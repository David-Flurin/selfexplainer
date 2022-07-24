import sys
import os

from models.selfexplainer import SelfExplainer

sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.assessment_metrics import prob_entropy, saliency, continuous_IOU, discrete_IOU, prob_sparsity, discrete_f1, soft_f1
from utils.assessment_metrics import mask_coverage, background_coverage, overlap, sim_ratio, f1s, auc
from data.dataloader import OIDataModule, ToyData_Saved_Module, VOCDataModule, VOC2012DataModule, COCODataModule, ToyDataModule
from models.classifier import VGG16ClassifierModel, Resnet50ClassifierModel
from models.resnet50 import Resnet50
from utils.helper import get_class_dictionary

from torchray.utils import get_device
import torch

from PIL import Image
import torchvision.transforms as transforms

from sklearn.metrics import f1_score

from tqdm import tqdm

def get_model_and_data(data_path, dataset_name, model_name, model_path):
    if dataset_name == "VOC":
        data_module = VOCDataModule(data_path, test_batch_size=1)
        if model_name == "vgg16":
            model = VGG16ClassifierModel.load_from_checkpoint(model_path, num_classes=20, dataset=dataset_name)
        elif model_name == "resnet50":
            model = Resnet50ClassifierModel.load_from_checkpoint(model_path, num_classes=20, dataset=dataset_name)
    elif dataset_name == "COCO":
        data_module = COCODataModule(data_path, test_batch_size=1)
        if model_name == "vgg16":
            model = VGG16ClassifierModel.load_from_checkpoint(model_path, num_classes=91, dataset=dataset_name)
        elif model_name == "resnet50":
            model = Resnet50ClassifierModel.load_from_checkpoint(model_path, num_classes=91, dataset=dataset_name)
    elif dataset_name in ["TOY", "TOY_MULTI"]:
        data_module = ToyData_Saved_Module(data_path, test_batch_size=1)
        model = Resnet50ClassifierModel.load_from_checkpoint(model_path, num_classes=8, dataset=dataset_name, multilabel = dataset_name == 'TOY_MULTI')
    elif dataset_name in ["OI", "OI_LARGE", "OI_SMALL"]:
        data_module = OIDataModule(data_path, test_batch_size=1)
        model = Resnet50ClassifierModel.load_from_checkpoint(model_path, num_classes=91, dataset=dataset_name)
   


    data_module.setup()

    return model, data_module


def get_selfexplainer_and_data(data_path, dataset_name, model_name, model_path, multilabel, aux_classifier):
    if dataset_name == "VOC":
        data_module = VOCDataModule(data_path, test_batch_size=1)
        model = SelfExplainer.load_from_checkpoint(model_path, num_classes=20, dataset=dataset_name, multilabel=multilabel, pretrained=False, aux_classifier=aux_classifier)
    elif dataset_name == "VOC2012":
        data_module = VOC2012DataModule(data_path, test_batch_size=1)
        model = SelfExplainer.load_from_checkpoint(model_path, num_classes=20, dataset=dataset_name, multilabel=multilabel, pretrained=False, aux_classifier=aux_classifier)
    elif dataset_name == "TOY":
        data_module = ToyData_Saved_Module(data_path, test_batch_size=1)
        model = SelfExplainer.load_from_checkpoint(model_path, num_classes=8, dataset=dataset_name, pretrained=False, multilabel=multilabel, aux_classifier=aux_classifier)
    elif dataset_name == "TOY_MULTI":
        data_module = ToyData_Saved_Module(data_path, test_batch_size=1)
        model = SelfExplainer.load_from_checkpoint(model_path, num_classes=8, dataset=dataset_name, pretrained=False, multilabel=multilabel, aux_classifier=aux_classifier)


    data_module.setup()

    return model, data_module

def segmented_generator(data_module, segmentations_path):
    """Generator that return all the segmented images"""
    for s in tqdm(data_module.test_dataloader()):
        img, meta = s
        x = img
        assert(len(x)==1)
        try:
            filename = Path(meta[0]["filename"]+".png")
            target_dict = get_class_dictionary('TOY', include_background_class=False)
            objects = [object[1] for object in meta[0]['objects']]
            category_id = [target_dict[e] for e in objects]
        except:
            fn = meta[0]['annotation']["filename"]
            if fn[-4] == '.':
                filename = Path(meta[0]['annotation']["filename"][:-4]+".png")
            else:
                filename = Path(meta[0]['annotation']["filename"]+".png")
            target_dict = get_class_dictionary('VOC', include_background_class=False)
            objects = meta[0]['annotation']['object']
            category_id = [target_dict[e["name"]] for e in objects]
        segmentation_filename =  segmentations_path / filename
        if not os.path.exists(segmentation_filename):
            continue
        else:
            yield x, category_id, filename   

def open_segmentation_mask(segmentation_filename, dataset_name):
    transformer = transforms.Compose([transforms.Resize((224, 224))])
    mask = Image.open(segmentation_filename).convert('L')
    mask = transformer(mask)
    mask = np.array(mask) / 255.0
    if dataset_name == "VOC":
        mask[mask > 0] = 1
    return mask

def get_path_mask(masks_path, dataset_name, model_name, method):
    return masks_path / Path('{}_{}_{}/'.format(dataset_name, model_name, method))

def gen_evaluation(data_path, masks_path, segmentations_path, dataset_name, model_name, model_path, method, compute_p=True, **kwargs):
    # Load the model and data
    model, data_module = get_model_and_data(data_path, dataset_name, model_name, model_path)
    # Path of the masks
    if method in ["0.5", "0", "1", "perfect"]:
        masks_path_method = None
    else:
        masks_path_method = get_path_mask(masks_path, dataset_name, model_name, method)
    
    if compute_p:
        device = get_device()
        model = model.to(device)
        model.eval()

    for x, category_id, filename in segmented_generator(data_module, segmentations_path):
        seg_mask = open_segmentation_mask(segmentations_path / filename, dataset_name)
        if method in ["0.5", "0", "1", "perfect"]:
            if method=="0":
                mask = np.zeros(seg_mask.shape, dtype=np.float32)
            elif method=="0.5":
                mask = 0.5*np.ones(seg_mask.shape, dtype=np.float32)
            elif method=="1":
                mask = np.ones(seg_mask.shape, dtype=np.float32)
            elif method=="perfect":
                mask = seg_mask.copy().astype(np.float32)
            else:
                raise ValueError("Something went wrong!")

        
        else:
            try:
                npz_name = Path(str(filename)[:-4] + ".npz")
                mask = np.load(masks_path_method / npz_name, dataset_name)["arr_0"]
            except:
                continue
        if np.sum(np.isnan(mask)):
            mask = np.zeros(shape=mask.shape, dtype=np.float32)
        if compute_p:
            x = x.to(device)
            logits = model.forward(x)
            p = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
            x_masked = torch.tensor(np.reshape(mask, [1,1, *mask.shape])).to(device) * x
            logits_mask = model.forward(x_masked)
            p_mask = torch.nn.functional.softmax(logits_mask, dim=1).detach().cpu().numpy().squeeze()
            x_background = torch.tensor(np.reshape(1-mask, [1,1, *mask.shape])).to(device) * x
            logits_background = model.forward(x_background)
            p_background = torch.nn.functional.softmax(logits_background, dim=1).detach().cpu().numpy().squeeze()
        else:
            p = None
            p_mask = None
            p_background = None
        yield mask, seg_mask, p, p_mask, p_background, category_id, x.detach().cpu().numpy().squeeze()

def selfexplainer_evaluation(data_module, masks_path, segmentations_path, dataset_name, model, model_name, method, multilabel):

    # Path of the masks
    if method in ["0.5", "0", "1", "perfect"]:
        masks_path_method = None
    else:
        masks_path_method = get_path_mask(masks_path, dataset_name, model_name, method)

    
    
    for x, category_id, filename in segmented_generator(data_module, segmentations_path):
        seg_mask = open_segmentation_mask(segmentations_path / filename, dataset_name)
        if method in ["0.5", "0", "1", "perfect"]:
            if method=="0":
                mask = np.zeros(seg_mask.shape, dtype=np.float32)
            elif method=="0.5":
                mask = 0.5*np.ones(seg_mask.shape, dtype=np.float32)
            elif method=="1":
                mask = np.ones(seg_mask.shape, dtype=np.float32)
            elif method=="perfect":
                mask = seg_mask.copy().astype(np.float32)
            else:
                raise ValueError("Something went wrong!")

        else:
            try:
                npz_name = Path(str(filename)[:-4] + ".npz")
                mask = np.load(masks_path_method / npz_name, dataset_name)["arr_0"]
            except:
                continue
        if np.sum(np.isnan(mask)):
            mask = np.zeros(shape=mask.shape, dtype=np.float32)



        logits_fn = torch.sigmoid if multilabel else lambda y: torch.nn.functional.softmax(y, dim=1)
        x = x.to(model.device)
        logits = model.forward(x)['image'][3]
        p = logits_fn(logits).detach().cpu().numpy().squeeze()

        x_masked = torch.tensor(np.reshape(mask, [1,1, *mask.shape])).to(model.device) * x
        logits_mask = model.forward(x_masked)['image'][3]
        p_mask = logits_fn(logits_mask).detach().cpu().numpy().squeeze()
        x_background = torch.tensor(np.reshape(1-mask, [1,1, *mask.shape])).to(model.device) * x
        logits_background = model.forward(x_background)['image'][3]
        p_background = logits_fn(logits_background).detach().cpu().numpy().squeeze()

        yield mask, seg_mask, p, p_mask, p_background, category_id, x.detach().cpu().numpy().squeeze()


def selfexplainer_compute_numbers(data_path, masks_path, segmentations_path, dataset_name, model_name, model_path, method, aux_classifier=False, multilabel=False):
#     sparsity = []
#     sparsity_masked = []
#     sparsity_background = []

#     entropy = []
#     entropy_masked = []
#     entropy_background = []

    d_f1_25 = []
    d_f1_50 = []
    d_f1_75 = []
    c_f1 = []
    a_f1s = []
    aucs = [] 

    d_IOU = []
    c_IOU = []
    sal = []

    over = []
    background_c = []
    mask_c = []
    sr = []

    print(model_path)

    model, data_module = get_selfexplainer_and_data(data_path, dataset_name, model_name, model_path, multilabel, aux_classifier)
    device = get_device()
    model = model.to(device)
    model.eval()



    for mask, seg_mask, p, p_mask, p_background, category_id, x in selfexplainer_evaluation(data_module, masks_path, segmentations_path, dataset_name, model, model_name, method, multilabel=multilabel):

#         sparsity.append(prob_sparsity(p))
#         sparsity_masked.append(prob_sparsity(p_mask))
#         sparsity_background.append(prob_sparsity(p_background))

#         entropy.append(prob_entropy(p))
#         entropy_masked.append(prob_entropy(p_mask))
#         entropy_background.append(prob_entropy(p_background))
        d_f1_25.append(discrete_f1(mask, seg_mask, 0.25))
        d_f1_50.append(discrete_f1(mask, seg_mask, 0.50))
        d_f1_75.append(discrete_f1(mask, seg_mask, 0.75))
        c_f1.append(soft_f1(mask, seg_mask))
        a_f1s.append(f1s(mask, seg_mask))
        aucs.append(auc(mask, seg_mask))

        d_IOU.append(discrete_IOU(mask, seg_mask))
        c_IOU.append(continuous_IOU(mask, seg_mask))

        sal.append(saliency(p_mask, category_id, mask))

        over.append(overlap(mask, seg_mask))
        background_c.append(background_coverage(mask, seg_mask))
        mask_c.append(mask_coverage(mask, seg_mask))

        sr.append(sim_ratio(mask, seg_mask))

        
    return d_f1_25,d_f1_50,d_f1_75,c_f1,a_f1s, aucs, d_IOU, c_IOU, sal, over, background_c, mask_c, sr

        
def compute_numbers(data_path, masks_path, segmentations_path, dataset_name, model_name, model_path, method, compute_p=True, aux_classifier=False):
#     sparsity = []
#     sparsity_masked = []
#     sparsity_background = []

#     entropy = []
#     entropy_masked = []
#     entropy_background = []

    d_f1_25 = []
    d_f1_50 = []
    d_f1_75 = []
    c_f1 = []
    a_f1s = []
    aucs = [] 

    d_IOU = []
    c_IOU = []
    sal = []

    over = []
    background_c = []
    mask_c = []
    sr = []




    for mask, seg_mask, p, p_mask, p_background, category_id, x in gen_evaluation(data_path, masks_path, segmentations_path, dataset_name, model_name, model_path, method, compute_p=compute_p, aux_classifier=aux_classifier):

#         sparsity.append(prob_sparsity(p))
#         sparsity_masked.append(prob_sparsity(p_mask))
#         sparsity_background.append(prob_sparsity(p_background))

#         entropy.append(prob_entropy(p))
#         entropy_masked.append(prob_entropy(p_mask))
#         entropy_background.append(prob_entropy(p_background))
        d_f1_25.append(discrete_f1(mask, seg_mask, 0.25))
        d_f1_50.append(discrete_f1(mask, seg_mask, 0.50))
        d_f1_75.append(discrete_f1(mask, seg_mask, 0.75))
        c_f1.append(soft_f1(mask, seg_mask))
        a_f1s.append(f1s(mask, seg_mask))
        aucs.append(auc(mask, seg_mask))

        d_IOU.append(discrete_IOU(mask, seg_mask))
        c_IOU.append(continuous_IOU(mask, seg_mask))

        sal.append(saliency(p_mask, category_id, mask))

        over.append(overlap(mask, seg_mask))
        background_c.append(background_coverage(mask, seg_mask))
        mask_c.append(mask_coverage(mask, seg_mask))

        sr.append(sim_ratio(mask, seg_mask))

        
    return d_f1_25,d_f1_50,d_f1_75,c_f1,a_f1s, aucs, d_IOU, c_IOU, sal, over, background_c, mask_c, sr
