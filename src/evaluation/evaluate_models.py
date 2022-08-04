import sys
import os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from torchray.utils import get_device
from timeit import default_timer

from models.selfexplainer import SelfExplainer
from data.dataloader import *
from utils.helper import *
from utils.image_display import *

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from compute_scores import compute_numbers, selfexplainer_compute_numbers


def compute_masks_and_f1(save_path, dataset, checkpoint, checkpoint_base_path, segmentations_directory, aux_classifier, multilabel):
    # Set up data module
    if dataset == "VOC":
        num_classes = 20
        data_path = Path(data_base_path) / "VOC2007"
        data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "VOC2012":
        num_classes = 20
        data_path = Path(data_base_path) / "VOC2012"
        data_module = VOC2012DataModule(data_path=data_path, test_batch_size=1)
    elif dataset == 'OI_SMALL':
        num_classes = 3
        data_path = Path(data_base_path) / "OI_SMALL"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)    
    elif dataset == 'OI':
        num_classes = 13
        data_path = Path(data_base_path) / "OI"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "OI_LARGE":
        num_classes = 20
        data_path = Path(data_base_path) / "OI_LARGE"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "TOY":
        num_classes = 8
        data_path = Path(data_base_path) / "TOY"
        data_module = ToyData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    elif dataset == "TOY_MULTI":
        num_classes = 8
        data_path = Path(data_base_path) / "TOY_MULTI"
        data_module = ToyData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    else:
        raise Exception("Unknown dataset " + dataset)

    save_path = save_path / Path('{}_{}_{}/'.format(dataset, "selfexplainer", checkpoint))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    model = SelfExplainer.load_from_checkpoint(checkpoint_base_path+checkpoint+".ckpt", num_classes=num_classes, multilabel=multilabel, dataset=dataset, pretrained=False, aux_classifier=aux_classifier)
    device = get_device()
    model.to(device)
    model.eval()

    data_module.setup()

    total_time = 0.0
    print(checkpoint_base_path+checkpoint+".ckpt")

    logits_fn = torch.sigmoid if multilabel else lambda x: torch.nn.functional.softmax(x, dim=1)
    thresh_preds = []
    preds = []
    trues = []
    #trues_roc = []

    #i = 0
    for batch in tqdm(data_module.test_dataloader()):
        image, annotations = batch



        filename = get_filename_from_annotations(annotations, dataset=dataset)
        segmentation_filename = (segmentations_directory / os.path.splitext(filename)[0]).with_suffix( '.png')
        
        
        if not os.path.exists(segmentation_filename):
            continue
        
        image = image.to(device)
        targets = get_targets_from_annotations(annotations, dataset=dataset)

        start_time = default_timer()
        output = model(image, targets)
        mask = output['image'][1]
        logits = (output['image'][3])
        pred = logits_fn(logits)
        preds.append(pred.detach().cpu().squeeze().numpy() >= 0.5)
        #thresh_preds.append(pred.detach().cpu().squeeze().numpy() > 0.5)
        trues.append(targets.int().cpu().squeeze().numpy())
        # if multilabel:
        #     trues_roc.append(targets.int().cpu().squeeze().numpy())
        # else:
        #     trues_roc.append(np.where(targets.int().cpu().squeeze().numpy() == 1)[0])


        end_time = default_timer()
        total_time += end_time - start_time

        save_mask(mask, save_path / filename, dataset)
        save_masked_image(image, mask, save_path / "images" / filename, dataset)

        #i += 1
        #if i == 6:
        #    break

    averages = ['micro', 'weighted']
    classification_metrics = {'f1':{}, 'precision':{}, 'recall':{}}
    #trues = np.stack(trues, axis=0)
    #trues_roc = np.stack(trues_roc, axis=0)
    for avg in averages:
        classification_metrics['f1'][avg] = f1_score(trues, preds, average=avg)
        classification_metrics['precision'][avg] = precision_score(trues, preds, average=avg)
        classification_metrics['recall'][avg] = recall_score(trues, preds, average=avg)

    np.savez(save_path / "classification_metrics.npz", classification_metrics=classification_metrics)



    print("Total time for masking process of the Selfexplainer with dataset {} and model {}: {} seconds".format(dataset, checkpoint, total_time))
    return classification_metrics

def compute_class_masks(save_path, dataset, checkpoint, checkpoint_base_path, masks_for_classes, aux_classifier, multilabel):
    # Set up data module
    if dataset == "VOC":
        num_classes = 20
        data_path = Path(data_base_path) / "VOC2007"
        data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "VOC2012":
        num_classes = 20
        data_path = Path(data_base_path) / "VOC2012"
        data_module = VOC2012DataModule(data_path=data_path, test_batch_size=1)
    elif dataset == 'OI_SMALL':
        num_classes = 3
        data_path = Path(data_base_path) / "OI_SMALL"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)    
    elif dataset == 'OI':
        num_classes = 13
        data_path = Path(data_base_path) / "OI"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "OI_LARGE":
        num_classes = 20
        data_path = Path(data_base_path) / "OI_LARGE"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "TOY":
        num_classes = 8
        data_path = Path(data_base_path) / "TOY"
        data_module = ToyData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    elif dataset == "TOY_MULTI":
        num_classes = 8
        data_path = Path(data_base_path) / "TOY_MULTI"
        data_module = ToyData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    else:
        raise Exception("Unknown dataset " + dataset)

    save_path = save_path / Path('{}_{}_{}/'.format(dataset, "selfexplainer", checkpoint))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    model = SelfExplainer.load_from_checkpoint(checkpoint_base_path+checkpoint+".ckpt", num_classes=num_classes, multilabel=multilabel, dataset=dataset, pretrained=False, aux_classifier=aux_classifier)
    device = get_device()
    model.to(device)
    model.eval()

    data_module.setup()

    total_time = 0.0
    print(checkpoint_base_path+checkpoint+".ckpt")

    logits_fn = torch.sigmoid if multilabel else lambda x: torch.nn.functional.softmax(x, dim=1)

    for batch in tqdm(data_module.test_dataloader()):
        image, annotations = batch

        filename = get_filename_from_annotations(annotations, dataset=dataset)
        
        image = image.to(device)
        targets = get_targets_from_annotations(annotations, dataset=dataset)

        target_classes = [index for index, value in enumerate(targets[0]) if value == 1.0]
        intersection = set(target_classes) & set(masks_for_classes)
        if intersection:
            output = model(image, targets)
            for mask_class in masks_for_classes:
                mask = output['image'][0][0][mask_class].sigmoid()
                save_mask(mask, save_path / "class_masks" 
                                                / "masks_for_class_{}".format(mask_class)
                                                / filename, dataset=dataset)


############################################## Change to your settings ##########################################################
masks_path = Path("data/OI_SMALL/")
data_base_path = Path("/scratch/snx3000/dniederb/datasets/")
VOC_segmentations_path = Path(data_base_path / 'VOC2007/VOCdevkit/VOC2007/SegmentationClass/')
VOC2012_segmentations_path = Path(data_base_path / 'VOC2012/VOCdevkit/VOC2012/SegmentationClass/')
TOY_segmentations_path = Path(data_base_path / 'TOY/segmentations/textures/')
TOY_MULTI_segmentations_path = Path(data_base_path / 'TOY_MULTI/segmentations/textures/')
OI_segmentations_path = Path(data_base_path / 'OI/test/segmentations/')
OI_LARGE_segmentations_path = Path(data_base_path / 'OI_LARGE/test/segmentations/')
OI_SMALL_segmentations_path = Path(data_base_path / 'OI_SMALL/test/segmentations/')

dataset = "OI_SMALL"
multilabel = False
classifiers = ["resnet50"]
checkpoints_base_path = "../checkpoints/OI_SMALL/"

checkpoints = ["3passes_1koeff_01" ]

load_file = ''
save_file = 'results/selfexplainer/OI_SMALL/1koeff_3passes.npz'
compute_masks = True
class_masks = True

masks_for_classes = [0, 2, 4, 6, 7, 9, 10, 11, 12, 14]


#################################################################################################################################

p = Path(save_file)
if not p.parent.is_dir():
    p.parent.mkdir(parents=True)

try:
    results = np.load(load_file, allow_pickle=True)["results"].item()
except:
    results = {}


for checkpoint in checkpoints:
    if not(checkpoint in results):
        results[checkpoint] = {}
        # try:
        if dataset == 'VOC':
            data_path = data_base_path / "VOC2007"
            segmentations_path = VOC_segmentations_path
        if dataset == 'VOC2012':
            data_path = data_base_path / "VOC2012"
            segmentations_path = VOC2012_segmentations_path
        elif dataset == 'TOY':
            data_path = data_base_path / "TOY"
            segmentations_path = TOY_segmentations_path
        elif dataset == 'TOY_MULTI':
            data_path = data_base_path / "TOY_MULTI"
            segmentations_path = TOY_MULTI_segmentations_path
        elif dataset == 'OI_SMALL':
            data_path = data_base_path / "OI_SMALL"
            segmentations_path = OI_SMALL_segmentations_path
        elif dataset == 'OI':
            data_path = data_base_path / "OI"
            segmentations_path = OI_segmentations_path
        elif dataset == 'OI_LARGE':
            data_path = data_base_path / "OI_LARGE"
            segmentations_path = OI_LARGE_segmentations_path
            
        model_name = checkpoint
        model_path = checkpoints_base_path + checkpoint + '.ckpt'

        if checkpoint.startswith('aux'):
            aux_classifier=True
        else:
            aux_classifier=False

        if class_masks:
            compute_class_masks(masks_path, dataset, checkpoint, checkpoints_base_path, masks_for_classes, aux_classifier, multilabel=multilabel)
            quit()
        
        if compute_masks:
            classification_metrics = compute_masks_and_f1(masks_path, dataset, checkpoint, checkpoints_base_path, segmentations_path, aux_classifier, multilabel=multilabel)
        else:
            save_path = masks_path / Path('{}_{}_{}/'.format(dataset, "selfexplainer", checkpoint))
            classification_metrics = np.load(save_path / 'classification_metrics.npz' , allow_pickle=True)["classification_metrics"].item()

        d_f1_25,d_f1_50,d_f1_75,c_f1,a_f1s, aucs, d_IOU, c_IOU, sal, over, background_c, mask_c, sr = selfexplainer_compute_numbers(data_path=data_path,
                                                                                                                        masks_path=masks_path, 
                                                                                                                        segmentations_path=segmentations_path, 
                                                                                                                        dataset_name=dataset, 
                                                                                                                        model_name='selfexplainer', 
                                                                                                                        model_path=model_path, 
                                                                                                                        method=checkpoint, 
                                                                                                                        aux_classifier=aux_classifier, 
                                                                                                                        multilabel=multilabel)


        d = {}
        d["d_f1_25"] = d_f1_25
        d["d_f1_50"] = d_f1_50
        d["d_f1_75"] = d_f1_75
        d["d_f1"] = ((np.array(d_f1_25) + np.array(d_f1_50) + np.array(d_f1_75)) /3).tolist()
        d["c_f1"] = c_f1
        d["a_f1s"] = a_f1s
        d["aucs"] = aucs
        d["d_IOU"] = d_IOU
        d["c_IOU"] = c_IOU
        d["sal"] = sal
        d["over"] = over
        d["background_c"] = background_c
        d["mask_c"] = mask_c
        d["sr"] = sr
        d['classification_metrics'] = classification_metrics
        results[checkpoint] = d
        print("Scores computed for: {} - {}".format(dataset, checkpoint))
        # except:
        #     print("Cannot compute scores for: {} - {} - {}!".format(dataset, classifier, method))



        np.savez(save_file, results=results)


