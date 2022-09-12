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
    elif dataset == "SYN":
        num_classes = 8
        data_path = Path(data_base_path) / "SYN"
        data_module = SyntheticData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    elif dataset == "SYN_MULTI":
        num_classes = 8
        data_path = Path(data_base_path) / "SYN_MULTI"
        data_module = SyntheticData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
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
    logits_fn = torch.sigmoid if multilabel else lambda x: torch.nn.functional.softmax(x, dim=1)
    preds = []
    trues = []

    for batch in tqdm(data_module.test_dataloader()):
        image, annotations = batch
        filename = get_filename_from_annotations(annotations, dataset=dataset)
        segmentation_filename = (segmentations_directory / os.path.splitext(filename)[0]).with_suffix( '.png')
        
        if not os.path.exists(segmentation_filename):
            continue
        
        image = image.to(device)
        targets = get_targets_from_annotations(annotations, dataset=dataset)

        output = model(image, targets)
        mask = output['image'][1]
        logits = output['image'][3]
        pred = logits_fn(logits)
        preds.append(pred.detach().cpu().squeeze().numpy() >= 0.5)
        trues.append(targets.int().cpu().squeeze().numpy())

        save_mask(mask, save_path / filename, dataset)
        save_masked_image(image, mask, save_path / "images" / filename, dataset)

    averages = ['micro', 'weighted']
    classification_metrics = {'f1':{}, 'precision':{}, 'recall':{}}
    for avg in averages:
        classification_metrics['f1'][avg] = f1_score(trues, preds, average=avg)
        classification_metrics['precision'][avg] = precision_score(trues, preds, average=avg)
        classification_metrics['recall'][avg] = recall_score(trues, preds, average=avg)

    np.savez(save_path / "classification_metrics.npz", classification_metrics=classification_metrics)
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
    elif dataset == "SYN":
        num_classes = 8
        data_path = Path(data_base_path) / "SYN"
        data_module = SyntheticData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    elif dataset == "SYN_MULTI":
        num_classes = 8
        data_path = Path(data_base_path) / "SYN_MULTI"
        data_module = SyntheticData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
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
                mask = output['image'][0][0][mask_class].sigmoid() if 'image' in output else output[0][0][mask_class].sigmoid()
                save_mask(mask, save_path / "class_masks" 
                                                / "masks_for_class_{}".format(mask_class)
                                                / filename, dataset=dataset)


############################################## Change to your settings ##########################################################
masks_path = Path("")
data_base_path = Path("/scratch/snx3000/dniederb/datasets/")

VOC_segmentations_path = Path(data_base_path / 'VOC2007/VOCdevkit/VOC2007/SegmentationClass/')
VOC2012_segmentations_path = Path(data_base_path / 'VOC2012/VOCdevkit/VOC2012/SegmentationClass/')
SYN_segmentations_path = Path(data_base_path / 'SYN/segmentations/textures/')
SYN_MULTI_segmentations_path = Path(data_base_path / 'SYN_MULTI/segmentations/textures/')
OI_segmentations_path = Path(data_base_path / 'OI/test/segmentations/')
OI_LARGE_segmentations_path = Path(data_base_path / 'OI_LARGE/test/segmentations/')
OI_SMALL_segmentations_path = Path(data_base_path / 'OI_SMALL/test/segmentations/')

dataset = "SYN_MULTI" # ['VOC', 'VOC2012', 'SYN', 'SYN_MULTI', 'OI_SMALL', 'OI', 'OI_LARGE'] 
multilabel = True

checkpoints_base_path = 'checkpoints/'
checkpoints = [''] # File name(s) without '.ckpt'-suffix
load_file = '' # if there is already a result file to which the results are appended
save_file = 'results.npz'

# If no masks for this model on the test set have been generated yet
compute_masks = True

# If we want to evaluate per-class masks
class_masks = False
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
        elif dataset == 'SYN':
            data_path = data_base_path / "SYN"
            segmentations_path = SYN_segmentations_path
        elif dataset == 'SYN_MULTI':
            data_path = data_base_path / "SYN_MULTI"
            segmentations_path = SYN_MULTI_segmentations_path
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

        d_f1_25,d_f1_50,d_f1_75,c_f1,a_f1s, aucs, d_IOU, c_IOU, sal, bg_entropy, combined_sal, over, background_c, mask_c, sr = selfexplainer_compute_numbers(data_path=data_path,
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
        d["combined_sal"] = combined_sal
        d['background_entropy'] = bg_entropy
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


