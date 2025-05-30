import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from compute_scores import compute_numbers

############################################## Change to your settings ##########################################################
masks_path = Path("/scratch/snx3000/dniederb/evaluation_data/baselines/")
data_base_path = Path("/scratch/snx3000/dniederb/datasets/")
VOC_segmentations_path = Path(data_base_path / 'VOC2007/VOCdevkit/VOC2007/SegmentationClass/')
VOC2012_segmentations_path = Path(data_base_path / 'VOC2012/VOCdevkit/VOC2012/SegmentationClass/')
TOY_segmentations_path = Path(data_base_path / 'TOY/segmentations/textures/')
TOY_MULTI_segmentations_path = Path(data_base_path / 'TOY_MULTI/segmentations/textures/')
OI_segmentations_path = Path(data_base_path / 'OI/test/segmentations/')
OI_LARGE_segmentations_path = Path(data_base_path / 'OI_LARGE/test/segmentations/')
OI_SMALL_segmentations_path = Path(data_base_path / 'OI_SMALL/test/segmentations/')


datasets = ["SYN"]
multilabel = True
classifiers = ["resnet50"]
resnet50_toy_checkpoint = '/scratch/snx3000/dniederb/checkpoints/resnet50/toy_singlelabel.ckpt'
resnet50_toy_multi_checkpoint = '/scratch/snx3000/dniederb/checkpoints/resnet50/toy_multilabel.ckpt'
resnet_50_oi_checkpoint = '/scratch/snx3000/dniederb/checkpoints/resnet50/oi_pretrained.ckpt'
resnet_50_oi_large_checkpoint = '/scratch/snx3000/dniederb/checkpoints/resnet50/oi_large_pretrained.ckpt'
resnet_50_oi_small_checkpoint = '/scratch/snx3000/dniederb/checkpoints/resnet50/oi_small_pretrained.ckpt'
resnet50_voc_checkpoint = "/scratch/snx3000/dniederb/checkpoints/resnet50/voc2007_pretrained.ckpt"
resnet50_coco_checkpoint = "../checkpoints/pretrained_classifiers/resnet50_coco.ckpt"
selfexplainer_toy_checkpoint = "../checkpoints/selfexplainer/toy.ckpt"
selfexplainer_voc_checkpoint = "../checkpoints/selfexplainer/voc.ckpt"

load_file = ''
save_file = 'results.npz'

methods = ["grad_cam", "rise"]
#################################################################################################################################

p = Path(save_file)
if not p.parent.is_dir():
    p.parent.mkdir(parents=True)

try:
    results = np.load(load_file, allow_pickle=True)["results"].item()
except:
    results = {}

for dataset in datasets:
    if not(dataset in results):
        results[dataset] = {}
    for classifier in classifiers:
        if not(classifier in results[dataset]):
            results[dataset][classifier] = {}
        for method in methods:
            if not(method in results[dataset][classifier]):
                results[dataset][classifier][method] = {}            
          
            if dataset == "VOC":
                data_path = data_base_path / "VOC2007"
                segmentations_path = VOC_segmentations_path
                model_path = resnet50_voc_checkpoint
            elif dataset == "TOY":
                data_path = data_base_path / "TOY"
                segmentations_path = TOY_segmentations_path
                model_path = resnet50_toy_checkpoint
            elif dataset == "TOY_MULTI":
                data_path = data_base_path / "TOY_MULTI"
                segmentations_path = TOY_MULTI_segmentations_path
                model_path = resnet50_toy_multi_checkpoint
            elif dataset == "OI":
                data_path = data_base_path / "OI"
                segmentations_path = OI_segmentations_path
                model_path = resnet_50_oi_checkpoint
            elif dataset == "OI_LARGE":
                data_path = data_base_path / "OI_LARGE"
                segmentations_path = OI_LARGE_segmentations_path
                model_path = resnet_50_oi_large_checkpoint
            elif dataset == 'OI_SMALL':
                data_path = data_base_path / 'OI_SMALL'
                segmentations_path = OI_SMALL_segmentations_path
                model_path = resnet_50_oi_small_checkpoint

            d_f1_25,d_f1_50,d_f1_75,c_f1,a_f1s, aucs, d_IOU, c_IOU, sal, bg_sal, combined_sal, combined_sal_wo_mean, log_a, log_pmask, log_entropy,bg_entropy, entropy,  over, background_c, mask_c, sr = compute_numbers(data_path=data_path,
                                                                                                                            masks_path=masks_path, 
                                                                                                                            segmentations_path=segmentations_path, 
                                                                                                                            dataset_name=dataset, 
                                                                                                                            model_name=classifier, 
                                                                                                                            model_path=model_path, 
                                                                                                                            method=method,
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
            d["bg_sal"] = bg_sal
            d["combined_sal"] = combined_sal
            d["combined_sal_wo_mean"] = combined_sal_wo_mean
            d['log_a'] = log_a
            d['log_pmask'] = log_pmask
            d['log_entropy'] = log_entropy
            d['background_entropy'] = bg_entropy
            d['normalized entropy'] = entropy

            d["over"] = over
            d["background_c"] = background_c
            d["mask_c"] = mask_c
            d["sr"] = sr
            results[dataset][classifier][method] = d
            print("Scores computed for: {} - {} - {}".format(dataset, classifier, method))
            # except:
            #     print("Cannot compute scores for: {} - {} - {}!".format(dataset, classifier, method))

            np.savez(save_file, results=results)

