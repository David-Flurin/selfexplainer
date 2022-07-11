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

from compute_scores import compute_numbers


def compute_masks(dataset, checkpoint, checkpoint_base_path, segmentations_directory):
    # Set up data module
    if dataset == "VOC":
        num_classes = 20
        data_path = Path(data_base_path) / "VOC2007"
        data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "TOY":
        num_classes = 8
        data_path = Path(data_base_path) / "TOY"
        data_module = ToyData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    else:
        raise Exception("Unknown dataset " + dataset)

    save_path = Path('{}_{}_{}/'.format(dataset, "selfexplainer", checkpoint))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    model = SelfExplainer.load_from_checkpoint(checkpoint_base_path+checkpoint+".ckpt", num_classes=num_classes, dataset=dataset, pretrained=False, aux_classifier=False)
    device = get_device()
    model.to(device)
    model.eval()

    data_module.setup()

    total_time = 0.0

    for batch in tqdm(data_module.test_dataloader()):
        image, annotations = batch

        filename = get_filename_from_annotations(annotations, dataset=dataset)
        segmentation_filename = (segmentations_directory / os.path.splitext(filename)[0]).with_suffix( '.png')

        if not os.path.exists(segmentation_filename):
            continue
        
        image = image.to(device)
        targets = get_targets_from_annotations(annotations, dataset=dataset, num_classes=num_classes)

        start_time = default_timer()
        mask = model(image, targets)['image'][1]
        end_time = default_timer()
        total_time += end_time - start_time

        save_mask(mask, save_path / filename, dataset)
        save_masked_image(image, mask, save_path / "images" / filename, dataset)


    print("Total time for masking process of the Selfexplainer with dataset {} and model {}: {} seconds".format(dataset, checkpoint, total_time))


############################################## Change to your settings ##########################################################
masks_path = Path(".")
data_base_path = Path("/scratch/snx3000/dniederb/datasets/")
VOC_segmentations_path = Path("/scratch/snx3000/dniederb/datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/")
TOY_segmentations_path = Path("/scratch/snx3000/dniederb/datasets/TOY/segmentations/textures/")

dataset = "TOY"
classifiers = ["resnet50"]
checkpoints_base_path = "../checkpoints/TOY/singlelabel/"
checkpoints = ["3_passes_unfrozen", "3_passes_frozen_first", "3_passes_frozen_final"]

load_file = 'results.npz'
save_file = 'results_toy_singlelabel.npz'

#################################################################################################################################

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
        elif dataset == 'TOY':
            data_path = data_base_path / "TOY"
            segmentations_path = TOY_segmentations_path
        
        model_name = checkpoint
        model_path = checkpoints_base_path + checkpoint + '.ckpt'

        #compute_masks(dataset, checkpoint, checkpoints_base_path, segmentations_path)
        

        d_f1_25,d_f1_50,d_f1_75,c_f1,a_f1s, aucs, d_IOU, c_IOU, sal, over, background_c, mask_c, sr = compute_numbers(data_path=data_path,
                                                                                                                        masks_path=masks_path, 
                                                                                                                        segmentations_path=segmentations_path, 
                                                                                                                        dataset_name=dataset, 
                                                                                                                        model_name='selfexplainer', 
                                                                                                                        model_path=model_path, 
                                                                                                                        method=checkpoint)


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
        results[checkpoint] = d
        print("Scores computed for: {} - {}".format(dataset, checkpoint))
        # except:
        #     print("Cannot compute scores for: {} - {} - {}!".format(dataset, classifier, method))

np.savez(save_file, results=results)


