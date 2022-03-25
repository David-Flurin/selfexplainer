import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from tqdm import tqdm
from pathlib import Path
from torchray.utils import get_device
from timeit import default_timer

from models.explainer_classifier import ExplainerClassifierModel
from data.dataloader import *
from utils.helper import *
from utils.image_display import *

############################## Change to your settings ##############################
dataset = 'VOC' # one of: ['VOC', 'COCO']
data_base_path = '../../datasets/'
classifier_type = 'vgg16' # one of: ['vgg16', 'resnet50']
explainer_classifier_checkpoint = '../checkpoints/explainer_vgg16_voc.ckpt'
VOC_segmentations_directory = '../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
COCO_segmentations_directory = './coco_segmentations/'
#####################################################################################

# Set up data module
if dataset == "VOC":
    num_classes = 20
    data_path = Path(data_base_path) / "VOC2007"
    data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
elif dataset == "COCO":
    num_classes = 91
    data_path = Path(data_base_path) / "COCO2014"
    data_module = COCODataModule(data_path=data_path, test_batch_size=1)
else:
    raise Exception("Unknown dataset " + dataset)

save_path = Path('{}_{}_{}/'.format(dataset, classifier_type, "explainer"))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

model = ExplainerClassifierModel.load_from_checkpoint(explainer_classifier_checkpoint, num_classes=num_classes, dataset=dataset, classifier_type=classifier_type)
device = get_device()
model.to(device)
model.eval()

data_module.setup()

total_time = 0.0

for batch in tqdm(data_module.test_dataloader()):
    image, annotations = batch

    filename = get_filename_from_annotations(annotations, dataset=dataset)
    if dataset == "VOC":
        segmentation_filename = VOC_segmentations_directory + os.path.splitext(filename)[0] + '.png'
    elif dataset == "COCO":
        segmentation_filename = COCO_segmentations_directory + os.path.splitext(filename)[0] + '.png'

    if not os.path.exists(segmentation_filename):
        continue
    
    image = image.to(device)
    targets = get_targets_from_annotations(annotations, dataset=dataset)

    start_time = default_timer()
    _, _, mask, _, _ = model(image, targets)
    end_time = default_timer()
    total_time += end_time - start_time

    save_mask(mask, save_path / filename)
    save_masked_image(image, mask, save_path / "images" / filename)

print("Total time for masking process of the Explainer with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))

