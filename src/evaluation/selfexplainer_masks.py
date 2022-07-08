import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from tqdm import tqdm
from pathlib import Path
from torchray.utils import get_device
from timeit import default_timer

from models.selfexplainer import SelfExplainer
from data.dataloader import *
from utils.helper import *
from utils.image_display import *

############################## Change to your settings ##############################
dataset = 'TOY' # one of: ['VOC', 'COCO']
data_base_path = '../../datasets/'
classifier_type = 'resnet50' # one of: ['vgg16', 'resnet50']
selfexplainer_checkpoint = '/home/david/Documents/Master/Thesis/selfexplainer/epoch=30-step=1549.ckpt'
VOC_segmentations_directory = '../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
TOY_segmentations_directory = '../../datasets/TOY/segmentations/textures/'
#####################################################################################

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

save_path = Path('{}_{}_{}/'.format(dataset, classifier_type, "selfexplainer"))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

model = SelfExplainer.load_from_checkpoint(selfexplainer_checkpoint, num_classes=num_classes, dataset=dataset, pretrained=False, aux_classifier=False)
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
    elif dataset == "TOY":
        segmentation_filename = TOY_segmentations_directory + os.path.splitext(filename)[0] + '.png'

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

print("Total time for masking process of the Selfexplainer with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))

