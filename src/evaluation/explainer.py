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
data_base_path = Path('/scratch/snx3000/dniederb/datasets/')
classifier_type = 'resnet50' # one of: ['vgg16', 'resnet50']
explainer_classifier_checkpoint = '/users/dniederb/selfexplainer/src/checkpoints/explainer/voc2007_steven.ckpt'

VOC_segmentations_path = Path(data_base_path / 'VOC2007/VOCdevkit/VOC2007/SegmentationClass/')
VOC2012_segmentations_path = Path(data_base_path / 'VOC2012/VOCdevkit/VOC2012/SegmentationClass/')
TOY_segmentations_path = Path(data_base_path / 'TOY/segmentations/textures/')
TOY_MULTI_segmentations_path = Path(data_base_path / 'TOY_MULTI/segmentations/textures/')
OI_segmentations_path = Path(data_base_path / 'OI/test/segmentations/')
OI_LARGE_segmentations_path = Path(data_base_path / 'OI_LARGE/test/segmentations/')
OI_SMALL_segmentations_path = Path(data_base_path / 'OI_SMALL/test/segmentations/')

#####################################################################################

if dataset == "VOC":
        num_classes = 20
        data_path = Path(data_base_path) / "VOC2007"
        data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
        segmentations_path = VOC_segmentations_path
elif dataset == "VOC2012":
    num_classes = 20
    data_path = Path(data_base_path) / "VOC2012"
    data_module = VOC2012DataModule(data_path=data_path, test_batch_size=1)
elif dataset == "OI":
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
    segmentation_filename = segmentations_path /(os.path.splitext(filename)[0] + '.png')

    if not os.path.exists(segmentation_filename):
        continue
    
    image = image.to(device)
    targets = get_targets_from_annotations(annotations, dataset=dataset)

    start_time = default_timer()
    _, _, mask, _, _ = model(image, targets)
    end_time = default_timer()
    total_time += end_time - start_time

    save_mask(mask, save_path / filename, dataset=dataset)
    save_masked_image(image, mask, save_path / "images" / filename, dataset=dataset)

print("Total time for masking process of the Explainer with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))

