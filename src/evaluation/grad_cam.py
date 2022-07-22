import pytorch_lightning as pl
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from pathlib import Path
from timeit import default_timer

from data.dataloader import *
from utils.helper import *
from utils.image_display import *
from models.resnet50 import Resnet50
from models.classifier import Resnet50ClassifierModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import *

############################## Change to your settings ##############################
dataset = 'VOC' # one of: ['VOC', 'TOY']
data_base_path = '/scratch/snx3000/dniederb/datasets/'
classifier_type = 'resnet50' # one of: ['vgg16', 'resnet50']
classifier_checkpoint = '../checkpoints/resnet_steven/voc2007_pretrained.ckpt'
VOC_segmentations_directory = '/scratch/snx3000/dniederb/datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
TOY_segmentations_directory = "/scratch/snx3000/dniederb/datasets/TOY/segmentations/textures/"
TOY_MULTI_segmentations_directory = "/scratch/snx3000/dniederb/datasets/TOY_MULTI/segmentations/textures/"
OI_segmentations_directory = '/scratch/snx3000/dniederb/datasets/OI/test/segmentations/'
OI_LARGE_segmentations_directory = '/scratch/snx3000/dniederb/datasets/OI_LARGE/test/segmentations/'
OI_SMALL_segmentations_directory = '/scratch/snx3000/dniederb/datasets/OI_SMALL/test/segmentations/'

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
elif dataset == 'TOY':
    num_classes = 8
    data_path = Path(data_base_path) / "TOY"
    data_module = ToyData_Saved_Module(data_path=data_path)
elif dataset == 'TOY_MULTI':
    num_classes = 8
    data_path = Path(data_base_path) / "TOY_MULTI"
    data_module = ToyData_Saved_Module(data_path=data_path)
elif dataset == 'OI_SMALL':
    num_classes = 3
    data_path = Path(data_base_path) / "OI_SMALL"
    data_module = OISmallDataModule(data_path=data_path)
elif dataset == 'OI':
    num_classes = 13
    data_path = Path(data_base_path) / "OI"
    data_module = OIDataModule(data_path=data_path)
elif dataset == 'OI_LARGE':
    num_classes = 20
    data_path = Path(data_base_path) / "OI_LARGE"
    data_module = OIDataModule(data_path=data_path)
else:
    raise Exception("Unknown dataset " + dataset)

save_path = Path('{}_{}_{}/'.format(dataset, classifier_type, "grad_cam"))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

total_time = 0.0

class GradCAMModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.use_cuda = (torch.cuda.device_count() > 0)
        # Set up model
        if classifier_type == "resnet50":
            self.model = Resnet50ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset='TOY' if dataset=='TOY_MULTI' else dataset, weighted_sampling=False, multiclass = True if dataset in ['TOY_MULTI', 'VOC'] else False)
            self.target_layer = self.model.feature_extractor[-2][-1]
        else:
            raise Exception("Unknown classifier type " + classifier_type)
        
        self.num_classes = num_classes

        self.cam = GradCAM(model=self.model, target_layer=self.target_layer, use_cuda=self.use_cuda)

    def forward(self, image, target):
        saliency = self.cam(input_tensor=image, target_category=target)

        return saliency

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset='TOY' if dataset=='TOY_MULTI' else dataset)
        filename = get_filename_from_annotations(annotations, dataset='TOY' if dataset=='TOY_MULTI' else dataset)
        if dataset == "VOC":
            segmentation_filename = VOC_segmentations_directory + os.path.splitext(filename)[0] + '.png'
        elif dataset == "TOY":
            segmentation_filename = TOY_segmentations_directory + filename + '.png'
        elif dataset == "TOY_MULTI":
            segmentation_filename = TOY_MULTI_segmentations_directory + filename + '.png'
        
        elif dataset == "OI_SMALL":
            segmentation_filename = OI_SMALL_segmentations_directory + filename + '.png'
        elif dataset == "OI":
            segmentation_filename = OI_segmentations_directory + filename + '.png'
        elif dataset == "OI_LARGE":
            segmentation_filename = OI_LARGE_segmentations_directory + filename + '.png'
        else:
            raise Exception("Illegal dataset: " + dataset)

        if not os.path.exists(segmentation_filename):
            return
        assert(targets.size()[0] == 1)

        start_time = default_timer()
        saliencies = torch.zeros(num_classes, 224, 224)
        for class_index in range(num_classes):
            if targets[0][class_index] == 1.0:
                saliencies[class_index] = torch.tensor(self(image, class_index)[0, :])

        saliency_map = saliencies.amax(dim=0)
        end_time = default_timer()
        global total_time
        total_time += end_time - start_time

        save_mask(saliency_map, save_path / filename, dataset='TOY' if dataset=='TOY_MULTI' else dataset)

model = GradCAMModel(num_classes=num_classes)
trainer = pl.Trainer(gpus=[0] if torch.cuda.is_available() else 0)
trainer.test(model=model, datamodule=data_module)

print("Total time for masking process of GradCAM with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))
