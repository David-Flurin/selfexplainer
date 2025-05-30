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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import *

############################## Change to your settings ##############################
dataset = 'VOC' # one of: ['VOC', 'VOC2012', 'SYN', 'SYN_MULTI', 'OI_SMALL', 'OI', 'OI_LARGE']
data_base_path = Path("/scratch/snx3000/dniederb/datasets/")

classifier_type = 'resnet50' 
classifier_checkpoint = '/scratch/snx3000/dniederb/checkpoints/resnet50/voc2007_pretrained.ckpt'

VOC_segmentations_path = Path(data_base_path / 'VOC2007/VOCdevkit/VOC2007/SegmentationClass/')
VOC2012_segmentations_path = Path(data_base_path / 'VOC2012/VOCdevkit/VOC2012/SegmentationClass/')
SYN_segmentations_path = Path(data_base_path / 'SYN/segmentations/textures/')
SYN_MULTI_segmentations_path = Path(data_base_path / 'SYN_MULTI/segmentations/textures/')
OI_segmentations_path = Path(data_base_path / 'OI/test/segmentations/')
OI_LARGE_segmentations_path = Path(data_base_path / 'OI_LARGE/test/segmentations/')
OI_SMALL_segmentations_path = Path(data_base_path / 'OI_SMALL/test/segmentations/')

save_base_path = Path('.')

# Whether to compute a target attribution mask (seg) or per-class masks (classes)
mode = 'seg' #['seg', 'classes']
masks_for_classes = [0, 2, 4, 6, 7, 9, 10, 11, 12, 14]

#####################################################################################
    
# Set up data module
if dataset == "VOC":
    num_classes = 20
    data_path = Path(data_base_path) / "VOC2007"
    data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
elif dataset == 'SYN':
    num_classes = 8
    data_path = Path(data_base_path) / "SYN"
    data_module = SyntheticData_Saved_Module(data_path=data_path)
elif dataset == 'SYN_MULTI':
    num_classes = 8
    data_path = Path(data_base_path) / "SYN_MULTI"
    data_module = SyntheticData_Saved_Module(data_path=data_path)
elif dataset == 'OI_SMALL':
    num_classes = 3
    data_path = Path(data_base_path) / "OI_SMALL"
    data_module = OIDataModule(data_path=data_path)
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

save_path = save_base_path / '{}_{}_{}/'.format(dataset, classifier_type, "grad_cam")
if not os.path.isdir(save_path):
    os.makedirs(save_path)

total_time = 0.0

class GradCAMModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.use_cuda = (torch.cuda.device_count() > 0)
        # Set up model
        if classifier_type == "resnet50":
            self.model = Resnet50.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset='SYN' if dataset=='SYN_MULTI' else dataset, weighted_sampling=False, fix_classifier_backbone=False, multilabel = True if dataset in ['SYN_MULTI', 'VOC'] else False)
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
        targets = get_targets_from_annotations(annotations, dataset='SYN' if dataset=='SYN_MULTI' else dataset)
        filename = get_filename_from_annotations(annotations, dataset='SYN' if dataset=='SYN_MULTI' else dataset)

        if mode == 'seg':
            if dataset == "VOC":
                segmentation_filename = VOC_segmentations_path / (os.path.splitext(filename)[0] + '.png')
            elif dataset == "SYN":
                segmentation_filename = SYN_segmentations_path / (filename + '.png')
            elif dataset == "SYN_MULTI":
                segmentation_filename = SYN_MULTI_segmentations_path / (filename + '.png')
            
            elif dataset == "OI_SMALL":
                segmentation_filename = OI_SMALL_segmentations_path / (filename + '.png')
            elif dataset == "OI":
                segmentation_filename = OI_segmentations_path / (filename + '.png')
            elif dataset == "OI_LARGE":
                segmentation_filename = OI_LARGE_segmentations_path / (filename + '.png')
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

            print(save_path)
            save_mask(saliency_map, save_path / filename, dataset='SYN' if dataset=='SYN_MULTI' else dataset)

        elif mode == 'classes':
            target_classes = [index for index, value in enumerate(targets[0]) if value == 1.0]
            intersection = set(target_classes) & set(masks_for_classes)
            if intersection:
                for mask_class in masks_for_classes:
                    saliency = torch.tensor(self(image, mask_class)[0, :])
                    saliency = saliency.nan_to_num(nan=0.0)

                    save_mask(saliency, save_path / "class_masks" 
                                                    / "masks_for_class_{}".format(mask_class)
                                                    / filename, dataset=dataset)

model = GradCAMModel(num_classes=num_classes)
trainer = pl.Trainer(gpus=[0] if torch.cuda.is_available() else 0)
trainer.test(model=model, datamodule=data_module)

print("Total time for masking process of GradCAM with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))
