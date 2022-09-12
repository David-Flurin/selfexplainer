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

from torchray.attribution.rise import rise

############################## Change to your settings ##############################
dataset = 'OI' # one of: ['VOC', 'VOC2012', 'SYN', 'SYN_MULTI', 'OI_SMALL', 'OI', 'OI_LARGE']
save_base_path = Path('/scratch/snx3000/dniederb/evaluation_data/baselines/')
data_base_path = '/scratch/snx3000/dniederb/datasets/'
classifier_type = 'resnet50'
classifier_checkpoint = '/scratch/snx3000/dniederb/checkpoints/resnet50/oi_pretrained.ckpt'
VOC_segmentations_directory = '/scratch/snx3000/dniederb/datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
SYN_segmentations_directory = "/scratch/snx3000/dniederb/datasets/SYN/segmentations/textures/"
SYN_MULTI_segmentations_directory = "/scratch/snx3000/dniederb/datasets/SYN_MULTI/segmentations/textures/"
OI_segmentations_directory = '/scratch/snx3000/dniederb/datasets/OI/test/segmentations/'
OI_LARGE_segmentations_directory = '/scratch/snx3000/dniederb/datasets/OI_LARGE/test/segmentations/'
OI_SMALL_segmentations_directory = '/scratch/snx3000/dniederb/datasets/OI_SMALL/test/segmentations/'

# Whether to compute a target attribution mask (seg) or per-class masks (classes)
mode = 'classes' #['seg', 'classes']
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

total_time = 0.0

save_path = save_base_path / '{}_{}_{}/'.format(dataset, classifier_type, "rise")
if not os.path.isdir(save_path):
    os.makedirs(save_path)

class RISEModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # Set up model
        if classifier_type == "resnet50":
            self.model = Resnet50.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset='SYN' if dataset=='SYN_MULTI' else dataset, weighted_sampling=False, multiclass = True if dataset in ['SYN_MULTI', 'VOC'] else False)
            self.target_layer = self.model.model.layer4[-1]
        else:
            raise Exception("Unknown classifier type " + classifier_type)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, image):
        saliency = rise(self.model, image)

        return saliency

    def test_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset='SYN' if dataset=='SYN_MULTI' else dataset)
        filename = get_filename_from_annotations(annotations, dataset='SYN' if dataset=='SYN_MULTI' else dataset)
        if mode == 'seg': 
            if dataset == "VOC":
                segmentation_filename = VOC_segmentations_directory + os.path.splitext  (filename)[0] + '.png'
            elif dataset == "SYN":
                segmentation_filename = SYN_segmentations_directory + filename + '.png'
            elif dataset == "SYN_MULTI":
                segmentation_filename = SYN_MULTI_segmentations_directory + filename + '.png'
            elif dataset == "OI_SMALL":
                segmentation_filename = OI_SMALL_segmentations_directory + filename + '.png'
            elif dataset == "OI":
                segmentation_filename = OI_segmentations_directory + filename + '.png'
            elif dataset == "OI_LARGE":
                segmentation_filename = OI_LARGE_segmentations_directory + filename + '.png'

            else:
                raise Exception("Illegal dataset: " + dataset)

            start_time = default_timer()
            saliency = self(image)
            saliencies = torch.zeros(num_classes, 224, 224)
            for class_index in range(num_classes):
                if targets[0][class_index] == 1.0:
                    class_sal = saliency[:, class_index].squeeze()
                    min_val = class_sal.min()
                    max_val = class_sal.max()
                    class_sal = class_sal - min_val
                    class_sal = torch.mul(class_sal, 1 / (max_val - min_val))
                    class_sal = class_sal.clamp(0, 1)
                    saliencies[class_index] = class_sal

            saliency_map = saliencies.amax(dim=0)
            end_time = default_timer()
            global total_time
            total_time += end_time - start_time

            save_mask(saliency_map, save_path / filename, dataset=dataset)

        elif mode == 'classes':
            target_classes = [index for index, value in enumerate(targets[0]) if value == 1.0]
            intersection = set(target_classes) & set(masks_for_classes)
            if intersection:
                saliency = self(image)
                for mask_class in masks_for_classes:
                    class_sal = saliency[:, mask_class].squeeze()
                    min_val = class_sal.min()
                    max_val = class_sal.max()
                    class_sal = class_sal - min_val
                    class_sal = torch.mul(class_sal, 1 / (max_val - min_val))
                    class_sal = class_sal.clamp(0, 1)

                    save_mask(class_sal, save_path / "class_masks" 
                                                       / "masks_for_class_{}".format(mask_class)
                                                       / filename, dataset=dataset)

model = RISEModel(num_classes=num_classes)
trainer = pl.Trainer(gpus=[0] if torch.cuda.is_available() else 0)
trainer.test(model=model, datamodule=data_module)

print("Total time for masking process of RISE with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))
