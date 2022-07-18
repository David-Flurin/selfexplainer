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
dataset = 'TOY' # one of: ['VOC', 'TOY']
data_base_path = '../../datasets/'
classifier_type = 'resnet50' # one of: ['vgg16', 'resnet50']
classifier_checkpoint = '/home/david/Documents/Master/Thesis/selfexplainer/src/checkpoints/resnet50/toy_singlelabel.ckpt'
VOC_segmentations_directory = '../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
TOY_segmentations_directory = "../../datasets/TOY/segmentations/textures/"
TOY_MULTI_segmentations_directory = "../../datasets/TOY_MULTI/segmentations/textures/"

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
else:
    raise Exception("Unknown dataset " + dataset)

total_time = 0.0

save_path = Path('{}_{}_{}/'.format(dataset, classifier_type, "rise"))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

class RISEModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # Set up model
        if classifier_type == "resnet50":
            self.model = Resnet50.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset='TOY' if dataset=='TOY_MULTI' else dataset)
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
        targets = get_targets_from_annotations(annotations, dataset='TOY' if dataset=='TOY_MULTI' else dataset, num_classes=num_classes)
        filename = get_filename_from_annotations(annotations, dataset='TOY' if dataset=='TOY_MULTI' else dataset)
        if dataset == "VOC":
            segmentation_filename = VOC_segmentations_directory + os.path.splitext(filename)[0] + '.png'
        elif dataset == "TOY":
            segmentation_filename = TOY_segmentations_directory + filename + '.png'
        elif dataset == "TOY_MULTI":
            segmentation_filename = TOY_MULTI_segmentations_directory + filename + '.png'
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

        save_mask(saliency_map, save_path / filename)

model = RISEModel(num_classes=num_classes)
trainer = pl.Trainer(gpus=[0] if torch.cuda.is_available() else 0)
trainer.test(model=model, datamodule=data_module)

print("Total time for masking process of RISE with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))
