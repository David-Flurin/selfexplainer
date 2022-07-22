from cv2 import threshold
import torch
import pytorch_lightning as pl
import json

from torch import  nn
from torch.optim import Adam
from pathlib import Path

from torchvision import models
#from torchviz import make_dot

from utils.helper import get_class_dictionary, get_targets_from_annotations, get_targets_from_segmentations, LogitStats, get_class_weights
from utils.metrics import MultiLabelMetrics, ClassificationMultiLabelMetrics
from plot import plot_class_metrics

import GPUtil
from matplotlib import pyplot as plt

VOC_segmentations_path = Path("../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/")
SMALLVOC_segmentations_path = Path("../../datasets/VOC2007_small/VOCdevkit/VOC2007/SegmentationClass/")


class Resnet50(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5,
                 gpu=0, metrics_threshold=0.5,  multilabel=False, weighted_sampling=True):

        super().__init__()

        self.gpu = gpu
        self.weighted_sampling = weighted_sampling
        self.model = models.resnet50(use_imagenet_pretraining=False, num_classes=num_classes)

        self.learning_rate = learning_rate
        self.dataset = dataset
        self.num_classes = num_classes

        self.multilabel = multilabel

        self.test_i = 0
        self.val_i = 0

        # ---------------- DEBUG -------------------
        self.i = 0.
    
       
        self.setup_losses()
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)


    def setup_losses(self):
        pos_weights = torch.ones(self.num_classes, device=self.device)*self.num_classes
        class_weights = torch.Tensor(get_class_weights(self.dataset), device=self.device)
        if not self.multilabel:
            if self.weighted_sampling:
                self.classification_loss_fn = nn.CrossEntropyLoss()
            else:
                #self.classification_loss_fn = nn.CrossEntropyLoss(weight = class_weights)
                self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            if self.weighted_sampling:
                self.classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones(self.num_classes, device=self.device)*self.num_classes/4)
            else:
                self.classification_loss_fn = nn.BCEWithLogitsLoss()




    def setup_metrics(self, num_classes, metrics_threshold):
        if self.dataset in ['COLOR', 'TOY', 'TOY_SAVED', 'SMALLVOC',  'VOC2012', 'VOC', 'OISMALL', 'OI', 'OI_LARGE']:
            self.train_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multilabel else 'ce')
            self.valid_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multilabel else 'ce')
            self.test_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multilabel else 'ce', classwise=True, dataset=self.dataset)
        else:
            self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    def forward(self, image):
        return self.model(image)
        
    def training_step(self, batch, batch_idx):
        #GPUtil.showUtilization()
        
        if self.dataset in ['VOC', 'SMALLVOC', 'VOC2012', 'OISMALL', 'OI', 'TOY', 'OI_LARGE']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset,  gpu=self.gpu)


        output = self(image)

        loss = self.classification_loss_fn(output, target_vector)

        self.log('loss', float(loss))
        
        self.train_metrics(output, target_vector.int())

        self.i += 1.
        self.log('iterations', self.i)
        return loss

    def training_epoch_end(self, outs):
        m = self.train_metrics.compute()
        self.log('train_metric', m)
        for k,v in m.items():
            self.log(f'{k}', v, prog_bar=True, logger=False)
        self.train_metrics.reset()
        '''
        self.f_tex_dist.print_distribution()
        self.b_text_dist.print_distribution()
        self.shapes_dist.print_distribution()
        '''
        
        for g in self.trainer.optimizers[0].param_groups:
            self.log('lr', g['lr'], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if self.dataset in ['VOC', 'SMALLVOC', 'VOC2012', 'OISMALL', 'OI_LARGE', 'OI']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)

    
        output = self(image)


        loss = self.classification_loss_fn(output, target_vector)

        self.valid_metrics(output, target_vector.int())
        self.val_i += 1
        return loss
   
    def validation_epoch_end(self, outs):
        m = self.valid_metrics.compute()
        self.log('valid_metric', m)
        for k,v in m.items():
            self.log(f'{k}', v, prog_bar=True, logger=False)
        self.valid_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.is_testing = True

    def test_step(self, batch, batch_idx):
        self.test_i += 1
        if self.dataset in ['VOC', 'SMALLVOC', 'VOC2012', 'OISMALL', 'OI', 'TOY', 'OI_LARGE']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)

        output = self(image, target_vector)
        loss = self.classification_loss_fn(output, target_vector)

        self.log('test_loss', loss)
        self.test_metrics(output['image'][3], target_vector.int())

    def test_epoch_end(self, outs):
        a_m = self.test_metrics.compute()
        m = a_m['Total']
        self.log('test_metric', m)
        for k,v in m.items():
            self.log(f'{k}', v, prog_bar=True, logger=False)

        (Path(self.save_path) / 'plots').mkdir(parents=True, exist_ok=True)
        plot_class_metrics(list(get_class_dictionary(self.dataset).keys()), a_m['Class'], Path(self.save_path) / 'plots' / 'class_metrics.png')
        with open(Path(self.save_path) / 'plots' / 'class_metrics.json', 'w') as jsonfile:
            a_m = dict_tensor_to_list(a_m)
            json.dump({'Classes': list(get_class_dictionary(self.dataset).keys()), "Metrics": a_m}, jsonfile)


        
        self.test_metrics.reset()
        #save_background_logits(self.test_background_logits, Path(self.save_path) / 'plots' / 'background_logits.png')


       
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
    '''
    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, threshold=0.001, min_lr=1e-5)
        lr_scheduler_config = {
        "scheduler": lr_scheduler,
        "interval": "epoch",
        "frequency": 1,
        "monitor": "loss",
        "strict": True,
        "name": None,
        }
        return {'optimizer': optim, 'lr_scheduler': lr_scheduler_config}
    '''
    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('frozen'):
                del checkpoint['state_dict'][k]


def dict_tensor_to_list(dict):
    if type(dict) == torch.Tensor:
        return dict.tolist()
    else:
        for k,v in dict.items():
            dict[k] = dict_tensor_to_list(v)
        return dict
