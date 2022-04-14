import torch
import pytorch_lightning as pl

from torch import nn, softmax
from torch.optim import Adam
from pathlib import Path

from copy import deepcopy

from torchvision import models
from utils.helper import get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution
from utils.image_display import save_all_class_masks, save_mask, save_masked_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics
from utils.weighting import softmax_weighting

class Classifier(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, gpu=0, profiler=None, metrics_threshold=-1.0, save_path="./results/"):

        super().__init__()

        self.gpu = gpu
        self.profiler = profiler

        self.learning_rate = learning_rate

        self.model = models.resnet50()
        self.model.fc = nn.Linear(2048, num_classes)
        self.dataset = dataset

        self.num_classes = num_classes

        self.setup_losses()
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        if self.dataset == 'TOY':
            self.f_tex_dist = Distribution()
            self.b_text_dist = Distribution()
            self.shapes_dist = Distribution()

        self.i = 0


    def setup_losses(self):
        self.classification_loss_fn = nn.BCEWithLogitsLoss()

    def setup_metrics(self, num_classes, metrics_threshold):
        self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    


    def forward(self, image, targets):
        logits = self.model(image) 
        return logits

    def training_step(self, a, batch_idx):
        image, annotations = a
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes = self.num_classes, gpu=self.gpu)

        if self.dataset == 'TOY':
            for a in annotations:
                for obj in a['objects']:
                    self.shapes_dist.update(obj[0])
                    self.f_tex_dist.update(obj[1])
            self.b_text_dist.update(a['background'])
           
        
        logits = self(image, targets)
        loss = self.classification_loss_fn(logits, targets)
        self.log('loss', float(loss))

        self.i += 1.
        self.log('iterations', self.i, prog_bar=True)
       
        self.train_metrics(logits, targets)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()
        # self.f_tex_dist.print_distribution()
        # self.b_text_dist.print_distribution()
        # self.shapes_dist.print_distribution()

    def validation_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)
        
        
        logits = self(image, targets)
        loss = self.classification_loss_fn(logits, targets)

        self.log('val_loss', loss)
        self.valid_metrics(logits, targets) 
   
    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)
        
        logits = self(image, targets)
        loss = self.classification_loss_fn(logits)

        self.log('test_loss', loss)
        self.test_metrics(logits, targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
