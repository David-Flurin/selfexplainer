import torch
import pytorch_lightning as pl

from torch import nn, softmax
from torch.optim import Adam
from pathlib import Path

from utils.helper import get_class_dictionary, get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution, get_targets_from_segmentations, LogitStats
from utils.image_display import save_all_class_masked_images, save_mask, save_masked_image, save_background_logits, save_image, save_all_class_masks, get_unnormalized_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss, bg_loss, background_activation_loss, relu_classification
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics, ClassificationMultiLabelMetrics
from utils.weighting import softmax_weighting

from models.DeepLabv3 import Deeplabv3Resnet50Model

class Simple_Model(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1., pretrained=False, use_similarity_loss=False, similarity_regularizer=1.0, use_background_loss=False, bg_loss_regularizer=1.0, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.04, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, use_perfect_mask=False, count_logits=False, save_masks=False, save_all_class_masks=False, 
                 gpu=0, profiler=None, metrics_threshold=0.5, save_path="./results/", objective='classification', class_loss='bce', frozen=False, freeze_every=20, background_activation_loss=False, bg_activation_regularizer=0.5, target_threshold=0.7, non_target_threshold=0.3, background_loss='logits_ce', aux_classifier=False, multiclass=False, class_only=False):

        super().__init__()

        self.i = 0
        self.gpu = gpu
        self.aux_classifier = aux_classifier
        self.model = Deeplabv3Resnet50Model(pretrained=False, num_classes=num_classes, aux_classifier=aux_classifier)
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.num_classes = num_classes
        self.multiclass = multiclass
        if not self.multiclass:
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()
        

    def forward(self, image, targets):
        segmentations, logits = self.model(image)
        

        return logits

        
    def training_step(self, batch, batch_idx):
        
        image, annotations = batch
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)


        output = self(image, target_vector)
        loss = self.classification_loss_fn(output, target_vector)

        self.i += 1.
        self.log('iterations', self.i)
        return loss
        

    def validation_step(self, batch, batch_idx):
        if self.dataset in ['VOC', 'SMALLVOC', 'OISMALL']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)

        output = self(image, target_vector)

        loss = self.classification_loss_fn(output, target_vector)
        
        self.log('loss', float(loss))
    
        self.i += 1.
        self.log('iterations', self.i)
        return loss
   




    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
    

    


