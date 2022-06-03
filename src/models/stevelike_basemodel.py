from cv2 import threshold
import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import os
import random

from matplotlib import pyplot as plt 


from torch import nn, softmax
from torch.optim import Adam
from pathlib import Path

from copy import deepcopy

import pickle

#from torchviz import make_dot

from utils.helper import get_class_dictionary, get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution, get_targets_from_segmentations, LogitStats
from utils.image_display import save_all_class_masked_images, save_mask, save_masked_image, save_background_logits, save_image, save_all_class_masks, get_unnormalized_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss, bg_loss, background_activation_loss, relu_classification
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics, ClassificationMultiLabelMetrics
from utils.weighting import softmax_weighting
from evaluation.compute_scores import selfexplainer_compute_numbers

import GPUtil

VOC_segmentations_path = Path("../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/")
SMALLVOC_segmentations_path = Path("../../datasets/VOC2007_small/VOCdevkit/VOC2007/SegmentationClass/")


class Slike_BaseModel(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1., pretrained=False, use_similarity_loss=False, similarity_regularizer=1.0, use_background_loss=False, bg_loss_regularizer=1.0, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.04, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, use_perfect_mask=False, count_logits=False, save_masks=False, save_all_class_masks=False, 
                 gpu=0, metrics_threshold=0.5, save_path="./results/", objective='classification', class_loss='bce', frozen=False, freeze_every=20, background_activation_loss=False, bg_activation_regularizer=0.5, target_threshold=0.7, non_target_threshold=0.3, background_loss='logits_ce', aux_classifier=False, multiclass=False, class_only=False):

        super().__init__()

        self.gpu = gpu

        self.learning_rate = learning_rate
        self.weighting_koeff = weighting_koeff

        self.classifier = None
        self.dataset = dataset
        self.num_classes = num_classes

        self.aux_classifier = aux_classifier

        self.class_only = class_only

        self.use_similarity_loss = use_similarity_loss
        self.similarity_regularizer = similarity_regularizer
        self.use_background_loss = use_background_loss
        self.background_loss = background_loss
        self.bg_loss_regularizer = bg_loss_regularizer
        self.use_weighted_loss = use_weighted_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_total_area_regularizer = mask_total_area_regularizer
        self.ncmask_total_area_regularizer = ncmask_total_area_regularizer
        self.use_mask_variation_loss = use_mask_variation_loss
        self.mask_variation_regularizer = mask_variation_regularizer


        self.use_background_activation_loss = background_activation_loss
        self.bg_activation_regularizer = bg_activation_regularizer

        self.save_path = save_path
        self.save_masked_images = save_masked_images
        self.save_masks = save_masks
        self.save_all_class_masks = save_all_class_masks

        self.objective = objective

        self.test_background_logits = []

        self.freeze_every = freeze_every

        self.multiclass = multiclass

        # self.attention_layer = None
        # if use_attention_layer:
        #     l = [
        #         nn.Conv2d(num_classes, )
        #     ]
        #     self.attention_layer = nn.Conv2d

        self.test_i = 0

        # ---------------- DEBUG -------------------
        self.i = 0.
        self.use_perfect_mask = use_perfect_mask
        self.count_logits = count_logits

        self.global_image_mask = None
        self.global_object_mask = None
        self.first_of_epoch = True
        self.same_images = {}

        #self.automatic_optimization = False
        # -------------------------------------------


        if self.dataset == 'TOY':
            self.f_tex_dist = Distribution()
            self.b_text_dist = Distribution()
            self.shapes_dist = Distribution()

        self.setup_losses(class_mask_min_area=class_mask_min_area, class_mask_max_area=class_mask_max_area, target_threshold=target_threshold, non_target_threshold=non_target_threshold)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)


    def setup_losses(self, class_mask_min_area, class_mask_max_area, target_threshold, non_target_threshold):
        if not self.multiclass:
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()
        # elif self.class_loss == 'threshold':
        #     self.classification_loss_fn = lambda logits, targets: relu_classification(logits, targets, target_threshold, non_target_threshold)
        # else:
        #     raise ValueError(f'Classification loss argument {self.class_loss} not known')

        if self.objective == 'segmentation':
            self.classification_loss_fn = nn.BCEWithLogitsLoss()
        

        self.total_variation_conv = TotalVariationConv()
        self.class_mask_area_loss_fn = ClassMaskAreaLoss(min_area=class_mask_min_area, max_area=class_mask_max_area, gpu=self.gpu)



    def setup_metrics(self, num_classes, metrics_threshold):
        if self.dataset in ['COLOR', 'TOY', 'TOY_SAVED', 'SMALLVOC', 'VOC', 'OISMALL']:
            self.train_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multiclass else 'ce')
            self.valid_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multiclass else 'ce')
            self.test_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multiclass else 'ce')
        else:
            self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

        if self.count_logits:
            self.logit_stats = {'image': LogitStats(self.num_classes)}
            if self.use_similarity_loss:
                self.logit_stats['object'] = LogitStats(self.num_classes)
            if self.use_background_loss:
                self.logit_stats['background'] = LogitStats(self.num_classes)

 

    def forward(self, image, targets):
        segmentations, logits = self.model(image)
        if self.class_only:
            return logits, None, None, None, None, None, None, None

        classifier_segmentations, classifier_logits = self.classifier(image)
        target_mask, non_target_mask = extract_masks(segmentations, targets, gpu=self.gpu)
        if self.use_similarity_loss:
            masked_image = target_mask.unsqueeze(1) * image
            _, logits_mask = self.classifier(masked_image)
        else:
            logits_mask = None
            
        if self.use_background_loss:   
            target_mask_inversed = torch.ones_like(target_mask) - target_mask
            target_mask_inversed = target_mask_inversed.unsqueeze(1)
            inverted_masked_image = target_mask_inversed * image
            _, logits_inversed_mask = self.classifier(inverted_masked_image)
        else:
            logits_inversed_mask = None        

        return logits, classifier_logits, logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations, classifier_segmentations

<<<<<<< HEAD
        
=======


>>>>>>> 7933ef6a7d96ead140c6afe644e1e10966afbfc6
    def training_step(self, batch, batch_idx):
        #GPUtil.showUtilization()
        
        if self.dataset in ['VOC', 'SMALLVOC', 'OISMALL']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)


        if not self.class_only and self.i == 0 and (self.use_similarity_loss or self.use_background_loss):
            self.classifier = deepcopy(self.model)
            for p in self.classifier.parameters():
               p.requires_grad = False
            self.classifier.freeze()

        loss = torch.zeros(1, device = image.device)
        logits, logits_classifier, logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations, segmentations_classifier = self(image, target_vector)
        #if self.class_only:
        if self.objective == 'classification':
            classification_loss_initial = self.classification_loss_fn(logits, target_vector)
        elif self.objective == 'segmentation':
            targets = targets.to(torch.float64)
            classification_loss_initial = self.classification_loss_fn(segmentations, targets)
        else:
            raise ValueError('Unknown objective')
        
        classification_loss = classification_loss_initial
        self.log('classification_loss', classification_loss)   

        loss += classification_loss
        
        obj_back_loss = torch.zeros((1), device=loss.device)
        if not self.class_only and self.use_similarity_loss:
            #similarity_loss = self.similarity_regularizer * mask_similarity_loss(output['object'][3], target_vector, output['image'][1], output['object'][1])
            logit_fn = torch.sigmoid if self.multiclass else lambda x: torch.nn.functional.softmax(x, dim=-1)
            similarity_loss = self.classification_loss_fn(logits_mask, logit_fn(logits_classifier))
            #similarity_loss = self.classification_loss_fn(logits_mask, target_vector)
            self.log('similarity_loss', similarity_loss)
            obj_back_loss += similarity_loss

        if not self.class_only and self.use_background_loss:
            if self.bg_loss == 'entropy':
                background_entropy_loss = self.bg_loss_regularizer * entropy_loss(logits_inversed_mask)
            elif self.bg_loss == 'distance':
                    background_entropy_loss = self.bg_loss_regularizer * bg_loss(output['background'][0], target_vector, self.background_loss)

            self.log('background_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted


        mask_loss = torch.zeros((1), device=loss.device)
        if not self.class_only and self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            mask_loss += mask_variation_loss
            self.log('Area TV loss', mask_variation_loss)

        
            
            
        
        if not self.class_only and self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(segmentations, target_vector)) #+ self.class_mask_area_loss_fn(output['object'][0], target_vector))
            self.log('Bounding area loss', mask_area_loss)
            mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
            mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            self.log('Mean area loss', mask_area_loss)
            mask_loss += mask_area_loss


        if self.use_similarity_loss or self.use_background_loss or self.use_mask_variation_loss or self.use_mask_area_loss:
            if self.use_weighted_loss:
                w_loss = weighted_loss(loss, obj_back_loss + mask_loss, 2, 0.2)
                self.log('weighted_loss', w_loss)
                loss += w_loss
            else:
                loss = loss + obj_back_loss + mask_loss
        
        '''
        masked_image = target_mask.unsqueeze(1) * image
        self.logger.experiment.add_image('Train Masked Images', get_unnormalized_image(masked_image), self.i, dataformats='NCHW')
        self.logger.experiment.add_image('Train Images', get_unnormalized_image(image), self.i, dataformats='NCHW')
        self.logger.experiment.add_image('Train 1PassOutput', target_mask.unsqueeze(1), self.i, dataformats='NCHW')

        self.log('loss', float(loss))
        '''
        if self.use_similarity_loss and not self.class_only:
            self.train_metrics(logits_mask, target_vector.int())
        else:
            self.train_metrics(logits, target_vector.int())

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

        self.first_of_epoch = True

        #print("Checking for same images")
        #print("Size of dict", len(self.same_images))
        #for k,v in self.same_images.items():
        #    if v > 1:
        #        print(k, v)

    def validation_step(self, batch, batch_idx):
        if self.dataset in ['VOC', 'SMALLVOC', 'OISMALL']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)


        if not self.class_only and self.i == 0 and (self.use_similarity_loss or self.use_background_loss):
           self.classifier = deepcopy(self.model)
           print('Copied model')
           for _,p in self.classifier.named_parameters():
               p.requires_grad_(False)


        logits, logits_classifier, logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations, segmentations_classifier = self(image, target_vector)
        if self.objective == 'classification':
            classification_loss_initial = self.classification_loss_fn(logits, target_vector)
        elif self.objective == 'segmentation':
            targets = targets.to(torch.float64)
            classification_loss_initial = self.classification_loss_fn(segmentations, targets)
        else:
            raise ValueError('Unknown objective')
        
        classification_loss = classification_loss_initial
        self.log('val_classification_loss', classification_loss)   

        loss = classification_loss
        
        obj_back_loss = torch.zeros((1), device=loss.device)
        if not self.class_only and self.use_similarity_loss:
            #similarity_loss = self.similarity_regularizer * mask_similarity_loss(output['object'][3], target_vector, output['image'][1], output['object'][1])
            logit_fn = torch.sigmoid if self.multiclass else lambda x: torch.nn.functional.softmax(x, dim=-1)
            similarity_loss = self.classification_loss_fn(logits_mask, logit_fn(logits_classifier))
            self.log('val_similarity_loss', similarity_loss)
            obj_back_loss += similarity_loss

        if self.use_background_loss:
            if self.bg_loss == 'entropy':
                background_entropy_loss = self.bg_loss_regularizer * entropy_loss(logits_inversed_mask)
            elif self.bg_loss == 'distance':
                    background_entropy_loss = self.bg_loss_regularizer * bg_loss(output['background'][0], target_vector, self.background_loss)

            self.log('val_background_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted


        mask_loss = torch.zeros((1), device=loss.device)
        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            mask_loss += mask_variation_loss

        
            #mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], target_vector)) #+ self.class_mask_area_loss_fn(output['object'][0], target_vector))
        mask_area_loss = self.mask_total_area_regularizer * target_mask.mean()
        mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
        self.log('val_mask_area_loss', mask_area_loss)
        if self.use_mask_area_loss:
            mask_loss += mask_area_loss


        if self.use_similarity_loss or self.use_background_loss or self.use_mask_variation_loss or self.use_mask_area_loss:
            if self.use_weighted_loss:
                w_loss = weighted_loss(loss, obj_back_loss + mask_loss, 2, 0.2)
                self.log('val_weighted_loss', w_loss)
                loss += w_loss
            else:
                loss = loss + obj_back_loss + mask_loss

        if self.i % 5 == 4:
            masked_image = target_mask.unsqueeze(1) * image
            self.logger.experiment.add_image('Val Masked Images', get_unnormalized_image(masked_image), self.i, dataformats='NCHW')
            self.logger.experiment.add_image('Val Images', get_unnormalized_image(image), self.i, dataformats='NCHW')
            self.logger.experiment.add_image('Val 1PassOutput', target_mask.unsqueeze(1), self.i, dataformats='NCHW')
        
        self.log('loss', float(loss))
       
        self.valid_metrics(logits_mask, target_vector.int())

        self.i += 1.
        self.log('iterations', self.i)
        return loss
   
    def validation_epoch_end(self, outs):
        m = self.valid_metrics.compute()
        self.log('valid_metric', m)
        for k,v in m.items():
            self.log(f'{k}', v, prog_bar=True, logger=False)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        self.test_i += 1
        if self.dataset in ['VOC', 'SMALLVOC', 'OISMALL']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)

        logits, logits_classifier, logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations, segmentations_classifier = self(image, target_vector)

        
        if self.save_masked_images and image.size()[0] == 1:
            filename = Path(self.save_path) / "masked_images" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, target_mask, filename, self.dataset)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)
            save_mask(target_mask, Path(self.save_path) / "masks" / filename, self.dataset)

        if self.save_all_class_masks and image.size()[0] == 1 and self.dataset == "VOC":
            filename = Path(self.save_path) / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masks(image, segmentations, filename, self.dataset)
        
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_mask = self.classification_loss_fn(logits_mask, labels)
        else:
            classification_loss_mask = self.classification_loss_fn(logits_mask, target_vector)

        classification_loss_inversed_mask = self.bg_loss_regularizer * entropy_loss(logits_inversed_mask)
        loss = classification_loss_mask + classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
            mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            loss += mask_area_loss

        self.log('test_loss', loss)
        
        if self.use_similarity_loss:
            self.test_metrics(logits_mask, target_vector.int())
        else:
            self.test_metrics(logits, target_vector.int())

    def test_epoch_end(self, outs):
        m = self.test_metrics.compute()
        self.log('test_metric', m)
        for k,v in m.items():
            self.log(f'{k}', v, prog_bar=True, logger=False)
        self.test_metrics.reset()
        #save_background_logits(self.test_background_logits, Path(self.save_path) / 'plots' / 'background_logits.png')


        #DEBUG
        if self.count_logits and self.save_path:
            dir = self.save_path + '/logit_stats'
            os.makedirs(dir)
            class_dict = get_class_dictionary(self.dataset)
            for k,v in self.logit_stats.items():
                v.plot(dir + f'/{k}.png', list(class_dict.keys()))

        # if self.dataset == "SMALLVOC":
        #     segmentations_path = SMALLVOC_segmentations_path
        # elif self.dataset == 'VOC':
        #     segmentations_path = VOC_segmentations_path
        # selfexplainer_compute_numbers(Path(self.save_path) / "masked_image", segmentations_path, self.dataset,  )


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
            if k.startswith('classifier'):
                del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('classifier'):
                del checkpoint['state_dict'][k]

    


