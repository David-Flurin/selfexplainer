import torch
from torchmetrics import Accuracy
import json

import pytorch_lightning
from torch import  nn
from torch.optim import Adam
from pathlib import Path
import pytorch_lightning as pl

from copy import deepcopy
from utils.helper import get_class_dictionary, get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution, get_targets_from_segmentations, LogitStats, get_VOC_dictionary, get_class_weights
from utils.image_display import save_all_class_masked_images, save_mask, save_masked_image, save_background_logits, save_image, save_all_class_masks, get_unnormalized_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss, bg_loss, relu_classification, similarity_loss_fn
from utils.metrics import ClassificationMultiLabelMetrics
from utils.weighting import softmax_weighting
#from plot import plot_class_metrics


VOC_segmentations_path = Path("../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/")
SMALLVOC_segmentations_path = Path("../../datasets/VOC2007_small/VOCdevkit/VOC2007/SegmentationClass/")


class BaseModel(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1., pretrained=False, use_similarity_loss=False, similarity_regularizer=1.0, use_background_loss=False, bg_loss_regularizer=1.0, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.05, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, save_masks=False, save_all_class_masks=False, 
                 non_target_threshold=0.3, background_loss='logits_ce', aux_classifier=False, multilabel=False, use_bounding_loss=False, similarity_loss_mode='rel', weighted_sampling=True, similarity_loss_scheduling=500, background_loss_scheduling=500, mask_loss_scheduling=1000, use_loss_scheduling=False,
                 gpu=0, profiler=None, metrics_threshold=0.5, save_path="./results/", class_loss='bce', frozen=False, freeze_every=20, target_threshold=0.7, use_mask_logit_loss=False, mask_logit_loss_regularizer=1.0, mask_loss_weighting_params=[5, 0.1], object_loss_weighting_params=[2, 0.2]):

        super().__init__()

        self.gpu = gpu
        self.profiler = profiler

        self.learning_rate = learning_rate
        self.weighting_koeff = weighting_koeff

        self.dataset = dataset
        self.num_classes = num_classes

        self.aux_classifier = aux_classifier

        self.use_similarity_loss = use_similarity_loss
        self.similarity_regularizer = similarity_regularizer
        self.similarity_loss_mode = similarity_loss_mode
        self.use_background_loss = use_background_loss
        self.background_loss = background_loss
        self.bg_loss_regularizer = bg_loss_regularizer
        self.use_weighted_loss = use_weighted_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_total_area_regularizer = mask_total_area_regularizer
        self.ncmask_total_area_regularizer = ncmask_total_area_regularizer
        self.use_bounding_loss = use_bounding_loss
        self.use_mask_variation_loss = use_mask_variation_loss
        self.mask_variation_regularizer = mask_variation_regularizer
        self.use_mask_logit_loss = use_mask_logit_loss
        self.mask_logit_loss_regularizer = mask_logit_loss_regularizer
        self.object_loss_weighting_params = object_loss_weighting_params
        self.mask_loss_weighting_params = mask_loss_weighting_params


        self.save_path = save_path
        self.save_masked_images = save_masked_images
        self.save_masks = save_masks
        self.save_all_class_masks = save_all_class_masks

        self.frozen = frozen
        self.freeze_every = freeze_every

        self.multilabel = multilabel

        self.test_i = 0
        self.val_i = 0

        # ---------------- DEBUG -------------------
        self.i = 0.

        self.global_image_mask = None
        self.global_object_mask = None
        self.same_images = {}
        self.sim_losses = {'object_0': 0., 'object_1': 0., 'object_2':0., 'object_3':0., 'object_4':0., 'object_5':0., 'object_6':0.}

        self.data_stats = {str(k):0. for k in range(self.num_classes)}

        #self.automatic_optimization = False
        # -------------------------------------------
        self.weighted_sampling = weighted_sampling

        self.setup_losses(class_mask_min_area=class_mask_min_area, class_mask_max_area=class_mask_max_area, target_threshold=target_threshold, non_target_threshold=non_target_threshold)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)


        # With loss scheduling active, we can decide after how many iterations the different losses are used. 
        # This was not used in the final Selfexplainer!
        self.use_loss_scheduling = use_loss_scheduling
        if use_loss_scheduling:
            self.use_similarity_loss = False
            self.use_background_loss = False
            self.use_mask_area_loss = False
            self.use_mask_variation_loss = False
            self.use_bounding_loss = False
        self.background_loss_scheduling = background_loss_scheduling
        self.similarity_loss_scheduling = similarity_loss_scheduling
        self.mask_loss_scheduling = mask_loss_scheduling



    def setup_losses(self, class_mask_min_area, class_mask_max_area, target_threshold, non_target_threshold):
        pos_weights = torch.ones(self.num_classes, device=self.device)*self.num_classes
        class_weights = torch.Tensor(get_class_weights(self.dataset), device=self.device)
        if not self.multilabel:
            self.classification_loss_fn = nn.CrossEntropyLoss()       
            self.similarity_loss_fn = nn.CrossEntropyLoss()
        else:
            if self.weighted_sampling:
                self.classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones(self.num_classes, device=self.device)*self.num_classes/4)
            else:
                self.classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights*class_weights)
            self.similarity_loss_fn = nn.BCEWithLogitsLoss()

        self.total_variation_conv = TotalVariationConv()
        self.class_mask_area_loss_fn = ClassMaskAreaLoss(min_area=class_mask_min_area, max_area=class_mask_max_area, gpu=self.gpu)

    def setup_metrics(self, num_classes, metrics_threshold):
        self.train_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multilabel else 'ce')
        self.valid_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multilabel else 'ce')
        self.test_metrics = ClassificationMultiLabelMetrics(metrics_threshold, num_classes=num_classes, gpu=self.gpu, loss='bce' if self.multilabel else 'ce', classwise=True, dataset=self.dataset)

    def forward(self, image, targets=None):
        output = {}
        output['image'] = self._forward(image, targets)

        i_mask = output['image'][1]

        
        if targets != None and self.use_similarity_loss:
            if self.multilabel: 
                i_masks = torch.sigmoid(output['image'][0])
                max_objects = 0
                for b in range(targets.size()[0]):
                    if targets[b].sum() > max_objects:
                        max_objects = int(targets[b].sum().item())
                for i in range(max_objects):
                    batch_indices = (targets.sum(1) > i).nonzero().squeeze(1)
                    new_batch = torch.index_select(image, 0, batch_indices)
                    seg_indices_list = []
                    for b_idx in batch_indices.tolist():
                        seg_indices_list.append((targets[b_idx] == 1.).nonzero()[i])
                    seg_indices = torch.cat(seg_indices_list)
                    inter = torch.index_select(i_masks, 0, batch_indices)
                    new_batch_masks = inter[torch.arange(inter.size(0)), seg_indices].unsqueeze(1)
                    new_batch_masked = new_batch_masks * new_batch
                    output[f'object_{i}'] = self._forward(new_batch_masked, targets)
            else:
                output['object_0'] = self._forward(i_mask.unsqueeze(1) * image, targets)
            
        
        if targets != None and self.use_background_loss:   
            target_mask_inversed = torch.ones_like(i_mask) - i_mask
            if image.dim() > 3:
                target_mask_inversed = target_mask_inversed.unsqueeze(1)
            inverted_masked_image = target_mask_inversed * image
            output['background'] = self._forward(inverted_masked_image, targets, frozen=self.frozen)
            
        # if len(output) == 1:
        #     return output['image']
        # else:
        return output

    def _forward(self, image, targets, frozen=False):
        if self.aux_classifier:
            if frozen:
                if self.training and image.size(0) == 1:
                    image = image.repeat(2,1,1,1)
                    segmentations, logits = self.frozen_model(image)
                    segmentations = segmentations[0].unsqueeze(0)
                    logits = logits[0].unsqueeze(0)
                elif self.training:
                    segmentations, logits = self.frozen_model(image)
                else:
                    segmentations, logits = self.model(image)

            else:
                if self.training and image.size(0) == 1:
                    image = image.repeat(2,1,1,1)
                    segmentations, logits = self.model(image)
                    segmentations = segmentations[0].unsqueeze(0)
                    logits = logits[0].unsqueeze(0)
                else:
                    segmentations, logits = self.model(image)
        else:
            if frozen:
                if self.training and image.size(0) == 1:
                    image = image.repeat(2,1,1,1)
                    segmentations = self.frozen_model(image)
                    segmentations = segmentations[0].unsqueeze(0)
                elif self.training:
                    segmentations = self.frozen_model(image)
                else:
                    segmentations = self.model(image)

            else:
                if self.training and image.size(0) == 1:
                    image = image.repeat(2,1,1,1)
                    segmentations = self.model(image)
                    segmentations = segmentations[0].unsqueeze(0)
                else:
                    segmentations = self.model(image)
        
        if targets != None:
            target_mask, non_target_mask = extract_masks(segmentations, targets, gpu=self.gpu) # [batch_size, height, width]
        else:
            target_mask = non_target_mask = None

        weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
        mask_logits = weighted_segmentations.sum(dim=(2,3))
        if not self.aux_classifier:
            logits = mask_logits
            

        
        #mask_logits = segmentations.mean((2,3))
        return segmentations, target_mask, non_target_mask, logits, mask_logits

    def check_loss_schedulings(self):
        if self.background_loss_scheduling <= self.i:
            self.use_background_loss = True
        if self.similarity_loss_scheduling <= self.i:
            self.use_similarity_loss = True
        if self.mask_loss_scheduling <= self.i:
            self.use_mask_area_loss = True


    def training_step(self, batch, batch_idx):
        if self.use_loss_scheduling:
            self.check_loss_schedulings()
        
        if self.dataset in ['TOY', 'VOC', 'VOC2012', 'OI_SMALL', 'OI', 'OI_LARGE', 'TOY_MULTI']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)

        if self.frozen and (self.use_similarity_loss or self.use_background_loss):
           self.frozen_model = deepcopy(self.model)
           for _,p in self.frozen_model.named_parameters():
               p.requires_grad_(False)

        output = self(image, target_vector)

        if self.use_similarity_loss or self.use_background_loss:
            classification_loss = self.classification_loss_fn(output['image'][3], target_vector)
        else:
            classification_loss = self.classification_loss_fn(output[3], target_vector)
        
        self.log('classification_loss', classification_loss.item(), on_epoch=False)   
        loss = classification_loss
        
        obj_back_loss = torch.tensor(0., device=image.device)
        # Foreground similarity loss
        if self.use_similarity_loss:
            if self.multilabel:
                sim_loss = similarity_loss_fn(output, target_vector, self.similarity_loss_fn, self.similarity_regularizer, mode=self.similarity_loss_mode)
            else:
                detached = output['image'][3].detach()
                probs = torch.nn.functional.softmax(detached, dim=-1)
                sim_loss = self.similarity_regularizer * self.classification_loss_fn(output['object_0'][3], probs)            

            self.log('similarity_loss', sim_loss.item(), on_epoch=False)
            obj_back_loss += sim_loss.squeeze()
            
        # Background entropy loss (in the final Self-explainer, 'entropy' loss was used)
        if self.use_background_loss:
            if self.background_loss == 'entropy':
                background_entropy_loss = self.bg_loss_regularizer * entropy_loss(output['background'][3])
            else:
                background_entropy_loss = self.bg_loss_regularizer * bg_loss(output['background'][0], target_vector, self.background_loss)

            self.log('background_loss', background_entropy_loss.item(), on_epoch=False)
            obj_back_loss += background_entropy_loss.squeeze() # Entropy loss is negative, so is added to loss here but actually its subtracted


        # Mask losses (TV loss is not used in final Self-explainer)
        mask_loss = torch.tensor(0., device=image.device)
        mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(output['image'][1] if 'image' in output else output[1])) #+ self.total_variation_conv(s_mask))
        self.log('TV mask loss', mask_variation_loss.item(), on_epoch=False)
        if self.use_mask_variation_loss:
            mask_loss += mask_variation_loss

        bounding_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0] if 'image' in output else output[0], target_vector)) #+ self.class_mask_area_loss_fn(output['object'][0], target_vectori))
        self.log('Bounding area loss', bounding_area_loss.item(), on_epoch=False)
        if self.use_bounding_loss:
            mask_loss += bounding_area_loss

        mask_area_loss = self.mask_total_area_regularizer * ((output['image'][1] if 'image' in output else output[1]).mean()) 
        mask_area_loss += self.ncmask_total_area_regularizer * ((output['image'][2] if 'image' in output else output[2]).mean())
        self.log('Mean area loss', mask_area_loss.item(), on_epoch=False)
        if self.use_mask_area_loss:
            mask_loss += mask_area_loss


        # Used in combination with auxiliary classifier head (not used in final Self-explainer)
        if self.use_mask_logit_loss:
            mask_logit_loss = self.mask_logit_loss_regularizer * self.classification_loss_fn(output['image'][4] if 'image' in output else output[4], target_vector)
            self.log('Mask logit loss', mask_logit_loss.item(), on_epoch=False)
            loss += mask_logit_loss

        # Weighting scheme for foreground, background and mask losses
        if self.use_weighted_loss:
            if self.use_similarity_loss or self.use_background_loss:
                loss += weighted_loss(loss, obj_back_loss, self.object_loss_weighting_params[0], self.object_loss_weighting_params[1])      
            if self.use_mask_area_loss or self.use_bounding_loss:
                loss  += weighted_loss(loss, mask_loss, self.mask_loss_weighting_params[0], self.mask_loss_weighting_params[1])  
        else:
            loss += obj_back_loss + mask_loss

        # Tensorboard logging 
        if self.i % 5 == 4 and self.logger != None:
            masked_image = (output['image'][1] if 'image' in output else output[1]).detach().unsqueeze(1) * image
            self.logger.experiment.add_image('Train Masked Images', get_unnormalized_image(masked_image), self.i, dataformats='NCHW')
            self.logger.experiment.add_image('Train Images', get_unnormalized_image(image), self.i, dataformats='NCHW')
            self.logger.experiment.add_image('Train 1PassOutput', (output['image'][1] if 'image' in output else output[1]).detach().unsqueeze(1), self.i, dataformats='NCHW')
            log_string = ''
            logit_fn = torch.sigmoid if self.multilabel else lambda x: torch.nn.functional.softmax(x, dim=-1)
            self.logger.experiment.add_image('Train Nontarget mask', (output['image'][2] if 'image' in output else output[2]).detach().unsqueeze(1), self.i, dataformats='NCHW')
            
            log_string += f'Batch {0}:  \n'
            logits_list = [f'{i:.3f}' for i in (output['image'][3] if 'image' in output else output[3]).detach().tolist()[0]] 
            logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
            log_string += f'1Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'

            probs_list = [f'{i:.2f}' for i in logit_fn(output['image'][3] if 'image' in output else output[3]).detach()[0]]
            probs_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(probs_list)
            log_string += f'1Pass Probs:&nbsp;&nbsp;{probs_string}  \n'

            if self.use_similarity_loss:
                logits_list = [f'{i:.3f}' for i in output['object_0'][3].detach().tolist()[0]] 
                logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
                log_string += f'2Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'
            if self.use_background_loss:
                logits_list = [f'{i:.3f}' for i in output['background'][3].detach().tolist()[0]] 
                logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
                log_string += f'3Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'

            log_string += '  \n'
            self.logger.experiment.add_text('Train Logits', log_string,  self.i)
                    
        self.log('loss', float(loss.item()), on_epoch=False)
        self.train_metrics((output['image'][3] if 'image' in output else output[3]).detach(), target_vector.int())

        pytorch_lightning.utilities.memory.garbage_collection_cuda()

        self.i += 1.
        return loss

    def training_epoch_end(self, outs):
        m = self.train_metrics.compute()
        self.log('train_metric', m)
        for k,v in m.items():
            self.log(f'{k}', v, prog_bar=True, logger=False)
        self.train_metrics.reset()
        
        for g in self.trainer.optimizers[0].param_groups:
            self.log('lr', g['lr'], prog_bar=True)


    def validation_step(self, batch, batch_idx):
        if self.dataset in ['TOY', 'VOC', 'SMALLVOC', 'VOC2012', 'OI_SMALL', 'OI', 'OI_LARGE']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)

        
        output = self(image, target_vector)

        classification_loss = self.classification_loss_fn(output['image'][3], target_vector)
        self.log('val_classification_loss', classification_loss)        

        loss = classification_loss
        
        obj_back_loss = torch.zeros((1), device=loss.device)
        # Foreground similarity loss
        if self.use_similarity_loss:
            
            if self.multilabel:
                sim_loss = similarity_loss_fn(output, target_vector, self.similarity_loss_fn, self.similarity_regularizer, mode=self.similarity_loss_mode)
            else:
                detached = output['image'][3].detach()
                probs = torch.nn.functional.softmax(detached, dim=-1)
                sim_loss = self.similarity_regularizer * self.classification_loss_fn(output['object_0'][3], probs) 
                
            self.log('val_similarity_loss', sim_loss)
            
            obj_back_loss += sim_loss

        #Background entropy loss (in the final Self-explainer, 'entropy' loss was used)
        if self.use_background_loss:
            if self.background_loss == 'entropy':
                background_entropy_loss = self.bg_loss_regularizer * entropy_loss(output['background'][3])
            else:
                background_entropy_loss = self.bg_loss_regularizer * bg_loss(output['background'][0], target_vector, self.background_loss)

            self.log('val_background_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted

        # Mask losses
        mask_loss = torch.zeros((1), device=loss.device)
        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(output['image'][1])) #+ self.total_variation_conv(s_mask))
            mask_loss += mask_variation_loss

        bounding_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], target_vector)) 
        self.log('val_bounding_area_loss', bounding_area_loss)
        if self.use_bounding_loss:
            mask_loss += bounding_area_loss

        mask_area_loss = self.mask_total_area_regularizer * (output['image'][1].mean()) 
        mask_area_loss += self.ncmask_total_area_regularizer * (output['image'][2].mean())
        self.log('val_mask_area_loss', mask_area_loss)
        if self.use_mask_area_loss:
            mask_loss += mask_area_loss

        # Weighting scheme for foreground, background and mask losses
        if self.use_weighted_loss:
            if self.use_similarity_loss or self.use_background_loss:
                loss += weighted_loss(loss, obj_back_loss, self.object_loss_weighting_params[0], self.object_loss_weighting_params[1])
            if self.use_mask_variation_loss or self.use_mask_area_loss:
                loss += weighted_loss(loss, mask_loss, self.mask_loss_weighting_params[0], self.mask_loss_weighting_params[1])
        else:
            loss = loss + obj_back_loss + mask_loss

        # Tensorboard logging
        masked_image = output['image'][1].unsqueeze(1) * image
        self.logger.experiment.add_image('Val Masked Images', get_unnormalized_image(masked_image), self.val_i, dataformats='NCHW')
        self.log('val_loss', float(loss))
        self.logger.experiment.add_image('Val Images', get_unnormalized_image(image), self.val_i, dataformats='NCHW')
        self.logger.experiment.add_image('Val 1PassOutput', output['image'][1].unsqueeze(1), self.val_i, dataformats='NCHW')
        if self.use_similarity_loss:
            self.logger.experiment.add_image('Val 2PassOutput', output['object_0'][1].unsqueeze(1), self.val_i, dataformats='NCHW')
        self.logger.experiment.add_image('Val Nontarget mask', output['image'][2].unsqueeze(1), self.val_i, dataformats='NCHW')

        log_string = ''
        logit_fn = torch.sigmoid if self.multilabel else lambda x: torch.nn.functional.softmax(x, dim=-1)

        for b in range(image.size()[0]):
            log_string += f'Batch {b}:  \n'
            logits_list = [f'{i:.3f}' for i in output['image'][3].tolist()[b]] 
            logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
            log_string += f'1Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'

            probs_list = [f'{i:.2f}' for i in logit_fn(output['image'][3]).detach()[b]]
            probs_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(probs_list)
            log_string += f'1Pass Probs:&nbsp;&nbsp;{probs_string}  \n'

            if self.use_similarity_loss:
                logits_list = [f'{i:.3f}' for i in output['object_0'][3].tolist()[b]] 
                logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
                log_string += f'2Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'
            if self.use_background_loss:
                logits_list = [f'{i:.3f}' for i in output['background'][3].tolist()[b]] 
                logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
                log_string += f'3Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'
            log_string += '  \n'
        self.logger.experiment.add_text('Val Logits', log_string,  self.val_i)


        self.valid_metrics(output['image'][3], target_vector.int())
        self.val_i += 1
        return loss
   
    def validation_epoch_end(self, outs):
        m = self.valid_metrics.compute()
        self.log('valid_metric', m)
        for k,v in m.items():
            self.log(f'{k}', v, prog_bar=True, logger=False)
        self.valid_metrics.reset()


    def test_step(self, batch, batch_idx):
        self.test_i += 1
        if self.dataset in ['TOY', 'VOC', 'SMALLVOC', 'VOC2012', 'OI_SMALL', 'OI', 'OI_LARGE', 'TOY_SAVED', 'TOY_MULTI']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)

        output = self(image, target_vector)

        if self.save_masked_images and image.size()[0] == 1 and self.test_i < 1000:
            filename = Path(self.save_path) / "masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, output['image'][1], filename, self.dataset, output['image'][3][0].sigmoid() if self.multilabel else torch.nn.functional.softmax(output['image'][3][0], dim=-1))
            filename = Path(self.save_path) / "inverse_masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            inverse = torch.ones_like(output['image'][1]) - output['image'][1]
            save_masked_image(image, inverse, filename, self.dataset)

            filename = Path(self.save_path) / "images" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_image(image, filename, self.dataset)

        if self.save_masks and image.size()[0] == 1 and self.test_i < 1000:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)
            for k, v in output.items():
                save_mask(v[1], Path(self.save_path) / f'masks_{k}_pass' / filename, self.dataset)

        if self.dataset != 'COLOR' and self.test_i < 1000 and self.save_all_class_masks and image.size()[0] == 1:
            filename = Path(self.save_path) / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masks(output['image'][0], filename, dataset=self.dataset)
            # if self.use_background_loss:
            #     filename = Path(self.save_path) / "all_class_masks_background" / get_filename_from_annotations(annotations, dataset=self.dataset)
            #     save_all_class_masks(output['background'][0], filename, dataset=self.dataset)
             
            filename = Path(self.save_path) / "class_masked_images" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masked_images(image, output['image'][0], filename, self.dataset, target_vector)

        classification_loss = self.classification_loss_fn(output['image'][3], target_vector)

        loss = classification_loss
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][3], target_vector, output['image'][1], output['object_0'][1])
            loss += similarity_loss

        # Foreground similarity loss
        if self.use_similarity_loss:
            if self.multilabel:
                sim_loss = similarity_loss_fn(output, target_vector, self.similarity_loss_fn, self.similarity_regularizer, mode=self.similarity_loss_mode)
            else:
                detached = output['image'][3].detach()
                probs = torch.nn.functional.softmax(detached, dim=-1)
                sim_loss = self.similarity_regularizer * self.classification_loss_fn(output['object_0'][3], probs)            

            self.log('similarity_loss', sim_loss.item(), on_epoch=False)
            obj_back_loss += sim_loss.squeeze()
            
        # Background entropy loss (in the final Self-explainer, 'entropy' loss was used)
        if self.use_background_loss:
            if self.background_loss == 'entropy':
                background_entropy_loss = self.bg_loss_regularizer * entropy_loss(output['background'][3])
            else:
                background_entropy_loss = self.bg_loss_regularizer * bg_loss(output['background'][0], target_vector, self.background_loss)

            self.log('background_loss', background_entropy_loss.item(), on_epoch=False)
            obj_back_loss += background_entropy_loss.squeeze() # Entropy loss is negative, so is added to loss here but actually its subtracted


        # Mask losses (TV loss is not used in final Self-explainer)
        mask_loss = torch.tensor(0., device=image.device)
        mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(output['image'][1] if 'image' in output else output[1])) #+ self.total_variation_conv(s_mask))
        self.log('TV mask loss', mask_variation_loss.item(), on_epoch=False)
        if self.use_mask_variation_loss:
            mask_loss += mask_variation_loss


        bounding_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0] if 'image' in output else output[0], target_vector)) #+ self.class_mask_area_loss_fn(output['object'][0], target_vectori))
        self.log('Bounding area loss', bounding_area_loss.item(), on_epoch=False)
        if self.use_bounding_loss:
            mask_loss += bounding_area_loss

        mask_area_loss = self.mask_total_area_regularizer * ((output['image'][1] if 'image' in output else output[1]).mean()) 
        mask_area_loss += self.ncmask_total_area_regularizer * ((output['image'][2] if 'image' in output else output[2]).mean())
        self.log('Mean area loss', mask_area_loss.item(), on_epoch=False)
        if self.use_mask_area_loss:
            mask_loss += mask_area_loss


        # Used in combination with auxiliary classifier head (not used in final Self-explainer)
        if self.use_mask_logit_loss:
            mask_logit_loss = self.mask_logit_loss_regularizer * self.classification_loss_fn(output['image'][4] if 'image' in output else output[4], target_vector)
            self.log('Mask logit loss', mask_logit_loss.item(), on_epoch=False)
            loss += mask_logit_loss

        # Weighting scheme for foreground, background and mask losses
        if self.use_weighted_loss:
            if self.use_similarity_loss or self.use_background_loss:
                loss += weighted_loss(loss, obj_back_loss, self.object_loss_weighting_params[0], self.object_loss_weighting_params[1])
                
            if self.use_mask_area_loss or self.use_bounding_loss:
                loss  += weighted_loss(loss, mask_loss, self.mask_loss_weighting_params[0], self.mask_loss_weighting_params[1])  
        else:
            loss += obj_back_loss + mask_loss
        
        self.log('test_loss', loss)
        self.test_metrics(output['image'][3], target_vector.int())

    def test_epoch_end(self, outs):
        a_m = self.test_metrics.compute()
        m = a_m['Total']
        self.log('test_metric', m)
        for k,v in m.items():
            self.log(f'{k}', v, prog_bar=True, logger=False)

        with open(Path(self.save_path) / 'plots' / 'class_metrics.json', 'w') as jsonfile:
            a_m = dict_tensor_to_list(a_m)
            json.dump({'Classes': list(get_class_dictionary(self.dataset).keys()), "Metrics": a_m}, jsonfile)
        
        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
  
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
