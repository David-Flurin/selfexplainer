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
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss, bg_loss, background_activation_loss, relu_classification, similarity_loss_fn
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics, ClassificationMultiLabelMetrics
from utils.weighting import softmax_weighting
from evaluation.compute_scores import selfexplainer_compute_numbers

import GPUtil
from matplotlib import pyplot as plt

VOC_segmentations_path = Path("../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/")
SMALLVOC_segmentations_path = Path("../../datasets/VOC2007_small/VOCdevkit/VOC2007/SegmentationClass/")


class BaseModel(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1., pretrained=False, use_similarity_loss=False, similarity_regularizer=1.0, use_background_loss=False, bg_loss_regularizer=1.0, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.05, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, use_perfect_mask=False, count_logits=False, save_masks=False, save_all_class_masks=False, 
                 gpu=0, profiler=None, metrics_threshold=0.5, save_path="./results/", objective='classification', class_loss='bce', frozen=False, freeze_every=20, background_activation_loss=False, bg_activation_regularizer=0.5, target_threshold=0.7, non_target_threshold=0.3, background_loss='logits_ce', aux_classifier=False, multiclass=False, use_bounding_loss=False):

        super().__init__()

        self.gpu = gpu
        self.profiler = profiler

        self.learning_rate = learning_rate
        self.weighting_koeff = weighting_koeff

        self.frozen = None
        self.dataset = dataset
        self.num_classes = num_classes

        self.aux_classifier = aux_classifier


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
        self.use_bounding_loss = use_bounding_loss
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

        self.frozen = frozen
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

    def forward(self, image, targets, perfect_mask = None):
        output = {}
        output['image'] = self._forward(image, targets)

        i_mask = output['image'][1]
        if perfect_mask != None:
            i_mask = perfect_mask
        
        if self.use_similarity_loss:
            masked_image = i_mask.unsqueeze(1) * image
            if self.i % 5 == 4:
                self.logger.experiment.add_image('Masked Images', get_unnormalized_image(masked_image), self.i, dataformats='NCHW')
            output['object'] = self._forward(masked_image, targets, frozen=self.frozen)
            sanity_check = self._forward(torch.zeros_like(masked_image), targets, frozen=False)
        
        if self.use_background_loss:   
            target_mask_inversed = torch.ones_like(i_mask) - i_mask
            if image.dim() > 3:
                target_mask_inversed = target_mask_inversed.unsqueeze(1)
            inverted_masked_image = target_mask_inversed * image
            # from matplotlib import pyplot as plt
            # fig = plt.figure(figsize=(10,10))
            # for b in range(image.size()[0]):
            #     fig.add_subplot(b+1,3,b*3+1)
            #     plt.imshow(image[b].detach().transpose(0,2))
            #     fig.add_subplot(b+1,3,b*3+2)
            #     plt.imshow(i_mask[b].detach().transpose(0,1))
            #     fig.add_subplot(b+1,3,b*3+3)
            #     plt.imshow(inverted_masked_image[b].detach().transpose(0,2))
            # plt.show()
            output['background'] = self._forward(inverted_masked_image, targets, frozen=self.frozen)
            
        return output

    def _forward(self, image, targets, frozen=False):
        if self.aux_classifier:
            if frozen:
                segmentations, logits = self.frozen(image)
            else:
                segmentations, logits = self.model(image) # [batch_size, num_classes, height, width]
        else:
            if frozen:
                segmentations = self.frozen(image)
            else:
                segmentations = self.model(image) # [batch_size, num_classes, height, width]
        
        target_mask, non_target_mask = extract_masks(segmentations, targets, gpu=self.gpu) # [batch_size, height, width]

        if not self.aux_classifier:
            weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
            logits = weighted_segmentations.sum(dim=(2,3))
        
        # logits = segmentations.mean((2,3))
        

        return segmentations, target_mask, non_target_mask, logits



    def measure_weighting(self, segmentations):
        with self.profiler.profile("softmax_weighing"):
            weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
            logits = weighted_segmentations.sum(dim=(2,3))
        return logits
        
    def training_step(self, batch, batch_idx):
        #GPUtil.showUtilization()
        
        if self.dataset in ['VOC', 'SMALLVOC', 'OISMALL']:
            image, annotations = batch
        else:
            image, seg, annotations = batch
            targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)


        if self.frozen and self.i % self.freeze_every == 0 and (self.use_similarity_loss or self.use_background_loss):
           self.frozen = deepcopy(self.model)
           for _,p in self.frozen.named_parameters():
               p.requires_grad_(False)

        
        # if self.first_of_epoch:
        #     self.first_of_epoch = False
        # for b in range(image.size()[0]):
        #     save_image(image[b], f'check_toydata/{b}_{random.randint(0, 10000)}.png', self.dataset)


        
        if self.use_perfect_mask:
            output = self(image, target_vector, torch.max(targets, dim=1)[0])
        else:
            output = self(image, target_vector)

        
        #print(output['image'][3])
        #print(target_vector)

        if self.use_background_loss:
            self.test_background_logits.append(output['background'][3].sum().item())

 


        if self.objective == 'classification':
            classification_loss_initial = self.classification_loss_fn(output['image'][3], target_vector)
            #classification_loss_object = self.classification_loss_fn(o_logits, targets)
            #classification_loss_background = self.classification_loss_fn(b_logits, targets)
        elif self.objective == 'segmentation':
            targets = targets.to(torch.float64)
            classification_loss_initial = self.classification_loss_fn(output['image'][0], targets)
        else:
            raise ValueError('Unknown objective')
        
        #print(classification_loss_initial)

        classification_loss = classification_loss_initial
        self.log('classification_loss', classification_loss)   

        if self.use_similarity_loss:
            self.log('classification_loss_1Pass', classification_loss)   
            self.log('classification_loss_2Pass', self.classification_loss_fn(output['object'][3], target_vector))     

        loss = classification_loss
        
        obj_back_loss = torch.zeros((1), device=loss.device)
        if self.use_similarity_loss:
            logit_fn = torch.sigmoid if self.multiclass else lambda x: torch.nn.functional.softmax(x, dim=-1)
            #similarity_loss = self.similarity_regularizer * similarity_loss_fn(logit_fn(output['image'][3].detach()),  logit_fn(output['object'][3]))
            similarity_loss = self.similarity_regularizer * self.classification_loss_fn(output['object'][3], logit_fn(output['image'][3]).detach())
            self.log('similarity_loss', similarity_loss)

            obj_back_loss += similarity_loss

        if self.use_background_loss:
            if self.bg_loss == 'entropy':
                background_entropy_loss = self.bg_loss_regularizer * entropy_loss(output['background'][3])
            elif self.bg_loss == 'distance':
                    background_entropy_loss = self.bg_loss_regularizer * bg_loss(output['background'][0], target_vector, self.background_loss)

            self.log('background_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted


        

        mask_loss = torch.zeros((1), device=loss.device)
        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(output['image'][1])) #+ self.total_variation_conv(s_mask))
            self.log('TV mask loss', mask_variation_loss)
            mask_loss += mask_variation_loss


        if self.use_bounding_loss:
            bounding_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], target_vector)) #+ self.class_mask_area_loss_fn(output['object'][0], target_vectori))
            self.log('Bounding area loss', bounding_area_loss)
            mask_loss += bounding_area_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_total_area_regularizer * (output['image'][1].mean()) #+ output['object'][1].mean())
            mask_area_loss += self.ncmask_total_area_regularizer * (output['image'][2].mean()) #+ output['object'][2].mean())
            self.log('Mean area loss', mask_area_loss)
            mask_loss += mask_area_loss


        if self.use_similarity_loss or self.use_background_loss:
            if self.use_weighted_loss:
                w_obj_back_loss = weighted_loss(loss, obj_back_loss, 5, 0.1)
                self.log('weighted_loss', w_obj_back_loss)
                w_mask_loss = weighted_loss(loss, mask_loss, 5, 0.1)
                self.log('Weighted mask losses', w_mask_loss)
                loss += w_obj_back_loss + w_mask_loss
                #w_m_loss = weighted_loss(loss, mask_loss, 10, 0.1)
                #self.log('weighted mask loss', w_m_loss)
                #loss += w_m_loss
            else:
                loss = loss + obj_back_loss + mask_loss

        if self.use_background_activation_loss:
            bg_logits_loss = self.bg_activation_regularizer * background_activation_loss(output['image'][1])
            self.log('bg_logits_loss', bg_logits_loss)
            loss = weighted_loss(loss, bg_logits_loss, 2, 0.1)
        
        

        if self.i % 5 == 4 and self.use_similarity_loss:
            self.logger.experiment.add_image('Train Images', get_unnormalized_image(image), self.i, dataformats='NCHW')
            self.logger.experiment.add_image('Train 1PassOutput', output['image'][1].unsqueeze(1), self.i, dataformats='NCHW')
            self.logger.experiment.add_image('Train 2PassOutput', output['object'][1].unsqueeze(1), self.i, dataformats='NCHW')
            log_string = ''
            logit_fn = torch.sigmoid if self.multiclass == 'bce' else lambda x: torch.nn.functional.softmax(x, dim=-1)
            
            log_string += f'Batch {0}:  \n'
            logits_list = [f'{i:.3f}' for i in output['image'][3].tolist()[0]] 
            logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
            log_string += f'1Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'

            probs_list = [f'{i:.2f}' for i in logit_fn(output['image'][3]).detach()[0]]
            probs_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(probs_list)
            log_string += f'1Pass Probs:&nbsp;&nbsp;{probs_string}  \n'

            logits_list = [f'{i:.3f}' for i in output['object'][3].tolist()[0]] 
            logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
            log_string += f'2Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'
            log_string += '  \n'
            self.logger.experiment.add_text('Train Logits', log_string,  self.i)

        self.log('loss', float(loss))
        
        self.train_metrics(output['image'][3], target_vector.int())

        #DEBUG

        #Save min and max logits
        if self.count_logits:
            for k, v in output.items():
                self.logit_stats[k].update(v[3])

        # self.global_image_mask = output['image'][1]
        # self.global_object_mask = output['object'][1]
                
        #GPUtil.showUtilization()  
        #d = make_dot(loss, params=dict(self.model.named_parameters())) 
        #d.render('backward_graph_unfrozen', format='png')  
        # output['image'][1].retain_grad()
        # output['object'][1].retain_grad()

        # o = self.optimizers()
        # self.manual_backward(loss)
        # o.step()
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

        
        if self.frozen and self.i % self.freeze_every == 0 and (self.use_similarity_loss or self.use_background_loss):
           self.frozen = deepcopy(self.model)
           for _,p in self.frozen.named_parameters():
               p.requires_grad_(False)
        
        if self.use_perfect_mask:
            output = self(image, target_vector, torch.max(targets, dim=1)[0])
        else:
            output = self(image, target_vector)


        if self.use_background_loss:
            self.test_background_logits.append(output['background'][3].sum().item())


        if self.objective == 'classification':
            classification_loss_initial = self.classification_loss_fn(output['image'][3], target_vector)
            #classification_loss_object = self.classification_loss_fn(o_logits, targets)
            #classification_loss_background = self.classification_loss_fn(b_logits, targets)
        elif self.objective == 'segmentation':
            targets = targets.to(torch.float64)
            classification_loss_initial = self.classification_loss_fn(output['image'][0], targets)
        else:
            raise ValueError('Unknown objective')
        
        classification_loss = classification_loss_initial
        self.log('val_classification_loss', classification_loss)        

        loss = classification_loss
        
        obj_back_loss = torch.zeros((1), device=loss.device)
        if self.use_similarity_loss:
            #similarity_loss = self.similarity_regularizer * mask_similarity_loss(output['object'][3], target_vector, output['image'][1], output['object'][1])
            logit_fn = torch.sigmoid if self.multiclass == 'bce' else lambda x: torch.nn.functional.softmax(x, dim=-1)
            similarity_loss = self.classification_loss_fn(output['object'][3], logit_fn(output['image'][3]).detach())
            self.log('val_similarity_loss', similarity_loss)

            obj_back_loss += similarity_loss

        if self.use_background_loss:
            if self.bg_loss == 'entropy':
                background_entropy_loss = self.bg_loss_regularizer * entropy_loss(output['background'][3])
            elif self.bg_loss == 'distance':
                    background_entropy_loss = self.bg_loss_regularizer * bg_loss(output['background'][0], target_vector, self.background_loss)

            self.log('val_background_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted


        

        mask_loss = torch.zeros((1), device=loss.device)
        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(output['image'][1])) #+ self.total_variation_conv(s_mask))
            mask_loss += mask_variation_loss

            #mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], target_vector)) #+ self.class_mask_area_loss_fn(output['object'][0], target_vector))
            mask_area_loss = self.mask_total_area_regularizer * (output['image'][1].mean()) #+ output['object'][1].mean())
            #mask_area_loss += self.ncmask_total_area_regularizer * (output['image'][2].mean()) #+ output['object'][2].mean())
            self.log('mask_area_loss', mask_area_loss)
            if self.use_mask_area_loss:
                mask_loss += mask_area_loss

        if self.use_similarity_loss or self.use_background_loss or self.use_mask_variation_loss or self.use_mask_area_loss:
            if self.use_weighted_loss:
                w_loss = weighted_loss(loss, obj_back_loss + mask_loss, 2, 0.2)
                self.log('Weighted loss', w_loss)
                loss += w_loss
            else:
                loss = loss + obj_back_loss + mask_loss

        if self.use_background_activation_loss:
            bg_logits_loss = self.bg_activation_regularizer * background_activation_loss(output['image'][1])
            self.log('val_bg_logits_loss', bg_logits_loss)
            loss = weighted_loss(loss, bg_logits_loss, 2, 0.1)
        

        if self.use_similarity_loss:
            self.log('val_loss', float(loss))
            self.logger.experiment.add_image('Val Images', get_unnormalized_image(image), self.i, dataformats='NCHW')
            self.logger.experiment.add_image('Val 1PassOutput', output['image'][1].unsqueeze(1), self.i, dataformats='NCHW')
            self.logger.experiment.add_image('Val 2PassOutput', output['object'][1].unsqueeze(1), self.i, dataformats='NCHW')
            log_string = ''
            for b in range(image.size()[0]):
                log_string += f'Batch {b}:  \n'
                logits_list = [f'{i:.3f}' for i in output['image'][3].tolist()[b]] 
                logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
                log_string += f'1Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'

                probs_list = [f'{i:.2f}' for i in logit_fn(output['image'][3]).detach()[b]]
                probs_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(probs_list)
                log_string += f'1Pass Probs:&nbsp;&nbsp;{probs_string}  \n'

                logits_list = [f'{i:.3f}' for i in output['object'][3].tolist()[b]] 
                logits_string = ",&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(logits_list)
                log_string += f'2Pass:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{logits_string}  \n'
                log_string += '  \n'
            self.logger.experiment.add_text('Val Logits', log_string,  self.i)

       
        self.valid_metrics(output['image'][3], target_vector.int())

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

        output = self(image, target_vector)

        

        # from matplotlib import pyplot as plt
        # plt.imshow(image[0].permute(1,2,0))
        # plt.show()
        

        # from matplotlib import pyplot as plt
        # fig = plt.figure(figsize=(10, 5))
        # fig.add_subplot(1,3,1)
        # plt.imshow(image[0])
        # fig.add_subplot(1,3,2)
        # plt.imshow(output['image'][0][0][0])
        # fig.add_subplot(1,3,3)
        # plt.imshow(output['image'][0][0][1])
        # plt.show()

        if self.save_masked_images and image.size()[0] == 1:
            filename = Path(self.save_path) / "masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, output['image'][1], filename, self.dataset)
            filename = Path(self.save_path) / "inverse_masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            inverse = torch.ones_like(output['image'][1]) - output['image'][1]
            save_masked_image(image, inverse, filename, self.dataset)

            filename = Path(self.save_path) / "images" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_image(image, filename, self.dataset)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)

            for k, v in output.items():
                save_mask(v[1], Path(self.save_path) / f'masks_{k}_pass' / filename, self.dataset)


        if self.test_i < 21 and self.save_all_class_masks and image.size()[0] == 1:
            filename = Path(self.save_path) / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masks(output['image'][0], filename, dataset=self.dataset)
            if self.use_background_loss:
                filename = Path(self.save_path) / "all_class_masks_background" / get_filename_from_annotations(annotations, dataset=self.dataset)
                save_all_class_masks(output['background'][0], filename, dataset=self.dataset)



        classification_loss = self.classification_loss_fn(output['image'][3], target_vector)
        # print(classification_loss)
        # print(output['image'][3])
        # print(target_vector)
        loss = classification_loss
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][3], target_vector, output['image'][1], output['object'][1])
            loss += similarity_loss

        if self.use_background_loss:
            if self.bg_loss == 'entropy':
                background_entropy_loss = self.bg_loss_regularizer * entropy_loss(output['background'][3])
            elif self.bg_loss == 'distance':
                background_entropy_loss = self.bg_loss_regularizer * bg_loss(output['background'][0], target_vector, self.background_loss)
            loss += background_entropy_loss

        os.makedirs(os.path.dirname(Path(self.save_path) / "test_losses" / filename), exist_ok=True)
        with open(Path(self.save_path) / "test_losses" / filename, 'w') as f:
            f.write(f'classification loss: {classification_loss}\n')
            if self.use_similarity_loss:
                f.write(f'similarity loss: {similarity_loss}\n')
            if self.use_background_loss:
                f.write(f'background loss: {background_entropy_loss}\n')

        '''
        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(t_seg, targets) + self.class_mask_area_loss_fn(s_seg, targets))
            mask_area_loss += self.mask_total_area_regularizer * (t_mask.mean() + s_mask.mean())
            mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
            loss += mask_area_loss

        if self.use_mask_coherency_loss:
            mask_coherency_loss = (t_mask - s_mask).abs().mean()
            loss += mask_coherency_loss
        '''

        self.log('test_loss', loss)

        



        self.test_metrics(output['image'][3], target_vector.int())

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


    # def configure_optimizers(self):
    #     return Adam(self.parameters(), lr=self.learning_rate)
    
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
    
    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('frozen'):
                del checkpoint['state_dict'][k]


