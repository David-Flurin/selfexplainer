import torch
import pytorch_lightning as pl
import os

from torch import nn, softmax
from torch.optim import Adam
from pathlib import Path

from copy import deepcopy

from models.DeepLabv3 import Deeplabv3Resnet50Model
from utils.helper import get_class_dictionary, get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution, get_targets_from_segmentations, LogitStats
from utils.image_display import save_all_class_masks, save_mask, save_masked_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics
from utils.weighting import softmax_weighting

import GPUtil
from matplotlib import pyplot as plt

class SelfExplainer(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1, pretrained=False, use_similarity_loss=False, use_entropy_loss=False, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.04, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, use_perfect_mask=False, count_logits=False, save_masks=False, save_all_class_masks=False, 
                 gpu=0, profiler=None, metrics_threshold=-1.0, save_path="./results/"):

        super().__init__()

        self.gpu = gpu
        self.profiler = profiler

        self.learning_rate = learning_rate
        self.weighting_koeff = weighting_koeff

        self.model = Deeplabv3Resnet50Model(num_classes=num_classes, pretrained=pretrained)
        self.frozen = None
        self.dataset = dataset
        self.num_classes = num_classes

        self.use_similarity_loss = use_similarity_loss
        self.use_entropy_loss = use_entropy_loss
        self.use_weighted_loss = use_weighted_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_total_area_regularizer = mask_total_area_regularizer
        self.ncmask_total_area_regularizer = ncmask_total_area_regularizer
        self.use_mask_variation_loss = use_mask_variation_loss
        self.mask_variation_regularizer = mask_variation_regularizer

        self.save_path = save_path
        self.save_masked_images = save_masked_images
        self.save_masks = save_masks
        self.save_all_class_masks = save_all_class_masks

        #DEBUG
        self.i = 0.
        self.use_perfect_mask = use_perfect_mask
        self.count_logits = count_logits

        if self.dataset == 'TOY':
            self.f_tex_dist = Distribution()
            self.b_text_dist = Distribution()
            self.shapes_dist = Distribution()

        self.setup_losses(dataset=dataset, class_mask_min_area=class_mask_min_area, class_mask_max_area=class_mask_max_area)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        GPUtil.showUtilization()

    def setup_losses(self, dataset, class_mask_min_area, class_mask_max_area):
        if dataset == "CUB":
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()

        self.total_variation_conv = TotalVariationConv()
        self.class_mask_area_loss_fn = ClassMaskAreaLoss(min_area=class_mask_min_area, max_area=class_mask_max_area)


    def setup_metrics(self, num_classes, metrics_threshold):
        if self.dataset == "CUB":
            self.train_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.valid_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.test_metrics = SingleLabelMetrics(num_classes=num_classes)
        else:
            self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

        if self.count_logits:
            self.logit_stats = {'image': LogitStats(self.num_classes)}
            if self.use_similarity_loss:
                self.logit_stats['object'] = LogitStats(self.num_classes)
            if self.use_entropy_loss:
                self.logit_stats['background'] = LogitStats(self.num_classes)

    def forward(self, image, targets, perfect_mask = None):
        output = {}
        output['image'] = self._forward(image, targets)
        

        i_mask = output['image'][1]
        if perfect_mask != None:
            i_mask = perfect_mask



        if self.use_similarity_loss:
            masked_image = i_mask.unsqueeze(1) * image
            output['object'] = self._forward(masked_image, targets, frozen=True)

        if self.use_entropy_loss:   
            target_mask_inversed = torch.ones_like(i_mask) - i_mask
            inverted_masked_image = target_mask_inversed.unsqueeze(1) * image
            output['background'] = self._forward(inverted_masked_image, targets, frozen=True)
            
        return output

    def _forward(self, image, targets, frozen=False):
        if frozen:
            segmentations = self.frozen(image)
        else:
            segmentations = self.model(image) # [batch_size, num_classes, height, width]
        target_mask, non_target_mask = extract_masks(segmentations, targets, gpu=self.gpu) # [batch_size, height, width]

        
        weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
        logits = weighted_segmentations.sum(dim=(2,3))
        #logits_mean = segmentations.mean((2,3))
        
        
        return segmentations, target_mask, non_target_mask, logits

    def measure_weighting(self, segmentations):
        with self.profiler.profile("softmax_weighing"):
            weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
            logits = weighted_segmentations.sum(dim=(2,3))
        return logits
        
    def training_step(self, batch, batch_idx):
        image, seg, annotations = batch
        targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)

        if self.dataset == 'TOY':
            for a in annotations:
                for obj in a['objects']:
                    self.shapes_dist.update(obj[0])
                    self.f_tex_dist.update(obj[1])
            self.b_text_dist.update(a['background'])
        
        if self.use_similarity_loss or self.use_entropy_loss:
            self.frozen = deepcopy(self.model)
            for _,p in self.frozen.named_parameters():
                p.requires_grad_(False)
        
        if self.use_perfect_mask:
            output = self(image, target_vector, torch.max(targets, dim=1)[0])
        else:
            output = self(image, target_vector)

        if self.dataset == "CUB":
            labels = target_vector.argmax(dim=1)
            classification_loss_initial = self.classification_loss_fn(output['image'][3], labels)
            #classification_loss_object = self.classification_loss_fn(o_logits, labels)
            #classification_loss_background = self.classification_loss_fn(b_logits, labels)
        else:
            classification_loss_initial = self.classification_loss_fn(output['image'][3], target_vector)
            #classification_loss_object = self.classification_loss_fn(o_logits, targets)
            #classification_loss_background = self.classification_loss_fn(b_logits, targets)

        classification_loss = classification_loss_initial
        self.log('classification_loss', classification_loss)

        #if classification_loss.item() > 0.5 and self.i > 20 and self.i % 2 == 0:
        # b_s,_,_,_ = image.size()
        # for b in range(2):
        #     f, axes = plt.subplots(3,8) 
        #     f.set_size_inches(12,8)
        #     for s in range(7):
        #         axes[0][s].imshow(output['image'][0][b][s].detach(), vmin=-5, vmax=5)
        #         axes[1][s].imshow(output['object'][0][b][s].detach(),vmin=-5, vmax=5)
        #     axes[0][7].imshow(output['image'][1][b].detach(), vmin=0, vmax=1)
        #     axes[1][7].imshow(output['object'][1][b].detach(), vmin=0, vmax=1)
        #     axes[2][0].imshow(image[b].T)
        # plt.show()

        loss = self.classification_loss_fn(output['image'][3], target_vector)

        
        obj_back_loss = torch.zeros((1), device=loss.device)
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][1], output['object'][1])
            print('Similarity loss', similarity_loss)
            self.log('similarity_loss', similarity_loss)
            obj_back_loss += similarity_loss

        if self.use_entropy_loss:
            background_entropy_loss = entropy_loss(output['background'][3])
            self.log('background_entropy_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted

        if self.use_similarity_loss or self.use_entropy_loss:
            if self.use_weighted_loss:
                loss = weighted_loss(classification_loss, obj_back_loss, 2, 0.2)
            else:
                loss = classification_loss + obj_back_loss

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(output['image'][1])) #+ self.total_variation_conv(s_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], target_vector)) #+ self.class_mask_area_loss_fn(output['object'][0], target_vector))
            mask_area_loss += self.mask_total_area_regularizer * (output['image'][1].mean()) #+ output['object'][1].mean())
            mask_area_loss += self.ncmask_total_area_regularizer * (output['image'][2].mean()) #+ output['object'][2].mean())
            self.log('mask_area_loss', mask_area_loss)
            loss += mask_area_loss

        # if self.use_mask_coherency_loss:
        #     mask_coherency_loss = (t_mask - s_mask).abs().mean()
        #     loss += mask_coherency_loss

        self.i += 1.
        self.log('iterations', self.i)

        self.log('loss', float(loss))
       
        self.train_metrics(output['image'][3], target_vector)

        #DEBUG

        #Save min and max logits
        if self.count_logits:
            for k, v in output.items():
                self.logit_stats[k].update(v[3])
                
        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute(), prog_bar=(self.dataset=='TOY'))
        self.train_metrics.reset()
        '''
        self.f_tex_dist.print_distribution()
        self.b_text_dist.print_distribution()
        self.shapes_dist.print_distribution()
        '''

    def validation_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)
        
        self.frozen = deepcopy(self.model)
        for _,p in self.frozen.named_parameters():
            p.requires_grad_(False)
        
        output = self(image, targets)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_initial = self.classification_loss_fn(output['image'][3], labels)
            #classification_loss_object = self.classification_loss_fn(o_logits, labels)
            #classification_loss_background = self.classification_loss_fn(b_logits, labels)
        else:
            classification_loss_initial = self.classification_loss_fn(output['image'][3], targets)
            #classification_loss_object = self.classification_loss_fn(o_logits, targets)
            #classification_loss_background = self.classification_loss_fn(b_logits, targets)

        classification_loss = classification_loss_initial #+ classification_loss_object + classification_loss_background

        loss = classification_loss
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][1], output['object'][1])
            loss += similarity_loss

        if self.use_entropy_loss:
            background_entropy_loss = entropy_loss(output['background'][3])
            self.log('background_entropy_loss', background_entropy_loss)
            loss += background_entropy_loss
        # if self.use_mask_variation_loss:
        #     mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
        #     loss += mask_variation_loss

        if self.use_mask_area_loss:
            #mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], targets) + self.class_mask_area_loss_fn(output['object'][0], targets))
            mask_area_loss = self.mask_total_area_regularizer * (output['image'][1].mean() + output['object'][1].mean())
            #mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
            self.log('mask_area_loss', mask_area_loss)
            loss += mask_area_loss

        # if self.use_mask_coherency_loss:
        #     mask_coherency_loss = (t_mask - s_mask).abs().mean()
        #     loss += mask_coherency_loss

        self.log('val_loss', loss)
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            self.valid_metrics(output['object'][3], labels)
        else:
            self.valid_metrics(output['object'][3], targets) 
   
    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        image, seg, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)
        output = self(image, targets)

        if self.save_masked_images and image.size()[0] == 1:
            filename = Path(self.save_path) / "masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, output['image'][1], filename)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)

            save_mask(output['image'][1], Path(self.save_path) / "masks" / filename)

        if self.save_all_class_masks and image.size()[0] == 1 and self.dataset == "VOC":
            filename = self.save_path / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masks(image, t_seg, filename)


        classification_loss = self.classification_loss_fn(output['image'][3], targets)

        loss = classification_loss
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][1], output['object'][1])
            loss += similarity_loss

        if self.use_entropy_loss:
            background_entropy_loss = entropy_loss(output['background'][3])
            #self.log('background entropy loss', background_entropy_loss)
            loss += background_entropy_loss

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

        



        self.test_metrics(output['image'][3], targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

        #DEBUG
        if self.count_logits and self.save_path:
            dir = self.save_path + '/logit_stats'
            os.makedirs(dir)
            class_dict = get_class_dictionary(self.dataset)
            for k,v in self.logit_stats.items():
                v.plot(dir + f'/{k}.png', list(class_dict.keys()))


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('frozen'):
                del checkpoint['state_dict'][k]
