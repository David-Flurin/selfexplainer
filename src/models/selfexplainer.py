import torch
import pytorch_lightning as pl

from torch import nn, softmax
from torch.optim import Adam
from pathlib import Path

from copy import deepcopy

from models.DeepLabv3 import Deeplabv3Resnet50Model
from utils.helper import get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution
from utils.image_display import save_all_class_masks, save_mask, save_masked_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics
from utils.weighting import softmax_weighting

class SelfExplainer(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1, use_similarity_loss=True, use_entropy_loss=True, gpu=0, profiler=None, metrics_threshold=-1.0, save_path="./results/"):

        super().__init__()

        self.gpu = gpu
        self.profiler = profiler

        self.learning_rate = learning_rate
        self.weighting_koeff = weighting_koeff

        self.model = Deeplabv3Resnet50Model(num_classes=num_classes)
        self.frozen = None
        self.dataset = dataset

        self.use_similarity_loss = use_similarity_loss
        self.use_entropy_loss = use_entropy_loss

        if self.dataset == 'TOY':
            self.f_tex_dist = Distribution()
            self.b_text_dist = Distribution()
            self.shapes_dist = Distribution()

        self.setup_losses(dataset=dataset)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        self.i = 0.

    def setup_losses(self, dataset):
        if dataset == "CUB":
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()

    def setup_metrics(self, num_classes, metrics_threshold):
        if self.dataset == "CUB":
            self.train_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.valid_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.test_metrics = SingleLabelMetrics(num_classes=num_classes)
        else:
            self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    def forward(self, image, targets):
        output = {}
        output['image'] = self._forward(image, targets)
        
        i_mask = output['image'][1]
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
        return segmentations, target_mask, non_target_mask, logits

    def measure_weighting(self, segmentations):
        with self.profiler.profile("softmax_weighing"):
            weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
            logits = weighted_segmentations.sum(dim=(2,3))
        return logits
        
    def training_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)

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

        classification_loss = classification_loss_initial
        self.log('classification_loss', classification_loss)

        loss = classification_loss
        
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][1], output['object'][1])
            loss += similarity_loss

        if self.use_entropy_loss:
            background_entropy_loss = entropy_loss(output['background'][3])
            self.log('background entropy loss', background_entropy_loss)
            loss += background_entropy_loss
        # if self.use_mask_variation_loss:
        #     mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
        #     loss += mask_variation_loss

        # if self.use_mask_area_loss:
        #     mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(t_seg, targets) + self.class_mask_area_loss_fn(s_seg, targets))
        #     mask_area_loss += self.mask_total_area_regularizer * (t_mask.mean() + s_mask.mean())
        #     mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
        #     loss += mask_area_loss

        # if self.use_mask_coherency_loss:
        #     mask_coherency_loss = (t_mask - s_mask).abs().mean()
        #     loss += mask_coherency_loss

        self.i += 1.
        self.log('iterations', self.i, prog_bar=True)

        self.log('loss', float(loss))
       
        '''
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            self.valid_metrics(o_logits, labels)
        else:
            self.valid_metrics(o_logits, targets)
        '''
        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()
        self.f_tex_dist.print_distribution()
        self.b_text_dist.print_distribution()
        self.shapes_dist.print_distribution()

    def validation_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)
        
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
            self.log('background entropy loss', background_entropy_loss)
            loss += background_entropy_loss
        # if self.use_mask_variation_loss:
        #     mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
        #     loss += mask_variation_loss

        # if self.use_mask_area_loss:
        #     mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(t_seg, targets) + self.class_mask_area_loss_fn(s_seg, targets))
        #     mask_area_loss += self.mask_total_area_regularizer * (t_mask.mean() + s_mask.mean())
        #     mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
        #     loss += mask_area_loss

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
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, gpu=self.gpu)
        t_seg, s_seg, t_mask, s_mask, t_ncmask, s_ncmask, t_logits, s_logits = self(image, targets)

        if self.save_masked_images and image.size()[0] == 1:
            filename = self.save_path / "masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, s_mask, filename)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)

            save_mask(s_mask, self.save_path / "masks" / filename)

        if self.save_all_class_masks and image.size()[0] == 1 and self.dataset == "VOC":
            filename = self.save_path / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masks(image, t_seg, filename)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_teacher = self.classification_loss_fn(t_logits, labels)
            classification_loss_student = self.classification_loss_fn(s_logits, labels)
        else:
            classification_loss_teacher = self.classification_loss_fn(t_logits, targets)
            classification_loss_student = self.classification_loss_fn(s_logits, targets)

        logits_similarity_loss = (t_logits - s_logits).abs().mean()
        classification_loss = classification_loss_teacher + classification_loss_student + logits_similarity_loss

        loss = classification_loss

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

        self.log('test_loss', loss)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            self.test_metrics(s_logits, labels)
        else:
            self.test_metrics(s_logits, targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
