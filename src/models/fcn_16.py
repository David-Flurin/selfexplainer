import os.path as osp

from torch.optim import Adam
from pathlib import Path

from torchvision import models


import torch.nn as nn
import torch
import numpy as np

import pytorch_lightning as pl

from utils.helper import get_filename_from_annotations, get_targets_from_annotations, get_targets_from_segmentations, extract_masks, Distribution
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss
from utils.image_display import save_all_class_masks, save_mask, save_masked_image
from utils.segmentationmetric import MultiLabelSegmentationMetrics
from utils.weighting import softmax_weighting

from copy import deepcopy

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN16(pl.LightningModule):

    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1, pretrained=False, use_similarity_loss=False, use_entropy_loss=False, use_weighted_loss=False,
    use_mask_area_loss=True, mask_area_constraint_regularizer=1.0, mask_total_area_regularizer=0.1, save_masked_images=False, save_masks=False, save_all_class_masks=False, gpu=0, profiler=None, metrics_threshold=-1.0, save_path="./results/"):
        super(FCN16, self).__init__()

        self.gpu = gpu
        self.profiler = profiler

        self.learning_rate = learning_rate
        self.weighting_koeff = weighting_koeff

        self.frozen = None
        self.dataset = dataset
        self.num_classes = num_classes

        self.model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=num_classes, progress=True)

        self.use_similarity_loss = use_similarity_loss
        self.use_entropy_loss = use_entropy_loss
        self.use_weighted_loss = use_weighted_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_total_area_regularizer = mask_total_area_regularizer

        self.save_path = save_path
        self.save_masked_images = save_masked_images
        self.save_masks = save_masks
        self.save_all_class_masks = save_all_class_masks

        if self.dataset == 'TOY':
            self.f_tex_dist = Distribution()
            self.b_text_dist = Distribution()
            self.shapes_dist = Distribution()

        self.setup_losses(dataset=dataset)
        self.setup_metrics()

        self.i = 0.

        self._init_model()

    def _init_model(self):

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, self.num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, self.num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            self.num_classes, self.num_classes, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            self.num_classes, self.num_classes, 32, stride=16, bias=False)

        #self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def setup_losses(self, dataset):
        if dataset == "CUB":
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()

    def setup_metrics(self):
        self.train_metrics = MultiLabelSegmentationMetrics()
        self.valid_metrics = MultiLabelSegmentationMetrics()
        self.test_metrics = MultiLabelSegmentationMetrics()

    def model_forward(self, x):
        # from matplotlib import pyplot as plt
        # plt.imshow(x[0].transpose(0,2))
        # plt.show()
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c

        h = self.upscore16(h)
        h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()

        return h

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
            segmentations = self.model(image)['out'] # [batch_size, num_classes, height, width]
        target_mask, non_target_mask = extract_masks(segmentations, targets, gpu=self.gpu) # [batch_size, height, width]
        
        weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
        logits = weighted_segmentations.sum(dim=(2,3))
        #logits_mean = segmentations.mean((2,3))



        return segmentations, target_mask, non_target_mask, logits

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
        
        output = self(image, target_vector)

        # from matplotlib import pyplot as plt
        # t = torch.max(output['image'][0][0].detach(), dim=0)[0]
        # # plt.imshow(output['image'][1][0].detach())
        # # plt.show()
        # for i in range(8):
        #     plt.imshow(output['image'][0][0][i].detach())
        #     plt.show()
        # for i in range(8):
        #     plt.imshow(targets[0][i])
        #     plt.show()

        # plt.imshow(torch.max(targets[0], dim=0)[0])
        # plt.show()

        seg_loss = self.classification_loss_fn(output['image'][0], targets)
        #classification_loss_object = self.classification_loss_fn(o_logits, targets)
        #classification_loss_background = self.classification_loss_fn(b_logits, targets)

        self.log('segmentation_loss', seg_loss)

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

        

        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][1], output['object'][1])
            self.log('similarity_loss', similarity_loss)
            obj_back_loss = similarity_loss

        if self.use_entropy_loss:
            background_entropy_loss = entropy_loss(output['background'][3])
            self.log('background_entropy_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted

        if self.use_similarity_loss or self.use_entropy_loss:
            if self.use_weighted_loss:
                loss = weighted_loss(seg_loss, obj_back_loss, 2, 0.2)
            else:
                loss = seg_loss + obj_back_loss
        else:
            loss = seg_loss

        # if self.use_mask_variation_loss:
        #     mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
        #     loss += mask_variation_loss

        # if self.use_mask_area_loss:
        #     #mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], targets) + self.class_mask_area_loss_fn(output['object'][0], targets))
        #     mask_area_loss = self.mask_total_area_regularizer * (output['image'][1].mean() + output['object'][1].mean())
        #     #mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
        #     self.log('mask_area_loss', mask_area_loss)
        #     loss += mask_area_loss

        # if self.use_mask_coherency_loss:
        #     mask_coherency_loss = (t_mask - s_mask).abs().mean()
        #     loss += mask_coherency_loss

        # self.i += 1.
        # self.log('iterations', self.i)

        self.log('loss', float(loss))
       
        self.train_metrics(output['image'][0], targets)
        return loss

    def training_epoch_end(self, outs):
        for g in self.trainer.optimizers[0].param_groups:
            self.log('lr', g['lr'], prog_bar=True)

        self.log('train_metrics', self.train_metrics.compute(), prog_bar=(self.dataset=='TOY'))
        self.train_metrics.reset()

    def test_step(self, batch, batch_idx):
        image, seg, annotations = batch
        targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)

        output = self(image, target_vector)

        if self.save_masked_images and image.size()[0] == 1:
            filename = Path(self.save_path) / "masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, output['image'][1], filename)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)

            save_mask(output['image'][1], Path(self.save_path) / "masks" / filename)

        if self.save_all_class_masks and image.size()[0] == 1 and self.dataset == "VOC":
            filename = self.save_path / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masks(image, t_seg, filename)


        seg_loss = self.classification_loss_fn(output['image'][0], targets)

        loss = seg_loss
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

        self.test_metrics(output['image'][0], targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
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


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    if target.dim() > 3:
        b, h, w, c = target.size()
        n_target = torch.zeros(b, h, w)
        n_target[target.sum(3) > 0.] = 1.
        target = n_target

    if len(input.size()) < 4:
        input = input.unsqueeze(1)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = torch.nn.functional.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = torch.nn.functional.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss
