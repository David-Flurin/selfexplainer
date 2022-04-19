import os.path as osp

from torch.optim import Adam


import torch.nn as nn
import torch
import numpy as np

import pytorch_lightning as pl

from utils.helper import get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics
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
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

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

        self._initialize_weights()

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

    def setup_metrics(self, num_classes, metrics_threshold):
        if self.dataset == "CUB":
            self.train_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.valid_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.test_metrics = SingleLabelMetrics(num_classes=num_classes)
        else:
            self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    def model_forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

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
            segmentations = self.model_forward(image) # [batch_size, num_classes, height, width]
        target_mask, non_target_mask = extract_masks(segmentations, targets, gpu=self.gpu) # [batch_size, height, width]
        
        weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
        logits = weighted_segmentations.sum(dim=(2,3))
        #logits_mean = segmentations.mean((2,3))

        return segmentations, target_mask, non_target_mask, logits

    def training_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)

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

        classification_loss_initial = self.classification_loss_fn(output['image'][3], targets)
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
                loss = weighted_loss(classification_loss, obj_back_loss, 2, 0.2)
            else:
                loss = classification_loss + obj_back_loss
        else:
            loss = classification_loss

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
       
        self.train_metrics(output['image'][3], targets)
        return loss


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('frozen'):
                del checkpoint['state_dict'][k]
