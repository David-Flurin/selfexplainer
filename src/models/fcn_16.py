import os.path as osp
import os

from torch.optim import Adam
from pathlib import Path
import math
from torchvision import models


import torch.nn as nn
import torch
import numpy as np

import pytorch_lightning as pl

from utils.helper import get_filename_from_annotations, get_targets_from_annotations, get_targets_from_segmentations, extract_masks, Distribution, LogitStats, get_class_dictionary
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
    use_mask_area_loss=True, mask_area_constraint_regularizer=1.0, mask_total_area_regularizer=0.1, use_perfect_mask=False, count_logits=False, save_masked_images=False, save_masks=False, save_all_class_masks=False, gpu=0, profiler=None, metrics_threshold=-1.0, save_path="./results/"):
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

         #DEBUG
        self.i = 0.
        self.use_perfect_mask = use_perfect_mask
        self.count_logits = count_logits

        if self.dataset == 'TOY':
            self.f_tex_dist = Distribution()
            self.b_text_dist = Distribution()
            self.shapes_dist = Distribution()

        self.setup_losses(dataset=dataset)
        self.setup_metrics()

       

        self._init_model()

    def _init_model(self):

        self.conv0 = ConvBlock(1, 3, 64, kernel=7)

        # conv1
        self.conv1 = ConvBlock(2, 64, 128)

        # conv2
        self.conv2 = ConvBlock(2, 128, 256)

        # conv3
        self.conv3 = ConvBlock(3, 256, 512)

        self.fc4 = nn.Conv2d(512, 1024, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout2d()

        # conv4
        #self.conv4 = ConvBlock(3, 1024, 2048)


        # # fc7
        # self.fc7 = nn.Conv2d(4096, 4096, 1)
        # self.relu7 = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d()

        # self.score_fr = nn.Conv2d(4096, self.num_classes, 1)
        # self.score_pool4 = nn.Conv2d(512, self.num_classes, 1)

        in_channels = 1024
        inter_channel = in_channels // 4
        self.score = nn.Sequential(
            nn.Conv2d(in_channels, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channel, self.num_classes, 1)
        )

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

        if self.count_logits:
            self.logit_stats = {'image': LogitStats(self.num_classes)}
            if self.use_similarity_loss:
                self.logit_stats['object'] = LogitStats(self.num_classes)
            if self.use_entropy_loss:
                self.logit_stats['background'] = LogitStats(self.num_classes)

    def model_forward(self, x):
        # from matplotlib import pyplot as plt
        # plt.imshow(x[0].transpose(0,2))
        # plt.show()
        h = self.conv0(x)
        h = self.conv1(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        h = self.conv2(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        h = self.conv3(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        #h = self.conv4(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        #h = self.conv5(h)

        # plt.imshow(torch.max(h[0].detach(), dim=0)[0])
        # plt.show()

        # h = self.relu6(self.fc6(h))
        # h = self.drop6(h)

        # h = self.relu7(self.fc7(h))
        # h = self.drop7(h)

        # h = self.score_fr(h)
        # h = self.upscore2(h)
        # upscore2 = h  # 1/16

        # h = self.score_pool4(pool4)
        # h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        # score_pool4c = h  # 1/16

        # h = upscore2 + score_pool4c

        # h = self.upscore16(h)
        # h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()
        h = self.relu4(self.fc4(h))
        h = self.drop4(h)

        h = self.score(h)

        h = torch.nn.functional.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return {'out': h}

    def forward(self, image, targets, perfect_mask = None):
        output = {}
        output['image'] = self._forward(image, targets)
        
        i_mask = output['image'][1]
        if perfect_mask != None:
            i_mask = perfect_mask
        
        if self.use_similarity_loss or self.use_mask_area_loss:
            masked_image = i_mask.unsqueeze(1) * image
            output['object'] = self._forward(masked_image, targets, frozen=True)

        if self.use_entropy_loss:   
            target_mask_inversed = torch.ones_like(i_mask) - i_mask
            inverted_masked_image = target_mask_inversed.unsqueeze(1) * image
            output['background'] = self._forward(inverted_masked_image, targets, frozen=True)
            
        return output

    def _forward(self, image, targets, frozen=False):
        if frozen:
            segmentations = self.frozen(image)['out']
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
        
        if self.use_perfect_mask:
            output = self(image, target_vector, torch.max(targets, dim=1)[0])
        else:
            output = self(image, target_vector)
        
        # t = torch.zeros(image.size())
        # output = self(t, target_vector)

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

        #loss = self.classification_loss_fn(output['image'][0], targets)
        loss = self.classification_loss_fn(output['image'][3], target_vector)


        #classification_loss_object = self.classification_loss_fn(o_logits, targets)
        #classification_loss_background = self.classification_loss_fn(b_logits, targets)

        self.log('classification_loss', loss)

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

        
        obj_back_loss = torch.zeros((1), device=loss.device)
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][1], output['object'][1])
            self.log('similarity_loss', similarity_loss)
            obj_back_loss += similarity_loss

        if self.use_entropy_loss:
            background_entropy_loss = entropy_loss(output['background'][3])
            self.log('background_entropy_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted

        if self.use_similarity_loss or self.use_entropy_loss:
            if self.use_weighted_loss:
                loss = weighted_loss(loss, obj_back_loss, 2, 0.2)
            else:
                loss = loss + obj_back_loss
        else:
            loss = loss

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

        # self.i += 1.
        # self.log('iterations', self.i)

        self.log('loss', float(loss))
       
        self.train_metrics(output['image'][0], targets)

        #DEBUG

        #Save min and max logits
        if self.count_logits:
            for k, v in output.items():
                self.logit_stats[k].update(v[3])

        return loss

    def training_epoch_end(self, outs):
        for g in self.trainer.optimizers[0].param_groups:
            self.log('lr', g['lr'], prog_bar=True)
        
        torch.set_printoptions(precision=4)

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

        #DEBUG
        if self.count_logits and self.save_path:
            dir = self.save_path + '/logit_stats'
            os.makedirs(dir)
            class_dict = get_class_dictionary(self.dataset)
            for k,v in self.logit_stats.items():
                v.plot(dir + f'/{k}.png', list(class_dict.keys()))

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, threshold=0.001, min_lr=1e-6)
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


class ConvBlock(nn.Module):

    def __init__(self, convolutions, in_channels, out_channels, kernel=3) -> None:
        super().__init__()
        layers = []
        for i in range(convolutions):
            if i == 0:
                i_c = in_channels
            else:
                i_c = out_channels
            layers += [
                nn.Conv2d(i_c, out_channels, kernel, stride=1, padding=math.floor(kernel)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
