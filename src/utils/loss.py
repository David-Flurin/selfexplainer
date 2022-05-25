import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import math

from torch import nn

from matplotlib import pyplot as plt

class TotalVariationConv(pl.LightningModule):
    def __init__(self):
        super().__init__()

        weights_right_variance = torch.tensor([[0.0, 0.0, 0.0],
                                              [0.0, 1.0, -1.0],
                                              [0.0, 0.0, 0.0]], device=self.device).view(1, 1, 3, 3)

        weights_down_variance = torch.tensor([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, -1.0, 0.0]], device=self.device).view(1, 1, 3, 3)

        self.variance_right_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=3, padding=1, padding_mode='reflect', groups=1, bias=False)
        self.variance_right_filter.weight.data = weights_right_variance
        self.variance_right_filter.weight.requires_grad = False

        self.variance_down_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=3, padding=1, padding_mode='reflect', groups=1, bias=False)
        self.variance_down_filter.weight.data = weights_down_variance
        self.variance_down_filter.weight.requires_grad = False

    def forward(self, mask):
        variance_right = self.variance_right_filter(mask.unsqueeze(1)).abs()

        variance_down = self.variance_down_filter(mask.unsqueeze(1)).abs()

        total_variance = (variance_right + variance_down).mean()
        return total_variance

class MaskAreaLoss():
    def __init__(self, image_size=224, min_area=0.0, max_area=1.0, gpu=0):
        self.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else "cpu")

        self.image_size = image_size
        self.min_area = min_area
        self.max_area = max_area

        assert(self.min_area >= 0.0 and self.min_area <= 1.0)
        assert(self.max_area >= 0.0 and self.max_area <= 1.0)
        assert(self.min_area <= self.max_area)
        
    def __call__(self, masks):
        batch_size = masks.size()[0]
        losses = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            mask = masks[i].flatten()
            sorted_mask, indices = mask.sort(descending=True)
            losses[i] += (self._min_mask_area_loss(sorted_mask) + self._max_mask_area_loss(sorted_mask)).mean()

        return losses.mean()

    def _min_mask_area_loss(self, sorted_mask):
        if (self.min_area == 0.0):
            return torch.tensor(0.0)

        ones_length = (int)(self.image_size * self.image_size * self.min_area)
        ones = torch.ones(ones_length, device=self.device)
        zeros = torch.zeros((self.image_size * self.image_size) - ones_length, device=self.device)
        ones_and_zeros = torch.cat((ones, zeros), dim=0)

        # [1, 1, 0, 0, 0] - [0.9, 0.9, 0.9, 0.5, 0.1] = [0.1, 0.1, -0.9, -0.5, -0.1] -> [0.1, 0.1, 0, 0, 0]
        loss = F.relu(ones_and_zeros - sorted_mask)

        return loss
    
    def _max_mask_area_loss(self, sorted_mask):
        if (self.max_area == 1.0):
            return torch.tensor(0.0)

        ones_length = (int)(self.image_size * self.image_size * self.max_area)
        ones = torch.ones(ones_length, device=self.device)
        zeros = torch.zeros((self.image_size * self.image_size) - ones_length, device=self.device)
        ones_and_zeros = torch.cat((ones, zeros), dim=0)

        # [0.9, 0.9, 0.9, 0.5, 0.1] - [1, 1, 1, 1, 0] = [-0.1, -0.1, -0.1, -0.5, 0.1] -> [0, 0, 0, 0, 0.1]
        loss = F.relu(sorted_mask - ones_and_zeros)

        return loss

class ClassMaskAreaLoss(MaskAreaLoss):
    def __call__(self, segmentations, target_vectors):
        masks = segmentations.sigmoid()
        batch_size, num_classes, h, w = masks.size()

        losses = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            class_indices = target_vectors[i].eq(1.0)
            class_masks = masks[i][class_indices]
            for j in range(class_masks.size()[0]):
                mask = class_masks[j].flatten()
                sorted_mask, indices = mask.sort(descending=True)
                losses[i] += (self._min_mask_area_loss(sorted_mask) + self._max_mask_area_loss(sorted_mask)).mean()

            losses[i] = losses[i].mean()

        return losses.mean()

def entropy_loss(logits):
    min_prob = 1e-16
    probs = F.softmax(logits, dim=-1).clamp(min=min_prob)
    log_probs = probs.log()
    entropy = (-probs * log_probs)
    entropy_loss = -entropy.mean()

    #b, c = logits.size()
    #sm = nn.functional.softmax(logits, dim=-1)
    #entropy_loss = abs(-(sm + (c-1)/c).log()).sum(1).mean()


    return entropy_loss


def mask_similarity_loss(logits, targets, imask, omask):
    '''Compute L1 loss over pixel distances between initial mask and object mask.'''

    #older version
    #batch_losses = ((imask - omask).abs() * imask).sum((1,2)) / imask.sum((1,2))
    
    #new version
    # abs_diff = (imask - omask).abs()
    # count = torch.where(abs_diff > 0.1, 1, 0).sum(1).sum(1) #.unsqueeze(1).unsqueeze(2)
    # abs_diff = torch.where(abs_diff >= 0.1, 0, abs_diff)
    # abs_diff_sum = abs_diff.sum(1).sum(1)
    # batch_losses = torch.div(abs_diff_sum, count + 1)

    #newer version
    # max_mask = torch.where(imask > omask, imask, omask)
    # batch_losses = ((imask - omask).abs() * max_mask).sum((1,2)) / max_mask.sum((1,2))

    return nn.functional.cross_entropy(logits, targets)

    return batch_losses.mean()

def weighted_loss(l_1, l_2, steepness, offset):
    loss1 = l_1.item()
    return l_1 + min(1., math.exp(-steepness * (loss1 - offset))) * l_2


# t = torch.zeros((2, 224, 224))
# t[0, 0:50, 0:50] += torch.ones((50, 50))
# t[1, 0:50, 0:50] += torch.ones((50, 50)) * 0.5
# z = torch.zeros((2, 224, 224))
# m = mask_similarity_loss(t, z)
# print(m)

def bg_loss(segmentations, target_vector, loss):
#def bg_loss(segmentations):
    b, c, h, w = segmentations.size()

    if loss == 'logits_ce': 
        mean = segmentations.mean((2,3))
        sm = nn.functional.softmax(mean, dim=-1)
        batch_losses = abs(-(sm + (c-1)/c).log()).sum(1)
    
    elif loss == 'segmentations_ce':
        batch_softmax = torch.zeros_like(segmentations)
        for i in range(b):
            exp_sum = segmentations[i].exp().sum()
            batch_softmax[i] = segmentations[i].exp() / exp_sum
        batch_mean = batch_softmax.sum(dim=(2,3))
        #batch_mean = batch_mean / batch_mean.sum()
        
        #batch_losses = (batch_mean[target_idx] -1/c).abs()
        batch_losses = (batch_mean - 1/c).square().sum(1).sqrt()
        #batch_loss = torch.nn.functional.cosine_similarity(batch_mean)
    
    elif loss == 'segmentations_distance':
        
        batch_losses = torch.zeros(b, device=segmentations.device)
        for i in range(b):
            exp_sum = segmentations[i].exp().sum()
            batch_softmax = segmentations[i].exp() / exp_sum
            target_idx = target_vector[i].eq(1.0)
            target_mean = batch_softmax[target_idx].mean(0)
            non_target_mean = batch_softmax[~target_idx].mean(0).detach()
            batch_losses[i] = (target_mean - non_target_mean).abs().mean((0,1))

    else:
        raise ValueError('Unknown background loss. Choose from [logits_ce, segmentations_ce, segmentations_distance')
    return batch_losses.mean()


def relu_classification(logits, targets, t_threshold, nt_threshold):
    batch_size = logits.size()[0]
    probs = F.softmax(logits, dim=-1)
    losses = torch.zeros(batch_size, device=logits.device)

    for b in range(batch_size):
        target_idx = targets[b].eq(1.0)
        non_target_idx = targets[b].eq(0.0)
        target_probs = probs[b][target_idx]
        non_target_probs = probs[b][non_target_idx]
        for p in target_probs:
            losses[b] += F.relu(t_threshold - p)
        for p in non_target_probs:
            losses[b] += F.relu(p - nt_threshold)
    
    return losses.mean()



def background_activation_loss(mask):
    t = torch.cat((mask[mask < 0.7], torch.tensor([1e-16], device=mask.device)))
    return t.mean()


t = torch.tensor([[.16, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]])
tt = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
print(entropy_loss(t))










