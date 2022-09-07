import pytorch_lightning as pl
from torch import nn, Tensor
from torchvision import models
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabV3

from collections import OrderedDict
from typing import Dict
from torch.nn import functional as F
from copy import deepcopy

class Deeplabv3Resnet50Model(pl.LightningModule):
    def __init__(self, pretrained=False, num_classes=20, aux_classifier=False):
        super().__init__()
        if pretrained:
            self.model = models.segmentation.deeplabv3(pretrained=pretrained, num_classes=21, aux_loss = aux_classifier)
            self.model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes, aux_loss= aux_classifier)

        self.aux_classifier = aux_classifier
        if self.aux_classifier:
            aux_head = AuxHead(2048, num_classes, deepcopy(self.model.backbone.layer4))
            self.model = ModifiedDeepLab(self.model.backbone, self.model.classifier, aux_head)

    def forward(self, x):
        x = self.model(x)
        if self.aux_classifier:
            return (x['out'], x['aux'])
        else:
            return x['out']

class AuxHead(nn.Module):
    def __init__(self, in_channel, num_classes, layer4):
        super().__init__()
        self.last_resnet_layer = layer4
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channel, num_classes)

    def forward(self, x):
        x = self.last_resnet_layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ModifiedDeepLab(DeepLabV3):

     def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            result["aux"] = x

        return result


