import pytorch_lightning as pl

from torchvision import models
import torch

class Deeplabv3Resnet50Model(pl.LightningModule):
    def __init__(self, pretrained=False, num_classes=20):
        super().__init__()
        if pretrained:
            self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=21)
            self.model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)


    def forward(self, x):
        x = self.model(x)['out']
        return x


