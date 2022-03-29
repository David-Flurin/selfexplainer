import pytorch_lightning as pl

from torchvision import models

class Deeplabv3Resnet50Model(pl.LightningModule):
    def __init__(self, pretrained=False, num_classes=20):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)['out']
        return x


