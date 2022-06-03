from models.no_lightning.base_nl import BaseModel
from models.stevelike_basemodel import Slike_BaseModel
from models.DeepLabv3 import Deeplabv3Resnet50Model
from torch import nn


class SelfExplainer(BaseModel):
    def __init__(self, **kwargs):
                 
        super().__init__(**kwargs)

        self.model = Deeplabv3Resnet50Model(num_classes=kwargs['num_classes'], pretrained=kwargs['pretrained'], aux_classifier=kwargs['aux_classifier'])
        if kwargs['aux_classifier']:
            self.bg_loss = 'entropy'
        else:
            self.bg_loss = 'distance'

        if kwargs['dataset'] == 'MNIST':
            self.model.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
class SelfExplainer_Slike(Slike_BaseModel):
    def __init__(self, **kwargs):
                 
        super().__init__(**kwargs)

        self.model = Deeplabv3Resnet50Model(num_classes=kwargs['num_classes'], pretrained=kwargs['pretrained'], aux_classifier=kwargs['aux_classifier'])
        if kwargs['aux_classifier']:
            self.bg_loss = 'entropy'
        else:
            self.bg_loss = 'distance'

        if kwargs['dataset'] == 'MNIST':
            self.model.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
