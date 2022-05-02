from models.base import BaseModel
from models.DeepLabv3 import Deeplabv3Resnet50Model


class SelfExplainer(BaseModel):
    def __init__(self, **kwargs):
                 
        super().__init__(**kwargs)

        self.model = Deeplabv3Resnet50Model(num_classes=kwargs['num_classes'], pretrained=kwargs['pretrained'])
        self.bg_loss = 'entropy'
        