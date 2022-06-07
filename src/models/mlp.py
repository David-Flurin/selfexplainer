from models.base import BaseModel
from torch import nn, float64, sigmoid, flatten
import copy


class MLP(BaseModel):

    def __init__(self, **kwargs):
                 
        parent_args = copy.deepcopy(kwargs)
        del parent_args['rgb']
        super().__init__(**parent_args)

        self.model = _MLP(kwargs['rgb'], kwargs['num_classes'], kwargs['aux_classifier']).float()

        if kwargs['aux_classifier']:
            self.bg_loss = 'entropy'
        else:
            self.bg_loss = 'distance'


class _MLP(nn.Module):

    def __init__(self, rgb, num_classes, aux_classifier):
        super().__init__()

        self.first = nn.Conv2d(3 if rgb else 1, 64, 1, dtype=float64)
        inter = []
        for i in range(10):
            inter.append(nn.Conv2d(64, 64, 1, dtype=float64))
            inter.append(nn.ReLU())
        self.intermediate = nn.Sequential(*inter)
        self.out = nn.Conv2d(64, num_classes, 1, dtype=float64)

        self.aux_classifier = aux_classifier
        if aux_classifier:
            self.aux = FCHead(64, num_classes)



    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.first(x)
        x = nn.functional.relu(x)
        x = self.intermediate(x)
        o = self.out(x)
        if not self.aux_classifier:
            return o
        else:
            l = self.aux(x.float())
            return o, l


class FCHead(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_channel, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        return self.fc2(x)