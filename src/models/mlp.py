from models.base import BaseModel
from torch import nn, float64, sigmoid


class MLP(BaseModel):

    def __init__(self, **kwargs):
                 
        del kwargs['rgb']
        super().__init__(**kwargs)

        self.model = _MLP(kwargs['rgb'], kwargs['num_classes'])

        self.bg_loss = 'distance'



class _MLP(nn.Module):

    def __init__(self, rgb, num_classes):
        super().__init__()

        self.first = nn.Conv2d(3 if rgb else 1, 64, 1, dtype=float64)
        inter = []
        for i in range(5):
            inter.append(nn.Conv2d(64, 64, 1, dtype=float64))
            inter.append(nn.ReLU())
        self.intermediate = nn.Sequential(*inter)
        self.out = nn.Conv2d(64, num_classes, 1, dtype=float64)


    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.first(x)
        x = nn.functional.relu(x)
        x = self.intermediate(x)
        x = self.out(x)
        return x