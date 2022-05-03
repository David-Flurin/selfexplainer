import torchvision
from torch import nn
import torch
import math

from models.base import BaseModel

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

class FCN16(BaseModel):

    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1, pretrained=False, use_similarity_loss=False, similarity_regularizer=1.0, use_entropy_loss=False, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.04, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, use_perfect_mask=False, count_logits=False, save_masks=False, save_all_class_masks=False, 
                 gpu=0, profiler=None, metrics_threshold=-1.0, save_path="./results/"):
                 
        super().__init__(num_classes=num_classes, dataset=dataset, learning_rate=learning_rate, weighting_koeff=weighting_koeff, pretrained=pretrained, use_similarity_loss=False, similarity_regularizer=similarity_regularizer, use_entropy_loss=use_entropy_loss, use_weighted_loss=use_weighted_loss,
        use_mask_area_loss=use_mask_area_loss, use_mask_variation_loss=use_mask_variation_loss, mask_variation_regularizer=mask_variation_regularizer, ncmask_total_area_regularizer=ncmask_total_area_regularizer, mask_area_constraint_regularizer=mask_area_constraint_regularizer, class_mask_min_area=class_mask_min_area, 
                 class_mask_max_area=class_mask_max_area, mask_total_area_regularizer=mask_total_area_regularizer, save_masked_images=save_masked_images, use_perfect_mask=use_perfect_mask, count_logits=count_logits, save_masks=save_masks, save_all_class_masks=save_all_class_masks, 
                 gpu=gpu, profiler=profiler, metrics_threshold=metrics_threshold, save_path=save_path)

       

        self.model = fcn(num_classes=num_classes)

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

        h = nn.functional.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return {'out': h}



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
    loss = nn.functional.nll_loss(log_p, target, weight=weight, reduction='sum')
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


class fcn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=num_classes, progress=True)

    def forward(self, x):
        return self.model(x)['out']
