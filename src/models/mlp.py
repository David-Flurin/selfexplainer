from models.base import BaseModel
from torch import nn, float64, sigmoid


class MLP(BaseModel):

    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1, pretrained=False, use_similarity_loss=False, similarity_regularizer=1.0, use_entropy_loss=False, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.04, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, use_perfect_mask=False, count_logits=False, save_masks=False, save_all_class_masks=False, 
                 gpu=0, profiler=None, metrics_threshold=-1.0, save_path="./results/", objective='classification', rgb=False):
                 
        super().__init__(num_classes=num_classes, dataset=dataset, learning_rate=learning_rate, weighting_koeff=weighting_koeff, pretrained=pretrained, use_similarity_loss=False, similarity_regularizer=similarity_regularizer, use_entropy_loss=use_entropy_loss, use_weighted_loss=use_weighted_loss,
        use_mask_area_loss=use_mask_area_loss, use_mask_variation_loss=use_mask_variation_loss, mask_variation_regularizer=mask_variation_regularizer, ncmask_total_area_regularizer=ncmask_total_area_regularizer, mask_area_constraint_regularizer=mask_area_constraint_regularizer, class_mask_min_area=class_mask_min_area, 
                 class_mask_max_area=class_mask_max_area, mask_total_area_regularizer=mask_total_area_regularizer, save_masked_images=save_masked_images, use_perfect_mask=use_perfect_mask, count_logits=count_logits, save_masks=save_masks, save_all_class_masks=save_all_class_masks, 
                 gpu=gpu, profiler=profiler, metrics_threshold=metrics_threshold, save_path=save_path, objective=objective)

        self.model = _MLP(rgb, num_classes)

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
        return sigmoid(x)