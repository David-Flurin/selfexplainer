import torch

from models.DeepLabv3 import Deeplabv3Resnet50Model
#from torchviz import make_dot

from utils.helper import get_class_dictionary, get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution, get_targets_from_segmentations, LogitStats
from utils.image_display import save_all_class_masked_images, save_mask, save_masked_image, save_background_logits, save_image, save_all_class_masks, get_unnormalized_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss, bg_loss, background_activation_loss, relu_classification
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics, ClassificationMultiLabelMetrics
from utils.weighting import softmax_weighting
from evaluation.compute_scores import selfexplainer_compute_numbers
from data.dataloader import VOCDataModule

class Simple_Model(torch.nn.Module):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1., pretrained=False, use_similarity_loss=False, similarity_regularizer=1.0, use_background_loss=False, bg_loss_regularizer=1.0, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.04, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, use_perfect_mask=False, count_logits=False, save_masks=False, save_all_class_masks=False, 
                 gpu=0, profiler=None, metrics_threshold=0.5, save_path="./results/", objective='classification', class_loss='bce', frozen=False, freeze_every=20, background_activation_loss=False, bg_activation_regularizer=0.5, target_threshold=0.7, non_target_threshold=0.3, background_loss='logits_ce', aux_classifier=False, multiclass=False, class_only=False):

        super().__init__()

        self.i = 0
        self.gpu = gpu
        self.aux_classifier = aux_classifier
        self.model = Deeplabv3Resnet50Model(pretrained=False, num_classes=num_classes, aux_classifier=aux_classifier)
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.num_classes = num_classes

    def forward(self, image, targets):
        segmentations, logits = self.model(image)
        return logits



model = Simple_Model(num_classes = 3, dataset='SMALLVOC', learning_rate = 1e-4, gpu=0, aux_classifier=True)

train_module = VOCDataModule('../datasets/VOC2007_small')
train_module.setup()
train_loader = train_module.train_dataloader()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)

for epoch in range(1):
    for i, batch in enumerate(train_loader):
        image, annotations = batch
        target_vector = get_targets_from_annotations(annotations, dataset=model.dataset, num_classes=model.num_classes, gpu=model.gpu)

        output = model(image, target_vector)
        loss = criterion(output, target_vector)

        loss.backward()
        optimizer.step()
