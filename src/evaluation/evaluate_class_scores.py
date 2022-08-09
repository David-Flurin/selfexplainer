import os
import torch
import sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchray.utils import get_device

from models.classifier import Resnet50ClassifierModel
from models.resnet50 import Resnet50
from models.selfexplainer import SelfExplainer
from utils.helper import get_class_dictionary, get_filename_from_annotations, get_targets_from_annotations

from data.dataloader import VOCDataModule


dataset = 'VOC'
num_classes = 20
img_path = Path('../../datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/')
classifier_checkpoint = Path('/home/david/Documents/Master/Thesis/selfexplainer/src/checkpoints/resnet50/voc2007_pretrained.ckpt')
selfexplainer_checkpoint = Path('/home/david/Documents/Master/Thesis/selfexplainer/src/checkpoints/selfexplainer/3passes_01_later.ckpt')

load_file = ''
save_file = 'results/class_masks/VOC2007/selfexplainer_3passes.npz'
data_base_path = Path("../../datasets/")
masks_base_path = Path('data')

methods = ['3passes_01']

mask_classes = ['aeroplane', 'bird', 'bottle', 'car', 'cat', 'cow', 'diningtable', 'dog', 'horse', 'person']
target_dict = get_class_dictionary(dataset)
inv_target_dict = {value: key for key, value in target_dict.items()}

class_count_dict = {'aeroplane': 205, 'bicycle': 250, 'bird': 289, 'boat': 176, 'bottle': 240, 'bus': 183, 'car': 775, 
                    'cat': 332, 'chair': 545, 'cow': 127, 'diningtable': 247, 'dog': 433, 'horse': 279, 'motorbike': 233, 
                    'person': 2097, 'pottedplant': 254, 'sheep': 98, 'sofa': 355, 'train': 259, 'tvmonitor': 255}

transformer = T.Compose([ T.Resize(size=(224,224)),
                          T.ToTensor(), 
                          T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])



classifier = Resnet50.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset)

if dataset == "VOC":
    num_classes = 20
    data_path = Path(data_base_path) / "VOC2007"
    data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
else:
    raise ValueError('Dataset not known')
data_module.setup()


device = get_device()
classifier.to(device)
classifier.eval()
for param in classifier.parameters():
    param.requires_grad_(False)

selfexplainer = None
if set(methods) - set(['gradcam', 'rise', 'explainer']):
    selfexplainer = SelfExplainer.load_from_checkpoint(selfexplainer_checkpoint, 
                        num_classes=num_classes, 
                        multilabel=True if dataset in ['VOC', 'TOY_MULTI'] else False, 
                        dataset=dataset, 
                        pretrained=False, 
                        aux_classifier=False)
    selfexplainer.to(device)
    selfexplainer.eval()
    for param in selfexplainer.parameters():
        param.requires_grad_(False)





#mask_classes = [target_dict[obj] for obj in mask_classes]

def compute_results(model, image, mask, class_id):
    thresholds = np.arange(0.1, 1.0, 0.1)
    outputs = []
    for threshold in thresholds:
        thresh_mask = (mask > threshold).float()
        masked_image = thresh_mask * image
        output_probs = torch.nn.Softmax(dim=1)(model(masked_image))
        outputs.append(output_probs[0][class_id].cpu().numpy())

    return np.mean(outputs)

all_scores = {method:{tc:{mc:[] for mc in ['None'] + mask_classes} for tc in mask_classes} for method in methods}
i = 0
for batch in tqdm(data_module.test_dataloader()):
    image, annotations = batch
    filename = get_filename_from_annotations(annotations, dataset=dataset)    #Only need filenames from the directory, and not masks, therefore we can just take any method here
    targets = get_targets_from_annotations(annotations, dataset=dataset)
    target_classes = [index for index, value in enumerate(targets[0]) if value == 1.0]

    image = image.to(device)

    output_probs = torch.nn.Softmax(dim=1)(classifier(image))[0]
    intersection = set(target_classes) & set([target_dict[obj] for obj in mask_classes])
    if intersection:
        for target_class in intersection:
            target_prob = output_probs[target_class].cpu().numpy()
            for method in methods:
                if method not in ['gradcam', 'rise', 'explainer']:
                    continue
                all_scores[method][inv_target_dict[target_class]]['None'].append(target_prob)


    if set(methods) - set(['gradcam', 'rise', 'explainer']):
        output_probs = torch.nn.Softmax(dim=1)(selfexplainer(image)[3])[0]
        intersection = set(target_classes) & set([target_dict[obj] for obj in mask_classes])
        if intersection:
            for target_class in intersection:
                target_prob = output_probs[target_class].cpu().numpy()
                all_scores[method][inv_target_dict[target_class]]['None'].append(target_prob)

    intersection = set(target_classes) & set([target_dict[obj] for obj in mask_classes])
    if intersection:
        for method in methods:
            for target_class in intersection:
                target_class_name = inv_target_dict[target_class]
                for mask_class in mask_classes:
                        masks_dir = (masks_base_path / '{}_{}_{}'.format(dataset, 'resnet50' if method in ['gradcam', 'rise', 'explainer'] else 'selfexplainer', method)
                                         / 'class_masks'
                                         / 'masks_for_class_{}'.format(target_dict[mask_class]))


                        mask = np.load(masks_dir / (filename[:-3]+'npz'))['arr_0']
                        mask = torch.tensor(np.reshape(mask, [1,1, *mask.shape]), device=device)
                        if method in ['gradcam', 'rise', 'explainer']:
                            score = compute_results(model=classifier, image=image, mask=mask, class_id=target_class)
                        else:
                                thresholds = np.arange(0.1, 1.0, 0.1)
                                outputs = []
                                for threshold in thresholds:
                                    thresh_mask = (mask > threshold).float()
                                    masked_image = thresh_mask * image
                                    output_probs = torch.nn.Softmax(dim=1)(model(masked_image)[3])
                                    outputs.append(output_probs[0][target_class].cpu().numpy())

                        all_scores[method][target_class_name][mask_class].append(score)



try:
    results = np.load(load_file, allow_pickle=True)['results'].item()
except:
    results = {}


for method in all_scores:
    results[method] = {}
    for target_class_name in all_scores[method]:
        results[method][target_class_name] = {}
        for mask_class in all_scores[method][target_class_name]:
            results[method][target_class_name][mask_class] = np.mean(all_scores[method][target_class_name][mask_class])
   
print(results)


            

# for method in methods:
#     if method not in results:
#         results[method] = {}
#     for target_class in mask_classes:
#         if target_class not in results[method]:
#             results[method][inv_target_dict[target_class]] = {}
#         for mask_class in mask_classes:
#             results[method][inv_target_dict[target_class]][inv_target_dict[mask_class]] = {}

#             masks_dir = (masks_base_path / '{}_{}_{}'.format(dataset, 'resnet50' if method != 'selfexplainer' else 'selfexplainer', method)
#                                          / 'class_masks'
#                                          / 'masks_for_class_{}'.format(mask_class))

#             #clear_output(wait=True)
#             print("Evaluating method {} for {} images with {} masks".format(method, 
#                                                                             inv_target_dict[target_class],
#                                                                             inv_target_dict[mask_class]))

#             all_scores = []
#             for filename in tqdm(masks_dir.glob('*.npz'), total=class_count_dict[inv_target_dict[target_class]]):
#                 jpeg_filename = os.path.splitext(filename.name)[0] + '.jpg'
#                 image = Image.open(img_path / jpeg_filename).convert("RGB")
#                 image = transformer(image).unsqueeze(0)
#                 image = image.to(device)

#                 mask = np.load(filename)['arr_0']
#                 mask = torch.tensor(np.reshape(mask, [1,1, *mask.shape]), device=device)
#                 score = compute_results(model=classifier if method != 'selfexplainer' else selfexplainer, image=image, mask=mask, class_id=target_class)
#                 all_scores.append(score)

#             results[method][inv_target_dict[target_class]][inv_target_dict[mask_class]]['mean_probs'] = np.mean(all_scores)

np.savez(save_file, results=results)
