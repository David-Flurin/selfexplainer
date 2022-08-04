import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T
from tqdm.notebook import tqdm
from PIL import Image
from pathlib import Path
from torchray.utils import get_device
from utils.helper import get_class_dictionary, get_filename_from_annotations

from models.classifier import Resnet50ClassifierModel
from data.dataloader import VOCDataModule


dataset = 'VOC'
num_classes = 20
data_base_path = Path("../../datasets/")
data_path = Path('../datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/')

classifier_type = 'vgg16'
classifier_checkpoint = Path('../src/checkpoints/pretrained_classifiers/vgg16_voc.ckpt')

masks_base_path = Path('../src/evaluation/masks/')

methods = ['explainer', 'grad_cam', 'rise', 'rt_saliency']

mask_objects = ['bottle', 'car', 'cat', 'dog', 'person']
target_dict = get_class_dictionary(dataset)
inv_target_dict = {value: key for key, value in target_dict.items()}

class_count_dict = {'aeroplane': 205, 'bicycle': 250, 'bird': 289, 'boat': 176, 'bottle': 240, 'bus': 183, 'car': 775, 
                    'cat': 332, 'chair': 545, 'cow': 127, 'diningtable': 247, 'dog': 433, 'horse': 279, 'motorbike': 233, 
                    'person': 2097, 'pottedplant': 254, 'sheep': 98, 'sofa': 355, 'train': 259, 'tvmonitor': 255}

transformer = T.Compose([ T.Resize(size=(224,224)),
                          T.ToTensor(), 
                          T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])



model = Resnet50ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset)

if dataset == "VOC":
    num_classes = 20
    data_path = Path(data_base_path) / "VOC2007"
    data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
else:
    raise ValueError('Dataset not known')
data_module.setup()


device = get_device()
model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad_(False)


mask_classes = [target_dict[obj] for obj in mask_objects]

def compute_results(model, image, mask, class_id):
    thresholds = np.arange(0.1, 1.0, 0.1)
    outputs = []
    for threshold in thresholds:
        thresh_mask = (mask > threshold).float()
        masked_image = thresh_mask * image
        output_probs = torch.nn.Softmax(dim=1)(model(masked_image))
        outputs.append(output_probs[0][class_id].cpu().numpy())

    return np.mean(outputs)

try:
    results = np.load('class_scores_results.npz', allow_pickle=True)['results'].item()
except:
    results = {}

all_scores = {k:[] for k in [target_dict[obj] for obj in mask_objects]}
for batch in tqdm(data_module.test_dataloader()):
    image, annotations = batch
    filename = get_filename_from_annotations(annotations, dataset=dataset)    #Only need filenames from the directory, and not masks, therefore we can just take any method here
    masks_dir = (masks_base_path / '{}_{}_{}'.format(dataset, classifier_type, "explainer") 
                                 / 'class_masks'
                                 / 'masks_for_class_{}'.format(target_class))

    print("Evaluating classifier on unmasked {} images".format(inv_target_dict[target_class]))

    
    image = image.to(device)

    output_probs = torch.nn.Softmax(dim=1)(model(image))[0]
    for target_class in [target_dict[obj] for obj in mask_objects]:
        target_prob = output_probs[target_class].cpu().numpy()
        all_scores[target_class].append(target_prob)
        

for target_class in [target_dict[obj] for obj in mask_objects]:
    for method in methods:
        if method not in results:
            results[method] = {}
        if inv_target_dict[target_class] not in results[method]:
            results[method][inv_target_dict[target_class]] = {}
            
        results[method][inv_target_dict[target_class]]['no_mask'] = {}
        results[method][inv_target_dict[target_class]]['no_mask']['mean_probs'] = np.mean(all_scores[target_class])


for method in methods:
    if method not in results:
        results[method] = {}
    for target_class in mask_classes:
        if target_class not in results[method]:
            results[method][inv_target_dict[target_class]] = {}
        for mask_class in mask_classes:
            results[method][inv_target_dict[target_class]][inv_target_dict[mask_class]] = {}

            masks_dir = (masks_base_path / '{}_{}_{}'.format(dataset, classifier_type, method)
                                         / 'class_masks'
                                         / 'masks_for_class_{}'.format(mask_class))

            clear_output(wait=True)
            print("Evaluating method {} for {} images with {} masks".format(method, 
                                                                            inv_target_dict[target_class],
                                                                            inv_target_dict[mask_class]))

            all_scores = []
            for filename in tqdm(masks_dir.glob('*.npz'), total=class_count_dict[inv_target_dict[target_class]]):
                jpeg_filename = os.path.splitext(filename.name)[0] + '.jpg'
                image = Image.open(data_path / jpeg_filename).convert("RGB")
                image = transformer(image).unsqueeze(0)
                image = image.to(device)

                mask = np.load(filename)['arr_0']
                mask = torch.tensor(np.reshape(mask, [1,1, *mask.shape]), device=device)
                score = compute_results(model=model, image=image, mask=mask, class_id=target_class)
                all_scores.append(score)

            results[method][inv_target_dict[target_class]][inv_target_dict[mask_class]]['mean_probs'] = np.mean(all_scores)

np.savez('class_scores_results.npz', results=results)