import sys
import os

sys.path.insert(0, os.path.abspath(".."))
import numpy as np
from pathlib import Path

from tqdm import tqdm
from pathlib import Path
from torchray.utils import get_device
from timeit import default_timer

from models.resnet50 import Resnet50
from data.dataloader import *
from utils.helper import *
from utils.image_display import *

from sklearn.metrics import f1_score, precision_score, recall_score
sys.path.insert(0, os.path.abspath("/users/dniederb/nn-explainer/"))

def compute_scores(dataset, checkpoint, checkpoint_base_path, multilabel):
    # Set up data module
    if dataset == "VOC":
        num_classes = 20
        data_path = Path(data_base_path) / "VOC2007"
        data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "VOC2012":
        num_classes = 20
        data_path = Path(data_base_path) / "VOC2012"
        data_module = VOC2012DataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "OI":
        num_classes = 13
        data_path = Path(data_base_path) / "OI"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "OI_LARGE":
        num_classes = 20
        data_path = Path(data_base_path) / "OI_LARGE"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "OI_SMALL":
        num_classes = 3
        data_path = Path(data_base_path) / "OI_SMALL"
        data_module = OIDataModule(data_path=data_path, test_batch_size=1)
    elif dataset == "TOY":
        num_classes = 8
        data_path = Path(data_base_path) / "TOY"
        data_module = ToyData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    elif dataset == "TOY_MULTI":
        num_classes = 8
        data_path = Path(data_base_path) / "TOY_MULTI"
        data_module = ToyData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    else:
        raise Exception("Unknown dataset " + dataset)

    model = Resnet50.load_from_checkpoint(checkpoint_base_path+checkpoint+".ckpt", num_classes=num_classes, multilabel=multilabel, weighted_sampling=False, dataset=dataset, pretrained=False, aux_classifier=aux_classifier)
    device = get_device()
    model.to(device)
    model.eval()

    data_module.setup()

    logits_fn = torch.sigmoid if multilabel else lambda x: torch.nn.functional.softmax(x, dim=1)
    preds = []
    trues = []

    for batch in tqdm(data_module.test_dataloader()):
        image, annotations = batch

        image = image.to(device)
        targets = get_targets_from_annotations(annotations, dataset=dataset)

        start_time = default_timer()
        logits = model(image)
        pred = logits_fn(logits)
        preds.append(pred.detach().cpu().squeeze().numpy() >= 0.5)
        trues.append(targets.int().cpu().squeeze().numpy())
        end_time = default_timer()
        total_time += end_time - start_time

    averages = ['micro', 'weighted']
    classification_metrics = {'f1':{}, 'precision':{}, 'recall':{}}
    for avg in averages:
        classification_metrics['f1'][avg] = f1_score(trues, preds, average=avg)
        classification_metrics['precision'][avg] = precision_score(trues, preds, average=avg)
        classification_metrics['recall'][avg] = recall_score(trues, preds, average=avg)

    return classification_metrics

############################################## Change to your settings ##########################################################
data_base_path = Path("/scratch/snx3000/dniederb/datasets/")

dataset = "OI_LARGE" # ['VOC', 'VOC2012', 'TOY', 'TOY_MULTI', 'OI_SMALL', 'OI', 'OI_LARGE'] 
multilabel = False

checkpoints_base_path = "/scratch/snx3000/dniederb/experiments/thesis/resnet50/OI_LARGE/check/tb_logs/Selfexplainer/version_0/checkpoints/"
checkpoints = ["epoch=1-step=26070"] # Evaluate multiple models at once

load_file = ''
save_file = 'resnet50_oilarge_check.npz'


#################################################################################################################################

p = Path(save_file)
if not p.parent.is_dir():
    p.parent.mkdir(parents=True)

try:
    results = np.load(load_file, allow_pickle=True)["results"].item()
except:
    results = {}


for checkpoint in checkpoints:
    if not(checkpoint in results):
        results[checkpoint] = {}
        # try:
        if dataset == 'VOC':
            data_path = data_base_path / "VOC2007"
        elif dataset == 'VOC2012':
            data_path = data_base_path / "VOC2012"
        elif dataset == 'TOY':
            data_path = data_base_path / "TOY"
        elif dataset == 'TOY_MULTI':
            data_path = data_base_path / "TOY_MULTI"
        elif dataset == 'OI_SMALL':
            data_path = data_base_path / "OI_SMALL"
        elif dataset == 'OI':
            data_path = data_base_path / "OI"
        elif dataset == 'OI_LARGE':
            data_path = data_base_path / "OI_LARGE"

        model_name = checkpoint
        model_path = checkpoints_base_path + checkpoint + '.ckpt'

        if checkpoint.startswith('aux'):
            aux_classifier=True
        else:
            aux_classifier=False
        
        classification_metrics = compute_scores(dataset, checkpoint, checkpoints_base_path, multilabel=multilabel)
     
        d = {}
        d['classification_metrics'] = classification_metrics
        results[checkpoint] = d
        print("Scores computed for: {} - {}".format(dataset, checkpoint))

        np.savez(save_file, results=results)


