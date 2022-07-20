import sys
import os

sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from torchray.utils import get_device
from timeit import default_timer

from models.resnet50 import Resnet50
from data.dataloader import *
from utils.helper import *
from utils.image_display import *

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



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
    elif dataset == "TOY":
        num_classes = 8
        data_path = Path(data_base_path) / "TOY"
        data_module = ToyData_Saved_Module(data_path=data_path, segmentation=False, test_batch_size=1)
    else:
        raise Exception("Unknown dataset " + dataset)

    model = Resnet50.load_from_checkpoint(checkpoint_base_path+checkpoint+".ckpt", num_classes=num_classes, multiclass=multilabel, dataset=dataset, pretrained=False, aux_classifier=aux_classifier)
    device = get_device()
    model.to(device)
    model.eval()

    data_module.setup()

    total_time = 0.0
    print(checkpoint_base_path+checkpoint+".ckpt")

    logits_fn = torch.sigmoid if multilabel else lambda x: torch.nn.functional.softmax(x, dim=1)
    thresh_preds = []
    preds = []
    trues = []
    #trues_roc = []

    #i = 0
    for batch in tqdm(data_module.test_dataloader()):
        image, annotations = batch

        image = image.to(device)
        targets = get_targets_from_annotations(annotations, dataset=dataset)

        start_time = default_timer()
        from matplotlib import pyplot as plt
        plt.imshow(image[0].permute(1,2,0))
        plt.show()
        torch.save(image,'eval_tensor.pt')
        logits = model(image)
        pred = logits_fn(logits)
        preds.append(pred.detach().cpu().squeeze().numpy() >= 0.5)
        #thresh_preds.append(pred.detach().cpu().squeeze().numpy() > 0.5)
        trues.append(targets.int().cpu().squeeze().numpy())
        end_time = default_timer()
        total_time += end_time - start_time

    averages = ['micro', 'weighted']
    classification_metrics = {'f1':{}, 'precision':{}, 'recall':{}}
    #trues = np.stack(trues, axis=0)
    #trues_roc = np.stack(trues_roc, axis=0)
    for avg in averages:
        classification_metrics['f1'][avg] = f1_score(trues, preds, average=avg)
        classification_metrics['precision'][avg] = precision_score(trues, preds, average=avg)
        classification_metrics['recall'][avg] = recall_score(trues, preds, average=avg)

    return classification_metrics

############################################## Change to your settings ##########################################################
data_base_path = Path("../../datasets/")

dataset = "TOY"
multilabel = False
checkpoints_base_path = "../checkpoints/resnet50/"
checkpoints = ["toy_singlelabel"]

load_file = ''
save_file = 'results/resnet_toy_singlelabel.npz'


#################################################################################################################################

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
        if dataset == 'VOC2012':
            data_path = data_base_path / "VOC2012"
        elif dataset == 'TOY':
            data_path = data_base_path / "TOY"
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


