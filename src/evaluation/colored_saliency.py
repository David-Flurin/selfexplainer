import pytorch_lightning as pl
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from pathlib import Path
from timeit import default_timer

from data.dataloader import *
from utils.helper import *
from utils.image_display import *
from models.resnet50 import Resnet50
from models.classifier import Resnet50ClassifierModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import *

from PIL import Image
from tqdm import tqdm

############################## Change to your settings ##############################
mask_base_path = '/scratch/snx3000/dniederb/evaluation_data/VOC/sanity_check/'

dataset = 'VOC' # one of: ['VOC', 'SYN']
data_base_path = Path("/scratch/snx3000/dniederb/datasets/")
classifier_type = 'selfexplainer' # one of: ['vgg16', 'resnet50']

method = '3passes_01_2503'

file_list = ['000010', '000043', '000067', '000075']


#####################################################################################
    
# Set up data module
if dataset == "VOC":
    num_classes = 20
    data_path = Path(data_base_path) / "VOC2007"
    data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
elif dataset == 'SYN':
    num_classes = 8
    data_path = Path(data_base_path) / "SYN"
    data_module = SyntheticData_Saved_Module(data_path=data_path)
elif dataset == 'SYN_MULTI':
    num_classes = 8
    data_path = Path(data_base_path) / "SYN_MULTI"
    data_module = SyntheticData_Saved_Module(data_path=data_path)
elif dataset == 'OI_SMALL':
    num_classes = 3
    data_path = Path(data_base_path) / "OI_SMALL"
    data_module = OIDataModule(data_path=data_path)
elif dataset == 'OI':
    num_classes = 13
    data_path = Path(data_base_path) / "OI"
    data_module = OIDataModule(data_path=data_path)
elif dataset == 'OI_LARGE':
    num_classes = 20
    data_path = Path(data_base_path) / "OI_LARGE"
    data_module = OIDataModule(data_path=data_path)
else:
    raise Exception("Unknown dataset " + dataset)

save_path = Path('{}_{}_{}/'.format(dataset, classifier_type, method))
save_path = mask_base_path / save_path 
if not os.path.isdir(save_path):
    os.makedirs(save_path)
os.makedirs(save_path / 'images_overlaid', exist_ok=True)

data_module.setup()

total_time = 0.0

for batch in tqdm(data_module.test_dataloader()):
        image, annotations = batch
        filename = get_filename_from_annotations(annotations, dataset=dataset)

        if filename[-4] == '.':
            filename = filename[:-4]

        if not os.path.exists(save_path / (filename + '.npz')):
            continue
        if len(file_list) > 0 and filename not in file_list:
            continue

        #mask = Image.open(save_path / (filename + '.png'))
        saliencies = np.load(save_path / (filename + '.npz'))['arr_0']        
        assert(saliencies.ndim == 2)

        masked_img = show_cam_on_image(get_unnormalized_image(image[0]).permute(1,2,0).numpy(), saliencies, use_rgb=True)
        Image.fromarray(masked_img).save(save_path / ('images_overlaid/' + filename + '.png'))
        

