import os
from matplotlib.pyplot import get
from sklearn import multiclass
import torch
import numpy as np

import pytorch_lightning as pl
import torchvision.transforms as T

from torchvision.datasets import VOCDetection, MNIST
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional
from pathlib import Path
from xml.etree.ElementTree import parse as ETparse
from utils.helper import get_class_dictionary
from data.dataset import ColorDataset, SyntheticDataset, SyntheticDataset_Saved, OIDataset

class VOCDataModule(pl.LightningDataModule):

    def __init__(self, data_path, train_batch_size=16, val_batch_size=16, test_batch_size=16, use_data_augmentation=False, weighted_sampling=False):
        super().__init__()

        self.data_path = Path(data_path)

        if os.path.exists(self.data_path) and len(os.listdir(self.data_path)) > 1:
            self.download = False
        else:
            self.download = True
        print('Data augmentation:', use_data_augmentation)
        self.train_transformer = get_training_image_transformer(use_data_augmentation)
        self.test_transformer = get_testing_image_transformer()

        self.weighted_sampling = weighted_sampling

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train = VOCDetection(self.data_path, year="2007", image_set="train", download=self.download, transform=self.train_transformer)
        self.val   = VOCDetection(self.data_path, year="2007", image_set="val", download=self.download, transform=self.test_transformer)
        self.test  = VOCDetection(self.data_path, year="2007", image_set="test", download=self.download, transform=self.test_transformer)

    def train_dataloader(self):
        if self.weighted_sampling:
            weights = self.calculate_weights()
            generator=torch.Generator(device='cpu')
            generator.manual_seed(42)
            return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available(), 
                sampler=WeightedRandomSampler(weights, len(weights), generator=generator))
        else:
            return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

    def calculate_weights(self):
        img_classes = {}
        voc_classes = get_class_dictionary('VOC', include_background_class=False)
        obj_img_count = [0]*len(voc_classes)

        for target in self.train.targets:
            annotation = self.train.parse_voc_xml(ETparse(target).getroot())
            target_indices = [0]*len(voc_classes)
            for object in annotation['annotation']['object']:
                target_indices[voc_classes[object['name']]] = 1
                obj_img_count[voc_classes[object['name']]] += 1
            img_classes[annotation['annotation']['filename']] = np.array(target_indices)
        
        class_weights = np.array([1/c for c in obj_img_count])
        img_weights =  [class_weights[targets == 1].sum() / targets.sum() for targets in img_classes.values()]
        return img_weights

class VOC2012DataModule(VOCDataModule):

    def __init__(self, data_path, train_batch_size=16, val_batch_size=16, test_batch_size=16, use_data_augmentation=False, weighted_sampling=False):
        super().__init__(data_path, train_batch_size=train_batch_size, val_batch_size=val_batch_size, test_batch_size=test_batch_size,
         use_data_augmentation=use_data_augmentation, weighted_sampling=weighted_sampling)   

    def setup(self, stage: Optional[str] = None):
        self.train = VOCDetection(self.data_path, year="2012", image_set="train", download=self.download, transform=self.train_transformer)
        self.val   = VOCDetection(self.data_path, year="2012", image_set="val", download=self.download, transform=self.test_transformer)
        self.test  = VOCDetection(self.data_path, year="2012", image_set="val", download=self.download, transform=self.test_transformer)

    def train_dataloader(self):
        if self.weighted_sampling:
            weights = self.calculate_weights()
            generator=torch.Generator(device='cpu')
            generator.manual_seed(42)
            return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available(),
                sampler=WeightedRandomSampler(weights, len(weights), generator=generator))
        else:
            return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def calculate_weights(self):
        img_classes = {}
        voc_classes = get_class_dictionary('VOC2012', include_background_class=False)
        obj_img_count = [0]*len(voc_classes)

        for target in self.train.targets:
            annotation = self.train.parse_voc_xml(ETparse(target).getroot())
            target_indices = [0]*len(voc_classes)
            counted_objects = []
            for object in annotation['annotation']['object']:
                if object['name'] in counted_objects:
                    continue
                counted_objects.append(object['name'])
                target_indices[voc_classes[object['name']]] = 1
                obj_img_count[voc_classes[object['name']]] += 1
            img_classes[annotation['annotation']['filename']] = np.array(target_indices)

        class_weights = np.array([1/c for c in obj_img_count])
        img_weights =  [class_weights[targets == 1].sum() / targets.sum() for targets in img_classes.values()]
        return img_weights

class OIDataModule(pl.LightningDataModule):

    def __init__(self, data_path, train_batch_size=16, val_batch_size=16, test_batch_size=1, use_data_augmentation=False, weighted_sampling=False):
        super().__init__()

        self.type = data_path.name
        self.data_path = Path(data_path)
        self.weighted_sampling = weighted_sampling

        self.train_transformer = get_training_image_transformer(use_data_augmentation)
        self.test_transformer = get_testing_image_transformer()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train = OIDataset(root=self.data_path / 'train', transform_fn=self.train_transformer)
        self.val = OIDataset(root=self.data_path / 'validation', transform_fn=self.test_transformer)
        self.test = OIDataset(root=self.data_path / 'test', transform_fn=self.test_transformer)

    def train_dataloader(self):
        if self.weighted_sampling:
            weights = self.calculate_weights()
            generator=torch.Generator(device='cpu')
            generator.manual_seed(23098471209)
            return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available(), 
                sampler=WeightedRandomSampler(weights, len(weights), generator=generator))
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())
        #return None
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

    def calculate_weights(self):
        oi_classes = get_class_dictionary(self.type, include_background_class=False)
        img_classes = {}
        obj_img_count = [0] * len(oi_classes)
        for img, classes in self.train.labels.items():
            target_indices = [0]*len(oi_classes)
            for c in classes:
                obj_img_count[oi_classes[c.lower()]] += 1
                target_indices[oi_classes[c.lower()]] = 1
            img_classes[img] = np.array(target_indices)

        class_weights = np.array([1/(c+1) for c in obj_img_count])
        class_weights = np.array([c if c < 1 else 0 for c in class_weights])

        img_weights =  [class_weights[targets == 1].sum() / targets.sum() for targets in img_classes.values()]
        return img_weights

class SyntheticDataModule(pl.LightningDataModule):

    def __init__(self, epoch_length, test_samples, segmentation=False, multilabel=False, train_batch_size=16, val_batch_size=16, test_batch_size=16, use_data_augmentation=False):
        super().__init__()

        self.epoch_length = epoch_length
        self.test_samples = test_samples
        self.segmentation = segmentation
        self.multilabel = multilabel

        self.train_transformer = get_training_image_transformer()
        self.test_transformer = get_training_image_transformer()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train = SyntheticDataset(self.epoch_length, transform_fn=self.train_transformer, segmentation=self.segmentation, multilabel=self.multilabel)
        self.test = SyntheticDataset(self.test_samples, transform_fn=self.test_transformer, segmentation=self.segmentation, multilabel=self.multilabel)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        #return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())
        return None

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

class SyntheticData_Saved_Module(pl.LightningDataModule):

    def __init__(self, data_path, segmentation=False, train_batch_size=16, val_batch_size=16, test_batch_size=1, use_data_augmentation=False):
        super().__init__()

        self.data_path = data_path
        self.segmentation = segmentation

        self.train_transformer = get_training_image_transformer()
        self.test_transformer = get_testing_image_transformer()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train = SyntheticDataset_Saved(self.data_path, 'train', transform_fn=self.train_transformer, segmentation=self.segmentation)
        self.val = SyntheticDataset_Saved(self.data_path, 'val', transform_fn=self.test_transformer, segmentation=self.segmentation)
        self.test = SyntheticDataset_Saved(self.data_path, 'test', transform_fn=self.test_transformer, segmentation=self.segmentation)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

class ColorDataModule(pl.LightningDataModule):

    def __init__(self, epoch_length, test_samples, segmentation=False, multiclass=False, train_batch_size=16, val_batch_size=16, test_batch_size=16, use_data_augmentation=False, rgb=False):
        super().__init__()

        self.epoch_length = epoch_length
        self.test_samples = test_samples
        self.segmentation = segmentation

        self.rgb = rgb

        self.multiclass = multiclass

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train = ColorDataset(self.epoch_length, rgb=self.rgb, transform_fn=None, segmentation=self.segmentation, multiclass=self.multiclass)
        self.test = ColorDataset(self.test_samples, rgb=self.rgb, segmentation=self.segmentation, multiclass=self.multiclass)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        #return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())
        return None

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())




def get_training_image_transformer(use_data_augmentation=False, bw=False):

    mean = [0.1307] if bw else [0.485, 0.456, 0.406]
    std = [0.3081] if bw else [0.229, 0.224, 0.225]
    if use_data_augmentation:
        transformer = T.Compose([ T.RandomHorizontalFlip(),
                                  T.RandomRotation(10),
                                  T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                  #T.Resize(256),
                                  #T.CenterCrop(224),
                                  T.Resize(size=(224,224)),
                                  T.ToTensor(), 
                                  T.Normalize(mean = mean, std = std)])
    else:
        transformer = T.Compose([ T.Resize(size=(224,224)),
                                  # T.Resize(256),
                                  # T.CenterCrop(224),
                                  T.ToTensor(), 
                                  T.Normalize(mean = mean, std = std)
                                  ])

    return transformer

def get_testing_image_transformer(bw = False):

    mean = [0.1307] if bw else [0.485, 0.456, 0.406]
    std = [0.3081] if bw else [0.229, 0.224, 0.225]
    transformer = T.Compose([ T.Resize(size=(224,224)),
                              T.ToTensor(), 
                              T.Normalize(mean = mean, std = std)])

    return transformer

def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    if len(batch[0]) == 2:
        target = [item[1] for item in batch]
        return data, target
    else:
        target = torch.stack([item[1] for item in batch])
        annotations = [item[2] for item in batch]
        return data, target, annotations

