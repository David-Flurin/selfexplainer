from multiprocessing.sharedctypes import Value
import os
import pickle
import pathlib
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from random import randint
import random
from xml.etree import cElementTree as ElementTree
import json
import numpy as np

from toy_dataset import generator
from color_dataset import generator as color_generator

class OISmallDataset(Dataset):
    def __init__(
        self,
        root,
        transform_fn=None,
        gt_field="ground_truth",
        classes=None,
    ):
        with open(root/'labels.json', 'r') as jsonfile:
            labels = json.load(jsonfile)
        self.root = root
        classes = labels['classes']
        
        labels = labels['labels']
        target_classes = {classes.index('Cat'):'cat', classes.index('Dog'):'dog', classes.index('Bird'):'bird'}
        self.labels = {}
        for img, l in labels.items():
            if l == None:
                continue
            for i, c in target_classes.items():
                if i in l:
                    self.labels[img] = [c] if img not in self.labels else self.labels[img] + [c]
        
        self.images = list(self.labels.keys())
        print('Cats:', sum('cat' in value for value in self.labels.values()))
        print('Dogs:', sum('dog' in value for value in self.labels.values()))
        print('Sheeps:', sum('bird' in value for value in self.labels.values()))

        self.transforms = transform_fn
        

    def __getitem__(self, idx):
        img_path = self.root / 'data' / (self.images[idx] +'.jpg')

        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, {'annotation':{'object': [{'name': name} for name in self.labels[self.images[idx]]], 'filename': self.images[idx]}}


    def __len__(self):
        return len(self.images)

class OIDataset(Dataset):
    def __init__(
        self,
        root,
        transform_fn=None,
    ):

        with open(root/'labels.json', 'r') as jsonfile:
            self.labels = json.load(jsonfile)
        self.root = root
        
        self.images = list(self.labels.keys())

        if root.parts[-2] == 'OI':
            self.t_class_list = ['Person', 'Cat', 'Dog', 'Bird', 'Horse', 'Sheep', 'Aeroplane', 'Bus', 'Car', 'Motorbike', 'Train', 'Bottle', 'Sofa']
        elif root.parts[-2] == 'OI_LARGE':
            self.t_class_list = ['Flower', 'Fish', 'Monkey', 'Cake', 'Sculpture', 'Lizard', 'Mobile phone', 'Camera', 'Bread', 'Guitar', 'Snake', 'Handbag', 'Pastry', 'Ball', 'Flag', 'Piano', 'Rabbit', 'Book', 'Mushroom', 'Dress']
        elif root.parts[-2] == 'OI_SMALL':
            self.t_class_list = ['Cat', 'Dog', 'Bird']
        else:
            raise ValueError(f'OI dataset type not known.')

        for c in self.t_class_list:
            s = sum(c in value for value in self.labels.values())
            print(f'{c}:', s)

        self.transforms = transform_fn
        

    def __getitem__(self, idx):
        img_path = self.root / 'data' / (self.images[idx] +'.jpg')
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, {'annotation':{'object': [{'name': name.lower()} for name in set.intersection(set(self.t_class_list), set(self.labels[self.images[idx]]))], 'filename': self.images[idx]}}


    def __len__(self):
        return len(self.images)

class ToyDataset(Dataset):
    def __init__(self, epoch_length, transform_fn=None, segmentation=True, multilabel=False, target='texture'):
        self.ids = range(0, epoch_length)
        self.transform = transform_fn
        self.segmentation = segmentation
        self.target = target
        base = '/'.join(generator.__file__.split('/')[0:-1])
        self.generator = generator.Generator(pathlib.Path(base, 'foreground.txt'), pathlib.Path(base, 'background.txt'))
        self.shapes = {}
        self.foreground_textures = {}
        self.background_textures = {}
        for t in self.generator.f_texture_names:
            self.foreground_textures[t] = 0
        for t in self.generator.b_texture_names:
            self.background_textures[t] = 0

        self.num_objects = 2 if multilabel else 1
        

    def __getitem__(self, index):
        sample = self.generator.generate_sample(randint(1, self.num_objects))       

        if self.transform is not None:
            img = self.transform(Image.fromarray(sample['image']))

        filename = ''
        for object in sample['objects']:
            filename += f'{object[1]}_'
        filename += f'{randint(0, 999):03d}'
        if self.segmentation:
            if self.target == 'texture':
                seg = torch.from_numpy(sample['seg_tex'])
            else:
                seg = torch.from_numpy(sample['seg_shape'])
            return img, seg, {'objects': sample['objects'], 'background': sample['background'], 'filename': filename}
        else:
            return img, {'objects': sample['objects'], 'background': sample['background'], 'filename': filename}

    def __len__(self):
        return len(self.ids)


class ToyDataset_Saved(Dataset):
    def __init__(self, root, mode, transform_fn=None, segmentation=False, target='texture'):
        self.root = root
        self.ids = []
        with open(os.path.join(root, 'imagesets', target+'s', mode+'.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                self.ids.append(l.rstrip())

        self.transform = transform_fn
        self.segmentation = segmentation
        self.target = target        

    def __getitem__(self, index):
        
        filename = self.ids[index]
        img =  Image.open(self.root/ 'images'/ f'{filename}.png').convert("RGB")  
        tree = ElementTree.parse(self.root / 'annotations'/ f'{filename}.xml')
        root = tree.getroot()
        annotations = XmlDictConfig(root)
        sample = {'objects': [], 'background': None}
        for _, objects in annotations['objects'].items():
            sample['objects'].append((objects['shape'], objects['texture']))
        sample['background'] = annotations['background']

        if self.transform is not None:
            img = self.transform(img)

        if self.segmentation:
            seg = Image.open(os.path.join(self.root, 'segmentations', self.target+"s", filename+'.png')).convert("RGB")
            seg = torchvision.transforms.functional.to_tensor(seg)
            return img, seg, {'objects': sample['objects'], 'background': sample['background'], 'filename': filename}
        else:
            return img, {'objects': sample['objects'], 'background': sample['background'], 'filename': filename}

    def __len__(self):
        return len(self.ids)


class ColorDataset(Dataset):
    def __init__(self, epoch_length, rgb, transform_fn=None, segmentation=False, multiclass=False):
        self.ids = range(0, epoch_length)
        self.transform = transform_fn
        self.segmentation = segmentation
        self.generator = color_generator.Generator(rgb)

        self.multiclass = multiclass

        if rgb:
            self.colors = [[255., 0., 0.], [0., 0., 255.], [255., 51., 204.], [0., 255., 0.]]
        else:
            self.colors = [170., 255., 85.]

    def __getitem__(self, index):
        
        if self.multiclass:
            num_classes = randint(1,2)
            if num_classes == 2:
                current_colors = random.sample(range(len(self.colors)-1), 2)
                sample = self.generator.generate_multilabel(self.colors[-1], [self.colors[i] for i in current_colors])
                logits = [0] * (len(self.colors) - 1)
                for i in current_colors:
                    logits[i] = 1
            else:
                current_color = randint(0,len(self.colors) -2)
                sample = self.generator.generate_sample(self.colors[-1], self.colors[current_color])
                logits = [0] * (len(self.colors) - 1)
                logits[current_color] = 1


        else:
            current_color = randint(0,len(self.colors) -2)
            sample = self.generator.generate_sample(self.colors[-1], self.colors[current_color])
            logits = [0] * (len(self.colors) - 1)
            logits[current_color] = 1

        seg = torch.from_numpy(sample)

        if self.transform is not None:
            img = self.transform(Image.fromarray(sample.astype(np.uint8)))
        else:
            img = torch.from_numpy(sample)
            if img.dim() == 3:
                img = img.permute(2,0,1)

        filename = ''
        filename += f'{randint(0, 9999):05d}'
        return img.float(), seg, {'logits': logits, 'filename': filename}
        

    def __len__(self):
        return len(self.ids)


class XmlDictConfig(dict):

    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself 
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a 
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})
