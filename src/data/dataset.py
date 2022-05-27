import os
import pickle
import pathlib
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from random import randint
from xml.etree import cElementTree as ElementTree
import json

from toy_dataset import generator
from color_dataset import generator as color_generator

class COCODataset(Dataset):
    def __init__(self, root, annotation, transform_fn=None):
        self.root = root
        self.transform = transform_fn
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objects = len(coco_annotation)
        cat_ids = []
        for i in range(num_objects):
            cat_ids.append(coco_annotation[i]['category_id'])

        targets = coco.getCatIds(catIds=cat_ids)

        my_annotation = {}
        my_annotation["targets"] = targets
        my_annotation["image_id"] = img_id
        my_annotation["filename"] = path

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

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
        self.img_to_class = {}
        for img, l in labels.items():
            if l == None:
                continue
            for i, c in target_classes.items():
                if i in l:
                    self.img_to_class[img] = [c] if img not in self.img_to_class else self.img_to_class[img] + [c]
        
        self.images = list(self.img_to_class.keys())
        print('Cats:', sum('cat' in value for value in self.img_to_class.values()))
        print('Dogs:', sum('dog' in value for value in self.img_to_class.values()))
        print('Sheeps:', sum('bird' in value for value in self.img_to_class.values()))

        self.transforms = transform_fn
        self.gt_field = gt_field
        

    def __getitem__(self, idx):
        img_path = self.root / 'data' / (self.images[idx] +'.jpg')

        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, {'annotation':{'object': [{'name': name} for name in self.img_to_class[self.images[idx]]], 'filename': self.images[idx]}}


    def __len__(self):
        return len(self.images)

    

class ToyDataset(Dataset):
    def __init__(self, epoch_length, transform_fn=None, segmentation=False, multiclass=False, target='texture'):
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

        self.num_objects = 2 if multiclass else 1
        

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
    def __init__(self, epoch_length, rgb, transform_fn=None, segmentation=False):
        self.ids = range(0, epoch_length)
        self.transform = transform_fn
        self.segmentation = segmentation
        self.generator = color_generator.Generator(rgb)

        if rgb:
            self.colors = [[255., 0., 0.], [0., 0., 255.], [0., 255., 0.]]
        else:
            self.colors = [170., 255., 85]

    def __getitem__(self, index):
        current_color = randint(0,1)
        sample = self.generator.generate_sample(self.colors[-1], self.colors[current_color])
        seg = torch.from_numpy(sample)

        if self.transform is not None:
            img = self.transform(Image.fromarray(sample))
        else:
            img = torch.from_numpy(sample)
            if img.dim() == 3:
                img = img.permute(2,0,1)

        filename = ''
        filename += f'{randint(0, 9999):05d}'
        logits = [1 - current_color, current_color]
        if self.segmentation:
            return img, seg, {'logits': logits, 'filename': filename}
        else:
            return img, {'logits': logits, 'filename': filename}

    def __len__(self):
        return len(self.ids)



class CUB200Dataset(Dataset):
    def __init__(self, root, annotations, transform_fn=None):
        self.root = root
        self.transform = transform_fn
        with open(annotations, 'rb') as fp:
            self.annotations = pickle.load(fp)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        filename = pathlib.PurePath(annotation['filename']).name

        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        class_label = annotation['class']['label']

        my_annotation = {}
        my_annotation["target"] = class_label
        my_annotation["filename"] = filename

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.annotations)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
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
