from multiprocessing.sharedctypes import Value
from os.path import join
import cv2
from random import sample
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
import math

import numpy as np

def get_i(idx, list):
    return [list[i] for i in idx]


class Generator:

    base = 'dataset'
    img_size = (10,10)
    bg_to_class_ratio = 0.7

    def __init__(self, rgb):


        base_dir = '/'.join(__file__.split('/')[0:-1])
        self.rgb = rgb


        
    def generate_sample(self, bg_value, fg_value, rgb=False):
        return self.__generate(bg_value, fg_value)

    
    def create(self, number_per_shape, proportions):

        proportions = [float(x) for x in proportions]
        if len(proportions) != 3 or sum(proportions) != 1.0:
            raise ValueError('\'Proportions\' must have three elements (Train, Val, Test) and must sum to 1')


        self.__directories()


        samples_shape = {}
        samples_texture = {}
        with tqdm(total=number_per_shape*len(self.shapes)) as pbar:
            num_length = math.ceil(math.log(number_per_shape*len(self.shapes), 10))+1
            i = 1
            for _ in range(number_per_shape):
                for shape in self.shapes:
                    filename = f'{i:0{num_length}d}'
                    shapes, textures = self.generate([shape], filename)
                    for s in shapes:
                        try:
                            samples_shape[s] += [filename]
                        except KeyError:
                            samples_shape[s] = [filename]
                    for t in textures:
                        try:
                            samples_texture[self.f_texture_names[t]] += [filename]
                        except KeyError:
                            samples_texture [self.f_texture_names[t]] = [filename]
                    
                    pbar.update()
                    i += 1
        
        

        train = []
        val = []
        test = []
        for k,v in samples_shape.items():
            v.sort(key=int)
            indices = range(0, len(v))
            total = len(indices)
            num_train = int(proportions[0] * total)
            num_val = int(proportions[1] * total)
            num_test = int(proportions[2] * total)
            num_train += (total - num_train - num_val - num_test)
            train_idx = sample(indices, num_train)
            indices = [i for i in indices if i not in train_idx]
            val_idx = sample(indices, num_val)
            test_idx = [i for i in indices if i not in val_idx]

            self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', f'{k}_train.txt'), get_i(train_idx, v))
            self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', f'{k}_val.txt'), get_i(val_idx, v))
            self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', f'{k}_test.txt'), get_i(test_idx, v))
            
            
            train += (get_i(train_idx, v))
            val += (get_i(val_idx, v))
            test += (get_i(test_idx, v))
        
        train.sort(key=int)
        val.sort(key=int)
        test.sort(key=int)
        self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', 'train.txt'), train)
        self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', 'val.txt'), val)
        self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', 'test.txt'), test)

        train = []
        val = []
        test = []
        for k,v in samples_texture.items():
            v.sort(key=int)
            indices = range(0, len(v))
            total = len(indices)
            num_train = int(proportions[0] * total)
            num_val = int(proportions[1] * total)
            num_test = int(proportions[2] * total)
            num_train += (total - num_train - num_val - num_test)
            train_idx = sample(indices, num_train)
            indices = [i for i in indices if i not in train_idx]
            val_idx = sample(indices, num_val)
            test_idx = [i for i in indices if i not in val_idx]

            self.__write_sample_list(Path(self.base, 'imagesets', 'textures', f'{k}_train.txt'), get_i(train_idx, v))
            self.__write_sample_list(Path(self.base, 'imagesets', 'textures', f'{k}_val.txt'), get_i(val_idx, v))
            self.__write_sample_list(Path(self.base, 'imagesets', 'textures', f'{k}_test.txt'), get_i(test_idx, v))
            
            
            train += (get_i(train_idx, v))
            val += (get_i(val_idx, v))
            test += (get_i(test_idx, v))

        train.sort(key=int)
        

                    
    def __write_sample_list(self, filename, list):
        with open(filename, 'w') as f:
            for sample in list:
                f.write(f'{sample}\n')

    

    def __generate(self, bg_value, fg_value):
        
        if self.rgb:
            pass
        else:
            sample_arr = np.random.choice([bg_value, fg_value], self.img_size[0]*self.img_size[1], p=[self.bg_to_class_ratio, 1 - self.bg_to_class_ratio])
            return np.reshape(sample_arr, self.img_size)


    def __save(self, sample, filename):
        
        annotation = ET.Element('annotation')
        xml_filename = ET.SubElement(annotation, 'filename')
        xml_filename.text = filename
        xml_objects = ET.SubElement(annotation, 'objects')
        
        for i, obj in enumerate(sample['objects']):
            xml_obj = ET.SubElement(xml_objects, f'object{i}')
            s = ET.SubElement(xml_obj, 'shape')
            s.text = obj[0]
            t = ET.SubElement(xml_obj, 'texture')
            t.text = obj[1]
        background = ET.SubElement(annotation, 'background')
        background.text = sample['background']


        cv2.imwrite(join(self.base, 'images', f'{filename}.jpg'), sample['image'])
        cv2.imwrite(join(self.base, 'segmentations', f'{filename}.jpg'), sample['seg'])

        tree = ET.ElementTree(annotation)
        tree.write(join(self.base, 'annotations', f'{filename}.xml'))




