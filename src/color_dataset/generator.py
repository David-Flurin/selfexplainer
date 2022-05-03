from multiprocessing.sharedctypes import Value
from os.path import join
from xml.dom import ValidationErr
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


        
    def generate_sample(self, bg_value, fg_value):
        return self.__generate(bg_value, fg_value)
    

    def __generate(self, bg_value, fg_value):
        
        if self.rgb:
            sample_arr = np.reshape(np.random.choice([0, 1], self.img_size[0]*self.img_size[1], p=[self.bg_to_class_ratio, 1 - self.bg_to_class_ratio]), self.img_size)
            values_arr = np.ones((*self.img_size, 3))
            values_arr *= bg_value
            values_arr[sample_arr == 1] = fg_value
            return values_arr
        else:
            sample_arr = np.random.choice([bg_value, fg_value], self.img_size[0]*self.img_size[1], p=[self.bg_to_class_ratio, 1 - self.bg_to_class_ratio])
            return np.reshape(sample_arr, self.img_size)


