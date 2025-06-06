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
    img_size = (5,5)
    class_to_bg_ratio = 1/3

    def __init__(self, rgb):


        base_dir = '/'.join(__file__.split('/')[0:-1])
        self.rgb = rgb


        
    def generate_sample(self, bg_value, fg_value):
        return self.__generate(bg_value, fg_value)
    

    def __generate(self, bg_value, fg_value):
        
        total = self.img_size[0] * self.img_size[1]
        if self.rgb:
            sample_arr = np.random.choice(total, round(total*self.class_to_bg_ratio), replace=False)
            values_arr = np.ones((total, 3))
            values_arr *= bg_value
            values_arr[sample_arr] = fg_value
            return np.reshape(values_arr, (*self.img_size, 3))
        else:
            sample_arr = np.random.choice(total, round(total*self.class_to_bg_ratio), replace=False)
            values_arr = np.ones(total)
            values_arr *= bg_value
            values_arr[sample_arr] = fg_value
            return np.reshape(values_arr, self.img_size)

    def generate_multilabel(self, bg_value, fg_values):
        total = self.img_size[0] * self.img_size[1]
        if self.rgb:
            sample_arr = np.random.choice(total, round(total*self.class_to_bg_ratio), replace=False)
            fg_1_arr = np.random.choice(sample_arr, round(total*self.class_to_bg_ratio*0.5), replace=False)
            fg_2_idx = np.isin(sample_arr, fg_1_arr)
            fg_2_arr = sample_arr[~fg_2_idx]
            values_arr = np.ones((total, 3))
            values_arr *= bg_value
            values_arr[fg_1_arr] = fg_values[0]
            values_arr[fg_2_arr] = fg_values[1]
            return np.reshape(values_arr, (*self.img_size, 3))
        else:
            pass



