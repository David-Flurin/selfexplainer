import torch
import cv2
import numpy as np

def softmax_weighting(t, k):
    ''' Multiply tensor with K, then apply softmax to obtain weights.
        If k = 0, this function is equivalent to Global Average Pooling.
        If k >> 1, it approximates Global Max Pooling.'''

    batch_size, num_classes, h, w = t.size()
    weighted = torch.zeros(batch_size, num_classes, h, w, device=('cuda' if t.is_cuda else 'cpu'))
    for i in range(batch_size):
        for c in range(num_classes):
            weights = torch.softmax(t[i, c].view(h*w) * k, 0)
            weighted[i, c] = torch.reshape(weights, (h, w)) * t[i, c]
    return  weighted


