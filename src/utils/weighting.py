import torch
import cv2
import numpy as np

def softmax_weighting(t, k):
    ''' Multiply tensor with K, then apply softmax to obtain weights.
        If k = 0, this function is equivalent to Global Average Pooling.
        If k = 1, it is normal softmax.
        If k >> 1, it is Global Max Pooling.'''

    batch_size, num_classes, h, w = t.size()
    weighted = torch.zeros(batch_size, num_classes, h, w, device=('cuda' if t.is_cuda else 'cpu'))
    for i in range(batch_size):
        for c in range(num_classes):
            weights = torch.softmax(t[i, c].view(h*w) * k, 0)
            weighted[i, c] = torch.reshape(weights, (h, w)) * t[i, c]
    return  weighted


# for i in range(3):
#     kernel = cv2.getGaussianKernel(224, 0.3*((224-1)*0.5 - 1) + 1.5)
#     kernel = np.dot(kernel, kernel.T)
#     kernel *= 1000
#     print(kernel.mean())
#     from matplotlib import pyplot as plt
#     plt.set_cmap('jet')
#     plt.imshow(kernel, vmin=0, vmax=0.2)
#     plt.show()
#     w_k = softmax_weighting(torch.tensor(kernel).unsqueeze(0).unsqueeze(0), i)*(224*224)
#     print(w_k.mean())
#     plt.imshow(w_k[0, 0], vmin=0, vmax=0.2)
#     plt.show()