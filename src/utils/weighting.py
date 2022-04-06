import torch

def softmax_weighting(t, k):
    ''' Multiply tensor with K, then apply softmax to obtain weights.
        If k = 0, this function is equivalent to Global Average Pooling.
        If k = 1, it is normal softmax.
        If k >> 1, it is Global Max Pooling.'''

    batch_size, num_classes, h, w = t.size()
    for i in range(batch_size):
        for c in range(num_classes):
            weights = torch.softmax(t[i, c].view(h*w) * k, 0)
            t[i, c] = torch.reshape(weights, (h, w))
    return  t

