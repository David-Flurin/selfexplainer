import torch

def softmax_weighting(t, k):
    ''' Multiply tensor with K, then apply softmax to obtain weights.
        If k = 0, this function is equivalent to Global Average Pooling.
        If k = 1, it is normal softmax.
        If k >> 1, it is Global Max Pooling.'''

    batch_size, num_classes, h, w = t.size()
    s = h*w
    for i in range(batch_size):
        for c in range(num_classes):
<<<<<<< HEAD
            weights = torch.softmax(t[i, c].flatten() * k, 0)
=======
            weights = torch.softmax(t[i, c].view(s) * k, 0)
>>>>>>> fd74c1f0263350a744c2342bd6a5f52b6c7ab301
            t[i, c] = torch.reshape(weights, (h, w))
    return  t

