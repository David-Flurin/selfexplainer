import numpy as np
import torch

from sklearn.metrics import roc_auc_score

#### PLAIN METRICS TO BENCHMARK 
def prob_sparsity(pvec):
    """Sparsity measure.
    
    For pvec of the masked image, we want this to be low.
    For pvec of the inverse masked image, we want this to be high.
    """
    return np.sum(pvec**2)


def prob_entropy(pvec):
    """Sparsity measure.
    
    For pvec of the masked image, we want this to be low.
    For pvec of the inverse masked image, we want this to be high.
    """
    return -np.sum(pvec * np.log(np.maximum(pvec, 1e-15)))

def saliency(pvec, c, mask):
    """
    Continuous saliency measure. 
    
    Adaptation from "Real Time Image Saliency for Black Box Classifiers
Piotr", Dabkowski and Gal.

    For pvec of the masked image, the lower the better for the masked image.
    
    This measure does not make sense for the inverse masked image.
    """
    a = np.maximum(np.mean(mask), 0.05)
    if isinstance(c, int):
        pclass = pvec[c]
    else:
        pclass = 0
        for e in c:
            pclass += pvec[e]
    return np.log(a) - np.log(pclass.mean())

def background_saliency(pvec, mask):
    """
    Continuous saliency measure for the inverse masked image.


    """
    a = np.maximum(np.mean(mask), 0.05)

    max_entropy = -np.log(1/pvec.size)

    entropy = (-pvec * np.log(pvec)).sum()
    normalized_entropy = entropy / max_entropy

    return np.log(a) - np.log(normalized_entropy), np.log(normalized_entropy)


def background_entropy(pvec, mask):
    """
    Continuous saliency measure for the inverse masked image.


    """
    a = np.maximum(np.mean(mask), 0.05)

    avg_probs = np.array([1/pvec.size]*pvec.size)
    max_entropy = (-avg_probs * np.log(avg_probs)).sum()
    
    entropy = (-pvec * np.log(pvec)).sum()
    normalized_entropy = entropy / max_entropy

    return np.log(a) - normalized_entropy, normalized_entropy, entropy

def combined_saliency(pvec, inv_pvec, c, mask):
    """
    Extended saliency measure. 
    
    Adaptation from "Real Time Image Saliency for Black Box Classifiers
Piotr", Dabkowski and Gal.

    For pvec of the masked image, the lower the better for the masked image.
    
    This measure does not make sense for the inverse masked image.
    """
    a = np.maximum(np.mean(mask), 0.05)

    avg_probs = np.array([1/inv_pvec.size]*inv_pvec.size)
    max_entropy = (-avg_probs * np.log(avg_probs)).sum()
    
    entropy = (-inv_pvec * np.log(inv_pvec)).sum()
    normalized_entropy = entropy / max_entropy

    if isinstance(c, int):
        pclass = pvec[c]
        inv_pclass = inv_pvec[c]
    else:
        pclass = 0
        inv_pclass = 0
        for e in c:
            pclass += pvec[e]
            inv_pclass += inv_pvec[e]

    return np.log(a) - (np.log(pclass.mean()) + np.log(normalized_entropy)).mean()

def combined_saliency_wo_mean(pvec, inv_pvec, c, mask):
    """
    Extended saliency measure. 
    
    Adaptation from "Real Time Image Saliency for Black Box Classifiers
Piotr", Dabkowski and Gal.

    For pvec of the masked image, the lower the better for the masked image.
    
    This measure does not make sense for the inverse masked image.
    """
    a = np.maximum(np.mean(mask), 0.05)

    avg_probs = np.array([1/inv_pvec.size]*inv_pvec.size)
    max_entropy = (-avg_probs * np.log(avg_probs)).sum()
    
    entropy = (-inv_pvec * np.log(inv_pvec)).sum()
    normalized_entropy = entropy / max_entropy

    if isinstance(c, int):
        pclass = pvec[c]
        inv_pclass = inv_pvec[c]
    else:
        pclass = 0
        inv_pclass = 0
        for e in c:
            pclass += pvec[e]
            inv_pclass += inv_pvec[e]

    return np.log(a) - np.log(pclass.mean()) + np.log(normalized_entropy)

def continuous_IOU(mask, seg):
    ### this is no longer the IoU but 1 + the Soergel distance (which is 1 - this ratio below)
    #intersection = np.sum(mask * seg)
    #union = np.sum(mask + seg)/2
    #union = np.sum(mask + seg) - intersection
    intersection = np.sum(np.minimum(mask, seg))
    union = np.sum(np.maximum(mask, seg))
    IOU = intersection/(union + 1e-15)
    return IOU
    
def discrete_IOU(mask, seg, thresh=0.5):

    """ Binarize the mask at 'tresh'. Then compute intersection (AND) and union (OR) and calculate intersection/union. """

    mask = mask>thresh
    seg = seg>thresh
    intersection = np.sum(np.logical_and(mask, seg)*1.)
    union = np.sum(np.logical_or(mask, seg)*1.)
    IOU = intersection/(union + 1e-15)
    return IOU

def soft_f1(p,q):

    """ Calculate continuous f1-score (without binarizing before). """

    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum(p*q)
    fp = np.sum(p*(1-q))
    fn = np.sum((1-p)* q)
    precision = tp/(tp+fp  + 1e-10)
    recall = tp/(tp+fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1

def f1(actual, predicted, label):

    """ A helper function to calculate f1-score for the given `label` """

    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum((actual==label) & (predicted==label))
    fp = np.sum((actual!=label) & (predicted==label))
    fn = np.sum((predicted!=label) & (actual==label))
    precision = tp/(tp+fp  + 1e-10)
    recall = tp/(tp+fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1

def f1_macro(actual, predicted):
    # `macro` f1- unweighted mean of f1 per label
    return np.mean([f1(actual, predicted, label) for label in [True, False]])

def discrete_f1(mask, seg, thresh=0.5):

    """ Binarize the mask at 'tresh', then calculate f1-score for both labels (True, False) and take mean."""

    mask = mask>thresh
    seg = seg>thresh
    return f1_macro(seg, mask)

def f1s(mask, seg):

    """ Calculate the discrete f1-score for each treshold from 0.1 to 1 in 0.1 steps."""

    return [discrete_f1(mask, seg, thresh) for thresh in np.arange(0.1, 1, 0.1)]

def auc(mask, seg):

    "Compute area under ROC "

    try:
        return roc_auc_score((seg.flatten()>0)*1., mask.flatten())
    except:
        return None

def sim_ratio(mask, seg):
    scp = np.sum(mask*seg)
    sr = scp / (scp + np.sum((mask - seg)**2) + 1e-10)
    
    return sr



def mask_coverage(mask, seg_mask):
    
    """Compute the true positive rate (proportion of correct classified foreground compared to foreground)."""

    seg_area = np.average(seg_mask)
    
    if seg_area > 0.0:
        seg_diff = np.clip(seg_mask - mask, a_min=0.0, a_max=None)
        return 1 - (np.average(seg_diff) * (1 / seg_area))
    else:
        return 0

def background_coverage(mask, seg_mask):

    """Compute the false positive rate (proportion of false classified foreground compared to background)."""
    
    seg_area = np.average(seg_mask)

    if seg_area < 1.0:
        non_seg_diff = np.clip(mask - seg_mask, a_min=0.0, a_max=None)
        return (np.average(non_seg_diff) * (1 / (1-seg_area)))
    else:
        return 0

def overlap(mask, seg_mask):

    "Return proportion where mask and groundtruth match. "
    
    return 1 - np.average(np.absolute(seg_mask - mask))



# t = np.array([0.22, 0.22, 0.22, 0.34])
# t1 = np.array([0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.19])
# p = np.array([0.05, 0.05, 0.05, 0.85])
# p1 = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.82])
# m = np.zeros((100, 100))
# m[0:40, 0:40] = np.ones((40,40))
# print(background_saliency(t, m))
# print(combined_saliency(p, t, 3, m))

# print(background_saliency(t1, m))
# print(combined_saliency(p1, t1, 9, m))