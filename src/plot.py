import os
from cv2 import sqrt
import cv2

import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import json
from utils.helper import get_class_dictionary
import torch
from matplotlib.ticker import FormatStrFormatter
from synthetic_dataset.generator import Generator
import pathlib
from xml.etree import cElementTree as ElementTree
from utils.weighting import softmax_weighting

from utils.assessment_metrics import background_entropy

from math import sqrt
import itertools
from functools import partial

import matplotlib.ticker as mticker
from cycler import cycler

from tqdm import tqdm

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

def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.
    
    """
    

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )
    
    columns_order = ['wall_time', 'name', 'step', 'value']
    
    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df.reset_index(drop=True)


def plot_losses(event_folder, names, save_path):
    df = convert_tb_data(event_folder)
    losses = {}
    for n in names:
        losses[n] = {}
    for r in df.iterrows():
        if r[1][1] in names:
            losses[r[1][1]][r[1][2]] = r[1][3] 
    
    steps = []
    for loss, s in losses.items():
        steps = list(set(steps) | set(s.keys()))
    steps = sorted(steps)
    #steps = [s for s in steps if s % 5 == 0]
    values = {}
    for n in names:
        values[n] = []
    for step in steps:
        for loss, s in losses.items():
            if step in s.keys():
                values[loss].append(s[step])

    for loss, value in values.items():
        if len(value) == 0:
            continue
        alpha = 0.8
        if loss == 'loss':
            alpha = 1.
        plt.plot(steps, value, label = loss, linewidth=1., alpha=alpha)
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def plot_class_metrics(labels, metrics, save_path):
    
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    rect_acc = ax.bar(x - width, metrics['Accuracy'].cpu(), width, label='Accuracy')
    rect_pre = ax.bar(x, metrics['Precision'].cpu(), width, label='Precision')
    rect_rec = ax.bar(x + width, metrics['Recall'].cpu(), width, label='Recall')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metrics')
    ax.set_title('Metrics per class')
    ax.set_xticks(x, labels)
    ax.legend()
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.bar_label(rect_acc, padding=3, fmt='%.2f')
    ax.bar_label(rect_pre, padding=3, fmt='%.2f')
    ax.bar_label(rect_rec, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metrics_from_file(jsonfile):
    with open(jsonfile, 'r') as f:
        metrics = json.load(f)
        metrics = dict_list_to_tensor(metrics)
        plot_class_metrics(list(get_class_dictionary('OI')), metrics['Class'], '../test.png')

def dict_list_to_tensor(dict):
    if type(dict) == list:
        return torch.Tensor(dict)
    elif type(dict) == float:
        return torch.Tensor([dict])
    else:
        for k,v in dict.items():
            dict[k] = dict_list_to_tensor(v)
        return dict



def filled_hist(ax, edges, values, bottoms=None, orientation='v',
                **kwargs):
    """
    Draw a histogram as a stepped patch.

    Parameters
    ----------
    ax : Axes
        The axes to plot to

    edges : array
        A length n+1 array giving the left edges of each bin and the
        right edge of the last bin.

    values : array
        A length n array of bin counts or values

    bottoms : float or array, optional
        A length n array of the bottom of the bars.  If None, zero is used.

    orientation : {'v', 'h'}
       Orientation of the histogram.  'v' (default) has
       the bars increasing in the positive y-direction.

    **kwargs
        Extra keyword arguments are passed through to `.fill_between`.

    Returns
    -------
    ret : PolyCollection
        Artist added to the Axes
    """
    print(orientation)
    if orientation not in 'hv':
        raise ValueError("orientation must be in {{'h', 'v'}} "
                         "not {o}".format(o=orientation))

    kwargs.setdefault('step', 'post')
    kwargs.setdefault('alpha', 0.7)
    edges = np.asarray(edges)
    values = np.asarray(values)
    if len(edges) - 1 != len(values):
        raise ValueError('Must provide one more bin edge than value not: '
                         'len(edges): {lb} len(values): {lv}'.format(
                             lb=len(edges), lv=len(values)))

    if bottoms is None:
        bottoms = 0
    bottoms = np.broadcast_to(bottoms, values.shape)

    values = np.append(values, values[-1])
    bottoms = np.append(bottoms, bottoms[-1])
    if orientation == 'h':
        return ax.fill_betweenx(edges, values, bottoms,
                                **kwargs)
    elif orientation == 'v':
        return ax.fill_between(edges, values, bottoms,
                               **kwargs)
    else:
        raise AssertionError("you should never be here")


def stack_hist(ax, stacked_data, sty_cycle, bottoms=None,
               hist_func=None, labels=None,
               plot_func=None, plot_kwargs=None):
    """
    Parameters
    ----------
    ax : axes.Axes
        The axes to add artists too

    stacked_data : array or Mapping
        A (M, N) shaped array.  The first dimension will be iterated over to
        compute histograms row-wise

    sty_cycle : Cycler or operable of dict
        Style to apply to each set

    bottoms : array, default: 0
        The initial positions of the bottoms.

    hist_func : callable, optional
        Must have signature `bin_vals, bin_edges = f(data)`.
        `bin_edges` expected to be one longer than `bin_vals`

    labels : list of str, optional
        The label for each set.

        If not given and stacked data is an array defaults to 'default set {n}'

        If *stacked_data* is a mapping, and *labels* is None, default to the
        keys.

        If *stacked_data* is a mapping and *labels* is given then only the
        columns listed will be plotted.

    plot_func : callable, optional
        Function to call to draw the histogram must have signature:

          ret = plot_func(ax, edges, top, bottoms=bottoms,
                          label=label, **kwargs)

    plot_kwargs : dict, optional
        Any extra keyword arguments to pass through to the plotting function.
        This will be the same for all calls to the plotting function and will
        override the values in *sty_cycle*.

    Returns
    -------
    arts : dict
        Dictionary of artists keyed on their labels
    """
    # deal with default binning function
    if hist_func is None:
        hist_func = np.histogram

    # deal with default plotting function
    if plot_func is None:
        plot_func = filled_hist

    # deal with default
    if plot_kwargs is None:
        plot_kwargs = {}
    print(plot_kwargs)
    try:
        l_keys = stacked_data.keys()
        label_data = True
        if labels is None:
            labels = l_keys

    except AttributeError:
        label_data = False
        if labels is None:
            labels = itertools.repeat(None)

    if label_data:
        loop_iter = enumerate((stacked_data[lab], lab, s)
                              for lab, s in zip(labels, sty_cycle))
    else:
        loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))

    arts = {}
    for j, (data, label, sty) in loop_iter:
        if label is None:
            label = 'dflt set {n}'.format(n=j)
        label = sty.pop('label', label)
        vals, edges = hist_func(data)
        if bottoms is None:
            bottoms = np.zeros_like(vals)
        top = bottoms + vals
        print(sty)
        sty.update(plot_kwargs)
        print(sty)
        ret = plot_func(ax, edges, top, bottoms=bottoms,
                        label=label, **sty)
        bottoms = top
        arts[label] = ret
    ax.legend(fontsize=10)
    return arts




def plot_generator_distribution(num_samples):
    g = Generator(pathlib.Path('/home/david/Documents/Master/Thesis/selfexplainer/src/toy_dataset', 'foreground.txt'), pathlib.Path('/home/david/Documents/Master/Thesis/selfexplainer/src/toy_dataset', 'background.txt'))
    # shapes = {k: {} for k in g.f_texture_names}
    # sizes = {k: [] for k in g.f_texture_names}
    # bg_tex = {k: {} for k in g.f_texture_names}

    # for i in tqdm(range(num_samples)):
    #     s = g.generate_sample(1)
    #     f_tex = s['objects'][0][1]
    #     shape = s['objects'][0][0]
    #     radius = s['objects'][0][2]
    #     b_tex = s['background']

    #     shapes[f_tex][shape] = shapes[f_tex].get(shape, 0) + 1
    #     sizes[f_tex].append(radius)
    #     bg_tex[f_tex][b_tex] = bg_tex[f_tex].get(b_tex, 0) + 1

    # for k,v in sizes.items():
    #     sizes[k] = sorted(v)

    # np.savez('generator_100000.npz', shapes=shapes, sizes=sizes, bg_tex=bg_tex)
    # quit()
    result = np.load('generator_100000.npz', allow_pickle=True)
    shapes = result['shapes'].item()
    sizes = result['sizes'].item()
    bg_tex = result['bg_tex'].item()
# --------------Shapes plot---------------------   
    plt.rcParams["font.family"] = "Roboto Mono"
    labels = list(shapes.keys())
    x = np.arange(len(labels))  # the label locations
    width = 0.05  # the width of the bars

    
    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(4)

    rects = []
    for i, shape in enumerate(shapes[list(shapes.keys())[0]].keys()):
        rects.append(ax.bar(x + (-3.5+i)*(width+0.01), [shapes[f][shape] for f in labels], width, label=shape, color='steelblue'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of samples')
    ax.set_xlabel('Classes (Texture)')
    ax.set_title('Distribution of object shapes per texture class')
    ax.set_xticks(x, labels)
    plt.axis([-1, 8, 1400, 1700])


    mean = int(num_samples/64)
    diff = []
    for shape in shapes:
        for t in shapes[shape]:
            diff.append(abs(shapes[shape][t]-mean)**2)
    std = sqrt(sum(diff) / 64)
    print(std)

    plt.axhline(y = mean, color = 'k', linewidth=0.5, linestyle = 'dashed')    
    plt.text(7.5,mean+3,'mean')
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # for rect in rects:
    #     ax.bar_label(rect, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig('shape_distribution.png')
    plt.close()
#-------------------------------------------------

#--------------Background plots-------------------
    labels = list(bg_tex.keys())
    x = np.arange(len(labels))  # the label locations
    width = 0.05  # the width of the bars

    
    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(4)

    rects = []
    for i, b_tex in enumerate(bg_tex[list(bg_tex.keys())[0]].keys()):
        rects.append(ax.bar(x + (-6.5+i)*(width+0.01), [bg_tex[f][b_tex] for f in labels], width, label=b_tex, color='tomato'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of samples')
    ax.set_xlabel('Classes (Textures)')
    ax.set_title('Distribution of background textures per texture class')
    ax.set_xticks(x, labels)
    plt.axis([-1, 8, 800, 1000])


    mean = int(num_samples/(14*8))
    diff = []
    for tex in bg_tex:
        for t in bg_tex[tex]:
            diff.append(abs(bg_tex[tex][t]-mean)**2)
    std = sqrt(sum(diff) / (14*8))
    print(std)
    plt.axhline(y = mean, color = 'k', linewidth=0.5, linestyle = 'dashed')    
    plt.text(7.5,mean+3,'mean')
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # for rect in rects:
    #     ax.bar_label(rect, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig('bgtex_distribution.png')
    plt.close()
#---------------------------------------------------

#--------------Radius plots-------------------
    # set up histogram function to fixed bins
    hist_func = partial(np.histogram, bins=51)

    # set up style cycles
    color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:8])
    label_cycle = cycler(label=['set {n}'.format(n=n) for n in range(4)])
    hatch_cycle = cycler(hatch=['/', '*', '+', '|'])

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)

    arts = stack_hist(ax, sizes, color_cycle,
                    hist_func=hist_func,
                    plot_kwargs=dict(edgecolor='w'))

    ax.set_xlabel('Object radius')
    ax.set_ylabel('Number of objects')
    ax.set_title('Distribution of object size per texture class')
    mean = int(num_samples/51)
    plt.axhline(y = mean, color = 'k', linewidth=0.5, linestyle = 'dashed')    
    plt.text(30,mean+3,'mean')
    

    plt.savefig('size_distribution.png')
#---------------------------------------------------

def plot_toydata_distribution(data_path):
    g = Generator(pathlib.Path('/home/david/Documents/Master/Thesis/selfexplainer/src/toy_dataset', 'foreground.txt'), pathlib.Path('/home/david/Documents/Master/Thesis/selfexplainer/src/toy_dataset', 'background.txt'))

    shapes = {k: {} for k in g.f_texture_names}
    #sizes = {k: [] for k in g.f_texture_names}
    bg_tex = {k: {} for k in g.f_texture_names}

    for file in tqdm(os.listdir(os.path.join(data_path,  'annotations'))):
        tree = ElementTree.parse(os.path.join(data_path, 'annotations', file))
        root = tree.getroot()
        annotations = XmlDictConfig(root)
        for _, objects in annotations['objects'].items():
            f_tex = objects['texture']
            shape = objects['shape']
            shapes[f_tex][shape] = shapes[f_tex].get(shape, 0) + 1
            #radius = objects[0][2]
        b_tex = annotations['background']

        
        #sizes[f_tex].append(radius)
        bg_tex[f_tex][b_tex] = bg_tex[f_tex].get(b_tex, 0) + 1

    # for k,v in sizes.items():
    #     sizes[k] = sorted(v)

    
# --------------Shapes plot---------------------    
    labels = list(shapes.keys())
    x = np.arange(len(labels))  # the label locations
    width = 0.05  # the width of the bars

    
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    rects = []
    for i, shape in enumerate(shapes[list(shapes.keys())[0]].keys()):
        rects.append(ax.bar(x + (-3.5+i)*(width+0.01), [shapes[f][shape] for f in labels], width, label=shape))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metrics')
    ax.set_title('Metrics per class')
    ax.set_xticks(x, labels)
    ax.legend()
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # for rect in rects:
    #     ax.bar_label(rect, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.show()
    plt.close()
#-------------------------------------------------

#--------------Background plots-------------------
    labels = list(bg_tex.keys())
    x = np.arange(len(labels))  # the label locations
    width = 0.05  # the width of the bars

    
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    rects = []
    for i, b_tex in enumerate(bg_tex[list(bg_tex.keys())[0]].keys()):
        rects.append(ax.bar(x + (-6.5+i)*(width+0.01), [bg_tex[f][b_tex] for f in labels], width, label=b_tex))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metrics')
    ax.set_title('Metrics per class')
    ax.set_xticks(x, labels)
    ax.legend()
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # for rect in rects:
    #     ax.bar_label(rect, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.show()
    plt.close()
#---------------------------------------------------

# #--------------Radius plots-------------------
#     # set up histogram function to fixed bins
#     hist_func = partial(np.histogram, bins=51)

#     # set up style cycles
#     color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:8])
#     label_cycle = cycler(label=['set {n}'.format(n=n) for n in range(4)])
#     hatch_cycle = cycler(hatch=['/', '*', '+', '|'])

#     # Fixing random state for reproducibility
#     np.random.seed(19680801)

#     stack_data = np.random.randn(4, 12250)
#     dict_data = dict(zip((c['label'] for c in label_cycle), stack_data))

#     fig, ax = plt.subplots(figsize=(9, 4.5), tight_layout=True)

#     arts = stack_hist(ax, sizes, color_cycle,
#                     hist_func=hist_func,
#                     plot_kwargs=dict(edgecolor='w'))

#     ax.set_xlabel('object radius')
#     ax.set_ylabel('number of objects')

#     plt.show()
#---------------------------------------------------

def plot_attention_pooling(mask_size):
    
    tot_attr = mask_size**2 * 0.1
    avg_mask = np.ones((mask_size, mask_size)) * 0.1
    kernel = cv2.getGaussianKernel(int(mask_size/1.5), 0.1*((int(mask_size/1.5)-1)/2 - 1) + 1.5)
    kernel = np.dot(kernel, kernel.T)
    narrow_mask = np.zeros((mask_size, mask_size))
    idx = mask_size // 6
    narrow_mask[idx:idx*5, idx:idx*5] = tot_attr * kernel
    kernel = cv2.getGaussianKernel(mask_size//3, 0.1*((mask_size//3-1)/3 - 1) + 1.5)
    kernel = np.dot(kernel, kernel.T)
    narrower_mask = np.zeros((mask_size, mask_size))
    idx = mask_size // 6
    narrower_mask[idx*2:idx*4, idx*2:idx*4] = tot_attr * kernel

    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft = False)

    weight = lambda x: softmax_weighting(torch.from_numpy(x).unsqueeze(0).unsqueeze(0), 2).sum()
    #plt.figure(figsize=(mask_size, mask_size))
    plt.imshow(avg_mask, cmap='jet',  vmin=0, vmax=2)
    plt.tight_layout
    plt.savefig('../../../figures/attention_pooling/avg_mask.png')
    print(weight(avg_mask))
    print(avg_mask.sum())
    plt.imshow(narrow_mask, cmap='jet',  vmin=0, vmax=5)
    plt.savefig('../../../figures/attention_pooling/narrow_mask.png')
    print(weight(narrow_mask))
    print(narrow_mask.sum())
    plt.imshow(narrower_mask, cmap='jet',  vmin=0, vmax=10)
    plt.savefig('../../../figures/attention_pooling/narrower_mask.png')
    print(weight(narrower_mask))
    print(narrower_mask.sum())



#plot_attention_pooling(240)
#plot_generator_distribution(100000)


def plot_deletion_metrics():
    bg_sal = []
    log_entropy = []
    bg_ent = []
    entropy = []
    unnormalized_entropy = []
    ts = []
    m = np.zeros((100, 100))
    for t in np.arange(10, 1, -0.2):
        p = torch.ones(10)
        p[9] = t
        o = torch.nn.functional.softmax(p, dim=0).numpy()
        m[0:32, 0:32] = np.ones((32, 32))
        sal, log = background_saliency(o, m)
        bg_sal.append(sal)
        log_entropy.append(log)
        bg_e, ent, un_entropy = background_entropy(o, m)
        bg_ent.append(bg_e)
        entropy.append(ent)
        ts.append(t)
        unnormalized_entropy.append(un_entropy)

    fig, ax1 = plt.subplots()
    ax1.plot(ts, bg_sal, color='red', label='log(area) - log(norm.entropy)')
    ax1.plot(ts, bg_ent, color='blue', label='log(area) - norm.entropy')
    plt.xlabel('t: [t, 1, 1, 1, 1, 1, 1, 1, 1, 1]')

    plt.legend()
    fig.tight_layout()
    plt.show()

    plt.xlabel = 'Logit value'
    fig, ax1 = plt.subplots()
    ax1.plot(ts, log_entropy, color='red', label=' log(norm.entropy)')
    ax1.plot(ts, entropy, color='blue', label=' norm.entropy')
    ax1.plot(ts, unnormalized_entropy, color='green', label='unnorm.entropy')
    plt.xlabel = 't: [t, 1, 1, 1, 1, 1, 1, 1, 1, 1]'

    plt.legend()
    fig.tight_layout()
    plt.show()


# qrand = np.sort(np.random.rand(60))[::-1]
# qmask = np.zeros(100)
# qmask[0:10] = 1
# qmask[10:70] = qrand
# qmask[70:] = 0
# qmin = np.zeros(100)
# qmin[0:10] = 1
# qmax = np.zeros(100)
# qmax[0:50] = 1
# qmask_rest = np.zeros(100)
# qmask_rest[50:70] = qrand[40:]
# fig = plt.figure(figsize=(10,1))
# plt.bar(np.linspace(0, 100, 100), np.zeros(100), width=1.1, color='black')
# plt.yticks([0,1])
# plt.show()
# fig = plt.figure(figsize=(10,1))
# plt.bar(np.linspace(0, 100, 100), qmin, width=1.1, color='red')
# plt.yticks([0,1])
# plt.show()
# fig = plt.figure(figsize=(10,1))
# plt.bar(np.linspace(0, 100, 100), qmax, width=1.1, color='blue')
# plt.yticks([0,1])
# plt.show()
# fig = plt.figure(figsize=(10,1))
# plt.bar(np.linspace(0, 100, 100), qmask_rest, width=1.1, color='black')
# plt.yticks([0,1])
# plt.show()

