import os

import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from matplotlib import pyplot as plt
import numpy as np


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
    rect_acc = ax.bar(x - width, metrics['Accuracy'].cpu(), width, label='Accuracy')
    rect_pre = ax.bar(x, metrics['Precision'].cpu(), width, label='Precision')
    rect_rec = ax.bar(x + width, metrics['Recall'].cpu(), width, label='Recall')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metrics')
    ax.set_title('Metrics per class')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rect_acc, padding=3)
    ax.bar_label(rect_pre, padding=3)
    ax.bar_label(rect_rec, padding=3)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

