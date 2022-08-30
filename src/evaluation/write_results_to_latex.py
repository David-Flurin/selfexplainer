import numpy as np
from statistics import mean
import latextable
from texttable import Texttable
import sys
import json

metrics_dict = {'d_f1_25': 'd_f1_25', 'd_f1_50': 'd_f1_50', 'd_f1_75': 'd_f1_75', 'd_f1': 'd_f1', 'c_f1': 'Continuous F1', 'a_f1s': 'Average F1', 'aucs': 'aucs', 'd_IOU': 'Discrete IoU', 'c_IOU': 'Continuous IoU', 'sal': 'Insertion', 'bg_sal': 'Deletion', 'background_entropy':'Deletion', 'combined_sal_wo_mean': 'Combined', 'over': 'over', 'background_c': 'Background cov.', 'mask_c': 'Mask cov.', 'sr': 'sr', 'classification_metrics': 'Classification metrics'}

def find_checkpoint(result_dict):
    results = {}
    for k in result_dict:
        if k in checkpoint_dict.keys():
            results[k] = result_dict[k]
        else:
            try:
                d = find_checkpoint(result_dict[k])
                results = {**results, **d}
            except:
                continue
    return results

def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


result_files = ['/home/david/Documents/Master/Thesis/selfexplainer/src/evaluation/results/selfexplainer/ablation/newer_saliency/toy_multi.npz']

try:
    checkpoint_dict = json.loads(sys.argv[2])
except:
    checkpoint_dict = None

find_best_metric = True


mode = 'micro'
checkpoint_dict = {'toy_multi_wo_background': 'L_{BE}', 'toy_multi_wo_similarity':'L_{obj}', 'toy_multi_wo_mask': 'L_{m}'}
metric_list = ['mask_c', 'background_c', 'c_IOU', 'sal', 'background_entropy', 'combined_sal_wo_mean', 'classification_metrics']
#metric_list = ['classification_metrics']

table = Texttable()
column_align = ['l']

results = {}
for results_file in result_files:
    with np.load(results_file, allow_pickle=True) as results_file:
        int_results = results_file["results"].item()
        results = merge(results, int_results)

results = find_checkpoint(results)

metric_names = [metrics_dict[m] for m in metric_list]
if 'classification_metrics' in metric_list:
    i = metric_names.index('Classification metrics')
    metric_names = metric_names[:i] + ['F1', 'Precision', 'Recall'] + metric_names[i+1:]
column_align += ['c' for i in range(len(metric_names))]
table.set_cols_align(column_align)
rows = [] 
rows.append([''] + metric_names)
for model in results:
    if checkpoint_dict and model in checkpoint_dict:
        model_name = checkpoint_dict[model]
    elif checkpoint_dict:
        continue
    else:
        model_name = model
    metrics = [model_name]
    for metric in metric_list:
        if metric not in results[model]:
            metrics.append(None)
        if metric == 'classification_metrics':
            for c_metric in results[model][metric]:
                metrics.append(results[model][metric][c_metric][mode])
        else:
            values = [x for x in results[model][metric] if x]
            try:
                metrics.append(mean(values))
            except:
                metrics.append(mean([mean(values) for values in results[model][metric]]))
    rows.append(metrics)
    
# Calculate best entry per metric and boldface it
if find_best_metric:
    best_metrics = dict.fromkeys(metric_names,(-1000, -1))
    if 'Insertion' in best_metrics:
        best_metrics['Insertion'] = (1000, -1)
    if 'Deletion' in best_metrics:
        best_metrics['Deletion'] = (1000, -1)
    if 'Combined' in best_metrics:
        best_metrics['Combined'] = (1000, -1)
    if 'Background cov.' in best_metrics:
        best_metrics['Background cov.'] = (1000, -1)

    for i, row in enumerate(rows[1:], 1):
        for j, m in enumerate(row[1:], 1):
            if rows[0][j] in ['Insertion', 'Deletion', 'Combined', 'Background cov.']:
                if m < best_metrics[rows[0][j]][0]:
                    best_metrics[rows[0][j]] = (m,i)
            else:
                if m > best_metrics[rows[0][j]][0]:
                    best_metrics[rows[0][j]] = (m,i)
    for metric, (best_m, best_idx) in best_metrics.items():
        rows[best_idx][rows[0].index(metric)] = f'\\bfseries{{{float(rows[best_idx][rows[0].index(metric)]):.3f}}}'
table.add_rows(rows)
latex_table = latextable.draw_latex(table, caption='Different models on Toy dataset')

latex_table = latex_table.replace('_', '\_')
print(latex_table)
