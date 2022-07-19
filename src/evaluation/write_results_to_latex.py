import numpy as np
from statistics import mean
import latextable
from texttable import Texttable
import sys
import json

result_files = ['results/resnet_toy_singlelabel.npz', 'results/results_toy_singlelabel.npz']

try:
    checkpoint_dict = json.loads(sys.argv[2])
except:
    checkpoint_dict = None


mode = 'micro'
checkpoint_dict = {"toy_singlelabel": "Resnet50 Classifier", "1_pass": "Simple Selfexplainer", "3_passes_frozen_final": "Selfexplainer"}
metric_dict = {'classification_metrics': 'classification_metrics'}

table = Texttable()
column_align = ['l']

results = {}
for results_file in result_files:
    with np.load(results_file, allow_pickle=True) as results_file:
        results = {**results, **(results_file["results"].item())}

metric_names = list(metric_dict.values())
if 'classification_metrics' in metric_dict.keys():
    i = metric_names.index('classification_metrics')
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
    for metric in metric_dict.keys():
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
best_metrics = dict.fromkeys(metric_names,(-1000, -1))
if 'sal' in best_metrics:
    best_metrics['sal'] = (1000, -1)
if 'background_c' in best_metrics:
    best_metrics['background_c'] = (1000, -1)

for i, row in enumerate(rows[1:], 1):
    for j, m in enumerate(row[1:], 1):
        if rows[0][j] in ['sal', 'background_c']:
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
