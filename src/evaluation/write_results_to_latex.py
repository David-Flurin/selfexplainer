import numpy as np
from statistics import mean
import latextable
from texttable import Texttable
import sys
import json

results_file = sys.argv[1]

try:
    checkpoint_dict = json.loads(sys.argv[2])
except:
    checkpoint_dict = None


mode = 'micro'

with open(results_file[:-3]+".txt", 'w') as table_file:
    table = Texttable()
    column_align = ['l']


    with np.load(results_file, allow_pickle=True) as results_file:
        results = results_file["results"].item()
        metric_names = list(next(iter(results.values())).keys())
        if 'classification_metrics' in metric_names:
            i = metric_names.index('classification_metrics')
            metric_names = metric_names[:i] + ['f1_class', 'prec', 'rec'] + metric_names[i+1:]
        print(metric_names)
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
            metric_list = results[model].values()
            for metric in results[model]:
                if metric == 'classification_metrics':
                    for c_metric in results[model][metric]:
                        metrics.append(results[model][metric][c_metric][mode])
                else:
                    try:
                        metrics.append(mean(results[model][metric]))
                    except:
                        metrics.append(mean([mean(values) for values in results[model][metric]]))
            rows.append(metrics)
        print(rows)
        
        # Calculate best entry per metric and boldface it
        best_metrics = dict.fromkeys(metric_names,(-1000, -1))
        best_metrics['sal'] = (1000, -1)
        for i, row in enumerate(rows[1:], 1):
            for j, m in enumerate(row[1:], 1):
                if rows[0][j] == 'sal':
                    if m < best_metrics['sal'][0]:
                        best_metrics['sal'] = (m,i)
                else:
                    if m > best_metrics[rows[0][j]][0]:
                        best_metrics[rows[0][j]] = (m,i)
        for metric, (best_m, best_idx) in best_metrics.items():
            rows[best_idx][rows[0].index(metric)] = f'\\bf{{{float(rows[best_idx][rows[0].index(metric)]):.3f}}}'
        table.add_rows(rows)
latex_table = latextable.draw_latex(table, caption='Different models on Toy dataset')

latex_table = latex_table.replace('_', '\_')
print(latex_table)
