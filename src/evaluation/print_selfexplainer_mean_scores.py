import numpy as np
from statistics import mean
import sys

filename = sys.argv[1]

with np.load(filename, allow_pickle=True) as file:
    results = file["results"].item()

    for model in results:
        print("\t" + model + "\n")
        for metric in results[model]:
            if metric != 'classification_metrics':
                values = [x for x in results[model][metric] if x]
                try:
                    print("\t\t\t{}: {}\n".format(metric, mean(values)))
                except:
                    values = [mean(value) for value in values]
                    print("\t\t\t{}: {}\n".format(metric, mean(values)))

            else:
                for single_metric in results[model][metric]:
                    print("\t\t\t{}: {}\n".format(single_metric,
                          results[model][metric][single_metric]))
