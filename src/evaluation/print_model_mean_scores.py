import numpy as np
from statistics import mean
import sys


with np.load(sys.argv[1], allow_pickle=True) as file:
        results = file["results"].item()
        for model in results:
                print("\t" + model + "\n")
                for metric in results[model]:
                        if metric != 'classification_metrics':
                            values = results[model][metric]
                            try:
                                    print("\t\t\t{}: {}\n".format(metric, mean(values)))
                            except:
                                    values = [mean(value) for value in values]
                                    print("\t\t\t{}: {}\n".format(metric, mean(values)))

                        else:
                            for single_metric in results[model][metric]:
                                print("\t\t\t{}: {}\n".format(single_metric, results[model][metric][single_metric]))



                

