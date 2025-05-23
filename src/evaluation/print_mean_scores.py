import numpy as np
from statistics import mean
import sys

filename = sys.argv[1]


with np.load(filename, allow_pickle=True) as file:
    results = file["results"].item()
    for dataset in results:
        print("\n" + dataset + "\n")
        for classifier in results[dataset]:
            print("\t" + classifier + "\n")
            for method in results[dataset][classifier]:
                print("\t\t" + method + "\n")
                for metric in results[dataset][classifier][method]:
                    values = results[dataset][classifier][method][metric]
                    values = [v for v in values if v is not None]
                    try:
                        print("\t\t\t{}: {}\n".format(metric, mean(values)))
                    except:
                        values = [mean(value) for value in values]
                        print("\t\t\t{}: {}\n".format(metric, mean(values)))
