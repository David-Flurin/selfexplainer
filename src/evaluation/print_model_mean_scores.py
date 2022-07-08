import numpy as np
from statistics import mean

with np.load("results.npz", allow_pickle=True) as file:
	results = file["results"].item()
	for dataset in results:
		print("\n" + dataset + "\n")
		for model in results[dataset]:
			print("\t" + model + "\n")
			for metric in results[dataset][model]:
				values = results[dataset][model][metric]
				try:
					print("\t\t\t{}: {}\n".format(metric, mean(values)))
				except:
					values = [mean(value) for value in values]
					print("\t\t\t{}: {}\n".format(metric, mean(values)))



		
