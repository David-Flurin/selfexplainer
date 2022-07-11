import numpy as np
from statistics import mean

with np.load("results_toy_singlelabel.npz", allow_pickle=True) as file:
	results = file["results"].item()
	for model in results:
		print("\t" + model + "\n")
		for metric in results[model]:
			values = results[model][metric]
			try:
				print("\t\t\t{}: {}\n".format(metric, mean(values)))
			except:
				values = [mean(value) for value in values]
				print("\t\t\t{}: {}\n".format(metric, mean(values)))



		
