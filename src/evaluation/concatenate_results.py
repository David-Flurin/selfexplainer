import numpy as np
import sys
import os

dir = sys.argv[1]
output_file = sys.argv[2]
all_results = {}
for file in os.listdir(dir):
    #filename = os.fsdecode(file)

    with np.load(os.path.join(dir, file), allow_pickle=True) as r_file:
        results = r_file["results"].item()
        for result in results:
            all_results[result] = results[result]

np.savez(os.path.join(dir, output_file), results=all_results)