# Self-Explainer

This repository contains the code to train the Self-Explainer and conduct the experiments from the report. Most of the code was directly taken from [NN-Explainer](https://github.com/stevenstalder/NN-Explainer) and was then adapted to fit the objective.

## Directories & Files
- main.py: Called to start a training
- plot.py: Some plotting functions used for the report
- models/: Contains Self-Explainer and Resnet50 modes
- data/: Defined dataloaders and datasets for the different datasets the Self-Explainer was tested on
- evaluation/: Different scripts used for the evaluation of the Self-Explainer and baselines
- utils/: Different helper functions
- synthetic_dataset/: Contains code to generate samples of the synthetic dataset.
- color_dataset/: Pixel dataset used in the early stage of the thesis to verify certain loss properties. 

## Run trainings

Run a training by executing main.py with a configuration file like:
```
python main.py -c config_files/configuration.cfg
```
To reproduce the results from the report, use the configuration files in the *config_files/* directory for the respective dataset. To change the synthetic dataset from single-label to multi-label mode, set the configuration field *multilabel=True*.

## Evaluate model
To evaluate a model, use the file *evaluate_selfexplainer.py* in the *evaluation/* directory. Change the settings defined in the file to fit your environment. The attribution masks are first generated and then evaluated with a bunch of metrics (see file *compute_scores.py*). To print averaged metrics, run
```
python print_selfexplainer_mean_scores.py path_to_results_file
```

With the script *evaluate_selfexplainer.py*, you can also generate class-specific masks.

## Synthetic dataset

To generate a new instance of the synthetic dataset, you have to run following code:
```
from synthetic_dataset.generator import Generator

g = Generator()
g.create_set(save_path, number_per_shape, proportions, multilabel)
```

The arguments of the method *create_set()* are:
- save\_path: Path to location where the new dataset is saved.
- number\_per\_shape: How many samples per shape are generated. E.g. number_per_shape=200 leads to 8\*200 = 1600 samples in total. 
- proportions: List of floats corresponding to the proportions of training, validation and test set. E.g. if we have a dataset with 10000 samples and proportions=[0.5, 0.2, 0.3], then the training set has 5000 samples, the validation set 2000 samples and the test set 3000 samples.
- multilabel: Whether the dataset also has samples with 2 target classes.

