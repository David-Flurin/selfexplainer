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

# Evaluate model
To evaluate a model, use the file *evaluate_selfexplainer.py* in the *evaluation/* directory. Change the settings defined in the file to fit your environment. The attribution masks are first generated and then evaluated with a bunch of metrics (see file *compute_scores.py*). To print averaged metrics, run
```
python print_selfexplainer_mean_scores.py path_to_results_file
```
