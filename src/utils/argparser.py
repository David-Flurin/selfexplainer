import argparse
from unittest import defaultTestLoader
from configargparse import ArgumentParser

# This file contains the declaration of our argument parser

# Needed to parse booleans from command line properly
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = ArgumentParser(description='Selfexplainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--arg_log', default=False, type=str2bool, help='save arguments to config file')

    # Dataset parameters
    parser.add_argument('--dataset', default='VOC', type=str, help='which dataset to use')
    parser.add_argument('--data_base_path', default='../datasets/', type=str, help='Base bath of the datasets. Should contain subdirectories with the different datasets.')
    parser.add_argument('--epoch_length', default=1000, type=int, help='Number of training samples per epoch when using synthetic dataset.')
    parser.add_argument('--test_samples', default=100, type=int, help='Number of test samples when using synthetic dataset.')
    parser.add_argument('--rgb', default=True, type=str2bool, help='Whether the color dataset generates grayscale or color images')
    parser.add_argument('--multilabel', default=False, type=str2bool, help='Whether the objective is multilabel or singlelabel.')
    parser.add_argument('--synthetic_segmentations', default=False, type=str2bool, help='Whether the synthetic dataset loader should provide segmentation masks.')


    # Data processing parameters
    parser.add_argument('--train_batch_size', default=16, type=int, help='batch size used for training')
    parser.add_argument('--val_batch_size', default=16, type=int, help='batch size used for validation')
    parser.add_argument('--test_batch_size', default=1, type=int, help='batch size used for testing')
    parser.add_argument('--use_data_augmentation', default=False, type=str2bool, help='set to true to enable data augmentation on training images')
    parser.add_argument('--weighted_sampling', default=False, type=str2bool, help='Use a WeightedRandomSampler to counteract class imbalance of dataset')


    # Trainer Parameters
    parser.add_argument('--seed', default=42, type=int, help='seed for all random number generators in pytorch, numpy, and python.random')
    parser.add_argument('--use_tensorboard_logger', default=False, type=str2bool, help='whether to use tensorboard')
    parser.add_argument('--checkpoint_callback', default=True, type=str2bool, help='if true, trained model will be automatically saved')
    parser.add_argument('--gpu', default=0, type=int, help='Number of the GPU to be used')

    # Early stopping Parameters
    parser.add_argument('--early_stop_min_delta', default=0.001, type=float, help='threshold for early stopping condition')
    parser.add_argument('--early_stop_patience', default=5, type=int, help='patience for early stopping to trigger')

    # General Model Parameters
    parser.add_argument('--train_model', default=True, type=str2bool, help='If True, specified model will be trained. If False, model will be tested.')
    parser.add_argument('--use_imagenet_pretraining', default=False, type=str2bool, help='If True, classifiers use a pretrained backbone from ImageNet pretraining')
    parser.add_argument('--fix_classifier_backbone', default=True, type=str2bool, help='Whether to fix the pretrained classifier backbone of ResNet50 during training.')
    parser.add_argument('--model_to_train', default='selfexplainer', type=str, help='which model architecture should be used for training or testing')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to the .ckpt file that contains the weights of a pretrained self-explainer.')
    parser.add_argument('--frozen', default=False, type=str2bool, help='If the object and background pass of selfexplainer models has frozen weights')
    parser.add_argument('--weighting_koeff', default=1.0, type=float, help='Koefficient of the softmax weighting scheme to generate logits.')
    parser.add_argument('--aux_classifier', default=False, type=str2bool, help='Use an auxiliary classifier head.')



    # Model-specific parameters
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate used by the Adam optimizer')
    parser.add_argument('--use_similarity_loss', default=True, type=str2bool, help='whether to use similaity loss between image and obejct pass.')
    parser.add_argument('--similarity_regularizer', default=0.5, type=float, help='loss weighting term for similarity loss')
    parser.add_argument('--similarity_loss_mode', default='rel', type=str, help='If similarity loss should be calculated relative to first pass or absolute to target labels (rel/abs)')
    parser.add_argument('--use_background_loss', default=True, type=str2bool, help='whether to use entropy loss on background logits.')
    parser.add_argument('--bg_loss_regularizer', default=1.0, type=float, help='Weight of background loss.')   
    parser.add_argument('--background_loss', default='entropy', type=str, help='Which background loss to use')    
    parser.add_argument('--use_weighted_loss', default=True, type=str2bool, help='whether to use a dynamically weighted loss.')
    parser.add_argument('--use_mask_variation_loss', default=False, type=str2bool, help='whether to use variation loss on the mask.')
    parser.add_argument('--use_mask_area_loss', default=True, type=str2bool, help='whether to use area loss on the mask.')
    parser.add_argument('--use_bounding_loss', default=True, type=str2bool, help='whether to use bounding loss on the mask.')
    parser.add_argument('--mask_variation_regularizer', default=1.0, type=float, help='loss weighting term for mask variation loss')
    parser.add_argument('--mask_area_constraint_regularizer', default=1.0, type=float, help='loss weighting term for overall mask area constraint (currently not used!)')
    parser.add_argument('--mask_total_area_regularizer', default=0.1, type=float, help='loss weighting term for the total area loss')
    parser.add_argument('--ncmask_total_area_regularizer', default=0.3, type=float, help='loss weighting term for the area constraints for the individual class segmentation masks')
    parser.add_argument('--use_loss_scheduling', default='False', type=str2bool, help='Use loss scheduling or not')
    parser.add_argument('--similarity_loss_scheduling', default='500', type=float, help='After how many iterations similarity loss is used.')
    parser.add_argument('--background_loss_scheduling', default='500', type=float, help='After how many iterations background entropy loss is used.')
    parser.add_argument('--mask_loss_scheduling', default='1000', type=float, help='After how many iterations mask loss is used.')
    parser.add_argument('--use_mask_logit_loss', default='False', type=str2bool, help='Mask Logit loss')
    parser.add_argument('--mask_logit_loss_regularizer', default='1.0', type=float, help='Mask Logit loss regularizer')
    parser.add_argument('--object_loss_weighting_params', default=[2, 0.2], type=float, nargs=2, help='Hyperparameters for object and background loss weighting')
    parser.add_argument('--mask_loss_weighting_params', default=[5, 0.1], type=float, nargs=2, help='Hyperparameters for mask loss weighting')

    parser.add_argument('--class_mask_min_area', default=0.05, type=float, help='minimum area for the area constraints for the individual class segmentation masks')
    parser.add_argument('--class_mask_max_area', default=0.3, type=float, help='maximum area for the area constraints for the individual class segmentation masks')

    # Image save parameters
    parser.add_argument('--save_masks', default=False, type=str2bool, help='If true, masks are saved to location specified by save_path (see below)')
    parser.add_argument('--save_masked_images', default=False, type=str2bool, help='If true, masked images are saved to location specified by save_path (see below)')
    parser.add_argument('--save_all_class_masks', default=False, type=str2bool, help='Unused.')
    parser.add_argument('--save_path', default='./results/', type=str, help='Path to where masks and/or masked images are saved if corresponding options are set to true.')

    # Metrics parameters
    parser.add_argument('--metrics_threshold', default=0.5, type=float, help='Threshold for logit to count as positive vs. negative prediction. Use -1.0 for Explainer and 0.0 for classifier.')

    return parser

def write_config_file(args, path='config.cfg'):
    with open(path, 'w') as f:
        for k in sorted(args.__dict__):
            if args.__dict__[k] != None:
                print(k, '=', args.__dict__[k], file=f)
