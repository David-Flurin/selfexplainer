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
    parser.add_argument('--epoch_length', default=1000, type=int, help='Number of training samples per epoch when using Toy dataset.')
    parser.add_argument('--test_samples', default=100, type=int, help='Number of test samples when using Toy dataset.')
    parser.add_argument('--rgb', default=False, type=str2bool, help='Whether the color dataset generates grayscale or color images')

    # Data processing parameters
    parser.add_argument('--train_batch_size', default=16, type=int, help='batch size used for training')
    parser.add_argument('--val_batch_size', default=16, type=int, help='batch size used for validation')
    parser.add_argument('--test_batch_size', default=1, type=int, help='batch size used for testing')
    parser.add_argument('--use_data_augmentation', default=False, type=str2bool, help='set to true to enable data augmentation on training images')

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
    # parser.add_argument('--fix_classifier_backbone', default=True, type=str2bool, help='Whether to fix the wait for the classifiers backbone')
    # parser.add_argument('--fix_classifier', default=True, type=str2bool, help='If True, classifier  is frozen. Strongly recommended for Explainer training.')
    parser.add_argument('--model_to_train', default='selfexplainer', type=str, help='which model architecture should be used for training or testing')
    # parser.add_argument('--classifier_type', choices=['vgg16', 'resnet50'], default='vgg16', type=str, help='type of classifier architecture to use')
    # parser.add_argument('--explainer_classifier_checkpoint', default=None, type=str, help='Path to the .ckpt file that contains the weights of a pretrained explainer. Also contains the weights for the associated classifier.')
    # parser.add_argument('--classifier_checkpoint', default=None, type=str, help='Path to the .ckpt file that contains the weights of a pretrained classifier.')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to the .ckpt file that contains the weights of a pretrained self-explainer.')
    parser.add_argument('--frozen', default=False, type=str2bool, help='If the object and background pass of selfexplainer models has frozen weights')
    parser.add_argument('--freeze_every', default=20, type=int, help='Every n iterations, the model is frozen for the object and background pass.')


    # Model-specific parameters
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate used by the Adam optimizer')
    parser.add_argument('--use_similarity_loss', default=False, type=str2bool, help='whether to use similaity loss between image and obejct pass.')
    parser.add_argument('--similarity_regularizer', default=1.0, type=float, help='loss weighting term for similarity loss')
    parser.add_argument('--use_entropy_loss', default=False, type=str2bool, help='whether to use entropy loss on background logits.')
    parser.add_argument('--use_weighted_loss', default=False, type=str2bool, help='whether to use a dynamically weighted loss.')
    parser.add_argument('--use_mask_variation_loss', default=True, type=str2bool, help='whether to use variation loss on the mask.')
    parser.add_argument('--use_mask_area_loss', default=True, type=str2bool, help='whether to use area loss on the mask.')
    # parser.add_argument('--use_mask_coherency_loss', default=True, type=str2bool, help='whether to use mask coherency loss (only for self-explainer architecture)')
    # parser.add_argument('--entropy_regularizer', default=1.0, type=float, help='loss weighting term for entropy loss')
    parser.add_argument('--mask_variation_regularizer', default=1.0, type=float, help='loss weighting term for mask variation loss')
    parser.add_argument('--mask_area_constraint_regularizer', default=1.0, type=float, help='loss weighting term for overall mask area constraint (currently not used!)')
    parser.add_argument('--mask_total_area_regularizer', default=0.1, type=float, help='loss weighting term for the total area loss')
    parser.add_argument('--ncmask_total_area_regularizer', default=0.3, type=float, help='loss weighting term for the area constraints for the individual class segmentation masks')
    parser.add_argument('--toy_segmentations', default=True, type=str2bool, help='Whether the toy dataset loader should provide segmentation masks.')
    parser.add_argument('--use_perfect_mask', default=False, type=str2bool, help='DEBUG: Whether the groundtruth mask is provided for the second and third pass')
    parser.add_argument('--count_logits', default=False, type=str2bool, help='DEBUG: Whether to generate statistics for logits.')
    parser.add_argument('--objective', default='classification', type=str, help='Classification or segmentation loss')
    parser.add_argument('--class_loss', default='bce', type=str, help='Which classification loss to use [bce, ce]')


    # parser.add_argument('--target_mask_min_area', default=0.05, type=float, help='minimum area for the overall mask area constraint (currently not used!)')
    # parser.add_argument('--target_mask_max_area', default=0.5, type=float, help='maximum area for the overall mask area constraint (currently not used!)')
    # parser.add_argument('--class_mask_min_area', default=0.05, type=float, help='minimum area for the area constraints for the individual class segmentation masks')
    # parser.add_argument('--class_mask_max_area', default=0.3, type=float, help='maximum area for the area constraints for the individual class segmentation masks')

    # Image display parameters
    # parser.add_argument('--show_images', default=False, type=str2bool, help='If true, displays images and corresponding masked images during testing. Requires testing batch size to be 1.')
    # parser.add_argument('--show_all_class_masks', default=False, type=str2bool, help='If true, displays individual class masks during testing. Requires VOC dataset. Requires testing batch size to be 1.')
    # parser.add_argument('--show_max_activation_for_class_id', default=None, type=int, help='If true, highlights point of maximum activation for given class id. Requires testing batch size to be 1.')
    parser.add_argument('--save_masks', default=False, type=str2bool, help='If true, masks are saved to location specified by save_path (see below)')
    parser.add_argument('--save_masked_images', default=False, type=str2bool, help='If true, masked images are saved to location specified by save_path (see below)')
    # parser.add_argument('--save_all_class_masks', default=False, type=str2bool, help='Unused.')
    parser.add_argument('--save_path', default='./results/', type=str, help='Path to where masks and/or masked images are saved if corresponding options are set to true.')

    # Metrics parameters
    parser.add_argument('--metrics_threshold', default=-1.0, type=float, help='Threshold for logit to count as positive vs. negative prediction. Use -1.0 for Explainer and 0.0 for classifier.')

    return parser

def write_config_file(args, path='config.cfg'):
    with open(path, 'w') as f:
        for k in sorted(args.__dict__):
            if args.__dict__[k] != None:
                print(k, '=', args.__dict__[k], file=f)
