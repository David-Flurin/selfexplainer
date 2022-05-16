from evaluation.plot import plot_losses

import torch
import os
import sys
import shutil
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler
from pathlib import Path
import pickle
import hashlib

from data.dataloader import ColorDataModule, ToyDataModule, VOCDataModule, COCODataModule, CUB200DataModule, ToyData_Saved_Module
from utils.argparser import get_parser, write_config_file
from models.selfexplainer import SelfExplainer
from models.classifier import Classifier
from models.fcn_16 import FCN16

from models.mlp import MLP
from utils.image_display import save_all_class_masked_images, save_masked_image

from toy_dataset import generator


main_dir = Path(os.path.dirname(os.path.abspath(__file__)))

parser = get_parser()
args = parser.parse_args()
print(args)
if args.arg_log:
    write_config_file(args)

#pl.seed_everything(args.seed)
profiler = AdvancedProfiler(dirpath=main_dir, filename='performance_report')


# Set up Logging
if args.use_tensorboard_logger:
    log_dir = Path(args.save_path, "tb_logs")
    logger = pl.loggers.TensorBoardLogger(log_dir, name="Selfexplainer")
else:
    logger = False

# Set up data module
if args.dataset == "VOC":
    data_path = main_dir / args.data_base_path / 'VOC2007'
    data_module = VOCDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 20
elif args.dataset == "SMALLVOC":
    data_path = main_dir / args.data_base_path / 'VOC2007_small'
    data_module = VOCDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 3
elif args.dataset == "COCO":
    data_path = main_dir / args.data_base_path / 'COCO2014'
    data_module = COCODataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 91
elif args.dataset == "CUB":
    data_path = main_dir / args.data_base_path / 'CUB200'
    data_module = CUB200DataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 200
elif args.dataset == "TOY":
    data_module = ToyDataModule(
        epoch_length=args.epoch_length, test_samples=args.test_samples, segmentation=(args.toy_segmentations), 
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, test_batch_size=args.test_batch_size
    )
    num_classes = 8 if args.model_to_train == 'fcn' else 8

elif args.dataset == "TOY_SAVED":
    data_path = main_dir / args.data_base_path / 'TOY'
    data_module = ToyData_Saved_Module(
        data_path=data_path, segmentation=(args.toy_segmentations), 
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, test_batch_size=args.test_batch_size
    )
    num_classes = 8 
elif args.dataset == "COLOR":
    data_module = ColorDataModule(
        epoch_length=args.epoch_length, test_samples=args.test_samples, segmentation=(args.toy_segmentations), 
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, test_batch_size=args.test_batch_size, rgb=args.rgb
    )
    num_classes = 2
else:
    raise Exception("Unknown dataset " + args.dataset)

# Create results folder
if args.save_path:
    new_version = 0
    if os.path.exists(args.save_path) and os.path.isdir(args.save_path):
        subdirs = os.listdir(args.save_path)
        versions = []
        for dirs in subdirs:
            d = dirs.split('_')
            if d[0] == 'version':
                versions.append(int(d[-1]))
        if len(versions) > 0:
            new_version = max(versions) + 1
    args.save_path += f'/version_{new_version}'
    os.makedirs(args.save_path)
    if len(sys.argv) > 1:
        shutil.copy(sys.argv[2], args.save_path)


# Set up model

if args.model_to_train == "selfexplainer":
    model = SelfExplainer(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, profiler=profiler, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, class_loss=args.class_loss, frozen=args.frozen, freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, save_all_class_masks=args.save_all_class_masks, objective=args.objective, background_loss=args.background_loss
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, profiler=profiler, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, class_loss=args.class_loss, frozen=args.frozen, freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, save_all_class_masks=args.save_all_class_masks, objective=args.objective, background_loss=args.background_loss
        )

elif args.model_to_train == "fcn":
    model = FCN16(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, profiler=profiler, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, frozen=args.frozen, freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, background_loss=args.background_loss, save_all_class_masks=args.save_all_class_masks
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, profiler=profiler, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, frozen=args.frozen, freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, background_loss=args.background_loss, save_all_class_masks=args.save_all_class_masks
        )
elif args.model_to_train == "mlp":
    model = MLP(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, profiler=profiler, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, objective=args.objective, class_loss = args.class_loss, rgb=args.rgb, frozen=args.frozen, freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, target_threshold=args.target_threshold, 
         non_target_threshold=args.non_target_threshold, background_loss=args.background_loss, save_all_class_masks=args.save_all_class_masks
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, profiler=profiler, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, objective=args.objective, class_loss=args.class_loss, rgb=args.rgb, frozen=args.frozen, freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, target_threshold=args.target_threshold, 
         non_target_threshold=args.non_target_threshold, background_loss=args.background_loss, save_all_class_masks=args.save_all_class_masks
        )
elif args.model_to_train == "classifier":
    model = Classifier(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, save_path=args.save_path, gpu=args.gpu, profiler=profiler
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, save_path=args.save_path
        )
else:
    raise Exception("Unknown model type: " + args.model_to_train)


print('Use variation loss:', model.use_mask_variation_loss)
print('Use area loss:', model.use_mask_area_loss)

# Define Early Stopping condition
early_stop_callback = EarlyStopping(
    monitor="loss" if args.dataset in ['TOY', 'COLOR'] else "val_loss",
    min_delta=args.early_stop_min_delta,
    patience=args.early_stop_patience,
    verbose=False,
    mode="min",
    #stopping_threshold=0.
)

#profiler = AdvancedProfiler(dirpath=main_dir, filename='performance_report')
trainer = pl.Trainer(
    logger = logger,
    callbacks = [early_stop_callback],
    gpus = [args.gpu] if torch.cuda.is_available() else 0,
    #detect_anomaly = True,
    log_every_n_steps = 5,
    enable_checkpointing = args.checkpoint_callback
    #amp_backend='apex',
    #amp_level='02'
    #profiler=profiler
)

if args.train_model:
    trainer.fit(model=model, datamodule=data_module)
    if logger:
        plot_dir = args.save_path + '/plots'
        os.makedirs(plot_dir)
        plot_losses(logger.log_dir, ['classification_loss', 'background_entropy_loss', 'similarity_loss', 'mask_area_loss', 'mask_loss', 'inv_mask_loss', 'bg_logits_loss', 'loss'], plot_dir)
        if args.dataset not in ['TOY', 'COLOR', 'TOY_SAVED']:
            plot_losses(logger.log_dir, ['val_classification_loss', 'val_background_entropy_loss', 'val_similarity_loss', 'val_mask_area_loss', 'val_mask_loss', 'val_inv_mask_loss', 'val_bg_logits_loss', 'val_loss')
    trainer.test(model=model, datamodule=data_module)
else:
    trainer.test(model=model, datamodule=data_module)


