from plot import plot_losses

import torch
import os
import sys
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from pathlib import Path

from data.dataloader import ColorDataModule, OIDataModule, ToyDataModule, VOCDataModule, ToyData_Saved_Module, VOC2012DataModule
from models.resnet50 import Resnet50
from utils.argparser import get_parser, write_config_file
from models.selfexplainer import SelfExplainer
from plot import plot_metrics_from_file

main_dir = Path(os.path.dirname(os.path.abspath(__file__)))

parser = get_parser()
args = parser.parse_args()
print(args)
if args.arg_log:
    write_config_file(args)

print('Dir: ',args.save_path)

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
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation, weighted_sampling=args.weighted_sampling
    )
    num_classes = 20
elif args.dataset == "VOC2012":
    data_path = main_dir / args.data_base_path / 'VOC2012'
    data_module = VOC2012DataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation, weighted_sampling=args.weighted_sampling
    )
    num_classes = 20
elif args.dataset == "OI_SMALL":
    data_path = main_dir / args.data_base_path / 'OI_SMALL'
    data_module = OIDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, 
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation, weighted_sampling=args.weighted_sampling
    )
    num_classes = 3
elif args.dataset == "OI":
    data_path = main_dir / args.data_base_path / 'OI'
    data_module = OIDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation, weighted_sampling=args.weighted_sampling
    )
    num_classes = 13
elif args.dataset == "OI_LARGE":
    data_path = main_dir / args.data_base_path / 'OI_LARGE'
    data_module = OIDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation, weighted_sampling=args.weighted_sampling
    )
    num_classes = 20
elif args.dataset == "SMALLVOC":
    data_path = main_dir / args.data_base_path / 'VOC2007_small'
    data_module = VOCDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 3
elif args.dataset == "TOY":
    data_module = ToyDataModule(
        epoch_length=args.epoch_length, test_samples=args.test_samples, segmentation=(args.toy_segmentations), multilabel=args.multilabel, 
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, test_batch_size=args.test_batch_size
    )
    num_classes = 8
elif args.dataset == "TOY_SAVED":
    data_path = main_dir / args.data_base_path / 'TOY'
    data_module = ToyData_Saved_Module(
        data_path=data_path, segmentation=(args.toy_segmentations), 
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, test_batch_size=args.test_batch_size
    )
    num_classes = 8 
elif args.dataset == "COLOR":
    data_module = ColorDataModule(
        epoch_length=args.epoch_length, test_samples=args.test_samples, segmentation=(args.toy_segmentations), multilabel=args.multilabel,
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, test_batch_size=args.test_batch_size, rgb=args.rgb
    )
    num_classes = 3
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
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, mask_variation_regularizer=args.mask_variation_regularizer,save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, class_loss=args.class_loss, frozen=args.frozen, 
         weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multilabel=args.multilabel, use_bounding_loss=args.use_bounding_loss, similarity_loss_mode=args.similarity_loss_mode, class_mask_max_area=args.class_mask_max_area, class_mask_min_area=args.class_mask_min_area, weighted_sampling=args.weighted_sampling, background_loss_scheduling=args.background_loss_scheduling, similarity_loss_scheduling=args.similarity_loss_scheduling, mask_loss_scheduling=args.mask_loss_scheduling, use_loss_scheduling=args.use_loss_scheduling,
         freeze_every=args.freeze_every, save_all_class_masks=args.save_all_class_masks, background_loss=args.background_loss, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, use_mask_logit_loss=args.use_mask_logit_loss, mask_logit_loss_regularizer=args.mask_logit_loss_regularizer, object_loss_weighting_params=args.object_loss_weighting_params, mask_loss_weighting_params=args.mask_loss_weighting_params
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, mask_variation_regularizer=args.mask_variation_regularizer,save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, class_loss=args.class_loss, frozen=args.frozen, 
         weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multilabel=args.multilabel, use_bounding_loss=args.use_bounding_loss, 
         similarity_loss_mode=args.similarity_loss_mode, weighted_sampling=args.weighted_sampling, background_loss_scheduling=args.background_loss_scheduling, similarity_loss_scheduling=args.similarity_loss_scheduling, mask_loss_scheduling=args.mask_loss_scheduling, use_loss_scheduling=args.use_loss_scheduling,
         freeze_every=args.freeze_every, save_all_class_masks=args.save_all_class_masks, background_loss=args.background_loss, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, use_mask_logit_loss=args.use_mask_logit_loss, mask_logit_loss_regularizer=args.mask_logit_loss_regularizer, object_loss_weighting_params=args.object_loss_weighting_params, mask_loss_weighting_params=args.mask_loss_weighting_params
        )


elif args.model_to_train == "resnet50":
    model = Resnet50(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, 
       gpu=args.gpu, metrics_threshold=args.metrics_threshold, multilabel=args.multilabel, weighted_sampling=args.weighted_sampling, use_imagenet_pretraining=args.use_imagenet_pretraining, fix_classifier_backbone=args.fix_classifier_backbone
    )

elif args.model_to_train == 'simple':
    model = Simple_Model(
        num_classes=num_classes, pretrained=False, aux_classifier=args.aux_classifier, learning_rate=args.learning_rate, dataset = args.dataset, multilabel=args.multilabel
    )
elif args.model_to_train == "resnet50_steven":
    model = Resnet50ClassifierModel(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, use_imagenet_pretraining=args.use_imagenet_pretraining, 
        fix_classifier_backbone=args.fix_classifier_backbone, metrics_threshold=args.metrics_threshold, multilabel=args.multilabel
    )
elif args.model_to_train == "resnet50_steven_original":
    model = Resnet50ClassifierModel(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, use_imagenet_pretraining=args.use_imagenet_pretraining, 
        fix_classifier_backbone=args.fix_classifier_backbone
    )
else:
    raise Exception("Unknown model type: " + args.model_to_train)


# Define Early Stopping condition
early_stop_callback = EarlyStopping(
    #monitor="loss" if args.dataset in ['TOY', 'COLOR'] else "val_loss",
    monitor='loss' if args.dataset == 'TOY' else 'val_loss',
    min_delta=args.early_stop_min_delta,
    patience=args.early_stop_patience,
    verbose=False,
    mode="min",
    #stopping_threshold=0.
)

checkpoint_callback = ModelCheckpoint(
    every_n_train_steps = 100
)

k_checkpoint_callback = ModelCheckpoint(
    monitor='loss' if args.dataset == 'TOY' else 'val_loss',
    save_top_k=10
)

profiler = AdvancedProfiler(dirpath=main_dir, filename='selfexplainer_model_report')
if args.dataset in ['OI_SMALL', 'OI', 'OI_LARGE']:
    trainer = pl.Trainer(
        logger = logger,
        callbacks = [early_stop_callback, k_checkpoint_callback],
        gpus = [args.gpu] if torch.cuda.is_available() else 0,
        #detect_anomaly = True,
        #log_every_n_steps = 80//args.train_batch_size,
        log_every_n_steps = 5,
        val_check_interval = 100,
        limit_val_batches = 50,
        enable_checkpointing = args.checkpoint_callback
        #amp_backend='apex',
        #amp_level='02'
        #profiler=profiler
    )
else:
    trainer = pl.Trainer(
        logger = logger,
        callbacks = [early_stop_callback, k_checkpoint_callback],
        gpus = [args.gpu] if torch.cuda.is_available() else 0,
        #detect_anomaly = True,
        #log_every_n_steps = 80//args.train_batch_size,
        log_every_n_steps = 5,
        # val_check_interval = 200,
        # limit_val_batches = 100,
        enable_checkpointing = args.checkpoint_callback,
        #amp_backend='apex',
        #amp_level='02'
        #profiler=profiler
    )


if args.train_model:
    trainer.fit(model=model, datamodule=data_module)
    if logger:
        plot_dir = args.save_path + '/plots'
        os.makedirs(plot_dir)
        plot_losses(logger.log_dir, ['classification_loss_1Pass', 'background_loss', 'similarity_loss', 'mask_area_loss', 'mask_loss', 'inv_mask_loss', 'bg_logits_loss'], plot_dir+'/train_losses.png')
        plot_losses(logger.log_dir, ['classification_loss_1Pass', 'classification_loss_2Pass'], plot_dir+'/classification_losses.png')
        plot_losses(logger.log_dir, ['classification_loss_1Pass', 'classification_loss_2Pass', 'similarity_loss'], plot_dir+'/classification_similarity_losses.png')


        if args.dataset not in ['TOY', 'COLOR', 'TOY_SAVED']:
           plot_losses(logger.log_dir, ['val_classification_loss', 'val_background_loss', 'val_similarity_loss', 'val_mask_area_loss', 'val_mask_loss', 'val_inv_mask_loss', 'val_bg_logits_loss'], plot_dir+'/val_losses.png')
    trainer.test(model=model, datamodule=data_module)
    #if logger:
        #if args.dataset not in ['TOY', 'COLOR', 'TOY_SAVED']:
        #    plot_losses(logger.log_dir, ['val_classification_loss', 'val_background_entropy_loss', 'val_similarity_loss', 'val_mask_area_loss', 'val_mask_loss', 'val_inv_mask_loss', 'val_bg_logits_loss', 'val_loss'], plot_dir+'/val_losses.png')
else:
    trainer.test(model=model, datamodule=data_module)


