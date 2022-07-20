from plot import plot_losses

import torch
import os
import sys
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from pathlib import Path

from data.dataloader import ColorDataModule, MNISTDataModule, OIDataModule, OISmallDataModule, ToyDataModule, VOCDataModule, COCODataModule, CUB200DataModule, ToyData_Saved_Module, VOC2012DataModule
from models.resnet50 import Resnet50
from utils.argparser import get_parser, write_config_file
from models.selfexplainer import SelfExplainer, SelfExplainer_Slike
from models.simple_model import Simple_Model
from models.explainer_classifier import ExplainerClassifierModel
from plot import plot_metrics_from_file

from models.mlp import MLP
from utils.image_display import save_all_class_masked_images, save_masked_image

from toy_dataset import generator

# generator = generator.Generator('/home/david/Documents/Master/Thesis/selfexplainer/src/toy_dataset/foreground.txt', '/home/david/Documents/Master/Thesis/selfexplainer/src/toy_dataset/background.txt')
# generator.create(500, [0, 0, 1], multiclass=True)
# quit()
main_dir = Path(os.path.dirname(os.path.abspath(__file__)))

parser = get_parser()
args = parser.parse_args()
print(args)
if args.arg_log:
    write_config_file(args)

#pl.seed_everything(args.seed)
#profiler = AdvancedProfiler(dirpath=main_dir, filename='performance_report')

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
elif args.dataset == "OISMALL":
    data_path = main_dir / args.data_base_path / 'OI_SMALL'
    data_module = OISmallDataModule(
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
elif args.dataset == "MNIST":
    data_path = main_dir / args.data_base_path / 'MNIST'
    data_module = MNISTDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 10
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
        epoch_length=args.epoch_length, test_samples=args.test_samples, segmentation=(args.toy_segmentations), multiclass=args.multiclass, 
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
        epoch_length=args.epoch_length, test_samples=args.test_samples, segmentation=(args.toy_segmentations), multiclass=args.multiclass,
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
         save_masks=args.save_masks, gpu=args.gpu, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, class_loss=args.class_loss, frozen=args.frozen, 
         weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multiclass=args.multiclass, use_bounding_loss=args.use_bounding_loss, similarity_loss_mode=args.similarity_loss_mode, class_mask_max_area=args.class_mask_max_area, class_mask_min_area=args.class_mask_min_area, weighted_sampling=args.weighted_sampling, background_loss_scheduling=args.background_loss_scheduling, similarity_loss_scheduling=args.similarity_loss_scheduling, mask_loss_scheduling=args.mask_loss_scheduling, use_loss_scheduling=args.use_loss_scheduling,
         freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, save_all_class_masks=args.save_all_class_masks, objective=args.objective, background_loss=args.background_loss, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, use_mask_logit_loss=args.use_mask_logit_loss, mask_logit_loss_regularizer=args.mask_logit_loss_regularizer, object_loss_weighting_params=args.object_loss_weighting_params, mask_loss_weighting_params=args.mask_loss_weighting_params
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, mask_variation_regularizer=args.mask_variation_regularizer,save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu,  use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, class_loss=args.class_loss, frozen=args.frozen, 
         weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multiclass=args.multiclass, use_bounding_loss=args.use_bounding_loss, 
         similarity_loss_mode=args.similarity_loss_mode, weighted_sampling=args.weighted_sampling, background_loss_scheduling=args.background_loss_scheduling, similarity_loss_scheduling=args.similarity_loss_scheduling, mask_loss_scheduling=args.mask_loss_scheduling, use_loss_scheduling=args.use_loss_scheduling,
         freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, save_all_class_masks=args.save_all_class_masks, objective=args.objective, background_loss=args.background_loss, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, use_mask_logit_loss=args.use_mask_logit_loss, mask_logit_loss_regularizer=args.mask_logit_loss_regularizer, object_loss_weighting_params=args.object_loss_weighting_params, mask_loss_weighting_params=args.mask_loss_weighting_params
        )

elif args.model_to_train == "explainer":
    model = ExplainerClassifierModel(
        num_classes=num_classes, dataset=args.dataset, classifier_type=args.classifier_type, classifier_checkpoint=args.classifier_checkpoint, fix_classifier=args.fix_classifier, learning_rate=args.learning_rate, 
        class_mask_min_area=args.class_mask_min_area, class_mask_max_area=args.class_mask_max_area, entropy_regularizer=args.entropy_regularizer, use_mask_variation_loss=args.use_mask_variation_loss, 
        mask_variation_regularizer=args.mask_variation_regularizer, use_mask_area_loss=args.use_mask_area_loss, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, 
        mask_total_area_regularizer=args.mask_total_area_regularizer, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, 
        save_masked_images=args.save_masked_images, save_masks=args.save_masks,
        save_all_class_masks=args.save_all_class_masks, save_path=args.save_path
    )
    print(model)

    if args.explainer_classifier_checkpoint != None:
        model = model.load_from_checkpoint(
            args.explainer_classifier_checkpoint,
            num_classes=num_classes, dataset=args.dataset, classifier_type=args.classifier_type, classifier_checkpoint=args.classifier_checkpoint, fix_classifier=args.fix_classifier, learning_rate=args.learning_rate, 
            class_mask_min_area=args.class_mask_min_area, class_mask_max_area=args.class_mask_max_area, entropy_regularizer=args.entropy_regularizer, use_mask_variation_loss=args.use_mask_variation_loss, 
            mask_variation_regularizer=args.mask_variation_regularizer, use_mask_area_loss=args.use_mask_area_loss, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, 
            mask_total_area_regularizer=args.mask_total_area_regularizer, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, 
            save_masked_images=args.save_masked_images, save_masks=args.save_masks, save_all_class_masks=args.save_all_class_masks, save_path=args.save_path
        )

elif args.model_to_train == "slike_selfexplainer":
    model = SelfExplainer_Slike(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, class_loss=args.class_loss, frozen=args.frozen, 
         freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, save_all_class_masks=args.save_all_class_masks, objective=args.objective, background_loss=args.background_loss, weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multiclass=args.multiclass, class_only=args.class_only,
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, class_loss=args.class_loss, frozen=args.frozen, 
         freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, save_all_class_masks=args.save_all_class_masks, objective=args.objective, background_loss=args.background_loss,  weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multiclass=args.multiclass, class_only=args.class_only,
        )

elif args.model_to_train == "fcn":
    model = FCN16(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, profiler=profiler, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, frozen=args.frozen, freeze_every=args.freeze_every, 
         background_activation_loss=args.background_activation_loss, background_loss=args.background_loss, save_all_class_masks=args.save_all_class_masks,  weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, profiler=profiler, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, frozen=args.frozen, freeze_every=args.freeze_every, 
         background_activation_loss=args.background_activation_loss, background_loss=args.background_loss, save_all_class_masks=args.save_all_class_masks,  weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer
        )
elif args.model_to_train == "mlp":
    model = MLP(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, objective=args.objective, class_loss = args.class_loss, 
         rgb=args.rgb, frozen=args.frozen, freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, target_threshold=args.target_threshold, 
         non_target_threshold=args.non_target_threshold, background_loss=args.background_loss, save_all_class_masks=args.save_all_class_masks,  weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multiclass=args.multiclass, similarity_loss_mode=args.similarity_loss_mode,
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, objective=args.objective, class_loss=args.class_loss, 
         rgb=args.rgb, frozen=args.frozen, freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, target_threshold=args.target_threshold, 
         non_target_threshold=args.non_target_threshold, background_loss=args.background_loss, save_all_class_masks=args.save_all_class_masks,  weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multiclass=args.multiclass, similarity_loss_mode=args.similarity_loss_mode,
        )
elif args.model_to_train == "resnet50":
    model = Resnet50(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, 
       gpu=args.gpu, metrics_threshold=args.metrics_threshold, multiclass=args.multiclass, weighted_sampling=args.weighted_sampling
    )

elif args.model_to_train == 'simple':
    model = Simple_Model(
        num_classes=num_classes, pretrained=False, aux_classifier=args.aux_classifier, learning_rate=args.learning_rate, dataset = args.dataset, multiclass=args.multiclass
    )
else:
    raise Exception("Unknown model type: " + args.model_to_train)


# Define Early Stopping condition
early_stop_callback = EarlyStopping(
    #monitor="loss" if args.dataset in ['TOY', 'COLOR'] else "val_loss",
    monitor='loss',
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
    monitor='loss',
    save_top_k=10
)

profiler = AdvancedProfiler(dirpath=main_dir, filename='selfexplainer_model_report')
if args.dataset in ['OISMALL', 'OI']:
    trainer = pl.Trainer(
        logger = logger,
        callbacks = [early_stop_callback, checkpoint_callback],
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


