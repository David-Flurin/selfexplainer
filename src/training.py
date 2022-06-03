from cgi import test
from evaluation.plot import plot_losses

import torch
import os
import sys
import shutil
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from pathlib import Path
import pickle
import hashlib

from torch.utils.tensorboard import SummaryWriter

from data.dataloader import ColorDataModule, MNISTDataModule, OISmallDataModule, ToyDataModule, VOCDataModule, COCODataModule, CUB200DataModule, ToyData_Saved_Module
from utils.argparser import get_parser, write_config_file
from models.no_lightning.selfexplainer_nl import SelfExplainer, SelfExplainer_Slike

from models.mlp import MLP
from utils.image_display import save_all_class_masked_images, save_masked_image

from toy_dataset import generator

from tqdm import tqdm


main_dir = Path(os.path.dirname(os.path.abspath(__file__)))

parser = get_parser()
args = parser.parse_args()
print(args)
if args.arg_log:
    write_config_file(args)

#pl.seed_everything(args.seed)
#profiler = AdvancedProfiler(dirpath=main_dir, filename='performance_report')

print('Dir: ',args.save_path)



# Set up data module
if args.dataset == "VOC":
    data_path = main_dir / args.data_base_path / 'VOC2007'
    data_module = VOCDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 20
elif args.dataset == "OISMALL":
    data_path = main_dir / args.data_base_path / 'OI_SMALL'
    data_module = OISmallDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 3
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

# Set up Logging
if args.use_tensorboard_logger:
    log_dir = Path(args.save_path, "tb_logs")
    writer=SummaryWriter(log_dir=log_dir)
else:
    writer=None

# Set up model

if args.model_to_train == "selfexplainer":
    model = SelfExplainer(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu, use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, class_loss=args.class_loss, frozen=args.frozen, 
         freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, save_all_class_masks=args.save_all_class_masks, objective=args.objective, background_loss=args.background_loss, weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multiclass=args.multiclass, use_bounding_loss=args.use_bounding_loss, writer=writer,
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, pretrained=args.use_imagenet_pretraining, use_weighted_loss=args.use_weighted_loss, 
        use_similarity_loss=args.use_similarity_loss, similarity_regularizer=args.similarity_regularizer, use_background_loss = args.use_background_loss, bg_loss_regularizer=args.bg_loss_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, use_mask_variation_loss=args.use_mask_variation_loss, save_path=args.save_path, save_masked_images=args.save_masked_images,
         save_masks=args.save_masks, gpu=args.gpu,  use_perfect_mask=args.use_perfect_mask, count_logits=args.count_logits, class_loss=args.class_loss, frozen=args.frozen, 
         freeze_every=args.freeze_every, background_activation_loss=args.background_activation_loss, save_all_class_masks=args.save_all_class_masks, objective=args.objective, background_loss=args.background_loss,  weighting_koeff=args.weighting_koeff, mask_total_area_regularizer=args.mask_total_area_regularizer, aux_classifier=args.aux_classifier, multiclass=args.multiclass, use_bounding_loss=args.use_bounding_loss, writer=writer,
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
else:
    raise Exception("Unknown model type: " + args.model_to_train)


class Trainer():
    def __init__(self, model, data_module, lr=1e-4):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr)

        data_module.setup()
        self.train_loader = data_module.train_dataloader()
        self.val_loader = data_module.val_dataloader()
        self.test_loader = data_module.test_dataloader()

    def fit(self, epochs):
        pbar = tqdm(total=len(self.train_loader), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', leave=True)
        postfix={}

        for e in range(epochs):
            pbar.set_description(f'Epoch: {e}', refresh=False)

            pbar.reset()
            pbar.set_postfix(postfix, refresh=False)
            pbar.set_postfix(postfix, refresh=False)
            for i, batch in enumerate(self.train_loader):
                loss = model.training_step(batch, i)
                loss.backward()
                self.optimizer.step()
                postfix['loss'] = loss.item()
                pbar.update(1)
                pbar.set_postfix(postfix, refresh=False)

            m = self.model.training_epoch_end()
            postfix['metrics'] = m
            pbar.set_postfix(postfix, refresh=False)

            #pbar.update(' '.join([f'{metric}: {value.item()}' for metric,value in m.items()]))


            if self.val_loader:
                self.model.eval()
                for i, batch in enumerate(self.val_loader):
                    loss = model.validation_step(batch.to(self.device, i))

                self.model.train()

            

    def test(self):
        pass


trainer = Trainer(model, data_module)
trainer.fit(10)




        

        


