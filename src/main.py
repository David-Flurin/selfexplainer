import torch
import os
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path

from data.dataloader import VOCDataModule, COCODataModule, CUB200DataModule
from utils.argparser import get_parser, write_config_file
from models.selfexplainer import SelfExplainer

main_dir = Path(os.path.dirname(os.path.abspath(__file__)))

parser = get_parser()
args = parser.parse_args()
print(args)
if args.arg_log:
    write_config_file(args)

pl.seed_everything(args.seed)

# Set up Logging
if args.use_tensorboard_logger:
    log_dir = "tb_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
else:
    raise Exception("Unknown dataset " + args.dataset)

# Set up model

if args.model_to_train == "selfexplainer":
    model = SelfExplainer(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, save_path=args.save_path
    )
    if args.checkpoint != None:
        model = model.load_from_checkpoint(
            args.fcnn_checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, save_path=args.save_path
        )
else:
    raise Exception("Unknown model type: " + args.model_to_train)

# Define Early Stopping condition
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=args.early_stop_min_delta,
    patience=args.early_stop_patience,
    verbose=False,
    mode="min",
)

trainer = pl.Trainer(
    logger = logger,
    callbacks = [early_stop_callback],
    gpus = [0] if torch.cuda.is_available() else 0,
    terminate_on_nan = True,
    checkpoint_callback = args.checkpoint_callback,
)

if args.train_model:
    trainer.fit(model=model, datamodule=data_module)
    trainer.test()
else:
    trainer.test(model=model, datamodule=data_module)
