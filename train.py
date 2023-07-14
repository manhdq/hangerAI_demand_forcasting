import os
import argparse
import wandb
import pandas as pd
from datetime import datetime

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import loggers as pl_loggers

from dataset.data_multitrends import ZeroShotDataset
from models.GTM import GTM


def run(args):
    print(args)
    # Seeds for reproducibility (By default we use the number 42)
    pl.seed_everything(args.seed)

    # Load sales data
    train_df = pd.read_csv(os.path.join(args.data_folder, 'train.csv'), parse_dates=['release_date'])
    test_df = pd.read_csv(os.path.join(args.data_folder, 'test.csv'), parse_dates=['release_date'])

    # Load category and color encodings
    cat_dict = torch.load(os.path.join(args.data_folder, 'category_labels.pt'))
    col_dict = torch.load(os.path.join(args.data_folder, 'color_labels.pt'))
    fab_dict = torch.load(os.path.join(args.data_folder, 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(os.path.join(args.data_folder, 'gtrends.csv'), index_col=[0], parse_dates=True)

    trainset = ZeroShotDataset(train_df, os.path.join(args.data_folder, "images"), gtrends, cat_dict, col_dict,
                                fab_dict, args.trend_len, train=True)
    testset = ZeroShotDataset(test_df, os.path.join(args.data_folder, "images"), gtrends, cat_dict, col_dict,
                                fab_dict, args.trend_len, train=False)

    # If you wish to debug with less data you can use this
    if args.quick_debug:
        print(f"Train set: {len(trainset)}")
        print(f"Test set: {len(testset)}")
        trainset = torch.utils.data.Subset(trainset, list(range(100)))
        testset = torch.utils.data.Subset(testset, list(range(100)))

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = GTM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_text=args.use_text,
        use_img=args.use_img,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
        gpu_num=args.gpu_num,
    )

    # Model Training
    # Define model saving procedure
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    model_savename = args.model_type + '_' + args.wandb_run

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, args.model_type,),
        filename=model_savename+'---{epoch}---'+dt_string,
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj, name=args.wandb_run)
    wandb_logger = pl_loggers.WandbLogger()
    wandb_logger.watch(model)

    # If you wish to use Tensorboard you can change the logger to:
    # tb_logger = pl_loggers.TensorBoardLogger(args.log_dir+'/', name=model_savename)
    trainer = pl.Trainer(gpus=[args.gpu_num], max_epochs=args.epochs, check_val_every_n_epoch=1,  ##TODO: Dynamic and change this
                         logger=wandb_logger, callbacks=[checkpoint_callback])

    # Fit model
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    # Print out path of best model
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument("--quick_debug", action="store_true")

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='manhdo')
    parser.add_argument('--wandb_proj', type=str, default='GTM')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)
