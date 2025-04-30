import os
import shutil
import argparse
import json
import logging
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current Directory: {current_dir}")

# Add custom module paths
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'losses'))
sys.path.append(os.path.join(current_dir, 'learning'))
sys.path.append(os.path.join(current_dir, 'models'))

# Custom utility imports
from utils.opts import arg_parse, get_criterion, get_model, get_learning_demo, get_optimizer, get_dataset
from utils.utils import *
from utils.mylogging import open_log
from utils import backup


def get_fraction_data_info(data_info, fraction=0.1, seed=42):
    """
    Returns a fraction of the training data based on unique patient IDs.

    Args:
        data_info (DataFrame): Full training data info
        fraction (float): Proportion of data to retain
        seed (int): Random seed for reproducibility

    Returns:
        DataFrame: Subsampled data info
    """
    from sklearn.model_selection import train_test_split
    unique_ids = data_info['patient_id'].astype(str).unique()
    selected_ids, _ = train_test_split(unique_ids, train_size=fraction, random_state=seed)
    return data_info[data_info['patient_id'].astype(str).isin(selected_ids)]


def get_data_info(args):
    """
    Load and preprocess the dataset CSVs for training, validation, and testing.

    Returns:
        datasets (list): Names of datasets
        train_data_info, valid_data_info, test_data_info (list of DataFrames)
    """
    dataset_cfg = read_yaml(args.dataset_config)
    logging.info(dataset_cfg)

    base_image_dir = args.image_dir
    args.image_dir = []

    for dataset in args.dataset:
        if dataset not in dataset_cfg:
            raise ValueError(f"DATASET: {dataset} is not supported.")
        args.image_dir.append(f"{base_image_dir}/{dataset_cfg[dataset]['image_dir']}")

    train_data_info, valid_data_info, test_data_info = [], [], []

    for dataset in args.dataset:
        if dataset in ['vindr', 'rsna', 'csaw', 'cmmd', 'embed', 'bmcd']:
            logging.info(f"DATASET: {dataset}")
            args.image_dir.append(f"{base_image_dir}/{dataset_cfg[dataset]['image_dir']}")
            data_info = pd.read_csv(f'{args.csv_dir}/external_datasets/{dataset}_data_info.csv')
            if dataset in ['rsna', 'embed']:
                data_info = data_info[data_info['label'] == 1]
            train_data_info.append(data_info.reset_index())
        else:
            # data_info = pd.read_csv(f'{args.csv_dir}/{args.task}/{dataset}_data_info.csv')
            data_info = pd.read_csv(f'{args.csv_dir}/cv_split/train_val_fold{args.fold}.csv')
            logging.info(f"DATASET: {dataset}")

            _train = data_info[data_info['split_group'] == 'train']
            if args.fraction != 1.0:
                _train = get_fraction_data_info(_train, args.fraction, args.seed)
                logging.info(f"DATASET: {dataset}, fraction: {args.fraction}")

            _valid = data_info[data_info['split_group'] == 'valid']
            _test = data_info[data_info['split_group'] == 'test']

            # Save splits to CSV
            _train.to_csv(f'{args.results_dir}/{dataset}_train_data_info.csv')
            _valid.to_csv(f'{args.results_dir}/{dataset}_valid_data_info.csv')
            _test.to_csv(f'{args.results_dir}/{dataset}_test_data_info.csv')

            train_data_info.append(_train.reset_index())
            valid_data_info.append(_valid.reset_index())
            test_data_info.append(_test.reset_index())

    return args.dataset, train_data_info, valid_data_info, test_data_info


def main():
    args = arg_parse()
    seed_reproducer(args.seed)

    open_log(args)
    logging.info(str(args).replace(',', "\n"))

    best_auc = 0.0

    # Load data and create dataloaders
    datasets_names, train_info, valid_info, test_info = get_data_info(args)
    custom_dataset, dataloader_fn = get_dataset(args)

    train_loader = dataloader_fn(custom_dataset, train_info, datasets_names, args, train=True)
    # valid_loader = dataloader_fn(custom_dataset, valid_info, datasets_names, args)
    # test_loader = dataloader_fn(custom_dataset, test_info, datasets_names, args)
    # Only validation and test loaders on the spr dataset
    valid_loader = dataloader_fn(custom_dataset, valid_info, ['spr'], args)
    test_loader = dataloader_fn(custom_dataset, test_info, ['spr'], args)

    logging.info('Finished loading data')

    # Save arguments to JSON
    with open(f'{args.results_dir}/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # Load model components
    criterion = get_criterion(args)
    model = get_model(args)
    train_fn, val_fn, test_fn = get_learning_demo(args)
    optimizer = get_optimizer(args, model)

    epoch_start = args.start_epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=1, mode='max',
                                                           patience=args.lr_decay_patient, factor=0.5)
    early_stopping = EarlyStopping(patience=args.early_stop_patient, verbose=True, mode='max')

    # Backup code/scripts
    set_backup(args.results_dir)

    # Resume from checkpoints
    if args.resume_retrain:
        checkpoint = torch.load(args.resume_retrain)
        model.load_state_dict(checkpoint['state_dict'])

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        best_auc = checkpoint.get('best_auc', 0.0)
        logging.info(f'Loaded from: {args.resume}')
        test_result = checkpoint.get('test_result')

    cudnn.benchmark = True

    for epoch in range(epoch_start, args.epochs + 1):
        if epoch == 0:
            # Evaluate initial model on test set
            test_result = test_fn(model, test_loader, criterion, args)
            logging.info(
                f'Init Model | Val Loss: {test_result["loss"]:.4f}, AUC: {test_result["auc"]:.4f}, ACC: {test_result["acc"]:.4f}')
        else:
            train_result = train_fn(model, train_loader, criterion, optimizer, epoch, args)
            logging.info(f'Epoch {epoch} | Train Loss: {train_result["loss"]:.4f}, AUC: {train_result["auc"]:.4f}, ACC: {train_result["acc"]:.4f}')

            valid_result = val_fn(model, valid_loader, criterion, args)
            logging.info(f'Epoch {epoch} | Val Loss: {valid_result["loss"]:.4f}, AUC: {valid_result["auc"]:.4f}, ACC: {valid_result["acc"]:.4f}')

            # Adjust learning rate
            scheduler.step(valid_result['auc'])
            for group in optimizer.param_groups:
                print(f"\n* Learning rate: {group['lr']:.2e} *\n")

            # Save best model
            is_best = valid_result['auc'] > best_auc
            best_auc = max(valid_result['auc'], best_auc)

            if is_best:
                test_result = test_fn(model, test_loader, criterion, args, save_pkl=f'best_{epoch}')
                logging.info(f'*** Best Epoch {epoch} | Test AUC: {test_result["auc"]:.4f}, ACC: {test_result["acc"]:.4f} ***')

            save_checkpoint(args.results_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
                'train_result': train_result,
                'valid_result': valid_result,
                'test_result': test_result,
            }, is_best)

            # Early stopping check
            early_stopping(valid_result['auc'])
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

    # Final evaluation on best checkpoint
    best_model = torch.load(f'{args.results_dir}/model_best.pth.tar')
    model.load_state_dict(best_model['state_dict'])
    final_test = test_fn(model, test_loader, criterion, args, save_pkl='best')

    logging.info(f'Final Test | AUC: {final_test["auc"]:.4f}, ACC: {final_test["acc"]:.4f}')


def set_backup(custom_backup_dir="custom_backups"):
    """
    Back up current scripts and imported modules for reproducibility.
    """
    backup_dir = os.path.join(custom_backup_dir, "mammo_cls")
    backup.save_script_backup(__file__, backup_dir)
    project_root = os.path.dirname(os.path.abspath(__file__))
    backup.backup_imported_modules(project_root, backup_dir)


if __name__ == '__main__':
    main()
