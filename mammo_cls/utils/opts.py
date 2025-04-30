import os
import torch
import logging
import argparse
import json
from datetime import datetime
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


def arg_parse():
    parser = argparse.ArgumentParser(description='FineTune Mammo_AGE model for downstream prediction tasks')

    # Training model parameters
    # ---------------------------------
    parser.add_argument('--task', default='classification', help='Task to be fine-tuned: [classification]')
    parser.add_argument('--num-output-neurons', default=2, type=int, help='Number of output neurons for the new task')
    parser.add_argument('--pretrained_model_path', default='/Path/to/checkpoint', type=str, help='Path to the pretrained model')
    parser.add_argument('--dont_load_pretrained_weight', default=False, action='store_true', help='Do not load pretrained weight, set to True for training from scratch')
    parser.add_argument('--freeze_backbone', default=False, action='store_true', help='Freeze the backbone model')
    parser.add_argument('--multi_view', action='store_true', help='If true, will use multi-view model.')
    parser.add_argument('--model_method', default='Finetune', type=str, help='model method in [Finetune, backbone].')
    parser.add_argument('--max_t', default=50, type=int, help='number of time sampling for the latent space')
    parser.add_argument('--use_sto', default=False, action='store_true', help='Use stochasticity in the model')
    parser.add_argument('--auxiliary_task', default=None, nargs='+', type=str, help='List of auxiliary tasks to enable (e.g., age, density, birads)')

    # Training parameters
    # ---------------------------------
    parser.add_argument('--debug', action='store_true', help='Quick setting params for debugging')
    parser.add_argument('--seed', default=5, type=int, help='seed for initializing training.')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer in [SGD, Adam, RMSprop].')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--schedule', default=[20, 40, 60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, metavar='N', help='gradient accumulation steps')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float, metavar='W', help='weight decay (default: 0.)', dest='weight_decay')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--num-workers', default='16', type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--training_step', default=1000000, type=int, metavar='N', help='Step for training each training loop')
    parser.add_argument('--val_step', default=1000000, type=int, metavar='N', help='Step for validating each val loop')
    parser.add_argument('--test_step', default=1000000, type=int, metavar='N', help='Step for testing each test loop')
    parser.add_argument('--early_stop_patient', default=10, type=int, metavar='N', help='manual epoch number (for early stopping)')
    parser.add_argument('--balance_training', action='store_true', help='Balance pos and neg sample for training')
    parser.add_argument('--lr_decay_patient', default=3, type=int, metavar='N', help='manual epoch number (for learning rate decay)')
    parser.add_argument('--fold', default=0, type=int, help='Cross validation')

    # Checkpoint
    # ---------------------------------
    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_retrain', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    # Dataset
    # ---------------------------------
    parser.add_argument('--dataset', default='[inhouse]', type=str, nargs='+', help='dataset in [inhouse, embed, csaw,].')
    parser.add_argument('--img-size', default=1024, type=int, help='size of image')
    parser.add_argument('--dataset-config', default='/Path/to/dataset/config.yaml', type=str, metavar='PATH', help='path to dataset config')
    parser.add_argument('--csv-dir', default='/Path/to/data/base_csv_folder', type=str, metavar='PATH', help='path to base folder for all datasets csv files')
    parser.add_argument('--image-dir', default='/Path/to/data/base_image_folder', type=str, metavar='PATH', help='path to base folder for all datasets image files')
    parser.add_argument('--fraction', default=1.0, type=float, help='fraction of data to be used for training')

    # Task specific parameters
    # ---------------------------------

    args = parser.parse_args()

    if args.model_method == 'Mammo_Clip' and not args.dont_load_pretrained_weight:
        model_method = "Mammo-CLIP" if not args.auxiliary_task else "Mammo-CLIP-Aux_Task"
        arch = "efficientnet_b2" if "b2" in args.pretrained_model_path.split('/')[-1] else "efficientnet_b5"
        size_fold = f"{args.img_size}_fold{args.fold}"
    elif args.model_method == 'MIRAI':
        model_method = "MIRAI" if not args.auxiliary_task else "MIRAI-Aux_Task"
        arch = "resnet18"
        size_fold = f"{args.img_size}_fold{args.fold}"
    elif args.model_method == 'RSNA':
        model_method = "RSNA" if not args.auxiliary_task else "RSNA-Aux_Task"
        arch = "convnext_small"
        size_fold = f"{args.img_size}_fold{args.fold}"
    else:
        raise ValueError("Invalid model method or pretrained weight option, please check the arguments.")

    if args.model_method == 'backbone':
        arch = f"{args}-backbone"

    if args.fraction != 1.0:
        size_fold = f"{size_fold}-fl-{args.fraction}"

    args.results_dir = f"{args.results_dir}/{args.task}/{'_'.join(args.dataset)}/{model_method}-{arch}/{size_fold}/"

    os.makedirs(args.results_dir, exist_ok=True)
    print(args.results_dir)
    return args


def get_criterion(args): # define loss function
    if args.task in ['classification']:
        if args.num_output_neurons == 1:
            criterion_BCE = nn.BCEWithLogitsLoss().cuda()
        else:
            criterion_BCE = nn.CrossEntropyLoss().cuda()
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")
    criterion = {'criterion_BCE': criterion_BCE,}
    return criterion


def get_model(args):
    """
    Initialize and return the appropriate fine-tuned model based on args.
    """
    task_class_map = {'age': 1, 'birads': 7, 'density': 4}
    aux_task_config = None

    if args.auxiliary_task:
        aux_task_config = {task: task_class_map[task] for task in args.auxiliary_task}

    # Select model class
    if args.model_method == 'Mammo_Clip':
        if aux_task_config:
            from models.mammoclip_finetune_model import MultiTaskMammoClip as DownstreamModel
        else:
            from models.mammoclip_finetune_model import FineTuneMammoClip as DownstreamModel

    elif args.model_method == 'MIRAI':
        if aux_task_config:
            from models.MIRAI_finetune_model import MultiTaskMIRAI as DownstreamModel
        else:
            from models.MIRAI_finetune_model import FineTuneMIRAI as DownstreamModel

    elif args.model_method == 'RSNA':
        if aux_task_config:
            raise NotImplementedError("Auxiliary task not implemented for RSNA")
        from models.RSNA_finetune_model import FineTuneRSNA as DownstreamModel

    else:
        raise NotImplementedError(f"Model method '{args.model_method}' not implemented.")

    # Instantiate model
    model_kwargs = dict(
        pretrained_model_path=args.pretrained_model_path,
        num_classes=args.num_output_neurons,
        freeze_backbone=args.freeze_backbone,
        dont_load_pretrained_weight=args.dont_load_pretrained_weight,
    )

    if aux_task_config:
        model_kwargs['auxiliary_task'] = aux_task_config

    model = DownstreamModel(**model_kwargs).cuda()
    logging.info(model)
    return model


def get_learning_demo(args):  # define train val test demo
    if args.task in ['classification']:
        from learning.baseline_learning_demo import get_train_val_test_demo
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    train, validate, test = get_train_val_test_demo()
    return train, validate, test


def get_dataset(args):
    if args.model_method in ['Mammo_Clip', 'MIRAI', 'RSNA']:
        from dataload.combine_data_loader import dataloador
        from dataload.custom_dataset import CustomDataset
    else:
        raise NotImplementedError
    return CustomDataset, dataloador


def get_optimizer(args, model): # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'Adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    return optimizer


def get_hyperparams_for_test(args):
    with open(args.results_dir + '/args.json', 'r') as f:
        raw_dict = json.load(f)

    args.task = raw_dict['task']
    args.num_output_neurons = raw_dict['num_output_neurons']
    args.pretrained_model_path = raw_dict['pretrained_model_path']
    args.dont_load_pretrained_weight = raw_dict['dont_load_pretrained_weight'] if 'dont_load_pretrained_weight' in raw_dict else False
    args.freeze_backbone = raw_dict['freeze_backbone']
    args.multi_view = raw_dict['multi_view']
    args.model_method = raw_dict['model_method']
    args.max_t = raw_dict['max_t']
    args.use_sto = raw_dict['use_sto']

    args.seed = raw_dict['seed']
    args.use_sto = raw_dict['use_sto']

    args.max_followup = raw_dict['max_followup']
    args.years_at_least_followup = raw_dict['years_at_least_followup']
    args.time_to_events_weights = raw_dict['time_to_events_weights']
    args.weight_class_loss = raw_dict['weight_class_loss']
    args.screening_clean = raw_dict['screening_clean'] if 'screening_clean' in raw_dict else False

    args.path_risk_model = f'{args.results_dir}'
    args.results_dir = f'{args.results_dir}/predict_result/'

    os.makedirs(args.results_dir, exist_ok=True)
    return args

