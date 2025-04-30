import gc
import pickle
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import warnings
from torchvision.transforms import v2 as T

from utils.utils import AverageMeter
from utils.metrics import calculate_classification_metrics as cal_metrics

warnings.filterwarnings('ignore')


def generate_prediction_df(data):
    """
    Generate prediction probabilities for the given data.

    Parameters:
    - data: Dictionary containing prediction data

    - Returns:
    - DataFrame: Contains patient_id, exam_id, img_id, view, laterality, risk_probabilitie, risk_label
    """
    metadata_info = {}
    metadata_info['patient_id'] = list(data['patient_id'])
    metadata_info['exam_id'] = list(data['exam_id'])
    metadata_info['img_id'] = list(data['img_id'])
    metadata_info['view'] = list(data['view'])
    metadata_info['laterality'] = list(data['laterality'])
    metadata_info['risk_probabilitie'] = list(data['risk_probabilitie'].reshape(-1))
    metadata_info['risk_label'] = list(data['risk_label'])
    metadata_info = pd.DataFrame(data=metadata_info)
    return metadata_info


def compute_losses(criterion, input, output, **kwargs):
    """
    Computes total loss for training/validation based on BCE and additional losses if available.

    Parameters:
    - args: Command-line or configuration arguments
    - criterion: Dictionary containing loss functions
    - input: Input batch containing ground truth
    - output: Model output containing predictions and optional loss

    Returns:
    - Combined loss value
    """
    risk_label = input['label'].float().cuda()
    pred_risk = output['risk'].reshape(-1)

    # Basic binary cross-entropy loss
    loss = criterion['criterion_BCE'](pred_risk, risk_label) * 10

    # If the model provides an additional loss term, add it
    if output['loss'] is not None:
        loss += output['loss']

    return loss


def input_and_output(args, model, input, **kwargs):
    """
    Feeds input through the model and extracts predictions and loss.

    Parameters:
    - args: Runtime configuration
    - model: PyTorch model
    - input: Batch of input data
    - kwargs: Additional arguments (e.g., inference flag)

    Returns:
    - Dictionary containing predictions and optional outputs/loss
    """
    img = input['img'].cuda()

    auxiliary_targets = None
    if args.auxiliary_task is not None:
        auxiliary_targets = {task: input[task].cuda() for task in args.auxiliary_task}

    output_dict = model(img, auxiliary_targets=auxiliary_targets)
    loss = output_dict.get('loss', None)

    return {
        'risk': output_dict['predict'],
        'loss': loss,
        'output_dict': output_dict if kwargs.get('inference', False) else None
    }


def direct_train(model, data_loader, criterion, optimizer, epoch, args):
    """
    Trains the model for one epoch.

    Returns:
    - Training loss and placeholder metrics
    """
    model.train()
    total_loss, total_num = 0.0, 0
    train_bar = tqdm(data_loader)
    losses = AverageMeter()
    i_debug = 0

    for input in train_bar:
        if args.debug and i_debug > 30:
            break
        i_debug += 1

        if args.debug:
            pickle.dump(input, open(f'{args.results_dir}/result_{i_debug}.pkl', 'wb'))

        output = input_and_output(args, model, input, train=True)
        loss = compute_losses(criterion, input, output)

        # Accumulate gradients if specified
        if args.accumulation_steps == 1:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            loss = loss / args.accumulation_steps
            loss.backward()
            if i_debug % args.accumulation_steps == 0 or i_debug == len(data_loader):
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size * args.accumulation_steps
        losses.update(loss.item())

        train_bar.set_description(
            f'Train Epoch: [{epoch}/{args.epochs}], lr: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {losses.avg:.3f}'
        )

        if i_debug // args.accumulation_steps >= args.training_step:
            break

    return {'loss': losses.avg, 'auc': 0.0, 'acc': 0.0, 'metrics': None}


def direct_validate(model, valid_loader, criterion, args):
    """
    Validates the model on the validation dataset.

    Returns:
    - Validation loss and performance metrics
    """
    model.eval()
    losses = AverageMeter()
    all_probs, all_labels = [], []
    total_loss = 0.0
    i_debug = 0

    with torch.no_grad():
        valid_bar = tqdm(valid_loader)
        for input in valid_bar:
            if args.debug and i_debug > 30:
                break
            i_debug += 1

            output = input_and_output(args, model, input)
            loss = compute_losses(criterion, input, output)

            labels = input['label'].cuda()
            risk = output['risk']

            losses.update(loss.item())
            total_loss += loss.item() * valid_loader.batch_size

            valid_bar.set_description(f'Valid MAE: {losses.avg:.4f}')

            # Use sigmoid for binary and softmax for multi-class
            pred_probs = F.sigmoid(risk) if args.num_output_neurons == 1 else F.softmax(risk, dim=-1)[:, -1]
            all_probs.append(pred_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if i_debug > args.val_step:
                break

    gc.collect()

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    metrics = cal_metrics(all_probs, all_labels)

    logging.info(f'Validation Loss: {losses.avg:.2f}')
    for key, val in metrics.items():
        logging.info(f'{key}: {val:.4f}')

    return {'loss': losses.avg, 'auc': metrics['AUC'], 'acc': metrics['ACC'], 'metrics': metrics}


def direct_test(model, test_loader, criterion, args, save_pkl=None, **kwargs):
    """
    Tests the model on the test dataset.

    Returns:
    - Test loss and performance metrics
    """
    model.eval()
    losses = AverageMeter()
    total_loss = 0.0
    i_debug = 0

    all_probs, all_labels = [], []
    all_patient_ids, all_exam_ids, all_img_ids = [], [], []
    all_views, all_lateralitys = [], []

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for input in test_bar:
            if args.debug and i_debug > 30:
                break
            i_debug += 1

            output = input_and_output(args, model, input, inference=True)
            loss = compute_losses(criterion, input, output)

            labels = input['label'].cuda()
            risk = output['risk']

            losses.update(loss.item())
            total_loss += loss.item() * test_loader.batch_size

            test_bar.set_description(f'Test Loss: {losses.avg:.4f}')

            # Predict probabilities
            pred_probs = F.sigmoid(risk) if args.num_output_neurons == 1 else F.softmax(risk, dim=-1)[:, -1]
            all_probs.append(pred_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            all_patient_ids.append(input['pid'])
            all_exam_ids.append(input['exam_id'])
            all_img_ids.append(input['img_id'])
            all_views.append(input['view'])
            all_lateralitys.append(input['laterality'])

            if "test_step" in args and i_debug > args.test_step:
                break

    gc.collect()

    all_data = {
        'patient_id': np.concatenate(all_patient_ids),
        'exam_id': np.concatenate(all_exam_ids),
        'img_id': np.concatenate(all_img_ids),
        'view': np.concatenate(all_views),
        'laterality': np.concatenate(all_lateralitys),
        'risk_probabilitie': np.concatenate(all_probs),
        'risk_label': np.concatenate(all_labels),
    }

    metrics = cal_metrics(all_data['risk_probabilitie'], all_data['risk_label'])

    if save_pkl is not None:
        pickle.dump(all_data, open(f'{args.results_dir}/result_{save_pkl}.pkl', 'wb'))
        metadata_info = generate_prediction_df(all_data)
        metadata_info.to_csv('{}/result_{}.csv'.format(args.results_dir, save_pkl), index=False)

    logging.info(f'Test Loss: {losses.avg:.2f}')
    for key, val in metrics.items():
        logging.info(f'{key}: {val:.4f}')

    return {'loss': losses.avg, 'auc': metrics['AUC'], 'acc': metrics['ACC'], 'metrics': metrics}


def get_train_val_test_demo():
    """
    Returns the training, validation, and test functions.

    - Returns:
    - Tuple: Contains training, validation, and test functions
    """
    return direct_train, direct_validate, direct_test
