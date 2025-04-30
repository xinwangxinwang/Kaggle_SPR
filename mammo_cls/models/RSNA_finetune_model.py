# Adjusted from https://github.com/yala/Mirai
import os
import logging
import torch
from torch import nn
from typing import Dict
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_dir)
import warnings

warnings.filterwarnings('ignore')
from timm.models import create_model



class FineTuneRSNA(nn.Module):
    def __init__(self,
                 pretrained_model_path: str = None,
                 num_classes: int = 2,
                 freeze_backbone: bool = True,
                 dont_load_pretrained_weight: bool = False, ):
        """
        Fine-tuned model based on Mirai with a new classification head.
        Args:
            pretrained_model_path (str): Path to the pretrained model weights (optional).
            num_classes (int): Number of classes for the classification task (default: 2 for binary classification).
            freeze_backbone (bool): Whether to freeze the backbone parameters.
            dont_load_pretrained_weight (bool): Whether to load the pretrained weights.
        """

        super(FineTuneRSNA, self).__init__()

        model_info = {'model_name': 'convnext_small.fb_in22k_ft_in1k_384',
                      'num_classes': 1, 'in_chans': 3, 'global_pool': 'max',}

        if pretrained_model_path:
            if not dont_load_pretrained_weight:
                logging.info(f'Loading image encoder weights from checkpoint: {pretrained_model_path}')

                model = create_model(
                    model_info['model_name'],
                    num_classes=model_info['num_classes'],
                    in_chans=model_info['in_chans'],
                    pretrained=False,
                    checkpoint_path=pretrained_model_path,
                    global_pool=model_info['global_pool'],)
            else:
                raise ValueError("Pretrained weight is required for loading the model.")
        else:
            raise ValueError("Pretrained model path is required for fine-tuning.")

        self.model = model
        # Replace final fc layer of the model
        self.model.head.fc = nn.Linear(
            in_features=self.model.head.fc.in_features,
            out_features=num_classes,
            bias=True)

        # Freeze backbone layers if required
        if freeze_backbone:
            raise ValueError("Freeze backbone is not supported in this model.")

    def forward(self, images, **kwargs):
        assert images.dim() == 4, f"Input must be 5D tensor, got {images.shape}"
        assert images.size(1) == 1, f"Expected single channel input, got {images.size(2)} channels"
        images = images.repeat(1, 3, 1, 1)  # [B, 4, 3, H, W]
        B, C, H, W = images.size()

        # get image features and predict
        predict = self.model(images)

        return {'predict': predict}


def load_model(path):
    model = torch.load(path, map_location='cpu')
    if isinstance(model, dict):
        model = model['model']

    if isinstance(model, nn.DataParallel):
        model = model.module.cpu()

    try:
        if hasattr(model, '_model'):
            _model = model._model
        else:
            _model = model
    except:
        pass
    model = _model
    return model


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FineTune Mammo_AGE model')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output neurons for the new task')
    parser.add_argument('--pretrained_model_path',
                        default='/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/challenges/Kaggle_SPR/third_party/weights/best_convnext_fold_0.pth.tar',
                        type=str, help='First stage model')
    parser.add_argument('--freeze_backbone', default=False, type=bool, help='Freeze the backbone model')
    parser.add_argument('--dont_load_pretrained_weight', default=False, type=bool,
                        help='Do not load the pretrained weights')
    args = parser.parse_args()

    # Initialize the model
    model = FineTuneRSNA(
        pretrained_model_path=args.pretrained_model_path,
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone,
        dont_load_pretrained_weight=args.dont_load_pretrained_weight)

    # Example input
    shape = (1, 1, 224, 224)
    input_a = torch.rand(*shape)

    # Forward pass
    output = model(input_a)
    print(output)
