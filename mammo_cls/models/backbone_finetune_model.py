"""
The script defines a new model that uses the pre-trained backbone from Mammo_AGE and adds a new classification head.
The new model is then initialized with the pre-trained weights and the backbone is frozen if required.
"""

import os
import json
import torch
import torch.nn as nn
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_dir)
from mammo_age.models.mammo_age_model import Mammo_AGE  # Ensure this is the correct import for your base model

# get Mammo_AGE pre-trained backbone encoder model (e.g., resnet18, resnet50, etc.)
class PretrainedBackbone(nn.Module):
    def __init__(self, pretrained_model_path: str = None,
                 freeze_backbone: bool = True,
                 dont_load_pretrained_weight: bool = False,):
        """
        Pretrained backbone model based on Mammo_AGE.

        Args:
            pretrained_model_path (str): Path to the pretrained model weights.
        """
        super(PretrainedBackbone, self).__init__()

        # Load pretrained weights
        if pretrained_model_path:
            # Load the json file for the model configuration
            with open(os.path.join(os.path.dirname(pretrained_model_path), 'args.json'), 'r') as f:
                _cfg = json.load(f)

            pretrained_Mammo_AGE = Mammo_AGE(_cfg['arch'], _cfg['num_output_neurons'], _cfg['nblock'],
                                      _cfg['hidden_size'], _cfg['second_stage'], _cfg['first_stage_model'])

            if not dont_load_pretrained_weight:
                # Load the pretrained model weights
                print(f"Loading pretrained model weights from: {pretrained_model_path}")
                state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))['state_dict']
                pretrained_Mammo_AGE.load_state_dict(state_dict, strict=True)
            else:
                print("Not loading pretrained weights, training from scratch.")
        else:
            raise ValueError("Pretrained model path is required.")

        self.num_feat = pretrained_Mammo_AGE.backbone_model.num_feat
        self.backbone = pretrained_Mammo_AGE.backbone_model.model

        # Freeze backbone layers if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the backbone
        return self.backbone(x)


class DownstreamModel(nn.Module):
    def __init__(self,
                 pretrained_model_path: str = None,
                 freeze_backbone: bool = True,
                 num_classes: int = 2,
                 dont_load_pretrained_weight: bool = False,):
        """
        Downstream model with a new classification head.

        Args:
            num_classes (int): Number of classes for the classification task (default: 2 for binary classification).
            num_feat (int): Number of features from the backbone model.
            hidden_size (int): Hidden size for the classification head.
        """
        super(DownstreamModel, self).__init__()

        # Load pretrained backbone model
        self.backbone = PretrainedBackbone(pretrained_model_path, freeze_backbone, dont_load_pretrained_weight)
        self.num_feat = self.backbone.num_feat

        # Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # New classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.num_feat, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = torch.cat([x, x, x], dim=1) # Concat x to 3 channels
        x = self.backbone(x) # Forward pass through the backbone
        x = self.avgpool(x) # Pooling layer
        x = torch.flatten(x, 1) # Flatten the output
        x = self.classifier(x) # Classification head
        return x


# Example usage
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FineTune Mammo_AGE model')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output neurons for the new task')
    parser.add_argument('--pretrained_model_path', default='/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/Mammo-AGE/logs_NC_rebuttal/add_acr2/VinDr_RSNA_Inhouse/resnet18/1024_fold0/model_best.pth.tar', type=str, help='First stage model')
    parser.add_argument('--freeze_backbone', default=True, type=bool, help='Freeze the backbone model')
    parser.add_argument('--dont_load_pretrained_weight', default=False, type=bool, help='Do not load the pretrained weights')
    args = parser.parse_args()

    # Initialize the model
    model = DownstreamModel(
        pretrained_model_path=args.pretrained_model_path,
        freeze_backbone=args.freeze_backbone,
        num_classes=args.num_classes,
        dont_load_pretrained_weight=args.dont_load_pretrained_weight
    )

    print(model)

    # Example input
    shape = (3, 1, 512, 512)
    input_a = torch.rand(*shape)

    # Forward pass
    output = model(input_a)
    print(output)