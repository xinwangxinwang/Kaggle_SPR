"""
Fine-tuned model based on Mammo_AGE with a new classification head.
"""

import os
import json
import torch
import torch.nn as nn
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_dir)
from mammo_age.models.mammo_age_model import Mammo_AGE  # Ensure this is the correct import for your base model

class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()
        self.hazard_fc = nn.Linear(num_features,  max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triagular_mask', mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size() #hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T) #expanded_hazards is (B,T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob


class FineTuneMammoAGE(nn.Module):
    def __init__(self, pretrained_model_path: str = None,
                 num_classes: int = 2,
                 freeze_backbone: bool = True,
                 dont_load_pretrained_weight: bool = False,):
        """
        Fine-tuned model based on Mammo_AGE with a new classification head.

        Args:
            pretrained_model_path (str): Path to the pretrained model weights (optional).
            num_classes (int): Number of classes for the classification task (default: 2 for binary classification).
            freeze_backbone (bool): Whether to freeze the backbone parameters.
        """
        super(FineTuneMammoAGE, self).__init__()

        # Load pretrained weights if a path is provided
        if pretrained_model_path:
            # Load the json file for the model configuration
            with open(os.path.join(os.path.dirname(pretrained_model_path), 'args.json'), 'r') as f:
                _cfg = json.load(f)

            self.backbone = Mammo_AGE(_cfg['arch'], _cfg['num_output_neurons'], _cfg['nblock'],
                                      _cfg['hidden_size'], _cfg['second_stage'], _cfg['first_stage_model'])

            if not dont_load_pretrained_weight:
                # Load the pretrained model weights
                print(f"Loading pretrained model weights from: {pretrained_model_path}")
                state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))['state_dict']
                self.backbone.load_state_dict(state_dict, strict=True)  # Allow partial loading due to head mismatch
            else:
                print("Not loading pretrained weights, training from scratch.")
        else:
            raise ValueError("Pretrained model path is required for fine-tuning.")

        # Freeze backbone layers if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_feat = self.backbone.backbone_model.num_feat
        hidden_size = _cfg['hidden_size']
        in_feat = num_feat * 5 + hidden_size * 4

        self.mlp = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(in_feat, num_feat), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(num_feat, hidden_size), nn.ReLU(),
            nn.Dropout(0.1),)

        self.risk_layer = Cumulative_Probability_Layer(hidden_size, num_classes)

    def forward(self, x, max_t=50, use_sto=False):
        # Forward pass through the backbone
        (_, _, global_emb, local_emb, _, _, _) = self.backbone(x, max_t, use_sto)
        # Concatenate the global and local embeddings
        combined_emb = torch.cat([global_emb] + local_emb, dim=1)
        # Forward pass through the new classifier
        predict = self.risk_layer(self.mlp(combined_emb))
        return {'predict': predict}


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
    model = FineTuneMammoAGE(
        pretrained_model_path=args.pretrained_model_path,
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone,
        dont_load_pretrained_weight=args.dont_load_pretrained_weight
    )

    # Example input
    shape = (3, 4, 1, 512, 512)
    input_a = torch.rand(*shape)

    # Forward pass
    output = model(input_a, max_t=50, use_sto=False)
    print(output)