# Adjusted from https://github.com/batmanlab/Mammo-CLIP
import os
import logging
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_dir)
from third_party.mammo_clip.efficientnet_custom import EfficientNet


class FineTuneMammoClip(nn.Module):
    def __init__(self,
                 pretrained_model_path: str = None,
                 num_classes: int = 2,
                 freeze_backbone: bool = True,
                 dont_load_pretrained_weight: bool = False, ):
        """
        Fine-tuned model based on Mammo-Clip with a new classification head.
        Args:
            pretrained_model_path (str): Path to the pretrained model weights (optional).
            num_classes (int): Number of classes for the classification task (default: 2 for binary classification).
            freeze_backbone (bool): Whether to freeze the backbone parameters.
            dont_load_pretrained_weight (bool): Whether to load the pretrained weights.
        """

        super(FineTuneMammoClip, self).__init__()
        if pretrained_model_path:
            ckpt = torch.load(pretrained_model_path, map_location="cpu")
            self.config = ckpt["config"]["model"]["image_encoder"]
            self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
            if not dont_load_pretrained_weight:
                image_encoder_weights = {}
                for k in ckpt["model"].keys():
                    if k.startswith("image_encoder."):
                        image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
                logging.info('Loading image encoder weights from checkpoint')
                self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
            else:
                logging.info('Not loading image encoder weights from checkpoint')
                self.image_encoder.load_state_dict(ckpt["model"], strict=False)
        else:
            raise ValueError("Pretrained model path is required for fine-tuning.")

        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]

        # Freeze backbone layers if required
        if freeze_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        in_feat = self.image_encoder.out_dim
        num_feat = self.image_encoder.out_dim
        hidden_size = 512

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_feat, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes))

        self.raw_features = None
        self.pool_features = None

    def get_image_encoder_type(self):
        return self.image_encoder_type

    def encode_image(self, image):
        if self.image_encoder_type == "cnn":
            if self.config["name"].lower() == "resnet152" or self.config["name"].lower() == "resnet101":
                image_features = self.image_encoder(image)
                return image_features
            else:
                input_dict = {"image": image, "breast_clip_train_mode": True}
                image_features, raw_features = self.image_encoder(input_dict)
                self.raw_features = raw_features
                self.pool_features = image_features
                return image_features
        else:
            image_features = self.image_encoder(image)
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features

    def forward(self, images, **kwargs):
        assert images.dim() == 4, f"Input must be 5D tensor, got {images.shape}"
        assert images.size(1) == 1, f"Expected single channel input, got {images.size(2)} channels"
        images = images.repeat(1, 3, 1, 1)  # [B, 4, 3, H, W]
        B, C, H, W = images.size()

        # get image features and predict
        image_feature = self.encode_image(images)

        # concat the features from all views
        image_feature = image_feature.view(B, -1)

        predict = self.classifier(image_feature)
        return {'predict': predict}


def load_image_encoder(config_image_encoder: Dict):
    if (
            config_image_encoder["source"].lower() == "cnn" and
            config_image_encoder["name"].lower() == "tf_efficientnetv2-detect"
    ):
        _image_encoder = EfficientNet.from_pretrained("efficientnet-b2", num_classes=1)
        _image_encoder.out_dim = 1408
    elif (
            config_image_encoder["source"].lower() == "cnn" and
            config_image_encoder["name"].lower() == "tf_efficientnet_b5_ns-detect"
    ):
        _image_encoder = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
        _image_encoder.out_dim = 2048
    else:
        raise KeyError(f"Not supported image encoder: {config_image_encoder}")
    return _image_encoder


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class):
        super().__init__()
        self.classification_head = nn.Linear(feature_dim, num_class)

    def forward(self, x):
        return self.classification_head(x)
    
    
class MultiTaskMammoClip(FineTuneMammoClip):
    def __init__(self,
                 pretrained_model_path: str = None,
                 num_classes: int = 2,
                 freeze_backbone: bool = True,
                 dont_load_pretrained_weight: bool = False,
                 auxiliary_task: dict = None):
        """
        Multi-task version of FineTuneMIRAI with optional auxiliary tasks.

        Args:
            auxiliary_task (dict): Optional dict like {'age': 1, 'birads': 7, 'density': 4}
        """

        super(MultiTaskMammoClip, self).__init__(pretrained_model_path,
                                                 num_classes,
                                                 freeze_backbone,
                                                 dont_load_pretrained_weight)
        
        self.auxiliary_task_heads = nn.ModuleDict()
        self.auxiliary_task = auxiliary_task or {}

        for task_name, task_classes in self.auxiliary_task.items():
            self.auxiliary_task_heads[task_name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.classifier[1].in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, task_classes)
            )

        # Update main classifier input size to include auxiliary predictions
        aux_total_dim = sum([v for v in self.auxiliary_task.values()])
        in_feat = self.classifier[1].in_features

        hidden_size, out_hidden_size = 512, 32
        self.hidden_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_feat, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, out_hidden_size)
        )

        in_feat_final = out_hidden_size + aux_total_dim
        self.classifier = nn.Linear(in_feat_final, num_classes)

    def forward(self, images, **kwargs):
        assert images.dim() == 4, f"Input must be 5D tensor, got {images.shape}"
        assert images.size(1) == 1, f"Expected single channel input, got {images.size(2)} channels"
        images = images.repeat(1, 3, 1, 1)  # [B, 4, 3, H, W]
        B, C, H, W = images.size()

        # get image features and predict
        image_feature = self.encode_image(images)

        # concat the features from all views
        image_feature = image_feature.view(B, -1)

        aux_preds = {}
        aux_features = []
        losses = {}

        auxiliary_targets = kwargs.get('auxiliary_targets', None)

        for task_name, head in self.auxiliary_task_heads.items():
            task_pred = head(image_feature)
            if self.auxiliary_task[task_name] == 1:
                task_pred = torch.sigmoid(task_pred)
            else:
                task_pred = F.softmax(task_pred, dim=-1)
            aux_preds[task_name] = task_pred
            aux_features.append(task_pred)

            if auxiliary_targets and task_name in auxiliary_targets:
                loss = self.aux_loss_compute(task_pred, auxiliary_targets[task_name], task_name)
                if loss is not None:
                    losses[f'loss_{task_name}'] = loss

        image_feature = self.hidden_layer(image_feature)
        if aux_features:
            aux_concat = torch.cat(aux_features, dim=1)
            enhanced_feature = torch.cat([image_feature, aux_concat], dim=1)
        else:
            enhanced_feature = image_feature  # No aux task present

        main_pred = self.classifier(enhanced_feature)

        # Total auxiliary loss
        total_loss = sum(losses.values()) if losses else None

        return {
            'predict': main_pred,
            **aux_preds,
            'losses': losses,
            'loss': total_loss  # total of all auxiliary losses
        }

    def aux_loss_compute(self, preds, labels, task_name):
        """
        Compute auxiliary task loss for available (non-missing) labels.

        Args:
            preds (Tensor): Predictions from auxiliary head. Shape: [B, num_classes] or [B, 1] for regression.
            labels (Tensor): Ground truth labels. Shape: [B]
            task_name (str): Name of the task (e.g., 'age', 'birads', etc.)

        Returns:
            loss (Tensor): Computed loss on valid samples, or None if all labels are missing.
        """
        # Create a mask for valid labels
        valid_mask = labels != -1
        if valid_mask.sum() == 0:
            return None  # All labels are missing, skip loss

        # Age is regression
        if task_name == 'age':
            preds = preds.squeeze()  # [B]
            loss_fn = nn.L1Loss()
            try:
                return loss_fn(preds[valid_mask], labels[valid_mask].float() / 100) * 10
            except Exception as e:
                logging.info(f"Error in age loss computation: {e}")
                return None

        # Classification (BIRADS, density, etc.)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(preds[valid_mask], labels[valid_mask].long()) * 3e-1


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FineTune Mammo_AGE model')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output neurons for the new task')
    parser.add_argument('--pretrained_model_path',
                        default='/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/proj-risk-foundation/others_proj/Mammo-CLIP/weights/efficientnet_b2.tar',
                        type=str, help='First stage model')
    parser.add_argument('--freeze_backbone', default=True, type=bool, help='Freeze the backbone model')
    parser.add_argument('--dont_load_pretrained_weight', default=False, type=bool,
                        help='Do not load the pretrained weights')
    args = parser.parse_args()


    # # Initialize the model
    # model = FineTuneMammoClip(
    #     pretrained_model_path=args.pretrained_model_path,
    #     num_classes=args.num_classes,
    #     freeze_backbone=args.freeze_backbone,
    #     dont_load_pretrained_weight=args.dont_load_pretrained_weight)
    #
    # # Example input
    # shape = (1, 1, 224, 224)
    # input_a = torch.rand(*shape)
    #
    # # Forward pass
    # output = model(input_a)
    # print(output)

    # # Demo for MultiTaskMIRAI
    # Simulate auxiliary task configuration
    aux_tasks = {'age': 1, 'birads': 7, 'density': 4}

    # Initialize the model
    model = MultiTaskMammoClip(
        pretrained_model_path=args.pretrained_model_path,
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone,
        dont_load_pretrained_weight=args.dont_load_pretrained_weight,
        auxiliary_task=aux_tasks)

    print(model)

    # Simulate input (B=4, C=1, H=224, W=224)
    inputs = torch.rand(4, 1, 224, 224)

    # Simulate targets
    targets = {
        'age': torch.tensor([45.0, -1.0, 50.0, 60.0]),  # One missing label
        'birads': torch.tensor([2, -1, 1, 4]),  # One missing
        'density': torch.tensor([1, 2, -1, 3])  # One missing
        # Note: main task label (e.g. 'label') is not needed
    }

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(inputs, auxiliary_targets=targets)

    print("\n=== Predictions ===")
    for k, v in output.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                print(f"{sub_k}: {sub_v}")
        elif isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape} → {v}")
        else:
            print(f"{k}: {v}")

