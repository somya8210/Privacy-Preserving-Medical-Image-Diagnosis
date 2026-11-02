import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def create_pretrained_model():
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    return model


def create_model_from_state(state, device):
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    model.load_state_dict(state)
    return model.to(device)
