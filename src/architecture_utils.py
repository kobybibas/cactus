import torch.nn as nn
import torchvision.models as models


def get_backbone(is_pretrained: bool):
    # Define the backbone
    backbone = models.resnet18(pretrained=is_pretrained)
    out_feature_num = backbone.fc.in_features

    layers = list(backbone.children())[:-1]
    backbone = nn.Sequential(*layers)

    return backbone, out_feature_num


def get_classifier(in_features: int, num_target_classes: int):
    classifier = nn.Linear(in_features, num_target_classes)
    return classifier


def get_cf_predictor(num_filters: int, cf_vector_dim: int):
    # Define the cf vector predictor
    cf_layers = nn.Sequential(
        nn.BatchNorm1d(num_filters),
        nn.Linear(num_filters, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, cf_vector_dim),
    )
    return cf_layers
