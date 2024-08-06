import torch
import torchvision
import torch.nn as nn

def backbone(name: str,
            pretrained: bool = False,
            finetuning: bool = False,
            num_classes: int = 10):
  
    model_families = {'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
                      'inception': ['googlenet', 'inception_v3'],
                      'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                      'resnext': ['resnext101'],
                      'efficient' : ['efficientnet_b0', 'efficientnet_b4', 'efficientnet_widese_b0', 'efficientnet_widese_b4'],
                      'vit': ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']}

    if not any(name in models for models in model_families.values()):
        print('name must belong to one of the following families: \n')
        for k, v in model_families.items():
            print(f'{k}: {v}')

    if name in model_families['densenet']:
        model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=pretrained)
        if finetuning!=False:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif name in model_families['inception']:
        model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=pretrained)
        if finetuning!=False:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    elif name in model_families['resnet']:
        model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=pretrained)
        if finetuning!=False:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    elif name=='resnext101':
        model=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resneXt')
        if finetuning!=False:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    elif name in model_families['efficient']:
        model=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_'+name, pretrained=pretrained)
        if finetuning!=False:
            in_features = model.classifier.fc.in_features
            model.classifier.fc = nn.Linear(in_features, num_classes)
    elif name in model_families['vit']:
        if name=='vit_b_16': model = torchvision.models.vit_b_16(weights='DEFAULT')
        if name=='vit_b_32': model = torchvision.models.vit_b_32(weights='DEFAULT')
        if name=='vit_l_16': model = torchvision.models.vit_l_16(weights='DEFAULT')
        if name=='vit_l_32': model = torchvision.models.vit_l_16(weights='DEFAULT')
        if name=='vit_h_14': model = torchvision.models.vit_h_14(weights='DEFAULT')
        if finetuning!=False:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
    return model