import torch
import torch.nn as nn
from .import_classifiers import *


class completenn(nn.Module):
    def __init__(self, model1, model2, model3):
        super(completenn, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        code1 = self.model1(x)
        code2 = self.model2(x)
        code3 = self.model3(x)
        x = torch.cat((code1.unsqueeze(1), code2.unsqueeze(1), code3.unsqueeze(1)), 1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model_families():
    model_families = {'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
                        'inception': ['googlenet', 'inception_v3'],
                        'resnet1': ['resnet18', 'resnet34', 'resnet50'],
                        'resnet2': ['resnet101', 'resnet152'],
                        'resnext': ['resnext101'],
                        'efficient' : ['efficientnet_b0', 'efficientnet_b4'],
                        'efficient_w': ['efficientnet_widese_b0', 'efficientnet_widese_b4'],
                        'vit_b': ['vit_b_16', 'vit_b_32'],
                        'vit_l': ['vit_l_16', 'vit_l_32']}
    return model_families

def call_saved_model(backbone_name:str):
    model_families = get_model_families()
    if not any(backbone_name in models for models in model_families.values()):
        print('backbone_name must belong to one of the following families: \n')
        for k, v in model_families.items():
            print(f'{k}: {v}')
    if backbone_name in model_families['densenet']: saved_model_name = 'dense'+backbone_name[-3:]
    if backbone_name in model_families['inception']: saved_model_name = 'googl'
    if backbone_name in model_families['resnet1']: saved_model_name = 'res'+backbone_name[-2:]
    if backbone_name in model_families['resnet2']: saved_model_name = 'res'+backbone_name[-3:]
    if backbone_name in model_families['resnext']: saved_model_name = 'resnxt101'
    if backbone_name in model_families['efficient']: saved_model_name = 'effb'+backbone_name[-1]
    if backbone_name in model_families['efficient_w']: saved_model_name = 'effwdb'+backbone_name[-1]
    if backbone_name in model_families['vit_b']: saved_model_name = 'vitb'+backbone_name[-2:]
    if backbone_name in model_families['vit_l']: saved_model_name = 'vitl'+backbone_name[-2:]
    return saved_model_name

def get_complete_model(backbone_name: str, models_dir:str):
    model_families = get_model_families()
    saved_model_name = call_saved_model(backbone_name=backbone_name)
    model_dm = backbone(backbone_name, pretrained=False, finetuning=True, num_classes=2)
    model_gan = backbone(backbone_name, pretrained=False, finetuning=True, num_classes=2)
    model_real = backbone(backbone_name, pretrained=False, finetuning=True, num_classes=2)
    model_dm.load_state_dict(torch.load(models_dir+'/bm-dm/'+saved_model_name+'.pt'))
    model_gan.load_state_dict(torch.load(models_dir+'/bm-gan/'+saved_model_name+'.pt'))
    model_real.load_state_dict(torch.load(models_dir+'/bm-real/'+saved_model_name+'.pt'))
    model_dm.eval()
    model_gan.eval()
    model_real.eval()
    if backbone_name in model_families['densenet']: 
        for model in [model_dm, model_gan, model_real]: model.classifier=nn.Identity()
    if backbone_name in model_families['inception']+model_families['resnet1']+model_families['resnet2']+model_families['resnext']: 
        for model in [model_dm, model_gan, model_real]: model.fc=nn.Identity()
    if backbone_name in model_families['efficient']+model_families['efficient_w']: 
        for model in [model_dm, model_gan, model_real]: model.classifier.fc=nn.Identity()
    if backbone_name in model_families['vit_b']+model_families['vit_l']: 
        for model in [model_dm, model_gan, model_real]: model.heads.head=nn.Identity()
    complete_model = completenn(model_dm, model_gan, model_real)
    for backbone_model in [complete_model.model1, complete_model.model2, complete_model.model3]:
        for param in backbone_model.parameters():
            param.requires_grad = False
    return complete_model