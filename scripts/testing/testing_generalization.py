import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import warnings, argparse
warnings.filterwarnings('ignore')

from dfx import get_path
from dfx import backbone
from dfx import (
    call_saved_model,
    get_complete_model
)
from dfx import (
    dataset_for_generaization,
    make_binary,
    get_trans
)
from dfx import testing


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str)
    parser.add_argument('--model_type', type=str, default='complete', choices=['complete', 'backbone'])
    parser.add_argument('--tests', type=str, action='append', default=['inner_gan', 'outer_gan', 'io_gan', 'inner_dm', 'outer_dm', 'io_dm', 'inner_all', 'outer_all', 'all'])
    parser.add_argument('--backbone_path', type=str, default=None)
    parser.add_argument('--plot_cm', type=bool, default=False)
    parser.add_argument('--save_cm', type=bool, default=False)
    parser.add_argument('--average', type=str, default='binary', choices=['binary', 'micro', 'macro'])

    args = parser.parse_args()

    return args


def main(parser):
    models_dir = get_path('models')
    generalization_path = get_path('data_generalization')

    complete_model = get_complete_model(parser.backbone, models_dir=models_dir) if parser.model_type=='complete' else backbone(parser.backbone, finetuning=True, num_classes=3)
    if not parser.backbone_path==None:
        complete_model.load_state_dict(torch.load(parser.backbone_path))
    else:
        saved_backbone_name = call_saved_model(backbone_name=parser.backbone)
        complete_model.load_state_dict(torch.load(models_dir+'/complete_models/'+saved_backbone_name+'.pt'))

    trans = get_trans(model_name=parser.backbone)
    loss = nn.CrossEntropyLoss()

    for folder in parser.tests:
        print(f'-   {folder}')
        test = make_binary(dataset_for_generaization(dset_dir=generalization_path+f'/{folder}', transforms=trans))
        testload = DataLoader(test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

        loss = nn.CrossEntropyLoss()
        testing(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=parser.plot_cm, save_cm=parser.save_cm, average=parser.average, convert_to_binary=True, saving_dir='', model_name=parser.backbone)


if __name__=='__main__':
    parser = get_parser()
    main(parser)
