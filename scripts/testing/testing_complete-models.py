import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings, argparse
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader

from dfx import get_path
from dfx import (
    call_saved_model,
    get_complete_model
)
from dfx import (
    mydataset,
    dataset_for_robustness,
    balance_test,
    balance_binary_test,
    get_trans
)
from dfx import testing

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--datasets_dir', type=str, default=None)
    parser.add_argument('--guidance_dir', type=str, default=None)
    parser.add_argument('--saving_dir', type=str, default=None)
    parser.add_argument('--robustness_dir', type=str, default=None)
    
    parser.add_argument('--classification_type', type=str, default='multi-class', choices=['binary', 'multi-class'])
    parser.add_argument('--plot_cm', type=bool, default=False)
    parser.add_argument('--save_cm', type=bool, default=False)
    parser.add_argument('--average', type=str, default='micro', choices=['binary', 'micro', 'macro'])
    parser.add_argument('--robustness_test', type=bool, default=False)
    parser.add_argument('--test_raw', type=bool, default=False)
    parser.add_argument('-rt', '--robustness_types', action='append', choices=['jpegQF90','jpegQF80','jpegQF70','jpegQF60','jpegQF50', 'GaussNoise-3', 'GaussNoise-7', \
                                                        'GaussNoise-15', 'mir-B', 'rot-45', 'rot-135', 'scaling-50', 'scaling-200'])

    args = parser.parse_args()
    return args


def main(parser):

    datasets_path = get_path('dataset') if parser.datasets_dir is not None else parser.datasets_dir
    guidance_path = get_path('guidance') if parser.guidance_dir is not None else parser.guidance_dir
    models_dir = get_path('models') if parser.saving_dir is not None else parser.saving_dir
    robustnessdset_path = get_path('data_robustness') if parser.robustness_dir is not None else parser.robustness_dir

    complete_model = get_complete_model(parser.backbone, models_dir=models_dir)
    if not parser.model_path==None:
        complete_model.load_state_dict(torch.load(parser.model_path))
    else:
        saved_backbone_name = call_saved_model(backbone_name=parser.backbone)
        complete_model.load_state_dict(torch.load(models_dir+'/complete/'+saved_backbone_name+'.pt'))

    trans = get_trans(model_name=parser.backbone)

    if parser.test_raw:
        print('-    RAW')
        if parser.classification_type=='binary':
            test = balance_binary_test(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=True, transforms=trans))
        else:
            test = balance_test(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=True, transforms=trans))
        testload = DataLoader(test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
        loss = nn.CrossEntropyLoss()
        testing(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=parser.plot_cm, save_cm=parser.save_cm, average=parser.average, convert_to_binary=True if parser.classification_type=='binary' else False, saving_dir='', model_name=parser.backbone)

    if parser.robustness_test:
        for testin in parser.robustness_types:
            print(f'-   {testin}')
            if parser.classification_type=='binary':
                test = balance_binary_test(dataset_for_robustness(dset_dir=robustnessdset_path+f'/testing_dset-{testin}', transforms=trans))
            else:
                test = balance_test(dataset_for_robustness(dset_dir=robustnessdset_path+f'/testing_dset-{testin}', transforms=trans))

            testload = DataLoader(test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
            loss = nn.CrossEntropyLoss()

            testing(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=parser.plot_cm, save_cm=parser.save_cm, average=parser.average, convert_to_binary=True if parser.classification_type=='binary' else False, saving_dir='', model_name=parser.backbone)


if __name__=='__main__':
    parser = get_parser()
    main(parser)