import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings
import argparse
warnings.filterwarnings('ignore')
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dfx import get_path
from dfx import ( 
    mydataset, 
    get_trans, 
    balance_binary_test, 
    make_train_valid,
    check_len
)
from dfx import (
    backbone,
    training,
    testing,
)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline', type=str, action='append')
    parser.add_argument('--mode_logs', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--scheduler_stepsize', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)

    args = parser.parse_args()

    return args


def main(parser):

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets_path = get_path('dataset')
    guidance_path = get_path('guidance')
    models_dir = get_path('models')

    for model_name in parser.baselines:
        trans = get_trans(model_name=model_name)

        dset = mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=False, transforms=trans)
        train, valid = make_train_valid(dset=dset, validation_ratio=0.2)

        perc_dm, perc_gan, perc_real = check_len(dset, binary=False, return_perc=True)

        trainload = DataLoader(train, batch_size=parser.batch_size, shuffle=True, num_workers=0, drop_last=False)
        validload = DataLoader(valid, batch_size=parser.batch_size, shuffle=True, num_workers=0, drop_last=False)

        test = balance_binary_test(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=True, transforms=trans))
        testload = DataLoader(test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

        loss = nn.CrossEntropyLoss(weight=torch.tensor([1-perc_dm,1-perc_gan,1-perc_real]).to(dev))

        model = backbone(model_name, pretrained=True, finetuning=True, num_classes=3)
        optimizer = Adam(model.parameters(), 
                        lr=parser.learning_rate, 
                        weight_decay=parser.weight_decay, 
                        betas=(0.9, 0.999))
        scheduler = StepLR(optimizer=optimizer, step_size=parser.scheduler_stepsize, gamma=parser.scheduler_gamma)

        training(model=model,
                loaders={'train': trainload, 'valid': validload},
                epochs=parser.epochs,
                optimizer=optimizer,
                loss_fn=loss,
                scheduler=scheduler,
                logs=parser.mode_logs,
                model_name=model_name+'-baseline',
                save_best_model=True,
                saving_path=models_dir+'/baselines/'+model_name+'.pt')
        testing(model=model, test_loader=testload, loss_fn=nn.CrossEntropyLoss(), plot_cm=True, average='micro')

if __name__=='__main__':
    parser = get_parser()
    main(parser)