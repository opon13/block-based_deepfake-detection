# %%
import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse

from dfx import get_path
from dfx import (
    call_saved_model,
    get_complete_model
)
from dfx import (
    mydataset,
    check_len,
    make_train_valid,
    get_trans
)
from dfx import training

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--backbone', type=str)
    parser.add_argument('-logs', '--mode_logs', type=str, default='online')
    parser.add_argument('-batch', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-sch', '--scheduler', type=bool, default=False)
    parser.add_argument('-sch_step', '--scheduler_stepsize', type=int, default=10)
    parser.add_argument('-sch_g', '--scheduler_gamma', type=float, default=0.1)

    args = parser.parse_args()
    return args


def main(parser):

    datasets_path = get_path('dataset')
    guidance_path = get_path('guidance')
    models_dir = get_path('models')

    conf = f'-b{parser.batch_size}-lr{parser.learning_rate}-wd{parser.weight_decay}-step{parser.scheduler_stepsize}-gamma{parser.scheduler_gamma}'

    batch_size = parser.batch_size

    trans = get_trans(model_name=parser.backbone)
        
    dset = mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=False, transforms=trans)
    train, valid = make_train_valid(dset=dset, validation_ratio=0.2)

    perc_dm, perc_gan, perc_real = check_len(dset, binary=False, return_perc=True)

    trainload = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    validload = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    loss = nn.CrossEntropyLoss(weight=torch.tensor([1-perc_dm,1-perc_gan,1-perc_real]).to(dev))

    # %% training procedure
    backbone_name = parser.backbone
    saved_backbone = call_saved_model(backbone_name)

    print(f'\n-  {backbone_name}\n')
    model_complete = get_complete_model(backbone_name, models_dir=models_dir)

    optimizer = Adam(model_complete.parameters(), 
                    lr=parser.learning_rate, 
                    weight_decay=parser.weight_decay, 
                    betas=(0.9, 0.999))
    scheduler = StepLR(optimizer=optimizer, step_size=parser.scheduler_stepsize, gamma=parser.scheduler_gamma) if parser.scheduler else None

    training(model=model_complete,
            loaders={'train': trainload, 'valid': validload},
            epochs=parser.epochs,
            optimizer=optimizer,
            loss_fn=loss,
            scheduler=scheduler,
            mode_logs=parser.mode_logs,
            model_name=backbone_name,
            save_best_model=True,
            saving_path=models_dir+'/grid_complete_models/'+saved_backbone+conf+'.pt')
    

if __name__=='__main__':
    parser = get_parser()
    main(parser)