import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from utils import AverageMeter
from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
import numpy as np
from utils import *
from sklearn.metrics import mean_absolute_error
from functools import partial
from torch.utils.data import DataLoader
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import RemoveMean
from ema import EMAHelper

import warnings
warnings.filterwarnings("ignore")

def val(model, dataloader, device):
    model.eval()

    pred_energy_list = []
    pred_force_list = []
    label_energy_list = []
    label_force_list = []
    fixed_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred_energy, pred_force = model(data)
            label_energy, label_force, fixed = data.y, data.force, data.fixed

            pred_energy_list.append(pred_energy.detach().cpu().numpy())
            label_energy_list.append(label_energy.detach().cpu().numpy())

            pred_force_list.append(pred_force.detach().cpu().numpy())
            label_force_list.append(label_force.detach().cpu().numpy())

            fixed_list.append(fixed.detach().cpu().numpy())

    pred_energy = np.concatenate(pred_energy_list, axis=0)
    label_energy = np.concatenate(label_energy_list, axis=0)

    pred_force = np.concatenate(pred_force_list, axis=0)
    label_force = np.concatenate(label_force_list, axis=0)

    fixed = np.concatenate(fixed_list, axis=0)
    mask = (fixed == 0)
    pred_force = pred_force[mask]
    label_force = label_force[mask]

    mae_energy = mean_absolute_error(pred_energy, label_energy)
    mae_force = mean_absolute_error(pred_force, label_force)

    model.train()

    return mae_energy, mae_force

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--data_type', type=str, choices=['50K', '200K'], default='50K', help='type of data')
    parser.add_argument('--model_type', type=str, choices=['E2GNN', 'vanilla', 'vanilla_nmu'], default='E2GNN', help='type of data')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--save_model', type=bool, default=True)

    args = parser.parse_args()
    data_root = args.data_root
    data_type = args.data_type
    model_type = args.model_type
    num_workers = args.num_workers
    batch_size = args.batch_size
    save_model = args.save_model

    if data_type == '50K':
        train_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train'), 'split':True, 'mode':'train', 'total_percentage':0.25, 'train_percentage':0.8, 'seed':123})
        valid_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train'), 'split':True, 'mode':'val', 'total_percentage':0.25, 'train_percentage':0.8, 'seed':123})
    else:
        train_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train'), 'split':True, 'mode':'train', 'total_percentage':1., 'train_percentage':0.8, 'seed':234})
        valid_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train'), 'split':True, 'mode':'val', 'total_percentage':1., 'train_percentage':0.8, 'seed':234})

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, otf_graph=False), num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, otf_graph=False), num_workers=num_workers)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f'{model_type}_{timestamp}'
    # name is the name displayed in UI
    # id is the name displayed in local dir
    wandb.init(project=f"E2GNN_OC20_{data_type}", 
            config={"train_len" : len(train_set), "valid_len" : len(valid_set)}, 
            name=log_name,
            id=log_name
            )

    device = torch.device('cuda:0')

    if model_type == 'vanilla':
        from vanilla import E2GNN
        model = E2GNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=6.0, max_neighbors=20, use_pbc=True, otf_graph=False, num_elements=118).to(device)
    elif model_type == 'vanilla_nmu':
        from vanilla_nmu import E2GNN
        model = E2GNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=6.0, max_neighbors=20, use_pbc=True, otf_graph=False, num_elements=118).to(device)
    else:
        from E2GNN import E2GNN
        model = E2GNN(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=6.0, max_neighbors=20, use_pbc=True, otf_graph=False, num_elements=118).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.8, patience = 5, min_lr = 1.e-6)
    criterion = nn.L1Loss()

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)

    running_loss = AverageMeter()
    running_loss_energy = AverageMeter()
    running_loss_force = AverageMeter()
    running_grad_norm = AverageMeter()
    running_best_mae = BestMeter("min")
    
    epochs = 2000
    if data_type == '50K':
        steps_per_epoch = 2000
    else:
        steps_per_epoch = 5000
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    global_step = 0
    global_epoch = 0
    early_stop_epoch = 20

    break_flag = False

    print("Start training")
    model.train()
    for epoch in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1    

            data = data.to(device)
            pred_energy, pred_force = model(data)
            label_energy, label_force, fixed = data.y, data.force, data.fixed

            mask = (fixed == 0)
            pred_force = pred_force[mask]
            label_force = label_force[mask]

            loss_force = 50 * criterion(pred_force, label_force)
            loss_energy = criterion(pred_energy, label_energy)
            loss = loss_force + loss_energy 
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=50,
            )
            optimizer.step()

            ema_helper.update(model)

            running_loss.update(loss.item()) 
            running_loss_force.update(loss_force.item(), label_force.size(0)) 
            running_loss_energy.update(loss_energy.item(), label_energy.size(0)) 
            running_grad_norm.update(grad_norm.item())

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                train_loss = running_loss.get_average()
                train_loss_force = running_loss_force.get_average()
                train_loss_energy = running_loss_energy.get_average()
                train_grad_norm = running_grad_norm.get_average()

                running_loss.reset()
                running_loss_force.reset()
                running_loss_energy.reset()
                running_grad_norm.reset()

                valid_mae_energy, valid_mae_force = val(ema_helper.ema_copy(model), valid_loader, device)
                scheduler.step(valid_mae_force)
                current_lr = optimizer.param_groups[0]['lr']

                log_dict = {
                    'train/epoch':global_epoch,
                    'train/loss':train_loss,
                    'train/lr': current_lr,
                    'train/grad_norm' : train_grad_norm,
                    'val/energy_mae':valid_mae_energy,
                    'val/forces_mae':valid_mae_force
                }
                wandb.log(log_dict)

                if valid_mae_force < running_best_mae.get_best():
                    running_best_mae.update(valid_mae_force)
                    if save_model:
                        msg = "epoch-%d, train_loss-%.4f, valid_mae_energy-%.4f, valid_mae_force-%.4f" \
                        % (global_epoch, train_loss, valid_mae_energy, valid_mae_force)
                        print(msg)
                        torch.save(ema_helper.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
                else:
                    count = running_best_mae.counter()
                    if count > early_stop_epoch:
                        best_force_mae = running_best_mae.get_best()
                        print(f"early stop in epoch {epoch}")
                        print({'best_force_mae':best_force_mae})
                        break_flag = True
                        break
                    
    wandb.finish()
# %%

