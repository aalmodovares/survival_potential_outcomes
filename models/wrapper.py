# file to store all model wrappers
from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple, Any, Optional, Dict

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from .utils import ClippedAdam, ClippedAdamW


class Wrapper(nn.Module):
    def __init__(self, name, device, experiment, model_path, save=True, optim_config=None, wandblog=False, disable_tqdm=False):
        super(Wrapper, self).__init__()
        '''
            name: str, Name of the model
            device: torch.device,
            experiment: str, Name of the experiment, only for saving, logging and loading
            model_path: str, Name of the path for saving the models
            save: bool,
            optim_config: Optional[Dict[str, Any]],
            wandblog: bool,
            disable_tqdm: bool
        '''
        self.name = name
        self.device = device
        self.model_path = model_path
        self.experiment = experiment
        self.save = save
        self.optim_config = optim_config
        self.wandblog = wandblog
        self.current_epoch = -1
        self.disable_tqdm = disable_tqdm

    def seed_everything(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_optim(self):
        if self.optim_config is None:
            self.optim_config = {'optimizer': 'Adam', 'lr': 0.005, 'weight_decay': 1e-5,
                            'scheduler': None,
                            'scheduler_config': {},
                            'early_stopping': {'activate': False, 'patience': 50, 'metric':None}}
        else:
            self.optim_config = self.optim_config
        if 'optimizer' not in self.optim_config:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        elif self.optim_config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.optim_config['lr'], weight_decay=self.optim_config['weight_decay'])
        elif self.optim_config['optimizer'] == 'ClippedAdam' or self.optim_config['optimizer'] == 'clipped-adam':
            self.optimizer = ClippedAdam(self.parameters(), lr=self.optim_config['lr'], weight_decay=self.optim_config['weight_decay'], clip_value=self.optim_config['clip_norm'])
        elif self.optim_config['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(self.parameters(), lr=self.optim_config['lr'], weight_decay=self.optim_config['weight_decay'])
        elif self.optim_config['optimizer'] == 'ClippedAdamW':
            self.optimizer = ClippedAdamW(self.parameters(), lr=self.optim_config['lr'], weight_decay=self.optim_config['weight_decay'], clip_value=self.optim_config['clip_norm'])
        else:
            raise Exception('Optimizer not implemented')
        if 'scheduler' in self.optim_config:
            if self.optim_config['scheduler'] == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.optim_config['scheduler_config'])
            elif self.optim_config['scheduler'] == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **self.optim_config['scheduler_config'])
            elif self.optim_config['scheduler'] == 'linear':
                self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, **self.optim_config['scheduler_config'])
            elif self.optim_config['scheduler'] == 'exponential':
                self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, **self.optim_config['scheduler_config'])
            elif self.optim_config['scheduler']:
                raise Exception('Scheduler not implemented')

    def forward(self, x_dict):
        x = x_dict['x']
        return self.net(x)

    @abstractmethod
    def loss(self, output, true_values:dict):
        pass

    def true_values(self, x, y, t=None, c=None, mask=None):
        if mask is None:
            # check dimension of c is one, otherwise it is mask
            if c is not None and len(c.shape) >1 and c.shape[1]>1:
                mask = c
                c = None
        return {'x': x, 'y': y, 't': t, 'c': c, 'mask': mask}

    def get_dataloader(self, X, y, t=None, c=None, mask=None, batch_size=256):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)
        data_tuple = (X, y)
        if t is not None:
            data_tuple = data_tuple + (torch.tensor(t, dtype=torch.float).to(self.device),)
        if c is not None:
            data_tuple = data_tuple + (torch.tensor(c, dtype=torch.float).to(self.device),)
        if mask is not None:
            data_tuple = data_tuple + (torch.tensor(mask, dtype=torch.float).to(self.device),)
        dataset = torch.utils.data.TensorDataset(*data_tuple)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def fit(self, X_train, y_train, t_train = None, c_train = None, mask_train=None,
                  X_val=None, y_val=None, t_val=None, c_val=None, mask_val=None,
                  epochs=1000, batch_size=256):
        # dataloader
        train_dataloader = self.get_dataloader(X=X_train, y=y_train, t=t_train, c=c_train, mask=mask_train, batch_size=batch_size)

        self.on_training_start()

        if X_val is not None:
            val_dataloader = self.get_dataloader(X=X_val, y=y_val, t=t_val, c=c_val, mask=mask_val, batch_size=batch_size)
        best_loss = 1e10
        train_loss_hist, val_loss_hist, best_loss_hist = {}, {}, {}
        patience_counter = 0
        self.current_epoch +=1
        progress_bar = tqdm(range(self.current_epoch, self.current_epoch+epochs), leave=False, disable=self.disable_tqdm)
        for epoch in progress_bar:
            self.current_epoch = epoch
            train_loss = 0
            train_loss_dict = {}
            val_loss = 0
            val_loss_dict = {}

            self.train()
            for i, train_tuple in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                train_dict = self.true_values(*train_tuple)

                if i == 0:
                    # line below for KAN implementations
                    self.on_epoch_start(epochs, train_dict)

                output = self(train_dict)
                loss_dict = self.loss(output, train_dict)
                loss = loss_dict['loss']
                loss += self.compute_regularization()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                if len(train_loss_dict) == 0:
                    train_loss_dict = {key: value.item() for key, value in loss_dict.items()}
                else:
                    train_loss_dict = {key: train_loss_dict[key] + value.item() for key, value in loss_dict.items()}
            train_loss = train_loss / len(train_dataloader)
            train_loss_dict = {key: value/len(train_dataloader) for key, value in train_loss_dict.items()}
            if self.wandblog:
                wandb.log({'train': train_loss_dict, 'Train Loss': train_loss}, step=self.current_epoch)

            dict_postfix = {'Epoch': self.current_epoch, 'Train Loss': train_loss}
            # progress_bar.set_postfix({'Epoch':self.current_epoch, 'Train Loss': train_loss})

            # ——— record into history ———
            for k, v in train_loss_dict.items():
                train_loss_hist.setdefault(k, []).append(v)

            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    for val_tuple in val_dataloader:
                        val_dict = self.true_values(*val_tuple)
                        output = self(val_dict)
                        loss_dict = self.loss(output, val_dict)
                        loss = loss_dict['loss']
                        val_loss += loss.item()
                        if len(val_loss_dict) == 0:
                            val_loss_dict = {key: value.item() for key, value in loss_dict.items()}
                        else:
                            val_loss_dict = {key: val_loss_dict[key] + value.item() for key, value in loss_dict.items()}
                val_loss = val_loss / len(val_dataloader)
                val_loss_dict = {key: value/len(val_dataloader) for key, value in val_loss_dict.items()}
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(val_loss)
                    if self.wandblog:
                        wandb.log({'lr': self.optimizer.param_groups[0]['lr']}, step=self.current_epoch)
                    dict_postfix['lr'] = self.optimizer.param_groups[0]['lr']
                if self.wandblog:
                    wandb.log({'val': val_loss_dict, 'Val Loss': val_loss}, step=self.current_epoch)
                # progress_bar.set_postfix({'Epoch': self.current_epoch, 'Train Loss': train_loss, 'Val Loss': val_loss})
                dict_postfix['Val Loss'] = val_loss
                # record into history
                for k, v in val_loss_dict.items():
                    val_loss_hist.setdefault(k, []).append(v)
                # implement early stopping
                if self.optim_config['early_stopping']['activate']:
                    if self.optim_config['early_stopping']['metric'] is not None:
                        metric_loss = val_loss_dict[self.optim_config['early_stopping']['metric']]
                    else:
                        metric_loss = val_loss
                    if metric_loss < best_loss:
                        best_loss = metric_loss
                        best_loss_dict = val_loss_dict
                        patience_counter = 0
                        if self.save:
                            # save model
                            # torch.save(self.state_dict(), f'{self.model_path}/best_model_{self.experiment}.pth')
                            torch.save(self, f'{self.model_path}/best_model_{self.experiment}.pth')
                    else:
                        patience_counter += 1
                        if (patience_counter > self.optim_config['early_stopping']['patience']):
                            print('Early stopping at epoch:', self.current_epoch)
                            break


            else:
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(train_loss)
                    if self.wandblog:
                        wandb.log({'lr': self.optimizer.param_groups[0]['lr']}, step=self.current_epoch)
            if self.save:
                torch.save(self, f'{self.model_path}/last_model_{self.experiment}.pth')
                # todo: save KAN
            progress_bar.set_postfix(dict_postfix)
        self.on_epoch_end()

        if best_loss_hist:
            return best_loss_hist
        elif val_loss_hist:
            return val_loss_hist
        else:
            return train_loss_hist

    @abstractmethod
    def predict(self, X_test, load_model: str):
        pass

    def compute_regularization(self):
        return torch.tensor(0.)
    def on_epoch_start(self, n_epochs, train_dict):
        pass
    def on_epoch_end(self):
        pass
    def on_training_start(self):
        pass

    def load(self, load_model: str = 'best'):
        if self.save:
            if load_model=='best':
                try:
                    print('Loading best...')
                    model = torch.load(f'{self.model_path}/best_model_{self.experiment}.pth', weights_only=False)
                    self.load_state_dict(model.state_dict())
                    # self.current_epoch = model.current_epoch
                except:
                    print('Loading Last..., best is not found')
                    self.load_state_dict(torch.load(f'{self.model_path}/last_model_{self.experiment}.pth', weights_only=False).state_dict())
            else:
                print('Loading last...')
                self.load_state_dict(torch.load(f'{self.model_path}/last_model_{self.experiment}.pth', weights_only=False).state_dict())

    @staticmethod
    def load_model(model_path):
        return torch.load(model_path)

    @torch.no_grad()
    def predict(self, x_test, load_model = None):
        if load_model is not None:
            self.load(load_model)
        self.eval()
        x_test = torch.tensor(x_test, dtype=torch.float).to(self.device)
        y_pred = self(x_test)
        return y_pred
