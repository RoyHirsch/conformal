import os
from ml_collections import config_dict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders
from conformal import get_conformal_module, get_sets_and_log_mets_baseline
from trainer import Trainer, get_optimizer, get_scheduler
import utils as utils


def get_config():

    def corrections(config):
        # if config.conformal_module_name == 'aps':
        #     config.plat_scaling = True
        return config
    
    cfg = config_dict.ConfigDict()

    # experiment
    cfg.name = 'temp'
    cfg.out_dir = '/home/royhirsch/conformal/exps'
    cfg.exp_dir = os.path.join(cfg.out_dir, cfg.name)

    cfg.dump_log = False
    cfg.comments = ''
    cfg.gpu_num = 0
    cfg.device = torch.device('cuda:{}'.format(cfg.gpu_num) if torch.cuda.is_available() else 'cpu')
    cfg.seed = 42

    # data
    cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r152/valid.pickle'
    cfg.num_train = 40000
    cfg.batch_size = 128
    cfg.num_workers = 4
    cfg.pin_memory = True
    
    # conformal
    cfg.alpha = 0.1
    cfg.plat_scaling = False
    cfg.conformal_module_name = 'aps'
    cfg.use_score_clipping = True

    # model
    cfg.norm = False
    cfg.drop_rate = 0.0
    cfg.hidden_dim = None

    # optim
    cfg.optimizer_name = 'sgd'
    cfg.scheduler_name = 'none'
    cfg.criteria_name = 'mse'
    cfg.lr = 1e-2
    cfg.wd = 1e-4

    # train
    cfg.num_epochs = 40
    cfg.val_interval = 1
    cfg.save_interval = 100
    cfg.monitor_met_name = 'val_loss'
    
    return corrections(cfg)


class NN(nn.Module):
    def __init__(self, input_dim=2048, out_dim=1, hidden_dim=None, drop_rate=0, norm=False, criteria_name='mse'):
        super().__init__()
        self.norm = norm
        self.hidden_dim = hidden_dim
        self.criteria_name = criteria_name
        if norm:
            self.norm = nn.LayerNorm(input_dim)
        if hidden_dim == None:
            self.layers = nn.Linear(input_dim, out_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            if drop_rate:
                layers.append(nn.Dropout(p=drop_rate))
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.layers = nn.Sequential(*layers)

        # self.fc2 = nn.Linear(100, 1)
        if criteria_name == 'bce':
            self.post = nn.Sigmoid()

    def forward(self, x):
        if self.norm:
            x = self.norm(x)
        if self.criteria_name == 'bce':
            return self.post(self.layers(x))
        else:
            return self.layers(x)


def experiment(config):
    conformal_module = get_conformal_module(config.conformal_module_name)
    train_dl, valid_dl, t = get_dataloaders(config, conformal_module)
    
    # get_sets_and_log_mets_baseline(conformal_module, train_dl, t, 'Train Optimal')
    # get_sets_and_log_mets_baseline(conformal_module, valid_dl, t, 'Valid Optimal')

    baseline_mets = conformal_module.baseline_calibrate(
        train_dl, valid_dl, t=t, alpha=config.alpha)
    utils.log('Baseline mets', baseline_mets)

    if config.use_score_clipping:
        conformal_module.set_score_clipping(baseline_mets['qhat'])

    train_dl, valid_dl, t = get_dataloaders(config, conformal_module)

    model = NN(hidden_dim=config.hidden_dim,
               norm=config.norm,
               criteria_name=config.criteria_name)
    model = model.to(config.device)
    logging.info(model)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    if config.criteria_name == 'mse':
        criteria = nn.MSELoss()
    elif config.criteria_name == 'bce':
        criteria = nn.BCELoss()
    else:
        raise ValueError

    trainer = Trainer(criteria,
                      utils.RegressionMetricLogger,
                      config=config)

    trainer.fit(model=model,
                train_loader=train_dl,
                test_loader=valid_dl,
                optimizer=optimizer,
                scheduler=scheduler,
                valid_loader=valid_dl)
    
    val_predict_out = trainer.predict(model, valid_dl)
    conformal_module.randomized = False
    sets = conformal_module.get_sets(
        val_predict_out['pred_scores'], val_predict_out['true_scores'],
        val_predict_out['cls_logits'], t)
    val_mets = conformal_module.get_conformal_mets(sets, val_predict_out['cls_labels'])
    utils.log('Val mets', val_mets)

    train_predict_out = trainer.predict(model, train_dl)
    sets = conformal_module.get_sets(
        train_predict_out['pred_scores'], train_predict_out['true_scores'],
        train_predict_out['cls_logits'], t)
    train_mets = conformal_module.get_conformal_mets(sets, train_predict_out['cls_labels'])
    utils.log('Train mets', train_mets)
    return trainer.history, val_predict_out, train_predict_out, val_mets, train_mets


if __name__ == '__main__':
    config = get_config()
    utils.seed_everything(config.seed)
    utils.create_logger(config.exp_dir, False)
    experiment(config)
