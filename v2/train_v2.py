import sys
sys.path.append('/home/royhirsch/conformal/')

import torch
import torch.nn as nn

import numpy as np
from trainer import Trainer, get_optimizer, get_scheduler
import utils as utils


def nll_loss(preds, target):
    pred_mean = preds[:,0]
    pred_sigma = preds[:,1]
    pred_sigma = torch.clip(torch.exp(pred_sigma), min=1e-4, max=1e10)
    square = torch.square(pred_mean - target)
    return torch.mean(torch.divide(square, pred_sigma) + torch.log(pred_sigma))


class NLLRegressionMetricLogger(utils.MetricLogger):
    def __init__(self, t=1.):
        super().__init__()
        self.t = t

    def reset(self):
        self._losses = []
        self._l2s = []
        self._set_sizes = []
        self._hits = []
        self._all_preds = []
        self._all_labels = []

    def update(self, pred_scores, true_scores):
        pred_scores = pred_scores[:,0].squeeze().detach().cpu()
        true_scores = true_scores.detach().cpu()

        self._all_labels += true_scores.numpy().tolist()
        self._all_preds += pred_scores.numpy().tolist()
        self._l2s += (torch.abs(pred_scores - true_scores)**2).numpy().tolist()

    def calc(self):
        return {'loss': np.mean(self._losses),
                'r^2': utils.calc_r2(np.asarray(self._all_labels),
                               np.asarray(self._all_preds)),
                'l2':  np.mean(self._l2s)}


class NN(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 out_dim=1,
                 hidden_dim=None,
                 drop_rate=0,
                 norm=False,
                 criteria_name='mse'):

        super().__init__()
        self.norm = norm
        self.hidden_dim = hidden_dim
        self.criteria_name = criteria_name
        if norm:
            self.norm = nn.LayerNorm(input_dim)
        if hidden_dim == None:
            self.layers = nn.Linear(input_dim, out_dim)
        elif isinstance(hidden_dim, int) or len(hidden_dim) == 1:
            layers = [nn.Linear(input_dim, hidden_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dim[0])]
            for i in range(1,len(hidden_dim)):
                if drop_rate:
                    layers.append(nn.Dropout(p=drop_rate))
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        
        last_in_features = layers[-1].out_features
        if drop_rate:
            layers.append(nn.Dropout(p=drop_rate))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(last_in_features, out_dim))
        self.layers = nn.Sequential(*layers)

        if criteria_name == 'bce':
            self.post = nn.Sigmoid()

    def forward(self, x):
        if self.criteria_name == 'bce':
            return self.post(self.layers(x))
        else:
            return self.layers(x)


def train(dls, config, input_key='embeds'):
    train_dl = dls['train']
    valid_dl = dls['valid']
    test_dl = dls['test']

    model = NN(input_dim=config.input_dim if input_key == 'embeds' else config.num_classes,
               hidden_dim=config.hidden_dim,
               out_dim=config.out_dim,
               norm=config.norm,
               drop_rate=config.drop_rate,
               criteria_name=config.criteria_name)

    model = model.to(config.device)
    print(model)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    if config.criteria_name == 'nll':
        criteria = nll_loss
        trainer = Trainer(criteria=criteria,
                        metric_logger=NLLRegressionMetricLogger,
                        config=config,
                        input_key=input_key)
    elif config.criteria_name == 'mse':
        criteria = nn.MSELoss()
        trainer = Trainer(criteria=criteria,
                        metric_logger=utils.RegressionMetricLogger,
                        config=config,
                        input_key=input_key)
    else:
        raise ValueError

    trainer.fit(model=model,
                train_loader=train_dl,
                test_loader=test_dl,
                optimizer=optimizer,
                scheduler=scheduler,
                valid_loader=valid_dl)
    
    train_predict_out = trainer.predict(model, train_dl)
    valid_predict_out = trainer.predict(model, valid_dl)
    test_predict_out = trainer.predict(model, test_dl)

    outs = {
        'train': train_predict_out,
        'valid': valid_predict_out,
        'test': test_predict_out,
        }
    return model, trainer.history, outs
