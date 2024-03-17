import os
from ml_collections import config_dict
import logging
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from mapie.metrics import classification_coverage_score
from sklearn.metrics import accuracy_score

from conformal import get_conformal_module, get_percentile, clip_scores
from data import read_and_split, get_dataloader
from trainer import Trainer, get_optimizer, get_scheduler
import utils as utils

# from mapie.classification import MapieClassifier
from mapie_classification import MapieClassifier


class MapieWrapper():
    def __init__(self) -> None:
        self.classes_ = None
        self.trained_ = False
        
    def fit(self, x, y) -> None:
        self.classes_ = np.arange(len(np.unique(y)))
        self.trained_ = True

    def predict_proba(self, x) -> np.ndarray:
        return x

    def predict(self, x) -> np.ndarray:
        return x.argmax(1)

    def __sklearn_is_fitted__(self):
        return True


def count_null_set(y: np.ndarray) -> int:
    count = 0
    for pred in y[:, :]:
        if np.sum(pred) == 0:
            count += 1
    return count


class MapieConformalModule():
    def __init__(self, method='aps', alpha=0.9, cv="prefit", random_state=42, include_last_label=None) -> None:
        self.method = method
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state

        if include_last_label == None:
            if self.method == 'aps':
                self.include_last_label = True
            elif 'random' in self.method:
                self.include_last_label = 'randomized'
            else:
                self.include_last_label = False
        else:
            self.include_last_label = include_last_label
        self.score_clip_value = None

    def to_probs(self, preds):
        return scipy.special.softmax(preds, 1)
    
    def fit(self, probs, labels):
        self.est = MapieWrapper()
        self.est.fit(probs, labels)
        self.mapie = MapieClassifier(estimator=self.est,
                        method=self.method,
                        cv=self.cv,
                        random_state=self.random_state) 
        self.mapie.fit(probs, labels)
        _  = self.mapie.predict(probs,
                                alpha=self.alpha,
                                include_last_label=self.include_last_label)
        self.qhat = self.mapie.quantiles_[0]

    def get_scores(self, probs, labels):
        mapie = MapieClassifier(estimator=self.est,
                                method=self.method,
                                cv=self.cv,
                                random_state=self.random_state) 
        mapie.fit(probs, labels)
        return mapie.conformity_scores_[:, 0]

    def get_sets(self, probs):
        sets = self.mapie.predict(probs,
                                  alpha=self.alpha,
                                  include_last_label=self.include_last_label)[1]
        return sets

    def get_sets_using_precalc_quantiles(self, probs, precalc_quantiles):
        sets = self.mapie.predict(probs,
                                  alpha=self.alpha,
                                  precalc_quantiles=precalc_quantiles,
                                  include_last_label=self.include_last_label)[1]
        # return [np.where(s)[0] for s in sets]
        return sets


    def get_conformal_mets(self, sets, labels):
        n_null = count_null_set(sets[:, :, 0])
        coverage = classification_coverage_score(labels, sets[:, :, 0])
        size = sets[:, :, 0].sum(axis=1).mean()
        return {'n_null': n_null, 'coverage': coverage, 'size': size}


def get_dataloaders(config, conformal_module, get_data_func=read_and_split):
    data = get_data_func(config)
    for k, v in data.items():
        logging.info('{} shape: {}'.format(k, v['labels'].shape))

    t = 1.
    dls = {}
    for k, v in data.items():
        v['probs'] = softmax(v['preds'] / t, 1)
        scores = conformal_module.get_scores(v['probs'], v['labels'])
        dl = get_dataloader(v['embeds'], v['probs'], v['labels'],
                            np.asarray(scores),
                            batch_size=config.batch_size,
                            shuffle=True if k == 'train' else False,
                            pin_memory=True)
        dls[k] = dl

    return dls, t
 

def predict_and_report_mets(conformal_module, trainer, model, dl, fold_name=''):
    predict_out = trainer.predict(model, dl)

    if conformal_module.score_clip_value:
        predict_out['pred_scores'] = clip_scores(
            predict_out['pred_scores'],
            conformal_module.score_clip_value)

    sets = conformal_module.get_sets_using_precalc_quantiles(
        predict_out['cls_probs'],
        predict_out['pred_scores'])
    mets = conformal_module.get_conformal_mets(
        sets, 
        predict_out['cls_labels'])
    utils.log(f'{fold_name} mets', mets)
    return predict_out, mets


def get_config():
    
    cfg = config_dict.ConfigDict()

    # experiment
    cfg.name = 'temp1'
    cfg.out_dir = '/home/royhirsch/conformal/exps'
    cfg.exp_dir = os.path.join(cfg.out_dir, cfg.name)

    cfg.dump_log = False
    cfg.comments = ''
    cfg.gpu_num = 0
    cfg.device = torch.device('cuda:{}'.format(cfg.gpu_num) if torch.cuda.is_available() else 'cpu')
    cfg.seed = 42

    # data
    # cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r152/valid.pickle'
    cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/tissuemnist_test.pickle'
    # cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/aug/imnet1k_r152/100k_train.pickle'
    cfg.num_train = 40000
    cfg.num_valid = 3500
    cfg.num_test = 3500
    cfg.batch_size = 128
    cfg.num_workers = 4
    cfg.pin_memory = True
    
    # conformal
    cfg.alpha = 0.1
    cfg.plat_scaling = True
    cfg.conformal_module_name = 'lac'
    cfg.use_score_clipping = True

    # model
    cfg.input_dim = 2048
    cfg.norm = False
    cfg.drop_rate = 0.0
    cfg.hidden_dim = 512

    # optim
    cfg.optimizer_name = 'adamw'
    cfg.scheduler_name = 'none'
    cfg.criteria_name = 'mse'
    cfg.lr = 5e-4
    cfg.wd = 1e-6

    # train
    cfg.num_epochs = 70
    cfg.val_interval = 5
    cfg.save_interval = 200
    cfg.monitor_met_name = 'val_loss'
    
    return cfg


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
            layers = [nn.Linear(input_dim, hidden_dim)]
            if drop_rate:
                layers.append(nn.Dropout(p=drop_rate))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.layers = nn.Sequential(*layers)

        if criteria_name == 'bce':
            self.post = nn.Sigmoid()

    def forward(self, x):
        if self.criteria_name == 'bce':
            return self.post(self.layers(x))
        else:
            return self.layers(x)


def cqr_loss(y_true, y_pred, gamma=0.9):
    true_is_bigger = (y_true - y_pred) * gamma
    pred_is_bigger = (y_pred - y_true) * (1. - gamma)
    return torch.mean(torch.where(y_true > y_pred, true_is_bigger, pred_is_bigger))


def run_experiment(config):
    # conformal_module = get_conformal_module(config.conformal_module_name)
    conformal_module = MapieConformalModule(
        config.conformal_module_name,
        config.alpha)
    data = read_and_split(config)
    conformal_module.fit(softmax(data['train']['preds'], 1),
                         data['train']['labels'])

    dls, t = get_dataloaders(config, conformal_module)
    train_dl = dls['train']
    valid_dl = dls['valid']
    test_dl = dls['test']

    # if use clipping, need to re-calc the scores for the datasets
    if config.use_score_clipping:
        qhat = conformal_module.qhat
        train_dl.dataset.scores = clip_scores(
            train_dl.dataset.scores, qhat)
        valid_dl.dataset.scores = clip_scores(
            valid_dl.dataset.scores, qhat)
        test_dl.dataset.scores = clip_scores(
            test_dl.dataset.scores, qhat)
        conformal_module.score_clip_value = qhat

    sets = conformal_module.get_sets(
        test_dl.dataset.cls_probs.numpy())
    mets = conformal_module.get_conformal_mets(
        sets, test_dl.dataset.cls_labels.numpy())
    utils.log('Baseline mets ', mets)

    model = NN(input_dim=config.input_dim,
               hidden_dim=config.hidden_dim,
               norm=config.norm,
               drop_rate=config.drop_rate,
               criteria_name=config.criteria_name)
    model = model.to(config.device)
    logging.info(model)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    if config.criteria_name == 'mse':
        criteria = nn.MSELoss()
    elif config.criteria_name == 'bce':
        criteria = nn.BCELoss()
    elif config.criteria_name == 'cqr':
        criteria = cqr_loss
    else:
        raise ValueError

    trainer = Trainer(criteria=criteria,
                      metric_logger=utils.RegressionMetricLogger,
                      config=config)

    trainer.fit(model=model,
                train_loader=train_dl,
                test_loader=valid_dl,
                optimizer=optimizer,
                scheduler=scheduler,
                valid_loader=valid_dl)

    train_dl.dataset.cls_labels = train_dl.dataset.cls_labels[:len(valid_dl.dataset)]
    train_predict_out, train_mets = predict_and_report_mets(
        conformal_module, trainer, model, train_dl, fold_name='Train')

    val_predict_out, val_mets = predict_and_report_mets(
        conformal_module, trainer, model, valid_dl, fold_name='Valid')
    
    test_predict_out, test_mets = predict_and_report_mets(
        conformal_module, trainer, model, test_dl, fold_name='Test')

    return {'history': trainer.history,
            'train_out': train_predict_out,
            'val_out': val_predict_out,
            'test_out': test_predict_out,
            'train_mets': train_mets,
            'val_mets': val_mets,
            'test_mets': test_mets}


if __name__ == '__main__':
    config = get_config()
    utils.seed_everything(config.seed)
    utils.create_logger(config.exp_dir, False)
    run_experiment(config)
