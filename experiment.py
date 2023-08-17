import os
from ml_collections import config_dict
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders
from trainer import Trainer, get_optimizer, get_scheduler
import utils as utils


class APS():
    score_clip_value = None

    def set_score_clipping(self, value):
        self.score_clip_value = value

    def get_scores(self, probs, labels):
        scores = []
        for p, l in zip(probs, labels):
            true_class_p = p[l]
            score = np.sum(p[p >= true_class_p])
            scores.append(score)
        scores = np.asarray(scores)

        if self.score_clip_value:
            inds = np.where(scores >= self.score_clip_value)[0]
            logging.info(f'Clipping {len(inds) / len(scores) * 100.:.2f}% of the scores to {self.score_clip_value:.2f}')
            scores[inds] = self.score_clip_value
        return scores

        # n = len(labels)
        # cal_pi = probs.argsort(1)[:, ::-1]
        # cal_srt = np.take_along_axis(probs, cal_pi, axis=1).cumsum(axis=1)
        # return np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        #     range(n), labels]
    
    def get_sets(self, scores, probs):
        sets = []
        for s, p, in zip(scores, probs):
            sorted_p = np.sort(p)[::-1]
            argsort_p = np.argsort(p)[::-1]
            cumsum_p = np.cumsum(sorted_p)            
            ind = np.where(cumsum_p >= s)[0][0]
            if self.score_clip_value and s == np.float32(self.score_clip_value):
                ind =- 1
            sets.append(tuple(argsort_p[:ind + 1]))
        return sets
    
    def get_conformal_mets(self, sets, labels):
        set_lens = []
        hits = []
        hits_per_label = {}
        for s, l in zip(sets, labels):
            set_lens.append(len(s))
            if l not in hits_per_label:
                hits_per_label[l] = []
            if l in s:
                hits.append(1)
                hits_per_label[l].append(1)
            else:
                hits.append(0)
                hits_per_label[l].append(0)

        acc = np.asarray(hits).mean()
        set_lens = np.asarray(set_lens)
        acc_per_label = {k: np.mean(v) for k, v in hits_per_label.items()}
        acc_per_label = np.asarray(list(acc_per_label.values()))
        return {'set_size_mean': set_lens.mean(),
                'set_size_std': set_lens.std(),
                'acc_per_label_mean': acc_per_label.mean(),
                'acc_per_label_max': acc_per_label.max(),
                'acc_per_label_min': acc_per_label.min(),
                'acc': acc}

    def calibrate_dls(self, calib_dl, val_dl, alpha):
        train_probs = calib_dl.dataset.cls_probs.numpy()
        train_labels = calib_dl.dataset.cls_labels.numpy()
        n = len(train_labels)

        calib_scores = self.get_scores(train_probs, train_labels)
        qhat = np.quantile(
            calib_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")

        val_probs = val_dl.dataset.cls_probs.numpy()
        val_labels = val_dl.dataset.cls_labels.numpy()

        val_pi = val_probs.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)
        prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)

        sets = []
        for i in range(len(prediction_sets)):
            sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
        mets = self.get_conformal_mets(sets, val_labels)
        mets['qhat'] = qhat
        return mets


def predict_and_report_mets(conformal_module, trainer, model, dl, fold_name=''):
    predict_out = trainer.predict(model, dl)
    scores = predict_out['pred_scores']

    if conformal_module.score_clip_value:
        score_clip_value = conformal_module.score_clip_value
        inds = np.where(scores >= score_clip_value)[0]
        logging.info(f'Clipping {len(inds) / len(scores) * 100.:.2f}% of the scores to {score_clip_value:.2f}')
        scores[inds] = score_clip_value
        predict_out['pred_scores'] = scores

    sets = conformal_module.get_sets(
        scores,
        predict_out['cls_probs'])
    mets = conformal_module.get_conformal_mets(sets, predict_out['cls_labels'])
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
    cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r152/valid.pickle'
    # cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/aug/imnet1k_r152/100k_train.pickle'
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
    cfg.input_dim = 2048
    cfg.norm = False
    cfg.drop_rate = 0.0
    cfg.hidden_dim = 512

    # optim
    cfg.optimizer_name = 'adamw'
    cfg.scheduler_name = 'none'
    cfg.criteria_name = 'mse'
    cfg.lr = 1e-3
    cfg.wd = 0

    # train
    cfg.num_epochs = 50
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


def run_experiment(config):
    conformal_module = APS()
    train_dl, valid_dl, t = get_dataloaders(config, conformal_module)
    
    baseline_mets = conformal_module.calibrate_dls(
        valid_dl, train_dl, alpha=config.alpha)
    utils.log('Baseline mets', baseline_mets)

    # train_dl.dataset.rand = 0.8

    # if use clipping, need to re-calc the scores for the datasets
    if config.use_score_clipping:
        conformal_module.set_score_clipping(baseline_mets['qhat'])
        train_dl, valid_dl, t = get_dataloaders(config, conformal_module)
        sets = conformal_module.get_sets(
            train_dl.dataset.scores.numpy(),
            train_dl.dataset.cls_probs.numpy())
        mets = conformal_module.get_conformal_mets(
            sets, train_dl.dataset.cls_labels.numpy())
        utils.log('Train mets after clipping', mets)

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
    
    val_predict_out, val_mets = predict_and_report_mets(
        conformal_module, trainer, model, valid_dl, fold_name='Valid')

    train_predict_out, train_mets = predict_and_report_mets(
        conformal_module, trainer, model, train_dl, fold_name='Train')
    
    return trainer.history, val_predict_out, train_predict_out, val_mets, train_mets


if __name__ == '__main__':
    config = get_config()
    utils.seed_everything(config.seed)
    utils.create_logger(config.exp_dir, False)
    run_experiment(config)
