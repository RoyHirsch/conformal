import os
from ml_collections import config_dict
import logging
import pandas as pd
import pickle
import torch
import torch.nn as nn

from conformal import get_conformal_module, get_percentile, clip_scores, calibrate_residual
from conformal_baselines import calc_baseline_mets
from data import get_dataloaders
from model import NN
from trainer import Trainer, get_optimizer, get_scheduler
import utils as utils


def predict_and_report_mets(conformal_module, trainer, model, dl, fold_name=''):
    predict_out = trainer.predict(model, dl)

    if conformal_module.score_clip_value:
        predict_out['pred_scores'] = clip_scores(
            predict_out['pred_scores'],
            conformal_module.score_clip_value)

    sets = conformal_module.get_sets(predict_out['pred_scores'], predict_out['cls_probs'])
    mets = conformal_module.get_conformal_mets(sets, predict_out['cls_labels'])
    utils.log(f'{fold_name} mets', mets)
    return predict_out, mets


def get_config():
    
    cfg = config_dict.ConfigDict()

    # experiment
    cfg.name = '0210_exp2'
    cfg.out_dir = '/home/royhirsch/conformal/exps/tissuemnist'
    cfg.exp_dir = os.path.join(cfg.out_dir, cfg.name)

    cfg.dump_log = True
    cfg.comments = ''
    cfg.gpu_num = 1
    cfg.device = torch.device('cuda:{}'.format(cfg.gpu_num) if torch.cuda.is_available() else 'cpu')
    cfg.seed = 42

    # data
    # cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r152/valid.pickle'
    cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/tissuemnist_test.pickle'
    # cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/aug/imnet1k_r152/100k_train.pickle'
    cfg.label_transform_name = 'none'
    cfg.num_train = 40000
    cfg.num_valid = 3500
    cfg.num_test = 3500
    cfg.batch_size = 128
    cfg.num_workers = 4
    cfg.pin_memory = True
    
    # conformal
    cfg.alpha = 0.1
    cfg.plat_scaling = True
    cfg.conformal_module_name = 'aps'
    cfg.use_score_clipping = True

    # model
    cfg.input_dim = 2048
    cfg.norm = False
    cfg.drop_rate = 0.0
    cfg.hidden_dim = 32

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


def run_experiment(config):
    logging.info('Config:')
    for k, v in config.items():
        logging.info(f'{k}: {v}')
    logging.info('')

    conformal_module = get_conformal_module(config.conformal_module_name)
    dls, t = get_dataloaders(config, conformal_module)
    train_dl = dls['train']
    valid_dl = dls['valid']
    test_dl = dls['test']
    baseline_mets = calc_baseline_mets(train_dl, valid_dl, alpha=config.alpha)
    # train_dl.dataset.rand = 0.5

    # if use clipping, need to re-calc the scores for the datasets
    if config.use_score_clipping:
        qhat = get_percentile(train_dl.dataset.scores, config.alpha)

        conformal_module.set_score_clipping(qhat)
        train_dl.dataset.scores = clip_scores(
            train_dl.dataset.scores, qhat)
        valid_dl.dataset.scores = clip_scores(
            valid_dl.dataset.scores, qhat)
        test_dl.dataset.scores = clip_scores(
            test_dl.dataset.scores, qhat)

        sets = conformal_module.get_sets(
            test_dl.dataset.scores.numpy(),
            test_dl.dataset.cls_probs.numpy())
        mets = conformal_module.get_conformal_mets(
            sets, test_dl.dataset.cls_labels.numpy())
        utils.log('Baseline mets (after clipping)', mets)
    else:
        baseline_mets = conformal_module.calibrate_dls(
            valid_dl, test_dl, alpha=config.alpha)
        utils.log('Baseline mets', baseline_mets)


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
        raise ValueError(f'Invalid loss name {config.criteria_name}')

    trainer = Trainer(criteria=criteria,
                      metric_logger=utils.RegressionMetricLogger,
                      config=config)

    trainer.fit(model=model,
                train_loader=train_dl,
                test_loader=valid_dl,
                optimizer=optimizer,
                scheduler=scheduler,
                valid_loader=valid_dl)

    # predict regression results for the trained model
    train_predict_out, train_mets = predict_and_report_mets(
        conformal_module, trainer, model, train_dl, fold_name='Train')

    val_predict_out, val_mets = predict_and_report_mets(
        conformal_module, trainer, model, valid_dl, fold_name='Valid')
    
    test_predict_out, test_mets = predict_and_report_mets(
        conformal_module, trainer, model, test_dl, fold_name='Test')

    # calibrate the predicted scores
    calibrated_test_pred_scores, qhat = calibrate_residual(
        val_predict_out['true_scores'],
        val_predict_out['pred_scores'],
        test_predict_out['true_scores'],
        test_predict_out['pred_scores'], 
        config.alpha)
    
    conformal_module = get_conformal_module(config.conformal_module_name)
    sets = conformal_module.get_sets(calibrated_test_pred_scores,
                                     test_predict_out['cls_probs'])
    calibrated_test_mets = conformal_module.get_conformal_mets(
        sets, test_predict_out['cls_labels'])
    utils.log('Calibrated test mets', calibrated_test_mets)

    baseline_mets['cp_net'] = test_mets
    baseline_mets['cp_net_post'] = calibrated_test_mets
    df_mets = pd.DataFrame(baseline_mets).T
    logging.info(df_mets)
    if config.dump_log:
        df_mets.to_csv(os.path.join(config.exp_dir, 'results.csv'))

    calibrated_test_mets['qhat'] = qhat
    return {'history': trainer.history,
            'train_out': train_predict_out,
            'val_out': val_predict_out,
            'test_out': test_predict_out,
            'train_mets': train_mets,
            'val_mets': val_mets,
            'test_mets': test_mets,
            'calibrated_test_mets': calibrated_test_mets}


if __name__ == '__main__':
    config = get_config()
    utils.seed_everything(config.seed)
    utils.create_logger(config.exp_dir, config.dump_log)
    outputs = run_experiment(config)
    if config.dump_log:
        with open(os.path.join(config.exp_dir, 'outputs.pickle'), 'wb') as f:
            pickle.dump(outputs, f)


