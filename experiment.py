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
from config import get_config_by_name
import utils as utils


def predict_and_report_mets(conformal_module, trainer, model, dl, fold_name=''):
    predict_out = trainer.predict(model, dl)

    sets = conformal_module.get_sets(predict_out['pred_scores'], predict_out['cls_probs'])
    mets = conformal_module.get_conformal_mets(sets, predict_out['cls_labels'])
    utils.log(f'{fold_name} mets', mets)
    return predict_out, mets


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
    baseline_mets = calc_baseline_mets(train_dl, valid_dl, alpha=config.alpha, k_raps=config.k_raps)

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
    conformal_module.score_clip_value = 0.99
    train_predict_out, train_mets = predict_and_report_mets(
        conformal_module, trainer, model, train_dl, fold_name='Train')

    val_predict_out, val_mets = predict_and_report_mets(
        conformal_module, trainer, model, valid_dl, fold_name='Valid')
    
    test_predict_out, test_mets = predict_and_report_mets(
        conformal_module, trainer, model, test_dl, fold_name='Test')

    # calibrate the predicted scores
    calibrated_test_pred_scores, qhat_below, qhat_above = calibrate_residual(
        train_predict_out,
        test_predict_out, 
        config.alpha,
        config.method_name)
    
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

    calibrated_test_mets['qhat_below'] = qhat_below
    calibrated_test_mets['qhat_above'] = qhat_above
    return {'history': trainer.history,
            'train_out': train_predict_out,
            'val_out': val_predict_out,
            'test_out': test_predict_out,
            'train_mets': train_mets,
            'val_mets': val_mets,
            'test_mets': test_mets,
            'calibrated_test_mets': calibrated_test_mets,
            'baseline_mets': baseline_mets}


if __name__ == '__main__':
    config = get_config_by_name('tissuemnist')
    utils.seed_everything(config.seed)
    utils.create_logger(config.exp_dir, config.dump_log)
    outputs = run_experiment(config)
    if config.dump_log:
        with open(os.path.join(config.exp_dir, 'outputs.pickle'), 'wb') as f:
            pickle.dump(outputs, f)


