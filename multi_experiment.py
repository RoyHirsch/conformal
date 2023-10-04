import os
import copy
import random
from tqdm import tqdm
from itertools import product
import numpy as np
import pandas as pd
import logging

from experiment import run_experiment
from config import get_config_by_name
import utils

class ExperimentsLogger():
    def __init__(self, save_dir, file_name='scan_res.csv',
                 sort_field='cp_net_post_sizes', periodic_save=5) -> None:
        self.save_dir = save_dir
        self.file_name = os.path.join(save_dir, file_name)
        self.sort_field = sort_field
        self.periodic_save = periodic_save
        self.df = pd.DataFrame(
            columns=['num', 'config', 'train_sizes', 'train_acc','val_sizes', 'val_acc',
                     'cp_net_sizes', 'cp_net_acc', 'cp_net_post_sizes', 'cp_net_post_acc'])
        self.i = 1
        self.baseline_mets = []

    def update(self, config, outputs):
        self.df.loc[len(self.df)] = [self.i,
                                     str(vars(config)), 
                                     outputs['train_mets']['size_mean'],
                                     outputs['train_mets']['acc'],
                                     outputs['val_mets']['size_mean'],
                                     outputs['val_mets']['acc'],
                                     outputs['test_mets']['size_mean'],
                                     outputs['test_mets']['acc'],
                                     outputs['calibrated_test_mets']['size_mean'],
                                     outputs['calibrated_test_mets']['acc']]
        self.i += 1
        self.baseline_mets.append(outputs['baseline_mets'])
        if self.i % self.periodic_save == 0:
            self.save()

    def save(self):
        self.df = self.df.sort_values(by=self.sort_field, ascending=False)
        self.df.to_csv(self.file_name)
        print('Best exp:')
        best_exp = self.df.iloc[0]
        print(best_exp['config'])
        print('Post-test acc: {:.4f} and size: {:.2f}'.format(
            best_exp['cp_net_post_acc'],
            best_exp['cp_net_post_sizes']))

    def save_summary(self):
        s = {}
        for exp in self.baseline_mets:
            for k, v in exp.items():
                if k not in s:
                    s[k] = {'size': [], 'acc': []}
                s[k]['size'].append(v['size_mean'])
                s[k]['acc'].append(v['acc'])

        for k, v in s.items():
            nv = np.asarray(s[k]['size'])
            s[k]['size'] = '{:.3f} \pm {:.3f}'.format(nv.mean(), nv.std())
            nv = np.asarray(s[k]['acc'])
            s[k]['acc'] = '{:.3f} \pm {:.3f}'.format(nv.mean(), nv.std())
        
        df = pd.DataFrame(s).T
        logging.info(df)
        df.to_csv(os.path.join(self.save_dir, 'summery.csv'))


def modify_config(base_config, params_to_modify):
    config = copy.deepcopy(base_config)
    for k, v in params_to_modify.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config


if __name__ == '__main__':

    dataset_name= 'organamnist'
    name = f'{dataset_name}_10r_0410'
    out_dir = f'/home/royhirsch/conformal/exps/{dataset_name}'
    num_repetitions = 10
    gpu_num = 1

    config = get_config_by_name(dataset_name)
    config.name = name
    config.out_dir = out_dir
    config.gpu_num = gpu_num
    config.exp_dir = os.path.join(config.out_dir, name)
    config.dump_log = True
    
    utils.create_logger(config.exp_dir, config.dump_log)
    if not os.path.exists(config.exp_dir):
        os.makedirs(config.exp_dir)

    logger = ExperimentsLogger(config.exp_dir)

    for seed in range(num_repetitions):
        modified_config = copy.deepcopy(config)
        modified_config.seed = seed
        outputs = run_experiment(modified_config)
        logger.update(modified_config, outputs)
        print('Finish exp: {}/{}'.format(seed , num_repetitions))
        logger.save()
    logger.save_summary()
