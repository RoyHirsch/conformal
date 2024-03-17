import sys
sys.path.append('/home/royhirsch/conformal/')

import numpy as np
import torch
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from evaluate import load_pickle
from conf_tools import get_logits_dataloader, platt_logits


def get_aps_scores(probs, labels):
    n = len(labels)
    cal_pi = probs.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(probs, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), labels
    ]
    return cal_scores


def get_global_threshold(calib_dataset, alpha, randomized=False):
    cal_smx = calib_dataset.cls_probs.numpy()
    cal_labels = calib_dataset.cls_labels.numpy()
    n = len(cal_labels)

    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_softmax_correct_class = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    if not randomized:
        cal_scores = cal_softmax_correct_class
    else:
        cumsum_index = np.where(cal_srt == cal_softmax_correct_class[:,None])[1]
        high = cal_softmax_correct_class
        low = np.zeros_like(high)
        low[cumsum_index != 0] = cal_srt[np.where(cumsum_index != 0)[0], cumsum_index[cumsum_index != 0]-1]
        cal_scores = np.random.uniform(low=low, high=high)
    qhat = np.quantile(
    cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    return qhat


def get_subset(data, indx):
    subset_data = {}
    for k, v in data.items():
        if v.ndim == 2:
            subset_data[k] = v[indx, :]
        else:
            subset_data[k] = v[indx]
    return subset_data

def split(data, par_test, par_valid, seed):
    indx = np.arange(len(data['labels']))
    indx_train, indx_test = train_test_split(indx, test_size=par_test, random_state=seed)
    indx_train, indx_valid = train_test_split(indx_train, test_size=par_valid, random_state=seed)
    return {
        'train': get_subset(data, indx_train),
        'valid': get_subset(data, indx_valid),
        'test': get_subset(data, indx_test)
        }

def get_two_datasets_by_threshold(config, threshold=0.9):
    data = load_pickle(config.file_name)
    max_logic = softmax(data['preds'], 1).max(1)

    above_thresh = np.where(max_logic >= threshold)[0]
    below_thresh = np.where(max_logic < threshold)[0]
    print(f'Num samples above: {len(above_thresh)} Num samples below: {len(below_thresh)} (threshold is {threshold})')

    above_data = get_subset(data, above_thresh)
    below_data = get_subset(data, below_thresh)

    return split(
        above_data, config.par_test, config.par_valid, seed=config.seed
        ), split(
        below_data, config.par_test, config.par_valid, seed=config.seed)


class ScoresDataset(torch.utils.data.Dataset):
    def __init__(self, embeds, cls_probs, cls_labels, scores, rand=0):
        self.embeds = embeds
        self.cls_probs = cls_probs
        self.cls_labels = cls_labels
        self.scores = scores
        self.rand = rand
        

    def __getitem__(self, i):
        return {'embeds': self.embeds[i],
                'probs': self.cls_probs[i],
                'labels': self.cls_labels[i],
                'scores': self.scores[i]}

    def __len__(self):
        return self.cls_labels.size(0)


def get_dataloader(embeds, cls_logits, cls_labels, scores,
                   batch_size=128, shuffle=False,
                   pin_memory=True):
    dataset = ScoresDataset(
        torch.from_numpy(embeds), 
        torch.from_numpy(cls_logits),
        torch.from_numpy(cls_labels).long(),
        torch.from_numpy(scores).float())
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def get_dataloaders_for_subset(config, data):
    for k, v in data.items():
        print('{} shape: {}'.format(k, v['labels'].shape))

    if config.plat_scaling:
        train_dataloader = get_logits_dataloader(data['train']['preds'],
                                                 data['train']['labels'])
        t = platt_logits(train_dataloader)
        print('Temp is {:.4f}'.format(t))
    else: 
        t = 1.

    dls = {}
    for k, v in data.items():
        v['probs'] = softmax(v['preds'] / t, 1)
        scores = get_aps_scores(v['probs'], v['labels'])
        dl = get_dataloader(v['embeds'], v['probs'], v['labels'],
                            np.asarray(scores),
                            batch_size=config.batch_size,
                            shuffle=True if k == 'train' else False,
                            pin_memory=True)
        dls[k] = dl

    return dls, t


def get_dataloaders(config):
    above_data, below_data = get_two_datasets_by_threshold(config, 1. - config.alpha)
    print('##### Above subset #####')
    above_dls, above_t = get_dataloaders_for_subset(config, above_data)
    above_qhat = get_global_threshold(
        above_dls['train'].dataset, alpha=config.alpha,
        randomized=True if 'rand' in config.conformal_module_name else False)
    print(f'Globel qhat: {above_qhat:.4f}')

    print('##### Below subset #####')
    below_dls, below_t = get_dataloaders_for_subset(config, below_data)
    below_qhat = get_global_threshold(
        below_dls['train'].dataset, alpha=config.alpha,
        randomized=True if 'rand' in config.conformal_module_name else False)
    print(f'Globel qhat: {below_qhat:.4f}')
    return above_dls, below_dls


if __name__ == '__main__':
    from config import get_config_by_name
    config = get_config_by_name('tissuemnist')
    above_dls, below_dls = get_dataloaders(config)

