import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], './conformal_classification/'))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
from tqdm import tqdm
import time
from scipy.special import softmax


class RepsDataset(torch.utils.data.Dataset):
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels

    def __getitem__(self, i):
        return self.logits[i], self.labels[i]

    def __len__(self):
        return len(self.labels)


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def split_data(data, n_calib, seed=None):
    labels = data['labels']
    n = len(labels)
    idx = np.arange(n)
    
    if seed:
        np.random.seed = seed
    np.random.shuffle(idx)

    valid_idx = idx[n_calib:]
    calib_idx = idx[:n_calib]
    
    valid_data = {'embeds': data['embeds'][valid_idx, :],
                 'preds': data['preds'][valid_idx, :],
                 'labels': data['labels'][valid_idx]}
    calib_data = {'embeds': data['embeds'][calib_idx, :],
                 'preds': data['preds'][calib_idx, :],
                 'labels': data['labels'][calib_idx]}

    return valid_data, calib_data


def platt(logits, labels, batch_size=128, max_iters=10, lr=0.01, epsilon=0.01):
    logits_loader = torch.utils.data.DataLoader(RepsDataset(logits, labels),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True)

    T = platt_logits(logits_loader, max_iters=max_iters, lr=lr, epsilon=epsilon)

    # print(f"Optimal T={T.item()}")
    return T.item() 


def platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 


class RegScore:
    @staticmethod
    def score(logits, labels, T=1):
        softmax_scores = softmax(logits/T, axis=1)
        return 1. - softmax_scores[np.arange(len(logits)), labels]
    
    @staticmethod
    def get_sets(logits, qhat, T=1):
        softmax_scores = softmax(logits/T, axis=1)
        return softmax_scores >= (1 - qhat)


class APSScore:
    @staticmethod
    def score(logits, labels, T=1):
        scores = softmax(logits/T, axis=1)
        cal_pi = scores.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(scores, cal_pi, axis=1).cumsum(axis=1)
        return np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
            range(len(labels)), labels]
    
    @staticmethod
    def get_sets(logits, qhat, T=1):
        scores = softmax(logits/T, axis=1)
        val_pi = scores.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(scores, val_pi, axis=1).cumsum(axis=1)
        return np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)


def calc_conf_qhat(logits, targets, T=1, alpha=0.1, score_fuc=RegScore.score):
    n = len(targets)
    scores = score_fuc(logits, targets, T)
    qhat = np.quantile(
        scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")
    return qhat 


def eval(logits, targets, qhat, T=1, get_sets_func=RegScore.get_sets):
    prediction_sets = get_sets_func(logits, qhat, T)

    empirical_coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]), targets].mean()

    sizes = prediction_sets.sum(1)
    m = {'size_mean': sizes.mean(),
         'size_std': sizes.std(),
         'coverage': empirical_coverage,
         'qhat': qhat}
    return m


def calibrate_and_calc(calib_logits, calib_labels,
                       val_logits, val_labels,
                       score_class,
                       alpha=0.1, ts=False):
    if ts:
        T = platt(calib_data['preds'], calib_data['labels'])    
    else: 
        T = 1

    qhat = calc_conf_qhat(calib_logits, calib_labels, T, alpha, score_fuc=score_class.score)
    mets = eval(val_logits, val_labels, qhat, T, get_sets_func=score_class.get_sets)
    return mets


class MetsAgg:
    def __init__(self):
        self.m = {}

    def update(self, m):
        for k, v in m.items():
            if k not in self.m:
                self.m[k] = []
            self.m[k].append(v)

    def calc(self):
        r = {}
        for k, v in self.m.items():
            r[k] = np.mean(v)
        return r


def results_print(mets, title=''):
    print(title)
    print('q_hat: {:.6f}'.format(mets['qhat']))
    print('Coverage: {:.4f}'.format(mets['coverage']))
    print('Sets size: {:.2f} ({:.2f})'.format(mets['size_mean'],
                                              mets['size_std']))
    
if __name__ == "__main__":
    device = torch.device('cuda:6')
    n_calib = 1000
    batch_size = 128
    num_worker = 4
    alpha = 0.1
    ts = True
    seed = None
    file_name = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r18/valid.pickle'

    reps = 20
    data = load_pickle(file_name)
    mets_agg = MetsAgg()
    for s in tqdm(range(reps)):
        valid_data, calib_data = split_data(data, n_calib, seed=str)
        mets = calibrate_and_calc(calib_data['preds'], calib_data['labels'],
                                valid_data['preds'], valid_data['labels'],
                                score_class=RegScore, alpha=alpha, ts=ts)
        mets_agg.update(mets)
        # results_print(mets, title='Reg')
    results_print(mets_agg.calc(), title='Mean Reg')
    
    mets_agg = MetsAgg()
    for s in tqdm(range(reps)):
        valid_data, calib_data = split_data(data, n_calib, seed=s)
        mets = calibrate_and_calc(calib_data['preds'], calib_data['labels'],
                                valid_data['preds'], valid_data['labels'],
                                score_class=APSScore, alpha=alpha, ts=ts)
        mets_agg.update(mets)
        # results_print(mets, title='Reg')
    results_print(mets_agg.calc(), title='Mean APS')
    