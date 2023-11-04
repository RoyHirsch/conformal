import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


def extract(dl):
    probs = dl.dataset.cls_probs.numpy()
    labels = dl.dataset.cls_labels.numpy()
    return probs, labels


def split(probs, labels, n_calib):
    idx = np.array([1] * n_calib + [0] * (probs.shape[0] - n_calib)) > 0
    np.random.shuffle(idx)
    cal_probs, val_probs = probs[idx,:], probs[~idx,:]
    cal_labels, val_labels = labels[idx], labels[~idx]
    return cal_probs, cal_labels, val_probs, val_labels


def get_calib_and_val_datasets(train_dl, val_dl, n_calib):
    cal_probs, cal_labels = extract(train_dl)
    val_probs, val_labels = extract(val_dl)
    # cal_probs, cal_labels, val_probs, val_labels = split(train_probs, train_labels, n_calib)

    return cal_probs, cal_labels, val_probs, val_labels


def naive(train_dl, val_dl, n_calib, alpha=0.1):
    cal_probs, cal_labels, val_probs, val_labels = get_calib_and_val_datasets(train_dl, val_dl, n_calib)
    sorted_val_probs = np.sort(val_probs, 1)[:,::-1]
    argsorted_val_probs = np.argsort(val_probs, 1)[:,::-1]
    cumsum_val_probs = np.cumsum(sorted_val_probs, 1)
    thresh = 1. - alpha

    sets = []
    scores = []
    for p, indxs in zip(cumsum_val_probs, argsorted_val_probs):
        i = np.where(p >= thresh)[0][0]
        scores.append(p[i])
        sets.append(tuple(indxs[:i + 1]))
    scores = np.asarray(scores)
    return (sets, val_labels)


def score(train_dl, val_dl, n_calib, alpha=0.1):
    cal_probs, cal_labels, val_probs, val_labels = get_calib_and_val_datasets(train_dl, val_dl, n_calib)
    n = len(cal_labels)

    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_scores = 1 - cal_probs[np.arange(n), cal_labels]
    # 2: get adjusted quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    prediction_sets = val_probs >= (1 - qhat)
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    return (sets, val_labels)


def aps(train_dl, val_dl, n_calib, alpha=0.1):
    cal_probs, cal_labels, val_probs, val_labels = get_calib_and_val_datasets(train_dl, val_dl, n_calib)
    n = len(cal_labels)

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_pi = cal_probs.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_probs, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    # Get the score quantile
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )
    # Deploy (output=list of length n, each element is tensor of classes)
    val_pi = val_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)

    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    return (sets, val_labels)


def aps_randomized(train_dl, val_dl, n_calib, alpha=0.1, randomized=True, no_zero_size_sets=True):
    cal_probs, cal_labels, val_probs, val_labels = get_calib_and_val_datasets(train_dl, val_dl, n_calib)
    n = len(cal_labels)

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_probs = cal_probs.astype(np.float64)
    cal_pi = cal_probs.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_probs, cal_pi, axis=1).cumsum(axis=1)
    cal_softmax_correct_class = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    if not randomized:
        cal_scores = cal_softmax_correct_class
    else:
        cumsum_index = np.where(cal_srt == cal_softmax_correct_class[:,None])[1]
        if cumsum_index.shape[0] != cal_srt.shape[0]:
            _, unique_indices = np.unique(np.where(
                cal_srt == cal_softmax_correct_class[:,None])[0], return_index=True)
            cumsum_index = cumsum_index[unique_indices]

        high = cal_softmax_correct_class
        low = np.zeros_like(high)
        low[cumsum_index != 0] = cal_srt[np.where(cumsum_index != 0)[0], cumsum_index[cumsum_index != 0]-1]
        cal_scores = np.random.uniform(low=low, high=high)

    # Get the score quantile
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )
    # Deploy (output=list of length n, each element is tensor of classes)
    val_pi = val_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)
    if not randomized:
        prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    else:
        n_val = val_srt.shape[0]
        cumsum_index = np.sum(val_srt <= qhat, axis=1)
        high = val_srt[np.arange(n_val), cumsum_index]
        low = np.zeros_like(high)
        low[cumsum_index > 0] = val_srt[np.arange(n_val), cumsum_index-1][cumsum_index > 0]
        prob = (qhat - low)/(high - low)
        rv = np.random.binomial(1,prob,size=(n_val))
        randomized_threshold = low
        randomized_threshold[rv == 1] = high[rv == 1]
        if no_zero_size_sets:
            randomized_threshold = np.maximum(randomized_threshold, val_srt[:,0])
        prediction_sets = np.take_along_axis(val_srt <= randomized_threshold[:,None], val_pi.argsort(axis=1), axis=1)
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    return (sets, val_labels)


def raps(train_dl, val_dl, n_calib, alpha=0.1, lam_reg=0.01, k_reg=5, disallow_zero_sets=False, rand=False):
    cal_probs, cal_labels, val_probs, val_labels = get_calib_and_val_datasets(train_dl, val_dl, n_calib)
    n = len(cal_labels)

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    reg_vec = np.array(k_reg*[0,] + (cal_probs.shape[1]-k_reg)*[lam_reg,])[None,:]
    cal_pi = cal_probs.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_probs,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]

    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

    # Deploy
    n_val = val_probs.shape[0]
    val_pi = val_probs.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_probs,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    return (sets, val_labels)


def calc_conformal_mets(sets, labels):
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
    return {'size_mean': set_lens.mean(),
            'size_std': set_lens.std(),
            'acc': acc}


def calc_baseline_mets(train_dl, val_dl, n_calib=0, alpha=0.1,
                       model_names=['naive', 'aps', 'aps_randomized', 'raps', 'raps_randomized'],
                       k_raps=5):
    mets = {}

    if 'naive' in model_names:
        sets, labels = naive(train_dl, val_dl, n_calib, alpha)
        mets['naive'] = calc_conformal_mets(sets, labels)

    if 'score' in model_names:
        sets, labels = score(train_dl, val_dl, n_calib, alpha)
        mets['score'] = calc_conformal_mets(sets, labels)

    if 'aps' in model_names:
        sets, labels = aps(train_dl, val_dl, n_calib, alpha)
        mets['aps'] = calc_conformal_mets(sets, labels)

    if 'aps_randomized' in model_names:
        sets, labels = aps_randomized(train_dl, val_dl, n_calib, alpha)
        mets['aps_randomized'] = calc_conformal_mets(sets, labels)

    if 'raps' in model_names:
        sets, labels = raps(train_dl, val_dl, n_calib, alpha, rand=False, k_reg=k_raps)
        mets['raps'] = calc_conformal_mets(sets, labels)
    
    if 'raps_randomized' in model_names:
        sets, labels = raps(train_dl, val_dl, n_calib, alpha, rand=True, k_reg=k_raps)
        mets['raps_randomized'] = calc_conformal_mets(sets, labels)

    return mets


def calc_baseline_details(train_dl, val_dl, n_calib=0, alpha=0.1,
                          model_names=['naive', 'aps', 'aps_randomized', 'raps', 'raps_randomized'],
                          k_raps=5):
    
    def helper(sets, labels):
        set_lens = []
        hits = []
        for s, l in zip(sets, labels):
            set_lens.append(len(s))
            sets.append(s)
            if l in s:
                hits.append(1)
            else:
                hits.append(0)
        return hits, set_lens
    
    mets = {}

    if 'naive' in model_names:
        sets, labels = naive(train_dl, val_dl, n_calib, alpha)
        hits, set_lens = helper(sets, labels)
        mets['naive_hits'] = hits
        mets['naive_size'] = set_lens
        mets['naive_set'] = sets

    if 'score' in model_names:
        sets, labels = score(train_dl, val_dl, n_calib, alpha)
        hits, set_lens = helper(sets, labels)
        mets['score_hits'] = hits
        mets['score_size'] = set_lens
        mets['score_set'] = sets

    if 'aps' in model_names:
        sets, labels = aps(train_dl, val_dl, n_calib, alpha)
        hits, set_lens = helper(sets, labels)
        mets['aps_hits'] = hits
        mets['aps_size'] = set_lens
        mets['aps_set'] = sets

    if 'aps_randomized' in model_names:
        sets, labels = aps_randomized(train_dl, val_dl, n_calib, alpha)
        hits, set_lens = helper(sets, labels)
        mets['aps_rand_hits'] = hits
        mets['aps_rand_size'] = set_lens
        mets['aps_rand_set'] = sets

    if 'raps' in model_names:
        sets, labels = raps(train_dl, val_dl, n_calib, alpha, rand=False, k_reg=k_raps)
        hits, set_lens = helper(sets, labels)
        mets['raps_hits'] = hits
        mets['raps_size'] = set_lens
        mets['raps_set'] = sets
    
    if 'raps_randomized' in model_names:
        sets, labels = raps(train_dl, val_dl, n_calib, alpha, rand=True, k_reg=k_raps)
        hits, set_lens = helper(sets, labels)
        mets['raps_rand_hits'] = hits
        mets['raps_rand_size'] = set_lens
        mets['raps_rand_set'] = sets

    return mets

if __name__ == '__main__':
    from config import get_config_by_name
    from conformal import get_conformal_module
    from data import get_dataloaders

    config = get_config_by_name('tissuemnist')
    config.plat_scaling = True
    conformal_module = get_conformal_module(config.conformal_module_name)

    dls, t = get_dataloaders(config, conformal_module)
    mets = calc_baseline_mets(dls['train'], dls['test'], n_calib=10000, alpha=0.1,
                              model_names=['naive', 'aps', 'aps_randomized', 'raps', 'raps_randomized'])

    for k, v in mets.items():
        print(k)
        for kk, vv in v.items():
            print(f'{kk}: {vv:.3f}')
        print('\n')
