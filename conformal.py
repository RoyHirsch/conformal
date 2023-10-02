from abc import ABC, abstractmethod
import logging
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import copy
import logging


def calibrate_residual(calib_true_scores, calib_pred_scores, 
                       valid_true_scores, valid_pred_scores, 
                       alpha):

    # get the quantile from the residuals
    diff = calib_true_scores - calib_pred_scores
    n = len(diff)
    qhat = np.quantile(
        diff, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")

    # modify the validation scores
    valid_pred_scores = copy.deepcopy(valid_pred_scores) + qhat
    n_over_one = (valid_pred_scores >= 1.).sum() / len(valid_pred_scores)
    logging.info('Residuals correction is {:.3f}, clip {:.2f}% of the samples'.format(
        qhat, n_over_one * 100.))
    return np.minimum(valid_pred_scores, (1 - alpha)), qhat


def get_percentile(scores, alpha=0.1):
    n = len(scores)
    qhat = np.quantile(
        scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")
    return qhat


def clip_scores(scores, clip_value):
    inds = np.where(scores >= clip_value)[0]
    logging.info(f'Clipping {len(inds) / len(scores) * 100.:.2f}% of the scores to {clip_value:.4f}')  # noqa: E501
    if isinstance(scores, torch.Tensor):
        scores[inds] = torch.tensor(clip_value, dtype=torch.float32)
    else:
        scores[inds] = clip_value
    return scores


class ConformalBase(ABC):
    score_clip_value = None

    @abstractmethod
    def get_scores(self, probs, labels, **kwargs):
        '''Calculated the conformal scores.
        
        Returns a numpy array with labels shape'''
        pass

    @abstractmethod
    def get_sets(self, scores, probs, **kwargs):
        '''Predict the set of classes given a score/threshold and the cls logits.
        
        Return list of class number tuples. 
        '''
        pass

    def calibrate_and_get_qhat(self, probs, labels, t=1., alpha=0.1, **kwargs):
        n = len(labels)
        scores = self.get_scores(probs, labels)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        return np.quantile(scores, q_level, method="higher")
    
    def calibrate_dls(self, train_dl, val_dl, t=1., alpha=0.1):
        train_probs = train_dl.dataset.cls_probs.numpy()
        train_labels = train_dl.dataset.cls_labels.numpy()

        val_probs = val_dl.dataset.cls_probs.numpy()
        val_labels = val_dl.dataset.cls_labels.numpy()
        return self.baseline_calibrate(
            train_probs, train_labels, val_probs, val_labels, t, alpha)

    def baseline_calibrate(
            self, train_probs, train_labels, val_probs, val_labels, t=1., alpha=0.1):
        qhat = self.calibrate_and_get_qhat(train_probs, train_labels, t, alpha)
        sets = self.get_sets(np.full_like(val_probs, qhat), val_probs)
        mets = self.get_conformal_mets(sets, val_labels)
        mets['qhat'] = qhat
        return mets

    @classmethod
    def clip_scores(self, scores):
        inds = np.where(scores >= self.score_clip_value)[0]
        logging.info(f'[Conf] Clipping {len(inds) / len(scores) * 100.:.2f}% of the scores to {self.score_clip_value:.4f}')  # noqa: E501
        scores[inds] = self.score_clip_value
        return scores

    @classmethod
    def set_score_clipping(self, value):
        self.score_clip_value = value
        
    @classmethod
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
        return {'size_mean': set_lens.mean(),
                'size_std': set_lens.std(),
                # 'acc_per_label_mean': acc_per_label.mean(),
                # 'acc_per_label_max': acc_per_label.max(),
                # 'acc_per_label_min': acc_per_label.min(),
                'acc': acc}

    @classmethod
    def get_sets_and_log_mets(self, pred_scores, true_scores,
                              pred_cls_logits, true_cls_labels, t=1.,
                              use_pred_scores=True, log_prefix='',
                              **kwargs):
        sets = self.get_sets(pred_scores, true_scores,
                             pred_cls_logits, true_cls_labels, t, use_pred_scores, 
                             **kwargs)
        mets = self.get_conformal_mets(sets, true_cls_labels)
        logging.info('{} mets: acc={:.3f} | set size: {:.2f} ({:.3f})'.format(
            log_prefix, mets['acc'], mets['set_size_mean'], mets['set_size_std']))


class APS(ConformalBase):

    def get_scores(self, probs, labels):
        scores = []
        for p, l in zip(probs, labels):
            true_class_p = p[l]
            score = np.sum(p[p >= true_class_p])
            scores.append(score)
        scores = np.asarray(scores)

        if self.score_clip_value:
            scores = self.clip_scores(scores)
        return scores

    def get_sets(self, scores, probs):
        sets = []
        for s, p, in zip(scores, probs):
            sorted_p = np.sort(p)[::-1]
            argsort_p = np.argsort(p)[::-1]
            try:
                cumsum_p = np.cumsum(sorted_p)            
                ind = np.where(cumsum_p >= (s - 1e-7))[0][0]
            except:
                print(s)
                print(cumsum_p)

            
            if self.score_clip_value and s == np.float32(self.score_clip_value):
                ind =- 1
            sets.append(tuple(argsort_p[:ind + 1]))
        return sets


class Naive(ConformalBase):

    def get_scores(self, probs, labels):
        n = len(labels)
        scores =  1. - probs[np.arange(n), labels]
        if self.score_clip_value:
            scores = self.clip_scores(scores)
        return scores


    def get_sets(self, scores, probs):
        if isinstance(scores, np.ndarray):
            scores = np.expand_dims(scores, 1)

        prediction_sets = probs >= (1 - scores)

        sets = []
        for row in prediction_sets:
            sets.append(tuple(np.where(row != 0)[0]))
        return sets


def get_conformal_module(conformal_module_name):
    if conformal_module_name == 'naive':
        return Naive()
    elif conformal_module_name == 'aps':
        return APS()
    else:
        raise ValueError


if __name__ == '__main__':
    from evaluate import load_pickle, split_data

    file_name = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r152/valid.pickle'
    conformal_module_name = 'aps'
    t = 1.
    alpha = 0.1

    data = load_pickle(file_name)
    train_data, val_data = split_data(data, 40000, seed=42)
    print('Train/Calib shape: {} | Val shape: {}'.format(train_data['labels'].shape,
                                                         val_data['labels'].shape))

    conformal = get_conformal_module(conformal_module_name)
    mets = conformal.baseline_calibrate(train_data['preds'], train_data['labels'],
                                        val_data['preds'], val_data['labels'])
    print(mets)

    qhat = conformal.calibrate_and_get_qhat(train_data['preds'], train_data['labels'], t, alpha)
    sets = conformal.get_sets(np.full(val_data['labels'].shape, qhat, ), 
                              np.full(val_data['labels'].shape, qhat, ), val_data['preds'], t)
    mets = conformal.get_conformal_mets(sets, val_data['labels'])
    mets['qhat'] = qhat
    print(mets)

