from abc import ABC, abstractmethod
import logging
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn


def get_sets_and_log_mets_baseline(conformal_module, dataloader, t=1., log_prefix='', **kwargs):
    pred_scores = None 
    true_scores = dataloader.dataset.scores.numpy()
    pred_cls_logits = dataloader.dataset.cls_logits.numpy()
    true_cls_labels = dataloader.dataset.cls_labels.numpy()
    sets = conformal_module.get_sets(pred_scores, true_scores, pred_cls_logits, 
                                     t, use_pred_scores=False, allow_zero_sets=False, **kwargs)
    mets = conformal_module.get_conformal_mets(sets, true_cls_labels)
    logging.info('{} mets: acc={:.3f} | set size: {:.2f} ({:.3f})'.format(
        log_prefix, mets['acc'], mets['set_size_mean'], mets['set_size_std']))


class ConformalBase(ABC):
    score_clip_value = None

    @abstractmethod
    def get_scores(self, logits, labels, t=1., **kwargs):
        '''Calculated the conformal scores.
        
        Returns a numpy array with labels shape'''
        pass

    @abstractmethod
    def get_sets(self, pred_scores, true_scores, pred_cls_logits,
                 t=1., use_pred_scores=True, **kwargs):
        '''Predict the set of classes given a score/threshold and the cls logits.
        
        Return list of class number tuples. 
        '''
        pass

    def calibrate_and_get_qhat(self, logits, labels, t=1., alpha=0.1, **kwargs):
        n = len(labels)
        scores = self.get_scores(logits, labels, t, **kwargs)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        return np.quantile(scores, q_level, method="higher")
    
    def calibrate_dls(self, train_dl, val_dl, t=1., alpha=0.1):
        train_logits = train_dl.dataset.cls_logits.numpy()
        train_labels = train_dl.dataset.cls_labels.numpy()

        val_logits = val_dl.dataset.cls_logits.numpy()
        val_labels = val_dl.dataset.cls_labels.numpy()
        return self.baseline_calibrate(
            train_logits, train_labels, val_logits, val_labels, t, alpha)

    def baseline_calibrate(
            self, train_logits, train_labels, val_logits, val_labels, t=1., alpha=0.1):
        qhat = self.calibrate_and_get_qhat(train_logits, train_labels, t, alpha)
        sets = self.get_sets(qhat, qhat, val_logits, t, use_pred_scores=True)
        mets = self.get_conformal_mets(sets, val_labels)
        mets['qhat'] = qhat
        return mets

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
        return {'set_size_mean': set_lens.mean(),
                'set_size_std': set_lens.std(),
                'acc_per_label_mean': acc_per_label.mean(),
                'acc_per_label_max': acc_per_label.max(),
                'acc_per_label_min': acc_per_label.min(),
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


class APSVersionTwoTODO(ConformalBase):

    @classmethod
    def _sort_sum(self, scores):
        I = scores.argsort(axis=1)[:,::-1]
        ordered = np.sort(scores,axis=1)[:,::-1]
        cumsum = np.cumsum(ordered,axis=1) 
        return I, ordered, cumsum

    @classmethod
    def _giq(self, scores, targets, I, ordered, cumsum, randomized, allow_zero_sets):
        """
            Generalized inverse quantile conformity score function.
            E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
        """
        E = -np.ones((scores.shape[0],))
        for i in range(scores.shape[0]):
            E[i] = self._get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:],randomized=randomized, allow_zero_sets=allow_zero_sets)

        return E

    @classmethod
    def _get_tau(self, score, target, I, ordered, cumsum, randomized, allow_zero_sets): # For one example
        idx = np.where(I==target)
        tau_nonrandom = cumsum[idx]

        if not randomized:
            return tau_nonrandom
        
        U = np.random.random()

        if idx == (0,0):
            if not allow_zero_sets:
                return tau_nonrandom
            else:
                return U * tau_nonrandom
        else:
            return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)]

    @classmethod
    def _gcq(self, scores, tau, I, ordered, cumsum, randomized, allow_zero_sets):
        if not isinstance(tau, float):
            tau = np.expand_dims(tau, 1)
            expended = True
        else:
            expended = False
        sizes_base = ((cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
        sizes_base = np.minimum(sizes_base, scores.shape[1]) # 1-1000

        if randomized:
            V = np.zeros(sizes_base.shape)
            for i in range(sizes_base.shape[0]):
                if expended:
                    _tau = tau[i]
                else:
                    _tau = tau
                V[i] = 1 / ordered[i, sizes_base[i] - 1] * \
                        (_tau - (cumsum[i, sizes_base[i] - 1] - ordered[i, sizes_base[i] - 1])) # -1 since sizes_base \in {1,...,1000}.

            sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
        else:
            sizes = sizes_base

        # TODO(royhirsch)
        # if tau == 1.0:
        #     sizes[:] = cumsum.shape[1] # always predict max size if alpha==0. (Avoids numerical error.)

        if not allow_zero_sets:
            sizes[sizes == 0] = 1 # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

        S = list()

        # Construct S from equation (5)
        for i in range(I.shape[0]):
            S = S + [I[i,0:sizes[i]],]

        return S

    def get_scores(self, logits, labels, t=1., randomized=True, allow_zero_sets=True):
        scores = softmax(logits/t, axis=1)
        I, ordered, cumsum = self._sort_sum(scores)
        return self._giq(scores, labels, I=I, ordered=ordered, cumsum=cumsum,
                        randomized=randomized, allow_zero_sets=allow_zero_sets).astype(float)

    def get_sets(self, pred_scores, true_scores, cls_logits, t=1., use_pred_scores=True,
                 randomized=True, allow_zero_sets=True):
        cls_logits = softmax(cls_logits / t , 1)
        scores = pred_scores if use_pred_scores else true_scores
        I, ordered, cumsum = self._sort_sum(cls_logits)
        sets = self._gcq(cls_logits, scores, I=I, ordered=ordered, cumsum=cumsum,
                     randomized=randomized, allow_zero_sets=allow_zero_sets)
        return sets

    def baseline_calibrate(self, train_dl, val_dl, t=1., alpha=0.1):
        train_logits = train_dl.dataset.cls_logits.numpy()
        train_labels = train_dl.dataset.cls_labels.numpy()
        qhat = self.calibrate_and_get_qhat(train_logits, train_labels, t,alpha=alpha)

        val_logits = val_dl.dataset.cls_logits.numpy()
        val_labels = val_dl.dataset.cls_labels.numpy()
        cls_logits = softmax(val_logits / t , 1)
        I, ordered, cumsum = self._sort_sum(val_logits)
        sets = self._gcq(cls_logits, qhat, I=I, ordered=ordered, cumsum=cumsum,
                     randomized=True, allow_zero_sets=False)
        mets = self.get_conformal_mets(sets, val_labels)
        mets['qhat'] = qhat
        return mets


class APS(ConformalBase):

    def get_scores(self, logits, labels, t=1., **kwargs):
        probs = softmax(logits / t, 1)
        scores = []
        for p, l in zip(probs, labels):
            true_class_p = p[l]
            score = np.sum(p[p >= true_class_p])
            scores.append(score)
        scores = np.asarray(scores)

        if self.score_clip_value:
            inds = np.where(scores >= self.score_clip_value)[0]
            logging.info(f'Clipping {len(inds) / len(scores) * 100.:.2f} of the scores to {self.score_clip_value:.2f}')
            scores[inds] = self.score_clip_value

        return scores

    def get_sets(self, pred_scores, true_scores, cls_logits, t=1., 
                 use_pred_scores=True, **kwargs):
        
        probs = softmax(cls_logits / t, 1)
        scores = pred_scores if use_pred_scores else true_scores
        if isinstance(scores, float) or scores.shape == ():
            scores = np.full((len(probs)), scores)

        sets = []
        for s, p, in zip(scores, probs):
            sorted_p = np.sort(p)[::-1]
            argsort_p = np.argsort(p)[::-1]
            cumsum_p = np.cumsum(sorted_p)
            inds = np.where(cumsum_p >= s)[0]
            if len(inds):
                sets.append(tuple(argsort_p[:inds[0] + 1]))
            else:
                print('Error')
        return sets

    def baseline_calibrate(
            self, train_logits, train_labels, val_logits, val_labels, t=1., alpha=0.1):
        
        qhat = self.calibrate_and_get_qhat(train_logits, train_labels, t, alpha)
        sets = self.get_sets(qhat, qhat, val_logits, t, use_pred_scores=True)
        mets = self.get_conformal_mets(sets, val_labels)
        mets['qhat'] = qhat
        return mets


class RandomizedAPS(ConformalBase):
    def __init__(self, no_zero_size_sets=True):
        self.no_zero_size_sets = no_zero_size_sets

    def _helper_get_sets(self, logits, qhat, t=1., randomized=True):
        probs = softmax(logits / t, 1)

        val_pi = probs.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(probs, val_pi, axis=1).cumsum(axis=1)
        if isinstance(qhat, np.ndarray):
            qhat = np.expand_dims(qhat, 1)

        if not randomized:
            prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
        else:
            n_val = val_srt.shape[0]
            cumsum_index = np.sum(val_srt <= qhat, axis=1)
            high = val_srt[np.arange(n_val), cumsum_index]
            low = np.zeros_like(high)
            low[cumsum_index > 0] = val_srt[np.arange(n_val), cumsum_index - 1][cumsum_index > 0]
            if isinstance(qhat, np.ndarray):
                qhat = np.squeeze(qhat)
            prob = (qhat - low) / (high - low)
            rv = np.random.binomial(1, prob, size=(n_val))
            randomized_threshold = low
            randomized_threshold[rv == 1] = high[rv == 1]
            if self.no_zero_size_sets:
                randomized_threshold = np.maximum(randomized_threshold, val_srt[:,0])
            prediction_sets = np.take_along_axis(val_srt <= randomized_threshold[:, None],
                                                 val_pi.argsort(axis=1), axis=1)
        sets = []
        for i in range(len(prediction_sets)):
            sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
        return sets

    def get_scores(self, logits, labels, t=1., randomized=True):
        n = len(labels)
        probs = softmax(logits / t, 1)
        cal_pi = probs.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(probs, cal_pi, axis=1).cumsum(axis=1)
        cal_softmax_correct_class = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
            range(n), labels
        ]
        if not randomized:
            scores = cal_softmax_correct_class
        else:
            cumsum_index = np.where(cal_srt == cal_softmax_correct_class[:,None])[1]
            high = cal_softmax_correct_class
            low = np.zeros_like(high)
            low[cumsum_index != 0] = cal_srt[np.where(cumsum_index != 0)[0], cumsum_index[cumsum_index != 0]-1]
            scores = np.random.uniform(low=low, high=high)
        return scores

    def get_sets(self, pred_scores, true_scores, cls_logits, t=1., 
                 use_pred_scores=True, randomized=True):
        
        scores = pred_scores if use_pred_scores else true_scores
        return self._helper_get_sets(cls_logits, scores, t, randomized)

    def baseline_calibrate(
            self, train_logits, train_labels, val_logits, val_labels, t=1., alpha=0.1):
        qhat = self.calibrate_and_get_qhat(train_logits, train_labels, t, alpha, randomized=True)
        sets = self.get_sets(qhat, qhat, val_logits, t, use_pred_scores=True, randomized=True)
        mets = self.get_conformal_mets(sets, val_labels)
        mets['qhat'] = qhat
        return mets


class Naive(ConformalBase):

    def get_scores(self, logits, labels, t=1., **kwargs):
        n = len(labels)
        probs = softmax(logits / t, 1)
        return 1. - probs[np.arange(n), labels]

    def get_sets(self, pred_scores, true_scores, pred_cls_logits,
                 t=1., use_pred_scores=True, **kwargs):
        probs = softmax(pred_cls_logits / t, 1)
        scores = pred_scores if use_pred_scores else true_scores

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
    elif conformal_module_name == 'rand_aps':
        return RandomizedAPS()
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

