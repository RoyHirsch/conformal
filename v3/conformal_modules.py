import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def platt_scale(logits, labels, max_iters=100, lr=0.01, epsilon=0.005):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(logits),
                                             torch.from_numpy(labels).long()) 
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True)

    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.]).cuda())
    optimizer = optim.SGD([T], lr=lr)
    for iter in tqdm(range(max_iters)):
        T_old = T.item()
        for x, targets in dataloader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T.item() 


class ConformalScore():
    def __init__(self, **kwargs):
        pass
    
    def get_scores(self, softmax_scores, labels, **kwargs):
        raise NotImplementedError

    def get_sets(self, softmax_scores, qhat, **kwargs):
        raise NotImplementedError


class APS(ConformalScore):
    def __init__(self, randomized=True, no_zero_size_sets=True, seed=0):
        self.randomized = randomized
        self.no_zero_size_sets = no_zero_size_sets
        self.seed = seed

    def get_scores(self, softmax_scores, labels):
        cal_pi = softmax_scores.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(softmax_scores, cal_pi, axis=1).cumsum(axis=1)
        cal_softmax_correct_class = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
            range(len(labels)), labels
        ]
        if not self.randomized:
            cal_scores = cal_softmax_correct_class
        else:
            np.random.seed(self.seed)
            cumsum_index = np.where(cal_srt == cal_softmax_correct_class[:,None])[1]
            if cumsum_index.shape[0] != cal_srt.shape[0]:
                _, unique_indices = np.unique(np.where(
                    cal_srt == cal_softmax_correct_class[:,None])[0], return_index=True)
                cumsum_index = cumsum_index[unique_indices]
            high = cal_softmax_correct_class
            low = np.zeros_like(high)
            low[cumsum_index != 0] = cal_srt[np.where(cumsum_index != 0)[0], cumsum_index[cumsum_index != 0]-1]
            cal_scores = np.random.uniform(low=low, high=high)
        return cal_scores


    def get_sets(self, softmax_scores, qhat):
        val_pi = softmax_scores.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(softmax_scores, val_pi, axis=1).cumsum(axis=1)
        if not self.randomized:
            prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
        else:
            np.random.seed(self.seed)
            n_val = val_srt.shape[0]
            if qhat.ndim == 1:
                cumsum_index = np.sum(val_srt <= np.expand_dims(qhat, 1), axis=1)
            else:
                cumsum_index = np.sum(val_srt <= qhat, axis=1)
            high = val_srt[np.arange(n_val), cumsum_index]
            low = np.zeros_like(high)
            low[cumsum_index > 0] = val_srt[np.arange(n_val), cumsum_index-1][cumsum_index > 0]
            prob = (qhat - low)/(high - low)
            # num_nans = np.isnan(prob).sum()
            # print("Number of NaNs:", num_nans)
            # prob = np.nan_to_num(prob, nan=0.0)
            # prob = np.clip(prob, 0, 1)
            rv = np.random.binomial(1,prob,size=(n_val))
            randomized_threshold = low
            randomized_threshold[rv == 1] = high[rv == 1]
            if self.no_zero_size_sets:
                randomized_threshold = np.maximum(randomized_threshold, val_srt[:,0])
            prediction_sets = np.take_along_axis(val_srt <= randomized_threshold[:,None], val_pi.argsort(axis=1), axis=1)
        return prediction_sets


class RAPS(ConformalScore):
    def __init__(self, lam_reg=0.01, k_reg=5, randomized=True, no_zero_size_sets=True, seed=0):
        self.lam_reg = lam_reg
        self.k_reg = k_reg
        self.randomized = randomized
        self.no_zero_size_sets = no_zero_size_sets
        self.seed = seed

    def get_scores(self, softmax_scores, labels):
        np.random.seed(self.seed)
        reg_vec = np.array(self.k_reg * [0, ] + (softmax_scores.shape[1] - self.k_reg) * [self.lam_reg, ])[None, :]
        n = softmax_scores.shape[0]
        cal_pi = softmax_scores.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(softmax_scores, cal_pi, axis=1).cumsum(axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_L = np.where(cal_pi == labels[:,None])[1]
        cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
        return cal_scores


    def get_sets(self, softmax_scores, qhat):
        np.random.seed(self.seed)
        reg_vec = np.array(self.k_reg * [0, ] + (softmax_scores.shape[1] - self.k_reg) * [self.lam_reg, ])[None, :]
        n = softmax_scores.shape[0]

        val_pi = softmax_scores.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(softmax_scores, val_pi, axis=1).cumsum(axis=1)
        val_srt_reg = val_srt + reg_vec
        if qhat.ndim == 1:
            qhat = np.expand_dims(qhat, 1)
        if self.randomized:
            indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n, 1) * val_srt_reg) <= qhat
        else:
              indicators = val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
        if self.no_zero_size_sets:
            indicators[:,0] = True
        prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1),axis=1)
        return prediction_sets


class Naive():
    def __init__(self, **kwargs):
        pass
    
    def get_scores(self, softmax_scores, labels, **kwargs):
        return 1 - softmax_scores[np.arange(len(labels)), labels]

    def get_sets(self, softmax_scores, qhat, **kwargs):
        if qhat.ndim == 1:
                qhat = np.expand_dims(qhat, 1)
        return softmax_scores >= (1 - qhat)


class SAPS():
    def __init__(self, weight=1, randomized=True, no_zero_size_sets=True, seed=0):
        self.weight = weight
        self.randomized = randomized
        self.no_zero_size_sets = no_zero_size_sets
        self.seed = seed

    def _sort_sum(self, probs):
        indices = np.argsort(-probs, axis=1)  # Get the indices that would sort the array
        ordered = np.take_along_axis(probs, indices, axis=1)          # Sort the array using the indices
        cumsum = np.cumsum(ordered, axis=1)        # Compute the cumulative sum of the sorted array
        return indices, ordered, cumsum

    def get_scores(self, softmax_scores, labels):
        indices, ordered, cumsum = self._sort_sum(softmax_scores)
        # Generate random values U from a uniform distribution with the same shape as the batch
        U = np.random.rand(*indices.shape)
        # Find the index where the sorted indices equal the label for each row
        idx = np.where(indices == labels[:, np.newaxis])
        # Compute the scores
        scores_first_rank = U[idx] * cumsum[idx]
        scores_usual = self.weight * (idx[1] - U[idx]) + ordered[:, 0]
        # Return the appropriate scores based on the condition
        return np.where(idx[1] == 0, scores_first_rank, scores_usual)

    def get_sets(self, softmax_scores, qhat):
        indices, ordered, cumsum = self._sort_sum(softmax_scores)
        # Set the weights for all but the first column
        ordered[:, 1:] = 1
        # Recalculate the cumulative sum with the new weights
        cumsum = np.cumsum(ordered, axis=-1)
        # Generate random values U from a uniform distribution with the same shape as probs
        U = np.random.rand(*softmax_scores.shape)
        # Calculate the ordered scores
        ordered_scores = cumsum - ordered * U
        # Sort indices to map back to the original order
        sorted_indices = np.argsort(indices, axis=-1)
        # Gather the scores according to the original indices
        scores = np.take_along_axis(ordered_scores, sorted_indices, axis=-1)
        if qhat.ndim == 1:
            qhat = np.expand_dims(qhat, 1)
        return scores <= qhat
