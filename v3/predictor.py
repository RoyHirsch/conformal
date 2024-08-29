import warnings
import math
import torch

from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores.base import BaseScore
from torch.utils.data import TensorDataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class APS(BaseScore):
    """
    Adaptive Prediction Sets (Romano et al., 2020)
    paper :https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf
    """

    def __call__(self, logits, label=None, seed=42):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if label is None:
            return self._calculate_all_label(probs)
        else:
            set_seed(seed)
            return self._calculate_single_label(probs, label)

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        # ordered: the ordered probabilities in descending order
        # indices: the rank of ordered probabilities in descending order
        # cumsum: the accumulation of sorted probabilities
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        scores_first_rank = U * cumsum[idx]
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
    

class RAPS(APS):
    """
    Regularized Adaptive Prediction Sets (Angelopoulos et al., 2020)
    paper : https://arxiv.org/abs/2009.14193
    
    :param penalty: the weight of regularization. When penalty = 0, RAPS=APS.
    :param kreg: the rank of regularization which is an integer in [0,labels_num].
    """

    def __init__(self, penalty, kreg=0):
        
        if penalty <= 0:
            raise ValueError("The parameter 'penalty' must be a positive value.")
        if kreg < 0:
            raise ValueError("The parameter 'kreg' must be a nonnegative value.")
        if type(kreg) != int:
            raise TypeError("The parameter 'kreg' must be a integer.")
        super(RAPS, self).__init__()
        self.__penalty = penalty
        self.__kreg = kreg

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores
    
    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        scores_first_rank = U * ordered[idx] + reg
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one] + reg
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


class SAPS(APS):
    """
    Sorted Adaptive Prediction Sets (Huang et al., 2023)
    paper: https://arxiv.org/abs/2310.06430
    
    :param weight: the weight of label ranking.
    """

    def __init__(self, weight):

        super(SAPS, self).__init__()
        if weight <= 0:
            raise ValueError("The parameter 'weight' must be a positive value.")
        self.__weight = weight

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        ordered[:, 1:] = self.__weight
        cumsum = torch.cumsum(ordered, dim=-1)
        U = torch.rand(probs.shape, device=probs.device)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        scores_first_rank = U * cumsum[idx]
        scores_usual = self.__weight * (idx[1] - U) + ordered[:, 0]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


class SplitPredictorVectorThreshold(SplitPredictor):
    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)

    def predict_with_logits(self, logits, q_hat=None):
        scores = self.score_function(logits).to(self._device)
        if q_hat is None:
            q_hat = self.q_hat
        if isinstance(q_hat, float):
            q_hat = torch.full_like(logits, q_hat)
        q_hat = q_hat.to(self._device)
        S = self._generate_prediction_set(scores, q_hat)
        
        return S

    def _generate_prediction_set(self, scores, q_hat):
        return [torch.argwhere(scores[i] <= q_hat[i]).reshape(-1).tolist() for i in range(scores.shape[0])]
    
    def set_deivce(self, device):
        self._device = device


if __name__ == '__main__':
    import torch.nn as nn
    from torchcp.classification.scores import APS

    x = nn.Softmax(1)(torch.randn(100, 10))
    y = torch.argmax(x, 1)
    
    x_test = nn.Softmax(1)(torch.randn(100, 10))
    y_test = torch.argmax(x_test, 1)
    
    predictor = SplitPredictorVectorThreshold(APS())
    predictor.calculate_threshold(x, y, 0.1)
    prediction_sets = predictor.predict_with_logits(x_test)
    res_dict = {"Coverage_rate": predictor._metric('coverage_rate')(prediction_sets, y_test),
            "Average_size": predictor._metric('average_size')(prediction_sets, y_test)}
    print(res_dict)
    
    prediction_sets = predictor.predict_with_logits(x_test, torch.randn(100))
    res_dict = {"Coverage_rate": predictor._metric('coverage_rate')(prediction_sets, y_test),
            "Average_size": predictor._metric('average_size')(prediction_sets, y_test)}
    print(res_dict)



