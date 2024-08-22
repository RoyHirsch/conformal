import warnings
import math
import torch

from torchcp.classification.predictors import SplitPredictor
from torch.utils.data import TensorDataset
import torch


class SplitPredictorVectorThreshold(SplitPredictor):
    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)

    def predict_with_logits(self, logits, q_hat=None):
        scores = self.score_function(logits).to(self._device)
        if q_hat is None:
            q_hat = self.q_hat
        if q_hat.ndim == 0:
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



