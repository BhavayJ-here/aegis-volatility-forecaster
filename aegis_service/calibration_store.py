from collections import deque
import numpy as np


class CalibrationStore:
    def __init__(self, initial_scores, alpha=0.10, window_size=500, gamma=0.01):
        self.alpha = alpha
        self.alpha_t = alpha
        self.gamma = gamma
        self.scores = deque(initial_scores[-window_size:], maxlen=window_size)
        self.q_hat = self.calculate_q_hat()

    def calculate_q_hat(self):
        n = len(self.scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha_t)) / n)
        return float(np.quantile(list(self.scores), level))

    def widen_interval(self, raw_lower, raw_upper):
        self.q_hat = self.calculate_q_hat()
        return raw_lower - self.q_hat, raw_upper + self.q_hat

    def update_after_actual(self, raw_lower, raw_upper, actual_value, was_covered):
        score = max(raw_lower - actual_value, actual_value - raw_upper)
        self.scores.append(score)

        # move alpha_t up if we were wrong, down if we were right
        error_t = 0 if was_covered else 1
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - error_t)
        self.alpha_t = min(max(self.alpha_t, 0.001), 0.5)

        self.q_hat = self.calculate_q_hat()


class MultiHorizonCalibrationStore:
    def __init__(self, initial_scores_by_horizon, alpha=0.10, window_size=500, gamma=0.01):
        self.stores = {}
        for horizon, scores in initial_scores_by_horizon.items():
            self.stores[horizon] = CalibrationStore(scores, alpha=alpha, window_size=window_size, gamma=gamma)

    def get_store(self, horizon):
        return self.stores[horizon]