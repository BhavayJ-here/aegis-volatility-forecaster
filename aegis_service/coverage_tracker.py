from collections import deque


class CoverageTracker:
    def __init__(self, window_size=200):
        self.recent_results = deque(maxlen=window_size)

    def record(self, was_covered):
        self.recent_results.append(1 if was_covered else 0)

    @property
    def rolling_coverage(self):
        if len(self.recent_results) == 0:
            return None
        return sum(self.recent_results) / len(self.recent_results)