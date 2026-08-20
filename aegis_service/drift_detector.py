class DriftDetector:
    def __init__(self, delta=0.005, threshold=5.0):
        self.delta = delta
        self.threshold = threshold
        self.running_average_error = 0.0
        self.num_points_seen = 0
        self.running_total = 0.0
        self.lowest_total_seen = 0.0

    def check(self, was_wrong):
        self.num_points_seen += 1
        self.running_average_error += (was_wrong - self.running_average_error) / self.num_points_seen
        self.running_total += was_wrong - self.running_average_error - self.delta
        self.lowest_total_seen = min(self.lowest_total_seen, self.running_total)

        # only fire if the gap from our best point gets big enough
        drift_amount = self.running_total - self.lowest_total_seen
        if drift_amount > self.threshold:
            self.reset()
            return True

        return False

    def reset(self):
        self.running_average_error = 0.0
        self.num_points_seen = 0
        self.running_total = 0.0
        self.lowest_total_seen = 0.0