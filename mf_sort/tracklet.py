import numpy as np
from .detection import Detection

class Tracklet():
    def __init__(self, id, mean, cov, min_hits=3):
        self.id = id
        
        self.min_hits = min_hits
        self.hit_streak = 1  # Number of consecutive detections for the tracker
        self.time_since_update = 0
        self.age = 1
        self.in_probation = True

        self.mean = mean
        self.cov = cov
        self.history = []


    def predict(self, kf):
        self.time_since_update += 1
        self.age += 1

        self.mean, self.cov = kf.predict(self.mean, self.cov)

    def update(self, kf, detection):
        self.hit_streak += 1
        self.time_since_update = 0
        if self.hit_streak >= self.min_hits:
            self.in_probation = False

        self.mean, self.cov = kf.update(self.mean, self.cov, detection.to_xyah())

    def get_bbox(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        
        return Detection(ret, 1.0)

    def get_state(self):
        return self.mean, self.cov

    def record_state(self):
        state = self.get_state()
        self.history.append(state)
