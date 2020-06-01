import numpy as np
from operator import itemgetter
from .kalman_filter import KalmanFilter
from .utils import hungarian_matching, iou
from .tracklet import Tracklet


class MF_SORT(object):
    def __init__(self, max_age=5, min_hits=3, gating_threshold=0.7):
        self.max_age = max_age
        self.min_hits = min_hits
        self.gating_threshold = gating_threshold
        self.tracklet_counter = 0

        self.kf = KalmanFilter()
        self.trackers = []
        self.lost_trackers = []

    def get_trackers(self):
        return self.trackers, self.lost_trackers

    def _matching_cascade(self, dets):
        unmatched_dets = dets.copy()
        matched_trks = set()
        unmatched_trks = set()

        # First round matching with cascade by age for confirmed trackers
        for l in range(self.max_age):
            trk_l = [trk for trk in self.trackers if trk.time_since_update == l and not trk.in_probation]
            matches_l, unmatched_dets, unmatched_trks_l = hungarian_matching(self.kf, unmatched_dets, trk_l, 9.488)
            matched_trks.update(matches_l)
            unmatched_trks.update(unmatched_trks_l)

        # Second round matching on remaining detections, trackers in probation,
        # and still unmatched trackers
        unconfirmed_trks = set([trk for trk in self.trackers if trk.in_probation])
        unmatched_trks.update(unconfirmed_trks)

        matches, unmatched_dets, unmatched_trks = hungarian_matching(self.kf, unmatched_dets, unmatched_trks, 13.277)
        matched_trks.update(matches)

        return matched_trks, unmatched_dets, unmatched_trks

    def _init_tracklet(self, dets):
        new_trackers = []
        for det in dets:
            # Gate detection by IOU with current trackers
            create = True
            for trk in self.trackers:
                if iou(det, trk.get_bbox()) > 0.7:
                    create = False
                    break
            if create:
                mean, cov = self.kf.initiate(det.to_xyah())
                trk = Tracklet(self.tracklet_counter, mean, cov, self.min_hits)
                new_trackers.append(trk)
                self.tracklet_counter += 1

        self.trackers += new_trackers

    def predict(self):
        for trk in self.trackers:
            trk.predict(self.kf)
        
    def update(self, dets):
        # Associate predictions to observations
        matched, unmatched_dets, unmatched_trks = self._matching_cascade(dets)
        new_trackers = []

        # Update matched trackers with assigned detections
        for det, trk in matched:
            trk.update(self.kf, det)
            new_trackers.append(trk)

        # Delete trackers with too high age
        for trk in unmatched_trks:
            if trk.time_since_update > self.max_age:
                self.lost_trackers.append(trk)
            else:
                new_trackers.append(trk)

        # Update the active trackers
        self.trackers = new_trackers

        # Create new trackers for unmatched detections
        self._init_tracklet(unmatched_dets)

        # Format output
        ret = []
        for trk in self.trackers:
            # If alive and not in probation, add it to the return
            if not trk.in_probation:
                d = trk.get_bbox()
                ret.append((d, trk.id))

        return ret
