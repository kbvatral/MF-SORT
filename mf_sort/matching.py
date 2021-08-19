from .distance import cascade_cost
from .kalman_utils import chi2inv95

from scipy.optimize import linear_sum_assignment
import numpy as np


def matching_cascade(max_age, tracks, dets, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(dets)))

    unmatched_det_ids = detection_indices
    matched_trks = []

    for a in range(1, max_age+1):
        if len(unmatched_det_ids) == 0:
            break
        if len(matched_trks) >= len(track_indices):
            break

        trk_ids = [t for t in track_indices if tracks[t].time_since_update == a]
        if len(trk_ids) == 0:
            continue

        matches, _, unmatched_det_ids = \
            min_cost_matching(cascade_cost, chi2inv95[4], tracks, dets, trk_ids, unmatched_det_ids)
        matched_trks += matches

    unmatched_track_ids = list(set(track_indices) - set(k for k, _ in matched_trks))
    
    return matched_trks, unmatched_track_ids, unmatched_det_ids

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices, detection_indices):
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    valid_dets = [detections[d] for d in detection_indices]
    valid_tracks = [tracks[t] for t in track_indices]
    cost_matrix = distance_metric(valid_dets, valid_tracks)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    matched_indices = linear_sum_assignment(cost_matrix)
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, d_idx in enumerate(detection_indices):
        if col not in matched_indices[: ,0]:
            unmatched_detections.append(d_idx)
    for row, t_idx in enumerate(track_indices):
        if row not in matched_indices[:, 1]:
            unmatched_tracks.append(t_idx)

    for row, col in matched_indices:
        trk_idx = track_indices[col]
        det_idx = detection_indices[row]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(trk_idx)
            unmatched_detections.append(det_idx)
        else:
            matches.append((trk_idx, det_idx))

    return matches, unmatched_tracks, unmatched_detections