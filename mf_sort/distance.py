import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from filterpy.stats import mahalanobis

def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes/detections
    """
    bb_test = bb_test.to_tlbr()
    bb_gt = bb_gt.to_tlbr()

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o


def iou_cost_function(detections, trackers):
    cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            cost_matrix[d, t] = 1 - iou(det, trk.get_bbox())
            
    return cost_matrix


def cascade_cost(detections, trackers):
    cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            mean, cov = trk.get_state()
            cost_matrix[d, t] = mahalanobis(det.to_xyah(), mean, cov)**2

    return cost_matrix
