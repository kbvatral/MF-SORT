import numpy as np
from filterpy.kalman import KalmanFilter

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

def construct_position_filter(bbox):
    xyah = bbox.to_xyah()

    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.x[:4] = xyah.reshape((4, 1))

    # F is the state transition function
    kf.F = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ])

    # H is the measurment function
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ])

    # P is Covariance matrix
    _std_weight_position = 1. / 20
    _std_weight_velocity = 1. / 160
    std = [
        2 * _std_weight_position * xyah[3],
        2 * _std_weight_position * xyah[3],
        1e-2,
        2 * _std_weight_position * xyah[3],
        10 * _std_weight_velocity * xyah[3],
        10 * _std_weight_velocity * xyah[3],
        1e-5,
        10 * _std_weight_velocity * xyah[3]
    ]
    kf.P = np.diag(np.square(std))

    # R is measurement noise
    std = [
        _std_weight_position * xyah[3],
        _std_weight_position * xyah[3],
        1e-1,
        _std_weight_position * xyah[3]
    ]
    kf.R = np.diag(np.square(std))

    # Q is process noise
    std_pos = [
        _std_weight_position * xyah[3],
        _std_weight_position * xyah[3],
        1e-2,
        _std_weight_position * xyah[3]]
    std_vel = [
        _std_weight_velocity * xyah[3],
        _std_weight_velocity * xyah[3],
        1e-5,
        _std_weight_velocity * xyah[3]]
    kf.Q = np.diag(np.square(np.r_[std_pos, std_vel]))

    return kf



def project(kf):
    mean = np.dot(kf.H, kf.x)
    cov = np.linalg.multi_dot((kf.H, kf.P, kf.H.T))
    return mean, cov+kf.R