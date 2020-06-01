import numpy as np
from mf_sort import MF_SORT, Detection
from mf_sort.utils import format_MOTChallenge
from tqdm import trange

detections_path = "example_data/overpass_detections.txt"
output_path = "example_data/overpass_track.txt"

all_dets = np.loadtxt(detections_path, delimiter=",")
frame_range = int(np.max(all_dets[:,0]))

# These are default parameters and can be changed
mot = MF_SORT(max_age=5, min_hits=3, gating_threshold=0.7)
tracks_by_frame = []

for frame_num in trange(frame_range, unit="frames"):
    # Extract detections from this frame
    dets = all_dets[all_dets[:, 0] == frame_num]
    dets = [Detection(r[2:6], r[6]) for r in dets]

    # Run the tracker
    mot.predict()
    trackers = mot.update(dets) # Returns tuples of (mf_sort.Detection, trk_num)
    # Use MOT Challenge formatting
    trackers = format_MOTChallenge(frame_num, trackers)
    tracks_by_frame.append(trackers)

trks = np.concatenate(tracks_by_frame)
np.savetxt(output_path, trks, delimiter=",", fmt="%d")