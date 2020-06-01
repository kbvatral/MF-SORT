import numpy as np
from mf_sort import MF_SORT, Detection
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

    # Format output to MOT Benchmark standard
    for trk, trk_num in trackers:
        x1 = int(trk.tlwh[0])
        y1 = int(trk.tlwh[1])
        w = int(trk.tlwh[2])
        h = int(trk.tlwh[3])

        trk_out = (frame_num, trk_num, x1, y1, w, h, 1, -1, -1, -1)
        tracks_by_frame.append(trk_out)

trks = np.array(tracks_by_frame)
np.savetxt(output_path, trks, delimiter=",", fmt="%d")