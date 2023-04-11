import numpy as np
from mf_sort import MF_SORT, Detection
from tqdm import trange

detections_path = "example_data/PETS09-S2L1.txt"
output_path = "example_data/PETS09-S2L1_track.txt"

all_dets = np.loadtxt(detections_path, delimiter=",")
frame_range = int(np.max(all_dets[:,0]))

# These are default parameters and can be changed
mot = MF_SORT()
track_output = []

for frame_num in trange(frame_range, unit="frames"):
    # Extract detections from this frame
    dets = all_dets[all_dets[:, 0] == frame_num]
    dets = [Detection(r[2:6], r[6]) for r in dets if r[1] == 0]

    # Run the tracker
    trks = mot.step(dets)
    
    for trk, trk_ID in trks:
        tlwh = np.squeeze(trk.tlwh.copy().astype("int"))
        track_output.append([frame_num, trk_ID, tlwh[0], tlwh[1], tlwh[2], tlwh[3], 1, -1, -1, -1])

trks = np.array(track_output)
np.savetxt(output_path, trks, delimiter=",", fmt="%d")