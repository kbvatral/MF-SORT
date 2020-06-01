import numpy as np
import cv2
from tqdm import trange
import imutils.video

video_path = "example_data/overpass.mp4"
tracks_path = "example_data/overpass_track.txt"
output_path = "example_data/overpass_track.avi"

# Load the tracks generated from MF-SORT
tracks = np.loadtxt(tracks_path, delimiter=",", dtype="int")
num_tracks = tracks[:, 1].max()
COLORS = np.random.randint(0, 255, size=(num_tracks+1, 3), dtype="int")

# Setup Video File
vs = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
W, H = None, None
writer = None
frame_total = imutils.video.count_frames(video_path)

# Loop through video fames with progress bar
for frame_number in trange(1, frame_total+1, unit="frames"):
    grabbed, frame = vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        H, W = frame.shape[:2]
    if writer is None:
        writer = cv2.VideoWriter(output_path, fourcc, 15, (W, H), True)

    # Extract the Tracks for the current frame
    frame_tracks = tracks[tracks[:, 0] == frame_number]
    for trk in frame_tracks:
        trk_ID, x, y, w, h = trk[1:6]

        # Draw a bounding box rectangle and label
        color = tuple(int(i) for i in COLORS[trk_ID]) # openCV is picky about data format for color
        text = "Tracker {}".format(trk_ID)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the frame to file
    writer.write(frame)

writer.release()
vs.release()