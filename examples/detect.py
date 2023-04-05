import numpy as np
import cv2
from tqdm import trange
from mf_sort.detector import Detector

video_path = "example_data/PETS09-S2L1.mp4"
out_path = "example_data/PETS09-S2L1.txt"

vs = cv2.VideoCapture(video_path)
num_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

detector = Detector("models/crowdhuman.pt", device="cuda")
all_detections = []

for i in trange(num_frames):
    grabbed, frame = vs.read()
    if not grabbed:
        break

    detections = detector([frame])[0]
    for det in detections:
        t,l,w,h = det.tlwh
        conf = det.confidence
        cls = det.cls
        all_detections.append([i, cls, t, l, w, h, conf, -1, -1, -1])

np.savetxt(out_path, np.array(all_detections), fmt=["%d","%d","%d","%d","%d","%d","%.2f", "%d", "%d", "%d"], delimiter=",")
vs.release()