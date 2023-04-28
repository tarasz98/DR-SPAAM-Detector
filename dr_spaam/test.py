import numpy as np
import matplotlib.pyplot as plt
from dr_spaam.detector import Detector

# Detector class wraps up preprocessing, inference, and postprocessing for DR-SPAAM.
# Checkout the comment in the code for meanings of the parameters.
ckpt = 'ckpts/dr_spaam_e40.pth'
detector = Detector(
    model_name="DR-SPAAM", 
    ckpt_file=ckpt, 
    gpu=True, 
    stride=1, 
    tracking=False
)

# set angular grid (this is only required once)
ang_inc = np.radians(0.5)  # angular increment of the scanner
num_pts = 450  # number of points in a scan
detector.set_laser_spec(ang_inc, num_pts)

odo_idx = 0
# inference
while True:
    scan = np.random.rand(num_pts)  # scan is a 1D numpy array with positive values
    dets_xy, dets_cls, instance_mask = detector(scan)  # get detection

    # confidence threshold
    cls_thresh = 0.2
    cls_mask = dets_cls > cls_thresh
    cls_mask = cls_mask.flatten('F')
    print(f"Detections: {dets_xy}")
    print(f"Detection mask: {cls_mask}")
    dets_xy = dets_xy[cls_mask]
    dets_cls = dets_cls[cls_mask]

    if 0xFF == ord('q'):
        break