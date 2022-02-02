from compute_mAP import compute_mAP
import numpy as np

def get_latency_from_LiteReconfig(filename):
    
    # Extract the mean and std of latency in GoF granularity, ignore incomplete GoFs
    column_dict = {"is_det_frame": 3, "si": 4, "lat_detection": 10, "lat_tracker": 11, "lat_overhead": 12}
    with open(filename) as fin:
        lines = fin.readlines()
    is_det_frames = [idx for idx, line in enumerate(lines) if eval(line.split()[column_dict["is_det_frame"]])]
    is_det_frames.append(len(lines))
    st_en = [(st, en) for st, en in zip(is_det_frames[:-1], is_det_frames[1:])]
    sis = [int(lines[st].split()[column_dict["si"]]) for st, en in st_en]
    lats_overhead, lats_detection, lats_tracker, lats_GoF = [], [], [], []
    lats_overhead_out, lats_detection_out, lats_tracker_out, lats_GoF_out = [], [], [], []
    for (st, en), si in zip(st_en, sis):
        lats_overhead0 = [float(line.split()[column_dict["lat_overhead"]]) for line in lines[st:en]]
        lats_detection0 = [float(line.split()[column_dict["lat_detection"]]) for line in lines[st:en]]
        lats_tracker0 = [float(line.split()[column_dict["lat_tracker"]]) for line in lines[st:en]]
        lat_mean0 = np.mean(lats_overhead0) + np.mean(lats_detection0) + np.mean(lats_tracker0)
        if en-st == si:
            lats_GoF.append(lat_mean0)
    return np.mean(lats_GoF), np.std(lats_GoF), np.percentile(lats_GoF, 95)

filename = "offline_logs_AE/executor_LiteReconfig_g0_lat33_tx2_lat.txt"
lat_mean, _, _ = get_latency_from_LiteReconfig(filename)
filename = "offline_logs_AE/executor_LiteReconfig_g0_lat33_tx2_det.txt"
meanAP = compute_mAP(gt="test/VID_testgt_full.txt", detection=filename)
print(f"LiteReconfig, no contention: {meanAP*100:.1f}% mAP, {lat_mean:.1f} ms mean latency.")
