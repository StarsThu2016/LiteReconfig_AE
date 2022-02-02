'''
LiteReconfig_MaxContent: LiteReconfig with one particular feature always on.

Usage:
python LiteReconfig_MaxContent.py --protocol SmartAdapt_RPN --gl 0 --lat_req 33.3 \
  --mobile_device=tx2 --output=test/executor_LiteReconfig_MaxContent_ResNet.txt
'''

import argparse
import numpy as np
from tqdm import tqdm
from helper_online import SchedulerOnline, MBODF

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate SmartAdapt.')
parser.add_argument('--protocol', dest='protocol', required=True, help='Protocol name.')
parser.add_argument('--gl', dest='gl', type=int, required=True, help='GPU contention level.')
parser.add_argument('--lat_req', dest='lat_req', type=float, help='Latency requirement in msec.')
parser.add_argument('--mobile_device', dest='mobile_device', required=True, help='tx2 or xv.')
parser.add_argument('--quick', dest='quick', type=int, help='Whether to run on 10% dataset.')
parser.add_argument('--output', dest='output', help='Output filename.')
parser.add_argument('--tv_version', dest='tv_version', help='torchvision version, e.g., 0.5.0')
parser.add_argument('--dataset_prefix', dest='dataset_prefix', help='Path to ILSVRC2015 dir.')
args = parser.parse_args()

metadata = "test/VID_testvideo_V2.txt"
with open(metadata) as fin:
    lines = fin.readlines()
if args.quick:  # 10% test dataset
    lines = lines[::10]
video_dirs = [x.strip().split()[0] for x in lines]
frame_cnts = {line.split()[0]:int(line.split()[1]) for line in lines}
contention_levels = {'cpu_level': 0, 'mem_bw_level': 0, 'gpu_level': args.gl}
scheduler = SchedulerOnline(user_requirement=args.lat_req, p95_requirement=True,
                            protocol=args.protocol, mobile_device=args.mobile_device,
                            tv_version=args.tv_version, dataset_prefix=args.dataset_prefix)
filename_pre = args.output.rsplit(".", 1)[0]
filename_det = f"{filename_pre}_lat{int(args.lat_req)}_g{args.gl}_{args.mobile_device}_det.txt"
filename_lat = f"{filename_pre}_lat{int(args.lat_req)}_g{args.gl}_{args.mobile_device}_lat.txt"
with open(filename_det, "w") as fout_det, open(filename_lat, "w") as fout_lat:
    executor = MBODF(feat=scheduler.feat, kernel="FRCNN+", frcnn_weight="models/ApproxDet.pb",
                     fout_det=fout_det, fout_lat=fout_lat)
    tqdm_desc = f"{args.protocol}, g{args.gl}/{args.lat_req:.1f} ms lat_req, on {len(video_dirs)} videos"
    for video_idx, video_dir in tqdm(enumerate(video_dirs), desc=tqdm_desc):
        frame_cnt, frame_idx = frame_cnts[video_dir], 0
        feature_cache = {"nobj": 0, "objsize": 0, "RPN": np.zeros((1024,)), "CPoP": np.zeros((31,))}
        while frame_idx < frame_cnt:
            config, img_pil, run_log = scheduler.schedule(video_dir, video_idx, frame_idx,
                                                          feature_cache, contention_levels)
            si, shape, nprop, tracker_name, ds = config
            frame_cnt_GoF = min(si, frame_cnt-frame_idx)
            executor.run(config, frame_cnt_GoF, video_dir, frame_idx, img_pil, feature_cache, run_log)
            frame_idx += frame_cnt_GoF
