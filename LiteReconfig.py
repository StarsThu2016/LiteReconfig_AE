'''
LiteReconfig w/ the cost benefit analyzer, a.k.a. the full version.

Example usage:
python LiteReconfig.py --gl 0 --lat_req 33.3 --mobile_device=tx2
'''

import argparse
import numpy as np
from tqdm import tqdm
from helper_online_dev import SchedulerCBOnline
from helper_online import MBODF

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate SmartAdapt_CB.')
parser.add_argument('--gl', dest='gl', type=int, required=True, help='GPU contention level.')
parser.add_argument('--lat_req', dest='lat_req', type=float, help='Latency requirement in msec.')
parser.add_argument('--mobile_device', dest='mobile_device', required=True, help='tx2 or xv.')
parser.add_argument('--cost_filename', dest='cost_filename',
  default='models/SmartAdapt_cost_20211009.pb', help='Cost weight file.')
parser.add_argument('--benefit_filename', dest='benefit_filename',
  default='models/SmartAdapt_benefit_20211009.pb', help='Benefit weight file.')
parser.add_argument('--quick', dest='quick', type=int, help='Whether to run on 10% dataset.')
parser.add_argument('--output', dest='output', default='test/executor_LiteReconfig.txt',
  help='Output filename.')
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
scheduler = SchedulerCBOnline(contention_levels=contention_levels, user_requirement=args.lat_req,
                              cost_filename=args.cost_filename,
                              benefit_filename=args.benefit_filename,
                              p95_requirement=True, mobile_device=args.mobile_device,
                              tv_version=args.tv_version, dataset_prefix=args.dataset_prefix)
filename_pre = args.output.rsplit(".", 1)[0]
filename_det = f"{filename_pre}_lat{int(args.lat_req)}_g{args.gl}_{args.mobile_device}_det.txt"
filename_lat = f"{filename_pre}_lat{int(args.lat_req)}_g{args.gl}_{args.mobile_device}_lat.txt"
with open(filename_det, "w") as fout_det, open(filename_lat, "w") as fout_lat:
    executor = MBODF(feat="RPN", kernel="FRCNN+", frcnn_weight="models/ApproxDet.pb",
                     fout_det=fout_det, fout_lat=fout_lat)
    tqdm_desc = f"LiteReconfig, g{args.gl}/{args.lat_req:.1f} ms lat_req, on {len(video_dirs)} videos"
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
