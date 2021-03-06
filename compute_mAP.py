'''
Compute the mAP metric
 - IoU threshold = 0.5
 - mAP averages AP of each class without considering the number of the object in each class
 - "false postive" includes the bounding boxes which have IoU with a ground truth that has already been detected.
 - recall = (true positive) / (number of ground truth objects)
 - precision = (true positive) / (true positive + false positive)
 - AP averages the precision at all recall points, e.g. 1/N, 2/N, .... (N-1)/N, N/N.

File format: we assume the ground truth file to have the following layout,
name_of_image  class_id   (ymin,xmin,ymax,xmax)  -- this line should not exist 
xxxx.jpg       1           10    200  30   400  

We assume the detection file to have the following layout,
name_of_image  class_id  confidence  (ymin,xmin,ymax,xmax)  -- this line should not exist 
xxxx.jpg       1         0.995         10    200  30   400  

Example usage: python3 compute_mAP.py --gt=VID_testgt_10percent.txt --detection=log_dettr/VID_testset_BLStaticDetTr_10percent_si20_c0m0g0_det.txt
'''

import numpy as np
# from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
from collections import defaultdict
import argparse

def iou(ymin, xmin, ymax, xmax, gt_ymin, gt_xmin, gt_ymax, gt_xmax):
    area1 = abs(ymax-ymin)*abs(xmax-xmin)
    area2 = abs(gt_ymax-gt_ymin)*abs(gt_xmax-gt_xmin)
    if max(ymin, gt_ymin) < min(ymax, gt_ymax) and max(xmin, gt_xmin) < min(xmax, gt_xmax):
        ydiff = min(ymax, gt_ymax) - max(ymin, gt_ymin)
        xdiff = min(xmax, gt_xmax) - max(xmin, gt_xmin)
        area_i = xdiff*ydiff
    else:
        return False
    iou_metric = area_i / (area1+area2-area_i)
    return iou_metric>=0.5

def import_gt_file(gt_file, video_name = "",filter=None):
    # File format: we assume the ground truth file to have the following layout,
    # name_of_image  class_id   (ymin,xmin,ymax,xmax)  -- this line should not exist 
    # xxxx.jpg       1           10    200  30   400  
    gt = defaultdict(lambda: [])    # maps "name_of_image" to "list of bboxes"
    gt_cnt_per_class = defaultdict(lambda: 0) # maps "class" to "number of gt boxes"
    if isinstance(gt_file, str):
        with open(gt_file) as f:
            lines = f.readlines()
    elif isinstance(gt_file,list):
        lines = gt_file
    else:
        print("Not Implemented!")
        raise NotImplementedError
    if filter is not None:
        import pickle
        filter = pickle.load(open(filter,'rb'))
    for line in lines:
        items = line.strip().split()
        name, cls = items[0], int(items[1])
        video_folder = name.rsplit("/", 1)[0]
        ymin, xmin, ymax, xmax = float(items[2]), float(items[3]), float(items[4]), float(items[5])
        if (video_name == "" and filter is None) or (video_folder == video_name) or (video_name == "" and video_folder in filter):
            gt[name].append((cls, ymin, xmin, ymax, xmax))
            gt_cnt_per_class[cls] += 1
    return gt, gt_cnt_per_class

def import_detection_file(detection_file, video_name = "",filter=None):
    # File format: we assume the detection file to have the following layout,
    # name_of_image  class_id  confidence  (ymin,xmin,ymax,xmax)  -- this line should not exist 
    # xxxx.jpg       1         0.995         10    200  30   400 
    detection_per_class = defaultdict(lambda: [])
    if isinstance(detection_file, str):
        with open(detection_file) as f:
            lines = f.readlines()
    elif isinstance(detection_file,list):
        lines = detection_file
    else:
        print("Not Implemented!")
        raise NotImplementedError
    if filter is not None:
        import pickle
        filter = pickle.load(open(filter,'rb'))
    # Extract the detection for each class
    for line in lines:
        items = line.strip().split()
        name, cls = items[0], int(items[1])
        video_folder = name.rsplit("/", 1)[0]
        if (video_name == "" and filter is None) or (video_folder == video_name) or (video_name == "" and video_folder in filter):
            detection_per_class[cls].append(line)

    # Sort the detection based on confidence (colomn 2)
    for cls in detection_per_class:
        detection_per_class[cls] = sorted(detection_per_class[cls], reverse = True, 
                                          key = lambda x: float(x.strip().split()[2]))
    return detection_per_class

def calculate_mAP(gt_list, detection_per_class_list):
    gt, gt_cnt_per_class = import_gt_file(gt_list)
    detection_per_class = import_detection_file(detection_per_class_list)
    ap_per_cls = {}
    precision_at_per_cls = {}
    for cls in gt_cnt_per_class:
        precision_at = [0 for _ in range(gt_cnt_per_class[cls])]
        T, F = 0, 0

        for line in detection_per_class[cls]:
            items = line.strip().split()
            name, conf = items[0], float(items[2])
            ymin, xmin, ymax, xmax = float(items[3]), float(items[4]), float(items[5]), float(items[6])
            hit = False
            for idx in range(len(gt[name])):
                gt_cls, gt_ymin, gt_xmin, gt_ymax, gt_xmax = gt[name][idx]
                if gt_cls == cls and iou(ymin, xmin, ymax, xmax, gt_ymin, gt_xmin, gt_ymax, gt_xmax):
                    hit = True
                    gt[name].pop(idx)
                    break
            if hit:
                T += 1
                precision_at[T - 1] = T / (T + F)
            else:
                F += 1
        # The precision_at "recall R" is the maximum precision out of all where recall >= R
        max_seen = 0
        for idx in reversed(range(gt_cnt_per_class[cls])):
            max_seen = max(max_seen, precision_at[idx])
            precision_at[idx] = max_seen

        # Calculate average precision when means the area under precision-recall curve
        ap_per_cls[cls] = np.mean(precision_at)
        precision_at_per_cls[cls] = precision_at

    # Print results for humans
    meanAP = np.mean([ap_per_cls[cls] for cls in ap_per_cls])
    #print("meanAP = {:.4f}".format(meanAP))
    return meanAP

def compute_mAP(gt, detection, video_name = None):
    if video_name:
        gt, gt_cnt_per_class = import_gt_file(gt, video_name = video_name)
        detection_per_class = import_detection_file(detection, video_name = video_name)
    else:
        gt, gt_cnt_per_class = import_gt_file(gt)
        detection_per_class = import_detection_file(detection)

    # Assign "True Positive" or "False Positive" for each detection
    #   then directly compute average precision
    ap_per_cls = {}
    precision_at_per_cls = {}

    for cls in gt_cnt_per_class:          
        precision_at = [0 for _ in range(gt_cnt_per_class[cls])]
        T, F = 0, 0

        for line in detection_per_class[cls]:
            items = line.strip().split()
            name, conf = items[0], float(items[2])
            ymin, xmin, ymax, xmax = float(items[3]), float(items[4]), float(items[5]), float(items[6])

            hit = False
            for idx in range(len(gt[name])):
                gt_cls, gt_ymin, gt_xmin, gt_ymax, gt_xmax = gt[name][idx]
                if gt_cls == cls and iou(ymin, xmin, ymax, xmax, gt_ymin, gt_xmin, gt_ymax, gt_xmax):
                    hit = True
                    gt[name].pop(idx)
                    break
            if hit:
                T += 1
                precision_at[T-1] = T/(T+F)
            else:
                F += 1

        # The precision_at "recall R" is the maximum precision out of all where recall >= R 
        max_seen = 0
        for idx in reversed(range(gt_cnt_per_class[cls])):
            max_seen = max(max_seen, precision_at[idx])
            precision_at[idx] = max_seen

        # Calculate average precision when means the area under precision-recall curve
        ap_per_cls[cls] = np.mean(precision_at)
        precision_at_per_cls[cls] = precision_at
    meanAP = np.mean([ap_per_cls[cls] for cls in ap_per_cls])
    return meanAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the mAP metric.')
    parser.add_argument('--gt', dest='gt', required=True, help='Ground truth file.')
    parser.add_argument('--detection', dest='detection', required=True, help='Detection file.')
    parser.add_argument('--video', dest='video', help='The video that we focus on.')
    args = parser.parse_args()

    # gt maps "name_of_image" to "list of bboxes"
    # gt_cnt_per_class maps "class" to "number of gt boxes"
    # detection_per_class maps "class" to "detection_on_this_class"
    if args.video:
        meanAP = compute_mAP(args.gt, args.detection, args.video)
    else:
        meanAP = compute_mAP(args.gt, args.detection, None)
    print("meanAP = {:.4f}".format(meanAP))
