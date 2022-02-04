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

Example usage: python3 utils_approxdet/compute_mAP.py --gt=test/VID_testgt_full.txt --detection=test/VID_test00106000_nprop100_shape576_det.txt --video=Data/VID/val/ILSVRC2015_val_00106000
'''

import numpy as np
from matplotlib import pyplot as plt
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the mAP metric.')
    parser.add_argument('--gt', dest='gt', required=True, help='Ground truth file.')
    parser.add_argument('--detection', dest='detection', required=True, help='Detection file.')
    parser.add_argument('--vis', dest='vis', help='A figure showing per-category ROC curve.')
    parser.add_argument('--video', dest='video', help='The video that we focus on.')
    parser.add_argument('--verbose', dest='verbose', help='Verbose level, 1=print FP, TP, #detection, #gt, ' + \
                                                          '2=additionaly, print TP and FP detection.' + \
                                                          '3=additionaly, print detections in the decreasing confidence order.')
    parser.add_argument('--filter',dest='filter',default=None,help="Add filters to detection and gt file to calculate separate mAP for different sets")
    args = parser.parse_args()

    # gt maps "name_of_image" to "list of bboxes"
    # gt_cnt_per_class maps "class" to "number of gt boxes"
    # detection_per_class maps "class" to "detection_on_this_class"
    
    if args.video:
        gt, gt_cnt_per_class = import_gt_file(args.gt, video_name = args.video)
        detection_per_class = import_detection_file(args.detection, video_name = args.video)
    else:
        gt, gt_cnt_per_class = import_gt_file(args.gt,filter=args.filter)
        detection_per_class = import_detection_file(args.detection,filter=args.filter)
    # Assign "True Positive" or "False Positive" for each detection
    #   then directly compute average precision
    ap_per_cls = {}
    precision_at_per_cls = {}
    
    # Print results for humans
    if args.verbose: 
        TP_prints, FP_prints, num_gt = [], [], 0
        num_detection_per_class = [len(detection_per_class[cls]) for cls in detection_per_class]
        print("Number of detections = {}".format(sum(num_detection_per_class)))
      
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
                
                # Print results for humans
                if args.verbose: 
                    TP_prints.append(("TP", name, cls, conf, ymin, xmin, ymax, xmax))
            else:
                F += 1
                
                # Print results for humans
                if args.verbose: 
                    FP_prints.append(("FP", name, cls, conf, ymin, xmin, ymax, xmax))
        
        # Print results for humans
        if args.verbose:
            num_gt += gt_cnt_per_class[cls]
               
        # The precision_at "recall R" is the maximum precision out of all where recall >= R 
        max_seen = 0
        for idx in reversed(range(gt_cnt_per_class[cls])):
            max_seen = max(max_seen, precision_at[idx])
            precision_at[idx] = max_seen
            
        # Calculate average precision when means the area under precision-recall curve
        ap_per_cls[cls] = np.mean(precision_at)
        precision_at_per_cls[cls] = precision_at

    # Print results for humans
    if args.verbose: 
        print("Number of ground detection = {}".format(num_gt))
        print("Number of TP, FP = {}, {} (note: the sum may not equal to number of detections, because the detections which does not belong to the class of the ground truth classes does not count.) ".format(len(TP_prints), len(FP_prints)))
        if args.verbose == "2": 
            TP_prints = sorted(TP_prints, key = lambda x: x[1])
            FP_prints = sorted(FP_prints, key = lambda x: x[1])
            print("True Positives are as follows: (path, class, confidence, ymin, xmin, ymax, xmax)")
            for t_or_f, name, cls, conf, ymin, xmin, ymax, xmax in TP_prints:
                print("{}: {} {:2d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(t_or_f, name, cls, conf, ymin, xmin, ymax, xmax))
            print("False Positives are as follows: (path, class, confidence, ymin, xmin, ymax, xmax)")
            for t_or_f, name, cls, conf, ymin, xmin, ymax, xmax in FP_prints:
                print("{}: {} {:2d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(t_or_f, name, cls, conf, ymin, xmin, ymax, xmax))
        if args.verbose == "3": 
            All_prints = TP_prints + FP_prints
            All_prints = sorted(All_prints, key = lambda x: x[3], reverse = True)
            print("All detections are as follows: (path, class, confidence, ymin, xmin, ymax, xmax)")
            for t_or_f, name, cls, conf, ymin, xmin, ymax, xmax in All_prints:
                print("{}: {} {:2d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(t_or_f, name, cls, conf, ymin, xmin, ymax, xmax))

    meanAP = np.mean([ap_per_cls[cls] for cls in ap_per_cls])
    print("meanAP = {:.4f}".format(meanAP))
        
    if args.vis:
        all_cls = [cls for cls in precision_at_per_cls]
        if len(all_cls) == 30:
            plt.figure(figsize=(24, 12))
            for cls in range(30):
                plt_row, plt_col = cls//6, cls%6
                plt.subplot(5,6,cls+1)
                N = len(precision_at_per_cls[cls])
                x = [y/(N-1) for y in range(N)]
                msg = "cls.{},mAP={:.3f}".format(cls, np.mean(precision_at_per_cls[cls]))
                plt.plot(x, precision_at_per_cls[cls], label = msg)
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.grid()
                plt.legend()
            plt.savefig(args.vis)
        elif len(all_cls) == 1:
            plt.figure(figsize = (3.2, 2.4), dpi = 300)
            plt.axes().set_position([0.18, 0.15, 0.79, 0.79])
            plt.axes().spines['top'].set_color('none')
            plt.axes().spines['right'].set_color('none')

            plt.xlabel('Recall', fontsize=8) # <=15 chacters
            plt.axes().xaxis.set_label_coords(0.5,-0.12)
            plt.xlim([0,1.05])
            plt.xticks(np.arange(0,1.01,0.2))
            plt.axes().set_xticklabels(labels = [0,0.2,0.4,0.6,0.8,1.0], fontsize=8)

            plt.ylabel('Precision', fontsize=8)
            plt.axes().yaxis.set_label_coords(-0.15,0.5)
            plt.ylim([0,1.05])
            plt.yticks(np.arange(0,1.01,0.2))
            plt.axes().set_yticklabels(labels = [0,0.2,0.4,0.6,0.8,1.0], fontsize=8)
            plt.axes().set_axisbelow(True)
            plt.grid()

            for cls in precision_at_per_cls:
                N = len(precision_at_per_cls[cls])
                x = [y/(N-1) for y in range(N)]
                msg = "class_id={}, mAP={:.3f}".format(cls, np.mean(precision_at_per_cls[cls]))
                plt.plot(x, precision_at_per_cls[cls], label = msg)
            plt.legend(fontsize=8)  
            plt.savefig(args.vis)
        else:
            print("Not support yet! len(all_cls) = {}".format(len(all_cls)))
