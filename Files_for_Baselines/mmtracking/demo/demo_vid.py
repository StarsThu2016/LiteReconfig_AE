from argparse import ArgumentParser
import os
import cv2
import time
import tqdm
import numpy as np


from mmtrack.apis import inference_vid, init_model


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--input', help='input video file')
    parser.add_argument('--output', help='output video file (mp4 format)')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--score-thr', type=float, default=0.6, help='bbox score threshold')
    args = parser.parse_args()

    dataset_prefix = '/home/nvidia/sdcard/ILSVRC2015'

    # setup for output logging.
    detoutput_filename = args.output.split(".")[0] + "_selsa_det_n.txt"
    latoutput_filename = args.output.split(".")[0] + "_selsa_lat_n.txt"
    fout_det = open(detoutput_filename, "w")
    fout_lat = open(latoutput_filename, "w")

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # cap = cv2.VideoCapture(args.input)
    with open(args.input) as f:
        lines = f.readlines()
        imgs_list = [line.strip() for line in lines]

    old_vid_name = imgs_list[0].split('/')[-2]
    frame_id = -1

    for img_path in tqdm.tqdm(imgs_list):

        vid_name = img_path.split('/')[-2]
        frame_id += 1

        if old_vid_name != vid_name:
            frame_id = 0

        full_path = os.path.join(dataset_prefix, img_path)
        frame = cv2.imread(full_path)
        h, w, _ = frame.shape
        # test a single image
        time1 = time.time()
        result = inference_vid(model, frame, frame_id)
        time2 = time.time()
        inf_lat = (time2-time1)*1e3
        bboxes, labels = model.show_result(
            frame, result, score_thr=args.score_thr, show=False)

        if args.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > args.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
        
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)

            xmin = bbox_int[0]
            ymin = bbox_int[1]
            xmax = bbox_int[2]
            ymax = bbox_int[3]
            if len(bbox) > 4:
                conf = bbox[-1]
        # for cls, conf, ymin, xmin, ymax, xmax in bboxes:
        #     # Detection results logging.
                print("{} {} {} {} {} {} {}".format(img_path, label, conf, ymin/h, xmin/w, ymax/h, xmax/w), file=fout_det)
                print(inf_lat - 200, file=fout_lat)
        # Latency results logging.

    fout_det.close()
    fout_lat.close()

if __name__ == '__main__':
    main()
