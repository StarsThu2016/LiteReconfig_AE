# Convert the detection results of our format to dict type ordered by each video.
import argparse
import pickle
from PIL import Image
import numpy as np
import os
import tqdm
import json

'''
Usage : python repp2ours.py --reppfile d0_test_repp_det.json \
          --output converted.txt \
          --dataset_prefix '/media/jay/New Volume/Datasets/ILSVRC2015/'
'''


def main():
    parser = argparse.ArgumentParser(description='For applying post processing')
    parser.add_argument('--reppfile', help='path to repp json file', type=str)
    parser.add_argument('--output', help='name of the output file', type=str)
    parser.add_argument('--dataset_prefix', help='path to the ILSVRC2015 VID dataset')
    args = parser.parse_args()

    fout_det = open(args.output, "w")

    with open(args.reppfile, 'rb') as jsonfile:
        predictions = json.load(jsonfile)

    for line in tqdm.tqdm(predictions):

        [img_path, cls, conf, ymin, xmin, ymax, xmax] = line.strip().split(' ')
        # print(xmin)
        # grab the h,w for un-normalizing.
        img_path = 'Data/VID/' + img_path + '.JPEG'
        # grab the h,w for un-normalizing.
        with open('LUT.json', 'rb') as jsonfile:
            LUT = json.load(jsonfile)
        [h, w] = LUT[img_path[:-12]]

        # normalize
        ymin = float(ymin) / h
        xmin = float(xmin) / w
        ymax = float(ymax) / h
        xmax = float(xmax) / w

        print("{} {} {} {} {} {} {}".format(img_path, cls, conf, ymin, xmin, ymax, xmax), file=fout_det)

    fout_det.close()

if __name__ == '__main__':
    main()
