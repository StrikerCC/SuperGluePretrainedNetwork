
from pathlib import Path
import argparse
import random

import cv2
import numpy as np
import matplotlib.cm as cm
import torch
from models.utils import read_image

from matcher import MatcherSuperGlue
import slam_lib.vis


def main():
    device = 'cuda'

    resize = [772, 516]
    img_0 = './data/graf_tf/img1_0_0.jpg'
    img_1 = './data/graf_tf/img1_0_1.jpg'

    img_0, img_1 = cv2.imread(img_0), cv2.imread(img_1)
    print(img_0.shape)
    print(img_1.shape)

    matcher = MatcherSuperGlue()

    pts_2d_0, pts_2d_1 = matcher.match(img_0, img_1, flag_vis=True)

    img3 = slam_lib.vis.draw_matches(img_0, pts_2d_0, img_1, pts_2d_1)
    cv2.namedWindow('hair_close_range_match', cv2.WINDOW_NORMAL)
    cv2.imshow('hair_close_range_match', img3)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
