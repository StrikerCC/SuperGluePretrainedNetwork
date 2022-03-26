
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer, read_image,
                          make_matching_plot_fast, frame2tensor)

# from slam_lib.models.matching import Matching
# from slam_lib.models.utils import (AverageTimer, VideoStreamer, read_image,
#                           make_matching_plot_fast, frame2tensor)

import slam_lib.vis
import slam_lib.mapping

torch.set_grad_enabled(False)


class MatcherSuperGlue:
    def __init__(self, cuda='0'):
        self.timer = AverageTimer(newline=True)

        # Load the SuperPoint and SuperGlue models.
        self.device = 'cuda:'+str(cuda) if torch.cuda.is_available() else 'cpu'
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # torch.set_grad_enabled(True)
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.5,
            }
        }

        # self.matching = Matching(config).eval().to(self.device)
        self.matching = Matching(config).to(self.device)
        print('SuperGlue', config)
        print('Running inference on device \"{}\"'.format(self.device))

        # load dataset
        # self.resize = [4096, 3000]
        # self.resize = [2048, 1500]
        # self.resize = [1365, 1000]
        # self.resize = [1024, 750]
        # self.resize = [512, 375]

        self.resize = [3088, 2064]
        # self.resize = [1544, 1032]
        # self.resize = [772, 516]
        # self.resize = [386, 258]

    def match(self, img_0, img_1, flag_vis=False):
        # torch.set_grad_enabled(False)
        # torch.set_grad_enabled(True)

        img_left, img_right = None, None
        if isinstance(img_0, str):
            img_left = cv2.imread(img_0)
        elif isinstance(img_0, np.ndarray):
            img_left = img_0

        if isinstance(img_1, str):
            img_right = cv2.imread(img_1)
        elif isinstance(img_1, np.ndarray):
            img_right = img_1

        gray_0, inp0, scales0 = read_image(img_0, self.device, self.resize, rotation=0, resize_float=True)
        gray_1, inp1, scales1 = read_image(img_1, self.device, self.resize, rotation=0, resize_float=True)

        pred = self.matching({'image0': inp0, 'image1': inp1})

        # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        # pred = {k: v[0].detach().numpy() for k, v in pred.items()}

        for k_, v_ in pred.items():
            try:
                pred[k_] = v_[0].cpu().numpy()
                print(k_)
            except:
                print(k_, 'no')

        kpts_0, kpts_1 = pred['keypoints0'], pred['keypoints1']
        # feats_0, feats_1 = pred['descriptors0'].T, pred['descriptors1'].T
        matches, conf = pred['matches0'], pred['matching_scores0']

        self.timer.update('matcher')

        # Keep the matching keypoints and scale points back
        valid = matches > -1
        mkpts_0 = kpts_0[valid]
        # mfeats_0 = feats_0[valid]
        mkpts_1 = kpts_1[matches[valid]]
        # mfeats_1 = feats_1[matches[valid]]
        # mconf = conf[valid]

        pts_2d_0, pts_2d_1 = slam_lib.mapping.scale_pts(scales0, kpts_0), slam_lib.mapping.scale_pts(scales1, kpts_1)
        mpts_2d_0, mpts_2d_1 = slam_lib.mapping.scale_pts(scales0, mkpts_0), slam_lib.mapping.scale_pts(scales1, mkpts_1)

        if flag_vis:
            img_pt = slam_lib.vis.draw_pts(img_0, pts_2d_0)
            cv2.namedWindow('pt0', cv2.WINDOW_NORMAL)
            cv2.imshow('pt0', img_pt)
            cv2.waitKey(0)

            img_pt = slam_lib.vis.draw_pts(img_1, pts_2d_1)
            cv2.namedWindow('pt1', cv2.WINDOW_NORMAL)
            cv2.imshow('pt1', img_pt)
            cv2.waitKey(0)

            img_pt = slam_lib.vis.draw_pts(img_0, mpts_2d_0)
            cv2.namedWindow('pt0', cv2.WINDOW_NORMAL)
            cv2.imshow('pt0', img_pt)
            cv2.waitKey(0)

            img_pt = slam_lib.vis.draw_pts(img_1, mpts_2d_1)
            cv2.namedWindow('pt1', cv2.WINDOW_NORMAL)
            cv2.imshow('pt1', img_pt)
            cv2.waitKey(0)

            img3 = slam_lib.vis.draw_matches(img_0, mpts_2d_0, img_1, mpts_2d_1)
            cv2.namedWindow('hair_close_range_match', cv2.WINDOW_NORMAL)
            cv2.imshow('hair_close_range_match', img3)
            cv2.waitKey(0)

        # return pts_2d_0, feats_0, pts_2d_1, feats_1
        return mpts_2d_0, mpts_2d_1

