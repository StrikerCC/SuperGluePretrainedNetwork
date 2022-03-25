import cv2

import slam_lib.vis


def main():
    img = slam_lib.vis.draw_pts(img, pts)
    cv2.namedWindow('pt', cv2.WINDOW_NORMAL)
    cv2.imshow('pt', img)
    cv2.waitKey(0)
