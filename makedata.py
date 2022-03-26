import cv2
import numpy as np

from dataset.dataset import DatasetImgTF
import slam_lib.feature
import slam_lib.vis


def main():
    wait_key = 1
    # dir_src_path = './data/hair_close_range/'
    # dir_tgt_path = './data/hair_close_range_tf/'

    # dir_src_path = './data/graf/'
    # dir_tgt_path = './data/graf_tf/'

    dir_src_path = './data/hair/'
    dir_tgt_path = './data/hair_tf/'

    dt = DatasetImgTF()
    dt.build_img_from_dir(dir_src_path, dir_tgt_path)
    # dt.load_img_from_dir(dir_tgt_path)

    print(len(dt))

    for i, data in enumerate(dt):
        img0, img1 = data['image0'], data['image1']
        tf = data['tf_0_2_1']
        print(img0.shape, img1.shape)
        print(tf.shape)

        pts0, feat0 = slam_lib.feature.sift_features(img0)
        pts1, feat1 = slam_lib.feature.sift_features(img1)

        match = dt.get_match_matrix(i, pts0, pts1)['scores']
        match = match[:-1, :-1]
        ids0 = []
        ids1 = []
        for id0, id0_ids1 in enumerate(match):
            id1 = np.argmax(id0_ids1)
            if match[id0, id1] == 1:
                ids0.append(id0)
                ids1.append(id1)

        pts0 = pts0[ids0]
        pts1 = pts1[ids1]

        img_pt = slam_lib.vis.draw_pts(img0, pts0)
        cv2.namedWindow('pt0', cv2.WINDOW_NORMAL)
        cv2.imwrite('./results/kpts0.jpg', img_pt)
        cv2.imshow('pt0', img_pt)
        cv2.waitKey(wait_key)

        img_pt = slam_lib.vis.draw_pts(img1, pts1)
        cv2.namedWindow('pt1', cv2.WINDOW_NORMAL)
        cv2.imwrite('./results/kpts1.jpg', img_pt)
        cv2.imshow('pt1', img_pt)
        cv2.waitKey(wait_key)

        img_match = slam_lib.vis.draw_matches_(img0, pts0, img1, pts1)
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.imwrite('./results/match.jpg', img_match)
        cv2.imshow('match', img_match)
        cv2.waitKey(wait_key)


if __name__ == '__main__':
    main()
