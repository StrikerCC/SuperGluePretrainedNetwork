import os
import shutil
import warnings

import cv2
import numpy as np
import transforms3d as tf3
import random

from dataset.format import name_is_img_file
import slam_lib.feature
import slam_lib.vis
import slam_lib.mapping
import slam_lib.format
import slam_lib.geometry
import slam_lib.models.utils


class Dataset:
    def __init__(self):
        self.img_paths = []
        self.label_paths = []

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        return cv2.imread(self.img_paths[item])
    
    def load_img_from_dir(self, dir_path):
        if not os.path.isdir(dir_path):
            warnings.warn(dir_path, 'doesn\'t exist in sys')
            return False

        '''load img file, find txt in same name'''
        for file_name in os.listdir(dir_path):
            if name_is_img_file(file_name):     # found a training data
                img_name = file_name
                img_path = dir_path + '/' + img_name
                '''record img and label path'''
                self.img_paths.append(img_path)         # add img path
        return True


class DatasetImgTF(Dataset):
    def __init__(self):
        super(DatasetImgTF, self).__init__()

        self.__img0_postfix = '_0.jpg'
        self.__img1_postfix = '_1.jpg'
        self.__tf_file_postfix = '_tf.txt'

        self.img0_filepaths = []
        self.img1_filepaths = []
        self.tf_0_2_1_filepath = []

        self.num_tf_each_data = 3

        self.__scale_range_x = (0.8, 2.0)
        self.__scale_range_y = (0.8, 2.0)
        self.__scale_range_x = (0.8, 1.0)
        self.__scale_range_y = (0.8, 1.0)

        self.__offset_range_x = (-300, 300)
        self.__offset_range_y = (-300, 300)
        self.__offset_range_x = (-3, 3)
        self.__offset_range_y = (-3, 3)

        self.__angle_range_x = (0, 0)
        self.__angle_range_y = (0, 0)
        self.__angle_range_z = (-np.pi/6, np.pi/6)

        # self.resize = (772, 516)
        # self.resize = (1544, 1032)
        self.resize = (3088, 2064)
        # self.resize = (800, 640)

    def __len__(self):
        return len(self.img0_filepaths)

    def __getitem__(self, item):
        return {'image0': cv2.imread(self.img0_filepaths[item]),
                'image1': cv2.imread(self.img1_filepaths[item]),
                'tf_0_2_1': np.loadtxt(self.tf_0_2_1_filepath[item])}

    def get_match_indices(self, item, pts0, pts1):
        if pts0 is None or len(pts0) == 0:
            return None
        pts0 = slam_lib.format.pts_2d_format(pts0)
        pts1 = slam_lib.format.pts_2d_format(pts1)

        '''tf pts 0'''
        data = self[item]
        tf_0_2_1 = data['tf_0_2_1']
        pts0_1 = slam_lib.mapping.transform_pt_2d(tf_0_2_1, pts0)

        '''match pts 1 with transformed pts 0'''
        ids_0_2_1 = slam_lib.geometry.nearest_neighbor_points_2_points(pts0_1, pts1, distance_min=5)

        '''fill match indices'''
        matches0, matches1 = np.zeros(pts0.shape[0]).astype(int) - 1, np.zeros(pts1.shape[0]).astype(int) - 1
        matches0[ids_0_2_1[:, 0].tolist()] = ids_0_2_1[:, 1]
        matches1[ids_0_2_1[:, 1].tolist()] = ids_0_2_1[:, 0]
        gt = {'matches0': matches0, 'matches1': matches1}
        return gt

    def get_match_matrix(self, item, pts0, pts1):
        if pts0 is None or len(pts0) == 0:
            return None
        pts0 = slam_lib.format.pts_2d_format(pts0)
        pts1 = slam_lib.format.pts_2d_format(pts1)

        data = self[item]
        tf_0_2_1 = data['tf_0_2_1']
        pts0_1 = slam_lib.mapping.transform_pt_2d(tf_0_2_1, pts0)

        m, n = len(pts0), len(pts1)
        # marked0, marked1 = np.zeros(m).astype(bool), np.zeros(n).astype(bool)
        match_ = np.zeros((m+1, n+1)).astype(float)
        match_[:, -1] = 1.0
        match_[-1, :] = 1.0

        ids_0_2_1 = slam_lib.geometry.nearest_neighbor_points_2_points(pts0_1, pts1, distance_min=50)

        if len(ids_0_2_1) > 0:
            match_[tuple(ids_0_2_1.T.tolist())] = 1.0
            match_[tuple(ids_0_2_1[:, 0].tolist()), -1] = 0.0
            match_[-1, tuple(ids_0_2_1[:, 1].tolist())] = 0.0
        return {'scores': match_}

    def tf_pts_0_2_1(self, item, pts0):
        if pts0 is None or len(pts0) == 0:
            return None
        pts0 = slam_lib.format.pts_2d_format(pts0)

        data = self[item]
        tf_0_2_1 = data['tf_0_2_1']

        pts1 = slam_lib.mapping.transform_pt_2d(tf_0_2_1, pts0)
        return pts1

    def load_img_from_dir(self, dir_path):
        assert os.path.isdir(dir_path)
        recorded = {}
        filenames = os.listdir(dir_path)
        filenames = sorted(filenames)
        filename_set = set(filenames)
        for filename in filenames:
            '''not a img'''
            if not name_is_img_file(filename):
                continue
            img_filename = filename                             # LeftCamera-Snapshot-20220221170353-29490143290143_13_0.jpg
            img_id = img_filename[:-len(self.__img0_postfix)]     # LeftCamera-Snapshot-20220221170353-29490143290143_13

            '''recorded already'''
            if img_id in recorded:
                continue

            img0_name, img1_name = img_id + self.__img0_postfix, img_id + self.__img1_postfix
            tf_0_2_1_name = img_id + self.__tf_file_postfix

            '''some file missing'''
            if not (img0_name in filename_set and
                    img1_name in filename_set and
                    tf_0_2_1_name in filename_set):
                warnings.warn('cannot find ' + img_id + ' in ' + dir_path)
                continue

            '''recording'''
            img0_path, img1_path = dir_path + '/' + img0_name, dir_path + '/' + img1_name
            tf_0_2_1_filepath = dir_path + '/' + tf_0_2_1_name
            self.img0_filepaths.append(img0_path)
            self.img1_filepaths.append(img1_path)
            self.tf_0_2_1_filepath.append(tf_0_2_1_filepath)

    def build_img_from_dir(self, src_dir_path, tgt_dir_path):
        super(DatasetImgTF, self).load_img_from_dir(src_dir_path)
        if os.path.isdir(tgt_dir_path):
            shutil.rmtree(tgt_dir_path)
        os.makedirs(tgt_dir_path)

        '''load img file, tf img, record tf in txt'''
        for img_src_path in self.img_paths:
            img_src_name = img_src_path[img_src_path.rfind('/')+1:]
            img_src_id = img_src_name[:img_src_name.rfind('.')]

            img_src = cv2.imread(img_src_path)

            '''preprocessing'''
            # resize
            w, h = img_src.shape[1], img_src.shape[0]
            img_size = (w, h)
            w_new, h_new = slam_lib.models.utils.process_resize(w, h, self.resize)
            resize_scales = (float(w) / float(w_new), float(h) / float(h_new))
            tf_resizing_2_original = np.array([[resize_scales[0], 0, 0],
                                               [0, resize_scales[1], 0],
                                               [0, 0, 1]])
            # tf_resizing_2_original = np.array([[1.0, 0, 0],
            #                                    [0, 1.0, 0],
            #                                    [0, 0, 1.0]])
            # recenter
            tf_upper_left_2_center = np.eye(3)
            tf_upper_left_2_center[0, -1] = -img_size[0] / 2
            tf_upper_left_2_center[1, -1] = -img_size[1] / 2

            '''do the tf'''
            for i_homo in range(self.num_tf_each_data):
                img0_path = tgt_dir_path + '/' + img_src_id + '_' + str(i_homo) + '_0' + '.jpg'
                img1_path = tgt_dir_path + '/' + img_src_id + '_' + str(i_homo) + '_1' + '.jpg'
                tf_path = tgt_dir_path + '/' + img_src_id + '_' + str(i_homo) + '_tf' + '.txt'

                '''do tf'''
                # rotation and translation
                tf_rigid = np.eye(3)
                tf_rigid[0, -1] = np.random.random() * (self.__offset_range_x[1] - self.__offset_range_x[0]) + \
                                   self.__offset_range_x[0]
                tf_rigid[1, -1] = np.random.random() * (self.__offset_range_y[1] - self.__offset_range_y[0]) + \
                                   self.__offset_range_y[0]
                rotation = tf3.euler.euler2mat(
                    np.random.random() * (self.__angle_range_x[1] - self.__angle_range_x[0]) + self.__angle_range_x[0],
                    np.random.random() * (self.__angle_range_y[1] - self.__angle_range_y[0]) + self.__angle_range_y[0],
                    np.random.random() * (self.__angle_range_z[1] - self.__angle_range_z[0]) + self.__angle_range_z[0]
                )
                tf_rigid = np.matmul(np.linalg.inv(tf_upper_left_2_center),
                                     np.matmul(tf_rigid, np.matmul(rotation, tf_upper_left_2_center)))

                # scaling
                tf_scale = np.eye(3)
                tf_scale[0, 0] = np.random.random() * (self.__scale_range_x[1] - self.__scale_range_x[0]) + self.__scale_range_x[
                    0]
                tf_scale[1, 1] = np.random.random() * (self.__scale_range_y[1] - self.__scale_range_y[0]) + self.__scale_range_y[
                    0]

                homograph_org0_2_org1 = np.matmul(tf_scale, tf_rigid)
                homograph_0_2_1 = np.matmul(np.matmul(np.linalg.inv(tf_resizing_2_original), homograph_org0_2_org1), tf_resizing_2_original)

                img0 = cv2.warpPerspective(img_src, M=np.linalg.inv(tf_resizing_2_original), dsize=self.resize)
                img1 = cv2.warpPerspective(img_src, M=np.matmul(np.linalg.inv(tf_resizing_2_original), homograph_org0_2_org1), dsize=self.resize)

                '''save img to jpg, tf to txt'''

                np.savetxt(tf_path, homograph_0_2_1)
                cv2.imwrite(img0_path, img0)
                cv2.imwrite(img1_path, img1)

                '''recording'''
                self.img0_filepaths.append(img0_path)
                self.img1_filepaths.append(img1_path)
                self.tf_0_2_1_filepath.append(tf_path)

        return True


class DatasetKeyPoints(Dataset):
    def __init__(self):
        super(DatasetKeyPoints, self).__init__()

    def __getitem__(self, item):
        return {'img': cv2.imread(self.img_paths[item]),
                'label': np.loadtxt(self.label_paths[item])}

    def load_img_label_from_dir(self, dir_path):
        """"""
        '''load img file'''
        super(DatasetKeyPoints, self).load_img_from_dir(dir_path)
        '''load img file, find txt in same name'''
        for img_path in self.img_paths:
            img_name = img_path[img_path.rfind('/')+1:]
            pts_filename = img_name[:img_name.rfind('.')] + '_pts.txt'
            feats_filename = img_name[:img_name.rfind('.')] + '_feats.txt'
            pts_filepath = dir_path + '/' + pts_filename
            feats_filepath = dir_path + '/' + feats_filename
            if not os.path.isfile(pts_filepath):
                '''compute label'''
                img = cv2.imread(img_path)
                pts, feats = slam_lib.feature.sift_features(img)
                np.savetxt(pts_filepath, pts)               # saving label and img to same dir
                # np.savetxt(feats_filepath, feats)         # saving label and img to same dir
            '''record label path'''
            self.label_paths.append(pts_filepath)     # add label path
        return True


class DatasetKeyPointsMatch(DatasetKeyPoints):
    def __init__(self):
        super(DatasetKeyPointsMatch, self).__init__()
        self.img0_paths = []
        self.label0_paths = []

        self.img1_paths = []
        self.label1_paths = []

        self.homography_bank = []
        self.num_tf_each_data = 3

        self.scale_range_x = (0.85, 1.0)
        self.scale_range_y = (0.85, 1.0)

        self.offset_range_x = (-300, 300)
        self.offset_range_y = (-300, 300)

        self.angle_range_x = (0, 0)
        self.angle_range_y = (0, 0)
        self.angle_range_z = (-np.pi, np.pi)

    def __len__(self):
        return len(self.img0_paths)

    def __getitem__(self, item):
        return {'image0': cv2.imread(self.img0_paths[item]),
                'sift_pts0': np.loadtxt(self.label0_paths[item]),
                'image1': cv2.imread(self.img1_paths[item]),
                'sift_pts1': np.loadtxt(self.label1_paths[item])}

    def load_homography_bank_from_txt(self, homography_path):
        assert os.path.isfile(homography_path)
        homos = np.loadtxt(homography_path)
        self.homography_bank = homos.reshape((-1, 3, 3))

    def load_img_label_from_dir(self, dir_path):
        assert os.path.isdir(dir_path)
        recorded = {}
        filenames = os.listdir(dir_path)
        filename_set = set(filenames)
        for filename in filenames:
            '''not a img'''
            if not name_is_img_file(filename):              # LeftCamera-Snapshot-20220221170353-29490143290143_13_0.jpg
                continue
            img_filename = filename[:filename.rfind('.')]   # LeftCamera-Snapshot-20220221170353-29490143290143_13_0
            img_id = img_filename[:-1]                      # LeftCamera-Snapshot-20220221170353-29490143290143_13_
            binary_id = img_filename[-3:-2]                 # 0 or 1

            '''recorded already'''
            if img_id in recorded:
                continue

            img0_name, img1_name = img_id + '0.jpg', img_id + '1.jpg'
            pts0_name, pts1_name = img_id + '0.txt', img_id + '1.txt'

            '''some file missing'''
            if not (img0_name in filename_set and
                    img1_name in filename_set and
                    pts0_name in filename_set and
                    pts1_name in filename_set):
                warnings.warn('cannot find ' + img_id + ' in ' + dir_path)
                continue

            '''recording'''
            img0_path, img1_path = dir_path + '/' + img0_name, dir_path + '/' + img1_name
            pts0_path, pts1_path = dir_path + '/' + pts0_name, dir_path + '/' + pts1_name
            self.img0_paths.append(img0_path)
            self.img1_paths.append(img1_path)
            self.label0_paths.append(pts0_path)
            self.label1_paths.append(pts1_path)

    def build_img_label_from_dir(self, src_dir_path, tgt_dir_path):
        super(DatasetKeyPointsMatch, self).load_img_label_from_dir(src_dir_path)
        '''for each source img and label'''
        for img_src_path, label_src_path in zip(self.img_paths, self.label_paths):
            '''path and id acquire'''
            img_src_name, pts_src_name = img_src_path[img_src_path.rfind('/')+1:], label_src_path[label_src_path.rfind('/')+1:]
            img_src_id, pts_src_id = img_src_name[:img_src_name.rfind('.')], pts_src_name[:pts_src_name.rfind('.')]

            '''data reading'''
            img_src = cv2.imread(img_src_path)
            pts_src = np.loadtxt(label_src_path)
            # pts_src = pts_src[int(len(pts_src)/4):int(len(pts_src)/2)]
            pts_src = pts_src[::3]

            '''pts filtering'''
            # TODO: cell down sampling

            '''img parameter'''
            img_size = img_src.shape[::-1][1:]
            tf_upper_left_2_center = np.eye(3)
            tf_upper_left_2_center[0, -1] = -img_size[0] / 2
            tf_upper_left_2_center[1, -1] = -img_size[1] / 2

            '''do the tf'''
            for i_homo in range(self.num_tf_each_data):
                img0_path, pts0_path = tgt_dir_path + '/' + img_src_id + '_' + str(i_homo) + '_0' + '.jpg', \
                                       tgt_dir_path + '/' + pts_src_id + '_' + str(i_homo) + '_0' + '.txt'
                img1_path, pts1_path = tgt_dir_path + '/' + img_src_id + '_' + str(i_homo) + '_1' + '.jpg', \
                                       tgt_dir_path + '/' + pts_src_id + '_' + str(i_homo) + '_1' + '.txt'
                img0 = img_src
                pts0 = pts_src

                homograph = np.eye(3)
                homograph[0, -1] = np.random.random() * (self.offset_range_x[1] - self.offset_range_x[0]) + self.offset_range_x[0]
                homograph[1, -1] = np.random.random() * (self.offset_range_y[1] - self.offset_range_y[0]) + self.offset_range_y[0]
                rotation = tf3.euler.euler2mat(
                    np.random.random() * (self.angle_range_x[1] - self.angle_range_x[0]) + self.angle_range_x[0],
                    np.random.random() * (self.angle_range_y[1] - self.angle_range_y[0]) + self.angle_range_y[0],
                    np.random.random() * (self.angle_range_z[1] - self.angle_range_z[0]) + self.angle_range_z[0]
                )
                scale = np.eye(3)
                scale[0, 0] = np.random.random() * (self.scale_range_x[1] - self.scale_range_x[0]) + self.scale_range_x[0]
                scale[1, 1] = np.random.random() * (self.scale_range_y[1] - self.scale_range_y[0]) + self.scale_range_y[0]

                homograph = np.matmul(np.linalg.inv(tf_upper_left_2_center), np.matmul(homograph, np.matmul(rotation, tf_upper_left_2_center)))
                homograph = np.matmul(scale, homograph)

                img1 = cv2.warpPerspective(img_src, M=homograph, dsize=img_size)
                pts1 = slam_lib.mapping.transform_pt_2d(tf=homograph, pts=pts_src)

                '''filter out of range pts in both src and tgt'''
                mask_x = np.logical_and(0 < pts1[:, 0], pts1[:, 0] < img_size[0])
                mask_y = np.logical_and(0 < pts1[:, 0], pts1[:, 1] < img_size[1])
                mask = np.logical_and(mask_x, mask_y)
                pts0 = pts0[mask]
                pts1 = pts1[mask]

                '''save source pts, target image, and target pts'''
                cv2.imwrite(img0_path, img0)
                cv2.imwrite(img1_path, img1)
                np.savetxt(pts0_path, pts0)
                np.savetxt(pts1_path, pts1)

                '''recording'''
                self.img0_paths.append(img0_path)
                self.img1_paths.append(img1_path)
                self.label0_paths.append(pts0_path)
                self.label1_paths.append(pts1_path)

            '''using loaded homography'''
            # homograph_bank = random.sample(list(self.homography_bank), k=min(self.num_tf_each_data, len(self.homography_bank)))
            # for i_homo, homograph in enumerate(homograph_bank):
            #     img0_path, pts0_path = tgt_dir_path + '/' + img_src_id + '_0_' + str(i_homo) + '.jpg', \
            #                            tgt_dir_path + '/' + pts_src_id + '_0_' + str(i_homo) + '.txt'
            #     img1_path, pts1_path = tgt_dir_path + '/' + img_src_id + '_1_' + str(i_homo) + '.jpg', \
            #                            tgt_dir_path + '/' + pts_src_id + '_1_' + str(i_homo) + '.txt'
            #     img0 = img_src
            #     pts0 = pts_src
            #     homograph = np.eye(3)
            #     homograph[0, -1] = 100
            #     homograph[1, -1] = 0
            #     rotation = tf3.euler.euler2mat(0, 0, np.pi/12)
            #     homograph = np.matmul(homograph, rotation)
            #
            #     img1 = cv2.warpPerspective(img_src, M=homograph, dsize=img_size)
            #     pts1 = slam_lib.mapping.transform_pt_2d(tf=homograph, pts=pts_src)
            #
            #     '''filter out of range pts in both src and tgt'''
            #     mask_x = np.logical_and(0 < pts1[:, 0], pts1[:, 0] < img_size[0])
            #     mask_y = np.logical_and(0 < pts1[:, 0], pts1[:, 1] < img_size[1])
            #     mask = np.logical_and(mask_x, mask_y)
            #     pts0 = pts0[mask]
            #     pts1 = pts1[mask]
            #
            #     '''save source pts, target image, and target pts'''
            #     cv2.imwrite(img0_path, img0)
            #     cv2.imwrite(img1_path, img1)
            #     np.savetxt(pts0_path, pts0)
            #     np.savetxt(pts1_path, pts1)
            #
            #     '''recording'''
            #     self.img0_paths.append(img0_path)
            #     self.img1_paths.append(img1_path)
            #     self.label0_paths.append(pts0_path)
            #     self.label1_paths.append(pts1_path)

        return True

