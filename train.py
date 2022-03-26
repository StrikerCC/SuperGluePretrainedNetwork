import torch
import numpy as np
import cv2
from models.matching import Matching
from models.utils import AverageTimer, VideoStreamer, read_image

import slam_lib.mapping
import slam_lib.vis
import matplotlib.pyplot as plt

from dataset.dataset import DatasetKeyPointsMatch, DatasetImgTF
import loss
from matcher import MatcherSuperGlue

device = 'cuda:3'


def change_pred_to_cpu_and_image_frame(pred, scales0, scales1):
    pred_cpu = {}
    for k_, v_ in pred.items():
        try:
            try:
                pred_cpu[k_] = v_[0].cpu().numpy()
                # print(k_)
            except:
                pred_cpu[k_] = v_[0].detach().numpy()
                print(k_)
        except:
            pass
    pred_cpu['keypoints0'], pred_cpu['keypoints1'] = slam_lib.mapping.scale_pts(scales0, pred_cpu['keypoints0']), slam_lib.mapping.scale_pts(scales1, pred_cpu['keypoints1'])
    return pred_cpu


def make_gt_from_pred(pred_cpu, dt, item, flag_vis=False):
    kpts_0_pred, kpts_1_pred = pred_cpu['keypoints0'], pred_cpu['keypoints1']

    # get gt match
    gt = {}
    # gt = dt.get_match_indices(item, kpts_0_pred, kpts_1_pred)
    gt = {**gt, **dt.get_match_matrix(item, kpts_0_pred, kpts_1_pred)}

    mscores = gt['scores']
    indices0 = mscores.argmax(1)[:-1]
    valid = indices0 < mscores.shape[1] - 1
    indices0[np.logical_not(valid)] = -1
    matches_0_gt = indices0

    # matches_0_gt = gt['matches0']
    pos_gt = matches_0_gt > -1
    mkpts_0_gt = kpts_0_pred[pos_gt]
    mkpts_1_gt = kpts_1_pred[matches_0_gt[pos_gt].astype(int)]

    '''statistics'''
    # get corrected match key-points
    matches_0_pred = pred_cpu['matches0']
    true_pred = matches_0_pred == matches_0_gt
    pos_pred = matches_0_pred > -1
    neg_pred = matches_0_pred <= -1
    true_pos = np.logical_and(pos_pred, true_pred)
    true_neg = np.logical_and(neg_pred, true_pred)

    mkpts_0_pos = kpts_0_pred[pos_pred]
    mkpts_1_pos = kpts_1_pred[matches_0_pred[pos_pred]]

    mkpts_0_true_pos = kpts_0_pred[true_pos]
    mkpts_1_true_pos = kpts_1_pred[matches_0_pred[true_pos]]

    # recall
    num_pos = np.sum(np.logical_and(pos_pred, pos_gt))       # get # of pts that get matched
    num_match_gt = np.sum(pos_gt)    # get # pts who can be matched
    recall = num_pos / num_match_gt if num_match_gt > 0 else 1.0

    # precision
    num_true = np.sum(true_pred)       # get # of pts that get matched correct
    num_pred = len(matches_0_pred)
    precision = num_true / num_pred if num_pred > 0 else 0.0

    '''vis'''
    if flag_vis:
        img_0, img_1 = dt[item]['image0'], dt[item]['image1']
        vis_gt_and_pred(1, img_0, kpts_0_pred, mkpts_1_gt, mkpts_0_pos, mkpts_0_true_pos, img_1, kpts_1_pred, mkpts_0_gt, mkpts_1_pos, mkpts_1_true_pos)

    return gt, recall, precision
    

def vis_gt_and_pred(wait: int, img_0, kpts_0_pred, mkpts_1_gt, mkpts_0_pos, mkpts_0_true_pos, img_1, kpts_1_pred, mkpts_0_gt, mkpts_1_pos, mkpts_1_true_pos):
    saving_dir = './results/'
    img_pt = slam_lib.vis.draw_pts(img_0, kpts_0_pred)
    cv2.imwrite(saving_dir + '/pt0.jpg', img_pt)
    cv2.namedWindow('pt0', cv2.WINDOW_NORMAL)
    cv2.imshow('pt0', img_pt)
    cv2.waitKey(wait)

    img_pt = slam_lib.vis.draw_pts(img_1, kpts_1_pred)
    cv2.imwrite(saving_dir + '/pt1.jpg', img_pt)
    cv2.namedWindow('pt1', cv2.WINDOW_NORMAL)
    cv2.imshow('pt1', img_pt)
    cv2.waitKey(wait)

    img3 = slam_lib.vis.draw_matches_(img_0, mkpts_0_gt, img_1, mkpts_1_gt)
    cv2.imwrite(saving_dir + '/match_gt.jpg', img3)
    cv2.namedWindow('match gt', cv2.WINDOW_NORMAL)
    cv2.imshow('match gt', img3)
    cv2.waitKey(wait)

    img_pt = slam_lib.vis.draw_pts(img_0, mkpts_0_pos)
    cv2.imwrite(saving_dir + '/mpt0.jpg', img_pt)
    cv2.namedWindow('mpt0', cv2.WINDOW_NORMAL)
    cv2.imshow('mpt0', img_pt)
    cv2.waitKey(wait)

    img_pt = slam_lib.vis.draw_pts(img_1, mkpts_1_pos)
    cv2.imwrite(saving_dir + '/mpt1.jpg', img_pt)
    cv2.namedWindow('mpt1', cv2.WINDOW_NORMAL)
    cv2.imshow('mpt1', img_pt)
    cv2.waitKey(wait)

    img3 = slam_lib.vis.draw_matches_(img_0, mkpts_0_pos, img_1, mkpts_1_pos)
    cv2.imwrite(saving_dir + '/match.jpg', img3)
    cv2.namedWindow('match pred', cv2.WINDOW_NORMAL)
    cv2.imshow('match pred', img3)
    cv2.waitKey(wait)

    img3 = slam_lib.vis.draw_matches_(img_0, mkpts_0_true_pos, img_1, mkpts_1_true_pos)
    cv2.imwrite(saving_dir + '/match_right.jpg', img3)
    cv2.namedWindow('match right', cv2.WINDOW_NORMAL)
    cv2.imshow('match right', img3)
    cv2.waitKey(wait)


def save_training_result(recall, precision, loss):
    fig1, ax1 = plt.subplots()
    ax1.plot(recall, label='recall')
    ax1.plot(precision, label='precision')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('%')
    ax1.set_title('recall & precision')
    ax1.legend()
    fig1.savefig('./results/r&p.png')
    # fig1.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(loss, label='loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('no unit')
    ax2.set_title('loss')
    ax2.legend()
    fig2.savefig('./results/loss.png')
    # fig2.show()


def train(dt, model, flag_vis=False):
    epochs = 100
    # resize = (400, 320)
    resize = (800, 640)
    # resize = [772, 516]
    # resize = (1544, 1032)

    # enable training
    torch.set_grad_enabled(True)
    torch.autograd.set_detect_anomaly(True)
    # print('Training ', model, 'at', model.cuda())

    optimizer = torch.optim.SGD(model.superglue.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.BCELoss()

    epoch_losses, epoch_recalls, epoch_precisions = [], [], []
    for epoch in range(epochs):
        epoch_loss, epoch_recall, epoch_precision = 0.0, 0.0, 0.0
        model.superglue.train()
        for i_data, data in enumerate(dt):
            img_0, img_1 = data['image0'], data['image1']
            gray_0, inp0, scales0 = read_image(img_0, device, resize, rotation=0, resize_float=True)
            gray_1, inp1, scales1 = read_image(img_1, device, resize, rotation=0, resize_float=True)

            pred = model({'image0': inp0, 'image1': inp1})
            pred_cpu = change_pred_to_cpu_and_image_frame(pred, scales0, scales1)

            gt, recall, precision = make_gt_from_pred(pred_cpu, dt, i_data, flag_vis=True)

            for k, v in gt.items():
                gt[k] = torch.from_numpy(v[None]).to(device)

            # back propagation
            optimizer.zero_grad()

            # loss_ = loss.nl_loss_1d_mscores(loss_fn, pred, gt)
            loss_ = loss.nl_loss_2d_mscores(loss_fn, pred, gt)

            '''logging'''
            epoch_loss += loss_.item() / len(dt)
            epoch_recall += recall / len(dt)
            epoch_precision += precision / len(dt)

            if i_data % 10 == 0:
                print('iter-', epoch, 'data-', i_data,
                      'average bce-loss:', epoch_loss*len(dt) / (i_data+1),
                      'average recall:', epoch_recall*len(dt) / (i_data+1),
                      'average precision:', epoch_precision*len(dt) / (i_data+1))

            '''exec'''
            loss_.backward()
            optimizer.step()

        '''epoch logging'''
        epoch_losses.append(epoch_loss), epoch_recalls.append(epoch_recall), epoch_precisions.append(epoch_precision)
        save_training_result(epoch_recalls, epoch_precisions, epoch_losses)
        print('iter-', epoch, 'bce-loss:', epoch_loss, 'recall:', epoch_recall, 'precision', epoch_precision)
    return


def main():
    """"""
    '''file paths'''
    match_data_dir = './data/hair_tf/'

    dt = DatasetImgTF()
    dt.load_img_from_dir(match_data_dir)

    # Load the SuperPoint and SuperGlue models.
    config = {
        'device': device,
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.5,
            'training': True
        }
    }

    matcher = Matching(config).to(device)

    train(dt, matcher, flag_vis=True)

    return


if __name__ == '__main__':
    main()
