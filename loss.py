import numpy as np
import torch
import numpy


def nl_loss_1d_mscores(ce_loss, pred, gt):
    assert pred

    indices0_pred = pred['matches0']         # use -1 for invalid hair_close_range_match
    indices1 = pred['matches1']         # use -1 for invalid dustbin match
    mscores0 = pred['matching_scores0']
    mscores1 = pred['matching_scores1']

    indices0_gt = gt['matches0']
    indices1_gt = gt['matches1']

    assert indices0_gt.size() == indices0_pred.size()
    assert indices1_gt.size() == indices1.size()

    '''get correct match from pred'''
    mask_match_correct = indices0_pred == indices0_gt

    ''' get mask for match between match object from gt'''
    mask_match_obj_gt = indices0_gt > -1

    ''' get mask for match to dustbin from gt'''
    mask_match0_dustbin = indices0_gt == -1
    mask_match1_dustbin = indices1_gt == -1

    ''' get correct match from gt obj '''
    # mask_match_correct_in_obj = mask_match_correct[mask_match_obj_gt]

    '''make pred match score'''
    mscores0_pred_obj = mscores0[torch.logical_and(mask_match_obj_gt, mask_match_correct)]
    mscores0_pred_dustbin = mscores0[mask_match0_dustbin]
    mscores1_pred_dustbin = mscores1[mask_match1_dustbin]
    mscores1_pred_general = torch.cat([mscores0_pred_obj, mscores0_pred_dustbin, mscores1_pred_dustbin], dim=0)

    '''make gt match score'''
    mscores0_gt_obj = mscores0.new_ones(mscores0_pred_obj.size())          # make wrong match 0
    # mscores0_gt_obj[mask_match_correct_in_obj] = 1.0                        # make correct match 1

    mscores0_gt_dustbin = mscores0.new_ones(mscores0_pred_dustbin.size())
    mscores1_gt_dustbin = mscores1.new_ones(mscores1_pred_dustbin.size())
    mscores1_gt_general = torch.cat([mscores0_gt_obj, mscores0_gt_dustbin, mscores1_gt_dustbin], dim=0)

    # loss_match_obj = ce_loss(mscores0_pred_obj, mscores0_gt_obj)
    loss_match_obj = ce_loss(mscores1_pred_general, mscores1_gt_general)

    return loss_match_obj


def nl_loss_2d_mscores(ce_loss, pred, gt):
    """"""
    '''build pred confusion matrix'''
    '''build gt confusion matrix'''
    '''losses'''

    confusion_matrix_pred = pred['scores']
    confusion_matrix_gt = gt['scores']
    confusion_matrix_gt = confusion_matrix_gt.type_as(confusion_matrix_pred)
    assert confusion_matrix_pred.size() == confusion_matrix_gt.size()
    loss_match_obj = ce_loss(confusion_matrix_pred, confusion_matrix_gt)

    return loss_match_obj