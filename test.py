import numpy as np
import torch
from models.superglue import arange_like
import matplotlib.pyplot as plt


def test_confusion_matrix():
    scores = torch.Tensor([[0.8, 0.1, 0.1],
                           [0.1, 0.9, 0.0],
                           [0.1, 0.0, 0.8]])

    # Get the matches with score above "match_threshold".
    # get indices
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    print('before filtering', indices0)
    print('before filtering', indices1)

    # filter indices by
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)    # mutual index
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)    #
    print(mutual0)
    print(mutual1)

    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)                    # matching score
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > 0.2)
    valid1 = mutual1 & valid0.gather(1, indices1)
    print(valid0)
    print(valid1)

    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    print('after filtering', indices0)
    print('after filtering', indices1)

    return {
        'matches0': indices0,  # use -1 for invalid hair_close_range_match
        'matches1': indices1,  # use -1 for invalid hair_close_range_match
        'matching_scores0': mscores0,
        'matching_scores1': mscores1,
    }


def test_indices():
    ids_0_2_1 = np.arange(0, 10)
    np.random.shuffle(ids_0_2_1)
    print(ids_0_2_1)

    ids_1_2_0 = np.zeros(ids_0_2_1.shape) - 1
    for i_0, id_0_2_1 in enumerate(ids_0_2_1):
        ids_1_2_0[id_0_2_1] = i_0
    print(ids_1_2_0)
    return ids_0_2_1


def test_plot():
    x = [i for i in range(10)]
    y = [i for i in range(1, 11)]
    z = [i*10 for i in range(10)]

    fig1, ax1 = plt.subplots()
    ax1.plot(x, label='recall')
    ax1.plot(y, label='precision')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('%')
    ax1.set_title('recall & precision')
    ax1.legend()

    fig1.savefig('./results/r&p.png')
    fig1.show()

    fig2, ax2 = plt.subplots()

    ax2.plot(z, label='loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('no unit')
    ax2.set_title('loss')
    ax2.legend()

    fig2.savefig('./results/loss.png')
    fig2.show()


def test_make_scores():
    """"""
    '''make hand-craft scores'''
    scores = np.asarray([np.eye(5)] * 3) - 0.1
    scores = torch.from_numpy(scores)
    scores[0, 0, 0] = 0
    scores[0, 0, 1] = 1

    scores[0, 1, 1] = 0
    scores[0, 1, 2] = 1

    scores[0, 2, 2] = 0
    scores[0, 2, 3] = 1

    scores[0, 3, 3] = 0
    scores[0, 3, 4] = 1

    scores[0, 4, 4] = 0
    scores[0, 4, 0] = 1

    # max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    max0, max1 = scores.max(2), scores.max(1)
    indices0, indices1 = max0.indices, max1.indices

    # filter indices by
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1,
                                                                indices0)   # mutual index, col max is also row max in that row
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1,
                                                                indices1)   # mutual index, row max is also col max in that col
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)                # matching score along col
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)     # matching score along row
    valid0 = mutual0 & (mscores0 > 0.2)                                     # filter low score along col
    valid1 = mutual1 & valid0.gather(1, indices1)                           # filter low score along row
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))       # fill invalid match indices with -1
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))       # fill invalid match indices with -1

    mask_dustbin0 = indices0[:, :-1] == scores.size()[1] - 1
    mask_dustbin1 = indices1[:, :-1] == scores.size()[0] - 1
    indices0_dustbin_marked = torch.where(torch.logical_not(mask_dustbin0), indices0[:, :-1], indices0.new_tensor(-1))  # fill invalid match indices with -1
    indices1_dustbin_marked = torch.where(torch.logical_not(mask_dustbin1), indices1[:, :-1], indices1.new_tensor(-1))  # fill invalid match indices with -1
    mscores0_dustbin_marked = torch.where(torch.logical_not(mask_dustbin0), mscores0[:, :-1],
                                          mscores0.new_tensor(0.0))  # fill invalid match indices with -1
    mscores1_dustbin_marked = torch.where(torch.logical_not(mask_dustbin1), mscores1[:, :-1],
                                          mscores1.new_tensor(0.0))  # fill invalid match indices with -1

    # make match score confusion matrix
    # indices0_valid, indices1_valid = indices0[valid0], indices1[valid1]
    # indices_valid = torch.cat([indices0_valid[:, :, None], indices1_valid[:, :, None]], dim=-1)
    # mscores = scores.new_zeros(scores.size())
    # mscores[indices_valid] = 1.0

    mscores = scores.new_zeros(scores.size())
    for i_batch in range(len(mscores)):
        # indices0_valid, indices1_valid = indices0[i_batch][valid0[i_batch]], indices1[i_batch][valid1[i_batch]]
        indices0_valid, indices1_valid = indices0[i_batch], indices1[i_batch]
        mscores0_valid = mscores0[i_batch]
        indices_valid_0 = torch.arange(0, indices0_valid.shape[0])[:, None]
        indices_valid_1 = indices0_valid[:, None]
        indices_valid = torch.cat([indices_valid_0, indices_valid_1], dim=-1)
        mscores[i_batch][indices_valid.T.tolist()] = mscores0_valid

    print(scores[0, :, :])
    # print(indices1[0, :])
    print(mutual0[0, :])
    print(mask_dustbin0[0, :])
    print(indices0[0, :])
    print(indices0_dustbin_marked[0, :])
    print(mscores0[0])
    print(mscores0_dustbin_marked[0])
    print(mscores[0, :, :])
    print(mscores[1, :, :])


if __name__ == '__main__':
    test_make_scores()
    # test_plot()
