import numpy as np
import torch
import torchvision
import cv2

def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        #derivative = torch.tensor([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        #hessian = torch.tensor([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            #hessianinv = torch.linalg.inv(hessian)
            offset = -hessianinv * derivative
            #offset = - torch.matmul(hessianinv, derivative)
            offset = np.squeeze(np.array(offset.T), axis=0)
            #offset = torch.transpose(offset,0,1).squeeze(0)
            coord += offset
    return coord


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i,j])
            #origin_max = torch.max(hm[i,j,:,:])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            #dr = torch.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i,j].copy()
            #dr[border: -border, border: -border,:,:] = hm[i,j,:,:].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            #dr = torchvision.transforms.GaussianBlur(kernel_size=(kernel, kernel))(dr)
            hm[i,j] = dr[border: -border, border: -border].copy()
            #hm[i,j,:,:] = dr[border: -border, border: -border,:,:].copy()
            hm[i,j] *= origin_max / np.max(hm[i,j])
            #hm[i,j] *= origin_max / torch.max(hm[i,j,:,:])
    return hm



def get_max_preds_dark(hm):
    coords, maxvals = get_max_preds(hm)
    heatmap_height = hm.shape[2]
    heatmap_width = hm.shape[3]
    

    # post-processing
    hm = gaussian_blur(hm, 11)
    hm = np.maximum(hm, 1e-10)
    #hm = torch.maximum(hm, 1e-10)
    hm = np.log(hm)
    #hm = torch.log(hm)
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            coords[n,p] = taylor(hm[n][p], coords[n][p])

    preds = coords.copy()
    preds[:, :, 0] = (preds[:, :, 0]) % heatmap_width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / heatmap_width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask

    # Transform back
    '''for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )'''

    return preds, maxvals

def get_max_preds_torch(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: torch.Tensor([batch_size, num_joints, height, width])
    '''
    #assert isinstance(batch_heatmaps, torch.tensor), \
        #'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)
    maxvals = torch.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = torch.tile(idx, (1, 1, 2)).type(torch.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = torch.tile(torch.gt(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.type(torch.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    elif hm_type == 'dark':
        pred, _ = get_max_preds_dark(output)
        target, _ = get_max_preds_dark(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros(len(idx))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0

    return acc, avg_acc, cnt, pred
