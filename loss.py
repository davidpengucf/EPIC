"""
Modified from https://github.com/microsoft/human-pose-estimation.pytorch
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metric import get_max_preds_torch
import einops
from model import RealNVP

class JointsOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.
    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        topk (int): Only top k joint losses are kept.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, topk=8, loss_weight=1.):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, loss):
        """Online hard keypoint mining."""
        ohkm_loss = 0.
        N = len(loss)
        for i in range(N):
            sub_loss = loss[i]
            _, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= N
        return ohkm_loss

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)
        if num_joints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not '
                             f'larger than num_joints ({num_joints}).')
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        losses = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                losses.append(
                    self.criterion(heatmap_pred * target_weight[:, idx],
                                   heatmap_gt * target_weight[:, idx]))
            else:
                losses.append(self.criterion(heatmap_pred, heatmap_gt))

        losses = [loss.mean(dim=1).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight

class EntropyLoss(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, mi=False):
        super(EntropyLoss, self).__init__()
        self.mi = mi

    def forward(self, target, target_weight=None):
        '''
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = heatmaps_gt + self.epsilon
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)
        if target_weight is not None:
            loss = loss * target_weight.view((B, K))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)'''
        
        B, K, _, _ = target.shape
        heatmaps_tgt = target.reshape((B, K, -1))
        entropy_tgt = - F.softmax(heatmaps_tgt,dim=-1) * F.log_softmax(heatmaps_tgt,dim=-1)
        entropy_loss = entropy_tgt.mean()
        
        heatmaps_mean = heatmaps_tgt.mean(dim=-2)
        entropy_mean = - F.softmax(heatmaps_mean,dim=-1) * F.log_softmax(heatmaps_mean,dim=-1)
        mean_loss = entropy_mean.mean() 
        
        if self.mi:
            
            return entropy_loss - mean_loss
        else:
            return - mean_loss

class RLELoss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_dis='laplace'):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_dis = q_dis

        self.flow_model = RealNVP()

    def forward(self, pred, sigma, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D*2]): Output regression,
                    including coords and sigmas.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        #pred = output[:, :, :2]
        #sigma = output[:, :, 2:4].sigmoid()
        sigma = sigma.softmax(dim=2)

        error = (pred - target) / (sigma + 1e-9)
        # (B, K, 2)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1],
                                             2)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_dis in ['laplace', 'gaussian', 'strict']
            if self.q_dis == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_weight(source, target, weight, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(weight * (XX + YY - XY -YX))
    return loss

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def mmd_loss(pred_hm, gt_hm, weight):
    B, C, H, W = pred_hm.shape
    loss = torch.tensor(0.).cuda()
    
    for i in range(B):
    

        #pred = pred_hm[i,j,:,:].flatten() * weight[i,j,:] #.reshape(1, -1)
        pred = pred_hm[i,:,:,:].reshape(C, -1)
        #print('pred:', pred.shape)
        #print('weight:', weight[i,:,:].shape)
        gt = gt_hm[i,:,:,:].reshape(C, -1)
        #print('pred:', type(pred))
        #print('gt:', type(gt))
        #print('weight:', type(weight[i,:,:]))
        loss += mmd_weight(pred, gt, weight[i,:,:])
            
    loss /= B * C 
    #print('loss:', loss.item())
    
    return loss


def mmd_loss_negative(pred_hm, gt_hm, weight):
    B, C, H, W = pred_hm.shape
    loss = torch.tensor(0.).cuda()
    
    for i in range(B):
        for j in range(C):
        
            pred = pred_hm[i,j,:,:].flatten() * weight[i,j,:] #.reshape(1, -1)
            #print('pred:', pred.shape)
            #print('weight:', weight[i,:,:].shape)
            stad = einops.repeat(pred, 'n -> k n', k=C-1)
            #print('select: ',torch.cat((pred_hm[i, :j,:,:], pred_hm[i,(j+1):,:,:]),dim=1).shape)
            if j != 0 and j!= C-1:
                #print('pred_1:', (torch.cat((pred_hm[i, :j,:,:].reshape(-1, H*W).unsqueeze(0), pred_hm[i,(j+1):,:,:].reshape(-1, H*W).unsqueeze(0)), dim=1).squeeze(0).shape))
                #print('weight_1:',torch.cat((weight[i, :j,:].unsqueeze(0), weight[i,(j+1):,:].unsqueeze(0)),dim=1).squeeze(0).shape)
                comp = torch.cat((pred_hm[i, :j,:,:].reshape(-1, H*W).unsqueeze(0), pred_hm[i,(j+1):,:,:].reshape(-1, H*W).unsqueeze(0)), dim=1).squeeze(0) * torch.cat((weight[i, :j,:].unsqueeze(0), weight[i,(j+1):,:].unsqueeze(0)),dim=1).squeeze(0)
            
            elif j == 0:
                
                comp = pred_hm[i,(j+1):,:,:].reshape(C-1, -1) * weight[i,(j+1):,:]
            else:
                comp = pred_hm[i,:j,:,:].reshape(C-1, -1) * weight[i,:j,:]
            loss += mmd(stad, comp)
        #gt = gt_hm[i,:,:,:].reshape(C, -1)
        #print('pred:', type(pred))
        #print('gt:', type(gt))
        #print('weight:', type(weight[i,:,:]))
        #loss += mmd_weight(pred, gt, weight[i,:,:])
            
    loss /= B * C * (C - 1)
    print('loss:', loss.item())
    
    return -loss

def oks_loss(pred_hm, gt_hm, weight, num_keypoints):
    var = torch.Tensor(np.array([4.])).cuda()
    area = torch.Tensor(np.array([256 * 256])).cuda()

    pred, _ = get_max_preds_torch(pred_hm)
    gt, _ = get_max_preds_torch(gt_hm)

    kpt_preds = pred.reshape(-1, pred.size(-1) // 2, 2)
    kpt_gts = gt.reshape(-1, gt.size(-1) // 2, 2)

    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
                       (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
    #print("squared_distance: ", squared_distance.shape)
    # assert (kpt_valids.sum(-1) > 0).all()
    squared_distance0 = squared_distance / (
            area * var * 2)
    squared_distance1 = torch.exp(-squared_distance0)
    #print("squared_distance1: ",squared_distance1.shape)
    #print("weight: ",weight.shape)
    squared_distance1 = torch.clamp(squared_distance1, min=0, max=100) * weight.reshape(-1,1)  # kpt_valids
    oks = squared_distance1.sum(dim=0) / weight.reshape(-1,1).sum(dim=0)
    #print("oks: ", oks)
    #print("oks: ", oks.shape)

    loss = -oks.log()

    return loss
 


class JointsMSELoss(nn.Module):
    """
    Typical MSE loss for keypoint detection.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean'):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_gt = target.reshape((B, K, -1))
        loss = self.criterion(heatmaps_pred, heatmaps_gt) * 0.5
        if target_weight is not None:
            loss = loss * target_weight.view((B, K, 1))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)


class JointsKLLoss(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = heatmaps_gt + self.epsilon
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)
        if target_weight is not None:
            loss = loss * target_weight.view((B, K))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)
