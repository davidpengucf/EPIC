import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from dalib.modules.gl import WarmStartGradientLayer
from metric import get_max_preds, get_max_preds_torch, get_max_preds_dark
import copy

from typing import Optional, Any, Tuple
#import numpy as np
#import torch.nn as nn
from torch.autograd import Function
#import torch
from einops.layers.torch import Rearrange
from torch import distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradientFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output * ctx.coeff, None


class WarmStartGradientLayer(nn.Module):
    """Warm Start Gradient Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Parameters:
            - **alpha** (float, optional): :math:`α`. Default: 1.0
            - **lo** (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            - **hi** (float, optional): Final value of :math:`\lambda`. Default: 1.0
            - **max_iters** (int, optional): :math:`N`. Default: 1000
            - **auto_step** (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class FastPseudoLabelGenerator2d(nn.Module):
    def __init__(self, sigma=2):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, heatmap: torch.Tensor):
        heatmap = heatmap.detach()
        height, width = heatmap.shape[-2:]
        idx = heatmap.flatten(-2).argmax(dim=-1) # B, K
        pred_h, pred_w = idx.div(width, rounding_mode='floor'), idx.remainder(width) # B, K
        delta_h = torch.arange(height, device=heatmap.device) - pred_h.unsqueeze(-1) # B, K, H
        delta_w = torch.arange(width, device=heatmap.device) - pred_w.unsqueeze(-1) # B, K, W
        gaussian = (delta_h.square().unsqueeze(-1) + delta_w.square().unsqueeze(-2)).div(-2 * self.sigma * self.sigma).exp() # B, K, H, W
        ground_truth = F.threshold(gaussian, threshold=1e-2, value=0.)

        ground_false = (ground_truth.sum(dim=1, keepdim=True) - ground_truth).clamp(0., 1.)
        return ground_truth, ground_false


class PseudoLabelGenerator2d(nn.Module):
    """
    Generate ground truth heatmap and ground false heatmap from a prediction.

    Args:
        num_keypoints (int): Number of keypoints
        height (int): height of the heatmap. Default: 64
        width (int): width of the heatmap. Default: 64
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    Inputs:
        - y: predicted heatmap

    Outputs:
        - ground_truth: heatmap conforming to Gaussian distribution
        - ground_false: ground false heatmap

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - ground_truth: :math:`(minibatch, K, H, W)`
        - ground_false: :math:`(minibatch, K, H, W)`
    """
    def __init__(self, num_keypoints, height=64, width=64, sigma=2):
        super(PseudoLabelGenerator2d, self).__init__()
        self.height = height
        self.width = width
        self.sigma = sigma
        #self.dark = dark

        heatmaps = np.zeros((width, height, height, width), dtype=np.float32)

        tmp_size = sigma * 3
        for mu_x in range(width):
            for mu_y in range(height):
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], width)
                img_y = max(0, ul[1]), min(br[1], height)

                heatmaps[mu_x][mu_y][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        self.heatmaps = heatmaps
        self.false_matrix = 1. - np.eye(num_keypoints, dtype=np.float32) #(21, 21)

    def forward(self, y):
        B, K, H, W = y.shape
        #print("model-PseudoLabelGenerator2d: y", y.shape) # (32, 21, 64, 64)
        #print("model-PseudoLabelGenerator2d: y", y)
        y = y.detach()
        preds, max_vals = get_max_preds(y.cpu().numpy())
        
        #if self.dark:
            #preds, max_vals = get_max_preds_dark(y.cpu().numpy())
        
        # B x K x (x, y)
        #print("model-PseudoLabelGenerator2d: preds before", preds.shape) # (32, 21, 2)
        #print("model-PseudoLabelGenerator2d: preds before", preds)
        preds = preds.reshape(-1, 2).astype(np.int)
        #print("model-PseudoLabelGenerator2d: preds after", preds.shape) # (672, 2)
        #print("model-PseudoLabelGenerator2d: preds after", preds)

        #print("model-PseudoLabelGenerator2d: heatmaps", self.heatmaps.shape)#(64, 64, 64, 64)
        #print("model-PseudoLabelGenerator2d: heatmaps", self.heatmaps)
        #print("model-PseudoLabelGenerator2d: heatmaps", self.heatmaps[preds[:, 0], preds[:, 1], :, :].shape)
        # (672, 64, 64)
        #print("model-PseudoLabelGenerator2d: heatmaps", self.heatmaps[preds[:, 0], preds[:, 1], :, :])


        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()
        # (32, 21, 64, 64)

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1)) #(32, 64*64, 21)
        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0, 2, 1)).reshape(B, K, H, W).copy()
        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)
    
class PseudoLabelGenerator2dDark(nn.Module):
    
    def __init__(self, num_keypoints, height=64, width=64, sigma=2):
        super(PseudoLabelGenerator2dDark, self).__init__()
        self.height = height
        self.width = width
        self.sigma = sigma
        #self.dark = dark

        heatmaps = np.zeros((width, height, height, width), dtype=np.float32)

        tmp_size = sigma * 3
        for mu_x in range(width):
            for mu_y in range(height):
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], width)
                img_y = max(0, ul[1]), min(br[1], height)

                heatmaps[mu_x][mu_y][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        self.heatmaps = heatmaps
        self.false_matrix = 1. - np.eye(num_keypoints, dtype=np.float32) #(21, 21)

    def forward(self, y):
        B, K, H, W = y.shape
        #print("model-PseudoLabelGenerator2d: y", y.shape) # (32, 21, 64, 64)
        #print("model-PseudoLabelGenerator2d: y", y)
        y = y.detach()
        preds, max_vals = get_max_preds_dark(y.cpu().numpy())
        
        #if self.dark:
            #preds, max_vals = get_max_preds_dark(y.cpu().numpy())
        
        # B x K x (x, y)
        #print("model-PseudoLabelGenerator2d: preds before", preds.shape) # (32, 21, 2)
        #print("model-PseudoLabelGenerator2d: preds before", preds)
        preds = preds.reshape(-1, 2).astype(np.int)
        #print("model-PseudoLabelGenerator2d: preds after", preds.shape) # (672, 2)
        #print("model-PseudoLabelGenerator2d: preds after", preds)

        #print("model-PseudoLabelGenerator2d: heatmaps", self.heatmaps.shape)#(64, 64, 64, 64)
        #print("model-PseudoLabelGenerator2d: heatmaps", self.heatmaps)
        #print("model-PseudoLabelGenerator2d: heatmaps", self.heatmaps[preds[:, 0], preds[:, 1], :, :].shape)
        # (672, 64, 64)
        #print("model-PseudoLabelGenerator2d: heatmaps", self.heatmaps[preds[:, 0], preds[:, 1], :, :])


        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()
        # (32, 21, 64, 64)

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1)) #(32, 64*64, 21)
        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0, 2, 1)).reshape(B, K, H, W).copy()
        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)


class RegressionDisparity(nn.Module):
    
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator2d, criterion: nn.Module):
        super(RegressionDisparity, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator

    def forward(self, y, y_adv, weight=None, mode='min'):
        assert mode in ['min', 'max']
        ground_truth, ground_false = self.pseudo_label_generator(y.detach())
        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)
        


class PoseResNet2d(nn.Module):
    
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet2d, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)

        if self.training:
            return y, y_adv
        else:
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()

        
class PoseResNet2d_GVB(nn.Module):
    
    def __init__(self, backbone, upsampling1, upsampling2, upsampling1_adv, upsampling2_adv, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet2d_GVB, self).__init__()
        self.backbone = backbone
        
        self.upsampling1 = upsampling1
        self.head1 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.upsampling2 = upsampling2
        self.head2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        
        self.upsampling1_adv = upsampling1_adv
        self.head1_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.upsampling2_adv = upsampling2_adv
        self.head2_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        
        f1 = self.upsampling1(x)
        f2 = self.upsampling2(x)
        #f = f1 - f2
        y1 = self.head1(f1)
        y2 = self.head2(f2)
        y = y1 - y2
        
        
        #f_adv = self.gl_layer(f)
        f1_adv = self.upsampling1_adv(x)
        f1_adv = self.gl_layer(f1_adv)
        f2_adv = self.upsampling2_adv(x)
        f2_adv = self.gl_layer(f2_adv)
        #f_adv = f2_adv - f1_adv
        y1_adv = self.head1_adv(f1_adv)
        y2_adv = self.head2_adv(f2_adv)
        y_adv = y1_adv - y2_adv
        
        
        #y = self.head(f)
        #y_adv = self.head_adv(f_adv)
        
        

        if self.training:
            return y1, y1_adv, y2, y2_adv, y, y_adv
        else:
            return y1

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
        ]

    def step(self):
        
        self.gl_layer.step()
        
        
class RegDAPoseResNetRLE(nn.Module):
   
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(RegDAPoseResNetRLE, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.regr = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(num_keypoints * 64 * 64, num_keypoints * 2),
            Rearrange('b (c d) -> b c d',d=2),    
        )
        
        self.regr_adv = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(num_keypoints * 64 * 64, num_keypoints * 2),
            Rearrange('b (c d) -> b c d',d=2),    
        )
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        
        y_cor,_ = get_max_preds_torch(y)
        y_adv_cor,_ = get_max_preds_torch(y_adv)
        
        y_sig = self.regr(y)
        y_adv_sig = self.regr_adv(y_adv)

        if self.training:
            return y, y_adv, y_cor, y_adv_cor, y_sig, y_adv_sig
        else:
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.regr.parameters(), 'lr': lr},
            {'params': self.regr_adv.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
        
# Copyright (c) OpenMMLab. All rights reserved.
#import torch
#import torch.nn as nn
#from torch import distributions


class RealNVP(nn.Module):
    

    @staticmethod
    def get_scale_net():
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def get_trans_net():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super(RealNVP, self).__init__()

        self.register_buffer('loc', torch.zeros(2))
        self.register_buffer('cov', torch.eye(2))
        self.register_buffer(
            'mask', torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))
        
        self.loc = self.loc.cuda()
        self.cov = self.cov.cuda()
        self.mask = self.mask.cuda()

        self.s = torch.nn.ModuleList(
            [self.get_scale_net().to(device) for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList(
            [self.get_trans_net().to(device) for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""

        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        z = z.cuda()
        log_det_jacob = log_det_jacob.cuda()
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            #print('self.mask[i]: ',self.mask[i].is_cuda)
            #print('self.mask[i]: ',(1-self.mask[i]).is_cuda)
            #print('z_: ',z_.is_cuda)
            #print('z_: ',self.s[i](z_).is_cuda)
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""

        z, log_det = self.backward_p(x)
        print("z: ",torch.isnan(z).any())
        print("log_det: ",torch.isnan(log_det).any())
        if torch.isnan(z).any():
            torch.nan_to_num(z, nan=0.1)
        if torch.isnan(log_det).any():
            torch.nan_to_num(log_det, nan=0.1)
        #print('z: ',z.shape)
        #print('z: ',z)
        #print("log_det: ",log_det.shape)
        #print("log_det: ",log_det)
        return self.prior.log_prob(z) + log_det
