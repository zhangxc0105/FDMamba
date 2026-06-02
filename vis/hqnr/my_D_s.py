import numpy as np
# from skimage.metrics import structural_similarity as ssim
from .interp23 import interp23
from .imresize import imresize
import torch
from torch.nn import functional as F
import math

#
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()


def create_window(window_size, channel):
    # _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    # _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    _2D_window = torch.ones(window_size, window_size)#.uniform_(0, 1)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def D_s(I_F, I_MS, I_MS_LR, I_PAN, ratio, S, q):
    """ if 0, Toolbox 1.0, otherwise, original QNR paper """
    flag_orig_paper = 0


    if (I_F.shape != I_MS.shape):
        print("The two images must have the same dimensions")
        return -1

    N = I_F.shape[0]
    M = I_F.shape[1]
    Nb = I_F.shape[2]
    max_val = 1

    if not isinstance(I_PAN, torch.Tensor):
        I_F = torch.from_numpy(I_F/max_val).permute(2, 0, 1).float()*max_val
        I_MS = torch.from_numpy(I_MS / max_val).permute(2, 0, 1).float() * max_val
        I_MS_LR = torch.from_numpy(I_MS_LR / max_val).permute(2, 0, 1).float() * max_val
        I_PAN = torch.from_numpy(I_PAN / max_val).float()

    if (flag_orig_paper == 0):
        """Opt. 1 (as toolbox 1.0)"""
        pan_filt = interp23(imresize(I_PAN, 1 / ratio), ratio)
    else:
        """ Opt. 2 (as paper QNR) """
        pan_filt = imresize(I_PAN, 1 / ratio)

    pan_filt = torch.from_numpy(pan_filt / max_val).unsqueeze(0).float()
    I_PAN = I_PAN.unsqueeze(0).unsqueeze(0)
    I_F = I_F.unsqueeze(0)
    Q_high, Q_high_map = uqi_ssim(I_F, I_PAN, Nb, win_size=S)

    I_MS = I_MS.unsqueeze(0)
    pan_filt = pan_filt.unsqueeze(0)

    if (flag_orig_paper == 0):
        """ Opt. 1 (as toolbox 1.0) """
        Q_low, Q_low_map = uqi_ssim(I_MS, pan_filt, Nb, win_size=S)
    else:
        """ Opt. 2 (as paper QNR) """
        Q_low, Q_low_map = uqi_ssim(I_MS_LR, pan_filt, Nb, win_size=S)

    D_s_index = np.abs(Q_high - Q_low) ** q
    D_s_map = np.abs(Q_high_map - Q_low_map) ** q
    
    D_s_index = D_s_index.sum()

    D_s_index = (D_s_index / Nb) ** (1 / q)
    D_s_map = (D_s_map / Nb) ** (1 / q)


    return D_s_index.item(), D_s_map.numpy().transpose([1, 2, 0])


def uqi_ssim(x, y, Nb, win_size):
    # cov_norm = win_size**2 / (win_size**2 - 1)
    window_ms = create_window(win_size, Nb) / (win_size**2)
    window_pan = create_window(win_size, 1) / (win_size**2)


    mu1 = F.conv2d(x, window_ms, groups=Nb, stride=win_size)
    mu2 = F.conv2d(y, window_pan, stride=win_size)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2  # 点乘产生误差

    sigma1_sq = F.conv2d(x * x, window_ms, groups=Nb, stride=win_size) - mu1_sq # 点乘产生误差
    sigma2_sq = F.conv2d(y * y, window_pan, stride=win_size) - mu2_sq
    sigma12 = F.conv2d(x * y, window_ms, groups=Nb, stride=win_size) - mu1_mu2 # 点乘产生误差

    V1 = 2.0 * sigma12  # * cov_norm
    V2 = sigma1_sq + sigma2_sq

    Q = (2 * V1 * mu1_mu2) / (V2 * (mu1_sq + mu2_sq))
    Q = Q.squeeze(0)
    return Q.mean(1).mean(1), Q


def cov(x, y):
    cov_xy = torch.matmul(x.transpose(-2, -1), y) / x.shape[-2]
    cov_yx = torch.matmul(y.transpose(-2, -1), x) / y.shape[-2]
    std_x = x.var(-2, keepdims=True)
    std_y = y.var(-2, keepdims=True)
    # print(cov_xy.shape, cov_yx.shape, std_x.shape, std_y.shape)

    return torch.cat([std_x, cov_xy, cov_yx, std_y], dim=-2)

def uqi(x, y, Nb, win_size):
    C, H, W = x.shape
    x = F.unfold(x, kernel_size=win_size, stride=win_size)
    x = x.permute(1, 0).reshape(-1, C, win_size**2, 1)
    y = F.unfold(y, kernel_size=win_size, stride=win_size)
    y = y.permute(1, 0).reshape(-1, win_size**2, 1)
    num_patches = y.shape[0]
    # print(x.shape, y.shape, mx.shape, my.shape)
    Q = torch.zeros([num_patches, C])
    for ii in range(Nb):
        xx = x[:, ii, ...]#.unsqueeze(1)
        mx = xx.mean(-2, keepdims=True)
        my = y.mean(-2, keepdims=True)
        # print(xx.shape, y.shape, mx.shape, my.shape)
        C = cov(xx-mx, y-my)
        # print(C.shape)
        C = C.reshape(num_patches, 2, 2)
        Q[:, ii, ...] = (4 * C[:, 0, 1] * mx[:, 0, 0]*my[:, 0, 0]) / ((C[:, 0, 0] + C[:, 1, 1]) * (mx[:, 0, 0]**2 + my[:, 0, 0]**2))

    return Q.mean(0)
