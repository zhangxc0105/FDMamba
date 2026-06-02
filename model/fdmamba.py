import time
import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=4):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution=None, num_tokens=64, inner_rank=64, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

        self.hprompt_proj = nn.Sequential(
            nn.Conv2d(dim, self.d_state, 1, 1, 0),
        )

    def forward(self, x, x_size, token, hf):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        hf = hf.view(B, C, -1).permute(0, 2, 1)

        pred_route = self.route(hf)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices)  # SGN-unfold

        semantic_prompt = semantic_neighbor(prompt, x_sort_indices)

        y = self.selectiveScan(semantic_x, semantic_prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse)  # SGN-fold

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  # mambairV2 needs only 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C, L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        # our ASE here ---
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class ASSMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            inner_rank=64,
            drop_path: float = 0.0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 32,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.inner_rank = inner_rank
        self.ln_1 = norm_layer(hidden_dim)
        self.assm = ASSM(dim=hidden_dim, d_state=d_state, inner_rank=inner_rank)
        self.drop_path = DropPath(drop_path)

        self.scale1 = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.scale2 = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ln_3 = nn.LayerNorm(hidden_dim)

        self.embeddingA = nn.Embedding(inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)
        self.convffn = ConvFFN(in_features=hidden_dim, hidden_features=hidden_dim, kernel_size=3)

    def forward(self, input, x_size, hf):
        # x [B, HW, C]
        x = self.ln_1(input)
        x = input * self.scale1 + self.drop_path(self.assm(x, x_size, self.embeddingA, hf))
        x = x * self.scale2 + self.convffn(self.ln_2(x), x_size)
        return x


class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Dropout2d(0.05)
        )

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class LayerNormProxy(nn.Module):
    """Copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class RountingFunction(nn.Module):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.05, proportion=45.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels * 3)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proportion = proportion / 180.0 * math.pi
        self.preset_angle = proportion / 180.0 * math.pi

        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                 groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                 groups=in_channels, bias=False)
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y_kernel = sobel_x_kernel.t()
        self.sobel_x.weight.data = sobel_x_kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.sobel_y.weight.data = sobel_y_kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)

        self.dropout_alpha = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels * 3, kernel_number, bias=True)

        self.dropout_theta = nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels * 3, kernel_number, bias=True)
        self.act_theta = nn.Tanh()

        self.act_alpha = nn.Sigmoid()

        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):
        grad_x = self.sobel_x(x)  # horizontal gradient
        grad_y = self.sobel_y(x)  # vertical gradient

        spatial = self.dwc(x)

        x = torch.cat([grad_x, grad_y, spatial], dim=1)  # [bs, in_channels*3, h, w]

        # normalization and activation
        x = self.norm(x)
        x = self.relu(x)

        # global context
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)  # [bs, in_channels*3]

        # predict kernel weights (alphas)
        alphas = self.dropout_alpha(x)
        alphas = self.fc_alpha(alphas)
        alphas = self.act_alpha(alphas)  # [bs, kernel_number]

        # predict angle offset (Δθ)
        delta_theta = self.dropout_theta(x)
        delta_theta = self.fc_theta(delta_theta)
        delta_theta = self.act_theta(delta_theta) * (math.pi / 4)  # limit Δθ to [-π/4, π/4]

        # compute final angle: θ_final = θ_preset + Δθ
        angles = delta_theta + self.preset_angle  # broadcast preset angle

        return alphas, angles


def _get_rotation_matrix(thetas):
    bs, g = thetas.shape
    device = thetas.device
    thetas = thetas.reshape(-1)  # [bs, n] --> [bs x n]

    x = torch.cos(thetas)
    y = torch.sin(thetas)
    x = x.unsqueeze(0).unsqueeze(0)  # shape = [1, 1, bs * g]
    y = y.unsqueeze(0).unsqueeze(0)
    a = x - y
    b = x * y
    c = x + y

    rot_mat_positive = torch.cat((
        torch.cat((a, 1 - a, torch.zeros(1, 7, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs * g, device=device), x - b, b, torch.zeros(1, 1, bs * g, device=device), 1 - c + b, y - b, torch.zeros(1, 3, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs * g, device=device), a, torch.zeros(1, 2, bs * g, device=device), 1 - a, torch.zeros(1, 3, bs * g, device=device)), dim=1),
        torch.cat((b, y - b, torch.zeros(1, 1, bs * g, device=device), x - b, 1 - c + b, torch.zeros(1, 4, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs * g, device=device), torch.ones(1, 1, bs * g, device=device), torch.zeros(1, 4, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs * g, device=device), 1 - c + b, x - b, torch.zeros(1, 1, bs * g, device=device), y - b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs * g, device=device), 1 - a, torch.zeros(1, 2, bs * g, device=device), a, torch.zeros(1, 2, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs * g, device=device), y - b, 1 - c + b, torch.zeros(1, 1, bs * g, device=device), b, x - b, torch.zeros(1, 1, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs * g, device=device), 1 - a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative = torch.cat((
        torch.cat((c, torch.zeros(1, 2, bs * g, device=device), 1 - c, torch.zeros(1, 5, bs * g, device=device)), dim=1),
        torch.cat((-b, x + b, torch.zeros(1, 1, bs * g, device=device), b - y, 1 - a - b, torch.zeros(1, 4, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs * g, device=device), 1 - c, c, torch.zeros(1, 6, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs * g, device=device), x + b, 1 - a - b, torch.zeros(1, 1, bs * g, device=device), -b, b - y, torch.zeros(1, 1, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs * g, device=device), torch.ones(1, 1, bs * g, device=device), torch.zeros(1, 4, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs * g, device=device), b - y, -b, torch.zeros(1, 1, bs * g, device=device), 1 - a - b, x + b, torch.zeros(1, 3, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs * g, device=device), c, 1 - c, torch.zeros(1, 1, bs * g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs * g, device=device), 1 - a - b, b - y, torch.zeros(1, 1, bs * g, device=device), x + b, -b), dim=1),
        torch.cat((torch.zeros(1, 5, bs * g, device=device), 1 - c, torch.zeros(1, 2, bs * g, device=device), c), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    mask = (thetas >= 0).unsqueeze(0).unsqueeze(0)
    mask = mask.float()  # shape = [1, 1, bs*g]
    rot_mat = mask * rot_mat_positive + (1 - mask) * rot_mat_negative  # shape = [k*k, k*k, bs*g]
    rot_mat = rot_mat.permute(2, 0, 1)  # shape = [bs*g, k*k, k*k]
    rot_mat = rot_mat.reshape(bs, g, rot_mat.shape[1], rot_mat.shape[2])  # shape = [bs, g, k*k, k*k]
    return rot_mat


def batch_rotate_multiweight(weights, lambdas, thetas):
    """Rotate weights based on learned angles.

    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        lambdas: tensor, shape = [batch_size, kernel_number]
        thetas: tensor, shape = [batch_size, kernel_number]

    Returns:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert thetas.shape == lambdas.shape
    assert lambdas.shape[1] == weights.shape[0]

    b = thetas.shape[0]
    n = thetas.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape

    if k == 3:
        # Stage 1:
        # input: thetas: [b, n]
        #        lambdas: [b, n]
        # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

        # Sub-Stage 1.1:
        # input: [b, n] kernel
        # output: [b, n, 9, 9] rotation matrix
        rotation_matrix = _get_rotation_matrix(thetas)

        # Sub-Stage 1.2:
        # input: [b, n, 9, 9] rotation matrix
        #        [b, n] lambdas
        #    --> [b, n, 1, 1] lambdas
        #    --> [b, n, 1, 1] lambdas dot [b, n, 9, 9] rotation matrix
        #    --> [b, n, 9, 9] rotation matrix with gate (done)
        # output: [b, n, 9, 9] rotation matrix with gate
        lambdas = lambdas.unsqueeze(2).unsqueeze(3)
        rotation_matrix = torch.mul(rotation_matrix, lambdas)

        # Sub-Stage 1.3: Reshape
        # input: [b, n, 9, 9] rotation matrix with gate
        # output: [b*9, n*9] rotation matrix with gate
        rotation_matrix = rotation_matrix.permute(0, 2, 1, 3)
        rotation_matrix = rotation_matrix.reshape(b * k * k, n * k * k)

        # Stage 2: Reshape
        # input: weights: [n, Cout, Cin, 3, 3]
        #             --> [n, 3, 3, Cout, Cin]
        #             --> [n*9, Cout*Cin] done
        # output: weights: [n*9, Cout*Cin]
        weights = weights.permute(0, 3, 4, 1, 2)
        weights = weights.contiguous().view(n * k * k, Cout * Cin)

        # Stage 3: torch.mm
        # [b*9, n*9] x [n*9, Cout*Cin]
        # --> [b*9, Cout*Cin]
        weights = torch.mm(rotation_matrix, weights)

        # Stage 4: Reshape Back
        # input: [b*9, Cout*Cin]
        #    --> [b, 3, 3, Cout, Cin]
        #    --> [b, Cout, Cin, 3, 3]
        #    --> [b * Cout, Cin, 3, 3] done
        # output: [b * Cout, Cin, 3, 3]
        weights = weights.contiguous().view(b, k, k, Cout, Cin)
        weights = weights.permute(0, 3, 4, 1, 2)
        weights = weights.reshape(b * Cout, Cin, k, k)
    else:
        thetas = thetas.reshape(-1)  # [bs, n] --> [bs x n]

        x = torch.cos(thetas)
        y = torch.sin(thetas)
        rotate_matrix = torch.tensor([[x, -y, 0], [y, x, 0]])
        rotate_matrix = rotate_matrix.unsqueeze(0).repeat(n, 1, 1)

        weights = weights.contiguous().view(n, Cout * Cin, k, k)

        grid = F.affine_grid(rotate_matrix, weights.shape)
        weights = F.grid_sample(weights, grid, mode='bilinear')

    return weights


class AdaptiveRotatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.rounting_func = rounting_func
        self.rotate_func = rotate_func

        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number,
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        alphas, angles = self.rounting_func(x)

        # rotate weight
        rotated_weight = self.rotate_func(self.weight, alphas, angles)

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]

        # adaptive conv over images using group conv
        out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))

        # reshape back
        out = out.reshape(bs, self.out_channels, *out.shape[2:])
        return out


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features),
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(0.01)
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class resblock(nn.Module):
    def __init__(self, channel):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=channel, init=0.01)

    def forward(self, x):
        rs1 = self.act(self.conv1(x))
        rs2 = self.conv2(rs1) + x
        return rs2


class HighFreqGate(nn.Module):
    def __init__(self, channels):
        super(HighFreqGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_high):
        return self.gate(x_high) * x_high


class combine(nn.Module):
    def __init__(self, channel):
        super(combine, self).__init__()
        self.resblock = resblock(channel=channel)
        self.a = nn.Parameter(torch.tensor(0.33), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.33), requires_grad=True)

    def forward(self, x1, x2, x3):
        rs1 = self.a * x1 + self.b * x2 + (1 - self.a - self.b) * x3
        rs2 = self.resblock(rs1)
        return rs2


class combine1(nn.Module):
    def __init__(self, channel):
        super(combine1, self).__init__()
        self.resblock = resblock(channel=channel)
        self.c = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x1, x2):
        rs1 = self.c * x1 + (1 - self.c) * x2
        rs2 = self.resblock(rs1)
        return rs2


class FFN(nn.Module):
    def __init__(self, in_channel, FFN_channel, out_channel):
        super(FFN, self).__init__()
        self.FFN_channel = FFN_channel
        self.out_channel = out_channel
        self.linear_1 = nn.Linear(in_channel, FFN_channel)
        self.conv1 = nn.Conv2d(FFN_channel, FFN_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(FFN_channel, FFN_channel, 1, 1, 0, bias=True)
        self.linear_2 = nn.Linear(FFN_channel, out_channel)
        self.act = nn.PReLU(num_parameters=FFN_channel, init=0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        rs1 = self.linear_1(x.permute(0, 2, 3, 1).reshape(B, -1, C)).permute(0, 2, 1).reshape(B, self.FFN_channel, H, W)
        rs2 = self.act(self.conv1(rs1))
        rs3 = self.conv2(rs2) + rs1
        rs4 = self.linear_2(rs3.permute(0, 2, 3, 1).reshape(B, -1, self.FFN_channel)).permute(0, 2, 1).reshape(B, self.out_channel, H, W)
        return rs4


class DownFRG(nn.Module):
    def __init__(self, dim, n_l_blocks=1, n_h_blocks=1, expand=2):
        super().__init__()
        self.dim = dim
        self.dwt = DWT()

        self.l_blk = nn.Sequential(*[ASSMBlock(dim) for _ in range(n_l_blocks)])

        self.h_fusion = SKFF(dim, height=3, reduction=2)

        routing_functionlh = RountingFunction(in_channels=dim, kernel_number=1, proportion=0)
        routing_functionhl = RountingFunction(in_channels=dim, kernel_number=1, proportion=90.0)
        routing_functionhh = RountingFunction(in_channels=dim, kernel_number=1, proportion=45.0)
        self.roteconvlh = AdaptiveRotatedConv2d(in_channels=dim, out_channels=dim, kernel_size=3, rounting_func=routing_functionlh)
        self.roteconvhl = AdaptiveRotatedConv2d(in_channels=dim, out_channels=dim, kernel_size=3, rounting_func=routing_functionhl)
        self.roteconvhh = AdaptiveRotatedConv2d(in_channels=dim, out_channels=dim, kernel_size=3, rounting_func=routing_functionhh)

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.dr = nn.Dropout(0.05)
        self.act = nn.PReLU(num_parameters=dim, init=0.01)
        self.pan_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.hf = HighFreqGate(dim)
        self.combine = combine(dim)
        self.mlp = FFN(in_channel=dim, FFN_channel=dim // 2, out_channel=dim)
        self.resblock = resblock(channel=dim)
        self.ca = ChannelAttention(dim)

    def forward(self, x, pan, ms):
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)

        pan = self.pan_conv(pan)
        pan_LL, pan_HL, pan_LH, pan_HH = self.dwt(pan)
        ms = self.mlp(ms)
        b, c, h, w = x_LL.shape

        x_HL = self.hf(x_HL) + pan_HL
        x_LH = self.hf(x_LH) + pan_LH
        x_HH = self.hf(x_HH) + pan_HH

        x_HL = self.dr(self.act(self.bn1(self.roteconvhl(x_HL))))
        x_LH = self.dr(self.act(self.bn2(self.roteconvlh(x_LH))))
        x_HH = self.dr(self.act(self.bn3(self.roteconvhh(x_HH))))

        x_h = self.ca(self.h_fusion([x_HL, x_LH, x_HH]))

        x_LL = self.resblock(self.combine(x_LL, pan_LL, ms))
        x_LL = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x_LL = l_layer(x_LL, [h, w], x_h)
        x_LL = rearrange(x_LL, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return x_LL, x_h


class upFRG(nn.Module):
    def __init__(self, dim, n_l_blocks=1, n_h_blocks=1, expand=2):
        super().__init__()
        self.iwt = IWT()
        self.l_blk = nn.Sequential(*[ASSMBlock(dim) for _ in range(n_l_blocks)])
        self.h_out_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, 3, 1, 1),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.05),
        )
        self.combine = combine1(dim)
        self.mlp = FFN(in_channel=dim, FFN_channel=dim // 2, out_channel=dim)
        self.resblock = resblock(channel=dim)
        self.dr = nn.Dropout(0.05)

    def forward(self, x_l, x_h, ms):
        b, c, h, w = x_l.shape

        ms = self.mlp(ms)
        x_l = self.resblock(self.combine(x_l, ms))

        x_l = rearrange(x_l, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x_l = l_layer(x_l, [h, w], x_h)
        x_l = rearrange(x_l, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        x_h = self.h_out_conv(x_h)
        x_l = self.iwt(torch.cat([x_l, x_h], dim=1))
        return x_l


class UNet(nn.Module):
    def __init__(self, in_chn=3, wf=48, n_l_blocks=[1, 1, 1], n_h_blocks=[1, 1, 1], ffn_scale=2):
        super(UNet, self).__init__()
        self.ps_down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.ps_down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
        )
        self.ps_down3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8)
        )
        self.pan_down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.pan_down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
        )
        self.pan_down3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8),
        )

        self.pro_01 = nn.Sequential(
            nn.Conv2d(in_chn * 2, wf, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=wf, init=0.01),
            nn.Conv2d(wf, wf, kernel_size=3, stride=1, padding=1),
        )

        # encoder of UNet-64
        self.down_group1 = DownFRG(wf, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], expand=ffn_scale)
        self.down_group2 = DownFRG(wf, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], expand=ffn_scale)
        self.down_group3 = DownFRG(wf, n_l_blocks=n_l_blocks[2], n_h_blocks=n_h_blocks[2], expand=ffn_scale)

        # decoder of UNet-64
        self.up_group3 = upFRG(wf, n_l_blocks=n_l_blocks[2], n_h_blocks=n_h_blocks[2], expand=ffn_scale)
        self.up_group2 = upFRG(wf, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], expand=ffn_scale)
        self.up_group1 = upFRG(wf, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], expand=ffn_scale)

        self.last = nn.Conv2d(wf, in_chn, kernel_size=3, stride=1, padding=1, bias=True)
        self.resblock = resblock(channel=wf)

    def forward(self, x, pan, up_ms, ms):
        img = x

        img_down1 = self.pan_down1(pan)
        img_down2 = self.pan_down2(pan)
        img_down3 = self.pan_down3(pan)

        ms_down1 = self.ps_down1(up_ms)
        ms_down2 = self.ps_down2(up_ms)
        ms_down3 = self.ps_down3(up_ms)

        # shallow conv
        x1 = self.pro_01(img)

        # UNet-64
        # Down-path (Encoder)
        x_ld1, x_H1 = self.down_group1(x1, pan, ms_down1)
        x_ld2, x_H2 = self.down_group2(x_ld1, img_down1, ms)
        x_ld3, x_H3 = self.down_group3(x_ld2, img_down2, ms_down3)

        # Up-path (Decoder)
        x_lu3 = self.up_group3(x_ld3, x_H3, ms_down3) + x_ld2
        x_lu2 = self.up_group2(x_lu3, x_H2, ms) + x_ld1
        x_lu1 = self.up_group1(x_lu2, x_H1, ms_down1)

        # Reconstruct
        out_1 = self.resblock(x_lu1) + x1

        return out_1


class raise_channel(nn.Module):
    def __init__(self, in_channel, target_channel):
        super(raise_channel, self).__init__()
        self.raise_conv = nn.Sequential(
            nn.Conv2d(in_channel, target_channel, 5, 1, 2, bias=True),
            nn.PReLU(num_parameters=target_channel, init=0.01),
            nn.Conv2d(target_channel, target_channel, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        x = self.raise_conv(x)
        return x


class reduce_channel(nn.Module):
    def __init__(self, ms_target_channel, L_up_channel):
        super(reduce_channel, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(ms_target_channel, ms_target_channel, 3, 1, 1, bias=True),
            nn.PReLU(num_parameters=ms_target_channel, init=0.01),
            nn.Conv2d(ms_target_channel, L_up_channel, 3, 1, 1, bias=True),
            nn.Conv2d(L_up_channel, L_up_channel, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.reduce_conv(x)


class Net(nn.Module):
    def __init__(self, num_channels=8, pan_channel=1, pan_target_channel=32, ms_target_channel=32,
                 wf=32, n_l_blocks=[1, 1, 1], n_h_blocks=[1, 1, 1], ffn_scale=2.0):
        super().__init__()
        self.pan_channel = pan_channel
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 16, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
        )
        self.pan_raise_channel = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
        self.lms_raise_channel = raise_channel(in_channel=num_channels, target_channel=ms_target_channel)
        self.ms_raise_channel = raise_channel(in_channel=num_channels, target_channel=ms_target_channel)
        self.reduce_channel = reduce_channel(ms_target_channel=ms_target_channel, L_up_channel=num_channels)

        self.restoration_network = UNet(in_chn=wf, wf=wf, n_l_blocks=n_l_blocks, n_h_blocks=n_h_blocks, ffn_scale=ffn_scale)
        self.act_1 = nn.PReLU(num_parameters=num_channels, init=0.01)
        self.act_2 = nn.PReLU(num_parameters=num_channels, init=0.01)

    def encode_and_decode(self, pan, lms_2, ms):
        x = torch.cat([pan, lms_2], dim=1)
        restoration = self.restoration_network(x, pan, lms_2, self.ms_raise_channel(ms))

        return restoration

    def forward(self, ms, lms, pan):
        pan = self.pan_raise_channel(pan)
        lms_1 = self.act_1(self.upsample(ms) + lms)
        lms_2 = self.lms_raise_channel(lms_1)

        restoration = self.encode_and_decode(pan, lms_2, ms)

        back = self.reduce_channel(restoration)
        restoration = self.act_2(back + lms_1)

        return restoration


def prepare_input(resolution):
    device = torch.device('cuda:0')
    return {
        'ms': torch.randn(1, 8, 64, 64, device=device),
        'lms': torch.randn(1, 8, 256, 256, device=device),
        'pan': torch.randn(1, 1, 256, 256, device=device),
    }


def measure_model(model: torch.nn.Module, device: torch.device = None):
    if device is None:
        device = next(model.parameters()).device

    print(f"{'=' * 60}")
    print(f"Model Analysis on {device}")
    print(f"{'=' * 60}")

    flops_str, params_str = get_model_complexity_info(
        model,
        input_res=(1,),
        input_constructor=prepare_input,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )
    print(f"{'Number of parameters':<30} {params_str}")

    dummy_inputs = prepare_input(None)
    flops = FlopCountAnalysis(model, (dummy_inputs['ms'], dummy_inputs['lms'], dummy_inputs['pan']))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    print(f"FLOPs: {flops.total() / 1e9:.2f}G")

    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output = model(**dummy_inputs)
        torch.cuda.synchronize(device)
        end = time.perf_counter()

        memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Memory used: {memory:.2f} MB")

    with torch.no_grad():
        for _ in range(10):  # warm up
            model(**dummy_inputs)
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(100):
            model(**dummy_inputs)
        torch.cuda.synchronize(device)
        end = time.perf_counter()

        avg_time = (end - start) * 10
        print(f"Average inference time: {avg_time:.2f} ms")


if __name__ == '__main__':
    import time
    import torch
    from ptflops import get_model_complexity_info
    from fvcore.nn import FlopCountAnalysis

    model = Net().to('cuda:0')
    model.eval()

    measure_model(model)

    # CANNet WFANet statistics
    # from thop import profile
    # flopsTP, _ = profile(model, inputs=(ms, lms, pan))
    # print("FLOPs(thop): ", flopsTP, "=", f"{flopsTP / 1e9}G")
