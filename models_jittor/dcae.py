"""
DCAE model — full Jittor rewrite.

All neural-network layers, helper functions, and the main DCAE class
originally defined in models/dcae.py, ported from PyTorch to Jittor.
"""

import jittor as jt
import jittor.nn as nn
from jittor import init
import numpy as np
import math

from .entropy_models import EntropyBottleneck, GaussianConditional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

# ---------------------------------------------------------------------------
# Weight-init helpers (replace timm trunc_normal_)
# ---------------------------------------------------------------------------

def trunc_normal_(var, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal initialization (in-place on a Jittor Var)."""
    data = np.random.normal(mean, std, var.shape).astype(np.float32)
    lo, hi = mean + a * std, mean + b * std
    mask = (data < lo) | (data > hi)
    while mask.any():
        resample = np.random.normal(mean, std, mask.sum()).astype(np.float32)
        data[mask] = resample
        mask = (data < lo) | (data > hi)
    var.assign(jt.array(data))
    return var


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularisation."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if not self.is_training() or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (jt.rand(shape) < keep).float32()
        return x * mask / keep


class Identity(nn.Module):
    def execute(self, x):
        return x

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv(in_channels, out_channels, kernel_size=3, stride=2):
    """AvgPool + 1x1 conv to replace strided conv (avoids Jittor im2col OOM)."""
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
    )


class DeconvBlock(nn.Module):
    """1x1 conv + Upsample + crop to match ConvTranspose output size.

    PyTorch ConvTranspose(k,s,p,op): out = (H-1)*s + k - 2*p + op
    For k=3,s=2,p=1,op=1: out = (H-1)*2 + 3 - 2 + 1 = 2H
    Upsample(2) gives 2H, so no crop needed for k=3.
    For k=5,s=2,p=2,op=1 (original model uses k=5): out = (H-1)*2 + 5 - 4 + 1 = 2H
    So both k=3 and k=5 produce exactly 2H — no crop needed!
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')

    def execute(self, x):
        return self.upsample(self.conv(x))


def deconv(in_channels, out_channels, kernel_size=3, stride=2):
    """ConvTranspose replacement for Jittor compatibility."""
    return DeconvBlock(in_channels, out_channels, stride)


def get_scale_table(min_val=SCALES_MIN, max_val=SCALES_MAX, levels=SCALES_LEVELS):
    return jt.exp(jt.linspace(math.log(min_val), math.log(max_val), levels))


def ste_round(x):
    """Straight-Through Estimator rounding."""
    return jt.round(x) - x.detach() + x

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else Identity()

    def execute(self, x):
        identity = self.skip(x)
        out = self.conv3(self.relu2(self.conv2(self.relu1(self.conv1(x)))))
        return out + identity


class ResidualBottleneckBlockWithStride(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv(in_ch, out_ch, kernel_size=3, stride=2)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def execute(self, x):
        return self.res3(self.res2(self.res1(self.conv(x))))


class ResidualBottleneckBlockWithUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=3, stride=2)

    def execute(self, x):
        return self.conv(self.res3(self.res2(self.res1(x))))

# ---------------------------------------------------------------------------
# Swin-Transformer blocks
# ---------------------------------------------------------------------------

class WMSA(nn.Module):
    """Window Multi-head Self-Attention (W-MSA / SW-MSA)."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, wmsa_type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = wmsa_type

        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        rp_shape = (self.n_heads, 2 * window_size - 1, 2 * window_size - 1)
        self.relative_position_params = nn.Parameter(jt.zeros(rp_shape))
        trunc_normal_(self.relative_position_params, std=0.02)
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    # ---- mask generation for SW-MSA ----
    def generate_mask(self, h, w, p, shift):
        if self.type == 'W':
            return None
        attn_mask = np.zeros((h, w, p, p, p, p), dtype=bool)
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        # rearrange 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)'
        mask = attn_mask.reshape(h * w, p * p, p * p)
        return jt.array(mask).unsqueeze(0).unsqueeze(0)  # (1,1,hw,p²,p²)

    # ---- relative position embedding ----
    def relative_embedding(self):
        ws = self.window_size
        cord = np.array([[i, j] for i in range(ws) for j in range(ws)])
        relation = cord[:, None, :] - cord[None, :, :] + ws - 1
        r0 = jt.array(relation[:, :, 0].astype(np.int32))
        r1 = jt.array(relation[:, :, 1].astype(np.int32))
        # params: (n_heads, 2ws-1, 2ws-1); use advanced indexing like PyTorch
        return self.relative_position_params[:, r0, r1]

    # ---- forward ----
    def execute(self, x):
        B, H, W, C = x.shape
        ws = self.window_size

        # cyclic shift for SW-MSA
        if self.type != 'W':
            x = jt.roll(x, shifts=(-(ws // 2), -(ws // 2)), dims=(1, 2))

        # b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c
        x = x.reshape(B, H // ws, ws, W // ws, ws, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        h_windows = x.shape[1]
        w_windows = x.shape[2]
        x = x.reshape(B, h_windows * w_windows, ws * ws, C)

        # QKV
        qkv = self.embedding_layer(x)  # (B, nw, np, 3*dim)
        qkv = qkv.reshape(B, h_windows * w_windows, ws * ws, 3, self.n_heads, self.head_dim)
        qkv = qkv.transpose(3, 0, 1, 2, 4, 5)  # (3, B, nw, np, n_heads, hd)
        qkv = qkv.reshape(3, B, h_windows * w_windows, ws * ws, self.n_heads, self.head_dim)
        q = qkv[0]  # (B, nw, np, n_heads, hd)
        k = qkv[1]
        v = qkv[2]
        # -> (B, nw, n_heads, np, hd) to match PyTorch einsum behavior
        q = q.transpose(0, 1, 3, 2, 4)
        k = k.transpose(0, 1, 3, 2, 4)
        v = v.transpose(0, 1, 3, 2, 4)

        # attention: einsum 'hbwpc' means h=n_heads, b=batch, w=nw, p=np, c=hd
        # with q shape (B, nw, n_heads, np, hd), einsum interprets as (b,w,h,p,c)
        # output will be ordered by first appearance: (B, nw, n_heads, np, np)
        sim = jt.linalg.einsum('bwhpc,bwhqc->bwhpq', q, k) * self.scale
        # relative_embedding: (n_heads, np, np) -> (1, 1, n_heads, np, np) for broadcasting
        sim = sim + self.relative_embedding().unsqueeze(0).unsqueeze(0)

        if self.type != 'W':
            # Create shifted window attention bias
            np_size = ws * ws
            shift = ws // 2

            # Create row/col indices for each patch position within a single window
            pos = jt.arange(np_size)
            r_i = pos // ws  # row index for i-th patch
            c_i = pos % ws   # col index for i-th patch
            r_j = (pos + shift) // ws  # shifted row for j-th patch
            c_j = (pos + shift) % ws   # shifted col for j-th patch

            diff_row = jt.abs(r_i.unsqueeze(1) - r_j.unsqueeze(0))
            diff_col = jt.abs(c_i.unsqueeze(1) - c_j.unsqueeze(0))

            valid = (diff_row < shift) & (diff_col < shift)
            attn_bias = jt.where(valid,
                                jt.zeros_like(diff_row).float(),
                                jt.full_like(diff_row, -1e4).float())

            # attn_bias: (np, np) -> reshape for broadcasting with sim (B, nw, nh, np, np)
            # Reshape to (1, 1, 1, np, np) and let broadcasting handle the rest
            attn_bias = attn_bias.reshape((1, 1, 1, np_size, np_size))
            sim = sim + attn_bias

        probs = nn.softmax(sim, dim=-1)
        output = jt.linalg.einsum('bwhij,bwhjc->bwhic', probs, v)

        # b nw n_heads np hd -> b nw np (n_heads hd)
        output = output.transpose(0, 1, 3, 2, 4).reshape(B, h_windows * w_windows, ws * ws, C)
        output = self.linear(output)

        # b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c
        output = output.reshape(B, h_windows, w_windows, ws, ws, C)
        output = output.transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)

        if self.type != 'W':
            output = jt.roll(output, shifts=(ws // 2, ws // 2), dims=(1, 2))
        return output


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True, groups=dim)

    def execute(self, x):
        # x: (B, H, W, C)
        x = x.transpose(0, 3, 1, 2)  # -> (B, C, H, W)
        x = self.dwconv(x)
        return x.transpose(0, 2, 3, 1)  # -> (B, H, W, C)


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = (hidden_features or in_features) // 2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self._hidden_features = hidden_features

    def execute(self, x):
        x = self.fc1(x)
        xv = x[..., :self._hidden_features]
        vg = x[..., self._hidden_features:]
        x = self.act(self.dwconv(xv)) * vg
        return self.fc2(x)


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        if trainable:
            self.scale = nn.Parameter(jt.full((dim,), float(init_value)))
        else:
            self.scale = jt.full((dim,), float(init_value))

    def execute(self, x):
        return x * self.scale


class ResScaleConvolutionGateBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size,
                 drop_path, block_type='W', input_resolution=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, block_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvolutionalGLU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim)
        self.res_scale_2 = Scale(input_dim)

    def execute(self, x):
        x = self.res_scale_1(x) + self.drop_path(self.msa(self.ln1(x)))
        x = self.res_scale_2(x) + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SwinBlockWithConvMulti(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size,
                 drop_path, block=ResScaleConvolutionGateBlock, block_num=2, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.block_num = block_num
        for i in range(block_num):
            ty = 'W' if i % 2 == 0 else 'SW'
            self.layers.append(block(input_dim, input_dim, head_dim, window_size, drop_path, block_type=ty))
        self.conv = conv(input_dim, output_dim, 3, 1)
        self.window_size = window_size

    def execute(self, x):
        # x: (B, C, H, W)
        H, W = x.shape[2], x.shape[3]
        ws = self.window_size
        # pad if spatial dims <= window_size
        if H <= ws or W <= ws:
            pad_row = (ws - H) // 2
            pad_col = (ws - W) // 2
            x = nn.pad(x, (pad_col, pad_col + 1, pad_row, pad_row + 1))

        # BCHW -> BHWC
        trans_x = x.transpose(0, 2, 3, 1)
        for i in range(self.block_num):
            trans_x = self.layers[i](trans_x)
        # BHWC -> BCHW
        trans_x = trans_x.transpose(0, 3, 1, 2)
        trans_x = self.conv(trans_x)
        # NOTE: resize un-pad is dead code in the original (resize=False always)
        return trans_x + x

# ---------------------------------------------------------------------------
# Attention / aggregation blocks for dictionary cross-attention
# ---------------------------------------------------------------------------

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # Use 1x1 conv for Jittor compatibility (avoids im2col OOM on large feature maps)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        avg_out = jt.mean(x, dim=1, keepdims=True)
        max_out = x.max(dim=1, keepdims=True)
        x = jt.concat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class ConvWithDW(nn.Module):
    def __init__(self, input_dim=320, output_dim=320):
        super().__init__()
        self.in_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(output_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=output_dim, bias=True)
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(output_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)

    def execute(self, x):
        return self.out_trans(self.act2(self.dw_conv(self.act1(self.in_trans(x)))))


class DenseBlock(nn.Module):
    def __init__(self, dim=320):
        super().__init__()
        self.layer_num = 3
        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.GELU(), ConvWithDW(dim, dim))
            for _ in range(self.layer_num)
        ])
        self.proj = nn.Conv2d(dim * (self.layer_num + 1), dim, kernel_size=1, padding=0, stride=1, bias=True)

    def execute(self, x):
        outputs = [x]
        for i in range(self.layer_num):
            outputs.append(self.conv_layers[i](outputs[-1]))
        return self.proj(jt.concat(outputs, dim=1))


class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)

    def execute(self, x):
        # x: (B, H, W, C) -> BCHW
        x = x.transpose(0, 3, 1, 2)
        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)
        return x.transpose(0, 2, 3, 1)  # -> BHWC


class MutiScaleDictionaryCrossAttentionGLU(nn.Module):
    """Core DCAE innovation — dictionary-based cross-attention."""

    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True):
        super().__init__()
        dict_dim = 32 * head_num
        self.head_num = head_num

        self.scale = jt.ones((head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)

        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)

        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)

        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)
        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))

        self.res_scale_1 = Scale(dict_dim)
        self.res_scale_2 = Scale(dict_dim)
        self.res_scale_3 = Scale(dict_dim)

    def execute(self, x, dt):
        B, C, H, W = x.shape
        # BCHW -> BHWC
        x = x.transpose(0, 2, 3, 1)
        x = self.x_trans(x)

        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)

        shortcut = x
        x = self.lnx(x)
        x = self.q_trans(x)
        # BHWC -> BCHW
        x = x.transpose(0, 3, 1, 2)

        # b (e c) h w -> b e (h w) c
        e = self.head_num
        c = x.shape[1] // e
        q = x.reshape(B, e, c, H, W).transpose(0, 1, 3, 4, 2).reshape(B, e, H * W, c)

        dt = self.dict_ln(dt)
        k = self.k(dt)
        # b n (e c) -> b e n c
        n = dt.shape[1]
        k = k.reshape(B, n, e, c).transpose(0, 2, 1, 3)
        dt_r = dt.reshape(B, n, e, c).transpose(0, 2, 1, 3)

        sim = jt.linalg.einsum('benc,bedc->bend', q, k)
        sim = sim * self.scale
        probs = nn.softmax(sim, dim=-1)
        output = jt.linalg.einsum('bend,bedc->benc', probs, dt_r)

        # b e (h w) c -> b h w (e c)
        output = output.reshape(B, e, H, W, c).transpose(0, 2, 3, 1, 4).reshape(B, H, W, e * c)

        output = self.linear(output) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)
        output = self.output_trans(output)
        # BHWC -> BCHW
        return output.transpose(0, 3, 1, 2)

# ---------------------------------------------------------------------------
# DCAE
# ---------------------------------------------------------------------------

class DCAE(nn.Module):
    """Dictionary-based Cross Attention Entropy model (Jittor compact version)."""

    def __init__(self, head_dim=None, drop_path_rate=0, N=96, M=160,
                 num_slices=5, max_support_slices=5, **kwargs):
        super().__init__()
        if head_dim is None:
            head_dim = [4, 8, 16, 16, 8, 4]  # scaled with N/M

        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.M = M

        feature_dim = [48, 72, 128]  # Jittor: reduced (was [96,144,256])
        block_num = [1, 1, 2]  # Jittor: reduced depth (was [1,2,12])

        dict_num = 32   # Jittor: reduced dictionary (was 128)
        dict_head_num = 8   # Jittor: reduced attention heads (was 20)
        dict_dim = 32 * dict_head_num
        self.dt = nn.Parameter(jt.randn((dict_num, dict_dim)))

        mlp_rate = 4
        qkv_bias = True
        self.dt_cross_attention = nn.ModuleList([
            MutiScaleDictionaryCrossAttentionGLU(
                input_dim=M * 2 + (M // num_slices) * i,
                output_dim=M, head_num=dict_head_num,
                mlp_rate=mlp_rate, qkv_bias=qkv_bias,
            ) for i in range(num_slices)
        ])

        # ---- encoder g_a ----
        basic_block = ResScaleConvolutionGateBlock
        swin_block = SwinBlockWithConvMulti

        self.m_down1 = [
            swin_block(feature_dim[0], feature_dim[0], head_dim[0], self.window_size, 0,
                        basic_block, block_num=block_num[0]),
            ResidualBottleneckBlockWithStride(feature_dim[0], feature_dim[1]),
        ]
        self.m_down2 = [
            swin_block(feature_dim[1], feature_dim[1], head_dim[1], self.window_size, 0,
                        basic_block, block_num=block_num[1]),
            ResidualBottleneckBlockWithStride(feature_dim[1], feature_dim[2]),
        ]
        self.m_down3 = [
            swin_block(feature_dim[2], feature_dim[2], head_dim[2], self.window_size, 0,
                        basic_block, block_num=block_num[2]),
            conv(feature_dim[2], M, kernel_size=3, stride=2),
        ]

        self.g_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(3, feature_dim[0]),
            *self.m_down1, *self.m_down2, *self.m_down3,
        )

        # ---- decoder g_s ----
        self.m_up1 = [
            swin_block(feature_dim[2], feature_dim[2], head_dim[3], self.window_size, 0,
                        basic_block, block_num=block_num[2]),
            ResidualBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1]),
        ]
        self.m_up2 = [
            swin_block(feature_dim[1], feature_dim[1], head_dim[4], self.window_size, 0,
                        basic_block, block_num=block_num[1]),
            ResidualBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0]),
        ]
        self.m_up3 = [
            swin_block(feature_dim[0], feature_dim[0], head_dim[5], self.window_size, 0,
                        basic_block, block_num=block_num[0]),
            ResidualBottleneckBlockWithUpsample(feature_dim[0], 3),
        ]

        self.g_s = nn.Sequential(
            deconv(M, feature_dim[2], kernel_size=3, stride=2),
            *self.m_up1, *self.m_up2, *self.m_up3,
        )

        # ---- hyperprior encoder h_a ----
        self.h_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(M, N),
            swin_block(N, N, 32, 4, 0, basic_block, block_num=1),
            conv(N, 192, kernel_size=3, stride=2),
        )

        # ---- hyperprior decoders h_z_s1 / h_z_s2 ----
        self.h_z_s1 = nn.Sequential(
            deconv(192, N, kernel_size=3, stride=2),
            swin_block(N, N, 32, 4, 0, basic_block, block_num=1),
            ResidualBottleneckBlockWithUpsample(N, M),
        )
        self.h_z_s2 = nn.Sequential(
            deconv(192, N, kernel_size=3, stride=2),
            swin_block(N, N, 32, 4, 0, basic_block, block_num=1),
            ResidualBottleneckBlockWithUpsample(N, M),
        )

        # ---- per-slice context transforms (all 1x1 conv for Jittor compatibility) ----
        self.cc_mean_transforms = nn.ModuleList([
            nn.Sequential(
                conv(M * 2 + (M // num_slices) * min(i, 5) + M, 224, stride=1, kernel_size=1),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=1),
                nn.GELU(),
                conv(128, M // num_slices, stride=1, kernel_size=1),
            ) for i in range(num_slices)
        ])
        self.cc_scale_transforms = nn.ModuleList([
            nn.Sequential(
                conv(M * 2 + (M // num_slices) * min(i, 5) + M, 224, stride=1, kernel_size=1),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=1),
                nn.GELU(),
                conv(128, M // num_slices, stride=1, kernel_size=1),
            ) for i in range(num_slices)
        ])
        self.lrp_transforms = nn.ModuleList([
            nn.Sequential(
                conv(M * 2 + (M // num_slices) * min(i + 1, 6) + M, 224, stride=1, kernel_size=1),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=1),
                nn.GELU(),
                conv(128, M // num_slices, stride=1, kernel_size=1),
            ) for i in range(num_slices)
        ])

        # ---- entropy models ----
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    # ------------------------------------------------------------------
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    # ------------------------------------------------------------------
    def aux_loss(self):
        return self.entropy_bottleneck.loss()

    # ------------------------------------------------------------------
    def execute(self, x):
        """Forward pass (differentiable, for training)."""
        b = x.shape[0]
        dt = self.dt.unsqueeze(0).expand(b, -1, -1)
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        # split y into slices along channel dim
        slice_ch = self.M // self.num_slices
        y_slices = [y[:, i * slice_ch:(i + 1) * slice_ch, :, :] for i in range(self.num_slices)]
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices]
            query = jt.concat([latent_scales, latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = jt.concat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)

            sc = self.cc_scale_transforms[slice_index](support)
            sc = sc[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(sc)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, sc, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = jt.concat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * jt.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = jt.concat(y_hat_slices, dim=1)
        means = jt.concat(mu_list, dim=1)
        scales = jt.concat(scale_list, dim=1)
        y_likelihoods = jt.concat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y},
        }

    # ------------------------------------------------------------------
    def compress(self, x):
        b = x.shape[0]
        dt = self.dt.unsqueeze(0).expand(b, -1, -1)
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        slice_ch = self.M // self.num_slices
        y_slices = [y[:, i * slice_ch:(i + 1) * slice_ch, :, :] for i in range(self.num_slices)]
        y_hat_slices, y_scales, y_means = [], [], []

        cdf = self.gaussian_conditional._quantized_cdf.numpy().tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.numpy().flatten().astype(np.int32).tolist()
        offsets = self.gaussian_conditional._offset.numpy().flatten().astype(np.int32).tolist()

        encoder = BufferedRansEncoder()
        symbols_list, indexes_list, y_strings = [], [], []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices]
            query = jt.concat([latent_scales, latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = jt.concat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            sc = self.cc_scale_transforms[slice_index](support)
            sc = sc[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(sc)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.numpy().reshape(-1).tolist())
            indexes_list.extend(index.numpy().reshape(-1).tolist())

            lrp_support = jt.concat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * jt.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(sc)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_strings.append(encoder.flush())

        return {"strings": [y_strings, z_strings], "shape": z.shape[-2:]}

    # ------------------------------------------------------------------
    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        b = z_hat.shape[0]
        dt = self.dt.unsqueeze(0).expand(b, -1, -1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional._quantized_cdf.numpy().tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.numpy().flatten().astype(np.int32).tolist()
        offsets = self.gaussian_conditional._offset.numpy().flatten().astype(np.int32).tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        slice_ch = self.M // self.num_slices
        for slice_index in range(self.num_slices):
            support_slices = y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices]
            query = jt.concat([latent_scales, latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = jt.concat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            sc = self.cc_scale_transforms[slice_index](support)
            sc = sc[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(sc)
            rv = decoder.decode_stream(index.numpy().reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = jt.array(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = jt.concat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * jt.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = jt.concat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp(0, 1)
        return {"x_hat": x_hat}

    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict=True):
        """Override to resize dynamic buffers before loading."""
        # Resize entropy_bottleneck quantiles if needed
        eb_prefix = "entropy_bottleneck."
        q_key = eb_prefix + "_quantiles"
        if q_key in state_dict:
            saved_q = state_dict[q_key]
            if hasattr(saved_q, "shape"):
                saved_shape = saved_q.shape
            else:
                saved_shape = np.array(saved_q).shape
            cur_shape = self.entropy_bottleneck._quantiles.shape
            if saved_shape != cur_shape:
                self.entropy_bottleneck._quantiles = jt.array(np.zeros(saved_shape, dtype=np.float32))

        # GaussianConditional buffers
        gc_prefix = "gaussian_conditional."
        for buf in ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"]:
            key = gc_prefix + buf
            if key in state_dict:
                val = state_dict[key]
                if hasattr(val, "numpy"):
                    val = val.numpy()
                setattr(self.gaussian_conditional, buf, jt.array(np.array(val)))

        # Jittor's Module.load_state_dict doesn't accept strict parameter
        super().load_state_dict(state_dict)
