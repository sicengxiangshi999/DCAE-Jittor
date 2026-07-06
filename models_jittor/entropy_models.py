"""
Entropy models for Jittor: GaussianConditional and EntropyBottleneck.
Replacements for compressai.entropy_models used by the DCAE model.
Uses compressai.ans (BufferedRansEncoder / RansDecoder) for actual rANS coding
since those are C++ bindings operating on Python lists, framework-agnostic.
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
import math

# Lazy imports – avoid triggering torch/CUDA conflicts when running Jittor-only
_BufferedRansEncoder = _RansDecoder = None


def _get_encoder():
    global _BufferedRansEncoder
    if _BufferedRansEncoder is None:
        from compressai.ans import BufferedRansEncoder
        _BufferedRansEncoder = BufferedRansEncoder
    return _BufferedRansEncoder


def _get_decoder():
    global _RansDecoder
    if _RansDecoder is None:
        from compressai.ans import RansDecoder
        _RansDecoder = RansDecoder
    return _RansDecoder


class GaussianConditional(nn.Module):
    """Gaussian conditional entropy model.

    Computes likelihoods under a Gaussian distribution parameterized by
    per-element scale and mean.  Maintains a discretized CDF table for
    rANS entropy coding during compress / decompress.
    """

    def __init__(self, scale_table=None):
        super().__init__()
        if scale_table is None:
            scale_table = self._default_scale_table()
        self.scale_table = jt.array(scale_table)
        # Buffers – populated by update()
        self._offset = jt.array(np.zeros(1, dtype=np.int32))
        self._quantized_cdf = jt.array(np.zeros((1, 1), dtype=np.int32))
        self._cdf_length = jt.array(np.zeros(1, dtype=np.int32))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_scale_table(min_val=0.11, max_val=256.0, levels=64):
        return np.exp(np.linspace(math.log(min_val), math.log(max_val), levels)).astype(np.float32)

    @staticmethod
    def _standardized_cumulative(inputs):
        """Phi(x) = 0.5 * erfc(-x / sqrt(2))"""
        # Use jt.erf (element-wise) instead of jt.math.erf
        const = -(2.0 ** -0.5)
        return 0.5 * (1.0 - jt.erf(const * inputs))

    # ------------------------------------------------------------------
    # Forward (training likelihood)
    # ------------------------------------------------------------------
    def execute(self, inputs, scales, means=None):
        """Return (quantized_inputs, likelihoods)."""
        quantized = self.quantize(inputs, "noise" if self.is_training() else "dequantize", means)
        likelihood = self._likelihood(inputs, scales, means)
        return quantized, likelihood

    # ------------------------------------------------------------------
    # Quantize / dequantize
    # ------------------------------------------------------------------
    def quantize(self, inputs, mode, means=None):
        if mode == "noise":
            noise = jt.rand(inputs.shape) - 0.5
            return inputs + noise
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        if mode == "symbols":
            values = jt.round(values)
            return values
        if mode == "dequantize":
            values = jt.round(values)
            if means is not None:
                values = values + means
            return values
        raise ValueError(f"Unknown quantize mode: {mode}")

    def dequantize(self, inputs, means=None):
        if means is not None:
            return inputs + means
        return inputs

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build_indexes(self, scales):
        scales = jt.maximum(scales, jt.array([0.11]))
        table = self.scale_table.reshape(1, -1)
        diff = scales.reshape((-1, 1)) - table
        # First index where scale_table >= scale
        mask = (diff >= 0).float32()
        indexes = mask.sum(dim=-1).int32() - 1
        indexes = jt.maximum(indexes, jt.zeros_like(indexes))
        indexes = jt.minimum(indexes, jt.array([len(self.scale_table) - 1], dtype=jt.int32))
        return indexes

    # ------------------------------------------------------------------
    # Update CDF table
    # ------------------------------------------------------------------
    def update_scale_table(self, scale_table, force=False):
        self.scale_table = jt.array(np.array(scale_table, dtype=np.float32))
        self.update()
        return True

    def update(self):
        # Use numpy to compute multiplier (non-differentiable constant)
        from scipy.special import erf
        const = -(2.0 ** -0.5)
        phi_neg_half = 0.5 * (1.0 - erf(const / np.sqrt(2)))
        multiplier = -phi_neg_half
        pmf_center = math.ceil(256.0 / 2.0) + abs(math.ceil(multiplier))
        pmf_length = 2 * pmf_center + 1
        min_allowed = 1
        max_length = pmf_length + min_allowed

        samples = np.abs(np.arange(-pmf_center, pmf_center + 1, dtype=np.float32))
        scales_np = self.scale_table.numpy().reshape(-1, 1)

        upper = 0.5 * (1.0 - erf((0.5 - samples.reshape(1, -1)) / scales_np))
        lower = 0.5 * (1.0 - erf((-0.5 - samples.reshape(1, -1)) / scales_np))
        pmf = upper - lower  # (num_scales, pmf_length)

        num_scales = len(scales_np.flatten())

        cdf = np.zeros((num_scales, pmf_length + 1), dtype=np.int32)
        tail_mass = np.zeros(num_scales, dtype=np.float64)
        cdf_length = np.zeros(num_scales, dtype=np.int32)

        for s in range(num_scales):
            pmf_s = pmf[s]
            pmf_s = np.maximum(pmf_s, 1e-10)
            pmf_s = pmf_s / pmf_s.sum()
            cdf_s = np.zeros(pmf_length + 1, dtype=np.int32)
            cdf_s[1:] = np.round(np.cumsum(pmf_s) * 65536).astype(np.int32)
            cdf_s[-1] = 65536
            overflow = np.where(cdf_s[1:] == cdf_s[:-1])[0]
            if len(overflow) > 0:
                cdf_length[s] = int(overflow[0]) + 2
            else:
                cdf_length[s] = pmf_length + 1
            tail_mass[s] = float(pmf_s[cdf_length[s] - 1:].sum())
            cdf[s] = cdf_s

        self._quantized_cdf = jt.array(cdf)
        self._cdf_length = jt.array(cdf_length)
        self._offset = jt.array(np.full(num_scales, -pmf_center, dtype=np.int32))

    # ------------------------------------------------------------------
    # Likelihood (differentiable, used in training)
    # ------------------------------------------------------------------
    def _likelihood(self, inputs, scales, means=None):
        values = inputs - means if means is not None else inputs
        scales = jt.maximum(scales, jt.array([0.11]))
        values = jt.abs(values)
        upper = self._standardized_cumulative((0.5 - values) / scales)
        lower = self._standardized_cumulative((-0.5 - values) / scales)
        return jt.maximum(upper - lower, jt.array([1e-9]))

    # ------------------------------------------------------------------
    # State dict helpers
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, *args, **kwargs):
        if "scale_table" in state_dict:
            st = state_dict["scale_table"]
            if hasattr(st, "numpy"):
                st = st.numpy()
            self.scale_table = jt.array(np.array(st, dtype=np.float32))
            self.update()
        for attr in ["_offset", "_quantized_cdf", "_cdf_length"]:
            if attr in state_dict:
                val = state_dict[attr]
                if hasattr(val, "numpy"):
                    val = val.numpy()
                setattr(self, attr, jt.array(np.array(val)))


class EntropyBottleneck(nn.Module):
    """Non-parametric entropy model using learned quantiles.

    A simplified Jittor port of CompressAI's EntropyBottleneck.
    During training, quantization is simulated with uniform noise;
    likelihoods are computed from a piecewise-linear CDF whose knot
    positions (quantiles) are learned via an auxiliary loss.
    """

    def __init__(self, channels, init_scale=10, filters=(3, 3, 3)):
        super().__init__()
        self.channels = channels
        self.init_scale = init_scale
        self.filters = tuple(filters)
        self.num_quantiles = 2 * self.init_scale + 1

        # Piecewise-linear CDF defined by learned quantile positions.
        # For each channel: CDF values at integer offsets in [-init_scale, init_scale]
        self._quantiles = nn.Parameter(
            jt.array(
                np.linspace(-init_scale, init_scale, self.num_quantiles)
                .reshape(self.num_quantiles, 1)
                .repeat(channels, axis=1)
                .astype(np.float32)
            )
        )

        # Target cumulative probabilities for auxiliary loss
        self._target = jt.array(
            np.linspace(0.01, 0.99, self.num_quantiles).astype(np.float32)
        )

        # CDF tables for rANS coding (populated by update())
        self._cdf = None
        self._cdf_length = None
        self._offset = None
        self._pmf = None

    # ------------------------------------------------------------------
    def _get_medians(self):
        """Return channel-wise median for zero-centring before quantization.
        Shape: (1, C, 1, 1) — broadcasts with (B, C, H, W).
        """
        mid = self.num_quantiles // 2
        # _quantiles: (num_q, C) -> (1, C) -> (1, C, 1, 1)
        return self._quantiles[mid:mid + 1].reshape((1, self.channels, 1, 1))

    # ------------------------------------------------------------------
    def loss(self):
        """Auxiliary loss: aligns learned quantiles with data distribution.

        Uses batch statistics from the most recent forward pass to guide
        quantile learning, with minimal memory overhead.
        """
        if not hasattr(self, '_last_data') or self._last_data is None:
            return jt.array([0.0])

        z = self._last_data  # (B, C, H, W)
        B, C = z.shape[0], z.shape[1]

        # Compute per-channel statistics by reducing batch+spatial separately
        z_flat = z.reshape(B, C, -1)  # (B, C, N)
        data_mean = jt.mean(z_flat, dim=0).mean(dim=1)  # (C,)
        data_var = jt.mean((z_flat - data_mean.unsqueeze(0).unsqueeze(-1)) ** 2, dim=0).mean(dim=1)  # (C,)
        data_std = jt.sqrt(data_var + 1e-8)

        # Quantile statistics per channel
        q_mean = jt.mean(self._quantiles, dim=0)  # (C,)
        q_var = jt.mean((self._quantiles - q_mean.unsqueeze(0)) ** 2, dim=0)  # (C,)
        q_std = jt.sqrt(q_var + 1e-8)

        # Loss 1: align quantile mean/std with data distribution
        mean_align = jt.mean((q_mean - data_mean) ** 2)
        std_align = jt.mean((q_std - data_std) ** 2)

        # Loss 2: uniform quantile spacing (no clustering)
        diffs = self._quantiles[1:] - self._quantiles[:-1]  # (nq-1, C)
        spacing_penalty = jt.mean(diffs ** 2)

        return mean_align + std_align + spacing_penalty

    # ------------------------------------------------------------------
    def execute(self, x):
        """Return (quantized, likelihoods)."""
        # Always save data for aux loss (needed for both train metric and test metric)
        self._last_data = x.detach()

        medians = self._get_medians()  # (1, C, 1, 1)
        x_centered = x - medians

        if self.is_training():
            noise = jt.rand(x.shape) - 0.5
            quantized = x_centered + noise
        else:
            quantized = jt.round(x_centered)

        # Piecewise-linear likelihood: P(q) = CDF(q+0.5) - CDF(q-0.5)
        # Approximate using the learned quantile positions
        upper = self._piecewise_cdf(quantized + 0.5)
        lower = self._piecewise_cdf(quantized - 0.5)
        likelihood = jt.maximum(jt.abs(upper - lower), jt.array([1e-9]))

        return quantized + medians, likelihood

    def _piecewise_cdf(self, x):
        """Evaluate approximate CDF at *x* using the learned quantiles.

        For each channel, linearly interpolate between the sorted quantile
        knot positions to get the CDF value.
        """
        sorted_q = jt.sort(self._quantiles, dim=0)[0]  # (num_q, C)
        # Normalised position: (x - min_q) / (max_q - min_q)  →  [0, 1]
        q_min = sorted_q[0:1].reshape((1, self.channels, 1, 1))  # (1, C, 1, 1)
        q_max = sorted_q[-1:].reshape((1, self.channels, 1, 1))  # (1, C, 1, 1)
        span = jt.maximum(q_max - q_min, jt.array([1e-6]))
        t = (x - q_min) / span
        t = jt.maximum(jt.array([0.0]), jt.minimum(jt.array([1.0]), t))
        return t

    # ------------------------------------------------------------------
    # Compress / decompress
    # ------------------------------------------------------------------
    def compress(self, x):
        medians = self._get_medians()
        x_centered = x - medians
        values = jt.round(x_centered)
        if self._cdf is None:
            self.update()

        pmf = self._pmf
        cdf = self._cdf
        cdf_lengths = self._cdf_length
        offsets = self._offset

        values_np = values.int32().numpy()
        B, C, H, W = values_np.shape
        values_flat = values_np.transpose(1, 0, 2, 3).reshape(C, -1)

        encoder = _get_encoder()()
        strings = []
        for c in range(C):
            cdf_c = cdf[c].tolist()
            cdf_len = int(cdf_lengths[c])
            off = int(offsets[c])
            syms = (values_flat[c] - off).clip(0, cdf_len - 2).tolist()
            idx = [0] * len(syms)
            encoder.encode_with_indexes(syms, idx, [cdf_c], [cdf_len], [0])
            strings.append(encoder.flush())
        return strings

    def decompress(self, strings, shape):
        if isinstance(shape, (tuple, list)) and len(shape) == 2:
            H, W = shape
        else:
            H, W = shape, shape
        C = self.channels

        if self._cdf is None:
            self.update()

        cdf = self._cdf
        cdf_lengths = self._cdf_length
        offsets = self._offset
        medians = self._get_medians()

        values = np.zeros((C, H * W), dtype=np.int32)
        for c in range(C):
            decoder = _get_decoder()()
            decoder.set_stream(strings[c])
            cdf_c = cdf[c].tolist()
            cdf_len = int(cdf_lengths[c])
            off = int(offsets[c])
            idx = [0] * (H * W)
            rv = decoder.decode_stream(idx, [cdf_c], [cdf_len], [0])
            values[c] = np.array(rv, dtype=np.int32) + off

        values = jt.array(values.reshape(C, 1, H, W).transpose(1, 0, 2, 3).astype(np.float32))
        return values + medians

    # ------------------------------------------------------------------
    def update(self, force=False):
        """Build PMF / CDF tables from learned quantiles for rANS coding."""
        sorted_q = np.sort(self._quantiles.numpy(), axis=0)
        C = self.channels
        max_range = int(max(abs(sorted_q[0]).max(), abs(sorted_q[-1]).max())) + 2
        min_val = -max_range
        max_val = max_range
        num_bins = max_val - min_val + 1

        pmf = np.zeros((C, num_bins + 1), dtype=np.float32)
        for c in range(C):
            knots = sorted_q[:, c]
            for b in range(num_bins):
                x = min_val + b + 0.5
                pos = np.searchsorted(knots, x)
                if pos == 0:
                    pmf[c, b] = 1e-9
                elif pos >= len(knots):
                    pmf[c, b] = 1e-9
                else:
                    t = (x - knots[pos - 1]) / max(knots[pos] - knots[pos - 1], 1e-6)
                    pmf[c, b] = max(t / len(knots), 1e-9)
            pmf[c] /= pmf[c].sum()

        cdf = np.zeros((C, num_bins + 2), dtype=np.int32)
        cdf_lengths = np.zeros(C, dtype=np.int32)
        offsets = np.full(C, min_val, dtype=np.int32)

        for c in range(C):
            cdf[c, 1:num_bins + 1] = np.round(np.cumsum(pmf[c]) * 65536).astype(np.int32)
            cdf[c, -1] = 65536
            nz = np.where(cdf[c, 1:] > cdf[c, :-1])[0]
            cdf_lengths[c] = int(nz[-1]) + 2 if len(nz) > 0 else 2

        self._pmf = pmf
        self._cdf = cdf.tolist()
        self._cdf_length = jt.array(cdf_lengths)
        self._offset = jt.array(offsets)
        return True

    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, *args, **kwargs):
        for attr in ["_quantiles", "_target", "_cdf_length", "_offset"]:
            key = attr
            if key in state_dict:
                val = state_dict[key]
                if hasattr(val, "numpy"):
                    val = val.numpy()
                setattr(self, attr, jt.array(np.array(val, dtype=np.float32 if attr in ("_quantiles", "_target") else np.int32)))
