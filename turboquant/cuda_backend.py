"""
CUDA backend for TurboQuant.

Wraps the QJL CUDA kernels (from github.com/amirzandieh/QJL) with
a Python-friendly interface that integrates into TurboQuant's two-stage
quantization pipeline.

Falls back to pure PyTorch if kernels are not available.
"""

import math
import os
import glob
import torch
import torch.nn as nn
from typing import Optional, Tuple
from scipy.linalg import hadamard

# Try importing CUDA kernels - they may not be built yet
_CUDA_AVAILABLE = False
try:
    _kernel_dir = os.path.dirname(os.path.abspath(__file__))
    import importlib.util
    for mod_name in ['cuda_qjl_quant', 'cuda_qjl_score', 'cuda_qjl_gqa_score', 'quantization']:
        # Search for any .so file that starts with the module name. (adds support for non-x86)
        so_files = glob.glob(os.path.join(_kernel_dir, f'{mod_name}.*.so'))
        so_path = so_files[0] if so_files else ""
        if os.path.exists(so_path):
            spec = importlib.util.spec_from_file_location(mod_name, so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            globals()[mod_name] = mod
    _CUDA_AVAILABLE = all(k in globals() for k in ['cuda_qjl_quant', 'cuda_qjl_score', 'cuda_qjl_gqa_score', 'quantization'])
except Exception as e:
    _CUDA_AVAILABLE = False
    _CUDA_LOAD_ERROR = str(e)


def is_cuda_available():
    return _CUDA_AVAILABLE


# ─── QJL CUDA kernel wrappers ───────────────────────────────────────

def qjl_quant(key_states, outlier_indices, rand_prj, outlier_sketch_dim):
    """Fused QJL quantization via CUDA kernel."""
    key_dtype = key_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): 'qjl_quant_half_half',
        (torch.half, torch.float): 'qjl_quant_half_float',
        (torch.float, torch.float): 'qjl_quant_float_float',
        (torch.bfloat16, torch.bfloat16): 'qjl_quant_bf16_bf16',
        (torch.bfloat16, torch.float): 'qjl_quant_bf16_float',
    }
    fn_name = dispatch.get((key_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: key={key_dtype}, proj={rand_dtype}")
    return getattr(cuda_qjl_quant, fn_name)(key_states, outlier_indices, rand_prj, outlier_sketch_dim)


def qjl_score(key_quant, key_outlier_quant, key_norm, key_outlier_norm,
              outlier_indices, query_sketch, query_states, rand_prj):
    """Fused QJL score computation via CUDA kernel."""
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): 'qjl_score_cuda_half_half',
        (torch.half, torch.float): 'qjl_score_cuda_half_float',
        (torch.float, torch.float): 'qjl_score_cuda_float_float',
        (torch.bfloat16, torch.bfloat16): 'qjl_score_cuda_bf16_bf16',
        (torch.bfloat16, torch.float): 'qjl_score_cuda_bf16_float',
    }
    fn_name = dispatch.get((query_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: query={query_dtype}, proj={rand_dtype}")
    return getattr(cuda_qjl_score, fn_name)(
        key_quant, key_outlier_quant, key_norm, key_outlier_norm,
        outlier_indices, query_sketch, query_states, rand_prj)


def qjl_gqa_score(key_quant, key_outlier_quant, key_norm, key_outlier_norm,
                   outlier_indices, query_sketch, query_states, rand_prj):
    """Fused QJL GQA score computation via CUDA kernel."""
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): 'qjl_gqa_score_cuda_half_half',
        (torch.half, torch.float): 'qjl_gqa_score_cuda_half_float',
        (torch.float, torch.float): 'qjl_gqa_score_cuda_float_float',
        (torch.bfloat16, torch.bfloat16): 'qjl_gqa_score_cuda_bf16_bf16',
        (torch.bfloat16, torch.float): 'qjl_gqa_score_cuda_bf16_float',
    }
    fn_name = dispatch.get((query_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: query={query_dtype}, proj={rand_dtype}")
    return getattr(cuda_qjl_gqa_score, fn_name)(
        key_quant, key_outlier_quant, key_norm, key_outlier_norm,
        outlier_indices, query_sketch, query_states, rand_prj)


def quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False):
    """Quantized batched matmul for value reconstruction."""
    assert len(fA.shape) == 4 and len(qB.shape) == 4
    B, nh, M, K = fA.shape
    feat_per_int = 32 // bits
    fA = fA.view(-1, M, K).contiguous()
    N = qB.shape[-1] * feat_per_int
    qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
    flatten_B = B * nh if not mqa else B

    scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()

    assert bits in [2, 4]

    dispatch = {
        torch.float16: 'batchedQuantizedMultiplyAccumulate_half',
        torch.float32: 'batchedQuantizedMultiplyAccumulate_float',
        torch.bfloat16: 'batchedQuantizedMultiplyAccumulate_bf16',
    }
    fn_name = dispatch.get(fA.dtype)
    if fn_name is None:
        raise TypeError(f"Unsupported dtype: {fA.dtype}")
    result = getattr(quantization, fn_name)(fA, qB, scales, zeros, bits, group_size, nh, mqa)
    return result.view(B, nh, result.shape[-2], result.shape[-1])


# ─── QJL Sketch (adapted from QJL repo) ────────────────────────────

class QJLSketch(nn.Module):
    """
    QJL random projection sketch for 1-bit quantization.
    Adapted from amirzandieh/QJL with CUDA kernel support.
    """

    def __init__(self, dim: Tuple[int, int], dim_outlier: int,
                 device=None, rng=None, rot=True, rht=False):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert len(dim) == 2, "dim should be (head_dim, sketch_dim)"
        self.dim = dim
        self.dim_outlier = dim_outlier

        self.proj_dir = self._init_proj_dir(rng).contiguous()
        self.proj_dir_score = self._init_rot_dir().contiguous() if rot else self.proj_dir
        if rht:
            self.proj_dir_score = self._compose_rht().contiguous()
        self.proj_dir_quant = self.proj_dir_score.transpose(0, 1).contiguous()

    def _init_proj_dir(self, rng):
        return torch.randn(self.dim, generator=rng, dtype=torch.float32, device=self.device)

    def _init_rot_dir(self):
        rot_matrices = []
        num_chunks = (self.dim[1] + self.dim[0] - 1) // self.dim[0]
        for i in range(num_chunks):
            start = i * self.dim[0]
            end = (i + 1) * self.dim[0]
            q, _ = torch.linalg.qr(self.proj_dir[:, start:end], mode='reduced')
            rot_matrices.append(q)
        return torch.cat(rot_matrices, dim=-1) * math.sqrt(self.dim[0])

    def _compose_rht(self):
        H = torch.from_numpy(
            hadamard(self.dim[0], dtype=float) / math.sqrt(self.dim[0])
        ).to(self.device)
        D = 2.0 * torch.randint(0, 2, (self.dim[0],), device=self.device) - 1.0
        HD = (H * D).to(self.proj_dir_score.dtype)
        return torch.einsum('dn,dm->mn', self.proj_dir_score, HD)

    def quantize_cuda(self, data, outlier_indices):
        """Quantize using fused CUDA kernel."""
        assert data.shape[-1] == self.dim[0]
        return qjl_quant(
            data.contiguous(), outlier_indices.contiguous(),
            self.proj_dir_quant, self.dim_outlier
        )

    def quantize_pytorch(self, data, outlier_mask):
        """Pure PyTorch fallback for quantization."""
        s = self.proj_dir_quant.shape[0]
        key_outlier = data * outlier_mask.unsqueeze(-2)
        key_inlier = data * (1 - outlier_mask.unsqueeze(-2))

        proj_dtype = self.proj_dir_quant.dtype
        sketched_outlier = torch.einsum('...nd,...sd->...ns', key_outlier.to(proj_dtype), self.proj_dir_quant)
        sketched_inlier = torch.einsum('...nd,...sd->...ns', key_inlier.to(proj_dtype), self.proj_dir_quant)

        bit_pack_len = 8
        sketched_outlier = sketched_outlier.view(*sketched_outlier.shape[:-1], -1, bit_pack_len)
        sketched_inlier = sketched_inlier.view(*sketched_inlier.shape[:-1], -1, bit_pack_len)

        enc_vec = 2 ** torch.arange(bit_pack_len, dtype=torch.uint8, device=data.device).view(1, 1, 1, -1)
        hash_outlier = ((sketched_outlier > 0) * enc_vec).sum(dim=-1, dtype=torch.uint8)
        hash_inlier = ((sketched_inlier > 0) * enc_vec).sum(dim=-1, dtype=torch.uint8)

        hash_outlier = hash_outlier[:, :, :, :, :s // 16]
        return hash_inlier, hash_outlier

    def quantize(self, data, outlier_indices):
        """Dispatch to CUDA or PyTorch."""
        if _CUDA_AVAILABLE and data.is_cuda:
            return self.quantize_cuda(data, outlier_indices)
        else:
            # Convert outlier_indices to mask for PyTorch path
            mask = torch.zeros(data.shape[:3] + (data.shape[-1],), device=data.device, dtype=data.dtype)
            for i in range(outlier_indices.shape[-1]):
                idx = outlier_indices[..., i].long()
                mask.scatter_(-1, idx.unsqueeze(-1), 1.0)
            return self.quantize_pytorch(data, mask)

    def calc_score_cuda(self, query, data_quant, outlier_quant, outlier_indices, norm_data, norm_outlier):
        """Compute attention scores using CUDA kernel."""
        sketched_q = torch.matmul(query.to(self.proj_dir_score.dtype), self.proj_dir_score)
        if data_quant.stride(-1) != 1:
            data_quant = data_quant.contiguous()
        return qjl_score(
            data_quant.contiguous(), outlier_quant.contiguous(),
            norm_data.contiguous(), norm_outlier.contiguous(),
            outlier_indices.contiguous(), sketched_q.contiguous(),
            query.contiguous(), self.proj_dir_score
        )

    def calc_score_pytorch(self, query, data_quant, outlier_quant, norm_data, norm_outlier, sketch_dim):
        """Pure PyTorch fallback for score computation."""
        # Unpack bit-packed quantized keys and compute inner products
        sketched_q = torch.matmul(query.to(self.proj_dir_score.dtype), self.proj_dir_score)

        bit_pack_len = 8
        B, H = data_quant.shape[:2]

        scores_list = []
        for n in range(data_quant.shape[2]):
            for g in range(data_quant.shape[3]):
                # Unpack bits
                k_packed = data_quant[:, :, n, g]  # (B, H, hash_dim)
                bits_unpacked = torch.zeros(B, H, sketch_dim, device=query.device)
                for byte_idx in range(k_packed.shape[-1]):
                    for bit in range(8):
                        dim_idx = byte_idx * 8 + bit
                        if dim_idx < sketch_dim:
                            bits_unpacked[:, :, dim_idx] = ((k_packed[:, :, byte_idx] >> bit) & 1).float() * 2 - 1

                ip = (sketched_q.squeeze(-2) * bits_unpacked).sum(dim=-1)
                scl = math.sqrt(math.pi / 2) / sketch_dim
                nk = norm_data[:, :, n, g]
                score = scl * nk * ip
                scores_list.append(score)

        return torch.stack(scores_list, dim=-1).unsqueeze(-1)

    def calc_score(self, query, data_quant, outlier_quant, outlier_indices, norm_data, norm_outlier):
        """Dispatch to CUDA or PyTorch."""
        if _CUDA_AVAILABLE and query.is_cuda:
            return self.calc_score_cuda(query, data_quant, outlier_quant, outlier_indices, norm_data, norm_outlier)
        raise RuntimeError("CUDA kernels required for calc_score. Build with: cd turboquant/csrc && python setup.py build_ext --inplace")


# ─── Key Quantizer with streaming support ──────────────────────────

class QJLKeyQuantizer:
    """
    Online key quantizer with streaming support.
    Buffers incoming keys and quantizes them in groups.
    """

    def __init__(self, qjl_sketch: QJLSketch, outliers_count: int,
                 buffer_size: int, group_size: int, qjl_dim: int):
        self.qjl_sketch = qjl_sketch
        self.outliers_count = outliers_count
        self.buffer_size = buffer_size
        self.group_size = group_size
        self.qjl_dim = qjl_dim
        self.seq_len = None
        self.outlier_indices = None
        self.key_states_quant = None
        self.key_outliers_quant = None
        self.key_outliers_norm = None
        self.key_states_norm = None
        self.key_residual = None

    def build_sketch(self, key_states: torch.Tensor):
        """Initial quantization of a batch of keys."""
        b, h, _, dim = key_states.shape
        self.seq_len = key_states.shape[-2]
        residual_size = self.seq_len % self.buffer_size

        if residual_size > 0:
            self.key_residual = key_states[:, :, self.seq_len - residual_size:, :]
        if residual_size == self.seq_len:
            return None

        num_groups = (self.seq_len - residual_size) // self.group_size
        key_states = key_states[:, :, :self.seq_len - residual_size, :].reshape(
            (b, h, num_groups, self.group_size, dim)).contiguous()

        norms = key_states.norm(dim=-2)
        _, outlier_indices = norms.topk(self.outliers_count, dim=-1)
        self.outlier_indices = outlier_indices.to(torch.uint8).contiguous()

        self.key_states_quant, self.key_outliers_quant, self.key_outliers_norm = \
            self.qjl_sketch.quantize(key_states, self.outlier_indices)
        self.key_states_norm = torch.norm(key_states, dim=-1)

    def update_sketch(self, key_states: torch.Tensor):
        """Append a single new key token."""
        assert key_states.shape[-2] == 1
        self.seq_len += 1

        if self.key_residual is not None:
            self.key_residual = torch.cat([self.key_residual, key_states], dim=-2)
        else:
            self.key_residual = key_states

        if self.seq_len % self.buffer_size != 0:
            return None

        b, h, _, dim = self.key_residual.shape
        self.key_residual = self.key_residual.reshape((b, h, -1, self.group_size, dim))

        norms = self.key_residual.norm(dim=-2)
        _, outlier_indices = norms.topk(self.outliers_count, dim=-1)
        outlier_indices = outlier_indices.to(torch.uint8)
        self.outlier_indices = torch.cat([self.outlier_indices, outlier_indices], dim=2).contiguous()

        kq, koq, kon = self.qjl_sketch.quantize(self.key_residual, outlier_indices)
        self.key_states_quant = torch.cat([self.key_states_quant, kq], dim=2).contiguous()
        self.key_outliers_quant = torch.cat([self.key_outliers_quant, koq], dim=2).contiguous()
        self.key_outliers_norm = torch.cat([self.key_outliers_norm, kon], dim=2).contiguous()

        residual_norm = torch.norm(self.key_residual, dim=-1)
        self.key_states_norm = torch.cat([self.key_states_norm, residual_norm], dim=2).contiguous()

        self.key_residual = None

    def attention_score(self, query_states: torch.Tensor) -> torch.Tensor:
        """Compute attention scores against all quantized keys."""
        residual = None
        if self.key_residual is not None:
            residual = torch.matmul(query_states, self.key_residual.transpose(-1, -2))

        scores = self.qjl_sketch.calc_score(
            query_states,
            self.key_states_quant,
            self.key_outliers_quant,
            self.outlier_indices,
            self.key_states_norm,
            self.key_outliers_norm,
        ).transpose(-1, -2)

        if residual is not None:
            return torch.cat([scores, residual], dim=-1)
        return scores
