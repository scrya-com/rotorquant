"""
Clifford algebra Cl(3,0) for RotorQuant.

Multivector basis: [1, e1, e2, e3, e12, e13, e23, e123]
                    grade-0  grade-1     grade-2        grade-3

The geometric product table is hardcoded for GPU efficiency.
Rotors R = exp(B/2) where B is a bivector — they act via R x R̃
and naturally preserve inner products, norms, and algebraic structure.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


# Cl(3,0) basis element indices
S, E1, E2, E3, E12, E13, E23, E123 = range(8)
MV_DIM = 8  # 2^3 components for Cl(3,0)


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Full Cl(3,0) geometric product: a * b

    Input:  a, b of shape (..., 8)
    Output: result of shape (..., 8)

    Multiplication table for Cl(3,0) with signature (+,+,+):
        e_i * e_i = +1  for i in {1,2,3}
        e_i * e_j = -e_j * e_i  for i != j

    Basis ordering: [s, e1, e2, e3, e12, e13, e23, e123] where
    e12 = e1 e2, e13 = e1 e3, e23 = e2 e3, e123 = e1 e2 e3.

    The Cayley table below was derived from scratch via bubble-sort sign
    tracking of basis-vector concatenations, collapsing e_i^2 = +1.
    Each term is the sign of moving the right operand's basis vectors
    past the left operand's basis vectors into canonical order. Spot
    check: e23 * e123 = (e2 e3)(e1 e2 e3); moving e1 leftward past
    e2 e3 picks up two sign flips, leaving e1 e2 e3 e2 e3 = -e1 after
    collapsing e_i^2 -> +1, so the (a23, b123) coefficient on r1 must
    be -1 (not +1 as in earlier revisions of this file).

    Verified by (a*b)*c == a*(b*c) on random multivectors; see
    tests/test_clifford.py::TestGeometricProduct::test_associativity.
    """
    # Unbind components
    a0, a1, a2, a3, a12, a13, a23, a123 = a.unbind(dim=-1)
    b0, b1, b2, b3, b12, b13, b23, b123 = b.unbind(dim=-1)

    # Grade 0 (scalar). e123 squares to -1 in Cl(3,0):
    # e123^2 = (e1 e2 e3)(e1 e2 e3) = -e1 e1 e2 e2 e3 e3 = -1.
    r0 = (a0*b0 + a1*b1 + a2*b2 + a3*b3
           - a12*b12 - a13*b13 - a23*b23 - a123*b123)

    # Grade 1 (vectors)
    r1 = (a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3
           - a23*b123 - a123*b23)
    r2 = (a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3
           + a13*b123 + a123*b13)
    r3 = (a0*b3 + a3*b0 + a1*b13 - a13*b1 + a2*b23 - a23*b2
           - a12*b123 - a123*b12)

    # Grade 2 (bivectors)
    r12 = (a0*b12 + a12*b0 + a1*b2 - a2*b1 - a13*b23 + a23*b13
            + a3*b123 + a123*b3)
    r13 = (a0*b13 + a13*b0 + a1*b3 - a3*b1 + a12*b23 - a23*b12
            - a2*b123 - a123*b2)
    r23 = (a0*b23 + a23*b0 + a2*b3 - a3*b2 - a12*b13 + a13*b12
            + a1*b123 + a123*b1)

    # Grade 3 (pseudoscalar)
    r123 = (a0*b123 + a123*b0 + a1*b23 + a23*b1 - a2*b13 - a13*b2
             + a3*b12 + a12*b3)

    return torch.stack([r0, r1, r2, r3, r12, r13, r23, r123], dim=-1)


def reverse(x: torch.Tensor) -> torch.Tensor:
    """
    Clifford reverse (reversion) x̃: reverses the order of basis vectors.

    Grade 0, 1: unchanged (sign = +1)
    Grade 2:    negated   (sign = -1)
    Grade 3:    negated   (sign = -1)

    This is used for rotor conjugation: R x R̃
    """
    signs = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1],
                         dtype=x.dtype, device=x.device)
    return x * signs


def multivector_norm_sq(x: torch.Tensor) -> torch.Tensor:
    """||x||² = <x x̃>_0  (scalar part of x * reverse(x))"""
    x_rev = reverse(x)
    product = geometric_product(x, x_rev)
    return product[..., 0]  # scalar part


def make_rotor(bivector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotor R = cos(θ/2) + sin(θ/2) * B̂  where B̂ is a unit bivector.

    bivector: (..., 3) — coefficients for [e12, e13, e23]
    angle:    (...,)   — rotation angle in radians

    Returns: (..., 8) multivector rotor
    """
    # Normalize bivector
    bv_norm = bivector.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    bv_hat = bivector / bv_norm

    half_angle = angle.unsqueeze(-1) / 2
    cos_ha = torch.cos(half_angle)
    sin_ha = torch.sin(half_angle)

    # R = cos(θ/2) + sin(θ/2) * (b12*e12 + b13*e13 + b23*e23)
    rotor = torch.zeros(*bivector.shape[:-1], 8, dtype=bivector.dtype, device=bivector.device)
    rotor[..., S] = cos_ha.squeeze(-1)
    rotor[..., E12] = sin_ha.squeeze(-1) * bv_hat[..., 0]
    rotor[..., E13] = sin_ha.squeeze(-1) * bv_hat[..., 1]
    rotor[..., E23] = sin_ha.squeeze(-1) * bv_hat[..., 2]
    return rotor


def make_random_rotor(shape: Tuple[int, ...], device='cpu', seed=None) -> torch.Tensor:
    """
    Generate a random rotor via random bivector + random angle.
    Returns a normalized rotor R with R R̃ = 1.
    """
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)

    # Random bivector direction
    full_shape = list(shape) + [3]
    bv = torch.randn(full_shape, generator=gen).to(device)
    # Random angle in [0, 2π)
    angle_shape = list(shape) if shape else [1]
    angle = torch.rand(angle_shape, generator=gen).to(device) * 2 * math.pi

    rotor = make_rotor(bv, angle)
    # Normalize: R / sqrt(R R̃)
    norm = multivector_norm_sq(rotor).abs().sqrt().unsqueeze(-1).clamp(min=1e-8)
    return rotor / norm


def rotor_sandwich(rotor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply rotor sandwich product: R x R̃
    This rotates x while preserving all algebraic structure.
    """
    rotor_rev = reverse(rotor)
    return geometric_product(geometric_product(rotor, x), rotor_rev)


def embed_vectors_as_multivectors(v: torch.Tensor) -> torch.Tensor:
    """
    Embed d-dimensional vectors into Cl(3,0) multivectors.

    For d divisible by 3: pack into grade-1 components (e1, e2, e3).
    For d not divisible by 3: also use scalar and bivector grades.

    v: (..., d) → (..., d_groups, 8)
    """
    d = v.shape[-1]
    # Pad to multiple of 3 if needed
    pad = (3 - d % 3) % 3
    if pad > 0:
        v = torch.nn.functional.pad(v, (0, pad))
    d_padded = v.shape[-1]
    n_groups = d_padded // 3

    # Reshape into groups of 3
    v_grouped = v.reshape(*v.shape[:-1], n_groups, 3)

    # Create multivectors with grade-1 components
    mv = torch.zeros(*v_grouped.shape[:-1], 8, dtype=v.dtype, device=v.device)
    mv[..., E1] = v_grouped[..., 0]
    mv[..., E2] = v_grouped[..., 1]
    mv[..., E3] = v_grouped[..., 2]

    return mv


def extract_vectors_from_multivectors(mv: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """
    Extract d-dimensional vectors from Cl(3,0) multivectors.
    Inverse of embed_vectors_as_multivectors.

    mv: (..., n_groups, 8) → (..., d)
    """
    v = torch.stack([mv[..., E1], mv[..., E2], mv[..., E3]], dim=-1)
    v = v.reshape(*mv.shape[:-2], -1)
    return v[..., :orig_dim]
