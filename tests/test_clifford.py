"""Tests for Cl(3,0) Clifford algebra primitives."""
import pytest
import torch
import math

from turboquant.clifford import (
    geometric_product, reverse, multivector_norm_sq,
    make_rotor, make_random_rotor, rotor_sandwich,
    embed_vectors_as_multivectors, extract_vectors_from_multivectors,
    MV_DIM,
)


class TestGeometricProduct:
    """Test the full Cl(3,0) geometric product table."""

    def test_scalar_times_scalar(self):
        a = torch.tensor([2.0, 0, 0, 0, 0, 0, 0, 0])
        b = torch.tensor([3.0, 0, 0, 0, 0, 0, 0, 0])
        r = geometric_product(a, b)
        assert torch.allclose(r[0], torch.tensor(6.0))
        assert torch.allclose(r[1:], torch.zeros(7))

    def test_e1_times_e1(self):
        """e1 * e1 = +1 in Cl(3,0)."""
        e1 = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])
        r = geometric_product(e1, e1)
        assert torch.allclose(r[0], torch.tensor(1.0), atol=1e-6)

    def test_e1_times_e2(self):
        """e1 * e2 = e12."""
        e1 = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])
        e2 = torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0])
        r = geometric_product(e1, e2)
        assert torch.allclose(r[4], torch.tensor(1.0), atol=1e-6)  # e12 component

    def test_e2_times_e1(self):
        """e2 * e1 = -e12 (anticommutativity)."""
        e1 = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])
        e2 = torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0])
        r = geometric_product(e2, e1)
        assert torch.allclose(r[4], torch.tensor(-1.0), atol=1e-6)

    def test_associativity(self):
        """(a * b) * c == a * (b * c).

        Associativity is a foundational axiom of any Clifford / geometric
        algebra. Prior revisions of this file carried sign errors in
        r1/r2/r3/r12/r13/r23/r123 that caused this test to fail and was
        previously marked xfail; the fix derives the Cayley table from
        scratch via bubble-sort sign tracking and restores associativity.
        """
        torch.manual_seed(42)
        a = torch.randn(8)
        b = torch.randn(8)
        c = torch.randn(8)
        lhs = geometric_product(geometric_product(a, b), c)
        rhs = geometric_product(a, geometric_product(b, c))
        assert torch.allclose(lhs, rhs, atol=1e-5)

    def test_associativity_batch(self):
        """Associativity across a batch of random multivectors."""
        torch.manual_seed(7)
        a = torch.randn(16, 8)
        b = torch.randn(16, 8)
        c = torch.randn(16, 8)
        lhs = geometric_product(geometric_product(a, b), c)
        rhs = geometric_product(a, geometric_product(b, c))
        assert torch.allclose(lhs, rhs, atol=1e-5)

    def test_scalar_identity_left(self):
        """1 * x == x for arbitrary multivector x."""
        torch.manual_seed(42)
        one = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
        x = torch.randn(8)
        assert torch.allclose(geometric_product(one, x), x, atol=1e-6)

    def test_scalar_identity_right(self):
        """x * 1 == x for arbitrary multivector x."""
        torch.manual_seed(42)
        one = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
        x = torch.randn(8)
        assert torch.allclose(geometric_product(x, one), x, atol=1e-6)

    def test_basis_vector_squares(self):
        """e_i * e_i = +1 for i in {1, 2, 3}; e_ij * e_ij = -1 for grade-2."""
        def basis(idx):
            v = torch.zeros(8)
            v[idx] = 1.0
            return v

        # Grade-1 squares
        for i in (1, 2, 3):
            r = geometric_product(basis(i), basis(i))
            expected = torch.zeros(8); expected[0] = 1.0
            assert torch.allclose(r, expected, atol=1e-6), f"e{i}^2 != +1"

        # Grade-2 squares are -1
        for i in (4, 5, 6):
            r = geometric_product(basis(i), basis(i))
            expected = torch.zeros(8); expected[0] = -1.0
            assert torch.allclose(r, expected, atol=1e-6), f"e_{i}^2 != -1"

        # Pseudoscalar squares to -1
        r = geometric_product(basis(7), basis(7))
        expected = torch.zeros(8); expected[0] = -1.0
        assert torch.allclose(r, expected, atol=1e-6), "e123^2 != -1"

    def test_basis_vector_anticommutation(self):
        """e_i * e_j = -e_j * e_i for i != j in {1,2,3}."""
        def basis(idx):
            v = torch.zeros(8)
            v[idx] = 1.0
            return v

        for i, j in [(1, 2), (1, 3), (2, 3)]:
            r_ij = geometric_product(basis(i), basis(j))
            r_ji = geometric_product(basis(j), basis(i))
            assert torch.allclose(r_ij, -r_ji, atol=1e-6), (
                f"e{i}*e{j} != -e{j}*e{i}"
            )

    def test_e1_e2_e3_chain_associates(self):
        """(e1 * e2) * e3 == e1 * (e2 * e3) == e123.

        Specific instance of associativity that was broken in the prior
        Cayley table. (e1*e2) = e12 and e12 * e3 = e123. Separately,
        (e2*e3) = e23 and e1 * e23 = e123. Both routes must agree.
        """
        e1 = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])
        e2 = torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0])
        e3 = torch.tensor([0, 0, 0, 1.0, 0, 0, 0, 0])

        left = geometric_product(geometric_product(e1, e2), e3)
        right = geometric_product(e1, geometric_product(e2, e3))

        expected = torch.zeros(8); expected[7] = 1.0  # e123
        assert torch.allclose(left, expected, atol=1e-6)
        assert torch.allclose(right, expected, atol=1e-6)
        assert torch.allclose(left, right, atol=1e-6)

    def test_e23_times_e123(self):
        """e23 * e123 = -e1.

        Derivation: (e2 e3)(e1 e2 e3); move e1 leftward past e2 and e3
        with two sign flips: e2 e3 e1 e2 e3 = e1 e2 e3 e2 e3 = -e1 after
        collapsing e_i^2 -> +1. This was wrong in the prior Cayley table.
        """
        e23 = torch.tensor([0, 0, 0, 0, 0, 0, 1.0, 0])
        e123 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1.0])
        r = geometric_product(e23, e123)
        expected = torch.tensor([0, -1.0, 0, 0, 0, 0, 0, 0])
        assert torch.allclose(r, expected, atol=1e-6)

    def test_e12_times_e3(self):
        """e12 * e3 = e123."""
        e12 = torch.tensor([0, 0, 0, 0, 1.0, 0, 0, 0])
        e3 = torch.tensor([0, 0, 0, 1.0, 0, 0, 0, 0])
        r = geometric_product(e12, e3)
        expected = torch.zeros(8); expected[7] = 1.0
        assert torch.allclose(r, expected, atol=1e-6)

    def test_batch_dimensions(self):
        """GP should work with batch dims."""
        torch.manual_seed(42)
        a = torch.randn(10, 8)
        b = torch.randn(10, 8)
        r = geometric_product(a, b)
        assert r.shape == (10, 8)
        # Verify first element matches unbatched
        r0 = geometric_product(a[0], b[0])
        assert torch.allclose(r[0], r0, atol=1e-6)

    def test_batch_2d(self):
        """GP with (batch, groups, 8)."""
        torch.manual_seed(42)
        a = torch.randn(5, 3, 8)
        b = torch.randn(5, 3, 8)
        r = geometric_product(a, b)
        assert r.shape == (5, 3, 8)


class TestReverse:
    def test_grade_signs(self):
        """Grade 0,1 unchanged; grade 2,3 negated."""
        x = torch.ones(8)
        x_rev = reverse(x)
        expected = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float)
        assert torch.allclose(x_rev, expected)

    def test_double_reverse_is_identity(self):
        torch.manual_seed(42)
        x = torch.randn(8)
        assert torch.allclose(reverse(reverse(x)), x)


class TestRotors:
    def test_rotor_is_normalized(self):
        """R R̃ = 1 (scalar part)."""
        r = make_random_rotor((), seed=42)
        norm_sq = multivector_norm_sq(r)
        assert torch.allclose(norm_sq, torch.tensor(1.0), atol=1e-5)

    def test_rotor_has_correct_structure(self):
        """Rotor should have non-zero: scalar, e12, e13, e23. Zero: e1, e2, e3, e123."""
        r = make_random_rotor((), seed=42)
        # Grade-1 and grade-3 should be zero
        assert torch.allclose(r[1], torch.tensor(0.0), atol=1e-7)
        assert torch.allclose(r[2], torch.tensor(0.0), atol=1e-7)
        assert torch.allclose(r[3], torch.tensor(0.0), atol=1e-7)
        assert torch.allclose(r[7], torch.tensor(0.0), atol=1e-7)

    def test_make_rotor_from_bivector(self):
        bv = torch.tensor([1.0, 0.0, 0.0])  # rotation in e12 plane
        angle = torch.tensor(math.pi / 2)
        r = make_rotor(bv, angle)
        # cos(pi/4) ~ 0.707, sin(pi/4) ~ 0.707
        assert abs(r[0].item() - math.cos(math.pi/4)) < 1e-5
        assert abs(r[4].item() - math.sin(math.pi/4)) < 1e-5

    def test_rotor_sandwich_preserves_norm(self):
        """||RvR̃|| = ||v||."""
        torch.manual_seed(42)
        r = make_random_rotor((), seed=42)
        v = torch.randn(8)
        v[0] = 0; v[4] = 0; v[5] = 0; v[6] = 0; v[7] = 0  # pure vector
        v_rot = rotor_sandwich(r, v)
        norm_orig = torch.sqrt((v[1]**2 + v[2]**2 + v[3]**2))
        norm_rot = torch.sqrt((v_rot[1]**2 + v_rot[2]**2 + v_rot[3]**2))
        assert torch.allclose(norm_orig, norm_rot, atol=1e-4)

    def test_identity_rotor(self):
        """R = [1, 0, 0, 0, 0, 0, 0, 0] should be identity."""
        r = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
        v = torch.tensor([0, 1.0, 2.0, 3.0, 0, 0, 0, 0])
        v_rot = rotor_sandwich(r, v)
        assert torch.allclose(v, v_rot, atol=1e-6)

    def test_different_seeds_give_different_rotors(self):
        r1 = make_random_rotor((), seed=42)
        r2 = make_random_rotor((), seed=99)
        assert not torch.allclose(r1, r2)

    def test_batch_rotors(self):
        """Multiple rotors applied to multiple vectors."""
        torch.manual_seed(42)
        rotors = torch.stack([make_random_rotor((), seed=i) for i in range(5)])
        v = torch.randn(5, 8)
        v[:, 0] = 0; v[:, 4:] = 0  # pure vectors
        v_rot = rotor_sandwich(rotors, v)
        assert v_rot.shape == (5, 8)


class TestEmbedExtract:
    def test_roundtrip_exact(self):
        """embed then extract should recover the original vector."""
        x = torch.randn(10, 128)
        mv = embed_vectors_as_multivectors(x)
        x_back = extract_vectors_from_multivectors(mv, 128)
        assert torch.allclose(x, x_back, atol=1e-6)

    def test_embed_shape(self):
        x = torch.randn(5, 128)
        mv = embed_vectors_as_multivectors(x)
        n_groups = (128 + 2) // 3  # 43
        assert mv.shape == (5, n_groups, 8)

    def test_embed_grade1_only(self):
        """Embedded vectors should only have grade-1 components."""
        x = torch.randn(3, 12)
        mv = embed_vectors_as_multivectors(x)
        # Scalar, bivector, trivector should be zero
        assert torch.allclose(mv[..., 0], torch.zeros_like(mv[..., 0]))
        assert torch.allclose(mv[..., 4], torch.zeros_like(mv[..., 4]))
        assert torch.allclose(mv[..., 5], torch.zeros_like(mv[..., 5]))
        assert torch.allclose(mv[..., 6], torch.zeros_like(mv[..., 6]))
        assert torch.allclose(mv[..., 7], torch.zeros_like(mv[..., 7]))

    def test_padding(self):
        """d not divisible by 3 should still work."""
        for d in [127, 128, 129, 130, 1]:
            x = torch.randn(2, d)
            mv = embed_vectors_as_multivectors(x)
            x_back = extract_vectors_from_multivectors(mv, d)
            assert torch.allclose(x, x_back, atol=1e-6), f"Failed for d={d}"

    def test_single_vector(self):
        """Should handle unbatched input via batch dim."""
        x = torch.randn(1, 64)
        mv = embed_vectors_as_multivectors(x)
        x_back = extract_vectors_from_multivectors(mv, 64)
        assert torch.allclose(x, x_back, atol=1e-6)
