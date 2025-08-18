"""
Tests for core functional base classes.
"""

import pytest
import torch

from quantumforge.core.functional_base import (
    GGAFunctional,
    HybridFunctional,
    LDAFunctional,
    MetaGGAFunctional,
)


class TestLDAFunctional(LDAFunctional):
    """Concrete LDA functional for testing."""

    def forward(self, rho, **_kwargs):
        """Simple test implementation - LDA exchange."""
        return (
            -0.75 * (3.0 / torch.pi) ** (1.0/3.0) * rho ** (4.0/3.0)
        )


class TestGGAFunctional(GGAFunctional):
    """Concrete GGA functional for testing."""

    def forward(self, rho, grad_rho=None, **_kwargs):
        """Simple test implementation - PBE-like."""
        # LDA part
        ex_lda = -0.75 * (3.0 / torch.pi) ** (1.0/3.0) * rho ** (4.0/3.0)

        if grad_rho is None:
            return ex_lda

        # Simple gradient correction
        grad_norm = torch.norm(grad_rho, dim=1, keepdim=True)
        s = grad_norm / (
            2.0 * (3.0 * torch.pi**2) ** (1.0/3.0) * rho ** (4.0/3.0)
        )

        # Simple enhancement factor
        enhancement = 1.0 + 0.1 * s**2

        return ex_lda * enhancement


class TestMetaGGAFunctional(MetaGGAFunctional):
    """Concrete meta-GGA functional for testing."""

    def forward(self, rho, grad_rho=None, tau=None, **_kwargs):
        """Simple test implementation - SCAN-like."""
        # Start with GGA-like part
        ex_lda = -0.75 * (3.0 / torch.pi) ** (1.0/3.0) * rho ** (4.0/3.0)

        if grad_rho is None and tau is None:
            return ex_lda

        enhancement = torch.ones_like(rho)

        if grad_rho is not None:
            grad_norm = torch.norm(grad_rho, dim=1, keepdim=True)
            s = grad_norm / (
                2.0 * (3.0 * torch.pi**2) ** (1.0/3.0) * rho ** (4.0/3.0)
            )
            enhancement *= (1.0 + 0.1 * s**2)

        if tau is not None:
            # Simple tau-dependent correction
            tau_unif = (
                0.3 * (3.0 * torch.pi**2) ** (2.0/3.0) * rho ** (5.0/3.0)
            )
            alpha = (
                tau
                - 0.125 * torch.norm(grad_rho, dim=1, keepdim=True) ** 2 / rho
            ) / tau_unif
            enhancement *= (1.0 + 0.05 * alpha)

        return ex_lda * enhancement


class TestHybridFunctional(HybridFunctional):
    """Concrete hybrid functional for testing."""

    def forward(self, rho, **_kwargs):
        """Simple test implementation."""
        return -0.75 * (3.0 / torch.pi) ** (1.0/3.0) * rho ** (4.0/3.0)

    def compute_exact_exchange(self, density_matrix, **_kwargs):
        """Placeholder exact exchange implementation."""
        return torch.sum(density_matrix**2) * 0.25  # Simplified


def test_lda_functional():
    """Test LDA functional implementation."""
    lda = TestLDAFunctional()

    # Test properties
    assert lda.name == "LDA"
    assert not lda.supports_gga
    assert not lda.supports_meta

    # Test forward pass
    rho = torch.rand(2, 1, 10, 10, 10)  # [batch, spin, nx, ny, nz]
    rho = rho.abs() + 1e-10  # Ensure positive density

    result = lda(rho)
    assert result.shape == rho.shape
    assert torch.all(result <= 0)  # Exchange energy should be negative


def test_gga_functional():
    """Test GGA functional implementation."""
    gga = TestGGAFunctional()

    # Test properties
    assert gga.name == "GGA"
    assert gga.supports_gga
    assert not gga.supports_meta

    # Test forward pass with gradient
    rho = torch.rand(2, 1, 10, 10, 10)
    rho = rho.abs() + 1e-10
    grad_rho = torch.rand(2, 3, 10, 10, 10)  # [batch, 3, nx, ny, nz]

    result = gga(rho, grad_rho=grad_rho)
    assert result.shape == rho.shape
    assert torch.all(result <= 0)


def test_meta_gga_functional():
    """Test meta-GGA functional implementation."""
    meta_gga = TestMetaGGAFunctional()

    # Test properties
    assert meta_gga.name == "Meta-GGA"
    assert meta_gga.supports_gga
    assert meta_gga.supports_meta

    # Test forward pass with gradient and tau
    rho = torch.rand(2, 1, 10, 10, 10)
    rho = rho.abs() + 1e-10
    grad_rho = torch.rand(2, 3, 10, 10, 10)
    tau = torch.rand(2, 1, 10, 10, 10).abs() + 1e-10

    result = meta_gga(rho, grad_rho=grad_rho, tau=tau)
    assert result.shape == rho.shape
    assert torch.all(result <= 0)


def test_hybrid_functional():
    """Test hybrid functional implementation."""
    hybrid = TestHybridFunctional()

    # Test properties
    assert hybrid.name == "Hybrid"
    assert hybrid.exact_exchange_fraction == 0.25

    # Test forward pass
    rho = torch.rand(2, 1, 10, 10, 10)
    rho = rho.abs() + 1e-10

    result = hybrid(rho)
    assert result.shape == rho.shape

    # Test exact exchange
    density_matrix = torch.rand(10, 10)
    ex_exact = hybrid.compute_exact_exchange(density_matrix)
    assert ex_exact.numel() == 1


def test_input_validation():
    """Test input validation for functionals."""
    lda = TestLDAFunctional()
    gga = TestGGAFunctional()

    # Test wrong dimensions
    rho_wrong = torch.rand(10, 10)  # Wrong dimensions

    with pytest.raises(ValueError, match="Expected rho to have 5 dimensions"):
        lda(rho_wrong)

    # Test GGA with wrong gradient shape
    rho = torch.rand(2, 1, 10, 10, 10)
    grad_rho_wrong = torch.rand(2, 2, 10, 10, 10)  # Should have 3 components

    with pytest.raises(
        ValueError, match="Expected grad_rho to have 3 components"
    ):
        gga(rho, grad_rho=grad_rho_wrong)

    # Test LDA with gradient (should raise error)
    grad_rho = torch.rand(2, 3, 10, 10, 10)

    with pytest.raises(ValueError, match="does not support GGA"):
        lda(rho, grad_rho=grad_rho)


def test_repr():
    """Test string representation of functionals."""
    lda = TestLDAFunctional()
    gga = TestGGAFunctional()

    assert "TestLDAFunctional" in repr(lda)
    assert "LDA" in repr(lda)
    assert "TestGGAFunctional" in repr(gga)
    assert "GGA" in repr(gga)


if __name__ == "__main__":
    pytest.main([__file__])
