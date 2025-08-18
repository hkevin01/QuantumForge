"""Convenience wrappers for QuantumForge CUDA ops."""
from __future__ import annotations

import torch

from quantumforge.cuda.ops import fd_gradient3d as _fd_grad
from quantumforge.cuda.ops import quadrature_batched as _quad_batched


def fd_gradient3d(
    values: torch.Tensor,
    spacing,
    boundary: str = "periodic",
) -> torch.Tensor:
    """CUDA finite-difference gradient; helpful error if CUDA missing."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: cannot run fd_gradient3d")
    return _fd_grad(values, spacing, boundary)


__all__ = ["fd_gradient3d"]


def quadrature_batched(
    values: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """CUDA batched quadrature with helpful CUDA availability check."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: cannot run quadrature_batched")
    return _quad_batched(values, weights)


__all__.append("quadrature_batched")
