"""Python loader and wrapper for CUDA batched quadrature op."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent
_SOURCES = [
    str(_THIS_DIR / "quadrature_batched_binding.cpp"),
    str(_THIS_DIR / "quadrature_batched_kernel.cu"),
]


@lru_cache(maxsize=None)
def _get_extension():
    return load(
        name="quantumforge_quadrature_batched",
        sources=_SOURCES,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def _ensure_registrations() -> None:
    if getattr(_ensure_registrations, "_did", False):
        return
    _get_extension()
    try:
        @torch.library.register_fake("quantumforge_ops::quadrature_batched")
        def _(values, _weights):
            # Normalize to [B, N], weights [N]
            if values.dim() in (1, 3):
                B = 1
            elif values.dim() in (2, 4):
                B = int(values.shape[0])
            else:
                raise AssertionError("invalid values dim")
            return torch.empty((B,), device=values.device, dtype=values.dtype)
    except RuntimeError:
        pass
    setattr(_ensure_registrations, "_did", True)


def quadrature_batched(
    values: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Integrate values over grid points with weights, batched.

    Shapes accepted:
      - values: [nx, ny, nz] or [B, nx, ny, nz] or [N] or [B, N]
      - weights: [nx, ny, nz] or [N]
    Returns:
      - [1] if values is unbatched, else [B]
    """
    if not values.is_cuda or not weights.is_cuda:
        raise ValueError("values and weights must be CUDA tensors")

    _ensure_registrations()

    # Ensure contiguity
    v = values.contiguous()
    w = weights.contiguous()

    out = torch.ops.quantumforge_ops.quadrature_batched(v, w)

    # For unbatched inputs, squeeze to scalar tensor shape [] for convenience
    if values.dim() in (1, 3):
        return out.reshape(())
    return out
