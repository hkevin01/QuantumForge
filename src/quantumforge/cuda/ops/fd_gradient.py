"""Python loader and wrapper for CUDA finite-difference gradient op."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# JIT compile and load the CUDA/C++ extension on first import
# Sources are located relative to this file
_THIS_DIR = Path(__file__).resolve().parent
_SOURCES = [
    str(_THIS_DIR / "fd_gradient_binding.cpp"),
    str(_THIS_DIR / "fd_gradient_kernel.cu"),
]


@lru_cache(maxsize=None)
def _get_extension():
    return load(
        name="quantumforge_fd_gradient",
        sources=_SOURCES,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def _ensure_registrations() -> None:
    if getattr(_ensure_registrations, "_did", False):
        return
    # Ensure C++ op is loaded before registering fake kernels
    _get_extension()
    try:
        @torch.library.register_fake("quantumforge_ops::fd_gradient3d")
        def _(values, _sx: float, _sy: float, _sz: float, _boundary_mode: int):
            # values: [nx, ny, nz] -> output: [3, nx, ny, nz]
            assert values.dim() == 3
            out_shape = (3, values.shape[0], values.shape[1], values.shape[2])
            return torch.empty(
                out_shape, device=values.device, dtype=values.dtype
            )
    except RuntimeError:
        # If already registered or not available, ignore
        pass
    setattr(_ensure_registrations, "_did", True)


def fd_gradient3d(
    values: torch.Tensor,
    spacing: tuple[float, float, float],
    boundary: str = "periodic",
) -> torch.Tensor:
    """Compute 3D finite-difference gradient on CUDA.

    Args:
        values: 3D tensor [nx, ny, nz] on CUDA device
        spacing: (sx, sy, sz)
        boundary: 'periodic' or 'zero'

    Returns:
        Tensor with shape [3, nx, ny, nz]
    """
    if not values.is_cuda:
        raise ValueError("values must be a CUDA tensor")
    if values.dim() != 3:
        raise ValueError("values must be 3D [nx, ny, nz]")

    sx, sy, sz = map(float, spacing)
    mode = 0 if boundary == "periodic" else 1

    _ensure_registrations()
    # Call custom operator via torch.ops registration
    # Schema: quantumforge_ops::fd_gradient3d
    out = torch.ops.quantumforge_ops.fd_gradient3d(
        values, float(sx), float(sy), float(sz), int(mode)
    )
    return out


