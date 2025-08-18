import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.unit]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_quadrature_batched_matches_cpu_sum_rect_rule_unbatched():
    from quantumforge.cuda.ops import quadrature_batched

    nx, ny, nz = 8, 6, 5
    sx, sy, sz = 0.3, 0.25, 0.2
    N = nx * ny * nz

    # Build a simple field f(x,y,z) and rectangular-rule weights
    x = torch.linspace(0, (nx - 1) * sx, nx)
    y = torch.linspace(0, (ny - 1) * sy, ny)
    z = torch.linspace(0, (nz - 1) * sz, nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    f = torch.sin(0.2 * X) + torch.cos(0.3 * Y) + 0.5 * torch.sin(0.4 * Z)

    weights = torch.full((N,), sx * sy * sz, dtype=f.dtype)

    # CPU reference
    ref = (f.flatten() * weights).sum()

    # CUDA op
    out = quadrature_batched(f.contiguous().cuda(), weights.cuda()).cpu()
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_quadrature_batched_matches_cpu_sum_rect_rule_batched():
    from quantumforge.cuda.ops import quadrature_batched

    B = 4
    nx, ny, nz = 8, 6, 5
    sx, sy, sz = 0.3, 0.25, 0.2
    N = nx * ny * nz

    x = torch.linspace(0, (nx - 1) * sx, nx)
    y = torch.linspace(0, (ny - 1) * sy, ny)
    z = torch.linspace(0, (nz - 1) * sz, nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    base = torch.sin(0.2 * X) + torch.cos(0.3 * Y) + 0.5 * torch.sin(0.4 * Z)

    # Create B different fields with small shifts
    vals = torch.stack([base + 0.1 * i for i in range(B)], dim=0)

    weights = torch.full((N,), sx * sy * sz, dtype=vals.dtype)

    # CPU reference per batch
    ref = (vals.view(B, -1) * weights.view(1, -1)).sum(dim=1)

    out = quadrature_batched(vals.contiguous().cuda(), weights.cuda()).cpu()
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
