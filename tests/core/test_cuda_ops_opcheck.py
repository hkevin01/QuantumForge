import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.unit]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_opcheck_custom_ops_registered_correctly():
    # Smoke-test opcheck on our registered ops for schema/fake/autograd wiring.
    # Note: opcheck won't validate numerics; we do that in dedicated tests.
    from quantumforge.cuda.ops import fd_gradient3d, quadrature_batched

    # fd_gradient3d
    nx, ny, nz = 6, 5, 4
    sx, sy, sz = 0.2, 0.25, 0.3
    field = torch.randn(nx, ny, nz, device="cuda", dtype=torch.float32)
    out = fd_gradient3d(field, (sx, sy, sz))
    assert out.shape == (3, nx, ny, nz)

    # opcheck requires OpOverload/CustomOpDef; call via torch.ops
    torch.library.opcheck(
        torch.ops.quantumforge_ops.fd_gradient3d.default,
        (field, float(sx), float(sy), float(sz), 0),
    )

    # quadrature_batched
    B = 3
    vals = torch.randn(B, nx, ny, nz, device="cuda", dtype=torch.float32)
    weights = torch.full((nx * ny * nz,), sx * sy * sz, device="cuda")
    q = quadrature_batched(vals, weights)
    assert q.shape == (B,)

    torch.library.opcheck(
        torch.ops.quantumforge_ops.quadrature_batched.default,
        (vals.contiguous(), weights.contiguous()),
    )
