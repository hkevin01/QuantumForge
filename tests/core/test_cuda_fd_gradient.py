import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.unit]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fd_gradient3d_matches_cpu_centered_diff():
    from quantumforge.cuda.ops import fd_gradient3d

    nx, ny, nz = 16, 12, 10
    sx, sy, sz = 0.2, 0.25, 0.3

    # Create a smooth test field on CPU
    x = torch.linspace(0, (nx - 1) * sx, nx)
    y = torch.linspace(0, (ny - 1) * sy, ny)
    z = torch.linspace(0, (nz - 1) * sz, nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    f = torch.sin(0.7 * X) * torch.cos(0.3 * Y) * torch.sin(0.5 * Z)

    # Numerical gradient on CPU using centered differences with periodic BCs
    def cpu_grad_centered(values, spacing):
        sx, sy, sz = spacing

        def roll(a, shift, dim):
            return torch.roll(a, shifts=shift, dims=dim)
        gx = (roll(values, -1, 0) - roll(values, 1, 0)) / (2 * sx)
        gy = (roll(values, -1, 1) - roll(values, 1, 1)) / (2 * sy)
        gz = (roll(values, -1, 2) - roll(values, 1, 2)) / (2 * sz)
        return torch.stack([gx, gy, gz], dim=0)

    grad_cpu = cpu_grad_centered(f, (sx, sy, sz))

    # Move to CUDA and evaluate custom op
    f_cuda = f.contiguous().cuda()
    grad_cuda = fd_gradient3d(f_cuda, (sx, sy, sz), boundary="periodic").cpu()

    # Compare
    torch.testing.assert_close(grad_cuda, grad_cpu, atol=1e-4, rtol=1e-4)
