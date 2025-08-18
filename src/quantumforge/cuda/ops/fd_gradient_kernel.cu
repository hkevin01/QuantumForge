#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace qf {

enum BoundaryMode : int {
    PERIODIC = 0,
    ZERO = 1,
};

template <typename scalar_t>
__global__ void fd_gradient3d_kernel(
    const scalar_t* __restrict__ values,
    scalar_t* __restrict__ grad,
    const int nx,
    const int ny,
    const int nz,
    const scalar_t sx,
    const scalar_t sy,
    const scalar_t sz,
    const int boundary_mode)
{
    // Flattened 3D index over grid points
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (idx >= total) return;

    // Recover i, j, k from idx (row-major)
    // values layout expected: [nx, ny, nz]
    int i = idx / (ny * nz);
    int rem = idx % (ny * nz);
    int j = rem / nz;
    int k = rem % nz;

    // Helper to compute linear index
    auto L = [ny, nz](int ii, int jj, int kk) {
        return (ii * ny + jj) * nz + kk;
    };

    // Neighbor indices with boundary handling
    auto neighbor = [&](int ii, int jj, int kk) {
        if (boundary_mode == BoundaryMode::PERIODIC) {
            int ni = (ii + nx) % nx;
            int nj = (jj + ny) % ny;
            int nk = (kk + nz) % nz;
            return L(ni, nj, nk);
        } else { // ZERO
            auto clampi = [&](int v, int lo, int hi) {
                return v < lo ? lo : (v > hi ? hi : v);
            };
            int ni = clampi(ii, 0, nx - 1);
            int nj = clampi(jj, 0, ny - 1);
            int nk = clampi(kk, 0, nz - 1);
            return L(ni, nj, nk);
        }
    };

    // Central differences
    scalar_t vxp = values[neighbor(i + 1, j, k)];
    scalar_t vxm = values[neighbor(i - 1, j, k)];
    scalar_t vyp = values[neighbor(i, j + 1, k)];
    scalar_t vym = values[neighbor(i, j - 1, k)];
    scalar_t vzp = values[neighbor(i, j, k + 1)];
    scalar_t vzm = values[neighbor(i, j, k - 1)];

    scalar_t gx = (vxp - vxm) / (2.0 * sx);
    scalar_t gy = (vyp - vym) / (2.0 * sy);
    scalar_t gz = (vzp - vzm) / (2.0 * sz);

    // Output layout: [3, nx, ny, nz]
    grad[0 * total + idx] = gx;
    grad[1 * total + idx] = gy;
    grad[2 * total + idx] = gz;
}

std::tuple<at::Tensor, at::Tensor>
fd_gradient3d_cuda_impl(const at::Tensor& values,
                        double sx, double sy, double sz,
                        int boundary_mode) {
    TORCH_CHECK(values.dim() == 3, "values must be 3D tensor [nx, ny, nz]");
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA device");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
    TORCH_CHECK(values.scalar_type() == at::kFloat || values.scalar_type() == at::kDouble,
                "values must be float32 or float64");

    const int nx = static_cast<int>(values.size(0));
    const int ny = static_cast<int>(values.size(1));
    const int nz = static_cast<int>(values.size(2));
    const int64_t total = static_cast<int64_t>(nx) * ny * nz;

    auto opts = values.options();
    at::Tensor grad = at::empty({3, nx, ny, nz}, opts);

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "fd_gradient3d_kernel", [&]() {
        fd_gradient3d_kernel<scalar_t><<<blocks, threads>>>(
            values.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            nx, ny, nz,
            static_cast<scalar_t>(sx),
            static_cast<scalar_t>(sy),
            static_cast<scalar_t>(sz),
            boundary_mode);
    });

    // Return gradient and preserve original values for potential backward (placeholder)
    return {grad, values};
}

} // namespace qf
