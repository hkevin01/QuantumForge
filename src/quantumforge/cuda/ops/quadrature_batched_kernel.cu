#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace qf {

// Kernel: for each batch b, compute sum_i values[b, i] * weights[i]
template <typename scalar_t>
__global__ void quadrature_batched_kernel(
    const scalar_t* __restrict__ values, // [B*N]
    const scalar_t* __restrict__ weights, // [N]
    scalar_t* __restrict__ out,           // [B]
    const int64_t N,
    const int64_t B)
{
    const int b = blockIdx.y; // batch index
    if (b >= B) return;

    // Grid-stride loop over spatial dimension N
    scalar_t sum = 0;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        sum += values[b * N + i] * weights[i];
    }

    // Shared-memory reduction within block
    extern __shared__ unsigned char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&out[b], sdata[0]);
    }
}

std::tuple<at::Tensor>
quadrature_batched_cuda_impl(
    const at::Tensor& values_in,
    const at::Tensor& weights_in)
{
    TORCH_CHECK(values_in.is_cuda(), "quadrature_batched: values must be CUDA tensor");
    TORCH_CHECK(weights_in.is_cuda(), "quadrature_batched: weights must be CUDA tensor");
    TORCH_CHECK(values_in.is_contiguous(), "quadrature_batched: values must be contiguous");
    TORCH_CHECK(weights_in.is_contiguous(), "quadrature_batched: weights must be contiguous");
    TORCH_CHECK(values_in.scalar_type() == at::kFloat || values_in.scalar_type() == at::kDouble,
                "quadrature_batched: values must be float32 or float64");
    TORCH_CHECK(weights_in.scalar_type() == values_in.scalar_type(),
                "quadrature_batched: weights dtype must match values");

    // Normalize shapes: values => [B, N], weights => [N]
    at::Tensor values = values_in;
    if (values.dim() == 3) {
        // [nx, ny, nz] -> [1, N]
        values = values.reshape({1, -1});
    } else if (values.dim() == 4) {
        // [B, nx, ny, nz] -> [B, N]
        values = values.reshape({values.size(0), -1});
    } else if (values.dim() == 1) {
        values = values.reshape({1, values.size(0)});
    } else if (values.dim() == 2) {
        // [B, N]
    } else {
        TORCH_CHECK(false, "quadrature_batched: values must be 1D, 2D, 3D, or 4D tensor");
    }

    at::Tensor weights = weights_in;
    if (weights.dim() == 3) {
        weights = weights.reshape({-1});
    } else if (weights.dim() == 1) {
        // ok
    } else {
        TORCH_CHECK(false, "quadrature_batched: weights must be 1D or 3D tensor");
    }

    const int64_t B = values.size(0);
    const int64_t N = values.size(1);
    TORCH_CHECK(weights.numel() == N, "quadrature_batched: weights numel must equal spatial size of values");

    auto opts = values.options();
    at::Tensor out = at::zeros({B}, opts);

    const int threads = 256;
    const int blocks_x = static_cast<int>((N + threads - 1) / threads);
    const dim3 grid(blocks_x, static_cast<unsigned int>(B), 1);
    const size_t shmem = threads * (values.scalar_type() == at::kFloat ? sizeof(float) : sizeof(double));

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "quadrature_batched_kernel", [&]() {
        quadrature_batched_kernel<scalar_t><<<grid, threads, shmem>>>(
            values.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N,
            B);
    });

    return {out};
}

} // namespace qf
