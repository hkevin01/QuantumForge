#include <torch/extension.h>
#include <vector>
#include <tuple>

namespace qf {
// Forward declaration implemented in fd_gradient_kernel.cu
std::tuple<at::Tensor, at::Tensor>
fd_gradient3d_cuda_impl(const at::Tensor& values,
                        double sx, double sy, double sz,
                        int boundary_mode);
}

namespace {

at::Tensor fd_gradient3d_cuda(at::Tensor values,
                               double sx, double sy, double sz,
                               int64_t boundary_mode) {
  TORCH_CHECK(values.is_cuda(), "fd_gradient3d: values must be CUDA tensor");
  TORCH_CHECK(values.dim() == 3, "fd_gradient3d: values must be [nx, ny, nz]");
  TORCH_CHECK(values.is_contiguous(), "fd_gradient3d: values must be contiguous");
  auto result = qf::fd_gradient3d_cuda_impl(values, sx, sy, sz, static_cast<int>(boundary_mode));
  return std::get<0>(result);
}

} // anonymous namespace

// Register custom operator schema and CUDA implementation
TORCH_LIBRARY(quantumforge_ops, m) {
  m.def("fd_gradient3d(Tensor values, float sx, float sy, float sz, int boundary_mode=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(quantumforge_ops, CUDA, m) {
  m.impl("fd_gradient3d", &fd_gradient3d_cuda);
}
