#include <torch/extension.h>
#include <tuple>

namespace qf {
// Forward declaration implemented in quadrature_batched_kernel.cu
std::tuple<at::Tensor>
quadrature_batched_cuda_impl(const at::Tensor& values,
                             const at::Tensor& weights);
}

namespace {

at::Tensor quadrature_batched_cuda(at::Tensor values,
                                   at::Tensor weights) {
  TORCH_CHECK(values.is_cuda(), "quadrature_batched: values must be CUDA tensor");
  TORCH_CHECK(weights.is_cuda(), "quadrature_batched: weights must be CUDA tensor");
  TORCH_CHECK(values.is_contiguous(), "quadrature_batched: values must be contiguous");
  TORCH_CHECK(weights.is_contiguous(), "quadrature_batched: weights must be contiguous");
  auto result = qf::quadrature_batched_cuda_impl(values, weights);
  return std::get<0>(result);
}

} // anonymous namespace

// Register custom operator schema and CUDA implementation
TORCH_LIBRARY(quantumforge_ops, m) {
  m.def("quadrature_batched(Tensor values, Tensor weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(quantumforge_ops, CUDA, m) {
  m.impl("quadrature_batched", &quadrature_batched_cuda);
}
