# CUDA Kernel Development Prompt

You are developing CUDA kernels for high-performance quantum chemistry computations.

## Context

- Project: QuantumForge - GPU-accelerated DFT computations
- Domain: Numerical algorithms for density functional theory
- Technology: CUDA C++ with PyTorch integration

## Coding Standards

- Use CUDA C++17 features
- Follow Google C++ style guide with 4-space indentation
- Use __device__ and __host__ annotations appropriately
- Implement proper error checking with CUDA_CHECK macros
- Add comprehensive comments explaining the algorithm

## Performance Requirements

- Optimize for memory coalescing
- Use shared memory for data reuse
- Implement thread divergence minimization
- Support arbitrary grid sizes and batch processing
- Target 256 threads per block as default

## Integration Requirements

- Use PyTorch tensors as input/output
- Support both float32 and float64 precision
- Implement proper CUDA stream management
- Add Python bindings using pybind11
- Include gradient computation support

## Best Practices

- Use thrust library for complex operations
- Implement bounds checking in debug mode
- Add comprehensive unit tests
- Profile with nsight-compute
- Document computational complexity

## Example Pattern

```cuda
template <typename T>
__global__ void operation_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int grid_size
) {
    // Implementation here
}
```
