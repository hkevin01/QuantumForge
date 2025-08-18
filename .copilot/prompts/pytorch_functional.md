# PyTorch Functional Development Prompt

You are developing PyTorch-based density functionals for quantum chemistry applications.

## Context
- Project: QuantumForge - GPU-accelerated DFT with ML functionals
- Domain: Density Functional Theory (DFT) and quantum chemistry
- Framework: PyTorch with CUDA acceleration

## Coding Standards
- Use PyTorch tensors for all computations
- Implement forward() and backward() methods for custom functions
- Support both CPU and CUDA tensors
- Include comprehensive docstrings with mathematical formulas
- Add type hints for all function parameters and returns
- Use automatic mixed precision (AMP) when beneficial

## Functional Requirements
- Input: electron density ρ(r) and its gradients ∇ρ(r)
- Output: exchange-correlation energy density ε_xc and potential v_xc
- Support batch processing for multiple molecules
- Ensure energy conservation and size consistency
- Implement proper derivative calculations for forces

## Best Practices
- Use torch.jit.script for performance-critical functions
- Implement gradient checkpointing for memory efficiency
- Add numerical stability checks (avoid division by zero)
- Use torch.autograd for automatic differentiation
- Include unit tests comparing against analytical derivatives

## Example Pattern
```python
class FunctionalBase(nn.Module):
    def forward(self, rho: torch.Tensor, grad_rho: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Implementation here
        return {"eps_xc": eps_xc, "v_xc": v_xc}
```
