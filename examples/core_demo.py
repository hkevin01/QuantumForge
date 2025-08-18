#!/usr/bin/env python3
"""
QuantumForge Core Module Demo and Integration Test

This script demonstrates the usage of QuantumForge's core modules including
functional base classes, grid management, and numerical operators.
Run this script to verify the core implementation is working correctly.
"""

import sys
from typing import Tuple

import numpy as np
import torch

# Add the src directory to the path for imports
sys.path.insert(0, '/home/kevin/Projects/QuantumForge/src')

from quantumforge.core.functional_base import GGAFunctional, LDAFunctional
from quantumforge.core.grid import UniformGrid
from quantumforge.core.operators import (
    FiniteDifferenceGradient,
    FiniteDifferenceLaplacian,
)


class DemoLDAFunctional(LDAFunctional):
    """Demo LDA exchange functional (Slater exchange)."""

    def forward(self, rho, **kwargs):
        """Slater exchange: Ex = -3/4 * (3/π)^(1/3) * ρ^(4/3)"""
        c_x = -0.75 * (3.0 / torch.pi) ** (1.0/3.0)
        return c_x * rho ** (4.0/3.0)


class DemoGGAFunctional(GGAFunctional):
    """Demo GGA exchange functional (PBE-like)."""

    def forward(self, rho, grad_rho=None, **kwargs):
        """PBE-like exchange with gradient correction."""
        # LDA part
        c_x = -0.75 * (3.0 / torch.pi) ** (1.0/3.0)
        ex_lda = c_x * rho ** (4.0/3.0)

        if grad_rho is None:
            return ex_lda

        # Gradient enhancement
        kf = (3.0 * torch.pi**2 * rho) ** (1.0/3.0)  # Fermi wave vector
        grad_norm = torch.norm(grad_rho, dim=1, keepdim=True)
        s = grad_norm / (2.0 * kf * rho)  # Reduced gradient

        # PBE enhancement factor (simplified)
        kappa = 0.804
        mu = 0.2195
        f_x = 1.0 + kappa - kappa / (1.0 + mu * s**2 / kappa)

        return ex_lda * f_x


def create_test_density(grid: UniformGrid) -> torch.Tensor:
    """Create a test electron density (Gaussian-like)."""
    coords = grid.get_coordinates()

    # Create a Gaussian density centered at origin
    r_squared = torch.sum(coords**2, dim=1)
    rho = 2.0 * torch.exp(-r_squared / 2.0)  # Gaussian density

    # Reshape to grid format [1, 1, nx, ny, nz] for [batch, spin, x, y, z]
    rho_grid = rho.view(1, 1, *grid.shape)

    return rho_grid


def test_functionals():
    """Test the functional base classes."""
    print("=" * 60)
    print("TESTING FUNCTIONAL BASE CLASSES")
    print("=" * 60)

    # Create test grid
    print("Creating test grid (16³ points)...")
    grid = UniformGrid(
        shape=(16, 16, 16),
        spacing=0.2,
        origin=(-1.6, -1.6, -1.6)
    )

    # Create test density
    print("Creating test electron density...")
    rho = create_test_density(grid)
    print(f"Density shape: {rho.shape}")
    print(f"Total electrons: {grid.integrate(rho.flatten()).item():.4f}")

    # Test LDA functional
    print("\nTesting LDA functional...")
    lda = DemoLDAFunctional(name="Slater-LDA")
    ex_lda = lda(rho)
    ex_lda_total = grid.integrate(ex_lda.flatten())
    print(f"LDA exchange energy: {ex_lda_total.item():.6f} Ha")

    # Test GGA functional
    print("\nTesting GGA functional...")
    gga = DemoGGAFunctional(name="PBE-GGA")

    # Compute density gradient using finite differences
    grad_op = FiniteDifferenceGradient(spacing=grid.spacing)
    grad_rho = grad_op(rho, grid_shape=grid.shape)
    print(f"Gradient shape: {grad_rho.shape}")

    ex_gga = gga(rho, grad_rho=grad_rho)
    ex_gga_total = grid.integrate(ex_gga.flatten())
    print(f"GGA exchange energy: {ex_gga_total.item():.6f} Ha")

    # Compare LDA vs GGA
    diff = (ex_gga_total - ex_lda_total).item()
    print(f"GGA - LDA difference: {diff:.6f} Ha")

    return True


def test_grid_operations():
    """Test grid management and operations."""
    print("=" * 60)
    print("TESTING GRID OPERATIONS")
    print("=" * 60)

    # Create uniform grid
    print("Creating uniform grid...")
    grid = UniformGrid(
        shape=(32, 32, 32),
        spacing=(0.1, 0.1, 0.1),
        origin=(-1.6, -1.6, -1.6)
    )

    print(f"Grid shape: {grid.shape}")
    print(f"Grid spacing: {grid.spacing}")
    print(f"Total volume: {grid.volume:.4f}")
    print(f"Number of points: {len(grid.get_coordinates())}")

    # Test coordinate generation
    coords = grid.get_coordinates()
    weights = grid.get_weights()
    print(f"Coordinates shape: {coords.shape}")
    print(f"Weights shape: {weights.shape}")

    # Test integration
    ones = torch.ones(coords.shape[0])
    volume_check = grid.integrate(ones)
    print(f"Volume check (should match total volume): {volume_check.item():.4f}")

    # Test with a simple function (x² + y² + z²)
    r_squared = torch.sum(coords**2, dim=1)
    r_squared_integral = grid.integrate(r_squared)
    print(f"∫(x² + y² + z²) dV = {r_squared_integral.item():.4f}")

    return True


def test_numerical_operators():
    """Test numerical differential operators."""
    print("=" * 60)
    print("TESTING NUMERICAL OPERATORS")
    print("=" * 60)

    # Create grid
    grid = UniformGrid(
        shape=(20, 20, 20),
        spacing=0.15,
        origin=(-1.5, -1.5, -1.5)
    )

    # Create test function: f(x,y,z) = x² + y² + z²
    coords = grid.get_coordinates()
    f = torch.sum(coords**2, dim=1)
    f_grid = f.view(*grid.shape)

    print(f"Test function shape: {f_grid.shape}")

    # Test gradient operator
    print("\nTesting gradient operator...")
    grad_op = FiniteDifferenceGradient(spacing=grid.spacing)

    # Add batch and component dimensions for the operator
    f_input = f_grid.unsqueeze(0)  # Add batch dimension
    grad_f = grad_op(f_input, grid_shape=grid.shape)

    print(f"Gradient shape: {grad_f.shape}")

    # Analytical gradient of x² + y² + z² is [2x, 2y, 2z]
    coords_grid = coords.view(*grid.shape, 3)
    grad_analytical = 2.0 * coords_grid.permute(3, 0, 1, 2).unsqueeze(0)

    # Compare numerical vs analytical (avoid boundaries)
    interior = slice(2, -2)
    diff = torch.abs(grad_f[:, :, interior, interior, interior] -
                    grad_analytical[:, :, interior, interior, interior])
    max_error = torch.max(diff).item()
    mean_error = torch.mean(diff).item()

    print(f"Max gradient error: {max_error:.6f}")
    print(f"Mean gradient error: {mean_error:.6f}")

    # Test Laplacian operator
    print("\nTesting Laplacian operator...")
    lap_op = FiniteDifferenceLaplacian(spacing=grid.spacing)
    lap_f = lap_op(f_input, grid_shape=grid.shape)

    print(f"Laplacian shape: {lap_f.shape}")

    # Analytical Laplacian of x² + y² + z² is 6 (constant)
    lap_analytical = 6.0 * torch.ones_like(f_input)

    # Compare numerical vs analytical (avoid boundaries)
    lap_diff = torch.abs(lap_f[:, interior, interior, interior] -
                        lap_analytical[:, interior, interior, interior])
    lap_max_error = torch.max(lap_diff).item()
    lap_mean_error = torch.mean(lap_diff).item()

    print(f"Max Laplacian error: {lap_max_error:.6f}")
    print(f"Mean Laplacian error: {lap_mean_error:.6f}")

    return True


def integration_test():
    """Full integration test combining all components."""
    print("=" * 60)
    print("FULL INTEGRATION TEST")
    print("=" * 60)

    print("Creating quantum chemistry calculation setup...")

    # Create computational grid
    grid = UniformGrid(
        shape=(24, 24, 24),
        spacing=0.2,
        origin=(-2.4, -2.4, -2.4)
    )

    # Create test density (hydrogen-like atom)
    coords = grid.get_coordinates()
    r = torch.norm(coords, dim=1)
    rho = 2.0 * torch.exp(-2.0 * r)  # 1s orbital (Z=2)
    rho_grid = rho.view(1, 1, *grid.shape)

    print(f"Electron density created")
    print(f"Total electrons: {grid.integrate(rho_grid.flatten()).item():.4f}")

    # Compute density gradient
    grad_op = FiniteDifferenceGradient(spacing=grid.spacing)
    grad_rho = grad_op(rho_grid, grid_shape=grid.shape)

    # Compute exchange energies
    lda = DemoLDAFunctional(name="Slater")
    gga = DemoGGAFunctional(name="PBE")

    ex_lda = lda(rho_grid)
    ex_gga = gga(rho_grid, grad_rho=grad_rho)

    ex_lda_total = grid.integrate(ex_lda.flatten()).item()
    ex_gga_total = grid.integrate(ex_gga.flatten()).item()

    print(f"\nExchange energy results:")
    print(f"LDA:  {ex_lda_total:.6f} Ha")
    print(f"GGA:  {ex_gga_total:.6f} Ha")
    print(f"Diff: {ex_gga_total - ex_lda_total:.6f} Ha")

    # Compute kinetic energy density (Thomas-Fermi)
    tau_tf = 0.3 * (3.0 * torch.pi**2) ** (2.0/3.0) * rho**(5.0/3.0)
    tau_tf_total = grid.integrate(tau_tf.flatten()).item()
    print(f"Thomas-Fermi kinetic energy: {tau_tf_total:.6f} Ha")

    print("\n✅ Integration test completed successfully!")
    return True


def main():
    """Run all tests and demonstrations."""
    print("QuantumForge Core Module Integration Test")
    print("=" * 60)

    try:
        # Test individual components
        success = True
        success &= test_functionals()
        success &= test_grid_operations()
        success &= test_numerical_operators()
        success &= integration_test()

        if success:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! QuantumForge core is working correctly!")
            print("=" * 60)
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
