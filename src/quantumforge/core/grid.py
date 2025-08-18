"""
Real-space grid management for quantum chemistry calculations.

This module provides classes for handling spatial discretization,
quadrature, and grid operations needed for density functional theory
calculations on real-space grids.
"""

import abc
from typing import Optional, Tuple, Union

import torch


class GridBase(abc.ABC):
    """Abstract base class for real-space grids used in DFT calculations."""

    def __init__(
        self,
        spacing: Union[float, Tuple[float, float, float]],
        origin: Optional[Tuple[float, float, float]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize the grid base class.

        Args:
            spacing: Grid spacing in atomic units (single value or per-axis)
            origin: Grid origin coordinates (default: center at zero)
            device: PyTorch device for tensor operations
            dtype: Data type for grid tensors
        """
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float64

        # Handle spacing specification
        if isinstance(spacing, (int, float)):
            self.spacing: Tuple[float, float, float] = (
                float(spacing), float(spacing), float(spacing)
            )
        else:
            self.spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

        self.origin = origin or (0.0, 0.0, 0.0)

        # Grid properties (to be set by subclasses)
        self.shape: Optional[Tuple[int, int, int]] = None
        self.coordinates: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.volume: Optional[float] = None

    @abc.abstractmethod
    def build_grid(self, **kwargs) -> None:
        """Build the grid coordinates and weights."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_coordinates(self) -> torch.Tensor:
        """Return grid point coordinates as [N, 3] tensor."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_weights(self) -> torch.Tensor:
        """Return integration weights as [N] tensor."""
        raise NotImplementedError

    def to(self, device: torch.device) -> "GridBase":
        """Move grid to specified device."""
        self.device = device
        if self.coordinates is not None:
            self.coordinates = self.coordinates.to(device)
        if self.weights is not None:
            self.weights = self.weights.to(device)
        return self

    def integrate(self, values: torch.Tensor) -> torch.Tensor:
        """Integrate values over the grid using quadrature weights.

        Args:
            values: Function values at grid points [..., N]

        Returns:
            Integrated value(s)
        """
        weights = self.get_weights()
        if values.shape[-1] != weights.shape[0]:
            raise ValueError(
                f"Value shape {values.shape} incompatible with "
                f"grid size {weights.shape[0]}"
            )
        return torch.sum(values * weights, dim=-1)

    def gradient(
        self,
        values: torch.Tensor,
        method: str = "finite_difference"
    ) -> torch.Tensor:
        """Compute gradient of values on the grid.

        Args:
            values: Function values at grid points [..., N]
            method: Gradient computation method

        Returns:
            Gradient as [..., 3, N] tensor
        """
        if method == "finite_difference":
            return self._finite_difference_gradient(values)
        else:
            raise ValueError(f"Unknown gradient method: {method}")

    @abc.abstractmethod
    def _finite_difference_gradient(self, values: torch.Tensor) -> torch.Tensor:
        """Compute finite difference gradient (to be implemented by subclasses)."""
        raise NotImplementedError

    def laplacian(
        self,
        values: torch.Tensor,
        method: str = "finite_difference"
    ) -> torch.Tensor:
        """Compute Laplacian of values on the grid.

        Args:
            values: Function values at grid points [..., N]
            method: Laplacian computation method

        Returns:
            Laplacian as [..., N] tensor
        """
        if method == "finite_difference":
            return self._finite_difference_laplacian(values)
        else:
            raise ValueError(f"Unknown Laplacian method: {method}")

    @abc.abstractmethod
    def _finite_difference_laplacian(self, values: torch.Tensor) -> torch.Tensor:
        """Compute finite difference Laplacian (to be implemented by subclasses)."""
        raise NotImplementedError


class UniformGrid(GridBase):
    """Uniform Cartesian grid for real-space DFT calculations."""

    def __init__(
        self,
        shape: Tuple[int, int, int],
        spacing: Union[float, Tuple[float, float, float]],
        origin: Optional[Tuple[float, float, float]] = None,
        **kwargs
    ):
        """Initialize uniform Cartesian grid.

        Args:
            shape: Number of grid points in each dimension
            spacing: Grid spacing in atomic units
            origin: Grid origin coordinates
        """
        super().__init__(spacing, origin, **kwargs)
        self.shape: Tuple[int, int, int] = (
            int(shape[0]), int(shape[1]), int(shape[2])
        )
        self.build_grid()

    def build_grid(self, **kwargs) -> None:
        """Build uniform grid coordinates and weights."""
        # Create 1D coordinate arrays for each dimension
        x = torch.linspace(
            self.origin[0],
            self.origin[0] + (self.shape[0] - 1) * self.spacing[0],
            self.shape[0],
            device=self.device,
            dtype=self.dtype
        )
        y = torch.linspace(
            self.origin[1],
            self.origin[1] + (self.shape[1] - 1) * self.spacing[1],
            self.shape[1],
            device=self.device,
            dtype=self.dtype
        )
        z = torch.linspace(
            self.origin[2],
            self.origin[2] + (self.shape[2] - 1) * self.spacing[2],
            self.shape[2],
            device=self.device,
            dtype=self.dtype
        )

        # Create meshgrid and flatten to get coordinate array
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        self.coordinates = torch.stack([
            X.flatten(), Y.flatten(), Z.flatten()
        ], dim=1)

        # Uniform weights for rectangular rule integration
        weight = self.spacing[0] * self.spacing[1] * self.spacing[2]
        total_points = self.shape[0] * self.shape[1] * self.shape[2]
        self.weights = torch.full(
            (total_points,),
            weight,
            device=self.device,
            dtype=self.dtype
        )

        # Total volume
        self.volume = float(weight * total_points)

    def get_coordinates(self) -> torch.Tensor:
        """Return grid coordinates as [N, 3] tensor."""
        if self.coordinates is None:
            self.build_grid()
        return self.coordinates

    def get_weights(self) -> torch.Tensor:
        """Return integration weights as [N] tensor."""
        if self.weights is None:
            self.build_grid()
        return self.weights

    def _finite_difference_gradient(
        self, values: torch.Tensor
    ) -> torch.Tensor:
        """Compute finite difference gradient using central differences."""
        # Reshape values to grid shape
        grid_values = values.view(-1, *self.shape)
        grad_x = torch.zeros_like(grid_values)
        grad_y = torch.zeros_like(grid_values)
        grad_z = torch.zeros_like(grid_values)

        # Central differences with boundary handling
        # X direction
        grad_x[:, 1:-1, :, :] = (
            grid_values[:, 2:, :, :] - grid_values[:, :-2, :, :]
        ) / (2 * self.spacing[0])
        grad_x[:, 0, :, :] = (
            grid_values[:, 1, :, :] - grid_values[:, 0, :, :]
        ) / self.spacing[0]
        grad_x[:, -1, :, :] = (
            grid_values[:, -1, :, :] - grid_values[:, -2, :, :]
        ) / self.spacing[0]

        # Y direction
        grad_y[:, :, 1:-1, :] = (
            grid_values[:, :, 2:, :] - grid_values[:, :, :-2, :]
        ) / (2 * self.spacing[1])
        grad_y[:, :, 0, :] = (
            grid_values[:, :, 1, :] - grid_values[:, :, 0, :]
        ) / self.spacing[1]
        grad_y[:, :, -1, :] = (
            grid_values[:, :, -1, :] - grid_values[:, :, -2, :]
        ) / self.spacing[1]

        # Z direction
        grad_z[:, :, :, 1:-1] = (
            grid_values[:, :, :, 2:] - grid_values[:, :, :, :-2]
        ) / (2 * self.spacing[2])
        grad_z[:, :, :, 0] = (
            grid_values[:, :, :, 1] - grid_values[:, :, :, 0]
        ) / self.spacing[2]
        grad_z[:, :, :, -1] = (
            grid_values[:, :, :, -1] - grid_values[:, :, :, -2]
        ) / self.spacing[2]

        # Stack and flatten back to original shape
        gradient = torch.stack([grad_x, grad_y, grad_z], dim=-2)
        return gradient.view(*values.shape[:-1], 3, -1)

    def _finite_difference_laplacian(
        self, values: torch.Tensor
    ) -> torch.Tensor:
        """Compute finite difference Laplacian using second-order stencil."""
        # Reshape values to grid shape
        grid_values = values.view(-1, *self.shape)
        laplacian = torch.zeros_like(grid_values)

        # Second derivatives in each direction
        # X direction
        laplacian[:, 1:-1, :, :] += (
            grid_values[:, 2:, :, :] - 2 * grid_values[:, 1:-1, :, :] +
            grid_values[:, :-2, :, :]
        ) / (self.spacing[0] ** 2)

        # Y direction
        laplacian[:, :, 1:-1, :] += (
            grid_values[:, :, 2:, :] - 2 * grid_values[:, :, 1:-1, :] +
            grid_values[:, :, :-2, :]
        ) / (self.spacing[1] ** 2)

        # Z direction
        laplacian[:, :, :, 1:-1] += (
            grid_values[:, :, :, 2:] - 2 * grid_values[:, :, :, 1:-1] +
            grid_values[:, :, :, :-2]
        ) / (self.spacing[2] ** 2)

        return laplacian.view(*values.shape)

    def __repr__(self) -> str:
        return (
            f"UniformGrid(shape={self.shape}, "
            f"spacing={self.spacing}, origin={self.origin})"
        )


class AdaptiveGrid(GridBase):
    """Adaptive grid with variable spacing around atomic centers."""

    def __init__(
        self,
        atomic_positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        box_size: Tuple[float, float, float],
        base_spacing: float = 0.2,
        refinement_radius: float = 2.0,
        refinement_factor: float = 4.0,
        **kwargs
    ):
        """Initialize adaptive grid around atomic centers.

        Args:
            atomic_positions: Atomic coordinates as [N_atoms, 3] tensor
            atomic_numbers: Atomic numbers as [N_atoms] tensor
            box_size: Simulation box dimensions
            base_spacing: Base grid spacing in regions far from atoms
            refinement_radius: Radius around atoms for grid refinement
            refinement_factor: Factor by which to refine grid near atoms
        """
        super().__init__(base_spacing, **kwargs)
        self.atomic_positions = atomic_positions.to(
            device=self.device, dtype=self.dtype
        )
        self.atomic_numbers = atomic_numbers.to(device=self.device)
        self.box_size = box_size
        self.base_spacing = base_spacing
        self.refinement_radius = refinement_radius
        self.refinement_factor = refinement_factor

    def build_grid(self, **kwargs) -> None:
        """Build adaptive grid with refinement around atomic centers.

        Args:
            **kwargs: Additional arguments including target_points
        """
        target_points = kwargs.get('target_points', None)
        # For now, implement a simplified uniform grid
        # Full adaptive implementation would require more complex algorithms
        if target_points is None:
            # Estimate grid size based on box size and spacing
            nx = int(self.box_size[0] / self.base_spacing)
            ny = int(self.box_size[1] / self.base_spacing)
            nz = int(self.box_size[2] / self.base_spacing)
        else:
            # Distribute points roughly evenly
            n_per_dim = int(target_points ** (1/3))
            nx = ny = nz = n_per_dim

        # Create uniform base grid for now
        # Note: Full adaptive implementation requires more complex algorithms
        # Currently using uniform grid as a starting point
        self.shape = (nx, ny, nz)
        self.spacing = (
            self.box_size[0] / nx,
            self.box_size[1] / ny,
            self.box_size[2] / nz
        )

        # Use uniform grid construction temporarily
        uniform_grid = UniformGrid(
            shape=self.shape,
            spacing=self.spacing,
            origin=self.origin,
            device=self.device,
            dtype=self.dtype
        )
        self.coordinates = uniform_grid.coordinates
        self.weights = uniform_grid.weights
        self.volume = uniform_grid.volume

    def get_coordinates(self) -> torch.Tensor:
        """Return grid coordinates as [N, 3] tensor."""
        if self.coordinates is None:
            self.build_grid()
        return self.coordinates

    def get_weights(self) -> torch.Tensor:
        """Return integration weights as [N] tensor."""
        if self.weights is None:
            self.build_grid()
        return self.weights

    def _finite_difference_gradient(
        self, values: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient (placeholder implementation)."""
        # For adaptive grids, this requires more sophisticated algorithms
        # For now, raise an error indicating this needs implementation
        raise NotImplementedError(
            "Gradient computation for adaptive grids not yet implemented"
        )

    def _finite_difference_laplacian(
        self, values: torch.Tensor
    ) -> torch.Tensor:
        """Compute Laplacian (placeholder implementation)."""
        # For adaptive grids, this requires more sophisticated algorithms
        # For now, raise an error indicating this needs implementation
        raise NotImplementedError(
            "Laplacian computation for adaptive grids not yet implemented"
        )

    def __repr__(self) -> str:
        return (
            f"AdaptiveGrid(n_atoms={len(self.atomic_positions)}, "
            f"box_size={self.box_size}, "
            f"base_spacing={self.base_spacing})"
        )
