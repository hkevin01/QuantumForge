"""
CUDA-accelerated numerical operators for quantum chemistry calculations.

This module provides optimized numerical operators including gradients,
Laplacians, and other differential operators needed for DFT calculations.
"""

import abc
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class NumericalOperatorBase(nn.Module, abc.ABC):
    """Abstract base class for numerical operators."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_cuda: bool = True,
    ):
        """Initialize numerical operator.

        Args:
            device: PyTorch device for computations
            dtype: Data type for tensors
            use_cuda: Whether to use CUDA acceleration when available
        """
        super().__init__()
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() and use_cuda
            else torch.device("cpu")
        )
        self.dtype = dtype or torch.float64
        self.use_cuda = use_cuda and torch.cuda.is_available()

    @abc.abstractmethod
    def forward(self, values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the numerical operator to input values."""
        raise NotImplementedError

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the appropriate device with correct dtype."""
        return tensor.to(device=self.device, dtype=self.dtype)


class FiniteDifferenceGradient(NumericalOperatorBase):
    """Finite difference gradient operator for structured grids."""

    def __init__(
        self,
        spacing: Union[float, Tuple[float, float, float]],
        boundary: str = "periodic",
        **kwargs
    ):
        """Initialize finite difference gradient operator.

        Args:
            spacing: Grid spacing (uniform or per-axis)
            boundary: Boundary condition handling ('periodic', 'zero')
        """
        super().__init__(**kwargs)

        if isinstance(spacing, (int, float)):
            self.spacing = (
                float(spacing), float(spacing), float(spacing)
            )
        else:
            self.spacing = (
                float(spacing[0]), float(spacing[1]), float(spacing[2])
            )

        self.boundary = boundary

    def forward(
        self,
        values: torch.Tensor,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute gradient using finite differences.

        Args:
            values: Input values [..., N] or [..., nx, ny, nz]
            grid_shape: Shape of structured grid if values are flattened

        Returns:
            Gradient tensor [..., 3, N] or [..., 3, nx, ny, nz]
        """
        values = self.to_device(values)

        # Handle grid shape
        if grid_shape is not None and values.dim() >= 1:
            total_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
            if values.shape[-1] == total_points:
                # Reshape flattened grid to structured form
                batch_shape = values.shape[:-1]
                values = values.view(*batch_shape, *grid_shape)

        if values.dim() < 3:
            raise ValueError(
                f"Expected at least 3D tensor for structured grid, "
                f"got {values.dim()}D"
            )

        return self._pytorch_gradient(values)

    def _pytorch_gradient(self, values: torch.Tensor) -> torch.Tensor:
        """Compute gradient using PyTorch operations."""
        # Ensure values are at least 3D for spatial dimensions
        spatial_dims = values.shape[-3:]
        batch_shape = values.shape[:-3]

        # Initialize gradient output
        grad = torch.zeros(
            *batch_shape, 3, *spatial_dims,
            device=values.device,
            dtype=values.dtype
        )

        # X-direction gradient (central differences)
        if spatial_dims[0] > 1:
            grad[..., 0, 1:-1, :, :] = (
                values[..., 2:, :, :] - values[..., :-2, :, :]
            ) / (2 * self.spacing[0])

            # Boundary conditions
            if self.boundary == "periodic":
                grad[..., 0, 0, :, :] = (
                    values[..., 1, :, :] - values[..., -1, :, :]
                ) / (2 * self.spacing[0])
                grad[..., 0, -1, :, :] = (
                    values[..., 0, :, :] - values[..., -2, :, :]
                ) / (2 * self.spacing[0])
            elif self.boundary == "zero":
                grad[..., 0, 0, :, :] = (
                    values[..., 1, :, :] / self.spacing[0]
                )
                grad[..., 0, -1, :, :] = (
                    -values[..., -2, :, :] / self.spacing[0]
                )

        # Y-direction gradient
        if spatial_dims[1] > 1:
            grad[..., 1, :, 1:-1, :] = (
                values[..., :, 2:, :] - values[..., :, :-2, :]
            ) / (2 * self.spacing[1])

            if self.boundary == "periodic":
                grad[..., 1, :, 0, :] = (
                    values[..., :, 1, :] - values[..., :, -1, :]
                ) / (2 * self.spacing[1])
                grad[..., 1, :, -1, :] = (
                    values[..., :, 0, :] - values[..., :, -2, :]
                ) / (2 * self.spacing[1])

        # Z-direction gradient
        if spatial_dims[2] > 1:
            grad[..., 2, :, :, 1:-1] = (
                values[..., :, :, 2:] - values[..., :, :, :-2]
            ) / (2 * self.spacing[2])

            if self.boundary == "periodic":
                grad[..., 2, :, :, 0] = (
                    values[..., :, :, 1] - values[..., :, :, -1]
                ) / (2 * self.spacing[2])
                grad[..., 2, :, :, -1] = (
                    values[..., :, :, 0] - values[..., :, :, -2]
                ) / (2 * self.spacing[2])

        return grad


class FiniteDifferenceLaplacian(NumericalOperatorBase):
    """Finite difference Laplacian operator for structured grids."""

    def __init__(
        self,
        spacing: Union[float, Tuple[float, float, float]],
        boundary: str = "periodic",
        **kwargs
    ):
        """Initialize finite difference Laplacian operator.

        Args:
            spacing: Grid spacing (uniform or per-axis)
            boundary: Boundary condition handling
        """
        super().__init__(**kwargs)

        if isinstance(spacing, (int, float)):
            self.spacing = (
                float(spacing), float(spacing), float(spacing)
            )
        else:
            self.spacing = (
                float(spacing[0]), float(spacing[1]), float(spacing[2])
            )

        self.boundary = boundary

    def forward(
        self,
        values: torch.Tensor,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute Laplacian using finite differences.

        Args:
            values: Input values [..., N] or [..., nx, ny, nz]
            grid_shape: Shape of structured grid if values are flattened

        Returns:
            Laplacian tensor with same shape as input
        """
        values = self.to_device(values)

        # Handle grid shape
        if grid_shape is not None and values.dim() >= 1:
            total_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
            if values.shape[-1] == total_points:
                batch_shape = values.shape[:-1]
                values = values.view(*batch_shape, *grid_shape)

        if values.dim() < 3:
            raise ValueError(
                f"Expected at least 3D tensor for structured grid, "
                f"got {values.dim()}D"
            )

        return self._pytorch_laplacian(values)

    def _pytorch_laplacian(self, values: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using PyTorch operations."""
        spatial_dims = values.shape[-3:]
        laplacian = torch.zeros_like(values)

        # Second derivatives in each direction
        # X-direction: d²/dx²
        if spatial_dims[0] > 2:
            laplacian[..., 1:-1, :, :] += (
                values[..., 2:, :, :] - 2 * values[..., 1:-1, :, :] +
                values[..., :-2, :, :]
            ) / (self.spacing[0] ** 2)

        # Y-direction: d²/dy²
        if spatial_dims[1] > 2:
            laplacian[..., :, 1:-1, :] += (
                values[..., :, 2:, :] - 2 * values[..., :, 1:-1, :] +
                values[..., :, :-2, :]
            ) / (self.spacing[1] ** 2)

        # Z-direction: d²/dz²
        if spatial_dims[2] > 2:
            laplacian[..., :, :, 1:-1] += (
                values[..., :, :, 2:] - 2 * values[..., :, :, 1:-1] +
                values[..., :, :, :-2]
            ) / (self.spacing[2] ** 2)

        return laplacian


class SpectralOperators(NumericalOperatorBase):
    """Spectral (FFT-based) differential operators for periodic systems."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        domain_size: Tuple[float, float, float],
        **kwargs
    ):
        """Initialize spectral operators.

        Args:
            grid_shape: Number of grid points in each dimension
            domain_size: Physical size of the domain
        """
        super().__init__(**kwargs)
        self.grid_shape = grid_shape
        self.domain_size = domain_size

        # Compute k-space grids
        self._setup_k_grids()

    def _setup_k_grids(self) -> None:
        """Set up k-space grids for spectral derivatives."""
        kx = 2 * torch.pi * torch.fft.fftfreq(
            self.grid_shape[0], self.domain_size[0] / self.grid_shape[0]
        )
        ky = 2 * torch.pi * torch.fft.fftfreq(
            self.grid_shape[1], self.domain_size[1] / self.grid_shape[1]
        )
        kz = 2 * torch.pi * torch.fft.fftfreq(
            self.grid_shape[2], self.domain_size[2] / self.grid_shape[2]
        )

        # Create 3D k-grids
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')

        # Register as buffers
        self.register_buffer('kx_grid', KX.to(dtype=self.dtype))
        self.register_buffer('ky_grid', KY.to(dtype=self.dtype))
        self.register_buffer('kz_grid', KZ.to(dtype=self.dtype))
        k2_grid = (KX**2 + KY**2 + KZ**2).to(dtype=self.dtype)
        self.register_buffer('k2_grid', k2_grid)

    def forward(
        self,
        values: torch.Tensor,
        operator: str = "laplacian",
        **kwargs
    ) -> torch.Tensor:
        """Apply spectral operator.

        Args:
            values: Input values [..., nx, ny, nz]
            operator: Type of operator ('gradient', 'laplacian')

        Returns:
            Result of spectral operation
        """
        values = self.to_device(values)

        if operator == "laplacian":
            return self._spectral_laplacian(values)
        elif operator == "gradient":
            return self._spectral_gradient(values)
        else:
            raise ValueError(f"Unknown spectral operator: {operator}")

    def _spectral_laplacian(self, values: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using FFT."""
        # FFT
        values_k = torch.fft.fftn(values, dim=(-3, -2, -1))

        # Apply Laplacian in k-space (multiply by -k²)
        laplacian_k = -self.k2_grid * values_k

        # Inverse FFT
        laplacian = torch.fft.ifftn(laplacian_k, dim=(-3, -2, -1)).real

        return laplacian

    def _spectral_gradient(self, values: torch.Tensor) -> torch.Tensor:
        """Compute gradient using FFT."""
        # FFT
        values_k = torch.fft.fftn(values, dim=(-3, -2, -1))

        # Apply gradient in k-space (multiply by ik)
        grad_x_k = 1j * self.kx_grid * values_k
        grad_y_k = 1j * self.ky_grid * values_k
        grad_z_k = 1j * self.kz_grid * values_k

        # Inverse FFT
        grad_x = torch.fft.ifftn(grad_x_k, dim=(-3, -2, -1)).real
        grad_y = torch.fft.ifftn(grad_y_k, dim=(-3, -2, -1)).real
        grad_z = torch.fft.ifftn(grad_z_k, dim=(-3, -2, -1)).real

        # Stack components
        gradient = torch.stack([grad_x, grad_y, grad_z], dim=-4)

        return gradient
