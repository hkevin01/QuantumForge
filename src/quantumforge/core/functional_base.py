"""
Core abstractions for QuantumForge density functionals.

This module provides the base classes and interfaces for implementing
density functionals in QuantumForge.
"""

import abc
from typing import Dict, Optional

import torch
import torch.nn as nn


class FunctionalBase(nn.Module, abc.ABC):
    """
    Abstract base class for all density functionals in QuantumForge.

    This class defines the interface that all density functionals must
    implement. It provides the basic structure for forward pass, gradient
    computation, and energy evaluation.

    Attributes:
        name: Human-readable name of the functional
        description: Detailed description of the functional
        supports_gga: Whether the functional supports generalized gradient
            approximation
        supports_meta: Whether the functional supports meta-GGA features
    """

    def __init__(
        self,
        name: str = "Unknown",
        description: str = "",
        supports_gga: bool = True,
        supports_meta: bool = False
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.supports_gga = supports_gga
        self.supports_meta = supports_meta

    @abc.abstractmethod
    def forward(
        self,
        rho: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute exchange-correlation energy density and potential.

        Args:
            rho: Electron density tensor of shape (batch, 1, nx, ny, nz)
            grad_rho: Density gradient tensor of shape (batch, 3, nx, ny, nz)
            tau: Kinetic energy density (for meta-GGA) of shape
                (batch, 1, nx, ny, nz)
            **kwargs: Additional functional-specific parameters

        Returns:
            Dictionary containing:
                - 'eps_xc': Exchange-correlation energy density per particle
                - 'v_xc': Exchange-correlation potential
                - 'dedgrho': Derivative w.r.t. gradient (for GGA)
                - 'dedtau': Derivative w.r.t. tau (for meta-GGA)
        """
        raise NotImplementedError

    def compute_energy(
        self,
        rho: torch.Tensor,
        weights: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute total exchange-correlation energy.

        Args:
            rho: Electron density
            weights: Quadrature weights for integration
            grad_rho: Density gradient (optional)
            tau: Kinetic energy density (optional)
            **kwargs: Additional parameters

        Returns:
            Total exchange-correlation energy
        """
        result = self.forward(rho, grad_rho, tau, **kwargs)
        eps_xc = result['eps_xc']

        # Integrate: E_xc = ∫ ρ(r) ε_xc(r) dr
        energy = torch.sum(rho * eps_xc * weights, dim=(-3, -2, -1))
        return energy

    def check_inputs(
        self,
        rho: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None
    ) -> None:
        """Validate input tensors for correctness."""
        if rho.dim() != 5:
            raise ValueError(
                f"Expected rho to have 5 dimensions, got {rho.dim()}"
            )

        if grad_rho is not None:
            if not self.supports_gga:
                raise ValueError(
                    f"Functional {self.name} does not support GGA"
                )
            if grad_rho.shape[1] != 3:
                raise ValueError(
                    f"Expected grad_rho to have 3 components, "
                    f"got {grad_rho.shape[1]}"
                )

        if tau is not None:
            if not self.supports_meta:
                raise ValueError(
                    f"Functional {self.name} does not support meta-GGA"
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class LDAFunctional(FunctionalBase):
    """Base class for Local Density Approximation (LDA) functionals."""

    def __init__(self, name: str = "LDA", **kwargs):
        super().__init__(
            name=name, supports_gga=False, supports_meta=False, **kwargs
        )


class GGAFunctional(FunctionalBase):
    """Base class for Generalized Gradient Approximation (GGA) functionals."""

    def __init__(self, name: str = "GGA", **kwargs):
        super().__init__(
            name=name, supports_gga=True, supports_meta=False, **kwargs
        )


class MetaGGAFunctional(FunctionalBase):
    """Base class for meta-GGA functionals."""

    def __init__(self, name: str = "Meta-GGA", **kwargs):
        super().__init__(
            name=name, supports_gga=True, supports_meta=True, **kwargs
        )


class HybridFunctional(FunctionalBase):
    """Base class for hybrid functionals with exact exchange."""

    def __init__(
        self,
        name: str = "Hybrid",
        exact_exchange_fraction: float = 0.25,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.exact_exchange_fraction = exact_exchange_fraction

    @abc.abstractmethod
    def compute_exact_exchange(
        self,
        density_matrix: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute exact exchange energy contribution."""
        raise NotImplementedError
