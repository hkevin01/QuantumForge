"""
Local Density Approximation (LDA) functionals.

This module implements production-ready LDA exchange and correlation
functionals that form the foundation of density functional theory.
"""

from typing import Optional

import torch

from ..core.functional_base import LDAFunctional


class SlaterExchange(LDAFunctional):
    """
    Slater local density approximation (LDA) exchange functional.

    This is the exact exchange energy density for a uniform electron gas,
    forming the foundation of density functional theory.

    Reference: Slater, J. C. Phys. Rev. 81, 385 (1951)
    """

    def __init__(self, name: str = "Slater"):
        super().__init__(name=name)
        # Slater exchange coefficient: -3/4 * (3/π)^(1/3)
        self.register_buffer('c_x', torch.tensor(-0.7385587663820223))

    def forward(
        self,
        rho: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Slater exchange energy density.        Args:
            rho: Electron density [batch, spin, nx, ny, nz]

        Returns:
            Exchange energy density with same shape as input
        """
        # Slater exchange: E_x = -3/4 * (3/π)^(1/3) * ρ^(4/3)
        return self.c_x * rho**(4.0/3.0)


class VWNCorrelation(LDAFunctional):
    """
    Vosko-Wilk-Nusair (VWN) local density approximation correlation functional.

    This is one of the most widely used LDA correlation functionals,
    providing a parametrization of quantum Monte Carlo results.

    Reference: Vosko, S. H., Wilk, L., & Nusair, M.
               Can. J. Phys. 58, 1200 (1980)
    """

    def __init__(self, name: str = "VWN"):
        super().__init__(name=name)

        # VWN parametrization constants
        # Paramagnetic case
        self.register_buffer('A_p', torch.tensor(0.0310907))
        self.register_buffer('b_p', torch.tensor(3.72744))
        self.register_buffer('c_p', torch.tensor(12.9352))
        self.register_buffer('x0_p', torch.tensor(-0.10498))

        # Ferromagnetic case
        self.register_buffer('A_f', torch.tensor(0.01554535))
        self.register_buffer('b_f', torch.tensor(7.06042))
        self.register_buffer('c_f', torch.tensor(18.0578))
        self.register_buffer('x0_f', torch.tensor(-0.32500))

        # Spin stiffness
        self.register_buffer('A_alpha', torch.tensor(0.0168869))
        self.register_buffer('b_alpha', torch.tensor(1.13107))
        self.register_buffer('c_alpha', torch.tensor(13.0045))
        self.register_buffer('x0_alpha', torch.tensor(-0.00475840))

    def forward(
        self,
        rho: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute VWN correlation energy density.        Args:
            rho: Electron density [batch, spin, nx, ny, nz]

        Returns:
            Correlation energy density
        """
        # Wigner-Seitz radius: rs = (3/(4πρ))^(1/3)
        rs = (3.0 / (4.0 * torch.pi * rho))**(1.0/3.0)
        x = torch.sqrt(rs)

        # For now, implement only the paramagnetic case (spin-unpolarized)
        # Full spin-polarized version requires spin densities

        ec_p = self._vwn_interp(x, self.A_p, self.b_p, self.c_p, self.x0_p)

        return rho * ec_p

    def _vwn_interp(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        x0: torch.Tensor
    ) -> torch.Tensor:
        """VWN interpolation formula."""
        X = x**2 + b*x + c
        X0 = x0**2 + b*x0 + c
        Q = torch.sqrt(4*c - b**2)

        term1 = torch.log(x**2 / X)
        term2 = (2*b / Q) * torch.atan(Q / (2*x + b))
        term3 = -(b*x0 / X0) * (
            torch.log((x - x0)**2 / X) +
            (2*(2*x0 + b) / Q) * torch.atan(Q / (2*x + b))
        )

        return A * (term1 + term2 + term3)
