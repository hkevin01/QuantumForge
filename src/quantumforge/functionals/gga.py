"""
Generalized Gradient Approximation (GGA) functionals.

This module implements GGA exchange and correlation functionals that include
gradient corrections to improve upon LDA.
"""

from typing import Optional

import torch

from ..core.functional_base import GGAFunctional
from .lda import VWNCorrelation


class PBEExchange(GGAFunctional):
    """
    Perdew-Burke-Ernzerhof (PBE) GGA exchange functional.

    One of the most successful and widely used GGA functionals,
    providing excellent performance for a wide range of systems.

    Reference: Perdew, J. P., Burke, K., & Ernzerhof, M.
               Phys. Rev. Lett. 77, 3865 (1996)
    """

    def __init__(self, name: str = "PBE"):
        super().__init__(name=name)

        # PBE exchange parameters
        self.register_buffer('kappa', torch.tensor(0.804))
        self.register_buffer('mu', torch.tensor(0.2195149727645171))

        # Slater exchange coefficient
        self.register_buffer('c_x', torch.tensor(-0.7385587663820223))

    def forward(
        self,
        rho: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute PBE exchange energy density.

        Args:
            rho: Electron density [batch, spin, nx, ny, nz]
            grad_rho: Density gradient [batch, 3, nx, ny, nz]
            tau: Kinetic energy density (unused for GGA)

        Returns:
            Exchange energy density
        """
        # LDA exchange energy density
        ex_lda = self.c_x * rho**(4.0/3.0)

        if grad_rho is None:
            return ex_lda

        # Fermi wave vector: kf = (3π²ρ)^(1/3)
        kf = (3.0 * torch.pi**2 * rho)**(1.0/3.0)

        # Gradient magnitude
        grad_norm = torch.norm(grad_rho, dim=1, keepdim=True)

        # Reduced gradient: s = |∇ρ|/(2kf*ρ)
        s = grad_norm / (2.0 * kf * rho)

        # PBE enhancement factor
        fx = 1.0 + self.kappa - self.kappa / (
            1.0 + self.mu * s**2 / self.kappa
        )

        return ex_lda * fx


class PBECorrelation(GGAFunctional):
    """
    Perdew-Burke-Ernzerhof (PBE) correlation functional.

    Gradient-corrected correlation functional that complements PBE exchange.

    Reference: Perdew, J. P., Burke, K., & Ernzerhof, M.
               Phys. Rev. Lett. 77, 3865 (1996)
    """

    def __init__(self, name: str = "PBE-corr"):
        super().__init__(name=name)

        # Initialize VWN for LDA correlation
        self.vwn = VWNCorrelation()

        # PBE correlation parameters
        self.register_buffer('beta', torch.tensor(0.06672455060314922))
        self.register_buffer('gamma', torch.tensor(0.031090690869654895))

    def forward(
        self,
        rho: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute PBE correlation energy density.

        Args:
            rho: Electron density [batch, spin, nx, ny, nz]
            grad_rho: Density gradient [batch, 3, nx, ny, nz]
            tau: Kinetic energy density (unused for GGA)

        Returns:
            Correlation energy density
        """
        # LDA correlation from VWN
        ec_lda_total = self.vwn(rho)
        ec_lda = ec_lda_total / rho  # Get energy density per particle

        if grad_rho is None:
            return ec_lda_total

        # Wigner-Seitz radius
        rs = (3.0 / (4.0 * torch.pi * rho))**(1.0/3.0)

        # Gradient magnitude
        grad_norm = torch.norm(grad_rho, dim=1, keepdim=True)

        # t parameter: t = |∇ρ|/(2*sqrt(4π)*sqrt(rs)*ρ)
        t = grad_norm / (
            2.0 * torch.sqrt(torch.tensor(4.0*torch.pi)) * torch.sqrt(rs) * rho
        )

        # A parameter (avoid division by zero)
        exp_term = torch.exp(-torch.abs(ec_lda) / self.gamma)
        A = self.beta / self.gamma / torch.clamp(exp_term - 1.0, min=1e-12)

        # H function
        t2 = t**2
        denominator = 1.0 + A * t2 + A**2 * t2**2
        H = self.gamma * torch.log(
            1.0 + self.beta * t2 / self.gamma *
            (1.0 + A * t2) / denominator
        )

        return rho * (ec_lda + H)


class BLYPExchange(GGAFunctional):
    """
    Becke88 exchange functional (part of BLYP).

    Reference: Becke, A. D. Phys. Rev. A 38, 3098 (1988)
    """

    def __init__(self, name: str = "Becke88"):
        super().__init__(name=name)

        # Becke88 parameter
        self.register_buffer('beta', torch.tensor(0.0042))

        # Slater exchange coefficient
        self.register_buffer('c_x', torch.tensor(-0.7385587663820223))

    def forward(
        self,
        rho: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute Becke88 exchange energy density."""
        # LDA exchange energy density
        ex_lda = self.c_x * rho**(4.0/3.0)

        if grad_rho is None:
            return ex_lda

        # Gradient magnitude
        grad_norm = torch.norm(grad_rho, dim=1, keepdim=True)

        # Reduced gradient
        rho_4_3 = rho**(4.0/3.0)
        x = grad_norm / rho_4_3

        # Becke88 enhancement factor
        x2 = x**2
        sinh_term = torch.asinh(x)
        fx = 1.0 - self.beta * x2 / (1.0 + 6.0 * self.beta * x * sinh_term)

        return ex_lda * fx


class LYPCorrelation(GGAFunctional):
    """
    Lee-Yang-Parr (LYP) correlation functional.

    Reference: Lee, C., Yang, W., & Parr, R. G.
               Phys. Rev. B 37, 785 (1988)
    """

    def __init__(self, name: str = "LYP"):
        super().__init__(name=name)

        # LYP parameters
        self.register_buffer('a', torch.tensor(0.04918))
        self.register_buffer('b', torch.tensor(0.132))
        self.register_buffer('c', torch.tensor(0.2533))
        self.register_buffer('d', torch.tensor(0.349))

    def forward(
        self,
        rho: torch.Tensor,
        grad_rho: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute LYP correlation energy density.

        Note: This is a simplified implementation for spin-unpolarized case.
        """
        if grad_rho is None:
            # Without gradients, return zero correlation
            return torch.zeros_like(rho)

        # Gradient magnitude
        grad_norm = torch.norm(grad_rho, dim=1, keepdim=True)

        # Parameters
        gamma = 2.0 * (1.0 - torch.pi**2 / 3.0)

        # Simplified LYP expression (spin-unpolarized)
        rho_inv = 1.0 / torch.clamp(rho, min=1e-12)

        # Leading terms
        term1 = -self.a * rho / (1.0 + self.d * rho**(-1.0/3.0))
        term2 = -self.a * self.b * gamma * rho**(5.0/3.0) * grad_norm**2
        term3 = term2 * rho_inv**2

        return term1 + term2 + term3


class BLYP(GGAFunctional):
    """
    BLYP (Becke-Lee-Yang-Parr) functional combining Becke88 exchange and LYP correlation.

    This functional combines:
    - Becke88 exchange functional
    - Lee-Yang-Parr (LYP) correlation functional

    Reference:
    Becke, A. D. Phys. Rev. A 38, 3098 (1988)
    Lee, C., Yang, W., & Parr, R. G. Phys. Rev. B 37, 785 (1988)
    """

    def __init__(self, name: str = "BLYP"):
        super().__init__(name=name)
        self.exchange = BLYPExchange()
        self.correlation = LYPCorrelation()

    def forward(self, rho: torch.Tensor, grad_rho: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute BLYP energy density.

        Args:
            rho: Electron density tensor
            grad_rho: Gradient of electron density
            tau: Kinetic energy density (unused for GGA)

        Returns:
            BLYP energy density tensor
        """
        ex = self.exchange(rho, grad_rho)
        ec = self.correlation(rho, grad_rho)
        return ex + ec
