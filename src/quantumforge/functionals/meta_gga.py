"""
Meta-GGA functionals for density functional theory.

This module implements meta-GGA functionals that depend on the kinetic energy density
in addition to the electron density and its gradient.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..core.functional_base import DFTFunctional


class SCANExchange(DFTFunctional):
    """
    SCAN (Strongly Constrained and Appropriately Normed) exchange functional.

    A meta-GGA exchange functional that incorporates the kinetic energy density
    and satisfies many known exact constraints.

    Reference:
    Sun, J., Ruzsinszky, A., & Perdew, J. P. (2015).
    Strongly constrained and appropriately normed semilocal density functional.
    Physical review letters, 115(3), 036402.
    """

    def __init__(self):
        super().__init__()
        self.name = "SCAN_X"

    def forward(self, rho: torch.Tensor, grad_rho: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute SCAN exchange energy density.

        Args:
            rho: Electron density tensor
            grad_rho: Gradient of electron density
            tau: Kinetic energy density

        Returns:
            Exchange energy density tensor
        """
        if grad_rho is None or tau is None:
            raise ValueError("SCAN exchange requires both gradient and kinetic energy density")

        # LDA exchange coefficient
        C_x = -3.0/(4.0*torch.pi) * (3.0*torch.pi**2)**(1./3.)

        # Reduced gradient and iso-orbital indicator
        rs = (3.0/(4.0*torch.pi*rho))**(1./3.)
        s = torch.norm(grad_rho, dim=-1, keepdim=True) / (2.0 * (3.0*torch.pi**2)**(1./3.) * rho**(4./3.))

        # Kinetic energy density ratio
        tau_w = torch.norm(grad_rho, dim=-1, keepdim=True)**2 / (8.0 * rho)
        alpha = (tau - tau_w) / tau_w

        # SCAN exchange enhancement factor (simplified)
        # This is a simplified version - full SCAN is much more complex
        h_x = 1.0 + 0.1 * s**2 / (1.0 + 0.1 * s**2) + 0.05 * alpha

        ex_lda = C_x * rho**(4./3.)
        return ex_lda * h_x


class SCANCorrelation(DFTFunctional):
    """
    SCAN (Strongly Constrained and Appropriately Normed) correlation functional.

    A meta-GGA correlation functional that incorporates the kinetic energy density
    and satisfies many known exact constraints.
    """

    def __init__(self):
        super().__init__()
        self.name = "SCAN_C"

    def forward(self, rho: torch.Tensor, grad_rho: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute SCAN correlation energy density.

        Args:
            rho: Electron density tensor
            grad_rho: Gradient of electron density
            tau: Kinetic energy density

        Returns:
            Correlation energy density tensor
        """
        if grad_rho is None or tau is None:
            raise ValueError("SCAN correlation requires both gradient and kinetic energy density")

        # Wigner-Seitz radius
        rs = (3.0/(4.0*torch.pi*rho))**(1./3.)

        # VWN correlation parameters (simplified)
        A = 0.0310907
        x0 = -0.10498
        b = 3.72744
        c = 12.9352

        x = torch.sqrt(rs)
        X = x**2 + b*x + c
        X0 = x0**2 + b*x0 + c
        Q = torch.sqrt(4.0*c - b**2)

        ec_lda = A * (torch.log(x**2/X) + 2.0*b/Q * torch.atan(Q/(2.0*x + b))
                      - b*x0/X0 * (torch.log((x-x0)**2/X) + 2.0*(b+2.0*x0)/Q *
                      torch.atan(Q/(2.0*x + b))))

        # Simplified SCAN correlation enhancement
        # (Real SCAN correlation is much more complex)
        grad_norm = torch.norm(grad_rho, dim=-1, keepdim=True)
        t = grad_norm / (2.0 * torch.sqrt(torch.pi) * rho**(5./3.))

        tau_w = grad_norm**2 / (8.0 * rho)
        alpha = (tau - tau_w) / tau_w

        H_c = 1.0 + 0.1 * t**2 + 0.05 * alpha**2

        return ec_lda * H_c
