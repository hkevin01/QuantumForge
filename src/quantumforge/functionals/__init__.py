"""
QuantumForge functional implementations.

This package provides concrete implementations of density functionals
including LDA, GGA, meta-GGA, and hybrid functionals.
"""

from .gga import BLYPExchange, LYPCorrelation, PBECorrelation, PBEExchange
from .hybrid import B3LYP, HSE06, PBE0
from .lda import SlaterExchange, VWNCorrelation
from .meta_gga import SCANCorrelation, SCANExchange

__all__ = [
    # LDA functionals
    "SlaterExchange",
    "VWNCorrelation",
    # GGA functionals
    "PBEExchange",
    "PBECorrelation",
    "BLYPExchange",
    "LYPCorrelation",
    # Meta-GGA functionals
    "SCANExchange",
    "SCANCorrelation",
    # Hybrid functionals
    "B3LYP",
    "PBE0",
    "HSE06",
]
