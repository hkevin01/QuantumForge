"""
QuantumForge core modules for quantum chemistry calculations.

This package provides the fundamental building blocks for density functional
theory (DFT) calculations, including functional abstractions, grid management,
and numerical operators.
"""

from .functional_base import (
    FunctionalBase,
    GGAFunctional,
    HybridFunctional,
    LDAFunctional,
    MetaGGAFunctional,
)
from .grid import AdaptiveGrid, GridBase, UniformGrid
from .operators import (
    FiniteDifferenceGradient,
    FiniteDifferenceLaplacian,
    NumericalOperatorBase,
    SpectralOperators,
)

__all__ = [
    # Functional base classes
    "FunctionalBase",
    "LDAFunctional",
    "GGAFunctional",
    "MetaGGAFunctional",
    "HybridFunctional",
    # Grid management
    "GridBase",
    "UniformGrid",
    "AdaptiveGrid",
    # Numerical operators
    "NumericalOperatorBase",
    "FiniteDifferenceGradient",
    "FiniteDifferenceLaplacian",
    "SpectralOperators",
]
