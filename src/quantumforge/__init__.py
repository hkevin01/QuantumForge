"""
QuantumForge: High-performance quantum chemistry framework.

A modern, GPU-accelerated framework for density functional theory (DFT)
calculations, designed for performance, flexibility, and ease of use.
"""

from . import core
from .utils import cuda_ops

__version__ = "0.1.0"
__author__ = "QuantumForge Development Team"
__email__ = "dev@quantumforge.org"

__all__ = ["core", "cuda_ops"]
