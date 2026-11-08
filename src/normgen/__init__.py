"""Generating normally distributed pseudo-random numbers from a uniform generator."""

from .inverse_cdf import norm_ppf
from .samplers import SAMPLERS, box_muller, inverse_transform, polar, rejection
from .uniform import LCG, NumpyUniform

__all__ = [
    "LCG",
    "SAMPLERS",
    "NumpyUniform",
    "box_muller",
    "inverse_transform",
    "norm_ppf",
    "polar",
    "rejection",
]

__version__ = "1.0.0"
