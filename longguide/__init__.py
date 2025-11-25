"""
LongGuide: Long-form Text Generation with Policy Optimization

A framework for improving long-form text generation through structured evaluation
metrics and policy optimization techniques.
"""

__version__ = "0.1.0"
__author__ = "Do Xuan Long"

from .guidelines import MetricsGuidelines, OutputConstraintsGuidelines

__all__ = [
    "MetricsGuidelines",
    "OutputConstraintsGuidelines"
]
