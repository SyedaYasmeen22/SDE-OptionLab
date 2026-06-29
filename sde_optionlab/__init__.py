"""SDE-OptionLab: a simulation-based pricing engine for spread and
better-of options on two (or more) correlated assets.
"""

from . import models, payoffs, pricers, greeks, diagnostics, estimation

__all__ = ["models", "payoffs", "pricers", "greeks", "diagnostics", "estimation"]
__version__ = "1.0.0"
