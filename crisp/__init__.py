"""
CRISP: Cosmic Ray Stochastic Interactions for Propagation

A convenience package to compute quantities related to the propagation 
of ultrahigh-energy cosmic rays based on closed form probability 
distribution functions.
"""

__version__ = "0.1.0"
__author__ = "Leonel Morejon"
__email__ = "leonel.morejon@uni-wuppertal.de"

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("crisp-py")
except PackageNotFoundError:
    __version__ = "unknown"

from . import core
from . import interaction_rates
from . import photonuclear_cross_sections
from . import background_photon_models
from . import continuous_losses
from . import source_models

__all__ = ['core', 'interaction_rates', 'photonuclear_cross_sections', 'background_photon_models', 'continuous_losses', 'source_models']
