"""
CRISP: Cosmic Ray Stochastic Interactions for Propagation

A convenience package to compute quantities related to the propagation 
of ultrahigh-energy cosmic rays based on closed form probability 
distribution functions.
"""

__version__ = "0.1.0"
__author__ = "Leonel Morejon"
__email__ = "leonel.morejon@uni-wuppertal.de"

import logging

# The package emits no logs by default: diagnostics are recorded at DEBUG
# level on the 'crisp' logger hierarchy and stay silent even when the
# application configures verbose logging. Developers enable them with:
#     logging.getLogger('crisp').setLevel(logging.DEBUG)
#     logging.basicConfig()      # or attach any handler
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.getLogger(__name__).setLevel(logging.WARNING)

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("crisp-pypi")
except PackageNotFoundError:
    __version__ = "unknown"

from . import core
from . import interaction_rates
from . import photonuclear_cross_sections
from . import background_photon_models
from . import continuous_losses
from . import source_models
from . import data_download
from .data_download import fetch_crpropa_tables, fetch_astrophomes, get_astrophomes_path

try:
    fetch_crpropa_tables(verbose=False)
except Exception:
    pass  # e.g. no network or disk access; download later

__all__ = [
    'core', 'interaction_rates', 'photonuclear_cross_sections',
    'background_photon_models', 'continuous_losses', 'source_models',
    'data_download', 'fetch_crpropa_tables', 'fetch_astrophomes',
    'get_astrophomes_path',
]
