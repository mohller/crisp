"""
CRISP: Cosmic Ray Stochastic Interactions for Propagation

A package to compute the propagation of ultrahigh-energy cosmic ray
nuclei through radiation fields (the CMB, the extragalactic background
light, or a custom photon field). Given an injected composition and
energy spectrum, it returns the exact, closed-form probability
distribution of surviving and fragment nuclei, and of secondary
particles (pions, nucleons, neutrinos, light nuclei) produced along the
way, at any distance from the source. The same interaction tensors also
give the closed form probability distribution of the distance itself at
which a nucleus first interacts or is absorbed (a "distance until
absorption" distribution, with its cdf, pdf, and moments), the
complementary, survival-analysis style view of the same cascade. The
core transport equation is solved as a matrix exponential rather than by
Monte Carlo sampling, so a result at one distance, or a thousand
distances at once, costs the same kind of calculation and carries no
statistical noise.

Main pieces, and how they fit together
---------------------------------------
- `photonuclear_cross_sections` defines the cross section models
  (`CRPropa_model`, `PSB_model`, `SimProp_model`, and others) that supply
  interaction rates for a given nuclear species and photon field. A
  cross section model is the main ingredient an `InteractionCore` is
  built from.
- `core` defines `InteractionCore`, the class that assembles those
  cross sections (plus optional photon fields, decay data, and
  photomeson kernels) into interaction tensors, then solves the
  transport equation, both for the propagated population at a distance
  (`species_evolution_boost_range` and the secondary production methods
  built on it) and for the distance-until-absorption distributions
  (`cdf_boost_range`/`pdf_boost_range` and their moments). This is the
  class most calculations start from.
- `source_models` defines astrophysical source classes (GRB jet
  models and related injection spectra) that produce the injection
  vector `InteractionCore` needs, for users who want a physically
  motivated source rather than a hand built one.
- `extragalactic` defines `ExtragalacticPropagation`, for propagation
  across cosmological distances where redshift evolution of the photon
  fields and adiabatic losses matter, built on top of `InteractionCore`.
- `interaction_rates`, `continuous_losses`, and `background_photon_models`
  are lower level physics modules (rate integrals, continuous energy
  loss rates, and photon field spectra) consumed by the modules above;
  most users will not need to call into them directly.
- `data_download` fetches and caches the CRPropa3 cross section tables
  and the AstroPhoMes photomeson data from their public repositories on
  first use, so cross section models can find their data files.

A minimal example
------------------
Inject iron-56 at a fixed energy, propagate it through the CMB with the
self-contained PSB cross sections (no external data download needed),
and read off the surviving fraction at a few distances:

>>> import numpy as np
>>> from crisp.core import InteractionCore
>>> from crisp.photonuclear_cross_sections import PSB_model
>>>
>>> core = InteractionCore(xsec_model=PSB_model())
>>> alpha, mass_range, true_range, _ = core.get_distribution_parameters(
...     mass_lims=(56, 0), injection_type=('only species', (26, 56)),
...     absorption_type=('only mass', []))
>>> distances = np.array([1.0, 10.0, 50.0, 200.0])  # Mpc
>>> P = core.species_evolution_boost_range(
...     distances, alpha, mass_range, core.boosts, true_range)
>>>
>>> # pick the boost bin nearest 99 EeV and read off Fe-56's own column
>>> i_100eev = np.argmin(np.abs(
...     core.energy_of_boost((26, 56), core.boosts) * 1e9 - 1e20))
>>> i_fe = true_range.index(core.species.index((26, 56)))
>>> P[i_100eev, :, i_fe]
array([0.99645193, 0.96508049, 0.83717775, 0.49121398])

Fe-56 injected at 99 EeV survives essentially unchanged to 1 Mpc, but
only half survives out to 200 Mpc; the rest has photodisintegrated into
lighter species, which `P`'s other columns hold (population is conserved
exactly across the full species set: `P[i_100eev].sum(axis=-1)` is 1 at
every distance here, since `absorption_type=('only mass', [])` tracks
every fragment down to free nucleons instead of discarding any of them).
See `InteractionCore`'s own docstring, and the notebooks under
`examples/`, for further worked cases (heavier nuclei, secondary particle
production, redshift-aware propagation, and source models).
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
from . import extragalactic
from . import data_download
from .data_download import fetch_crpropa_tables, fetch_astrophomes, get_astrophomes_path

try:
    fetch_crpropa_tables(verbose=False)
except Exception:
    pass  # e.g. no network or disk access; download later

__all__ = [
    'core', 'interaction_rates', 'photonuclear_cross_sections',
    'background_photon_models', 'continuous_losses', 'source_models',
    'extragalactic', 'data_download', 'fetch_crpropa_tables',
    'fetch_astrophomes', 'get_astrophomes_path',
]
