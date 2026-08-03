"""Interaction rate integrals: turning a cross section and a photon field
into a rate in 1/s or 1/Mpc.

`InteractionCore` calls into this module to build its interaction
tensors; the functions here are the rate integrals themselves,
independent of the species bookkeeping that lives in `core.py`. Two
families are provided:

- Non-photon rates that depend only on the particle's own kinematics
  and an external field: `interaction_rate_adiabatic` (Hubble-like
  expansion), `interaction_rate_acceleration` (Bohm-like, via
  `gyroradius`), and `interaction_rate_synchrotron`.
- Photon-field interaction rates, which fold a photonuclear cross
  section against an isotropic photon spectral density: the direct,
  one-Lorentz-factor-at-a-time integrators
  `interaction_rate_from_cross_section` /
  `interaction_rate_from_cross_section_boosts`, and the batched,
  FFT-based `compute_rates` / `exact_rates_for_sigma`, which evaluate
  every cross-section row on the whole boost grid in one log-space
  convolution. `InteractionCore` uses the batched path; the direct
  integrators are mainly useful for a one-off cross-check.

Most users will not call into this module directly; it is consumed by
`core.InteractionCore` while it builds its tensors from a
`photonuclear_cross_sections` model and a target photon field.
"""

import numpy as np
from astropy.constants import c, hbar, alpha, m_p
from astropy.units import cm, GeV, g, s
from .photonuclear_cross_sections import *

def gyroradius(Z, B, E):
    """Computes the gyroradius for a particles with charge Z in
    units of the elementary charge e, under a magnetic field intensity
    B in Gauss and the energy E in GeV. The radius returned is in units
    of meters.
    The gyroradius rg is computed as 
        rg = E/Ze/B
    where using the relation in cgs: e*G = 4.8E10 g*cm/s and the
    conversion factor GeV = 1.602E-10 kg*m2/s2 in the prefactor.

    Parameters
    ----------
    Z : particle's atomic number
    B : mean magnetic flux density in Gauss
    E : particle energies in GeV
    """
    factor = (GeV / (4.8e-10 * g * cm / s**2)).to('m').value
    
    return factor * E / Z / B 

def interaction_rate_adiabatic(energies, radius):
    """Returns the adiabatic interaction rate

    Parameters
    ----------
    energies  : particle energies in GeV
    radius : shell radius in m
    """

    return c.value / radius * np.ones_like(energies)


def interaction_rate_acceleration(energies, Z, eta, mgn_field):
    """Returns the acceleration interaction rate

    Parameters
    ----------
    energies  : particle energies in GeV
    Z         : particle's atomic number
    eta       : acceleration efficiency (0..1 dimensionless)
    mgn_field : mean magnetic flux density in Gauss
    """
    Rg = gyroradius(Z, mgn_field, energies)

    return eta * c.value / Rg


def interaction_rate_synchrotron(energies, Z, A, mgn_field):
    """Returns the synchrotron interaction rate for nucleons and nuclei

    The power emitted by a relativistic particle with total charge q,
    mass m, and kinetic energy E (relative velocity beta), under the
    influence of a magnetic field B is:

    P = e^2/(6 pi eps0) c Z^2 gamma^4 / Rg^2

    In a very relativistic scenario, beta ~ 1 and the kinetic energy
    is most of the total energy (E = gamma * m * c^2). 
    Using the expression:

    alpha = 1/137 = e^2 / (4 pi eps0 hbar c)

    The energy loss rate per unit time is results:

    t^-1 = P / E = 2/3 hbar alpha c^2 Z^2 gamma^4 / Rg^2 / E

    Parameters
    ----------
    energies  : particle energies in GeV
    A         : particle's nucleon number
    Z         : particle's atomic number
    mgn_field : mean magnetic flux density in Gauss
    """

    m = A * (m_p*c**2).to('GeV').value  # nuclear mass in GeV
    Rg = gyroradius(Z, mgn_field, energies)
    h_alpha_c2 = (hbar * alpha * c**2).to('GeV * m2 / s').value

    return  2/3 * h_alpha_c2 * Z**2 * (energies / m)**4 / Rg**2 / energies


def interaction_rate_from_cross_section_boosts(boosts, ng, eg, cs):
    """Returns the interaction rate from the cross section and the photon spectrum,
    taking the Lorentz factors directly.

    Boost-native variant of interaction_rate_from_cross_section: the rate
    integral only depends on the Lorentz factor, so no nuclear mass enters.
    Prefer this entry point when working on a boost grid; use the nuclear
    masses (e.g. crisp.data.nucleardecays.nuclear_mass_GeV) only to convert
    between energies and boosts at the interface.

    Parameters
    ----------
    boosts    : uhecr's Lorentz factors
    ng        : a function describing the photon spectral density. Should take
                energy in GeV and return photon density in GeV^-1 cm^-3
    eg        : photon grid for the cross section (energy in nucleus rest frame in GeV)
    cs        : cross section for photonuclear interaction evaluated in eg, given in cm^2
    Returns
    -------
    rates     : interaction rates corresponding to cross section provided in s^-1
    """
    (ymin, ymax), f = get_interp_response_function(eg, cs)
    y = np.logspace(-3., np.log10(ymax), 100)

    rates = []
    for boost in boosts:
        epsilon = y / boost / 2
        rates.append(np.trapezoid(ng(epsilon) * f(y) / boost, y))

    return c.to('cm/s').value * np.array(rates)


def interaction_rate_from_cross_section(energies, A, ng, eg, cs):
    """Returns the interaction rate from the cross section and the photon spectrum

    Parameters
    ----------
    energies  : uhecr's energies in GeV
    A         : uhecr's mass in GeV/c2 (nucleon number typically)
    ng        : a function describing the photon spectral density. Should take 
                energy in GeV and return photon density in GeV^-1 cm^-3
    eg        : photon grid for the cross section (energy in nucleus rest frame in GeV)
    cs        : cross section for photonuclear interaction evaluated in eg, given in cm^2
    Returns
    -------
    rates     : interaction rates corresponding to cross section provided in s^-1
    """
    m = A * .939  # nuclear mass in GeV
    (ymin, ymax), f = get_interp_response_function(eg, cs)
    y = np.logspace(-3., np.log10(ymax), 100)

    rates = []
    for Ej in energies:
        boost = Ej / m
        epsilon = y / boost / 2
        rates.append(np.trapezoid(ng(epsilon) * f(y) / boost, y))

    rates = c.to('cm/s').value * np.array(rates)

    return rates


def compute_rates(pdensity, pgrid, eweighted_xsec, egrid, boostgrid=None,
                  common_bounds=(-9, 9), N=3001):
    """Computes interaction rates from a table of energy weighted cross sections
    and a function giving the photon densities.

    Implements the isotropic-field rate as a single log-space convolution of
    the photon energy density with the energy-weighted cross section
    g(y) = (2/y^2) int_0^y y' sigma(y') dy', evaluated for the whole boost
    grid and all cross-section rows at once (one FFT per call). Above the
    tabulated sigma support the inner integral saturates, so g continues
    analytically as g(y_max) (y_max / y)^2 — without this tail the rates are
    underestimated at boosts where 2 Gamma eps_peak exceeds the sigma range.

    Parameters
    ----------
    pdensity : a function yielding target photon spectral density in [eV^-1 cm^-3] and taking energy in eV
    pgrid : photon energy grid in eV
    eweighted_xsec : energy weighted cross section table in mb
    egrid : energy grid in MeV for the energy weighted cross section table
    boostgrid : Lorentz factors at which to evaluate the rates
    common_bounds : log10 range in eV of the internal grid (must cover the
                    photon field support and the sigma support egrid*1e6)
    N : number of points of the internal grid

    Returns
    -------
    a table of interaction rates in 1/Mpc
    """
    from astropy import units as u
    from scipy.signal import fftconvolve
    from scipy.interpolate import interp1d

    common_grid = np.logspace(*common_bounds, N) # in eV
    t = np.log(common_grid)
    dt = np.diff(t)[0]
    conv_grid = np.linspace(t[0]-t[-1], t[-1]-t[0], 2*N-1)

    photon_edens = np.interp(common_grid, pgrid, (pgrid * pdensity(pgrid)), left=0, right=0)

    eweighted_xsec = np.atleast_2d(eweighted_xsec)
    ewxsec_interp = interp1d(egrid * 1e6, eweighted_xsec, bounds_error=False, fill_value=0)(common_grid)

    # analytic continuation above the sigma support: the inner integral has
    # saturated, so the energy-weighted cross section falls as 1/y^2
    ymax = egrid[-1] * 1e6
    tail = common_grid > ymax
    ewxsec_interp[:, tail] = eweighted_xsec[:, -1:] * (ymax / common_grid[tail])**2

    ewxsec_interp *= u.mbarn.to('cm^2')

    inter_rates = fftconvolve(np.repeat(np.atleast_2d(photon_edens), len(eweighted_xsec), axis=0),
                                        ewxsec_interp[:, ::-1], mode='full', axes=1)

    inter_rates *= dt / u.cm.to('Mpc')
    interp_rates = interp1d(conv_grid, inter_rates, kind='cubic',
                            fill_value=0, bounds_error=False)

    return interp_rates(-np.log(2*boostgrid))


def exact_rates_for_sigma(boosts, target_photons, eps_GeV, sigma_rows_mb):
    """Batched exact isotropic-field interaction rates of cross-section rows.

    Wraps compute_rates with an internal grid sized from the photon-field
    support and the boost range (so the analytic 1/y^2 tail of the
    energy-weighted cross section is captured up to 2 Gamma_max
    eps_field_max). Rates are clipped at zero (cubic interpolation of the
    convolution can leave tiny negatives).

    Parameters
    ----------
    boosts        : Lorentz-factor grid
    target_photons: photon field n_gamma(eps) in GeV^-1 cm^-3, eps in GeV
    eps_GeV       : photon-energy grid where the cross sections are sampled
    sigma_rows_mb : array (n_rows, len(eps_GeV)) of cross sections in mb

    Returns
    -------
    rates : ndarray (n_rows, len(boosts)) in Mpc^-1
    """
    eps_GeV = np.asarray(eps_GeV)
    eps_MeV = eps_GeV * 1e3
    eweighted = energy_weight_sigma(sigma_rows_mb, eps_MeV)

    # photon-field support (eV), probed on a wide grid
    probe = np.logspace(-9, 15, 200)
    n_probe = np.asarray(target_photons(probe * 1e-9))
    support = probe[n_probe > 0]
    field_lo = support[0] if support.size else 1e-9
    field_hi = support[-1] if support.size else 1e9

    lo = min(-9.0, np.log10(field_lo))
    hi = max(9.0, np.log10(eps_GeV[-1] * 1e9),
             np.log10(2 * boosts[-1] * field_hi))
    n_points = int((hi - lo) * 167) + 1

    pgrid = np.logspace(lo, hi, n_points)
    pdensity_eV = lambda e: np.asarray(target_photons(np.asarray(e) * 1e-9)) * 1e-9

    # group xsecs by chun to speed up the convolution and keep the 
    # transient array size bounded. Large reaction networks with hundreds 
    # of species, can put n_rows in the tens of thousands, where the 
    # unchunked array reaches multiple GB per InteractionCore construction.
    eweighted = np.atleast_2d(eweighted)
    row_chunk = 2000
    if eweighted.shape[0] > row_chunk:
        rates = np.concatenate([
            compute_rates(pdensity_eV, pgrid, eweighted[i:i + row_chunk],
                         eps_MeV, boostgrid=boosts, common_bounds=(lo, hi),
                         N=n_points)
            for i in range(0, eweighted.shape[0], row_chunk)
        ], axis=0)
    else:
        rates = compute_rates(pdensity_eV, pgrid, eweighted, eps_MeV,
                              boostgrid=boosts, common_bounds=(lo, hi), N=n_points)
    return np.clip(rates, 0.0, None)
