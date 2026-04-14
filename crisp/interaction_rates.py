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

    Parameters:
    -----------
    Z : particle's atomic number
    B : mean magnetic flux density in Gauss
    E : particle energies in GeV
    """
    factor = (GeV / (4.8e-10 * g * cm / s**2)).to('m').value
    
    return factor * E / Z / B 

def interaction_rate_adiabatic(energies, radius):
    """Returns the adiabatic interaction rate

    Parameters:
    -----------
    energies  : particle energies in GeV
    radius : shell radius in m
    """

    return c.value / radius * np.ones_like(energies)


def interaction_rate_acceleration(energies, Z, eta, mgn_field):
    """Returns the acceleration interaction rate

    Parameters:
    -----------
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

    Parameters:
    -----------
    energies  : particle energies in GeV
    A         : particle's nucleon number
    Z         : particle's atomic number
    mgn_field : mean magnetic flux density in Gauss
    """

    m = A * (m_p*c**2).to('GeV').value  # nuclear mass in GeV
    Rg = gyroradius(Z, mgn_field, energies)
    h_alpha_c2 = (hbar * alpha * c**2).to('GeV * m2 / s').value

    return  2/3 * h_alpha_c2 * Z**2 * (energies / m)**4 / Rg**2 / energies


def interaction_rate_from_cross_section(energies, A, ng, eg, cs):
    """Returns the interaction rate from the cross section and the photon spectrum
    Parameters:
    -----------
    energies  : uhecr's energies in GeV
    A         : uhecr's mass in GeV/c2 (nucleon number typically)
    ng        : a function describing the photon spectral density. Should take 
                energy in GeV and return photon density in GeV^-1 cm^-3
    eg        : photon grid for the cross section (energy in nucleus rest frame in GeV)
    cs        : cross section for photonuclear interaction evaluated in eg, given in cm^2
    Returns:
    --------
    rates     : interaction rates corresponding to cross section provided in s^-1
    """
    m = A * .939  # nuclear mass in GeV
    (ymin, ymax), f = get_interp_response_function(eg, cs)
    y = np.logspace(-3., np.log10(ymax), 100)

    rates = []
    for Ej in energies:
        boost = Ej / m
        epsilon = y / boost / 2
        rates.append(np.trapz(ng(epsilon) * f(y) / boost, y))

    rates = c.to('cm/s').value * np.array(rates)

    return rates


def compute_rates(pdensity, pgrid, eweighted_xsec, egrid, boostgrid=None):
    """Computes interaction rates from a table of energy weighted cross sections
    and a function giving the photon densities.

    Arguments
    ---------
    pdensity : a function yielding target photon spectral density in [eV^-1 cm^-3] and taking energy in eV
    pgrid : photon energy grid in eV
    eweighted_xsec : energy weighted cross section table in mb
    egrid : energy grid in MeV for the energy weighted cross section table

    Returns
    -------
    a table of interaction rates in 1/Mpc
    """
    from scipy.signal import fftconvolve
    from scipy.interpolate import interp1d

    N = 3001
    common_grid = np.logspace(-9, 9, N) # in eV
    t = np.log(common_grid)
    dt = np.diff(t)[0]
    conv_grid = np.linspace(t[0]-t[-1], t[-1]-t[0], 2*N-1)

    photon_edens = np.interp(common_grid, pgrid, (pgrid * pdensity(pgrid)), left=0, right=0)

    ewxsec_interp = interp1d(egrid * 1e6, eweighted_xsec, bounds_error=False, fill_value=0)(common_grid)
    ewxsec_interp *= u.mbarn.to('cm^2')

    inter_rates = fftconvolve(np.repeat(np.atleast_2d(photon_edens), len(eweighted_xsec), axis=0),
                                        ewxsec_interp[:, ::-1], mode='full', axes=1)

    inter_rates *= dt / u.cm.to('Mpc')
    interp_rates = interp1d(conv_grid, inter_rates, kind='cubic',
                            fill_value=0, bounds_error=False)

    return interp_rates(-np.log(2*boostgrid))
