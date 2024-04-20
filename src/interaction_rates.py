import numpy as np
from astropy.constants import c, hbar, alpha, m_p
from astropy.units import erg, km, cm, GeV, g, s
from numpy import log, log10, logspace
from photonuclear_cross_sections import *

# physical constants
c = c.to('cm/s').value # speed of light
ergs2GeV = erg.to('GeV')  # energy conversion factor from ergs to GeV
km2cm = km.to('cm')  # distance conversion factor from km to cm

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
    radius : shell radius in cm
    """

    return c / radius * np.ones_like(energies)


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

    return 1e-17 * eta * c**2 * Z * mgn_field / energies


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
    energies  : particle energies in GeV
    A         : particle's nucleon number
    ng        : a function describing the photon spectral density. Should take 
                energy in GeV and return photon density in GeV^-1 cm^-3
    eg        : photon energy in nucleus rest frame in GeV 
    cs        : cross section in cm^2
    Returns:
    --------
    rates     : interaction rates corresponding to cross section provided in s^-1
    """
    m = A * .939  # nuclear mass in GeV
    (ymin, ymax), f = get_interp_response_function(eg, cs)
    y = logspace(-3., np.log10(ymax), 100)

    rates = []
    for Ej in energies:
        boost = Ej / m
        epsilon = y / boost
        rates.append(np.trapz(ng(epsilon) * f(y) * y * np.log(10) / boost,
                             x=np.log10(y)))

    rates = c * np.array(rates)

    return rates

