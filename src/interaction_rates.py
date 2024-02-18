import numpy as np
from astropy.constants import c
from astropy.units import erg, km
from numpy import log, log10, logspace
from photonuclear_cross_sections import *

# physical constants
c = c.to('cm/s').value # speed of light
ergs2GeV = erg.to('GeV')  # energy conversion factor from ergs to GeV
km2cm = km.to('cm')  # distance conversion factor from km to cm

def interaction_rate_adiabatic(energies, radius, boost):
    """Returns the adiabatic interaction rate

    Parameters:
    -----------
    energies  : particle energies in GeV
    radius : shell radius in cm
    boost : Lorentz boost of shell
    """

    return 2 * boost * c / radius * np.ones_like(energies)


def interaction_rate_acceleration(energies, Z, eta, mgn_field):
    """Returns the acceleration interaction rate

    Parameters:
    -----------
    energies  : particle energies in GeV
    Z         : particle's atomic number
    eta       : acceleration efficiency (0..1 dimensionless)
    mgn_field : mean magnetic flux density in Gauss
    """

    return 1e-17 * eta * c**2 * Z * mgn_field / energies


def interaction_rate_synchrotron(energies, Z, A, mgn_field):
    """Returns the synchrotron interaction rate for nucleons and nuclei

    The power emitted by a relativistic particle with total charge q,
    mass m, and kinetic energy E (relative velocity beta), under the
    influence of a magnetic field B is:

    P = (q^2 * B*E*beta)^2 / (6 * pi * eps0 * m^4 * c^5)

    In a very relativistic scenario, beta ~ 1 and the kinetic energy
    is most of the total energy (E = gamma * m * c^2). The energy loss
    rate per unit time is results:

    t^-1 = P / E = q^4 * B^2 * E / (6 * pi * eps0 * m^4 * c^5)

    where M = m*c^2 in GeV, and eps0 is the vacuum permittivity with value
    ~8.854E-12 F / m = 8.854E-12 C^2 (m^3/s^2)^-1 kg^-1.

    The formula can be simplified by making the following substitutions:
    q = Z * qe  with Z atomic number, and qe the electron charge in Coulomb

    (...to be completed ...)

    Parameters:
    -----------
    energies  : particle energies in GeV
    A         : particle's nucleon number
    Z         : particle's atomic number
    mgn_field : mean magnetic flux density in Gauss
    """

    m = A * .939  # nuclear mass in GeV

    return Z**4 * mgn_field**2 / m**4 * (3e8)**3 * 1.602e-19 * 1e-37 \
        / (9 * np.pi * 8.854e-12) * energies
    # return Z**4 * (4. / 3 * 6.6524e-25 / c**3 / A**2 / 1e-24**2 *
    #                mgn_field**2 / 8 / np.pi * energies / 624.15e9)


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

