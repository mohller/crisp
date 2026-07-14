from pickle import load
import numpy as np
from numpy import pi, expm1, array, vectorize, logspace, log, log10, loadtxt, newaxis, minimum
from numpy import trapezoid
from scipy.constants import h, c, electron_volt, Boltzmann
import os
from importlib.resources import files as _pkg_files

# Resolve the data directory relative to this package, regardless of install method
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def target_photons_spectrum(Emin=1e-6, Emax=1e-4, Ebr=1e-3, si1=1, si2=2, normal=None):
    """Returns a photon spectrum modeled as a broken power law

    Arguments:
    ----------
    Emin: [float]
        lower energy for which the spectrum has a non zero value in GeV
    Emax: [float] 
        higher energy for which the spectrum has a non zero value in GeV
    Ebr : [float]
        break-point energy for of the spectrum in GeV
    si1 : [float]
        power law index of the lower energy part
    si2 : [float]
        power law index of the higher energy part
    normal: [(e1, e2), norm]
        Normalization parameters:
            (e1, e2) - energy range
            norm - value of integral of fluence over the 
            given range (integral of E * dN/dE)
    Returns:
    --------

    """

    if normal is None:
        e1, e2, norm = Emin, Emax, 1.
    else:
        (e1, e2), norm = normal
        print('normal parameters:', e1, e2, norm)

    A = 1.  # normalization constant of the spectrum

    def spectrum(e):
        if (e < Emin) or (e > Emax):
            nk = 0
        elif e <= Ebr:
            nk = (Ebr / e)**si1
        else:
            nk = (Ebr / e)**si2

        return A * nk

    egrid = logspace(log10(e1), log10(e2), 1000)
    dnde = array([spectrum(e) for e in egrid])
    Fluence_integral = trapezoid(egrid**2 * dnde * log(10), x=log10(egrid))

    A = norm / Fluence_integral  # renormalizing the spectrum

    return vectorize(spectrum)


def _normalized_spectrum(shape, Emin, Emax, normal):
    """Window a spectral shape to [Emin, Emax] and normalize the energy
    integral: with normal = ((e1, e2), u), int_e1^e2 E n(E) dE = u
    [GeV / cm^3]. Shared by the GRB prompt-spectrum builders below."""
    def windowed(e):
        e = np.asarray(e, dtype=float)
        return np.where((e >= Emin) & (e <= Emax), shape(np.clip(e, Emin, Emax)), 0.0)

    if normal is None:
        return windowed
    (e1, e2), u = normal
    egrid = logspace(log10(e1), log10(e2), 4000)
    A = u / trapezoid(egrid * windowed(egrid), egrid)
    return lambda e: A * windowed(e)


def band_photon_spectrum(E_peak, alpha=-1.1, beta=-2.2, Emin=1e-9, Emax=1e-2,
                         E0=1e-6, normal=None):
    """Band photon spectrum of GRB prompt emission (Band et al. 1993), in the
    form of De Lia & Tamborra 2024 (JCAP 10, 054), Eq. (2.3); energies in GeV:

        n(E) = C (E/E0)^alpha exp[-(alpha+2) E / E_peak]          E <= E_c
               C (E/E0)^beta  e^(beta-alpha) (E_c/E0)^(alpha-beta)  E > E_c

    with the transition energy E_c = (alpha - beta)/(alpha + 2) E_peak (the
    two branches are continuous there). Fermi-motivated defaults
    alpha = -1.1, beta = -2.2.

    Arguments:
    ----------
    E_peak : comoving peak energy [GeV]
    alpha, beta : low/high spectral indices (beta < alpha < -... typical)
    Emin, Emax : support of the spectrum [GeV]
    E0     : reference energy of the power laws [GeV]
    normal : ((e1, e2), u) — normalize int E n dE over (e1, e2) to u
             [GeV/cm^3], as in target_photons_spectrum

    Returns callable e[GeV] -> n(e) [GeV^-1 cm^-3 up to normalization].
    """
    E_c = (alpha - beta) / (alpha + 2.0) * E_peak

    def shape(e):
        lo = (e / E0)**alpha * np.exp(-(alpha + 2.0) * e / E_peak)
        hi = (e / E0)**beta * np.exp(beta - alpha) * (E_c / E0)**(alpha - beta)
        return np.where(e <= E_c, lo, hi)

    return _normalized_spectrum(shape, Emin, Emax, normal)


def fastcooling_photon_spectrum(E_c, E_peak, index_hi=2.1, Emin=1e-9, Emax=1e-2,
                                normal=None):
    """Fast-cooling synchrotron / self-Compton photon spectrum — the joint
    broken power law of De Lia & Tamborra 2024, Eq. (2.5); energies in GeV:

        n(E) = C (E/E_c)^(-2/3)                                E <= E_c
               C (E/E_c)^(-3/2)                        E_c  <= E <= E_peak
               C (E_peak/E_c)^(-3/2) (E/E_peak)^(-index_hi)     E > E_peak

    (continuous at both breaks by construction). index_hi = (k + 2)/2 with k
    the accelerated-particle index. Normalization as in band_photon_spectrum.
    """
    def shape(e):
        seg1 = (e / E_c)**(-2 / 3)
        seg2 = (e / E_c)**(-3 / 2)
        seg3 = (E_peak / E_c)**(-3 / 2) * (e / E_peak)**(-index_hi)
        return np.where(e <= E_c, seg1, np.where(e <= E_peak, seg2, seg3))

    return _normalized_spectrum(shape, Emin, Emax, normal)


def black_body_spectral_radiance(T, erange):
    """
    Computes the spectral radiance of a black body per unit spectrum per steradian.
    
    Parameters
    ----------
    T : float
        temperature in Kelvin 

    erange : array_like 
        photon energies in eV

    Returns
    -------
    B : array_like
        spectral radiance in W/sr/m^2/Hz
    """
    x = erange * electron_volt / Boltzmann / T
    return 2/(h*c)**2 * (erange*electron_volt)**3 / expm1(minimum(x, 709.))


def black_body_spectral_radiance_wavelength(T, lamrange):
    """
    Computes the spectral radiance of a black body per unit spectrum per steradian.
    
    Parameters
    ----------
    T : float
        temperature in Kelvin 

    lamrange : array_like 
        photon wavelengths in m

    Returns
    -------
    B : array_like
        spectral radiance in W/sr/m^3
    """
    return black_body_spectral_radiance(T, h*c/lamrange/electron_volt) * c / lamrange**2

To = 2.725 # Kelvin, CMB temperature

# CMB photon density in m^-3, takes energy in eV
cmb_photon_density = lambda T, erange: 4 * pi / h / c * black_body_spectral_radiance(T, erange) / erange

# CMB photon energy density in cm^-3 takes energy in GeV
cmb_photon_density_GeVcm3 = lambda erange: cmb_photon_density(To, erange * 1e9) / 1e6 * 1e9 # 1e6 for m3 to cm3 and 1e9 for eV to GeV

cmb = black_body_spectral_radiance


### Target photon fields for EBL models..

# Model by Gilmore 2012, takes energy in eV and returns density in m^-3 eV^-1
with open(os.path.join(_DATA_DIR, 'Gilmore12_splinterp.pkl'), 'rb') as file:
    eblg_interp = load(file)

# Model by Saldana-Lopez 2021, takes energy in eV and returns density in m^-3 eV^-1
with open(os.path.join(_DATA_DIR, 'SaldanaLopez21_splinterp.pkl'), 'rb') as file:
    ebls_interp = load(file)

# Model by Andrews 2018, takes energy in eV and returns density in m^-3 eV^-1
with open(os.path.join(_DATA_DIR, 'Andrews18_splinterp.pkl'), 'rb') as file:
    ebla_interp = load(file)


def create_interpolated_EBLmodel_Asndrews18(ebl_filename):
    """ Creates an interpolated version of the ebl model.
        Based on the data file found in CRPropa3-data/tables/Andrews17/table_file.dat
    """
    import pickle
    import astropy.units as u
    from astropy.constants import hbar, c
    from scipy.interpolate import RectBivariateSpline

    zlist = array([0., 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.])
    ebla = loadtxt(ebl_filename)


    elist = (2*pi*hbar*c / (ebla[:, 0] * 1e-6 * u.m)).to('eV').value[::-1] # energies 
    density_grid = (1 / pi / hbar**2 / c**3 * (ebla[:, 1:] * 1e-9 * u.J / u.s / u.m**2) * (ebla[:, 0][:, newaxis] * 1e-6 * u.m)**2).to('1/(eV*m^3)').value[::-1, :]

    ebla_interp = RectBivariateSpline(elist, zlist, density_grid, s=0)

    with open(os.path.join(_DATA_DIR, 'Andrews18_splinterp.pkl'), 'wb') as file:
        pickle.dump(ebla_interp, file)


def create_interpolated_EBLmodel_Gilmore12(ebl_filename):
    """ Creates an interpolated version of the ebl model.
        Based on the data file found in CRPropa3-data/tables/EBL_Gilmore_2012/eblflux_fixed.dat
    """
    import pickle
    import astropy.units as u
    from astropy.constants import hbar, c
    from scipy.interpolate import RectBivariateSpline

    zlist = array([0.0, 0.015, 0.025, 0.044, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0])
    eblg = loadtxt(ebl_filename)


    elist = (2*pi*hbar*c / (eblg[:, 0] * 1e-10 * u.m)).to('eV').value[::-1] # energies 
    density_grid = (4*pi / c * eblg[:, 0][:, newaxis] * eblg[:, 1:] * u.erg / u.s / u.cm**2 / (2*pi*hbar*c / (eblg[:, 0][:, newaxis] * 1e-10 * u.m))**2).to('1/(eV*m^3)').value[::-1, :]

    eblg_interp = RectBivariateSpline(elist, zlist, density_grid / (1 + zlist[newaxis, :])**3, s=0)

    with open(os.path.join(_DATA_DIR, 'Gilmore12_splinterp.pkl'), 'wb') as file:
        pickle.dump(eblg_interp, file)


def create_interpolated_EBLmodel_SaldanaLopez21(ebl_filename):
    """ Creates an interpolated version of the ebl model.
        Based on the data file found in CRPropa3-data/tables/EBL_SaldanaLopez_2021/ebl_saldana21_comoving.txt
    """
    import pickle
    import astropy.units as u
    from astropy.constants import hbar, c
    from scipy.interpolate import RectBivariateSpline

    zlist = array([0., 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.])
    ebls = loadtxt(ebl_filename)


    elist = (2*pi*hbar*c / (ebls[:, 0] * 1e-6 * u.m)).to('eV').value[::-1] # energies 
    density_grid = (1 / pi / hbar**2 / c**3 * (ebls[:, 1:] * 1e-9 * u.J / u.s / u.m**2) * (ebls[:, 0][:, newaxis] * 1e-6 * u.m)**2).to('1/(eV*m^3)').value[::-1, :]

    ebls_interp = RectBivariateSpline(elist, zlist, density_grid, s=0)

    with open(os.path.join(_DATA_DIR, 'SaldanaLopez21_splinterp.pkl'), 'wb') as file:
        pickle.dump(ebls_interp, file)
