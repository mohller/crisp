
from numpy import pi, exp, array, vectorize, logspace, log10, trapz
from scipy.constants import h, c, electron_volt, Boltzmann


def target_photons_spectrum(Emin=1e-6, Emax=1e4, Ebr=1e3, si1=1, si2=2, normal=None):
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
            given range (integral of E**2 * dN/dE)
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
    Fluence_integral = trapz(egrid**2 * dnde * log(10), x=log10(egrid))

    A = norm / Fluence_integral  # renormalizing the spectrum

    return vectorize(spectrum)


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
    return 2/(h*c)**2 * (erange*electron_volt)**3 / (exp(erange*electron_volt/Boltzmann/T) - 1)


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

# photon density in m^-3 eV^-1
cmb_photon_density = lambda T, erange: 4 * pi / c * black_body_spectral_radiance(T, erange) / erange / h

# photon energy density in GeV / cm^3
cmb_photon_density_GeVcm3 = lambda erange: cmb_photon_density(To, erange * 1e9) / erange / 1e9

cmb = black_body_spectral_radiance