from pickle import load
from numpy import pi, exp, array, vectorize, logspace, log, log10, trapz, loadtxt, newaxis
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

# CMB photon density in m^-3 eV^-1
cmb_photon_density = lambda T, erange: 4 * pi / h / c * black_body_spectral_radiance(T, erange) / erange

# CMB photon energy density in GeV / cm^3
cmb_photon_density_GeVcm3 = lambda erange: cmb_photon_density(To, erange * 1e9) / erange / 1e9

cmb = black_body_spectral_radiance


### Target photon fields for EBL models..

# Model by Gilmore 2012, takes energy in eV and returns density in m^-3 eV^-1
with open('../data/Gilmore12_splinterp.pkl', 'rb') as file:
    eblg_interp = load(file)

# Model by Saldana-Lopez 2012, takes energy in eV and returns density in m^-3 eV^-1
with open('../data/SaldanaLopez21_splinterp.pkl', 'rb') as file:
    ebls_interp = load(file)

# Model by Andrews 2018, takes energy in eV and returns density in m^-3 eV^-1
with open('../data/Andrews18_splinterp.pkl', 'rb') as file:
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

    with open('../data/Andrews18_splinterp.pkl', 'wb') as file:
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

    with open('../data/Gilmore12_splinterp.pkl', 'wb') as file:
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

    with open('SaldanaLopez21_splinterp.pkl', 'wb') as file:
        pickle.dump(ebls_interp, file)
