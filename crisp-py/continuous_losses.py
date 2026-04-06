import numpy as np
from scipy.integrate import cumulative_trapezoid
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c, m_e, alpha, e, eps0, k_B, hbar, m_p

def Bpp_generic(Z, mA, g, z=0, phot_dens=None):
    """Compute pair production losses for an arbitrary photon field
        Based on the papers:
        Blumenthal, G. R. (1970) PRD 1(6), 1596
        "Energy Loss of High-Energy Cosmic Rays in Pair-Producing Collisions with Ambient Photons."
        https://doi.org/10.1103/PhysRevD.1.1596
        "Reaction rate and energy loss rate for photopair production by relativistic nuclei"
        M. CHodorowski, A. Zdziarski, M. Sikora, ApJ 400, 181-185, 1992

        B = -1/E dE/dt = -1/g dg/dt
        Values given in Mpc^-1

        Arguments:
        Z, mA: charge number and mass (in units of proton mass) of the nuclear species
        g: lorentz boost(s) for which to evaluate Bpp
        z: redshift for which to evaluate Bpp
        phot_dens: target photon field density taking photon energies in eV and returning density in [eV^-1 m^-3]
    """
    mec2 = (m_e * c**2).to('eV')
    mpc2 = (m_p * c**2).to('eV')
    r_e = e.si**2 / (4*np.pi*eps0.to(u.C**2/u.eV/u.m) * m_e.to(u.eV * u.s**2/u.m**2) * c**2)

    phi_lo = lambda k: np.pi / 12 * (k - 2)**4 / (1 + np.polyval([-3.879e-6, 1.137e-3, 0.1459, 0.8048, 0], k-2))
    phi_B70 = lambda x: x * np.polyval([8/3., -14.45, 50.96, -86.07], np.log(x))
    phi_hi = lambda x: phi_B70(x) / (1 - np.polyval([1837, 78.35, 2.91, 0], 1/x))
    phi = lambda x: np.where(x < 25, phi_lo(x), phi_hi(x))
    
    integral_grid = []
    for gval in g / mec2.value:
        nu = np.logspace(np.log10(1 / gval), 4, 300)
        integral_grid.append(np.trapz(phot_dens(nu, z) * phi(2*gval*nu) / nu**2, nu))

    bpp = mec2.value**2 * (alpha*r_e**2 / u.m**3 * mec2 / mpc2 / g**2 / 2 * np.array(integral_grid)).to('1/Mpc').value

    return bpp * Z**2/mA

def Bpp_Blumenthal(Z, A, g, z=0):
    """Compute pair production losses for interactions with the CMB
    Based on the papers:
        Blumenthal, G. R. (1970) PRD 1(6), 1596
        "Energy Loss of High-Energy Cosmic Rays in Pair-Producing Collisions with Ambient Photons."
        https://doi.org/10.1103/PhysRevD.1.1596
        "Reaction rate and energy loss rate for photopair production by relativistic nuclei"
        M. CHodorowski, A. Zdziarski, M. Sikora, ApJ 400, 181-185, 1992

        B = -1/E dE/cdt = -1/g dg/cdt
        Values given in Mpc^-1
    """
    hbc = (hbar * c).to('eV m')
    kTo = (k_B * 2.7 * u.K).to('eV')
    mec2 = (m_e * c**2).to('eV')
    mpc2 = (m_p * c**2).to('eV')
    r_e = e.si**2 / (4*np.pi*eps0.to(u.C**2/u.eV/u.m) * m_e.to(u.eV * u.s**2/u.m**2) * c**2)
    alph_re2me2 = alpha*r_e**2 * mec2**2
    

    phi_lo = lambda k: np.pi / 12 * (k - 2)**4 / (1 + np.polyval([-3.879e-6, 1.137e-3, 0.1459, 0.8048, 0], k-2))
    phi_B70 = lambda x: x * np.polyval([8/3., -14.45, 50.96, -86.07], np.log(x))
    phi_hi = lambda x: phi_B70(x) / (1 - np.polyval([1837, 78.35, 2.91, 0], 1/x))
    phi = lambda x: np.where(x < 25, phi_lo(x), phi_hi(x))

    
    chi = np.logspace(.4, 6, 500)
    nu_eval = 10**np.linspace(-4, 1.15, 100)
    fnu_eval = np.array([nuval**2 * np.trapz(phi(chi) / (np.exp(nuval * chi) - 1), chi) for nuval in nu_eval] )
    fnu = lambda nu: np.interp(nu, nu_eval, fnu_eval)
    
    # f_nu = np.vectorize(f_nu)
    nu = (mec2/(2*kTo*g*(1+z))).value
    bpp = (alph_re2me2 * kTo**2 / (hbc**3) / np.pi**2 * fnu( nu ) / g / (1+z) / mpc2).to('1/Mpc').value

    return bpp * Z**2/A * (1 + z)**3

def Bpp_crpropa(Z, A, g, z):
    """Compute pair production losses
    Based on CRPropa's implementation of the original formula:
        Blumenthal, G. R. (1970) PRD 1(6), 1596
        "Energy Loss of High-Energy Cosmic Rays in Pair-Producing Collisions with Ambient Photons."
        https://doi.org/10.1103/PhysRevD.1.1596
        B = -1/E dE/dt = -1/g dg/dt
        Values given in Mpc^-1
    """    
    with open('/home/leonel/Downloads/Bpp_data', 'rb') as fileobj:
        data = np.load(fileobj)

    return np.interp(g, data[0, :], Z**2/A * data[1, :] * (1 + z)**3)

def dlngdl_tot_proton(z, lng, Bpp=Bpp_Blumenthal):
    """Computes all CEL (adiabatic + pair production) 
       the derivate of the ln of the boost for protons
       as a function of comoving distance in Mpc^-1
    """
    return (Bpp(1, 1, np.exp(lng), z) + cosmo.H(z).value ) / c.to(u.km/u.s).value

def dlngdz_tot_proton(z, lng, Bpp=Bpp_Blumenthal):
    """Computes all CEL (adiabatic + pair production) 
       the derivate of the ln of the boost for protons 
       as a function of redshift
    """
    return -(1 + Bpp(1, 1, np.exp(lng), z) * c.to(u.km/u.s).value / cosmo.H(z).value) / (1+z)

def dlngdz_tot_nucleus(z, lng, Z, A, Bpp=Bpp_Blumenthal):
    """Computes all CEL (adiabatic + pair production) 
       the derivate of the ln of the boost for protons
    """
    return (1 - Z**2/A) / (1+z) + Z**2/A * dlngdz_tot_proton(z, lng, Bpp=Bpp)

def dlngdz_tot(z, lng, Z, A, Bpp=Bpp_Blumenthal):
    """All CEL (adiabatic + pair production) 
       the derivate of the ln of the boost
    """
    return (1 - c.to(u.km/u.s).value * Bpp(Z, A, np.exp(lng), z)/cosmo.H(z).value) / (1+z)

def dln_comg_dz(z, lng, Z=1, A=1, Bpp=Bpp_Blumenthal):
    """The effects of pair production on the comoving boost.
       Computes the derivate of the ln of the comoving boost.
    """
    return - c.to(u.km/u.s).value * Bpp(Z, A, np.exp(lng), z) / cosmo.H(z).value / (1+z)

dlngdz_tot = dlngdz_tot_nucleus
Bpp_none = lambda Z, A, g, z: np.zeros_like(g)  # no pair production losses
dlngdz_none = lambda z, lng, Z=1, A=1: dlngdz_tot_nucleus(z, lng, Z, A, Bpp=Bpp_none) # no pair production losses

def g_in_z(g0, Z=1, A=1, z0=2, dlngdz=dlngdz_tot_proton, lngcutoff=7, zcutoff=1e-7):
    """Computes the evolution of boost over redshift under the effect
    of continuous energy losses.
    Arguments:
    ----------
    g0: starting boost
    Z, A: Nuclear species charge and mass numbers
    z0: starting redshift
    dlngdz: a function describing the rate of boost reduction. 
            It should take (z, lng, Z, A) as parameters and return
            the derivative of lng on the redshift. By default, takes 
            dlngdz_tot which includes adiabatic and pair production losses.
    lngcutoff, zcutoff: minimal lng and redshift respectively where to stop 
            the computation. The computation stops at whichever is reached first.
    """

    z = np.logspace(np.log10(z0), np.log10(zcutoff))

    integrated_lng_proton = cumulative_trapezoid(dlngdz(z, np.log(g0)), z, initial=0)

    integrated_lng_nucleus_scaling = np.log(g0) + (1 - Z**2/A) * np.log((1+z)/(1+z[0])) + Z**2/A * (integrated_lng_proton[0]-integrated_lng_proton)

    return z, np.exp(integrated_lng_nucleus_scaling)

def Lprime_trapz(z):
    """This computes the interaction depth due to cosmological compresion 
    of the target photon fields. Exact formula numerically integrated.
    """
    d_H = cosmo.hubble_distance.value # Mpc
    return cumulative_trapezoid(-(1+z)**2 * d_H * cosmo.H0/cosmo.H(z), z, initial=0)

def g_in_z_evolution(g0, gf=1e6, z0=.1, zf=1e-7, Z=1, A=1):
    """Computes the evolution of comoving boost with redshift under
    the effect of pair production losses.
    """

    gvalues = np.logspace(np.log10(g0), np.log10(gf), 5000)

    log10z0, log10zf = np.log10(z0), np.log10(zf)
    zvalues = np.logspace(log10z0, log10zf, 5000)

    thickness = Z**2 / A * pthickness(10**log10z0, zvalues)

    B70zgrid = np.interp(universal_bh_loss_B70(gvalues, gvalues[0]), thickness, zvalues)

    return gvalues, B70zgrid


absolute_thickness = lambda z: np.interp(z, np.logspace(-7, 5, 1000), Lprime_trapz(np.logspace(-7, 5, 1000)))
pthickness = lambda z0, zf: absolute_thickness(zf) - absolute_thickness(z0)
thickness_to_lookback_distance = lambda th: np.interp(th, -absolute_thickness(np.logspace(-6, 3, 1000)), cosmo.lookback_distance(np.logspace(-6, 3, 1000)).value)
lookback_distance_to_thickness = lambda lbd: np.interp(lbd, cosmo.lookback_distance(np.logspace(-6, 3, 1000)).value, -absolute_thickness(np.logspace(-6, 3, 1000)))

gvgrid = np.logspace(13, 7, 5000)
funvals_B70 = cumulative_trapezoid(1 / gvgrid / Bpp_Blumenthal(1, 1, gvgrid, 0), gvgrid, initial=0) + 1 / gvgrid[0] / Bpp_Blumenthal(1, 1, gvgrid[0], 0)
universal_thickness_B70 = lambda g0, gf: np.interp(gf, gvgrid[::-1], funvals_B70[::-1]) - np.interp(g0, gvgrid[::-1], funvals_B70[::-1])