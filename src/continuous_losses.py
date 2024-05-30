import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
# from astropy.cosmology import z_at_value
from astropy.constants import c, m_e, alpha, e, eps0

def dlngdl_tot_proton(z, lng):
    """Computes all CEL (adiabatic + pair production) 
       the derivate of the ln of the boost for protons
       as a function of comoving distance in Mpc^-1
    """
    return (Bpp(1, 1, np.exp(lng), z) + cosmo.H(z).value ) / c.to(u.km/u.s).value

def dlngdz_tot_proton(z, lng):
    """Computes all CEL (adiabatic + pair production) 
       the derivate of the ln of the boost for protons 
       as a function of reddshift
    """
    return -(1 + Bpp(1, 1, np.exp(lng), z) * c.to(u.km/u.s).value / cosmo.H(z).value) / (1+z)

def dlngdz_tot_nucleus(z, lng, Z, A):
    """Computes all CEL (adiabatic + pair production) 
       the derivate of the ln of the boost for protons
    """
    return (1 - Z**2/A) / (1+z) + Z**2/A * dlngdz_tot_proton(z, lng)

def dlngdz_tot(z, lng, Z, A):
    """All CEL (adiabatic + pair production) 
       the derivate of the ln of the boost
    """
    return (1 - c.to(u.km/u.s).value * Bpp(Z, A, np.exp(lng), z)/cosmo.H(z).value) / (1+z)

dlngdz_tot = dlngdz_tot_nucleus

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
    from scipy.integrate import cumtrapz

    z = np.logspace(np.log10(z0), np.log10(zcutoff))

    integrated_lng_proton = cumtrapz(dlngdz(z, np.log(g0)), z, initial=0)

    integrated_lng_nucleus_scaling = np.log(g0) + (1 - Z**2/A) * np.log((1+z)/(1+z[0])) + Z**2/A * (integrated_lng_proton[0]-integrated_lng_proton)

    return z, np.exp(integrated_lng_nucleus_scaling)


def my_g_in_z(g0, Z=1, A=1, z0=2, dlngdz=dlngdz_tot, lngcutoff=7, zcutoff=1e-7):
    """Computes the evolution of boost over redshift under the effect
    of continuous energy losses.

    !!! This is possibly wrong for nuclei !!!

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
    lng0 = np.log(g0)
    zvals = [z0]
    dz=1e-3
    lngvals = [lng0]
    
    for i in range(int((1/dz - 1)*2)):
        zvals.append(zvals[-1] - dz)
        lngvals.append(lngvals[-1] + dz*dlngdz(.2, lngvals[-1], Z, A))

        if (lngvals[-1] < lngcutoff) or (zvals[-1] < zcutoff):
            break
    gvals = np.exp(np.array(lngvals))

    return zvals, gvals

def Badiab(z):
    """Compute adiabatic losses
       B = -1/g dg/dt
    """
    B = cosmo.H(z) / c.to(u.km/u.s) # 1/Mpc
     
    return B.value

def Bpp_Blumenthal(Z, A, g, z=0):
    """Compute pair production losses
    Based on the paper:
        Blumenthal, G. R. (1970) PRD 1(6), 1596
        "Energy Loss of High-Energy Cosmic Rays in Pair-Producing Collisions with Ambient Photons."
        https://doi.org/10.1103/PhysRevD.1.1596
        B = -1/g dg/dt = 
    """
    hbc = 0.1973269804e-6 # eV*m
    kTo = 8.617333262e-5 * 2.71 #eV
    alph_re2me2 = alpha*r_e**2*(m_e.to(u.eV * u.s**2 / u.m**2) * c**2)**2
    r_e = e.si**2 / (4*np.pi*eps0.to(u.C**2/u.eV/u.m) * m_e.to(u.eV * u.s**2/u.m**2) * c**2)
    
    from scipy.integrate import quad

    phi_in_xi = lambda xi: xi*np.polyval([2.667, -14.45, 50.95, -86.07], np.log(xi))

    def f_nu(nu):
        """Calculating the function f(nu) from Blumenthal
        """
        integrand_nu = lambda xi: phi_in_xi(xi)/(np.exp(nu*xi) - 1)
        return nu**2*quad(integrand_nu, 20, np.inf)[0]

    f_nu = np.vectorize(f_nu)

    rate = alph_re2me2.value * kTo**2 / (hbc**3) / np.pi**2 * 1e-2 * f_nu(5.11e5/(2*kTo*g)) / g / 1e9 / 3.24e-25 # 1/Mpc

    return rate * Z**2/A

def Bpp(Z, A, g, z=0):
    """Compute pair production losses
       Based on tabulated data from CRPRopa
        B = -1/g dg/dt = 
       Values given in Mpc^-1
    """
    with open('/home/leonel/Downloads/Bpp_data', 'rb') as fileobj:
        data = np.load(fileobj)

    interpolator = interp1d(data[0, :], data[1, :], kind='linear', fill_value=(data[1, 0], data[1, -1]), bounds_error=False)

    return Z**2/A * interpolator((1+z)*g) * (1+z)**3