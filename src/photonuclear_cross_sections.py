"""Utility functions to model the total photonuclear cross section for any
nuclear species.
"""

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline
import os
main_path = os.path.abspath(__file__)

def pgamma(eps_r):
    """Photonuclear cross section in the energy range .1-1e4 GeV
    taken from Rachen PhD Thesis. 

    Returns the cross section in cm2.
    """
    exp = np.exp

    def Qf(eps_r, eps_th, w):
        out = np.zeros(eps_r.shape)
        out[np.argwhere(eps_r > w + eps_th)] = 1
        
        idcs = np.argwhere(eps_r <= w + eps_th)
        out[idcs] = (eps_r[idcs] - eps_th)/w

        return out
    
    def direct(eps_r):
        """Computes the direct contribution to the proton photonuclear cross section
        """
        def pl(eps_r, eps_th, eps_max, alpha):
            A = alpha * eps_max / eps_th
            out = np.zeros(eps_r.shape)
    
            idcs = np.argwhere(eps_r > eps_th)
            out[idcs] = ((eps_r[idcs] - eps_th)/(eps_max - eps_th))**(A-alpha)*(eps_max/eps_r[idcs])**A
            
            return out
    
        dir1 = 92.7*pl(eps_r, 0.152, 0.25, 2.0) + 40.0*exp(-(eps_r - 0.29)**2/0.002) - 15.*exp(-(eps_r - 0.37)**2/0.002)
        dir2 = 37.7*pl(eps_r, 0.4, 0.6, 2)

        return dir1 + dir2

    def multipion(eps_r):
        """Computes the multipion contribution to the proton photonuclear cross section
        """
        smp1 = 80.3*Qf(eps_r, 0.5, 0.1)*s**(-0.34)
        smp1[smp1 < 0] = 0
        smp2 = np.zeros(eps_r.shape)
        idcs = np.argwhere(eps_r > .85)
        smp2[idcs] = (1 - exp(-(eps_r[idcs] - 0.85)/0.69))*(29.3*s[idcs]**(-0.34) + 59.3*s[idcs]**0.095)
        
        return smp1 + smp2

    def resonances(eps_r):
        """Computes the multipion contribution to the proton photonuclear cross section
        """
        resdata = [
            [r"$Delta$(1232)", 31.125, 1.231 , 0.11, 0.17],
            ["N(1440)", 1.389, 1.440, 0.35, 0.38],
            ["N(1520)",  25.567, 1.515, 0.11, 0.38],
            ["N(1535)",  6.948, 1.525, 0.10 , 0.38],
            ["N(1650)", 2.779, 1.675, 0.16, 0.38],
            ["N(1680)", 17.508, 1.680, 0.125, 0.38],
            [r"$Delta$(1700)", 11.116, 1.690, 0.29, 0.38],
            [r"$Delta$(1905)", 1.667, 1.895, 0.35 , 0.38],
            [r"$Delta$(1950)", 11.116, 1.950, 0.30, 0.38 ]]
    
        def fun1(eps_r, sigma0, M, Gamma):
            fun0 = sigma0*Gamma**2*s**2/((s - M**2)**2 + Gamma**2*s)
    
            return fun0 / eps_r**2
    
        suma = sum([fun1(eps_r, sigma0, M, Gamma)*Qf(eps_r, 0.152, w) for _, sigma0, M, Gamma, w in resdata])
        
        return suma

    mubarn_to_cm2 = 1e-30
    mp = .938
    s = mp**2 + 2*mp*eps_r
    
    return mubarn_to_cm2 * (resonances(eps_r) + multipion(eps_r) + direct(eps_r))


def Spread_GDR(A, Z):
    """Literal quote from *"The width of the resonance is also
    strongly influencedby the shell structure of the nuclei.
    The systematics showvalues ranging from about 4-5 MeV for
    closed shell nucleiup to about 8 MeV for nuclei between
    closed shells."

    * data extracted from https://cyclotron.tamu.edu/wci3/newer/chapV_1.pdf
    """
    cshell_No = np.array([2., 8., 20., 28., 50., 82.])  # nuclear magic numbers
    dNo = cshell_No[1:] - cshell_No[:-1]

    if Z is None:
        Z = int(A / 2)

    N = A - Z
    G = 4.5 + 3. * (min(abs(N - cshell_No[:-1]) / dNo) +
                    min(abs(Z - cshell_No[:-1]) / dNo))  # based on docstring

    # print A, Z, G, 4.5 + 3. * (min(abs(N - cshell_No[:-1]) / dNo) +
    #                            min(abs(Z - cshell_No[:-1]) / dNo))
    return G


def Emax_GDR(A):
    """Returns energy of the GDR peak in MeV
    * data extracted from https://cyclotron.tamu.edu/wci3/newer/chapV_1.pdf
    """
    Em = 31.2 / A**(1. / 3) + 20.6 / A**(1. / 6)

    return Em


def Lorentzian(Evals, A, Z=None, Scale=1 / np.pi, TRKnorm=True):
    """
    TRKnorm: Can be normalized to comply with the Thomas-Reiche-Kuhn rule!
    A: nucleus
    Em: value for the peak in MeV
    Peak value is 2*Scale/pi/G.

    Returns: cross section in milibarns
    """
    Em = Emax_GDR(A)
    G = Spread_GDR(A, Z)
    cs = Scale * G / 2. / ((Evals - Em)**2 + G**2 / 4.)

    if Z is None:
        Z = int(A / 2.)
    N = A - Z

    if TRKnorm:
        x = np.linspace(1, 60, 90)
        y = Scale * G / 2. / ((x - Em)**2 + G**2 / 4.)
        cs *= 60. * N * Z / A / np.trapz(y, x)  # MeV * mb

    # print 60.*Z*N/A, np.trapz(cs, Evals)

    return cs


def get_interp_response_function(epsilon, cs):
    """Returns an interpolator with the response function

    Arguments:
    ----------
    epsilon : array, photon energies in nucleus rest frame desired units
    cs      : array, cross section values corresponding to the given energies
              in desired units
    """
    y = epsilon / 2.
    f = cumtrapz(epsilon * cs, x=epsilon, initial=0) / (2 * y**2)

    interp_f = InterpolatedUnivariateSpline(y, f, ext=1)  # ext=1 to return
    # zeros outside of range

    return (y[0], y[-1]), interp_f


def universal_function(energy_grid):
    """Returns the universal function on a fixed energy range
    """
    from pickle import load as pickle_load
    from scipy.interpolate import UnivariateSpline

    with open('EXFOR_data/universal-spline.pkl', 'rb') as f:
        tck = pickle_load(f, encoding='latin1')

    egrid = energy_grid[energy_grid < 1.9]
    egrid = egrid[egrid > .2]  # hardcoded! improve later

    univ_spl = UnivariateSpline._from_tck(tck)

    cs_univ = univ_spl(egrid)
    e_min, e_max = egrid[0], egrid[-1]

    return e_min, e_max, cs_univ


def cs_photomeson(Evals, A):
    """Models the nonelastic photonuclear cross section in the photomeson region
    and returns it in cm2 units.
    For nucleons (A = 1) returns a model of the resonance region which differs
    from the nucleus' cross section.

    Arguments:
    ----------

    """    
    cs_grid = pgamma(Evals)

    if A > 1:
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        path_to_file = os.path.join(main_path, 'data/universal-spline.pkl')
        with open(path_to_file, 'rb') as f:
            tck = pickle_load(f, encoding='latin1')
    
        univ_spl = UnivariateSpline._from_tck(tck)
        
        idcs = np.argwhere((.2 < Evals) * (Evals < 1.9))  # selecting resonance regions
        cs_grid[idcs] = univ_spl(Evals[idcs])  # univ function for nuclei

    return A * cs_grid


def cs_photodisinteg(Evals, A, Z):
    """wrapper function to get model of photodisintegration cross section
    takes the energy in GeV and returns in cm2
    """
    return Lorentzian(Evals * 1e3, A, Z) * 1e-27


def cs_photonuclear(Evals, A, Z):
    """Returns the photonuclear cross section by combining the the photodisintegration
       and the photomeson regions.

       Parameters
       ----------
       Evals : array-like
            photon energies in the nucleus rest frame in GeV
       A, Z  : integers
            mass and atomic number correspondingly 
    """
    idcs_pd = np.argwhere(Evals <= .21)  # .21 GeV point separating photodis and photomes
    idcs_pm = np.argwhere(.21 < Evals )  # .21 GeV point separating photodis and photomes

    # if idcs_pd
    if len(idcs_pd) == 0:
        cs_pdis = np.array([])
    else:
        cs_pdis = cs_photodisinteg(Evals[idcs_pd], A, Z).flatten()
    
    if len(idcs_pm) == 0:
        cs_pmes = np.array([])
    else:
        cs_pmes = cs_photomeson(Evals[idcs_pm], A).flatten()

    return np.concatenate([cs_pdis, cs_pmes])


def main():
    import matplotlib.pyplot as plt
    e = np.linspace(.21, 1.89, 50)
    e = np.logspace(-1, 4, 1000)
    plt.plot(e, cs_photomeson(e, 14) / 14., label='Nitrongen (A=14)')
    plt.plot(e, cs_photomeson(e, 1), label='Nucleon (A=1)')
    
    plt.semilogx()
    plt.xlabel('E GeV')
    plt.ylabel(r'$\sigma / A {\rm [cm^2]}$')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
