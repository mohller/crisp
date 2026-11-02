"""Continuous energy-loss (CEL) rates: Bethe-Heitler pair production and
the redshift/boost evolution built on it.

Where the rest of the package treats photodisintegration and photomeson
interactions as discrete, boost-changing events handled by
`InteractionCore`'s transport tensors, pair production on the photon
field is smooth enough on cosmic-ray timescales to be modeled as a
continuous drag on the Lorentz factor instead. `Bpp_generic` computes
that drag, B = -(1/gamma) dgamma/dt in Mpc^-1, for an arbitrary target
photon field; `Bpp_Blumenthal` specializes it to the CMB with a
precomputed lookup (much faster, since the CMB shape never changes);
`Bpp_crpropa` reads a CRPropa-exported table for a cross-check.

The `dlngdz_*` / `dlngdl_tot_proton` functions combine that pair-production
drag with the adiabatic (Hubble) term into a single d(ln gamma)/dz or
d(ln gamma)/dl derivative, which `g_in_z` integrates to trace how a
particle's boost falls off as it propagates from injection redshift z0
down to z ~ 0. `Lprime_trapz` and the `thickness`/`pthickness` family
compute the corresponding comoving path length (interaction depth) as a
function of redshift; `extragalactic.py` pins its own effective
thickness against `Lprime_trapz` for the CMB's coherent-inhomogeneity
treatment (see that module's docstring), and `source_models.py`'s
`loss_rates(include_pair=True)` calls `Bpp_generic` for the
pair-production rate on a source's own photon field.

Correspondence with the methods paper (Morejon & Kampert, Sect. 3):
Eq. 33 is `Bpp_Blumenthal`/`Bpp_generic` (Z^2/A times the proton rate,
Eq. 34); Eq. 35, the physical-boost form with the adiabatic term, is the
`dlngdz_*` family integrated by `g_in_z`; Eq. 27 (thickness) is
`Lprime_trapz`; Eq. 37, the separated form Phi(gamma_c2) - Phi(gamma_c1)
= Z^2/A delta, is `universal_thickness_B70` (Phi at z = 0) applied over a
whole path by `g_in_z_evolution`, and `bh_log_boost_shift` (Phi at any
fixed z, inverted). The propagator in `extragalactic.py` works with the
comoving form, Eq. 36, where the adiabatic term is removed exactly by
the relabel gamma_c = gamma/(1+z): it therefore never needs the Eq.-35
functions (they remain as diagnostics), and applies Eq. 37 per segment
through `bh_log_boost_shift`.

Most users will not call into this module directly.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid, solve_ivp
from scipy.interpolate import CubicSpline
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c, m_e, alpha, e, eps0, k_B, hbar, m_p

from .background_photon_models import To as T_CMB   # 2.725 K, the package-wide CMB temperature

# Chodorowski, Zdziarski & Sikora (1992) fits to phi(kappa), kappa = 2 gamma eps / m_e c^2 >= 2
_phi_lo = lambda k: np.pi / 12 * (k - 2)**4 / (1 + np.polyval([-3.879e-6, 1.137e-3, 0.1459, 0.8048, 0], k - 2))
_phi_B70 = lambda x: x * np.polyval([8/3., -14.45, 50.96, -86.07], np.log(x))
_phi_hi = lambda x: _phi_B70(x) / (1 - np.polyval([1837, 78.35, 2.91, 0], 1 / x))
_phi = lambda x: np.where(x < 25, _phi_lo(x), _phi_hi(np.maximum(x, 25)))

_MEC2 = (m_e * c**2).to('eV').value
_R_E = e.si**2 / (4*np.pi*eps0.to(u.C**2/u.eV/u.m) * m_e.to(u.eV * u.s**2/u.m**2) * c**2)
# B = _K_CMB * T^2 * f(nu) / (g (1+z)) * (1+z)^3 * Z^2/A, nu = m_e c^2 / (2 k T g (1+z))   [1/Mpc, T in K]
_K_CMB = (alpha * _R_E**2 * (m_e * c**2)**2 * (k_B * u.K)**2 / (hbar * c)**3 / np.pi**2
          / (m_p * c**2)).to('1/Mpc').value
_KB_EV = (k_B * u.K).to('eV').value


def _cmb_f_table(lnu_min=np.log(1e-9), lnu_max=np.log(60.), n_nu=500):
    """f(nu) = nu^2 int_2^inf phi(chi) / (e^{nu chi} - 1) dchi on a log-nu grid,
    integrated in ln(chi - 2) so the (chi - 2)^4 threshold region is resolved."""
    lnu = np.linspace(lnu_min, lnu_max, n_nu)
    s = np.linspace(np.log(1e-4), np.log(1e11), 6000)
    chi = 2 + np.exp(s)
    w = _phi(chi) * np.exp(s)                                       # dchi = (chi - 2) ds
    x = np.exp(lnu)[:, None] * chi[None, :]
    occ = np.exp(-x) / -np.expm1(-x)                                # 1/(e^x - 1) without overflow
    f = np.exp(2 * lnu) * trapezoid(occ * w[None, :], s, axis=1)
    return lnu, np.log(f)


_LNU, _LNF = _cmb_f_table()
_LNF_SPLINE = CubicSpline(_LNU, _LNF)


def _cmb_f(nu):
    """f(nu) from the cached table: 0 above the table (B < 1e-40 /Mpc there), and the
    first-bin value continued as nu^2 ln^3 behaviour is irrelevant below nu = 1e-9
    (gamma > 1e18)."""
    lnu = np.log(np.asarray(nu, dtype=float))
    f = np.exp(_LNF_SPLINE(np.clip(lnu, _LNU[0], _LNU[-1])))
    return np.where(lnu > _LNU[-1], 0.0, f)


def Bpp_generic(Z, mA, g, z=0, phot_dens=None, eps_max=1e4, n_eps=300):
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
        eps_max: upper photon energy of the integral [eV] (raise it for X-ray/gamma-ray source fields)
        n_eps: integration points, log-spaced from the threshold m_e c^2 / g to eps_max
    """
    mec2 = (m_e * c**2).to('eV')
    mpc2 = (m_p * c**2).to('eV')
    g = np.atleast_1d(np.asarray(g, dtype=float))

    integral_grid = []
    for gval in g / mec2.value:
        lnu = np.linspace(np.log(1 / gval), np.log(max(eps_max, 2 / gval)), n_eps)
        nu = np.exp(lnu)
        # in ln(eps): dnu = nu dln(nu)
        integral_grid.append(trapezoid(phot_dens(nu, z) * _phi(2*gval*nu) / nu, lnu))

    bpp = mec2.value**2 * (alpha*_R_E**2 / u.m**3 * mec2 / mpc2 / g**2 / 2 * np.array(integral_grid)).to('1/Mpc').value

    return bpp * Z**2/mA

def Bpp_Blumenthal(Z, A, g, z=0, T0=T_CMB):
    """Compute pair production losses for interactions with the CMB
    Based on the papers:
        Blumenthal, G. R. (1970) PRD 1(6), 1596
        "Energy Loss of High-Energy Cosmic Rays in Pair-Producing Collisions with Ambient Photons."
        https://doi.org/10.1103/PhysRevD.1.1596
        "Reaction rate and energy loss rate for photopair production by relativistic nuclei"
        M. CHodorowski, A. Zdziarski, M. Sikora, ApJ 400, 181-185, 1992

        B = -1/E dE/cdt = -1/g dg/cdt
        Values given in Mpc^-1

        T0: CMB temperature today [K]; defaults to background_photon_models.To so the
            pair losses see the same CMB as the photonuclear tensors.
        The universal function f(nu) is tabulated once at import (cubic spline in
        ln f vs ln nu, threshold resolved), so a call costs microseconds.
    """
    g = np.asarray(g, dtype=float)
    nu = _MEC2 / (2 * _KB_EV * T0 * g * (1 + z))
    bpp = _K_CMB * T0**2 * _cmb_f(nu) / g / (1 + z)

    return bpp * Z**2/A * (1 + z)**3


def _cmb_H_table(n=20000):
    """H(nu) = int dln(nu) / (nu f(nu)) on a fine grid: the characteristic of
    d ln(gamma)/dl = -Bpp(gamma) at fixed z, since d nu/dl = K' (1+z)^3 Z^2/A nu^2 f(nu)."""
    lnu = np.linspace(_LNU[0], _LNU[-1], n)
    integrand = np.exp(-lnu) / _cmb_f(np.exp(lnu))
    H = cumulative_trapezoid(integrand, lnu, initial=0) + integrand[0]   # offset > 0 for log interp
    return lnu, np.log(H)


_H_LNU, _H_LNH = _cmb_H_table()


def bh_log_boost_shift(Z, A, g, z, L, T0=T_CMB):
    """Exact decrease of ln(gamma) over a proper length L [Mpc] at fixed redshift z
    under CMB pair production alone, i.e. the solution of d ln(g)/dl = -Bpp(Z, A, g, z)
    (not the one-step Euler value Bpp(g) * L, which is off by tens of percent once
    Bpp L >~ 0.2, e.g. 50-200 Mpc segments at z >~ 1).

    This is paper Eq. 37 at fixed z, Phi(gamma_c2) - Phi(gamma_c1) = Z^2/A delta
    with delta = (1+z)^3 L, inverted for the final boost: H(nu) below is Phi
    written in nu = m_e c^2 / (2 k T0 (1+z) g), which absorbs the (1+z)^2
    blueshift of Eq. 36 so one table serves every redshift.

    Returns ln(g_start / g_end), same shape as g (in boost-grid bins: divide by dln).
    Particles whose characteristic leaves the table (B < 1e-40 /Mpc) are frozen there."""
    g = np.asarray(g, dtype=float)
    nu = _MEC2 / (2 * _KB_EV * T0 * g * (1 + z))
    lnu = np.log(nu)
    # d nu / dl = Kp nu^2 f(nu), Kp = (1+z)^3 Z^2/A _K_CMB T0^2 * 2 k T0 / m_e c^2
    Kp = (1 + z)**3 * Z**2 / A * _K_CMB * T0**2 * 2 * _KB_EV * T0 / _MEC2
    lnH0 = np.interp(np.clip(lnu, _H_LNU[0], _H_LNU[-1]), _H_LNU, _H_LNH)
    lnH1 = np.logaddexp(lnH0, np.log(Kp * L)) if L > 0 else lnH0
    lnu1 = np.interp(lnH1, _H_LNH, _H_LNU)
    return np.where(lnu >= _H_LNU[-1], 0.0, np.maximum(lnu1 - lnu, 0.0))

def Bpp_crpropa(Z, A, g, z):
    """Compute pair production losses based on CRPropa's tabulated data.

    This function requires an external binary data file (``Bpp_data``) exported
    from CRPropa.  Pass the path to that file via the ``data_path`` argument,
    or set the environment variable ``CRISP_BPP_DATA`` before importing.

    Parameters
    ----------
    Z, A : int
        Charge and mass numbers of the nucleus.
    g : array_like
        Lorentz boost(s).
    z : float
        Redshift.

    Raises
    ------
    FileNotFoundError
        If the Bpp_data file cannot be found.
    """
    import os
    data_path = os.environ.get('CRISP_BPP_DATA', '')
    if not data_path or not os.path.isfile(data_path):
        raise FileNotFoundError(
            "Bpp_crpropa requires an external CRPropa data file.\n"
            "Set the environment variable CRISP_BPP_DATA to its path, e.g.:\n"
            "  export CRISP_BPP_DATA=/path/to/Bpp_data"
        )
    with open(data_path, 'rb') as fileobj:
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
       the derivate of the ln of the boost for a nucleus (Z, A), same sign
       convention as dlngdz_tot_proton: -(1 + Z^2/A c Bpp_p / H) / (1+z).
       Only the pair term scales with Z^2/A; the adiabatic -1/(1+z) does not.
       This is paper Eq. 35 written per unit redshift (a diagnostic: the
       propagator uses the comoving Eq. 36/37 form instead, see module doc).
    """
    return (Z**2/A - 1) / (1+z) + Z**2/A * dlngdz_tot_proton(z, lng, Bpp=Bpp)

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

    The proton derivative is evaluated along the trajectory (ODE solve), not at
    the injection boost g0: freezing Bpp at g0 overestimates the losses once the
    boost falls towards the pair threshold (a factor ~40 in g(0) for g0 = 1e10
    injected at z = 1).
    """

    z = np.logspace(np.log10(z0), np.log10(zcutoff))
    k = Z**2/A

    # true d ln g / dz = (1 - k)/(1+z) - k * dlngdz_proton  (dlngdz_* use the opposite sign)
    rhs = lambda zz, y: [(1 - k) / (1 + zz) - k * float(np.squeeze(dlngdz(zz, y[0])))]
    stop = lambda zz, y: y[0] - lngcutoff
    stop.terminal = True
    sol = solve_ivp(rhs, (z[0], z[-1]), [np.log(g0)], t_eval=z, events=stop,
                    rtol=1e-6, atol=1e-9)

    return sol.t, np.exp(sol.y[0])

def Lprime_trapz(z):
    """This computes the interaction depth due to cosmological compresion 
    of the target photon fields. Exact formula numerically integrated.
    """
    d_H = cosmo.hubble_distance.value # Mpc
    return cumulative_trapezoid(-(1+z)**2 * d_H * cosmo.H0/cosmo.H(z), z, initial=0)

def g_in_z_evolution(g0, gf=1e6, z0=.1, zf=1e-7, Z=1, A=1):
    """Computes the evolution of comoving boost with redshift under
    the effect of pair production losses: paper Eq. 37 applied over the
    whole path, Z^2/A delta(z0, z) = Phi(gamma_c) - Phi(g0), inverted for
    z on a grid of gamma_c from g0 down to gf.

    Approximation: Phi is the z = 0 function (`universal_thickness_B70`),
    while the exact CMB scaling evaluates the rate at (1+z)^2 gamma_c
    (Eq. 36), so the result is exact only near z = 0. For a z-exact
    evolution chain `bh_log_boost_shift` over short segments, as
    `ExtragalacticPropagation` does.
    """

    gvalues = np.logspace(np.log10(g0), np.log10(gf), 5000)

    log10z0, log10zf = np.log10(z0), np.log10(zf)
    zvalues = np.logspace(log10z0, log10zf, 5000)

    thickness = Z**2 / A * pthickness(10**log10z0, zvalues)

    B70zgrid = np.interp(universal_thickness_B70(gvalues, gvalues[0]), thickness, zvalues)

    return gvalues, B70zgrid


absolute_thickness = lambda z: np.interp(z, np.logspace(-7, 5, 1000), Lprime_trapz(np.logspace(-7, 5, 1000)))
pthickness = lambda z0, zf: absolute_thickness(zf) - absolute_thickness(z0)
thickness_to_lookback_distance = lambda th: np.interp(th, -absolute_thickness(np.logspace(-6, 3, 1000)), cosmo.lookback_distance(np.logspace(-6, 3, 1000)).value)
lookback_distance_to_thickness = lambda lbd: np.interp(lbd, cosmo.lookback_distance(np.logspace(-6, 3, 1000)).value, -absolute_thickness(np.logspace(-6, 3, 1000)))

# Phi of paper Eq. 37 for protons at z = 0, Phi(g) = int dg / (g Bpp(g)) [Mpc]
# (bh_log_boost_shift holds the z-general version)
# (floor at gamma = 10^7.5: below it the CMB rate is < 1e-14 /Mpc and reaches
# exactly zero past the f(nu) table, where 1/Bpp would be infinite)
gvgrid = np.logspace(13, 7.5, 5000)
funvals_B70 = cumulative_trapezoid(1 / gvgrid / Bpp_Blumenthal(1, 1, gvgrid, 0), gvgrid, initial=0) + 1 / gvgrid[0] / Bpp_Blumenthal(1, 1, gvgrid[0], 0)
universal_thickness_B70 = lambda g0, gf: np.interp(gf, gvgrid[::-1], funvals_B70[::-1]) - np.interp(g0, gvgrid[::-1], funvals_B70[::-1])