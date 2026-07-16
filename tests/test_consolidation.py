"""Consolidation suite for the unified InteractionCore.

History: phases 1-4 built the argument-driven InteractionCore and proved every
legacy class equal to its unified configuration at machine precision (the
equivalence report lives in the git history of this file). In phase 5 the
legacy classes became deprecated thin wrappers (CRPropa scaffolding, the
UHECR_Source pair and the GDRA core were deleted), so this suite now checks:

  1. the wrappers delegate exactly to the unified construction and emit
     DeprecationWarning;
  2. the physics invariants and absolute normalizations that gated the
     migration (conservation, tally identities, GZK photopion benchmark,
     mass-independence of the boost-native tensors).

Run with pytest, or standalone:  python tests/test_consolidation.py
"""
import sys
import os
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crisp.core import (InteractionCore, InteractionCore_CRPdata_CMB,
                        InteractionCore_CRPdata_EBL, InteractionCore_Source,
                        InteractionCore_PSB_CMB, InteractionCore_SimProp_CMB)
from crisp.photonuclear_cross_sections import PSB_model, SimProp_model
from crisp.background_photon_models import cmb_photon_density_GeVcm3
from crisp.data.nucleardecays import NuclearDataTable, nuclear_mass_GeV

A_LIGHT = np.array([4., 3., 3., 2., 1., 1.])
Z_LIGHT = np.array([2., 2., 1., 1., 1., 0.])

MACHINE = 1e-10   # "machine precision" pass threshold for relative differences
CONSERVE = 1e-12  # static A/Z conservation threshold

_cache = {}

def cached(key, builder):
    if key not in _cache:
        _cache[key] = builder()
    return _cache[key]

def decay_table():
    return cached('decays', lambda: NuclearDataTable().prepare_decay_table())

def psb_xsec():
    return cached('psb_model', PSB_model)

def simprop_xsec():
    return cached('simprop_model', lambda: SimProp_model(M=2))

def rel_diff(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(b).max(), 1e-300)

def static_imbalance(core):
    """(A, Z) conservation imbalance of tensor + light_prod_tensor, relative.

    Photomeson-ejected nucleons are budgeted in core.photomeson_ejecta (their
    spectral placement lives in the recoil kernel) — the budget closes the
    static balance of the boost-preserving matrices."""
    A_sp = np.array([s[1] for s in core.species], float)
    Z_sp = np.array([s[0] for s in core.species], float)
    scale = np.abs(core.tensor).max()
    ej = getattr(core, 'photomeson_ejecta', None)
    if ej is not None:
        for ni, nuc in enumerate(core.nuclei):
            si = core.species.index(tuple(nuc))
            imbA[si] += ej['p'][ni] + ej['n'][ni]
            imbZ[si] += ej['p'][ni]
    imbA = (np.einsum('j,ijb->ib', A_sp, core.tensor)
            + np.einsum('l,lijb->ib', A_LIGHT, core.light_prod_tensor))
    imbZ = (np.einsum('j,ijb->ib', Z_sp, core.tensor)
            + np.einsum('l,lijb->ib', Z_LIGHT, core.light_prod_tensor))
    return np.abs(imbA).max() / scale, np.abs(imbZ).max() / scale

def deprecated(builder):
    """Construct via builder asserting a DeprecationWarning is emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        obj = builder()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
        'wrapper did not emit DeprecationWarning'
    return obj

def check_pair(old, new, label, tol=MACHINE):
    """Assert tensor-level equivalence of two cores and report the numbers."""
    assert new.species == old.species, f'{label}: species lists differ'
    assert sorted(new.nuclei) == sorted(old.nuclei), f'{label}: nuclei sets differ'
    idx = [new.nuclei.index(nuc) for nuc in old.nuclei]
    d_rates = rel_diff(new.all_rates[idx], old.all_rates)
    d_tensor = rel_diff(new.tensor, old.tensor)
    d_light = rel_diff(new.light_prod_tensor, old.light_prod_tensor)
    print(f'  {label}: all_rates {d_rates:.2e}, tensor {d_tensor:.2e}, light {d_light:.2e}')
    assert d_rates < tol and d_tensor < tol and d_light < tol, f'{label}: tensors differ'
    for core, tag in [(old, 'old'), (new, 'new')]:
        a, z = static_imbalance(core)
        assert a < CONSERVE and z < CONSERVE, f'{label} ({tag}): conservation violated'


# ------------------------------------------------------------- wrapper delegation

def test_wrapper_psb_cmb():
    """InteractionCore_PSB_CMB delegates to InteractionCore(xsec_model=PSB_model())."""
    wrapper = deprecated(lambda: InteractionCore_PSB_CMB())
    direct = cached('new_psb_nubase', lambda: InteractionCore(
        xsec_model=psb_xsec(), decays=decay_table()))
    # the wrapper defaults to decays=None (builtin valley fallback); compare a
    # like-for-like direct construction
    direct_nodecays = InteractionCore(xsec_model=PSB_model())
    check_pair(wrapper, direct_nodecays, 'PSB wrapper vs direct')
    # physics carried over from the completion: PSB now emits secondary He4
    assert np.abs(direct.light_prod_tensor[0]).max() > 0


def test_wrapper_simprop_cmb():
    """InteractionCore_SimProp_CMB(M=2) delegates to the unified construction."""
    wrapper = deprecated(lambda: InteractionCore_SimProp_CMB(M=2))
    direct = InteractionCore(xsec_model=SimProp_model(M=2))
    check_pair(wrapper, direct, 'SimProp wrapper vs direct')


def test_wrapper_crpdata_cmb():
    """InteractionCore_CRPdata_CMB(xsec_model=m) delegates to InteractionCore(xsec_model=m)."""
    m, d = simprop_xsec(), decay_table()
    wrapper = deprecated(lambda: InteractionCore_CRPdata_CMB(xsec_model=m, decays=d))
    direct = cached('new_crpdata', lambda: InteractionCore(xsec_model=m, decays=d))
    check_pair(wrapper, direct, 'CRPdata_CMB wrapper vs direct')


def test_wrapper_crpdata_ebl():
    """InteractionCore_CRPdata_EBL(z=0) delegates with the same EBL photon field."""
    try:
        from crisp.background_photon_models import eblg_interp
        ebl = lambda e: eblg_interp(e * 1e9, 0).flatten() * 1e3
        ebl(np.array([1e-9, 1e-8]))
    except Exception as exc:
        raise unittest.SkipTest(f'EBL data unavailable: {exc}')

    m, d = psb_xsec(), decay_table()
    wrapper = deprecated(lambda: InteractionCore_CRPdata_EBL(xsec_model=m, z=0, decays=d))
    direct = InteractionCore(xsec_model=m, target_photons=ebl, decays=d)
    check_pair(wrapper, direct, 'CRPdata_EBL wrapper vs direct')


def source_pair():
    """Source wrapper and its direct unified equivalent (same grids/photomeson)."""
    m, d = psb_xsec(), decay_table()
    old = cached('old_source', lambda: deprecated(lambda: InteractionCore_Source(
        epsrange=(1e-12, 1e-2), target_photon_spectrum=cmb_photon_density_GeVcm3,
        xsec_model=m, decays=d)))
    new = cached('new_source', lambda: InteractionCore(
        xsec_model=m, target_photons=cmb_photon_density_GeVcm3,
        photomeson='kernels', decays=d,
        boosts=np.logspace(0, 12, 131), eps=np.logspace(-2, 6, 300)))
    return old, new


def test_wrapper_source():
    """InteractionCore_Source delegates, including the photomeson kernels."""
    old, new = source_pair()
    check_pair(old, new, 'Source wrapper vs direct')
    assert rel_diff(new.pion_prod_tensor, old.pion_prod_tensor) < MACHINE
    assert rel_diff(new.proton_recoil_tensor, old.proton_recoil_tensor) < MACHINE


def test_source_production_methods():
    """pion_production / proton_recoil_production agree between the wrapper and
    the base-class implementations."""
    old, new = source_pair()
    alpha, mr, tr, _ = old.get_distribution_parameters(
        mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        absorption_type=('only mass', [1]))
    br = old.boosts[100:120]
    L = np.linspace(0.0, 50.0, 6)
    kw = dict(alpha=alpha, mass_range=mr, boost_range=br, true_range=tr)

    d_pion = rel_diff(new.pion_production(L, **kw), old.pion_production(L, **kw))
    d_rec = rel_diff(new.proton_recoil_production(L, **kw),
                     old.proton_recoil_production(L, **kw))
    print(f'  pion_production {d_pion:.2e}, proton_recoil_production {d_rec:.2e}')
    assert d_pion < MACHINE and d_rec < MACHINE

    # the P= fast path (reusing a precomputed evolution) must reproduce the
    # internally solved one
    Pev = new.species_evolution_boost_range(L, alpha, mr, br, tr)
    d_p1 = rel_diff(new.pion_production(L, P=Pev, **kw), new.pion_production(L, **kw))
    d_p2 = rel_diff(new.proton_recoil_production(L, P=Pev, **kw),
                    new.proton_recoil_production(L, **kw))
    d_p3 = rel_diff(new.light_secondaries_production(L, P=Pev, **kw),
                    new.light_secondaries_production(L, **kw))
    print(f'  P= reuse: pion {d_p1:.2e}, recoil {d_p2:.2e}, light {d_p3:.2e}')
    assert d_p1 < MACHINE and d_p2 < MACHINE and d_p3 < MACHINE


# ------------------------------------------------------------------ mass claims

def test_tensors_independent_of_masses():
    """The boost-native tensors must not depend on the masses= choice."""
    m, d = simprop_xsec(), decay_table()
    nub = cached('new_crpdata', lambda: InteractionCore(xsec_model=m, decays=d))
    leg = InteractionCore(xsec_model=m, decays=d, masses='legacy')
    assert np.array_equal(nub.tensor, leg.tensor), 'tensor depends on masses='
    assert np.array_equal(nub.light_prod_tensor, leg.light_prod_tensor)
    print('  tensors bit-identical between masses=nubase and masses=legacy')


def test_species_masses_and_helpers():
    """nubase masses are ~0.7-1% below A*0.939; helpers round-trip."""
    core = cached('new_crpdata', lambda: InteractionCore(
        xsec_model=simprop_xsec(), decays=decay_table()))
    approx = np.array([A * 0.939 for _, A in core.species])
    ratio = core.species_masses / approx
    print(f'  m_nubase / (A*0.939): min {ratio.min():.5f}, max {ratio.max():.5f}')
    assert 0.985 < ratio.min() and ratio.max() < 1.005
    assert np.isclose(nuclear_mass_GeV(1, 1), 0.9382720, atol=1e-5)
    assert np.isclose(nuclear_mass_GeV(0, 1), 0.9395654, atol=1e-5)

    gamma = 3.7e9
    E = core.energy_of_boost((26, 56), gamma)
    assert np.isclose(core.boost_of_energy((26, 56), E), gamma, rtol=1e-14)
    print(f'  E(Fe56, {gamma:.1e}) = {E:.4e} GeV; round-trip exact')


# --------------------------------------------------------------- physics battery

def test_dynamic_three_way_balance():
    """Unified PSB + decays: heavy + light + absorbed = 56 through propagation."""
    core = cached('new_psb_nubase', lambda: InteractionCore(
        xsec_model=psb_xsec(), decays=decay_table()))
    alpha, mr, tr, _ = core.get_distribution_parameters(
        mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        absorption_type=('only mass', [1]))
    A_sp = np.array([s[1] for s in core.species], float)
    A_tr = A_sp[[mr.index(v) for v in tr]]
    br = core.boosts[80:120]
    L = np.concatenate([[0.0], np.geomspace(1e-2, 500, 50)])
    H = core.species_evolution_boost_range(L, alpha, mr, br, tr)
    N = core.light_cascade_production(L, alpha, mass_range=mr, boost_range=br, true_range=tr)
    tot = (np.einsum('i,bli->bl', A_tr, H[:, :, :-1])
           + np.einsum('l,lbi->bi', A_LIGHT, N) + (-H[:, :, -1]))
    res = np.abs(tot - 56.0).max() / 56.0
    print(f'  dynamic three-way balance residual: {res:.2e}')
    assert res < 1e-8


def test_light_secondaries_rate_matches_cumulative_emission():
    """Integrating light_secondaries_production along L must reproduce the exact
    cumulative 'emission' tally of light_production_cumulative. (Pins a
    rate-squaring bug found via the GRB template: the yield tensor already
    contains the channel rates, and was being multiplied by the rate matrix
    again, giving Mpc^-2 'rates' -- a factor ~10^12 in dense photon fields.)"""
    from scipy.integrate import cumulative_trapezoid
    core = cached('new_psb_nubase', lambda: InteractionCore(
        xsec_model=psb_xsec(), decays=decay_table()))
    alpha, mr, tr, _ = core.get_distribution_parameters(
        mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        absorption_type=('only mass', [1]))
    br = core.boosts[85:110:6]
    L = np.concatenate([[0.0], np.geomspace(1e-2, 300, 200)])
    rate = np.asarray(core.light_secondaries_production(
        L, alpha=alpha, mass_range=mr, boost_range=br, true_range=tr))
    C_em = core.light_production_cumulative(
        L, alpha, channel='emission', mass_range=mr, boost_range=br, true_range=tr)
    cum = cumulative_trapezoid(rate, L, axis=-1, initial=0)
    err = np.abs(cum - C_em).max() / max(C_em.max(), 1e-12)
    print(f'  integral(rate) vs exact emission tally: {err:.2e} (trapezoid accuracy)')
    assert err < 2e-2


def test_neutrino_production_counts_and_energies():
    """pi -> mu -> e chain: 3 nu per charged pion, flavor ratio 2:1, <E> ~ E_pi/4."""
    core = cached('new_psb_nubase', lambda: InteractionCore(
        xsec_model=psb_xsec(), decays=decay_table()))
    n_b = len(core.boosts)
    N_pi = np.zeros((n_b, 3))
    i0 = 120                        # a high pion bin, away from the grid edge
    N_pi[i0] = [0., 5., 10.]        # cumulative pions over three distances
    E_nu, N_nu = core.neutrino_production(N_pion=N_pi, charged_fraction=1/3)

    n_mu = N_nu['nu_mu'].sum(axis=0)
    n_e = N_nu['nu_e'].sum(axis=0)
    assert np.allclose(n_mu, 2 * (1/3) * N_pi[i0], rtol=1e-3), 'nu_mu count != 2 per charged pion'
    assert np.allclose(n_e, 1 * (1/3) * N_pi[i0], rtol=1e-12), 'nu_e count != 1 per charged pion'
    print(f'  flavor ratio nu_mu : nu_e = {n_mu[-1] / n_e[-1]:.3f} (expect 2)')

    tot = N_nu['nu_mu'][:, -1] + N_nu['nu_e'][:, -1]
    E_mean = (E_nu * tot).sum() / tot.sum()
    E_pi = 0.13957039 * core.boosts[i0]
    print(f'  <E_nu> / E_pi = {E_mean / E_pi:.3f} (textbook ~0.25)')
    assert 0.2 < E_mean / E_pi < 0.3


def test_neutrino_production_via_pion_production():
    """The pion-path entry point runs end to end on a kernels-enabled core."""
    _, core = source_pair()
    alpha, mr, tr, _ = core.get_distribution_parameters(
        mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        absorption_type=('only mass', [1]))
    br = core.boosts[100:106]
    L = np.linspace(0.0, 50.0, 4)
    E_nu, N_nu = core.neutrino_production(L, alpha=alpha, mass_range=mr,
                                          boost_range=br, true_range=tr)
    N_pi = core.pion_production(L, alpha=alpha, mass_range=mr,
                                boost_range=br, true_range=tr)
    total_nu = N_nu['nu_mu'][:, -1].sum() + N_nu['nu_e'][:, -1].sum()
    print(f'  total nu per injected Fe at L=50 Mpc: {total_nu:.3e} '
          f'(= {total_nu / max(N_pi[:, -1].sum(), 1e-300):.3f} per lumped pion; expect ~1)')
    assert np.isfinite(total_nu) and total_nu > 0
def test_fft_rates_match_brute_force_reference():
    """compute_rates (the default rate_method='fft') against a dense direct
    quadrature of the exact isotropic-field integral, on identical sigma
    support. Also pins the analytic 1/y^2 continuation above the support,
    without which rates at 2*Gamma*eps_peak > sigma range are underestimated
    (the 'direct' method lacks it and falls behind there)."""
    from scipy.integrate import cumulative_trapezoid
    from scipy.constants import parsec
    from crisp.interaction_rates import compute_rates

    Mpc_cm = parsec * 1e8
    psb = psb_xsec()
    eps = 1e-3 * np.logspace(-1, 2.1, 300)
    eps_dense = 1e-3 * np.logspace(-1, 2.1, 12000)
    sig_dense = 1e-27 * psb.cross_section(eps_dense * 1e3, 26, 56)
    I_inner = cumulative_trapezoid(eps_dense * sig_dense, eps_dense, initial=0)
    eth = eps_dense[np.argmax(sig_dense > 0)]

    def ref_rate(gamma):
        lo = max(1e-18, eth / (2 * gamma) / 30)
        elab = np.logspace(np.log10(lo), -6, 20000)
        I = np.interp(2 * gamma * elab, eps_dense, I_inner, left=0, right=I_inner[-1])
        return np.trapezoid(cmb_photon_density_GeVcm3(elab) / elab**2 * I,
                            elab) / (2 * gamma**2) * Mpc_cm

    gammas = np.logspace(10.3, 14, 8)
    r_ref = np.array([ref_rate(g) for g in gammas])

    eMeV = eps * 1e3
    ew = 2 / eMeV**2 * cumulative_trapezoid(psb.cross_section(eMeV, 26, 56) * eMeV,
                                            eMeV, initial=0)
    pdens_eV = lambda e: cmb_photon_density_GeVcm3(np.asarray(e) * 1e-9) * 1e-9
    bounds, n_pts = (-9, 15), int(24 * 167) + 1
    r_fft = compute_rates(pdens_eV, np.logspace(*bounds, n_pts), ew, eMeV,
                          boostgrid=gammas, common_bounds=bounds, N=n_pts)[0]
    rel = np.abs(r_fft / r_ref - 1)
    print(f'  fft vs brute force (Gamma 2e10..1e14): max rel {rel.max():.2e}')
    assert rel.max() < 1e-2


def test_rate_method_fft_vs_direct():
    """The default 'fft' construction agrees with 'direct' below the boost
    where 2*Gamma*eps_peak reaches the sigma support, conserves nucleons, and
    is faster; above it, 'fft' keeps the (correct) 1/y^2 tail."""
    import time
    m = psb_xsec()
    t0 = time.time(); fft = InteractionCore(xsec_model=m); t_fft = time.time() - t0
    t0 = time.time(); direct = InteractionCore(xsec_model=m, rate_method='direct'); t_dir = time.time() - t0

    a, _ = static_imbalance(fft)
    assert a < CONSERVE

    pre = fft.boosts < 3e10
    idx = [fft.nuclei.index(n) for n in direct.nuclei]
    fa, da = fft.all_rates[idx][:, pre], direct.all_rates[:, pre]
    mask = da > 1e-6 * da.max()
    rel = np.abs(fa[mask] - da[mask]) / da[mask]
    print(f'  fft {t_fft:.1f}s vs direct {t_dir:.1f}s; pre-tail max rel {rel.max():.2e}; '
          f'high-boost tail ratio {fft.all_rates[:, fft.boosts > 3e11].sum() / direct.all_rates[:, fft.boosts > 3e11].sum():.2f}')
    assert rel.max() < 2e-2


def test_pion_kernel_absolute_normalization():
    """The kernel row sums equal the exact isotropic-field photomeson rate
    (each parent row is normalized to it; the head-on mapping only shapes the
    secondary spectrum). GZK benchmark: at Gamma = 1e11 on the CMB the exact
    rate is ~0.027 /Mpc — the threshold regime of the GZK rise, below the
    head-on peak-approximation value (~0.07) quoted before the normalization.
    Also pins the cm -> Mpc conversion (a factor-3e38 units bug was fixed here)."""
    from crisp.core import build_pion_prod_kernel, build_proton_recoil_kernel
    from crisp.interaction_rates import exact_rates_for_sigma
    from crisp.photonuclear_cross_sections import pgamma_components

    boosts = np.logspace(9, 12, 16)
    e_grid = np.logspace(np.log10(0.145), 4.0, 500)
    comps = pgamma_components(e_grid)
    sig_tot = 1e27 * sum(np.clip(c, 0, None) for c in comps.values())
    r_exact = exact_rates_for_sigma(boosts, cmb_photon_density_GeVcm3,
                                    e_grid, sig_tot)[0]

    # one nucleon leaves every interaction: the nucleon kernel rows equal the
    # exact isotropic interaction rate identically
    K_N = build_proton_recoil_kernel(boosts, cmb_photon_density_GeVcm3)
    row = K_N[0].sum(axis=1)
    mask = row > 0        # inside the head-on kinematic window
    d = np.abs(row[mask] - r_exact[mask]).max() / r_exact.max()
    print(f'  nucleon kernel row-sum == exact isotropic rate: {d:.2e}')
    assert d < 1e-10

    # the pion rows carry the physical multiplicity (>= 1 interaction count)
    K_pi = build_pion_prod_kernel(boosts, cmb_photon_density_GeVcm3)
    rate = K_pi[0].sum(axis=1)[int(np.argmin(np.abs(boosts - 1e11)))]
    print(f'  pion rate at Gamma=1e11 on the CMB: {rate:.3e} /Mpc (lit. rise 0.02-0.1)')
    assert 0.02 < rate < 0.15
    mult = np.divide(K_pi[0].sum(axis=1)[mask], row[mask]).max()
    print(f'  max pion multiplicity per interaction on this grid: {mult:.2f}')
    assert 1.0 <= mult < 3.01


def test_species_kernels_and_helicity_chain():
    """The charge/species-resolved photomeson kernels and the helicity decay
    chain (Huemmer+10): isospin mirror symmetry, the one-nucleon-per-
    interaction identity, physical charged fractions, exact nu counts and
    the nu/nubar detail, and the closed-form cooling limits."""
    from crisp.core import build_photomeson_species_kernels
    from crisp.interaction_rates import exact_rates_for_sigma
    from crisp.photonuclear_cross_sections import pgamma_components

    boosts = np.logspace(9, 12, 16)
    Ks = build_photomeson_species_kernels(boosts, cmb_photon_density_GeVcm3)

    # isospin mirror: neutron parents are the pi+ <-> pi-, p <-> n reflection
    assert np.array_equal(Ks['pi+'][0], Ks['pi-'][1])
    assert np.array_equal(Ks['pi0'][0], Ks['pi0'][1])
    assert np.array_equal(Ks['p'][0], Ks['n'][1])

    e = np.logspace(np.log10(0.145), 4, 500)
    comps = pgamma_components(e)
    r_ex = exact_rates_for_sigma(boosts, cmb_photon_density_GeVcm3, e,
                                 1e27 * sum(np.clip(c, 0, None) for c in comps.values()))[0]
    nuc = (Ks['p'] + Ks['n'])[0].sum(axis=1)
    m = nuc > 0
    assert np.abs(nuc[m] - r_ex[m]).max() / r_ex.max() < 1e-10

    lum = (Ks['pi+'] + Ks['pi-'] + Ks['pi0'])[0].sum(axis=1)
    f_ch = np.divide((Ks['pi+'] + Ks['pi-'])[0].sum(axis=1)[m], lum[m])
    print(f'  charged pion fraction range on the CMB grid: '
          f'{f_ch.min():.3f} .. {f_ch.max():.3f}')
    assert np.all(f_ch >= 1 / 3 - 1e-12) and np.all(f_ch <= 1.0)

    # helicity chain: 3 nu per charged pion, nu/nubar splits, cooling limits
    core = source_pair()[1]
    n_b = len(core.boosts)
    N_pi = np.zeros((n_b, 1))
    N_pi[120] = 6.0
    E_nu, N_nu = core.neutrino_production(N_pion=N_pi, charged_fraction=1.0)
    d = N_nu['detail']
    tot = sum(v.sum() for v in d.values())
    print(f'  nu per charged pion: {tot / 6.0:.4f} (expect 3, minus grid-floor tail)')
    assert abs(tot / 18.0 - 1) < 1e-2
    # symmetric pi+/pi- input -> symmetric nu/nubar per flavor
    assert np.allclose(d['nu_mu'], d['nubar_mu'], rtol=1e-12)
    assert np.allclose(d['nu_e'], d['nubar_e'], rtol=1e-12)

    _, N_cold = core.neutrino_production(N_pion=N_pi, charged_fraction=1.0, B_gauss=1e-12)
    assert rel_diff(N_cold['nu_mu'], N_nu['nu_mu']) < 1e-10   # B -> 0: no cooling
    _, N_hot = core.neutrino_production(N_pion=N_pi, charged_fraction=1.0, B_gauss=1e6)
    assert N_hot['nu_mu'].sum() < 1e-3 * N_nu['nu_mu'].sum()  # strong suppression
    f = core.decay_before_cooling(E_nu, 0.1057, 2.197e-6, 5e5)
    assert np.all(np.diff(f) <= 1e-15)                        # monotone in energy


def test_pgamma_components_decomposition():


def test_pion_kernel_nubase_mass_delta():
    """Documented delta: the pion kernel with the real proton mass vs 0.939."""
    from crisp.core import build_pion_prod_kernel
    boosts = np.logspace(8, 13, 26)
    k_leg = build_pion_prod_kernel(boosts, cmb_photon_density_GeVcm3)
    k_nub = build_pion_prod_kernel(boosts, cmb_photon_density_GeVcm3,
                                   mp_GeV=nuclear_mass_GeV(1, 1))
    delta = rel_diff(k_nub, k_leg)
    print(f'  pion kernel nubase-vs-legacy proton mass: {delta:.2e} '
          f'(boost-bin remapping from a 0.08% mass shift)')
    assert delta < 0.5, 'mass shift should only move rates between adjacent bins'
    # total pion production per parent boost is mass-independent (only the
    # pion-boost binning moves)
    d_tot = rel_diff(k_nub.sum(axis=2), k_leg.sum(axis=2))
    print(f'  per-parent totals: {d_tot:.2e}')
    assert d_tot < MACHINE


def test_psb_model_c12_channel_completion():
    """Model-level documented change: the C-12 total at 100 MeV doubled when the
    A=10-22 channels were completed (the old table dropped remnant masses 8-5,
    half the PSB1976 multiplicity weight; at 100 MeV the total is pure
    quasi-deuteron plateau, proportional to the summed weights)."""
    xs = psb_xsec().cross_section(np.array([100.0]), 6, 12)
    print(f'  PSB C-12 total at 100 MeV: {xs[0]:.4f} mb (old stale baseline: 0.79235)')
    assert np.isclose(xs[0], 2 * 0.79235, rtol=1e-3)


# ------------------------------------------------------------------- known gaps

def test_gdra_placeholder():
    """The GDRA core class was removed in phase 5 (pre-existing species explosion).

    The GDR_atlas cross-section model remains available; using it through
    InteractionCore(xsec_model=GDR_atlas(...)) needs a species filter
    (e.g. restricting to A <= 56 ground states) and a channel-table pass first.
    """
    raise unittest.SkipTest('GDRA core removed; GDR_atlas model needs a species filter to be usable')


def test_crpropa_removed():
    """The CRPropa file-driven cores were testing scaffolding (user decision,
    2026-07-09) and were deleted in phase 5, not replicated."""
    import crisp.core as cc
    for name in ['InteractionCore_CRPropA', 'InteractionCore_CRPropA_pdis',
                 'InteractionCore_UHECR_Source', 'InteractionCore_GDRA_CMB']:
        assert not hasattr(cc, name), f'{name} should have been removed'
    print('  scaffolding classes absent, as intended')


# ---------------------------------------------------------------- plain runner

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        print(f'== {name} ==')
        try:
            fn()
            print('   PASS')
        except unittest.SkipTest as exc:
            print(f'   SKIP ({exc})')
        except AssertionError as exc:
            print(f'   FAIL: {exc}')
            failed.append(name)
        print()
    print(f'{len(tests) - len(failed)}/{len(tests)} passed' + (f'; FAILED: {failed}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
