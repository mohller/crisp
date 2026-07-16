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
    imbA = (np.einsum('j,ijb->ib', A_sp, core.tensor)
            + np.einsum('l,lijb->ib', A_LIGHT, core.light_prod_tensor))
    imbZ = (np.einsum('j,ijb->ib', Z_sp, core.tensor)
            + np.einsum('l,lijb->ib', Z_LIGHT, core.light_prod_tensor))
    ej = getattr(core, 'photomeson_ejecta', None)
    if ej is not None:
        for ni, nuc in enumerate(core.nuclei):
            si = core.species.index(tuple(nuc))
            imbA[si] += ej['p'][ni] + ej['n'][ni]
            imbZ[si] += ej['p'][ni]
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

    # weights= is the weighted ladder total: identical to summing per-slice folds
    w = 1.0 / br
    ref = sum(w[i] * new.pion_production(L, alpha=alpha, mass_range=mr,
                                         boost_range=br[i:i + 1], true_range=tr,
                                         P=Pev[i:i + 1])
              for i in range(len(br)))
    d_w = rel_diff(new.pion_production(L, P=Pev, weights=w, **kw), ref)
    print(f'  weights= vs per-slice sum: {d_w:.2e}')
    assert d_w < 1e-12

    # photomeson_ejecta_production == the manual fold of the ejecta budget
    ej = new.photomeson_ejecta
    rows = np.zeros((len(tr), len(new.boosts)))
    for k, t in enumerate(tr):
        za = tuple(new.species[t])
        if za in new.nuclei:
            ni = new.nuclei.index(za)
            rows[k] = ej['p'][ni] + ej['n'][ni]
    from scipy.interpolate import interp1d as i1d
    ej_b = i1d(new.boosts, rows, kind='previous', bounds_error=False,
               fill_value=0.0)(br)
    ref_w = np.einsum('bmi,ib->bm', np.asarray(Pev)[:, :, :len(tr)], ej_b)
    got = new.photomeson_ejecta_production(L, P=Pev, **kw).sum(axis=0)
    d_e = rel_diff(got, ref_w)
    print(f'  photomeson_ejecta_production vs manual fold: {d_e:.2e}')
    assert d_e < MACHINE


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
    # 1% tolerance: the full decay spectra put a small tail below the energy
    # grid floor, which is dropped by convention
    assert np.allclose(n_mu, 2 * (1/3) * N_pi[i0], rtol=1e-2), 'nu_mu count != 2 per charged pion'
    assert np.allclose(n_e, 1 * (1/3) * N_pi[i0], rtol=1e-2), 'nu_e count != 1 per charged pion'
    print(f'  flavor ratio nu_mu : nu_e = {n_mu[-1] / n_e[-1]:.3f} (expect 2)')

    tot = N_nu['nu_mu'][:, -1] + N_nu['nu_e'][:, -1]
    E_mean = (E_nu * tot).sum() / tot.sum()
    E_pi = 0.13957039 * core.boosts[i0]
    print(f'  <E_nu> / E_pi = {E_mean / E_pi:.3f} (textbook ~0.25)')
    assert 0.2 < E_mean / E_pi < 0.3


def test_neutrino_production_via_pion_production():
    """The pion-path entry point runs end to end on a kernels-enabled core.

    The parent boosts must sit in the physical photopion regime of the CMB
    (Gamma >~ 1e10): below it the kernel rows are Planck-tail suppressed to
    the underflow level and the test would assert on numerical dust."""
    _, core = source_pair()
    alpha, mr, tr, _ = core.get_distribution_parameters(
        mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        absorption_type=('only mass', [1]))
    br = core.boosts[110:116]
    L = np.linspace(0.0, 50.0, 4)
    E_nu, N_nu = core.neutrino_production(L, alpha=alpha, mass_range=mr,
                                          boost_range=br, true_range=tr)
    N_pi = core.pion_production(L, alpha=alpha, mass_range=mr,
                                boost_range=br, true_range=tr)
    total_nu = N_nu['nu_mu'][:, -1].sum() + N_nu['nu_e'][:, -1].sum()
    ratio = total_nu / N_pi[:, -1].sum()
    print(f'  total nu per injected Fe at L=50 Mpc: {total_nu:.3e} '
          f'(= {ratio:.3f} per lumped pion; 3 x charged fraction)')
    assert np.isfinite(total_nu) and total_nu > 0
    # 3 x physical charged fraction of the interaction-type mix: between the
    # pure-Delta 1.0 and the all-charged 3.0
    assert 1.0 < ratio < 2.5


def test_model_rack_channel_union():
    """Model_Rack keeps every model's channels (union per nucleus, dedup) and
    distinguishes the interaction groups."""
    from crisp.photonuclear_cross_sections import Model_Rack, Photomeson_Superposition
    pdis = psb_xsec()
    pm = Photomeson_Superposition(pdis.nuclei)
    rack = Model_Rack(models=(pdis, pm))

    assert rack.photodisintegration_models == [pdis]
    assert rack.photomeson_models == [pm]
    assert sorted(rack.nuclei) == sorted(set(pdis.nuclei))   # pm nuclei are a subset

    for nuc in [(26, 56), (6, 12), (2, 4)]:
        i, j, k = rack.nuclei.index(nuc), pdis.nuclei.index(nuc), pm.nuclei.index(nuc)
        union = set(rack.channels[i])
        assert set(map(tuple, pdis.channels[j])) <= union, f'{nuc}: pdis channel lost'
        assert set(map(tuple, pm.channels[k])) <= union, f'{nuc}: photomeson channel lost'
    print('  rack union keeps all channels of both groups')


def test_photomeson_rates_in_cascade():
    """photomeson='kernels' adds the superposition A -> A-1 rates to the
    cascade: the per-nucleus rate increment equals the rate of the
    superposition sigma row exactly; conservation holds; the p/n hook is set
    and shifts the light-matrix diagonals; a rack that already contains a
    photomeson model is not wrapped again."""
    from crisp.photonuclear_cross_sections import Model_Rack, Photomeson_Superposition

    kern = cached('psb_photomeson', lambda: InteractionCore(
        xsec_model=PSB_model(), photomeson='kernels'))
    none = InteractionCore(xsec_model=PSB_model(), eps=kern.eps)

    i, j = kern.nuclei.index((26, 56)), none.nuclei.index((26, 56))
    inc = kern.all_rates[i] - none.all_rates[j]
    pm = kern._collect_photomeson_models(kern.xsec_model)
    assert len(pm) == 1 and type(pm[0]).__name__ == 'Photomeson_Superposition'
    r_pm = kern._rates_of_sigma_rows(pm[0].cross_section(kern.eps * 1e3, 26, 56))[0]
    mask = r_pm > 1e-8 * r_pm.max()
    d = np.abs(inc[mask] - r_pm[mask]).max() / r_pm.max()
    b = int(np.argmin(np.abs(kern.boosts - 1e11)))
    print(f'  Fe56 photomeson rate at Gamma=1e11: {inc[b]:.3f} /Mpc '
          f'({inc[b]/56:.4f} per nucleon); increment==row identity {d:.2e}')
    assert d < MACHINE

    a, z = static_imbalance(kern)
    assert a < CONSERVE and z < CONSERVE

    assert hasattr(kern, 'photomeson_rates_pn') and kern.photomeson_rates_pn[b] > 0
    M = kern._build_light_interaction_matrix(kern.boosts[[b]])
    M0 = none._build_light_interaction_matrix(none.boosts[[b]])
    shift = (M0[4, 4, 0] - M[4, 4, 0])
    print(f'  p/n light-matrix absorption: {shift:.4f} /Mpc == hook {kern.photomeson_rates_pn[b]:.4f}')
    assert np.isclose(shift, kern.photomeson_rates_pn[b], rtol=1e-12)

    # pre-composed rack: no double counting
    pdis = PSB_model()
    rack = Model_Rack(models=(pdis, Photomeson_Superposition(pdis.nuclei)))
    manual = InteractionCore(xsec_model=rack, photomeson='kernels', eps=kern.eps,
                             boosts=kern.boosts)
    k = manual.nuclei.index((26, 56))
    assert np.abs(manual.all_rates[k] - kern.all_rates[i]).max() < 1e-12 * kern.all_rates[i].max()
    print('  pre-composed rack identical to auto-wrap (no double counting)')


def test_photomeson_model_astrophomes():
    """A richer photomeson model (AstroPhoMes) joins the cascade through the
    rack: its channels survive the union and contribute nonzero rates.

    The repository is resolved via crisp.data_download.get_astrophomes_path
    (ASTROPHOMES_PATH env var / local cache); no download is attempted here."""
    from crisp.data_download import get_astrophomes_path
    from crisp.photonuclear_cross_sections import Model_Rack, Photomeson, load_astrophomes

    try:
        get_astrophomes_path(auto_download=False, verbose=False)
    except FileNotFoundError as exc:
        raise unittest.SkipTest(str(exc))

    pdis = psb_xsec()
    xspm = Photomeson(pmm=load_astrophomes(auto_download=False),
                      filter_nuclei=lambda nuc: nuc in pdis.nuclei)
    rack = Model_Rack(models=(pdis, xspm))

    i = rack.nuclei.index((26, 56))
    pm_only = set(map(tuple, xspm.channels[xspm.nuclei.index((26, 56))]))
    assert pm_only <= set(rack.channels[i]), 'AstroPhoMes channels lost in the rack'

    # channels list every inclusive daughter (untracked ones are resolved by
    # the core), and cross_section answers only for listed remnants
    e_pm = np.logspace(2.2, 6, 40)  # MeV
    for nuc in [(26, 56), (4, 9), (1, 2)]:
        chans = xspm.channels[xspm.nuclei.index(nuc)]
        assert chans, f'{nuc}: no photomeson channels listed'
        per_ch = np.array([xspm.cross_section(e_pm, *nuc, rem=ch).max() for ch in chans])
        assert per_ch.max() > 0, f'{nuc}: listed channels answer zero sigma'
        assert xspm.cross_section(e_pm, *nuc, rem=(9, 39)).max() == 0, \
            f'{nuc}: sigma leaks into unlisted channels'
    assert xspm.channels[xspm.nuclei.index((1, 2))] == [(1, 1)]  # nonel-only deuteron

    core = InteractionCore(xsec_model=rack, photomeson='kernels',
                           eps=np.logspace(-4, 4, 650))
    assert core._collect_photomeson_models(core.xsec_model) == [xspm]  # no auto-wrap
    a, z = static_imbalance(core)
    print(f'  AstroPhoMes rack: conservation A {a:.1e}, Z {z:.1e}')
    assert a < CONSERVE
    none = InteractionCore(xsec_model=psb_xsec(), eps=core.eps)
    k, j = core.nuclei.index((26, 56)), none.nuclei.index((26, 56))
    b = int(np.argmin(np.abs(core.boosts - 1e12)))
    print(f'  Fe56 AstroPhoMes photomeson rate at 1e12: {core.all_rates[k][b] - none.all_rates[j][b]:.3f} /Mpc')
    assert core.all_rates[k][b] > none.all_rates[j][b]


def test_photomeson_inclusive_scaling():
    """The empirical photomeson mass scalings (AstroPhoMes EmpiricalModel,
    Morejon+19) through the hybrid loader and photomeson_scaling='inclusive':
    the genuine model's restored multiplicity table, the hybrid's exclusive
    A-1 channels with improved cross sections, the sigma-level A^alpha_pi
    scaling, the fold factors, and default-path invariance."""
    from crisp.data_download import get_astrophomes_path
    from crisp.photonuclear_cross_sections import Model_Rack, Photomeson, load_astrophomes

    try:
        get_astrophomes_path(auto_download=False, verbose=False)
    except FileNotFoundError as exc:
        raise unittest.SkipTest(str(exc))

    # genuine fixed model: non-empty channel table (guards the upstream
    # isinstance(k, str) regression that silently emptied it on py3)
    emp = load_astrophomes(model='EmpiricalModel', auto_download=False)
    assert len(emp.multiplicity) > 0 and (5626, 100) in emp.multiplicity
    fe_mult = sum(v for (m, d), v in emp.multiplicity.items() if m == 5626)
    print(f'  genuine EmpiricalModel: {len(emp.multiplicity)} entries, '
          f'Fe-56 total multiplicity {fe_mult:.2f} (inclusive table)')

    # hybrid: SPM exclusive channel structure + EmpiricalModel cross sections
    hyb = load_astrophomes(model='EmpiricalModel', channels='superposition',
                           auto_download=False)
    spm = load_astrophomes(auto_download=False)
    assert sorted(k for k in hyb.multiplicity if k[0] == 5626) == \
           sorted(k for k in spm.multiplicity if k[0] == 5626)
    assert (hyb.cs_nonel(5626)[1] >= 0).all()          # threshold clip

    pdis = psb_xsec()
    xspm = Photomeson(pmm=hyb, filter_nuclei=lambda nuc: nuc in pdis.nuclei)
    eps = np.logspace(2.2, 6, 80)                      # MeV
    assert xspm.cross_section(eps, 26, 56, rem=(26, 55)).max() > 0

    # sigma level: A^(alpha_pi - 1) suppression at low energies, charge-blind
    r = {}
    for c in (2, 3, 4):
        num = xspm.inclusive_cross_section(eps, 26, 56, c)
        den = (26 * xspm.inclusive_cross_section(eps, 1, 1, c)
               + 30 * xspm.inclusive_cross_section(eps, 0, 1, c))
        m = den > 0
        r[c] = num[m] / den[m]
    lowE = r[2][:6].mean()
    print(f'  sigma-level Fe pion factor at threshold: {lowE:.3f} '
          f'(56^(alpha_pi-1) ~ 0.26 with the fade)')
    assert 0.20 < lowE < 0.40
    # charge-blind at sigma level (1e-15 on the native pmm grid; the residual
    # here is np.interp resampling of differently-shaped sigma onto eps)
    assert np.abs(r[2] - r[4]).max() < 0.01
    spm_w = Photomeson(pmm=spm, filter_nuclei=lambda nuc: nuc in pdis.nuclei)
    num = spm_w.inclusive_cross_section(eps, 26, 56, 2)
    den = (26 * spm_w.inclusive_cross_section(eps, 1, 1, 2)
           + 30 * spm_w.inclusive_cross_section(eps, 0, 1, 2))
    m = den > 0
    assert np.allclose(num[m] / den[m], 1.0, atol=1e-9)

    # core factors + fold identity on the GRB-like Band field
    from crisp.background_photon_models import band_photon_spectrum
    tp = band_photon_spectrum(2.5e-7, normal=((1e-9, 1e-2), 7.6e5))
    rack = Model_Rack(models=(pdis, xspm))
    kw = dict(xsec_model=rack, target_photons=tp, photomeson='kernels',
              boosts=np.logspace(0, 12, 131), eps=np.logspace(-2, 6, 300))
    core = InteractionCore(photomeson_scaling='inclusive', **kw)
    twin = InteractionCore(**kw)
    assert not hasattr(twin, 'photomeson_fold_scaling')

    F = core.photomeson_fold_scaling
    i_fe = core._pm_scaling_index[(26, 56)]
    bgrid = core.boosts
    f_lo = np.interp(3e4, bgrid, F['pion'][i_fe])      # threshold-regime boosts
    f_hi = np.interp(1e10, bgrid, F['pion'][i_fe])
    print(f'  Fe fold factor: {f_lo:.3f} at 3e4, {f_hi:.3f} at 1e10 '
          f'(alpha_pi -> 1 at high energies)')
    assert 0.2 < f_lo < 0.6 and 0.9 < f_hi <= 1.001
    sig = F['pion'][i_fe] != 1.0
    assert np.abs(F['pi+'][i_fe] - F['pi-'][i_fe])[sig].max() < 0.05
    assert 0.5 < np.interp(3e8, bgrid, F['N'][i_fe]) < 1.2

    i_sp = core.species.index((26, 56))
    fold_kw = dict(alpha=np.ones(1), mass_range=[i_sp], boost_range=np.array([3e8]),
                   true_range=[i_sp], P=np.ones((1, 2, 1)))
    L = np.array([0.0, 1e-9])
    y1 = core.photomeson_production('pi+', L, **fold_kw)[:, -1]
    y0 = twin.photomeson_production('pi+', L, **fold_kw)[:, -1]
    m = y0 > 0
    f_exp = np.interp(3e8, bgrid, F['pi+'][i_fe])
    d = np.abs(y1[m] / y0[m] - f_exp).max()
    print(f'  fold ratio == stored factor at boost 3e8: {f_exp:.4f} (delta {d:.1e})')
    assert d < 1e-12


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
    # near threshold both kernels sit at their FFT/interpolation noise floor
    # (~14 orders of magnitude below the peak); a ratio of two noise-floor
    # values is meaningless and environment-dependent, so restrict the
    # multiplicity check to rows with a physically significant rate (same
    # 1e-3-of-max significance cut used in test_sophia_spectrum_kernels)
    sig = row > 1e-3 * row.max()
    mult = np.divide(K_pi[0].sum(axis=1)[sig], row[sig]).max()
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
    """pgamma_components (the Rachen parametrization split into resonances /
    direct / multi-pion) sums to pgamma exactly, with the pieces peaking
    where the physics says."""
    from crisp.photonuclear_cross_sections import pgamma, pgamma_components
    e = np.logspace(-1, 4, 1500)
    c = pgamma_components(e)
    assert np.array_equal(c['resonances'] + c['direct'] + c['multipion'], pgamma(e))
    e_res = e[c['resonances'].argmax()]
    print(f'  resonance peak at eps_r = {e_res:.2f} GeV; multipion/resonances '
          f'at 100 GeV = {c["multipion"][e > 100][0] / c["resonances"][e > 100][0]:.1f}')
    assert 0.3 < e_res < 0.4                            # Delta(1232) region
    assert c['multipion'][-1] > c['resonances'][-1]     # continuum wins at high E


def test_photomeson_only_core():
    """A photomeson-only core (no photodisintegration) constructs, conserves,
    and the rack decomposes additively: full == pdis-only + photomeson-only."""
    from crisp.photonuclear_cross_sections import Model_Rack, Photomeson_Superposition

    pdis_model = psb_xsec()
    pm_model = Photomeson_Superposition(pdis_model.nuclei)

    # default eps tops out below the 145 MeV photomeson threshold -> wide grid
    pm_only = InteractionCore(xsec_model=pm_model, eps=np.logspace(-4, 4, 650))
    a, z = static_imbalance(pm_only)
    print(f'  photomeson-only core: conservation A {a:.1e}, Z {z:.1e}')
    assert a < CONSERVE and z < CONSERVE

    # photomeson nucleons are budgeted as wide-spectrum ejecta, one per event:
    # the budget equals the total rate, and the narrow light yields carry none
    ej = pm_only.photomeson_ejecta
    d_ej = rel_diff(ej['p'] + ej['n'], pm_only.all_rates)
    print(f'  ejecta identity (p+n budget == all_rates, 1 nucleon/event): {d_ej:.2e}')
    assert d_ej < 1e-12
    ly = np.abs(pm_only.light_prod_tensor[4:]).max()   # p, n narrow yields
    resolved = np.abs(pm_only.light_prod_tensor).max() # decay-resolution lights remain
    print(f'  narrow p/n yields from photomeson channels: {ly:.2e} '
          f'(resolution lights {resolved:.2e})')

    pdis_only = InteractionCore(xsec_model=pdis_model, eps=pm_only.eps)
    full = InteractionCore(xsec_model=Model_Rack(models=(pdis_model, pm_model)),
                           eps=pm_only.eps)
    i_f = full.nuclei.index((26, 56))
    i_p = pdis_only.nuclei.index((26, 56))
    i_m = pm_only.nuclei.index((26, 56))
    add = np.abs(full.all_rates[i_f] - pdis_only.all_rates[i_p]
                 - pm_only.all_rates[i_m]).max() / full.all_rates[i_f].max()
    print(f'  rack additivity (full == pdis + photomeson): {add:.2e}')
    assert add < 1e-10

    # the decomposition utility recovers the same split from the full core
    dec = full.rates_by_interaction()
    d_sum = np.abs(sum(dec.values()) - full.all_rates).max() / full.all_rates.max()
    i_dm = full.nuclei.index((26, 56))
    d_pm = rel_diff(dec['photomeson'][i_dm], pm_only.all_rates[i_m])
    d_pd = rel_diff(dec['photodisintegration'][i_dm], pdis_only.all_rates[i_p])
    print(f'  rates_by_interaction: sum == all_rates {d_sum:.2e}, '
          f'pm == pm-only core {d_pm:.2e}, pdis == pdis-only core {d_pd:.2e}')
    assert d_sum < 1e-8 and d_pm < 1e-10 and d_pd < 1e-10
    single = full.rates_by_interaction(nucleus=(26, 56))
    assert np.array_equal(single['photomeson'], dec['photomeson'][i_dm])


def test_source_model_physics_methods():
    """The source-model physics lives in the library, not in notebooks:
    acceleration/loss balance -> E'_max, the Appendix-D injection spectrum
    (energy closure to the baryon budget), frame conversion, the core's
    conservation diagnostic, and cumulative= on the production folds."""
    from scipy.constants import c
    from scipy.integrate import cumulative_trapezoid
    from crisp.source_models import OneZoneISModel

    ph = OneZoneISModel(photon_energy_min=1e-7, photon_energy_max=3e-4,
                           photon_energy_brk=1e-6, variability_timescale=.01,
                           bulk_lorentz_factor=300, photon_luminosity=1e53,
                           baryonic_loading=10, photon_energy=100, redshift=2)
    core = InteractionCore(xsec_model=psb_xsec(), target_photons=ph.target_photons,
                           photomeson='kernels', boosts=np.logspace(0, 12, 131),
                           eps=np.logspace(-4, 4, 650))

    a, z = core.conservation_imbalance()
    a2, z2 = static_imbalance(core)
    print(f'  conservation_imbalance: A {a:.2e}, Z {z:.2e} (== suite helper)')
    assert np.isclose(a, a2, rtol=1e-9) and np.isclose(z, z2, rtol=1e-9)
    assert a < CONSERVE

    Emax = ph.max_energy((26, 56), core)
    rates = ph.loss_rates((26, 56), core)
    acc_at = ph.acceleration_rate((26, 56), Emax)
    tot_at = np.exp(np.interp(np.log(Emax), np.log(rates['E']),
                              np.log(rates['total'])))
    print(f"  E'_max(Fe-56) = {Emax:.3e} GeV; acc/loss there = {acc_at / tot_at:.3f}")
    assert 1e7 < Emax < 1e9 and abs(acc_at / tot_at - 1) < 0.05

    q, info = ph.injection_spectrum((26, 56), interaction_core=core)
    g = np.logspace(1, 8, 4000)
    E = core.energy_of_boost((26, 56), g)
    t_cross = ph.get_parameter('shell_width').to('m').m / c
    u_inj = np.trapezoid(q(g) * E, np.log(g)) * t_cross
    u_target = 10 * ph.get_parameter('em_density').m
    print(f"  injection closure: {u_inj / u_target:.5f} x eta u_gamma  (C' = {info['C']:.3e})")
    assert abs(u_inj / u_target - 1) < 1e-3
    assert np.isclose(ph.observed_energy(1.0), 100.0)

    alpha, mr, tr, _ = core.get_distribution_parameters(
        mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        absorption_type=('only mass', [1]))
    L = np.logspace(-19, -16, 12)
    br2 = np.array([1e3, 1e4])
    P = core.species_evolution_boost_range(L, alpha, mr, br2, tr)
    kw = dict(alpha=alpha, mass_range=mr, boost_range=br2, true_range=tr, P=P)
    r = np.asarray(core.light_secondaries_production(L, **kw))
    d1 = rel_diff(core.light_secondaries_production(L, cumulative=True, **kw),
                  cumulative_trapezoid(r, L, axis=-1, initial=0))
    e = core.photomeson_ejecta_production(L, **kw)
    d2 = rel_diff(core.photomeson_ejecta_production(L, cumulative=True, **kw),
                  cumulative_trapezoid(e, L, axis=-1, initial=0))
    print(f'  cumulative= identities: light {d1:.2e}, ejecta {d2:.2e}')
    assert d1 < MACHINE and d2 < MACHINE


def test_frame_conversions():
    """The frame-aware parameter API: schema-declared kinds/native frames,
    get_parameter(frame=), the frames= input declaration, and the comoving
    adiabatic rate Gamma c / R (De Lia & Tamborra Eq. 3.3)."""
    from crisp.source_models import InternalShockModel, OneZoneISModel, ureg

    T = dict(redshift=2.0, bulk_lorentz_factor=300.0,
             iso_energy=4.5e54 * ureg.erg, duration=30.0 * ureg.second,
             eps_d=0.2, eps_e=0.01, eps_A=0.1, eps_B=0.1, k_index=2.2)
    m_eng = InternalShockModel(variability_timescale=0.5 / 3.0, **T)
    m_obs = InternalShockModel(variability_timescale=0.5,
                               frames={'variability_timescale': 'observer'},
                               **T)

    # frames= input declaration == explicit engine-frame value
    for par in ('radius', 'shell_width', 'magnetic_field', 'em_density'):
        a = m_eng.get_parameter(par).m
        b = m_obs.get_parameter(par).m
        assert abs(a / b - 1) < 1e-12, par

    # output conversions: t~ (engine, native) x(1+z) observed, xGamma comoving
    t_nat = m_obs.get_parameter('variability_timescale').m
    assert abs(t_nat - 0.5 / 3.0) < 1e-12
    assert abs(m_obs.get_parameter('variability_timescale',
                                   frame='observer').m - 0.5) < 1e-12
    assert abs(m_obs.get_parameter('variability_timescale',
                                   frame='comoving').m - 300 * 0.5 / 3) < 1e-9
    # energies: comoving native -> observer = x Gamma/(1+z)
    e_nat = m_obs.get_parameter('photon_energy_brk').m
    e_ob = m_obs.get_parameter('photon_energy_brk', frame='observer').m
    assert abs(e_ob / e_nat - 300 / 3.0) < 1e-9

    # error paths: unknown frame; no rule declared; frames= without value
    for bad in (lambda: m_obs.get_parameter('radius', frame='observer'),
                lambda: m_obs.get_parameter('variability_timescale',
                                            frame='lab')):
        try:
            bad()
            raise AssertionError('must raise')
        except ValueError:
            pass
    try:
        InternalShockModel(variability_timescale=0.5,
                           frames={'nonexistent_par': 'observer'}, **T)
        raise AssertionError('frames= for a missing parameter must raise')
    except ValueError:
        pass

    # adiabatic rate is the comoving Gamma c / R (both GRB model families)
    oz = OneZoneISModel(photon_energy_min=1e-7, photon_energy_max=3e-4,
                        photon_energy_brk=1e-6, variability_timescale=.01,
                        bulk_lorentz_factor=300, photon_luminosity=1e53,
                        baryonic_loading=10, photon_energy=100, redshift=2)
    core = InteractionCore(xsec_model=psb_xsec(), target_photons=oz.target_photons,
                           photomeson='kernels', boosts=np.logspace(0, 12, 61),
                           eps=np.logspace(-4, 4, 300))
    lr = oz.loss_rates((1, 1), core, include_ic=True)
    R_cm = oz.get_parameter('radius').to('cm').m
    expect = 300 * 2.99792458e10 / R_cm
    print(f"  adiabatic rate: {lr['adiabatic'][0]:.4e} /s == Gamma c/R "
          f"{expect:.4e}")
    assert abs(lr['adiabatic'][0] / expect - 1) < 1e-6

    # inverse Compton: at exact equipartition (B' = sqrt(8 pi u_gamma)) the
    # Thomson-regime IC rate equals synchrotron; deep Klein-Nishina kills it
    b = core.boosts
    r_ic, r_sy = lr['inverse_compton'], lr['synchrotron']
    i_lo = int(np.argmin(np.abs(b - 1e2)))       # b_KN = 4 g eps/m << 1
    i_hi = int(np.argmin(np.abs(b - 1e10)))
    print(f'  IC/synchrotron: {r_ic[i_lo] / r_sy[i_lo]:.3f} (Thomson), '
          f'{r_ic[i_hi] / r_sy[i_hi]:.2e} (deep KN)')
    assert 0.8 < r_ic[i_lo] / r_sy[i_lo] < 1.05
    assert r_ic[i_hi] / r_sy[i_hi] < 1e-3
    assert 'inverse_compton' not in oz.loss_rates((1, 1), core)  # default off
    d_tot = lr['total'] - oz.loss_rates((1, 1), core)['total'] - r_ic
    assert np.abs(d_tot).max() < 1e-12 * lr['total'].max()

    # kappa=True: photonuclear entries become energy-loss (cooling) rates
    lr_k = oz.loss_rates((1, 1), core, kappa=True)
    act = lr['photomeson'] > 1e-3 * lr['photomeson'].max()
    kap_p = lr_k['photomeson'][act] / lr['photomeson'][act]
    kap_p = kap_p[kap_p > 0]       # top-of-grid CIC clamp rows clip to 0
    print(f'  proton photomeson kappa: {kap_p.min():.2f}..{kap_p.max():.2f}')
    assert 0.1 < kap_p.min() and kap_p.max() < 0.7
    lrFe = oz.loss_rates((26, 56), core)
    lrFe_k = oz.loss_rates((26, 56), core, kappa=True)
    actF = lrFe['photonuclear'] > 1e-3 * lrFe['photonuclear'].max()
    kap_F = lrFe_k['photonuclear'][actF] / lrFe['photonuclear'][actF]
    print(f'  Fe-56 photonuclear kappa: {kap_F.min():.3f}..{kap_F.max():.3f}')
    assert 0.005 < kap_F.min() and kap_F.max() < 0.30
    assert oz.max_energy((26, 56), core, kappa=True) \
        >= oz.max_energy((26, 56), core)


def test_nucleon_reprocessing():
    """Multi-generation photomeson transport of free nucleons
    (nucleon_transport_matrix / reprocessed_nucleons / cascade_nucleon_source):
    exact number conservation by construction, the recoil-kernel block
    identities, the optically-thin one-generation limit, spectral degradation
    and the thick-field sub-threshold pile-up, the Duhamel source integral,
    and the cumulative=False fold pin."""
    from scipy.linalg import expm
    from scipy.integrate import cumulative_trapezoid
    from crisp.core import deposit_log_cic

    _, core = source_pair()                      # cached CMB kernels core
    n_b = len(core.boosts)
    i_p, i_n = core.species.index((1, 1)), core.species.index((0, 1))
    Kp, Kn = core.photomeson_kernels['p'], core.photomeson_kernels['n']
    R = (Kp[0] + Kn[0]).sum(axis=1)              # exact proton rate [1/Mpc]
    rmax = R.max()

    # 1. generator: rows sum to zero; blocks are exactly the fold's kernels
    # (the n -> p block additionally carries the beta-decay diagonal)
    from crisp.core import c_in_Mpc_sec, get_nucid
    M = core.nucleon_transport_matrix()
    assert np.abs(M.sum(axis=1)).max() < 1e-12 * np.abs(M).max()
    assert np.array_equal(M[:n_b, n_b:], Kn[0])
    G = M[n_b:, :n_b]
    assert np.array_equal(G - np.diag(np.diag(G)),
                          Kp[1] - np.diag(np.diag(Kp[1])))
    tau_n = core.decays[get_nucid((0, 1))]['decay_time']
    lam = 1.0 / (core.boosts * tau_n * c_in_Mpc_sec)
    assert np.allclose(np.diag(G) - np.diag(Kp[1]), lam, rtol=1e-12)
    offdiag = M[:n_b, :n_b] - np.diag(np.diag(M[:n_b, :n_b]))
    assert np.array_equal(offdiag, Kp[0] - np.diag(np.diag(Kp[0])))

    # neutron decay closed form: pure-n injection below the photopion
    # threshold follows exp(-L/L_dec), protons the complement, exactly
    ilow = int(np.argmin(np.abs(core.boosts - 1e6)))
    assert R[ilow] < 1e-3 * lam[ilow]
    L_dec = 1.0 / lam[ilow]
    n0n = np.zeros((n_b, 2))
    n0n[ilow, 1] = 1.0
    Ld = np.linspace(0.0, 2 * L_dec, 6)
    nd = core.reprocessed_nucleons(Ld, injection=n0n)
    assert np.allclose(nd[ilow, :, 1], np.exp(-Ld / L_dec), rtol=1e-9)
    assert np.allclose(nd[ilow, :, 0], 1 - np.exp(-Ld / L_dec), atol=1e-9)
    print(f'  neutron decay: tau_n = {tau_n:.1f} s (nubase), closed form to 1e-9')

    # cooling drift (energy_loss=): conservative upwind advection — the mean
    # ln(gamma) drifts at exactly -b while number is conserved (pure-drift
    # check with protons below the photomeson threshold)
    dln = np.log(core.boosts[1] / core.boosts[0])
    n0d = np.zeros((n_b, 2))
    n0d[ilow, 0] = 1.0
    b0 = 10 * dln                                # 10 bins over unit path
    Lb = np.linspace(0.0, 1.0, 5)
    ncool = core.reprocessed_nucleons(Lb, injection=n0d,
                                      energy_loss=np.full((n_b, 2), b0))
    tot = ncool.sum(axis=(0, 2))
    assert np.abs(tot - 1).max() < 1e-12
    lngb = np.log(core.boosts)
    mbar = (ncool.sum(axis=2) * lngb[:, None]).sum(axis=0) / tot
    assert np.allclose(mbar, lngb[ilow] - b0 * Lb, atol=1e-9)
    print(f'  cooling drift: <ln gamma> follows -b L exactly '
          f'({(mbar[0] - mbar[-1]) / dln:.1f} bins), number conserved')

    # 2. number conservation of the transport at several optical depths
    # (the solver caps the stiff decay pair per interval — exact to e^-60 —
    # so conservation stays at machine precision despite lam ~ 1e11/Mpc
    # at the grid floor)
    i0 = int(np.argmin(np.abs(core.boosts - 1e11)))
    n0 = np.zeros((n_b, 2))
    n0[i0, 0] = 1.0
    for tau in (0.01, 1.0):
        nt = core.reprocessed_nucleons(np.linspace(0.0, tau / rmax, 4),
                                       injection=n0)
        assert np.abs(nt.sum(axis=(0, 2)) - 1).max() < 1e-12

    # 3. + 5. delta injection: chained == single-interval, degradation,
    # no upward flow
    L = np.linspace(0.0, 5.0 / rmax, 9)
    n = core.reprocessed_nucleons(L, injection=n0)
    assert np.abs(n.sum(axis=(0, 2)) - 1).max() < 1e-12
    fresh = core.reprocessed_nucleons(np.array([L[0], L[-1]]),
                                      injection=n0)[:, -1, :]
    assert np.abs(fresh - n[:, -1, :]).max() < 1e-10 * fresh.max()
    lng = np.log(core.boosts)
    mbar = (n.sum(axis=2) * lng[:, None]).sum(axis=0) / n.sum(axis=(0, 2))
    assert np.all(np.diff(mbar) <= 1e-12)
    assert n[i0 + 1:, :, :].max() == 0.0
    print(f'  transport at tau=5: total conserved, <ln Gamma> {mbar[0]:.2f} -> {mbar[-1]:.2f}')

    # 6. thick field: sub-threshold occupation grows monotonically
    ith = int(np.argmax(R > 0.01 * rmax))
    below = n[:ith].sum(axis=(0, 2))
    assert np.all(np.diff(below) >= -1e-15)

    # 4. optically-thin limit: transport nu == one-generation fold + O(tau^2)
    tau = 1e-2
    Lt = np.linspace(0.0, tau / rmax, 6)
    P_tr = core.reprocessed_nucleons(Lt, injection=n0)
    _, N1 = core.neutrino_production(Lt, mass_range=[i_p, i_n],
                                     true_range=[i_p, i_n],
                                     boost_range=core.boosts, P=P_tr)
    P_1g = np.zeros((n_b, len(Lt), 1))
    P_1g[i0, :, 0] = np.exp(-R[i0] * Lt)
    _, N0 = core.neutrino_production(Lt, mass_range=[i_p], true_range=[i_p],
                                     boost_range=core.boosts, P=P_1g)
    t1, t0 = N1['nu_mu'][:, -1].sum(), N0['nu_mu'][:, -1].sum()
    print(f'  thin limit: transport/one-generation nu_mu = {t1 / t0:.6f}')
    assert abs(t1 / t0 - 1) < 5 * tau**2 and t1 >= t0 - 1e-15

    # 7. Duhamel: source-only total equals the trapezoid source integral
    rng = np.random.default_rng(7)
    q = rng.random((n_b, len(L), 2)) \
        * np.exp(-0.5 * ((lng[:, None, None] - 23) / 2)**2)
    ns = core.reprocessed_nucleons(L, source=q)
    tot = np.trapezoid(q.sum(axis=(0, 2)), L)
    assert abs(ns[:, -1, :].sum() - tot) < 1e-12 * tot

    # 8. cumulative=False pin + deposit conservation
    P_one = np.zeros((n_b, len(L), 1))
    P_one[i0, :, 0] = 1.0
    kwf = dict(mass_range=[i_p], true_range=[i_p], boost_range=core.boosts,
               P=P_one)
    rate = core.photomeson_production('p', L, cumulative=False, **kwf)
    cum = core.photomeson_production('p', L, **kwf)
    d = np.abs(cumulative_trapezoid(rate, L, axis=-1, initial=0.0) - cum).max()
    assert d < 1e-10 * max(cum.max(), 1e-300)
    w = rng.random(20)
    g = np.exp(rng.uniform(np.log(core.boosts[0]), np.log(core.boosts[-1]), 20))
    dep = deposit_log_cic(core.boosts, g, w)
    assert abs(dep.sum() - w.sum()) < 1e-12 * w.sum()

    # neutron-decay antineutrinos: one nubar_e per neutron with the boosted
    # beta spectrum — count conserved, mean fraction <E_nu>/E_n ~ 5.1e-4,
    # endpoint below 2 Q / m_n
    n_in = np.zeros(n_b)
    ihi = int(np.argmin(np.abs(core.boosts - 1e8)))
    n_in[ihi] = 1.0
    nd = core.neutron_decay_neutrinos(n_in)
    assert abs(nd.sum() - 1.0) < 1e-9
    E_g = 0.13957039 * core.boosts
    E_n = 0.93957 * core.boosts[ihi]
    fbar = (nd * E_g).sum() / E_n
    print(f'  neutron-decay nubar_e: count {nd.sum():.6f}, '
          f'<E>/E_n = {fbar:.2e} (expect ~5.1e-4)')
    assert 4.0e-4 < fbar < 6.5e-4
    assert E_g[nd > 0].max() < 2 * 0.782e-3 / 0.93957 * E_n * 1.1

    # cascade_nucleon_source: nonnegative, and its transport closes on the
    # source integral (three-term assembly is rate-consistent)
    alpha, mr, tr, _ = core.get_distribution_parameters(
        mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        absorption_type=('only mass', [1]))
    br = np.logspace(9.5, 11.5, 8)
    Lc = np.linspace(0.0, 3.0, 6)
    EV = core.species_evolution_boost_range(Lc, alpha=alpha, mass_range=mr,
                                            boost_range=br, true_range=tr)
    qs = core.cascade_nucleon_source(Lc, alpha=alpha, mass_range=mr,
                                     boost_range=br, true_range=tr, P=EV,
                                     weights=np.full(len(br), 0.1))
    assert qs.shape == (n_b, len(Lc), 2) and (qs >= 0).all()
    nf = core.reprocessed_nucleons(Lc, source=qs)
    tot = np.trapezoid(qs.sum(axis=(0, 2)), Lc)
    print(f'  cascade source: {tot:.3e} nucleons/inj integrated, transport closes '
          f'{abs(nf[:, -1, :].sum() / tot - 1):.1e}')
    assert abs(nf[:, -1, :].sum() - tot) < 1e-12 * tot


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

def test_sophia_spectrum_kernels():
    """photomeson_spectra=<AstroPhoMes model>: kernels built from the full
    SOPHIA x-distributions — same exact-rate normalization as the Rachen
    kernels, but with the complete secondary energy spread and real
    neutron-parent tables."""
    from crisp.data_download import get_astrophomes_path
    from crisp.interaction_rates import exact_rates_for_sigma
    from crisp.photonuclear_cross_sections import (Photomeson_Superposition,
                                                   load_astrophomes)
    from crisp.background_photon_models import cmb_photon_density_GeVcm3

    try:
        get_astrophomes_path(auto_download=False, verbose=False)
    except FileNotFoundError as exc:
        raise unittest.SkipTest(str(exc))

    pmm = load_astrophomes(auto_download=False)
    kw = dict(xsec_model=Photomeson_Superposition([(2, 4), (2, 3), (1, 2)]),
              target_photons=cmb_photon_density_GeVcm3, photomeson='kernels',
              boosts=np.logspace(0, 12, 131), eps=np.logspace(-2, 6, 300))
    core = InteractionCore(photomeson_spectra=pmm, **kw)
    ref = InteractionCore(**kw)

    i11 = int(np.argmin(np.abs(core.boosts - 1e11)))
    R_s = (core.photomeson_kernels['p'][0]
           + core.photomeson_kernels['n'][0]).sum(axis=1)
    R_r = (ref.photomeson_kernels['p'][0]
           + ref.photomeson_kernels['n'][0]).sum(axis=1)
    print(f'  GZK proton rate at 1e11: SOPHIA {R_s[i11]:.4f} vs '
          f'Rachen {R_r[i11]:.4f} /Mpc')
    assert abs(R_s[i11] / R_r[i11] - 1) < 0.15

    # nucleon multiplicity per interaction ~ 1 (tables), <= 1.35 everywhere
    sig = R_r > 1e-3 * R_r.max()
    r_ex = exact_rates_for_sigma(core.boosts, cmb_photon_density_GeVcm3,
                                 np.asarray(pmm.egrid, dtype=float),
                                 np.vstack([pmm.cs_proton_grid,
                                            pmm.cs_neutron_grid]) * 1e-3)
    mult_N = R_s[sig] / r_ex[0][sig]
    assert 0.9 < mult_N.min() and mult_N.max() < 1.35

    # the point of the feature: the secondary spectra are genuinely BROAD
    lg = np.log10(core.boosts)
    widths = {}
    for lab, c in [('SOPHIA', core), ('Rachen', ref)]:
        row = c.pion_prod_tensor[0][i11]
        cum = np.cumsum(row) / row.sum()
        widths[lab] = np.interp(0.75, cum, lg) - np.interp(0.25, cum, lg)
    print(f"  pion secondary IQR at 1e11: SOPHIA {widths['SOPHIA']:.2f} vs "
          f"Rachen {widths['Rachen']:.2f} decades")
    assert widths['SOPHIA'] > 3 * widths['Rachen']

    # the decay chain is untouched: still 3 nu per charged pion
    br = core.boosts[110:116]
    P1 = np.ones((len(br), 2, 1))
    i_p = core.species.index((1, 1))
    fold_kw = dict(alpha=np.ones(1), mass_range=[i_p], true_range=[i_p],
                   boost_range=br, P=P1)
    L = np.array([0.0, 1.0])
    n_pi = (core.photomeson_production('pi+', L, **fold_kw)[:, -1].sum()
            + core.photomeson_production('pi-', L, **fold_kw)[:, -1].sum())
    _, N_nu = core.neutrino_production(L, **fold_kw)
    n_nu = N_nu['nu_mu'][:, -1].sum() + N_nu['nu_e'][:, -1].sum()
    print(f'  nu per charged pion (SOPHIA kernels): {n_nu / n_pi:.3f}')
    assert abs(n_nu / n_pi - 3.0) < 0.05


def test_kaon_component():
    """K+/K- from the SOPHIA tables — the last Huemmer et al. (2010)
    inventory item: the kernels carry the strangeness channel with the
    right threshold ordering and charge asymmetry, the K -> mu nu_mu chain
    (BR 0.6356) adds exactly 3*BR neutrinos per decayed kaon with the
    (1-rK) mK/mpi energy placement, defaults stay bit-exact, and the kaon
    component evades the synchrotron cooling that suppresses the pions —
    their reason for tracking kaons at all."""
    from crisp.data_download import get_astrophomes_path
    from crisp.photonuclear_cross_sections import (Photomeson_Superposition,
                                                   load_astrophomes)
    from crisp.background_photon_models import cmb_photon_density_GeVcm3

    try:
        get_astrophomes_path(auto_download=False, verbose=False)
    except FileNotFoundError as exc:
        raise unittest.SkipTest(str(exc))

    pmm = load_astrophomes(auto_download=False)
    core = InteractionCore(
        xsec_model=Photomeson_Superposition([(2, 4), (2, 3), (1, 2)]),
        target_photons=cmb_photon_density_GeVcm3, photomeson='kernels',
        photomeson_spectra=pmm,
        boosts=np.logspace(0, 12, 131), eps=np.logspace(-2, 6, 300))

    # kernels: thresholds (K+ ~1 GeV associated production, K- ~1.5 GeV
    # pair production, both far above the pion turn-on) and K+ >= K-
    b = core.boosts
    RK_p = core.photomeson_kernels['K+'][0].sum(axis=1)
    RK_m = core.photomeson_kernels['K-'][0].sum(axis=1)
    Rpi = core.pion_prod_tensor[0].sum(axis=1)
    ipi = np.argmax(Rpi > 1e-6 * Rpi.max())
    ikp = np.argmax(RK_p > 1e-6 * RK_p.max())
    ikm = np.argmax(RK_m > 1e-6 * RK_m.max())
    print(f'  first active boost: pi {b[ipi]:.2e}, K+ {b[ikp]:.2e}, '
          f'K- {b[ikm]:.2e}')
    assert b[ikp] > 3 * b[ipi] and b[ikm] > b[ikp]
    assert (RK_p - RK_m).min() > -1e-12 * RK_p.max()

    # tables: near-threshold K+ dominance, high-energy K+/K- convergence
    xw = np.asarray(pmm.xwidths, dtype=float)
    e_grid = np.asarray(pmm.egrid, dtype=float)
    mult = {pid: (np.asarray(pmm.redist_proton[pid], dtype=float)
                  * xw).sum(axis=1) for pid in (50, 51)}
    i2 = int(np.argmin(np.abs(e_grid - 2.0)))
    i4 = int(np.argmin(np.abs(e_grid - 1e4)))
    print(f'  K-/K+ multiplicity: {mult[51][i2] / mult[50][i2]:.3f} at 2 GeV, '
          f'{mult[51][i4] / mult[50][i4]:.3f} at 1e4 GeV')
    assert mult[51][i2] < 0.3 * mult[50][i2]
    assert 0.8 < mult[51][i4] / mult[50][i4] < 1.2

    # chain: count normalization and energy placement on the shared grid
    CK = core._kaon_decay_chain()
    C = core._pion_decay_chain()
    assert abs(CK['box'][60].sum() - 1) < 1e-6        # one nu per decay
    dl = np.log(b[1]) - np.log(b[0])
    off = np.log((1 - CK['rK']) * CK['mK'] / C['mpi']) / dl
    jmax = int(np.max(np.nonzero(CK['box'][60])[0]))
    print(f'  kaon box endpoint offset: {jmax - 60} bins (expect ~{off:.1f})')
    assert abs(jmax - 60 - off) <= 1.0                # (1-rK) mK/mpi endpoint
    assert jmax > np.max(np.nonzero(C['box'][60])[0])  # harder than pion box

    # counting: 3*BR extra nu per decayed kaon; defaults bit-exact
    L = np.array([0.0, 1.0])
    i_p = core.species.index((1, 1))
    w = np.zeros(len(b))
    w[124] = 1.0                                      # boost 2.9e11, K-active
    fk = dict(alpha=None, mass_range=[i_p], boost_range=b, true_range=[i_p],
              P=np.ones((len(b), len(L), 1)), weights=w)
    _, N0 = core.neutrino_production(L, **fk)
    _, N1 = core.neutrino_production(L, **fk, kaons=True)
    _, N2 = core.neutrino_production(L, **fk, kaons=False)
    assert all(np.array_equal(N0['detail'][k], N2['detail'][k])
               for k in N0['detail'])
    tot0 = sum(N0['detail'][k][:, -1].sum() for k in N0['detail'])
    tot1 = sum(N1['detail'][k][:, -1].sum() for k in N1['detail'])
    nK = sum(core._photomeson_fold(core.photomeson_kernels[s], L,
                                   scaling_group=s, **fk)[:, -1].sum()
             for s in ('K+', 'K-'))
    per_K = (tot1 - tot0) / nK
    print(f'  extra nu per decayed kaon: {per_K:.3f} (3*BR = {3 * 0.6356:.3f})')
    assert abs(per_K / (3 * 0.6356) - 1) < 0.05

    # cooling: kaons evade the suppression that crushes the pions
    f_pi = core.decay_before_cooling(0.13957039 * b, 0.13957039, 2.6033e-8, 1e6)
    f_K = core.decay_before_cooling(CK['mK'] * b, CK['mK'], CK['tauK'], 1e6)
    assert (f_K[80:] / f_pi[80:]).min() > 10
    _, N0B = core.neutrino_production(L, **fk, B_gauss=1e6)
    _, N1B = core.neutrino_production(L, **fk, B_gauss=1e6, kaons=True)
    hi = b > 1e10                                     # the extreme end

    def kaon_share(Nk, Nn):
        t_k = sum(Nk['detail'][k][hi, -1].sum() for k in Nk['detail'])
        t_n = sum(Nn['detail'][k][hi, -1].sum() for k in Nn['detail'])
        return (t_k - t_n) / t_k
    s_unc, s_coo = kaon_share(N1, N0), kaon_share(N1B, N0B)
    print(f'  kaon share above E = 1e10 m_pi: uncooled {s_unc:.4f}, '
          f'B = 1e6 G {s_coo:.4f} (x{s_coo / s_unc:.0f})')
    assert s_coo > 30 * s_unc and s_coo > 0.02

    # kaons='K+' (Huemmer/NeuCosmA scope): the K- side is dropped — the
    # extra nu over kaons=False is 3*BR per decayed K+ only, the K- part
    # of the kaons=True output is 3*BR per K-, and the dropped content is
    # dominated by the direct nubar_mu box (the antineutrino shoulder)
    _, Nkp = core.neutrino_production(L, **fk, kaons='K+')
    tot_kp = sum(Nkp['detail'][k][:, -1].sum() for k in Nkp['detail'])
    nKp = core._photomeson_fold(core.photomeson_kernels['K+'], L,
                                scaling_group='K+', **fk)[:, -1].sum()
    nKm = nK - nKp
    per_Kp = (tot_kp - tot0) / nKp
    per_Km = (tot1 - tot_kp) / nKm
    print(f"  kaons='K+': extra nu per K+ {per_Kp:.3f}, "
          f"K- difference per K- {per_Km:.3f} (both ~ 3*BR)")
    assert abs(per_Kp / (3 * 0.6356) - 1) < 0.05
    assert abs(per_Km / (3 * 0.6356) - 1) < 0.05
    d_nbm = (N1['detail']['nubar_mu'] - Nkp['detail']['nubar_mu'])[:, -1].sum()
    assert d_nbm > 0.5 * 0.6356 * nKm          # K- box lives in nubar_mu
    try:
        core.neutrino_production(L, **fk, kaons='K-')
        raise AssertionError("kaons='K-' must raise")
    except ValueError:
        pass

    # error path: kaons=True needs the strangeness kernels
    Kp = core.photomeson_kernels.pop('K+')
    try:
        core.neutrino_production(L, **fk, kaons=True)
        raise AssertionError('kaons=True without K+ kernels must raise')
    except ValueError as exc:
        assert 'photomeson_spectra' in str(exc)
    finally:
        core.photomeson_kernels['K+'] = Kp


def test_secondary_cooling_migration():
    """cooling='migrate': the exact cooled-decay transport replacing the
    no-migration drop treatment. The closed form F(E|E0) =
    exp(-E_c^2 (1/E^2 - 1/E0^2)) is row-stochastic (cooling migrates
    energy, never number): far above the break the secondaries pile up at
    E_br/sqrt(3) instead of vanishing, and the end-to-end nu count per
    charged pion stays exactly the uncooled 3 while the drop treatment
    deletes everything."""
    from crisp.data_download import get_astrophomes_path
    from crisp.photonuclear_cross_sections import (Photomeson_Superposition,
                                                   load_astrophomes)
    from crisp.background_photon_models import cmb_photon_density_GeVcm3

    try:
        get_astrophomes_path(auto_download=False, verbose=False)
    except FileNotFoundError as exc:
        raise unittest.SkipTest(str(exc))

    pmm = load_astrophomes(auto_download=False)
    core = InteractionCore(
        xsec_model=Photomeson_Superposition([(2, 4), (2, 3), (1, 2)]),
        target_photons=cmb_photon_density_GeVcm3, photomeson='kernels',
        photomeson_spectra=pmm,
        boosts=np.logspace(0, 12, 131), eps=np.logspace(-2, 6, 300))

    b = core.boosts
    mpi, tau_pi, Bg = 0.13957039, 2.6033e-8, 1e6
    E = mpi * b
    S = core.cooled_decay_matrix(E, mpi, tau_pi, Bg)

    # row-stochastic (decays on-grid here: the break is mid-grid)
    f = core.decay_before_cooling(E, mpi, tau_pi, Bg)
    E_br = E[int(np.argmin(np.abs(f - 0.5)))]
    assert np.abs(S.sum(axis=1) - 1).max() < 1e-12

    # pile-up at E_br / sqrt(3) for injection far above the break
    i_hi = int(np.argmin(np.abs(E - 1e4 * E_br)))
    jpk = int(np.argmax(S[i_hi]))
    print(f'  pile-up: {E[i_hi]:.1e} GeV -> {E[jpk]:.2e} '
          f'(E_br/sqrt3 = {E_br / np.sqrt(3):.2e})')
    assert abs(np.log(E[jpk] / (E_br / np.sqrt(3)))) < 2 * np.log(b[1] / b[0])

    # identity limit far below the break
    assert S[20, 20] > 1 - 2 * (E[20] / E_br) ** 2

    # closed form vs direct quadrature of p(E|E0), one hard row
    f0 = float(core.decay_before_cooling(E[-1], mpi, tau_pi, Bg))
    E_c2 = E[-1] ** 2 * f0 / (1 - f0) / 2
    E0 = E[i_hi]
    Eq = np.logspace(np.log10(E[0] * 0.9), np.log10(E0), 400000)
    Eq = Eq[Eq <= E0]
    p = (2 * E_c2 / Eq ** 3) * np.exp(-E_c2 * (1 / Eq ** 2 - 1 / E0 ** 2))
    edges = np.concatenate([[E[0] ** 2 / E[1]], np.sqrt(E[:-1] * E[1:]),
                            [E[-1] ** 2 / E[-2]]])
    for j in np.nonzero(S[i_hi] > 1e-3)[0]:
        m_ = (Eq >= edges[j]) & (Eq < min(edges[j + 1], E0))
        q = np.trapezoid(p[m_], Eq[m_])
        assert abs(S[i_hi, j] - q) < 1e-3 * S[i_hi].max()

    # end-to-end: number conserved, energy migrated (deep above the break)
    L = np.array([0.0, 1.0])
    i_p = core.species.index((1, 1))
    br = b[95:105]
    fk = dict(alpha=np.ones(1), mass_range=[i_p], true_range=[i_p],
              boost_range=br, P=np.ones((len(br), 2, 1)))
    n_pi = (core.photomeson_production('pi+', L, **fk)[:, -1].sum()
            + core.photomeson_production('pi-', L, **fk)[:, -1].sum())
    res = {}
    for lab, kw in [('uncooled', {}), ('drop', dict(B_gauss=Bg)),
                    ('migrate', dict(B_gauss=Bg, cooling='migrate'))]:
        _, N = core.neutrino_production(L, **fk, **kw)
        n_nu = N['nu_mu'][:, -1].sum() + N['nu_e'][:, -1].sum()
        tot = sum(N['detail'][k][:, -1] for k in N['detail'])
        res[lab] = (n_nu / n_pi,
                    np.exp((tot * np.log(E)).sum() / max(tot.sum(), 1e-300)))
    print(f"  nu per charged pion at B=1e6 G: uncooled {res['uncooled'][0]:.4f}"
          f", drop {res['drop'][0]:.4f}, migrate {res['migrate'][0]:.4f}; "
          f"mean E_nu {res['uncooled'][1]:.1e} -> {res['migrate'][1]:.1e} GeV")
    assert abs(res['migrate'][0] / res['uncooled'][0] - 1) < 1e-5
    assert abs(res['migrate'][0] - 3) < 0.05
    assert res['drop'][0] < 1.0
    assert res['migrate'][1] < 0.1 * res['uncooled'][1]

    # defaults bit-exact, error paths
    _, N0 = core.neutrino_production(L, **fk, B_gauss=Bg)
    _, N1 = core.neutrino_production(L, **fk, B_gauss=Bg, cooling='drop')
    assert all(np.array_equal(N0['detail'][k], N1['detail'][k])
               for k in N0['detail'])
    for kw in (dict(cooling='migrate'), dict(cooling='bogus')):
        try:
            core.neutrino_production(L, **fk, **kw)
            raise AssertionError(f'{kw} must raise')
        except ValueError:
            pass


def test_empirical_channels():
    """channels='empirical': the Morejon et al. (2019) fragment physics as
    exclusive residual channels — one heavy survivor per event, weights
    renormalized to exactly one event, light fragments through the ejecta
    budget. Carries <Delta A> ~ 5.7 for Fe-56 instead of the superposition
    skeleton's 1."""
    from crisp.data_download import get_astrophomes_path
    from crisp.photonuclear_cross_sections import load_astrophomes

    try:
        get_astrophomes_path(auto_download=False, verbose=False)
    except FileNotFoundError as exc:
        raise unittest.SkipTest(str(exc))

    pm = load_astrophomes(model='EmpiricalModel', channels='empirical',
                          auto_download=False)
    fe = [k for k in pm.incl_idcs if k[0] == 5626]
    w = np.array([float(np.asarray(pm.multiplicity[k])) for k in fe])
    dA = np.array([56 - k[1] // 100 for k in fe], float)
    print(f'  Fe-56: {len(fe)} residual channels, weight sum {w.sum():.6f}, '
          f'<Delta A> = {w @ dA:.2f}')
    assert abs(w.sum() - 1) < 1e-12                 # exactly one event
    assert 5.0 < w @ dA < 6.5                       # the Morejon mass loss
    assert all(k[1] // 100 >= 28 for k in fe)       # no fragment channels
    assert w.min() >= 1e-3 / 1.2                    # trimmed tail

    # every kept mother is a one-event mixture; light mothers keep
    # physical residuals (Li-6 -> He/t/d, not bare nucleons)
    mothers = sorted({k[0] for k in pm.incl_idcs})
    for m in mothers[::40]:
        wm = sum(float(np.asarray(pm.multiplicity[k]))
                 for k in pm.incl_idcs if k[0] == m)
        assert abs(wm - 1) < 1e-12, m
    li6 = [(k[1] // 100, k[1] % 100) for k in pm.incl_idcs if k[0] == 603]
    assert li6 and min(a for a, z in li6) >= 3

    try:
        load_astrophomes(model='EmpiricalModel', channels='bogus',
                         auto_download=False)
        raise AssertionError('bad channels= must raise')
    except ValueError:
        pass

    # fragment_yields: the model's own inclusive fragment mix, rescaled to
    # close A and Z exactly against the channel table; one struck nucleon
    # per event goes to the wide budget
    from crisp.photonuclear_cross_sections import Photomeson
    xspm = Photomeson(pmm=pm, filter_nuclei=lambda n: n == (26, 56))
    fy = xspm.fragment_yields(26, 56)
    A_L = np.array([4., 3., 3., 2., 1., 1.])
    Z_L = np.array([2., 2., 1., 1., 1., 0.])
    dA = w @ np.array([56 - k[1] // 100 for k in fe], float)
    dZ = w @ np.array([26 - k[1] % 100 for k in fe], float)
    assert abs(A_L @ fy['narrow'] + 1.0 - dA) < 1e-9
    assert abs(Z_L @ fy['narrow'] + fy['wide_p'] - dZ) < 1e-9
    assert fy['narrow'].min() >= 0 and 0 < fy['wide_p'] < 1
    assert abs(fy['wide_p'] + fy['wide_n'] - 1) < 1e-12
    print(f"  Fe-56 narrow yields [He4,He3,t,d,p,n]: "
          + np.array2string(fy['narrow'], precision=3)
          + f"; struck p-fraction {fy['wide_p']:.3f}")

    # the superposition hybrid has no fragment data -> None (bit-exact path)
    pm_sup = load_astrophomes(model='EmpiricalModel',
                              channels='superposition', auto_download=False)
    xs_sup = Photomeson(pmm=pm_sup, filter_nuclei=lambda n: n == (26, 56))
    assert xs_sup.fragment_yields(26, 56) is None


def test_nuclear_decay_rates():
    """nuclear_decay_On=True: spontaneous decays of TRACKED species enter
    the main tensor as boost-diluted jump rates 1/(gamma tau c) per
    branching (Be-7 EC -> Li-7 pinned against nubase), reported under
    rates_by_interaction()['decay']; default off stays bit-identical and
    conservation stays machine-exact (EC preserves A)."""
    import tempfile
    from crisp.photonuclear_cross_sections import Inclusive_model
    from crisp.background_photon_models import cmb_photon_density_GeVcm3
    from crisp.core import c_in_Mpc_sec
    from crisp.data.nucleardecays import NuclearDataTable

    eps = np.linspace(2, 150, 30)
    sig = 40.0 * np.exp(-0.5 * ((eps - 25) / 8.0) ** 2) + 1.0
    tmp = tempfile.mkdtemp()
    np.savetxt(f'{tmp}/grid.txt', eps)
    # identity-exact chain: Be-7 -> Li-6 + p, Li-7 -> Li-6 + n,
    # Li-6 -> He4 + d, He-4 -> 2 d, d -> p + n; Be-7 decays by electron
    # capture to Li-7 (nubase: t1/2 = 53.22 d, BR 1), both tracked
    with open(f'{tmp}/nonel.txt', 'w') as f:
        for nucid, fac in ((704, 1.0), (703, 0.9), (603, 0.8),
                           (402, 0.5), (201, 0.3)):
            f.write(f'{nucid} ' + ' '.join(f'{fac * s:.5f}' for s in sig) + '\n')
    with open(f'{tmp}/incl.txt', 'w') as f:
        for mo, prod, mult, fac in ((704, 603, 1.0, 1.0), (704, 101, 1.0, 1.0),
                                    (703, 603, 1.0, 0.9), (703, 100, 1.0, 0.9),
                                    (603, 402, 1.0, 0.8), (603, 201, 1.0, 0.8),
                                    (402, 201, 2.0, 0.5),
                                    (201, 101, 1.0, 0.3), (201, 100, 1.0, 0.3)):
            f.write(f'{mo} {prod} '
                    + ' '.join(f'{mult * fac * s:.5f}' for s in sig) + '\n')

    decays = NuclearDataTable().prepare_decay_table()
    kw = dict(target_photons=cmb_photon_density_GeVcm3, photomeson='kernels',
              boosts=np.logspace(0, 12, 25), eps=np.logspace(-4, 2, 120))
    m = Inclusive_model(egrid=f'{tmp}/grid.txt', nonel=f'{tmp}/nonel.txt',
                        incl=f'{tmp}/incl.txt', max_mass=16, cache=False)
    core_on = InteractionCore(xsec_model=m, decays=decays,
                              nuclear_decay_On=True, **kw)
    core_off = InteractionCore(xsec_model=m, decays=decays, **kw)

    i_be = core_on.species.index((4, 7))
    i_li = core_on.species.index((3, 7))
    tau = decays[704]['decay_time']                  # mean life, seconds
    lam = 1.0 / (core_on.boosts * tau * c_in_Mpc_sec)
    jump = core_on.tensor[i_be, i_li] - core_off.tensor[i_be, i_li]
    assert np.allclose(jump, lam, rtol=1e-6)         # the EC jump rate
    print(f'  Be-7 EC jump at gamma=1e6: {jump[12]:.3e} /Mpc '
          f'(= 1/(gamma tau c), tau {tau/86400:.1f} d)')

    dec = core_on.rates_by_interaction((4, 7))['decay']
    assert np.allclose(dec, lam, rtol=1e-12)
    assert 'decay' not in core_off.rates_by_interaction((4, 7))
    assert np.array_equal(core_off.tensor,
                          InteractionCore(xsec_model=m, decays=decays, **kw).tensor)

    a_on, _ = core_on.conservation_imbalance()
    a_off, _ = core_off.conservation_imbalance()
    print(f'  conservation A: on {a_on:.1e} / off {a_off:.1e}')
    assert a_on < CONSERVE and a_off < CONSERVE


def test_inclusive_model():
    """Inclusive_model: user tables (egrid/nonel/incl, nucid = 100A + Z)
    satisfying the mass-closure identity sum_d A_d sigma_d = A sigma_nonel;
    heavy survivors as verbatim channels, deficit events routed to light
    species (capacity-capped), light yields = the inclusive multiplicity
    excess, inert-species treatment of stable-but-untracked remnants —
    conservation machine-exact through a full core build."""
    import tempfile
    from crisp.photonuclear_cross_sections import Inclusive_model
    from crisp.background_photon_models import cmb_photon_density_GeVcm3

    eps = np.linspace(2, 150, 30)
    sig = 40.0 * np.exp(-0.5 * ((eps - 25) / 8.0) ** 2) + 1.0
    tmp = tempfile.mkdtemp()
    np.savetxt(f'{tmp}/grid.txt', eps)
    # identity-exact chain: every mother's products close A and Z.
    # C-12: 75% -> B-11 + p, 25% -> Be-9 + He3 (Be-9 = the inert case);
    # B-11 -> Be-10 + p; Be-10 -> 2 He4 + 2 n (deficit routing: one He4
    # becomes the channel, one the yield); He-4 -> 2 d (routing to a light
    # mother); d -> p + n (routing to the proton species itself)
    with open(f'{tmp}/nonel.txt', 'w') as f:
        for nucid, fac in ((1206, 1.0), (1105, 0.9), (1004, 0.8),
                           (402, 0.4), (201, 0.2)):
            f.write(f'{nucid} ' + ' '.join(f'{fac * s:.5f}' for s in sig) + '\n')
    with open(f'{tmp}/incl.txt', 'w') as f:
        def row(mo, prod, mult, fac):
            f.write(f'{mo} {prod} '
                    + ' '.join(f'{mult * fac * s:.5f}' for s in sig) + '\n')
        row(1206, 1105, 0.75, 1.0)
        row(1206, 904, 0.25, 1.0)
        row(1206, 101, 0.75, 1.0)
        row(1206, 302, 0.25, 1.0)
        row(1105, 1004, 1.0, 0.9)
        row(1105, 101, 1.0, 0.9)
        row(1004, 402, 2.0, 0.8)
        row(1004, 100, 2.0, 0.8)
        row(402, 201, 2.0, 0.4)
        row(201, 101, 1.0, 0.2)
        row(201, 100, 1.0, 0.2)

    m = Inclusive_model(egrid=f'{tmp}/grid.txt', nonel=f'{tmp}/nonel.txt',
                        incl=f'{tmp}/incl.txt', max_mass=16, cache=False)
    assert m.nuclei == [(1, 2), (2, 4), (4, 10), (5, 11), (6, 12)]
    e_t = np.array([10.0, 30.0, 80.0])
    ch = (m.cross_section(e_t, 6, 12, rem=(5, 11))
          + m.cross_section(e_t, 6, 12, rem=(4, 9)))
    assert np.allclose(ch, m.cross_section(e_t, 6, 12), rtol=1e-4)
    r = m.cross_section(e_t, 6, 12, rem=(5, 11)) \
        / m.cross_section(e_t, 6, 12, rem=(4, 9))
    assert np.allclose(r, 3.0, rtol=1e-4)          # per-energy branching

    mult = m.light_inclusive_multiplicity(6, 12)
    assert mult is not None
    # sigma_incl/sigma_nonel: p = 0.75, He3 = 0.25
    ii = int(np.argmin(np.abs(m.eps - 30.0)))
    print(f'  C-12 light multiplicities at 30 MeV [He4, He3, .., p, n]: '
          f'{np.round(mult[:, ii], 3)}')
    assert abs(mult[4, ii] - 0.75) < 1e-4 and abs(mult[1, ii] - 0.25) < 1e-4

    # light yields = inclusive content minus the one-per-event channel
    # allocation: Be-10 keeps ONE He4 (of multiplicity 2) and both n;
    # He-4 keeps one d; d's own p is fully consumed by its (1, 1) channel
    e_pk = np.array([25.0])
    tot10 = m.cross_section(e_pk, 4, 10)[0]
    ly10 = m.light_yield_sigma(e_pk, 4, 10)
    assert abs(ly10[0, 0] / tot10 - 1.0) < 1e-4    # He4 yield
    assert abs(ly10[5, 0] / tot10 - 2.0) < 1e-4    # both neutrons
    assert abs(m.cross_section(e_pk, 4, 10, rem=(2, 4))[0] / tot10
               - 1.0) < 1e-4                       # the He4 channel
    ly4 = m.light_yield_sigma(e_pk, 2, 4)
    assert abs(ly4[3, 0] / m.cross_section(e_pk, 2, 4)[0] - 1.0) < 1e-4
    lyd = m.light_yield_sigma(e_pk, 1, 2)
    assert abs(lyd[4, 0]) < 1e-6                   # p went to the channel
    assert abs(lyd[5, 0] / m.cross_section(e_pk, 1, 2)[0] - 1.0) < 1e-4

    core = InteractionCore(xsec_model=m,
                           target_photons=cmb_photon_density_GeVcm3,
                           photomeson='kernels',
                           boosts=np.logspace(0, 12, 61),
                           eps=np.logspace(-4, 2, 200))
    a, z = core.conservation_imbalance()
    print(f'  core conservation: A {a:.1e}, Z {z:.1e}; '
          f'Be-9 inert-tracked: {(4, 9) in core.species}')
    assert (4, 9) in core.species                  # stable untracked -> inert
    assert a < CONSERVE


def test_tabulated_disintegration():
    """TabulatedDisintegration: user-supplied totals + multiplicities as
    exclusive residual channels — per-energy renormalization (exactly one
    residual per interaction), scalar AND energy-dependent weights, and
    machine-exact conservation through a full core build (a connected
    A = 9..2 chain, with Be-8 exercising the decay resolution)."""
    import io as _io
    from crisp.photonuclear_cross_sections import TabulatedDisintegration
    from crisp.background_photon_models import cmb_photon_density_GeVcm3

    eps = np.linspace(5, 200, 40)
    sig = 60.0 * np.exp(-0.5 * ((eps - 20) / 6.0) ** 2) + 1.5
    tot = 'eps_MeV ' + ' '.join(f'{e:.3f}' for e in eps) + '\n'
    for (Z, A), fac in [((4, 9), 1.0), ((3, 7), 0.8), ((3, 6), 0.7),
                        ((2, 4), 0.5)]:
        tot += f'{Z} {A} ' + ' '.join(f'{fac * s:.5f}' for s in sig) + '\n'
    w_ed = 1.0 + (eps > 40) * 3.0                 # per-energy weights
    mult = ('4 9 4 8 3.0\n'
            '4 9 3 7 ' + ' '.join(f'{w:.3f}' for w in w_ed) + '\n'
            '3 7 3 6 1.0\n'
            '3 6 2 4 1.0\n'
            '2 4 1 3 0.7\n2 4 1 2 0.3\n')

    m = TabulatedDisintegration(totals=_io.StringIO(tot),
                                multiplicities=_io.StringIO(mult))
    assert m.nuclei == [(2, 4), (3, 6), (3, 7), (4, 9)]
    e_test = np.array([10.0, 30.0, 60.0, 150.0])
    ch_sum = (m.cross_section(e_test, 4, 9, rem=(4, 8))
              + m.cross_section(e_test, 4, 9, rem=(3, 7)))
    assert np.allclose(ch_sum, m.cross_section(e_test, 4, 9), rtol=1e-12)
    r_lo = m.cross_section(np.array([20.0]), 4, 9, rem=(4, 8)) \
        / m.cross_section(np.array([20.0]), 4, 9, rem=(3, 7))
    r_hi = m.cross_section(np.array([100.0]), 4, 9, rem=(4, 8)) \
        / m.cross_section(np.array([100.0]), 4, 9, rem=(3, 7))
    print(f'  branching ratios: {r_lo[0]:.3f} below / {r_hi[0]:.3f} above '
          f'the 40 MeV step (expect 3.000 / 0.750)')
    assert abs(r_lo[0] - 3.0) < 1e-9 and abs(r_hi[0] - 0.75) < 1e-9
    assert np.allclose(m.cross_section(e_test, 4, 9, nloss=2),
                       m.cross_section(e_test, 4, 9, rem=(3, 7)))

    core = InteractionCore(xsec_model=m,
                           target_photons=cmb_photon_density_GeVcm3,
                           photomeson='kernels',
                           boosts=np.logspace(0, 12, 61),
                           eps=np.logspace(-4, 2, 200))
    a, z = core.conservation_imbalance()
    print(f'  core conservation: A {a:.1e}, Z {z:.1e}; '
          f'tensor max {np.abs(core.tensor).max():.3e}')
    assert np.abs(core.tensor).max() > 0
    assert a < CONSERVE

    for bad in ('4 9 5 10 1.0\n', '4 9 4 8 1 2 3\n'):
        try:
            TabulatedDisintegration(totals=_io.StringIO(tot),
                                    multiplicities=_io.StringIO(bad))
            raise AssertionError('must raise: ' + bad.strip())
        except ValueError:
            pass


def test_gdr_atlas_quasi_deuteron():
    """The Levinger quasi-deuteron term of the GDR atlas: the Chadwick+91
    formula for all nuclei, routed through its physical n + p channel
    (Z-1, A-2) instead of the GDR neutron-loss branchings, with the channel
    sum reproducing the total identically."""
    from crisp.photonuclear_cross_sections import GDR_atlas

    m = GDR_atlas()
    eps = np.linspace(10.5, 139.5, 400)

    # formula: L (NZ/A) sigma_d f, L = 6.5 (397.8 = 6.5 x 61.2)
    qd80 = float(m.quasi_deuteron_cross_section(np.array([80.0]), 26, 56)[0])
    hand = (397.8 * 26 * 30 / 56
            * np.polyval([9.3537e-9, -3.4762e-6, 4.1222e-4, -9.8343e-3, 8.3714e-2], 80.0)
            * (80.0 - 2.224)**1.5 / 80.0**3)
    assert np.isclose(qd80, hand, rtol=1e-12)

    # Pauli-blocking factor continuous at both joints (24.2 rounding at 140)
    lo = m.quasi_deuteron_cross_section(np.array([19.999, 20.001]), 26, 56)
    hi = m.quasi_deuteron_cross_section(np.array([139.99, 140.01]), 26, 56)
    assert abs(lo[1] / lo[0] - 1) < 1e-3 and abs(hi[1] / hi[0] - 1) < 1e-2

    # channel structure: QD remnant present; sum rule to machine precision
    i = m.nuclei.index((26, 56))
    assert (25, 54) in m.channels[i]
    tot = m.total_cross_section(eps, 26, 56)
    csum = sum(np.asarray(m.cross_section(eps, 26, 56, rem=tuple(r)))
               for r in m.channels[i])
    assert np.abs(csum - tot).max() < 1e-12 * tot.max()
    d_qd = np.abs(np.asarray(m.cross_section(eps, 26, 56, rem=(25, 54)))
                  - m.quasi_deuteron_cross_section(eps, 26, 56)).max()
    assert d_qd < 1e-12 * tot.max()
    print(f'  QD at 80 MeV (Fe-56): {qd80:.3f} mb; channel sum == total, '
          f'n+p channel (25, 54) present')

    # PSB-style energy regions: exclusive 1n/2n at the GDR proper, the
    # multiplicity table above 30 MeV — pinned via the channel-sigma-weighted
    # mean mass loss per interaction
    def mean_dA(e0, e1):
        eg = np.linspace(e0, e1, 200)
        sig = np.array([np.trapezoid(m.cross_section(eg, 26, 56, rem=tuple(r)), eg)
                        for r in m.channels[i]])
        dA = np.array([56 - r[1] for r in m.channels[i]])
        return (sig * dA).sum() / sig.sum()

    dA_gdr, dA_hi = mean_dA(12., 25.), mean_dA(35., 60.)
    print(f'  mean Delta A per interaction: {dA_gdr:.2f} at the GDR peak '
          f'(TALYS ~1.14), {dA_hi:.2f} at 35-60 MeV')
    assert 1.1 < dA_gdr < 1.35
    assert 2.5 < dA_hi < 4.5


def test_gdra_construction():
    """The unified core on the (filtered) IAEA GDR atlas — the phase-5 gap is
    closed: filter_nuclei bounds the species set (the raw atlas spans 8980
    nuclides up to A = 339 and must NEVER be used unfiltered) and the decay
    ladder resolves the atlas's proton-rich neutron-loss remnants. Also pins
    the quasi-deuteron n + p channel emitting direct protons, and the Fe-56
    CMB rate against PSB."""
    from crisp.photonuclear_cross_sections import GDR_atlas

    gdra = GDR_atlas(filter_nuclei=lambda n: n[1] <= 56 and n[0] >= 1)
    core = InteractionCore(xsec_model=gdra, decays=decay_table(),
                           boosts=np.logspace(6, 14, 201))
    a, z = core.conservation_imbalance()
    print(f'  GDRA A<=56 core: {len(core.species)} species, '
          f'conservation A {a:.1e}, Z {z:.2f} (beta+ of the n-loss remnants)')
    # inert species (stable untracked remnants) now tracked: was < 600
    assert 500 < len(core.species) < 700
    assert a < CONSERVE

    # the quasi-deuteron channel emits one proton AND one neutron
    # (boost-preserving light yields): the atlas has a direct-p channel
    p_direct = core.light_prod_tensor[4].max()
    n_direct = core.light_prod_tensor[5].max()
    print(f'  direct yields: p {p_direct:.3e}, n {n_direct:.3e} /Mpc (QD gives p > 0)')
    assert p_direct > 0 and n_direct > p_direct

    # physics anchor: Fe-56 total rate on the CMB at the GDR overlap vs PSB
    psb = InteractionCore(xsec_model=psb_xsec(), boosts=core.boosts)
    b = int(np.argmin(np.abs(core.boosts - 1.4e10)))
    r_g = core.all_rates[core.nuclei.index((26, 56))][b]
    r_p = psb.all_rates[psb.nuclei.index((26, 56))][b]
    print(f'  Fe-56 CMB rate at 1.4e10: GDRA {r_g:.2f} vs PSB {r_p:.2f} /Mpc '
          f'(ratio {r_g / r_p:.2f})')
    assert 0.5 < r_g / r_p < 1.5


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
