"""Extragalactic propagation from cosmological redshifts: the coherent-
inhomogeneity (CI) treatment of the methods paper, Sect. 3.

The CMB is an exact CI: its blackbody blueshift gives the identity

    Lambda(gamma, z) = (1 + z)^3 Lambda_0((1 + z) gamma),

so the z = 0 interaction tensor of a single InteractionCore is reused
UNCHANGED (evaluated at the blueshifted boost and rescaled) while the
adiabatic redshift of the particles is a rigid, species- and
boost-independent shift in ln(gamma) (Sect. 3.1: change of variable, no
new transport). The corresponding effective thickness
delta(z) = int (1+z)^2 d_H H0/H(z') dz' (Eq. 27) is exposed as
``thickness`` and pinned against continuous_losses.Lprime_trapz.

The EBL does NOT factorize this way (its (energy, z) evolution is tabulated,
not a blueshifted blackbody), so with an EBL model the propagation runs as a
quasi-homogeneous chain (Sect. 3.2): redshift segments short enough that the
CI description holds within each, one InteractionCore per segment built on
the proper CMB(z_j) + EBL(z_j) field at the segment midpoint, with the
adiabatic relabel applied between segments.

Bethe-Heitler pair production (pair=True) adds the Z^2/A-scaled continuous
drift of continuous_losses, a dispersive term (species-dependent), applied
per species row between segments.
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import cumulative_trapezoid

from .core import InteractionCore, shift_log_boost
from .background_photon_models import cmb_photon_density
from . import continuous_losses as _cl


def cmb_density_at_z(z):
    """Proper CMB spectral density at redshift z: blackbody at T0 (1+z).

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    callable
        n(e_GeV) in GeV^-1 cm^-3 (the house convention).
    """
    from .background_photon_models import To
    T = To * (1.0 + z)
    return lambda e: cmb_photon_density(T, np.asarray(e) * 1e9) / 1e6 * 1e9


def _ebl_spline_at_z(spline, z, e_lo=1.3e-12, e_hi=1.2e-8):
    """Proper EBL density at redshift z from an (energy [eV], z) spline in
    m^-3 eV^-1 storing the COMOVING density (all three bundled models do as
    verified: their n(e, z)/n(e, 0) ratios are O(1), no (1+z)^3 growth).
    Returns a callable n(e_GeV) in GeV^-1 cm^-3, windowed to
    [e_lo, e_hi] GeV, the common support of the bundled models."""
    def n(e):
        e = np.asarray(e, dtype=float)
        ec = np.clip(e, e_lo, e_hi)
        val = np.clip(spline(ec * 1e9, z).flatten(), 0.0, None) * 1e3
        return np.where((e >= e_lo) & (e <= e_hi), val, 0.0) * (1 + z) ** 3
    return n


def ebl_gilmore_at_z(z):
    """Proper Gilmore-2012 EBL density at redshift z (see _ebl_spline_at_z).

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    callable
        n(e_GeV) in GeV^-1 cm^-3, windowed to the model's common support.
    """
    from .background_photon_models import eblg_interp
    return _ebl_spline_at_z(eblg_interp, z)


def ebl_saldana_at_z(z):
    """Proper Saldana-Lopez-2021 EBL density at redshift z (tabulated to
    z = 6; see _ebl_spline_at_z).

    Parameters
    ----------
    z : float
        Redshift (<= 6, the table's tabulated range).

    Returns
    -------
    callable
        n(e_GeV) in GeV^-1 cm^-3, windowed to the model's common support.
    """
    from .background_photon_models import ebls_interp
    return _ebl_spline_at_z(ebls_interp, z)


def ebl_andrews_at_z(z):
    """Proper Andrews-2018 EBL density at redshift z (tabulated to z = 6;
    see _ebl_spline_at_z).

    Parameters
    ----------
    z : float
        Redshift (<= 6, the table's tabulated range).

    Returns
    -------
    callable
        n(e_GeV) in GeV^-1 cm^-3, windowed to the model's common support.
    """
    from .background_photon_models import ebla_interp
    return _ebl_spline_at_z(ebla_interp, z)


class ExtragalacticPropagation:
    """Propagates a nuclear injection from a source at redshift z_src to
    Earth with the CI machinery (module docstring). Heavy-cascade
    occupations only (light yields / neutrinos: use the arrival state with
    the existing core folds).

    Arguments:
    ----------
    xsec_model : photodisintegration model for the cascade (e.g. PSB_model()).
    ebl_model  : None (CMB only, single-core exact-identity path) or a
                 callable z -> (callable e_GeV -> GeV^-1 cm^-3 proper
                 density), e.g. ebl_gilmore_at_z.
    boosts     : log-uniform boost grid (default np.logspace(6, 11.5, 120)).
    eps        : photon grid for the cores [GeV].
    cosmology  : astropy cosmology (default: continuous_losses' WMAP9).
    n_seg      : number of redshift segments for propagate(); default sized
                 so each segment's argument drift 2 dln(1+z) <= 0.15.
    pair       : include Bethe-Heitler pair losses as a per-species
                 continuous drift between segments (default False).
    core_kwargs: forwarded to every InteractionCore (decays=, photomeson=,
                 ...). The species set is shared across segments.
    """

    def __init__(self, xsec_model=None, ebl_model=None, boosts=None, eps=None,
                 cosmology=None, n_seg=None, pair=False, verbose=True,
                 **core_kwargs):
        self.xsec_model = xsec_model
        self.ebl_model = ebl_model
        self.cosmology = cosmology if cosmology is not None else _cl.cosmo
        self.boosts = np.logspace(6, 11.5, 120) if boosts is None \
            else np.asarray(boosts, dtype=float)
        dln = np.diff(np.log(self.boosts))
        if not np.allclose(dln, dln[0], rtol=1e-8):
            raise ValueError('boosts must be log-uniform (the adiabatic '
                             'relabel between segments is a rigid shift)')
        self._dln = float(dln[0])
        self.eps = eps
        self.n_seg = n_seg
        self.pair = pair
        self.verbose = verbose
        self._core_kwargs = dict(core_kwargs)
        self._cores = {}                       # z_mid -> InteractionCore

        if verbose and ebl_model is None:
            print('CMB-only CI: exact single-core identity path '
                  '(CMB dominates the photonuclear rates for comoving '
                  'boosts >~ 5e9; add ebl_model= below that).')

    # ------------------------------------------------------------------ #
    def thickness(self, z):
        """Effective thickness delta(z) = int_0^z (1+z')^2 d_H H0/H dz'
        [Mpc] (methods paper Eq. 27) the CI variable replacing proper
        distance for z=0-tabulated CMB rates."""
        zgrid = np.linspace(0.0, float(z), 2049)
        d_H = self.cosmology.hubble_distance.value
        integ = (1 + zgrid) ** 2 * d_H \
            * (self.cosmology.H0 / self.cosmology.H(zgrid)).value
        return float(np.trapezoid(integ, zgrid))

    def _proper_length(self, z_lo, z_hi):
        """Proper path length [Mpc] traversed between the two redshifts."""
        return float((self.cosmology.lookback_distance(z_hi)
                      - self.cosmology.lookback_distance(z_lo)).value)

    # ------------------------------------------------------------------ #
    def _core_at(self, z_mid):
        """Segment core on the proper CMB(z)+EBL(z) field (cached). For the
        CMB-only path, one z=0 core serves all segments via the identity."""
        key = 0.0 if self.ebl_model is None else round(float(z_mid), 6)
        if key not in self._cores:
            if self.ebl_model is None:
                fields = cmb_density_at_z(0.0)
            else:
                fields = (cmb_density_at_z(z_mid), self.ebl_model(z_mid))
            self._cores[key] = InteractionCore(
                xsec_model=self.xsec_model, target_photons=fields,
                boosts=self.boosts, eps=self.eps, **self._core_kwargs)
        return self._cores[key]

    def _reduced_tensor(self, core, mr, tr, gamma_eval):
        """Restricted transition tensor at the boosts gamma_eval, with the
        absorbed column as the row deficit (same reduction as
        core.species_evolution_boost_range; positive-inflow convention)."""
        # clip to the tabulated range: the identity path evaluates at the
        # blueshifted boost (1+z) gamma, which can overshoot the grid top by
        # ~ln(1+z); 'previous'-step interpolation continues the top bin flat
        gamma_eval = np.clip(gamma_eval, core.boosts[0], core.boosts[-1])
        rt = np.array(core.interpolator(gamma_eval))
        rt = rt[np.ix_(mr, mr, range(rt.shape[-1]))]
        n_m = len(mr)
        for b in range(rt.shape[-1]):
            np.fill_diagonal(rt[:, :, b], 0.0)
        rt -= np.stack([np.diag(row) for row in rt.sum(axis=1).T], axis=2)
        idx = [mr.index(t) for t in tr]
        red = rt[np.ix_(idx, idx, range(rt.shape[-1]))]
        absorbed_in = -red.sum(axis=1)          # (n_tr, n_b), >= 0
        return red, absorbed_in

    # ------------------------------------------------------------------ #
    def propagate(self, z_src, injection, mass_lims=(56, 0), n_seg=None,
                  history=False):
        """Propagate `injection`, a dict {(Z, A): weights_on_boosts} of
        SOURCE-frame counts per boost bin, from z_src to Earth.

        Returns a dict: 'boosts' (the z=0 grid), 'species' (tracked list),
        'occupations' (n_b, n_sp), 'absorbed' (n_b,), 'diagnostics'.
        With history=True, also 'history': the per-segment chain 
        ('z_edges' (n_seg+1, source -> 0) and lists 'states' / 'absorbed'
        of length n_seg+1 holding the state at every edge (index 0 = the
        injection at the source)) e.g. for transition-density (pdf)
        reconstructions along the path.
        """
        first_sp = next(iter(injection))
        core0 = self._core_at(0.0 if self.ebl_model is None
                              else 0.5 * float(z_src))
        _, mr, tr, _ = core0.get_distribution_parameters(
            mass_lims=mass_lims, injection_type=('only species', first_sp),
            absorption_type=('only mass', [1]))
        species = [core0.species[t] for t in tr]

        n_seg = n_seg or self.n_seg \
            or max(1, int(np.ceil(2.0 * np.log1p(float(z_src)) / 0.15)))
        z_edges = np.expm1(np.linspace(np.log1p(float(z_src)), 0.0,
                                       n_seg + 1))

        n_b = len(self.boosts)
        state = np.zeros((n_b, len(species)))
        for sp, w in injection.items():
            state[:, species.index(tuple(sp))] += np.asarray(w, dtype=float)
        absorbed = np.zeros(n_b)
        n_init = state.sum()

        drifts = []
        hist_states, hist_abs = [state.copy()], [absorbed.copy()]
        for z_hi, z_lo in zip(z_edges[:-1], z_edges[1:]):
            z_mid = np.sqrt((1 + z_hi) * (1 + z_lo)) - 1.0
            L_seg = self._proper_length(z_lo, z_hi)
            core = self._core_at(z_mid)

            if self.ebl_model is None:
                # exact CMB identity: z=0 tensor at the blueshifted boost,
                # rescaled by (1+z)^3
                red, absin = self._reduced_tensor(
                    core, mr, tr, (1 + z_mid) * self.boosts)
                scale = (1 + z_mid) ** 3
            else:
                red, absin = self._reduced_tensor(core, mr, tr, self.boosts)
                scale = 1.0

            # segment evolution with the absorbed column augmented (rows of
            # M_aug sum to zero -> per-bin number conservation is exact;
            # this cascade tensor is boost-preserving)
            n_tr = len(species)
            for b in range(n_b):
                M_aug = np.zeros((n_tr + 1, n_tr + 1))
                M_aug[:n_tr, :n_tr] = red[:, :, b]
                M_aug[:n_tr, n_tr] = absin[:, b]
                v = np.append(state[b], absorbed[b]) @ expm(
                    M_aug * scale * L_seg)
                state[b], absorbed[b] = v[:n_tr], v[n_tr]

            # adiabatic relabel: gamma -> gamma (1+z_lo)/(1+z_hi), identical
            # for every species (coherent); pair=True adds the Z^2/A-scaled
            # Bethe-Heitler drift (dispersive, per species row)
            shift = np.log((1 + z_hi) / (1 + z_lo)) / self._dln
            if self.pair:
                bh = np.asarray(_cl.Bpp_Blumenthal(
                    1, 1, self.boosts, z_mid), dtype=float)   # Mpc^-1
                extra = bh * L_seg / self._dln                # bins
                for si, (Z, A) in enumerate(species):
                    state[:, si:si + 1] = shift_log_boost(
                        self.boosts, state[:, si:si + 1], shift + Z**2 / A * extra)
            else:
                state = shift_log_boost(self.boosts, state, shift)
            # the absorbed pool is free nucleons (A = 1): under pair=True it
            # keeps drifting with Z^2/A = 1 until Earth
            abs_shift = shift + extra if self.pair else shift
            absorbed = shift_log_boost(
                self.boosts, absorbed[:, None], abs_shift)[:, 0]
            drifts.append(2.0 * np.log((1 + z_hi) / (1 + z_lo)))
            hist_states.append(state.copy())
            hist_abs.append(absorbed.copy())

        diag = {'n_seg': n_seg, 'drift_per_segment': drifts,
                'thickness': self.thickness(z_src),
                'number_leakage': float(
                    1.0 - (state.sum() + absorbed.sum())
                    / max(n_init, 1e-300))}
        if self.verbose:
            print(f'CI propagation: z = {z_src} in {n_seg} segments, '
                  f'max argument drift/segment {max(drifts):.3f}; '
                  f'delta(z) = {diag["thickness"]:.1f} Mpc')
        out = {'boosts': self.boosts, 'species': species,
               'occupations': state, 'absorbed': absorbed,
               'diagnostics': diag}
        if history:
            out['history'] = {'z_edges': z_edges, 'states': hist_states,
                              'absorbed': hist_abs}
        return out
