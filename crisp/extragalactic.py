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

``ebl_scaling`` (opt-in, default off) approximates that chain instead of
building one full InteractionCore per segment. The interaction rate (cross
section convolved against the photon field) scales far more smoothly with
redshift than the raw EBL field does, since the convolution's narrow
support averages over most of the field's fine structure.
EBLSegmentRateScaling calibrates a boost-resolved correction between two
redshifts from a small probe-species set and validates it against held-out
species. ExtragalacticPropagation builds a handful of EBL-only reference
cores instead of one per segment, and assembles each segment's tensor as
CMB (exact identity) plus EBL (nearest reference, rescaled) plus decays
(a single redshift-independent core), relying on InteractionCore's tensor
being additive in the target photon field. See ``validate_ebl_scaling``
for a checkable error estimate on a given species list and EBL model.

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


class EBLSegmentRateScaling:
    """Calibrates and validates a boost-resolved rate-scaling correction
    between two redshifts for an EBL model.

    The interaction rate (cross section convolved against the photon
    field) evolves far more smoothly with redshift than the raw EBL field
    does. This class computes correction(boost) = rate(boost, z2) /
    rate(boost, z1) from a small set of cheap probe species, then applies
    that correction to approximate a rate at z2 from an already-known
    rate at z1, without a new `InteractionCore` build. Accuracy is
    checked against held-out species via `validate`, not assumed.

    Works with any EBL model exposing the same interface as
    `ebl_gilmore_at_z` (z -> callable(eps_GeV) -> density [GeV^-1 cm^-3]).

    Parameters
    ----------
    ebl_at_z : callable
        z -> callable(eps_GeV) -> photon density [GeV^-1 cm^-3].
    boosts : array_like
        Boost grid the correction is calibrated and evaluated on.
    probe_species : list of (Z, A), optional
        Species used to calibrate the correction. Default a spread
        across the mass range: He-4 and Fe-56.
    xsec_model : Cross_Section_Model, optional
        Source of probe cross sections. Default `PSB_model()`.
    erange : tuple of float, optional
        Photon energy window (MeV) the probe cross sections are sampled
        over. Default (0.2, 140.0).
    rate_floor : float, optional
        Boosts where either rate is below this fraction of that
        segment's peak rate are excluded. Default 1e-6.

    Attributes
    ----------
    factors_ : dict
        {(z1, z2): ndarray of shape (len(boosts),)}, cached corrections.
    coverage_ : dict
        {(z1, z2): fraction of boosts with a usable correction}. Below
        1.0 flags a redshift range where the EBL model's own tabulation
        runs out, not an approximation failure.
    """

    def __init__(self, ebl_at_z, boosts, probe_species=None, xsec_model=None,
                erange=(0.2, 140.0), rate_floor=1e-6):
        from .photonuclear_cross_sections import PSB_model

        self.ebl_at_z = ebl_at_z
        self.boosts = np.asarray(boosts, dtype=float)
        self.probe_species = probe_species or [(2, 4), (26, 56)]
        self.xsec_model = xsec_model or PSB_model()
        self.erange = erange
        self.rate_floor = rate_floor
        self.factors_ = {}
        self.coverage_ = {}
        self._rate_cache = {}

    def _rate(self, sp, z):
        key = (sp, float(z))
        if key not in self._rate_cache:
            from .interaction_rates import exact_rates_for_sigma
            eps_MeV = np.linspace(*self.erange, 200)
            sigma = self.xsec_model.total_cross_section(eps_MeV, *sp).reshape(1, -1)
            eps_GeV = eps_MeV * 1e-3
            self._rate_cache[key] = exact_rates_for_sigma(
                self.boosts, self.ebl_at_z(z), eps_GeV, sigma)[0]
        return self._rate_cache[key]

    def factor(self, z1, z2):
        """Boost-resolved correction rate(z2) / rate(z1), cached.

        Parameters
        ----------
        z1, z2 : float
            Reference and target redshift.

        Returns
        -------
        ndarray, shape (len(boosts),)
            NaN at boosts with no usable probe-species coverage; check
            `coverage_[(z1, z2)]` before trusting a segment fully.
        """
        key = (float(z1), float(z2))
        if key in self.factors_:
            return self.factors_[key]

        available = [sp for sp in self.probe_species if sp in self.xsec_model.nuclei]
        if not available:
            raise ValueError(
                f'none of probe_species {self.probe_species} are tracked by '
                f'xsec_model (has {len(self.xsec_model.nuclei)} nuclei)')

        log_ratios = []
        for sp in available:
            r1, r2 = self._rate(sp, z1), self._rate(sp, z2)
            thresh = self.rate_floor * max(r1.max(), r2.max())
            valid = (r1 > thresh) & (r2 > thresh)
            lr = np.full_like(self.boosts, np.nan)
            lr[valid] = np.log(r2[valid] / r1[valid])
            log_ratios.append(lr)
        stacked = np.vstack(log_ratios)
        usable = np.any(np.isfinite(stacked), axis=0)
        factor = np.full(len(self.boosts), np.nan)
        if usable.any():
            factor[usable] = np.exp(np.nanmean(stacked[:, usable], axis=0))
        self.factors_[key] = factor
        self.coverage_[key] = float(usable.mean())
        return factor

    def apply(self, rate_at_z1, z1, z2):
        """Approximate the rate at z2 from an already-known rate at z1.

        Parameters
        ----------
        rate_at_z1 : ndarray, shape (len(boosts),)
        z1, z2 : float

        Returns
        -------
        ndarray
            Approximate rate at z2, same shape as `rate_at_z1`.
        """
        return rate_at_z1 * self.factor(z1, z2)

    def validate(self, z_nodes, held_out_species, xsec_model=None, erange=None):
        """Checks calibrated corrections against directly computed rates
        for species not used in calibration, over consecutive pairs of
        `z_nodes`.

        Parameters
        ----------
        z_nodes : array_like
            Redshift nodes; consecutive pairs are checked.
        held_out_species : list of (Z, A)
            Species to validate against; should not overlap
            `probe_species` for a meaningful test.
        xsec_model, erange : optional
            Override the calibration's own xsec_model/erange (e.g. to
            cross-check against a different cross-section model).

        Returns
        -------
        dict
            {(Z, A): {(z1, z2): relative_error_array}} plus a 'summary'
            key with overall max/mean/median/p95 relative error.
        """
        z_nodes = np.asarray(z_nodes, dtype=float)
        xm = xsec_model or self.xsec_model
        er = erange or self.erange
        eps_MeV = np.linspace(*er, 200)
        eps_GeV = eps_MeV * 1e-3

        from .interaction_rates import exact_rates_for_sigma

        results = {}
        all_errors = []
        for sp in held_out_species:
            if sp not in xm.nuclei:
                continue
            sigma = xm.total_cross_section(eps_MeV, *sp).reshape(1, -1)
            true_rates = {
                z: exact_rates_for_sigma(self.boosts, self.ebl_at_z(z), eps_GeV, sigma)[0]
                for z in z_nodes
            }
            per_segment = {}
            for z1, z2 in zip(z_nodes[:-1], z_nodes[1:]):
                r1_true, r2_true = true_rates[z1], true_rates[z2]
                r2_approx = self.apply(r1_true, z1, z2)
                thresh = self.rate_floor * max(r1_true.max(), r2_true.max())
                valid = (r1_true > thresh) & (r2_true > thresh) & np.isfinite(r2_approx)
                rel_err = np.full_like(self.boosts, np.nan)
                rel_err[valid] = np.abs(r2_approx[valid] - r2_true[valid]) / r2_true[valid]
                per_segment[(z1, z2)] = rel_err
                all_errors.append(rel_err[valid])
            results[sp] = per_segment

        flat = np.concatenate(all_errors) if all_errors else np.array([])
        results['summary'] = {
            'n_species': len([s for s in held_out_species if s in xm.nuclei]),
            'n_points': flat.size,
            'max_relative_error': float(flat.max()) if flat.size else float('nan'),
            'mean_relative_error': float(flat.mean()) if flat.size else float('nan'),
            'median_relative_error': float(np.median(flat)) if flat.size else float('nan'),
            'p95_relative_error': float(np.percentile(flat, 95)) if flat.size else float('nan'),
        }
        return results

    def report(self, validation_result):
        """Human-readable summary of a `validate()` result."""
        s = validation_result['summary']
        lines = []
        low = [(z1, z2, c) for (z1, z2), c in self.coverage_.items() if c < 0.99]
        if low:
            lines.append('Coverage gaps (EBL model tabulation, not '
                         'approximation failure):')
            for z1, z2, c in low:
                lines.append(f"  z={z1:.2f}->{z2:.2f}: only {c:.0%} of "
                            f"boosts have a calibrated correction")
        lines += [
            f"Validated against {s['n_species']} held-out species, "
            f"{s['n_points']} (species x segment x boost) points.",
            f"  median relative error: {s['median_relative_error']:.1%}",
            f"  mean relative error:   {s['mean_relative_error']:.1%}",
            f"  95th pct rel. error:   {s['p95_relative_error']:.1%}",
            f"  worst-case rel. error: {s['max_relative_error']:.1%}",
        ]
        worst_segment, worst_val = None, -1.0
        for sp, per_seg in validation_result.items():
            if sp == 'summary':
                continue
            for seg, arr in per_seg.items():
                valid = arr[np.isfinite(arr)]
                if valid.size and valid.max() > worst_val:
                    worst_val = valid.max()
                    worst_segment = (sp, seg)
        if worst_segment:
            sp, (z1, z2) = worst_segment
            lines.append(f"  worst single point: species {sp}, segment "
                        f"z={z1:.2f}->{z2:.2f}, {worst_val:.1%} error")
        return '\n'.join(lines)


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
    ebl_scaling: None (default): today's exact per-segment quasi-
                 homogeneous chain, one full InteractionCore per segment.
                 'auto': fast, opt-in approximation. Builds a handful of
                 EBL-only reference cores (ln(1+z)-spaced, z capped at
                 min(z_src, 3.0), 5 references by default) instead of one
                 per segment, and rescales the nearest reference with a
                 calibrated, boost-resolved correction. Or an explicit
                 ascending array of reference redshifts to use instead of
                 the automatic grid. Has a known, checkable error budget;
                 see validate_ebl_scaling(). Only used when ebl_model is
                 also given.
    ebl_scaling_probe_species: species used to calibrate ebl_scaling's
                 correction. Default a spread across the mass range
                 (He-4, Fe-56).
    """

    def __init__(self, xsec_model=None, ebl_model=None, boosts=None, eps=None,
                 cosmology=None, n_seg=None, pair=False, verbose=True,
                 ebl_scaling=None, ebl_scaling_probe_species=None,
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
        self.ebl_scaling = ebl_scaling
        self.ebl_scaling_probe_species = ebl_scaling_probe_species
        self._core_kwargs = dict(core_kwargs)
        self._cores = {}                       # z_mid -> InteractionCore
        self._ebl_ref_cores = {}                # z_ref -> InteractionCore
        self._fast_cmb_core = None              # decay-free CMB-only core
        self._decay_core = None                 # decay-only core
        self._ebl_scaler = None                 # EBLSegmentRateScaling

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

    def _reduced_tensor(self, core, mr, tr, gamma_eval, boost_scale=None):
        """Restricted transition tensor at the boosts gamma_eval, with the
        absorbed column as the row deficit (same reduction as
        core.species_evolution_boost_range; positive-inflow convention).

        boost_scale, if given, is an array of shape (len(gamma_eval),)
        multiplying the tensor per boost bin before the diagonal is
        recomputed, so per-bin number conservation holds regardless of
        the scale's accuracy (used by the ebl_scaling fast path)."""
        # clip to the tabulated range: the identity path evaluates at the
        # blueshifted boost (1+z) gamma, which can overshoot the grid top by
        # ~ln(1+z); 'previous'-step interpolation continues the top bin flat
        gamma_eval = np.clip(gamma_eval, core.boosts[0], core.boosts[-1])
        rt = np.array(core.interpolator(gamma_eval))
        rt = rt[np.ix_(mr, mr, range(rt.shape[-1]))]
        if boost_scale is not None:
            rt = rt * np.asarray(boost_scale, dtype=float)[None, None, :]
        n_m = len(mr)
        for b in range(rt.shape[-1]):
            np.fill_diagonal(rt[:, :, b], 0.0)
        rt -= np.stack([np.diag(row) for row in rt.sum(axis=1).T], axis=2)
        idx = [mr.index(t) for t in tr]
        red = rt[np.ix_(idx, idx, range(rt.shape[-1]))]
        absorbed_in = -red.sum(axis=1)          # (n_tr, n_b), >= 0
        return red, absorbed_in

    # ------------------------------------------------------------------ #
    # ebl_scaling fast path: three field-additive builders, summed instead
    # of one full combined core per segment.
    #
    # decays=<user's own value>, nuclear_decay_On=False on the CMB-only and
    # EBL-ref cores below (not decays=None): InteractionCore's species
    # topology depends on whether a decay table is present at all, not
    # just on nuclear_decay_On (_resolve_unstable_products falls back to a
    # same-mass tracked stand-in for any untracked remnant absent from the
    # decay table, which is wrong for particle-unstable remnants like B-9
    # that a real table resolves correctly). Forcing decays=None here would
    # silently change which tensor entries are nonzero relative to the
    # user's own decays= setting, verified directly on a CRPropa network:
    # a genuine C-10 -> B-9 channel gets misrouted onto Be-9 (same mass,
    # untracked otherwise) when decays=None, and correctly resolved when
    # the real table is present, even with nuclear_decay_On=False. Keeping
    # decays consistent (only toggling nuclear_decay_On) keeps this core's
    # topology identical to the decay-only core's and to the exact path's,
    # so the three tensors stay additive regardless of the network.
    def _fast_cmb_core_at(self):
        """CMB-only core, no decay transitions added (they live on
        _decay_only_core, added once)."""
        if self._fast_cmb_core is None:
            kwargs = dict(self._core_kwargs)
            kwargs['nuclear_decay_On'] = False
            self._fast_cmb_core = InteractionCore(
                xsec_model=self.xsec_model, target_photons=cmb_density_at_z(0.0),
                boosts=self.boosts, eps=self.eps, **kwargs)
        return self._fast_cmb_core

    def _ebl_ref_core_at(self, z_ref):
        """EBL-only reference core at z_ref, no decay transitions added."""
        key = round(float(z_ref), 6)
        if key not in self._ebl_ref_cores:
            kwargs = dict(self._core_kwargs)
            kwargs['nuclear_decay_On'] = False
            self._ebl_ref_cores[key] = InteractionCore(
                xsec_model=self.xsec_model, target_photons=self.ebl_model(z_ref),
                boosts=self.boosts, eps=self.eps, **kwargs)
        return self._ebl_ref_cores[key]

    def _decay_only_core(self):
        """Single, redshift-independent core carrying only decay
        transitions: a zero photon field makes every photonuclear/
        photomeson rate exactly zero, verified directly (tensor(field,
        nuclear_decay_On=True) == tensor(field, nuclear_decay_On=False)
        + tensor(zero field, nuclear_decay_On=True), decays= held fixed,
        to double precision)."""
        if self._decay_core is None:
            zero_field = lambda e: np.zeros_like(np.atleast_1d(np.asarray(e, dtype=float)))
            self._decay_core = InteractionCore(
                xsec_model=self.xsec_model, target_photons=zero_field,
                boosts=self.boosts, eps=self.eps, **self._core_kwargs)
        return self._decay_core

    def _resolve_ebl_refs(self, z_src):
        """Reference redshift grid for ebl_scaling='auto', or the
        user-given explicit array."""
        if isinstance(self.ebl_scaling, str) and self.ebl_scaling == 'auto':
            z_max = min(float(z_src), 3.0)
            return np.expm1(np.linspace(0.0, np.log1p(z_max), 5))
        return np.sort(np.asarray(self.ebl_scaling, dtype=float))

    def _ebl_scaler_at(self):
        """Lazily-built EBLSegmentRateScaling for this instance's
        ebl_model, boosts, and probe species."""
        if self._ebl_scaler is None:
            self._ebl_scaler = EBLSegmentRateScaling(
                self.ebl_model, self.boosts,
                probe_species=self.ebl_scaling_probe_species,
                xsec_model=self.xsec_model)
        return self._ebl_scaler

    def validate_ebl_scaling(self, z_src=None, species=None, xsec_models=None,
                             tolerance=None):
        """Checks ebl_scaling's approximation for THIS instance's
        ebl_model, boosts, and reference grid, against held-out species.

        Parameters
        ----------
        z_src : float, optional
            Source redshift the reference grid is resolved for (only
            matters for ebl_scaling='auto'). Default 3.0.
        species : list of (Z, A), optional
            Held-out species to validate against. Default a spread not
            overlapping the default probe set (N-14, Si-28).
        xsec_models : list of Cross_Section_Model, optional
            Cross-section models to validate against, in addition to
            this instance's own xsec_model. Default none.
        tolerance : float, optional
            If given, also returns 'passed': worst-case relative error
            <= tolerance.

        Returns
        -------
        dict
            One EBLSegmentRateScaling.validate() result per cross-section
            model tried, plus a combined 'report' string.
        """
        if self.ebl_model is None or self.ebl_scaling is None:
            raise ValueError('validate_ebl_scaling() only applies when '
                             'ebl_model and ebl_scaling are both set')
        z_nodes = self._resolve_ebl_refs(z_src if z_src is not None else 3.0)
        species = species or [(7, 14), (14, 28)]
        models = [self.xsec_model] + list(xsec_models or [])

        scaler = self._ebl_scaler_at()
        out = {}
        reports = []
        for xm in models:
            result = scaler.validate(z_nodes, species, xsec_model=xm)
            if tolerance is not None:
                result['summary']['passed'] = \
                    result['summary']['max_relative_error'] <= tolerance
            out[getattr(xm, '__class__', type(xm)).__name__] = result
            reports.append(scaler.report(result))
        out['report'] = '\n\n'.join(reports)
        return out

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
        fast = self.ebl_model is not None and self.ebl_scaling is not None

        first_sp = next(iter(injection))
        if self.ebl_model is None:
            core0 = self._core_at(0.0)
        elif fast:
            core0 = self._fast_cmb_core_at()
        else:
            core0 = self._core_at(0.5 * float(z_src))
        _, mr, tr, _ = core0.get_distribution_parameters(
            mass_lims=mass_lims, injection_type=('only species', first_sp),
            absorption_type=('only mass', [1]))
        species = [core0.species[t] for t in tr]

        n_seg = n_seg or self.n_seg \
            or max(1, int(np.ceil(2.0 * np.log1p(float(z_src)) / 0.15)))
        z_edges = np.expm1(np.linspace(np.log1p(float(z_src)), 0.0,
                                       n_seg + 1))

        ebl_refs = self._resolve_ebl_refs(z_src) if fast else None
        ebl_scaler = self._ebl_scaler_at() if fast else None
        if fast:
            # cheap insurance: the three field-additive builders must agree
            # on species/index layout before their reduced tensors are summed
            decay_core = self._decay_only_core()
            ref_core = self._ebl_ref_core_at(float(ebl_refs[0]))
            if not (core0.species == decay_core.species == ref_core.species):
                raise RuntimeError(
                    'ebl_scaling fast path: CMB-only, decay-only, and EBL '
                    'reference cores disagree on species layout')

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

            if self.ebl_model is None:
                # exact CMB identity: z=0 tensor at the blueshifted boost,
                # rescaled by (1+z)^3
                core = self._core_at(z_mid)
                red, absin = self._reduced_tensor(
                    core, mr, tr, (1 + z_mid) * self.boosts)
                scale = (1 + z_mid) ** 3
            elif fast:
                # CMB (exact identity) + EBL (nearest reference, rescaled)
                # + decays (redshift-independent), summed before expm:
                # generators of independent simultaneous processes add,
                # propagators do not.
                red_cmb, absin_cmb = self._reduced_tensor(
                    self._fast_cmb_core_at(), mr, tr, (1 + z_mid) * self.boosts)
                z_ref = float(ebl_refs[np.argmin(np.abs(ebl_refs - z_mid))])
                correction = ebl_scaler.factor(z_ref, z_mid)
                red_ebl, absin_ebl = self._reduced_tensor(
                    self._ebl_ref_core_at(z_ref), mr, tr, self.boosts,
                    boost_scale=correction)
                red_dcy, absin_dcy = self._reduced_tensor(
                    self._decay_only_core(), mr, tr, self.boosts)
                cmb_scale = (1 + z_mid) ** 3
                red = red_cmb * cmb_scale + red_ebl + red_dcy
                absin = absin_cmb * cmb_scale + absin_ebl + absin_dcy
                scale = 1.0
            else:
                core = self._core_at(z_mid)
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
                    / max(n_init, 1e-300)),
                'ebl_scaling': 'auto' if fast and self.ebl_scaling == 'auto'
                              else ('explicit' if fast else None),
                'ebl_refs': list(ebl_refs) if fast else None}
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
