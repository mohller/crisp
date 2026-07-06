"""
Test suite for the crisp package.

Run with:
    pip install crisp-py          # or: pip install -e .
    pip install pytest
    pytest tests/test_crisp.py -v
"""

import numpy as np
import pytest
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_allclose(a, b, rtol=1e-4, msg=""):
    """Thin wrapper so failures print the actual values."""
    np.testing.assert_allclose(a, b, rtol=rtol, err_msg=msg)


# ---------------------------------------------------------------------------
# 1. Import tests – these must always pass after install
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_top_level(self):
        import crisp
        assert hasattr(crisp, "__version__")

    def test_import_background_photon_models(self):
        import crisp.background_photon_models

    def test_import_photonuclear_cross_sections(self):
        import crisp.photonuclear_cross_sections

    def test_import_continuous_losses(self):
        import crisp.continuous_losses

    def test_import_interaction_rates(self):
        import crisp.interaction_rates

    def test_import_source_models(self):
        import crisp.source_models

    def test_import_core(self):
        import crisp.core


# ---------------------------------------------------------------------------
# 2. Data files – verify they are bundled and loadable
# ---------------------------------------------------------------------------

class TestDataFiles:
    """Check that the data/ directory is accessible regardless of how the
    package was installed (pip install vs git clone + editable install)."""

    def test_data_dir_exists(self):
        import os, crisp
        data_dir = os.path.join(os.path.dirname(crisp.__file__), "data")
        assert os.path.isdir(data_dir), (
            f"data/ not found at {data_dir} — did you move it inside crisp/?"
        )

    def test_ebl_pickles_present(self):
        import os, crisp
        data_dir = os.path.join(os.path.dirname(crisp.__file__), "data")
        for fname in [
            "Gilmore12_splinterp.pkl",
            "SaldanaLopez21_splinterp.pkl",
            "Andrews18_splinterp.pkl",
            "universal-spline.pkl",
        ]:
            assert os.path.isfile(os.path.join(data_dir, fname)), (
                f"Missing data file: {fname}"
            )

    def test_csv_files_present(self):
        import os, crisp
        data_dir = os.path.join(os.path.dirname(crisp.__file__), "data")
        for fname in ["PSB1976.csv", "Atomic_abundances.csv"]:
            assert os.path.isfile(os.path.join(data_dir, fname)), (
                f"Missing data file: {fname}"
            )


# ---------------------------------------------------------------------------
# 3. EBL / background photon models
# ---------------------------------------------------------------------------

class TestBackgroundPhotonModels:
    @pytest.fixture(autouse=True)
    def _import(self):
        import crisp.background_photon_models as bpm
        self.bpm = bpm

    def test_ebl_interpolators_loaded(self):
        """EBL objects should be RectBivariateSpline instances."""
        from scipy.interpolate import RectBivariateSpline
        assert isinstance(self.bpm.eblg_interp, RectBivariateSpline)
        assert isinstance(self.bpm.ebls_interp, RectBivariateSpline)
        assert isinstance(self.bpm.ebla_interp, RectBivariateSpline)

    def test_ebl_gilmore12_value(self):
        """Gilmore 2012 EBL density at 1 eV, z=0 (regression)."""
        val = self.bpm.eblg_interp(np.array([1.0]), np.array([0.0]))[0, 0]
        assert_allclose(val, 2886.9, rtol=1e-2,
                        msg="Gilmore12 EBL value changed unexpectedly")

    def test_ebl_saldana_lopez_value(self):
        """Saldana-Lopez 2021 EBL density at 1 eV, z=0 (regression)."""
        val = self.bpm.ebls_interp(np.array([1.0]), np.array([0.0]))[0, 0]
        assert_allclose(val, 2637.6, rtol=1e-2)

    def test_ebl_andrews18_value(self):
        """Andrews 2018 EBL density at 1 eV, z=0 (regression)."""
        val = self.bpm.ebla_interp(np.array([1.0]), np.array([0.0]))[0, 0]
        assert_allclose(val, 3393.5, rtol=1e-2)

    def test_ebl_positive(self):
        """EBL densities must be positive for any physical (E, z) point."""
        e_vals = np.array([0.01, 0.1, 1.0, 10.0])
        z_vals = np.array([0.0, 0.5, 1.0, 2.0])
        for model in (self.bpm.eblg_interp, self.bpm.ebls_interp, self.bpm.ebla_interp):
            for z in z_vals:
                vals = model(e_vals, np.array([z]))
                assert np.all(vals >= 0), "EBL interpolator returned negative values"

    def test_broken_power_law_callable(self):
        """target_photons_spectrum should return a callable."""
        f = self.bpm.target_photons_spectrum()
        assert callable(f)
        assert f(1e-5) >= 0

    def test_cmb_spectrum_shape(self):
        """Black-body spectral radiance should return correct shapes."""
        T = 2.725
        erange = np.logspace(-4, -2, 10)
        result = self.bpm.black_body_spectral_radiance(T, erange)
        assert result.shape == erange.shape
        assert np.all(result > 0)


# ---------------------------------------------------------------------------
# 4. Photonuclear cross sections
# ---------------------------------------------------------------------------

class TestPhotonuclearCrossSections:
    @pytest.fixture(autouse=True)
    def _import(self):
        import crisp.photonuclear_cross_sections as pcs
        self.pcs = pcs

    def test_psb_model_instantiates(self):
        psb = self.pcs.PSB_model()
        assert psb is not None

    def test_psb_carbon12_regression(self):
        """PSB cross section for C-12 at 100 MeV (regression value)."""
        psb = self.pcs.PSB_model()
        xs = psb.cross_section(np.array([100.0]), 6, 12)
        assert_allclose(xs[0], 0.79235, rtol=1e-3)

    def test_psb_fe56_regression(self):
        """PSB cross section for Fe-56 at 100 MeV (regression value)."""
        psb = self.pcs.PSB_model()
        xs = psb.cross_section(np.array([100.0]), 26, 56)
        assert_allclose(xs[0], 6.59402, rtol=1e-3)

    def test_psb_above_threshold_is_zero(self):
        """PSB model should return 0 above its tabulated energy range."""
        psb = self.pcs.PSB_model()
        xs = psb.cross_section(np.array([1000.0]), 6, 12)
        assert xs[0] == 0.0

    def test_psb_cross_section_nonnegative(self):
        psb = self.pcs.PSB_model()
        e = np.linspace(10, 300, 100)
        xs = psb.cross_section(e, 26, 56)
        assert np.all(xs >= 0)

    def test_gdr_atlas_instantiates(self):
        gdr = self.pcs.GDR_atlas()
        assert gdr is not None

    def test_photomeson_requires_pmm(self):
        """Photomeson needs an external AstroPhoMes pmm object — without one it
        should raise AttributeError on pmm.nonel_idcs, not an import error."""
        with pytest.raises((AttributeError, TypeError)):
            self.pcs.Photomeson(np.logspace(0, 4, 100), 1)


# ---------------------------------------------------------------------------
# 5. Continuous energy losses
# ---------------------------------------------------------------------------

class TestContinuousLosses:
    @pytest.fixture(autouse=True)
    def _import(self):
        import crisp.continuous_losses as cl
        self.cl = cl

    def test_bpp_blumenthal_proton_regression(self):
        """Bpp for a proton at g=1e10, z=0 (regression value in Mpc-1)."""
        bpp = self.cl.Bpp_Blumenthal(1, 1, np.array([1e10]), 0)
        assert_allclose(bpp[0], 7.002e-4, rtol=1e-2)

    def test_bpp_blumenthal_nuclear_scaling(self):
        """Bpp must scale as Z^2/A relative to a proton at the same boost."""
        g = np.array([1e10])
        bpp_p  = self.cl.Bpp_Blumenthal(1, 1, g, 0)[0]
        bpp_fe = self.cl.Bpp_Blumenthal(26, 56, g, 0)[0]
        expected_ratio = 26**2 / 56
        assert_allclose(bpp_fe / bpp_p, expected_ratio, rtol=1e-6,
                        msg="Z^2/A scaling broken for Fe-56")

    def test_bpp_blumenthal_positive(self):
        """Pair production losses must be strictly positive."""
        g = np.logspace(9, 12, 20)
        bpp = self.cl.Bpp_Blumenthal(1, 1, g, 0)
        assert np.all(bpp > 0)

    def test_bpp_crpropa_raises_without_env(self, monkeypatch):
        """Bpp_crpropa should raise FileNotFoundError if env var not set."""
        monkeypatch.delenv("CRISP_BPP_DATA", raising=False)
        with pytest.raises(FileNotFoundError, match="CRISP_BPP_DATA"):
            self.cl.Bpp_crpropa(1, 1, np.array([1e10]), 0)

    def test_dlngdz_tot_proton_sign(self):
        """Energy losses must decelerate the particle (negative dlng/dz)."""
        import numpy as np
        result = self.cl.dlngdz_tot_proton(0.1, np.log(1e10))
        assert result < 0, "dlngdz_tot_proton should be negative (energy loss)"

    def test_g_in_z_returns_arrays(self):
        """g_in_z should return two arrays of equal length."""
        z_out, g_out = self.cl.g_in_z(1e10, Z=1, A=1, z0=1.0)
        assert len(z_out) == len(g_out)
        assert len(z_out) > 1

    def test_g_in_z_boost_decreases(self):
        """Boost should decrease monotonically from source to observer."""
        z_out, g_out = self.cl.g_in_z(1e10, Z=1, A=1, z0=1.0)
        # g decreases as z decreases toward 0 (energy lost en route)
        assert g_out[-1] < g_out[0]


# ---------------------------------------------------------------------------
# 6. Interaction rates
# ---------------------------------------------------------------------------

class TestInteractionRates:
    @pytest.fixture(autouse=True)
    def _import(self):
        import crisp.interaction_rates as ir
        self.ir = ir

    def test_module_has_key_attributes(self):
        """Spot-check that key callables are present."""
        assert hasattr(self.ir, "interaction_rate") or any(
            callable(getattr(self.ir, name))
            for name in dir(self.ir)
            if not name.startswith("_")
        ), "interaction_rates module exposes no callable"


# ---------------------------------------------------------------------------
# 7. Channel-number parsing
# ---------------------------------------------------------------------------

class TestChannelNumberParsing:
    """Unit tests for get_particle_numbers, which decodes CRPropa channel IDs.

    Channel encoding:
        CN = nN*100000 + nP*10000 + nH2*1000 + nH3*100 + nHe3*10 + nHe4*1
    Return order: [nHe4, nHe3, nH3, nH2, nP, nN]
    """

    def test_single_proton(self):
        from crisp.core import get_particle_numbers
        assert get_particle_numbers(10000) == [0, 0, 0, 0, 1, 0]

    def test_single_neutron(self):
        from crisp.core import get_particle_numbers
        assert get_particle_numbers(100000) == [0, 0, 0, 0, 0, 1]

    def test_single_alpha(self):
        from crisp.core import get_particle_numbers
        assert get_particle_numbers(1) == [1, 0, 0, 0, 0, 0]

    def test_single_he3(self):
        from crisp.core import get_particle_numbers
        assert get_particle_numbers(10) == [0, 1, 0, 0, 0, 0]

    def test_single_tritium(self):
        from crisp.core import get_particle_numbers
        assert get_particle_numbers(100) == [0, 0, 1, 0, 0, 0]

    def test_single_deuterium(self):
        from crisp.core import get_particle_numbers
        assert get_particle_numbers(1000) == [0, 0, 0, 1, 0, 0]

    def test_two_protons_one_neutron(self):
        from crisp.core import get_particle_numbers
        assert get_particle_numbers(120000) == [0, 0, 0, 0, 2, 1]

    def test_alpha_plus_proton(self):
        from crisp.core import get_particle_numbers
        # nHe4=1, nP=1 → CN = 1 + 10000 = 10001
        assert get_particle_numbers(10001) == [1, 0, 0, 0, 1, 0]

    def test_zero_channel(self):
        from crisp.core import get_particle_numbers
        assert get_particle_numbers(0) == [0, 0, 0, 0, 0, 0]

    def test_all_nonnegative(self):
        """Particle counts are always non-negative for any valid channel number."""
        from crisp.core import get_particle_numbers
        for cn in [1, 10, 100, 1000, 10000, 100000, 10001, 110000, 11001]:
            assert all(n >= 0 for n in get_particle_numbers(cn))

    def test_nucleon_count_consistent(self):
        """sum(A_l * n_l) is the correct total nucleon count for several channels."""
        from crisp.core import get_particle_numbers
        A_light = np.array([4, 3, 3, 2, 1, 1])  # He4, He3, H3, H2, p, n
        cases = [
            (10000,  1),   # 1p
            (100000, 1),   # 1n
            (1,      4),   # 1 alpha
            (10001,  5),   # 1 alpha + 1p
            (110000, 2),   # 1p + 1n
            (20000,  2),   # 2p
        ]
        for cn, expected_A in cases:
            particles = np.array(get_particle_numbers(cn))
            assert A_light.dot(particles) == expected_A, (
                f"Channel {cn}: expected {expected_A} nucleons, "
                f"got {A_light.dot(particles)}"
            )


# ---------------------------------------------------------------------------
# 8. Cascade nucleon and charge conservation  (PSB CMB model)
# ---------------------------------------------------------------------------

class TestCascadeNucleonConservation:
    """Verify that the disintegration cascade conserves nucleon and charge numbers.

    Uses the self-contained PSB CMB model (no external data files required).

    Two complementary checks are performed at each boost/distance:

    Static (rate-matrix level):
        For each parent species i,
            sum_j A_j * M[i,j,b]  +  sum_l A_l * sum_j Y[l,i,j,b]  =  0
        where M is the interaction tensor and Y is the light-yield tensor.

    Dynamic (propagation level):
        Augment the state vector to [P_species | N_light]:
            d[P; N]/dL = [P; N] @ [[M, K]; [0, 0]]
        where K[i, l] = sum_j Y[l,i,j,b] is the total light-particle production
        rate from species i.  Then at every distance L:
            sum_i A_i * P_i(L)  +  sum_l A_l * N_l(L)  =  A_injection
    """

    # Light-particle properties (order matches light_prod_tensor axis 0)
    _A_LIGHT = np.array([4.0, 3.0, 3.0, 2.0, 1.0, 1.0])  # He4, He3, H3, H2, p, n
    _Z_LIGHT = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 0.0])

    # Representative distances [Mpc] to probe during propagation
    _L_VALUES = [0.0, 1.0, 10.0, 100.0]

    @pytest.fixture(scope="class")
    def psb_cascade(self):
        """Build the PSB CMB interaction core once per test class."""
        from crisp.core import InteractionCore_PSB_CMB
        return InteractionCore_PSB_CMB()

    @pytest.fixture(scope="class")
    def fe56_injection(self, psb_cascade):
        """Return injection vector for a pure Fe-56 beam."""
        species = psb_cascade.species
        if (26, 56) not in species:
            pytest.skip("Fe-56 not in PSB species list")
        idx = species.index((26, 56))
        alpha = np.zeros(len(species))
        alpha[idx] = 1.0
        return alpha

    @pytest.fixture(scope="class")
    def boost_idx(self, psb_cascade):
        """Boost index where interaction rates are significant (log10(Γ) ≈ 10)."""
        # boosts = logspace(6, 14, 201)  →  index 100 ≈ 10^10
        return 100

    # ------------------------------------------------------------------
    # 9.1  Static conservation check
    # ------------------------------------------------------------------

    def test_static_nucleon_balance(self, psb_cascade):
        """Nucleon-weighted row sums of the rate matrix must vanish.

        For each parent i and every boost b:
            sum_j A_j * M[i,j,b] + sum_l A_l * sum_j Y[l,i,j,b] = 0

        This is equivalent to verifying that no nucleons are created or
        destroyed by any single interaction channel.
        """
        species = psb_cascade.species
        A_sp = np.array([s[1] for s in species], dtype=float)

        tensor = psb_cascade.tensor          # (n, n, n_boosts)
        lyield = psb_cascade.light_prod_tensor  # (6, n, n, n_boosts)

        # Nucleon flux from heavy products: sum_j A_j * M[i,j,b]
        heavy_flow = np.einsum("j,ijb->ib", A_sp, tensor)
        # Nucleon flux from light secondaries: sum_l A_l * sum_j Y[l,i,j,b]
        light_flow = np.einsum("l,lijb->ib", self._A_LIGHT, lyield)

        imbalance = heavy_flow + light_flow

        # Only check species that actually participate in interactions
        active = np.any(np.abs(tensor) > 0, axis=(1, 2))
        scale = np.abs(tensor).max()

        np.testing.assert_allclose(
            imbalance[active], 0.0,
            atol=1e-8 * scale,
            err_msg="Nucleon conservation violated in rate matrix "
                    "(A-weighted row sum is non-zero)",
        )

    def test_static_charge_balance(self, psb_cascade):
        """Charge-weighted row sums of the rate matrix must vanish.

        For each parent i and every boost b:
            sum_j Z_j * M[i,j,b] + sum_l Z_l * sum_j Y[l,i,j,b] = 0
        """
        species = psb_cascade.species
        Z_sp = np.array([s[0] for s in species], dtype=float)

        tensor = psb_cascade.tensor
        lyield = psb_cascade.light_prod_tensor

        heavy_flow = np.einsum("j,ijb->ib", Z_sp, tensor)
        light_flow = np.einsum("l,lijb->ib", self._Z_LIGHT, lyield)

        imbalance = heavy_flow + light_flow

        active = np.any(np.abs(tensor) > 0, axis=(1, 2))
        scale = np.abs(tensor).max()

        np.testing.assert_allclose(
            imbalance[active], 0.0,
            atol=1e-8 * scale,
            err_msg="Charge conservation violated in rate matrix "
                    "(Z-weighted row sum is non-zero)",
        )

    # ------------------------------------------------------------------
    # 9.2  Probability conservation (sanity check for the tensor itself)
    # ------------------------------------------------------------------

    def test_probability_conservation(self, psb_cascade, fe56_injection, boost_idx):
        """Total probability must equal 1 at every distance."""
        alpha = fe56_injection
        M = psb_cascade.tensor[:, :, boost_idx].astype(float)

        for L in self._L_VALUES:
            P_L = alpha @ expm(M * L)
            np.testing.assert_allclose(
                P_L.sum(), 1.0, rtol=1e-6,
                err_msg=f"sum(P) ≠ 1 at L={L} Mpc  (probability not conserved)",
            )

    # ------------------------------------------------------------------
    # 9.3  Dynamic nucleon conservation
    # ------------------------------------------------------------------

    def _augmented_evolve(self, cascade, alpha, boost_idx, L):
        """Evolve [P_species | N_light] jointly and return state at distance L.

        The augmented generator is:
            augM = [[M, K], [0, 0]]

        where K[i, l] = sum_j light_prod_tensor[l, i, j, b] is the total
        rate of producing light particle l from species i.
        """
        n_sp = len(cascade.species)
        n_lt = len(self._A_LIGHT)

        M = cascade.tensor[:, :, boost_idx].astype(float)
        # K shape: (n_sp, n_lt) — production rate matrix
        K = cascade.light_prod_tensor[:, :, :, boost_idx].sum(axis=2).T

        augM = np.block([
            [M,                            K],
            [np.zeros((n_lt, n_sp)),       np.zeros((n_lt, n_lt))],
        ])

        state0 = np.concatenate([alpha, np.zeros(n_lt)])
        return state0 @ expm(augM * L)

    def test_dynamic_nucleon_conservation(self, psb_cascade, fe56_injection, boost_idx):
        """Injecting Fe-56, the nucleon count must stay at 56 at every distance.

        Nucleon count = sum_i A_i * P_i(L)  +  sum_l A_l * N_l(L)
        where N_l is the cumulative number of light particles of type l.
        """
        n_sp = len(psb_cascade.species)
        A_sp = np.array([s[1] for s in psb_cascade.species], dtype=float)
        A_inj = float(A_sp @ fe56_injection)

        for L in self._L_VALUES:
            state = self._augmented_evolve(psb_cascade, fe56_injection, boost_idx, L)
            P_L = state[:n_sp]
            N_L = state[n_sp:]

            nucleon_count = A_sp @ P_L + self._A_LIGHT @ N_L
            np.testing.assert_allclose(
                nucleon_count, A_inj, rtol=1e-5,
                err_msg=f"Nucleon count {nucleon_count:.6f} ≠ {A_inj} at L={L} Mpc",
            )

    def test_dynamic_charge_conservation(self, psb_cascade, fe56_injection, boost_idx):
        """Injecting Fe-56, the total charge must stay at 26 at every distance.

        Charge count = sum_i Z_i * P_i(L)  +  sum_l Z_l * N_l(L)
        """
        n_sp = len(psb_cascade.species)
        Z_sp = np.array([s[0] for s in psb_cascade.species], dtype=float)
        Z_inj = float(Z_sp @ fe56_injection)

        for L in self._L_VALUES:
            state = self._augmented_evolve(psb_cascade, fe56_injection, boost_idx, L)
            P_L = state[:n_sp]
            N_L = state[n_sp:]

            charge_count = Z_sp @ P_L + self._Z_LIGHT @ N_L
            np.testing.assert_allclose(
                charge_count, Z_inj, rtol=1e-5,
                err_msg=f"Charge count {charge_count:.6f} ≠ {Z_inj} at L={L} Mpc",
            )

    def test_species_account_for_all_nucleons_at_long_distance(self, psb_cascade, fe56_injection, boost_idx):
        """At large L, nucleons must be distributed among surviving species + light.

        Verifies that the cascade does not silently absorb or lose nucleons as
        the nuclear population fully disintegrates.  This is the most sensitive
        end-to-end check: errors in remnant indexing or branching-ratio sums
        tend to accumulate and become visible here.
        """
        n_sp = len(psb_cascade.species)
        A_sp = np.array([s[1] for s in psb_cascade.species], dtype=float)
        A_inj = float(A_sp @ fe56_injection)

        # Use a large distance where a substantial fraction of Fe-56 has decayed
        L_large = 500.0  # Mpc

        state = self._augmented_evolve(psb_cascade, fe56_injection, boost_idx, L_large)
        P_L = state[:n_sp]
        N_L = state[n_sp:]

        # Probability still sums to 1
        np.testing.assert_allclose(
            P_L.sum(), 1.0, rtol=1e-5,
            err_msg=f"sum(P) ≠ 1 at L={L_large} Mpc",
        )
        # Nucleon count still equals injection
        nucleon_count = A_sp @ P_L + self._A_LIGHT @ N_L
        np.testing.assert_allclose(
            nucleon_count, A_inj, rtol=1e-5,
            err_msg=f"Nucleon count {nucleon_count:.6f} ≠ {A_inj} at L={L_large} Mpc",
        )

    def test_multiple_injection_species(self, psb_cascade, boost_idx):
        """Conservation holds when multiple species are injected simultaneously."""
        species = psb_cascade.species
        n_sp = len(species)
        A_sp = np.array([s[1] for s in species], dtype=float)

        # Inject equal parts of every species with A ≥ 4
        alpha = np.array([1.0 if s[1] >= 4 else 0.0 for s in species])
        if alpha.sum() == 0:
            pytest.skip("No species with A >= 4 in PSB model")
        alpha /= alpha.sum()
        A_inj = float(A_sp @ alpha)

        for L in [1.0, 50.0]:
            state = self._augmented_evolve(psb_cascade, alpha, boost_idx, L)
            P_L = state[:n_sp]
            N_L = state[n_sp:]

            nucleon_count = A_sp @ P_L + self._A_LIGHT @ N_L
            np.testing.assert_allclose(
                nucleon_count, A_inj, rtol=1e-5,
                err_msg=f"Mixed-injection nucleon count {nucleon_count:.6f} ≠ {A_inj} at L={L} Mpc",
            )


# ---------------------------------------------------------------------------
# InteractionCore save / load
# ---------------------------------------------------------------------------

class TestInteractionCoreSaveLoad:
    """Verify that save()/load() recreates the PSB model exactly."""

    @pytest.fixture(scope="class")
    def ic(self):
        from crisp.core import InteractionCore_PSB_CMB
        return InteractionCore_PSB_CMB()

    @pytest.fixture(scope="class")
    def ic_loaded(self, ic, tmp_path_factory):
        path = str(tmp_path_factory.mktemp("ic") / "psb")
        ic.save(path)
        from crisp.core import InteractionCore_PSB_CMB
        ic2 = InteractionCore_PSB_CMB()
        ic2.load(path)
        return ic2

    def test_tensor_round_trip(self, ic, ic_loaded):
        np.testing.assert_array_equal(ic.tensor, ic_loaded.tensor)

    def test_light_prod_tensor_round_trip(self, ic, ic_loaded):
        np.testing.assert_array_equal(ic.light_prod_tensor, ic_loaded.light_prod_tensor)

    def test_boosts_round_trip(self, ic, ic_loaded):
        np.testing.assert_array_equal(ic.boosts, ic_loaded.boosts)

    def test_nuclei_round_trip(self, ic, ic_loaded):
        assert ic.nuclei == ic_loaded.nuclei

    def test_species_round_trip(self, ic, ic_loaded):
        assert ic.species == ic_loaded.species

    def test_all_branchings_round_trip(self, ic, ic_loaded):
        for orig, loaded in zip(ic.all_branchings, ic_loaded.all_branchings):
            np.testing.assert_array_equal(orig, loaded)

    def test_interpolator_matches(self, ic, ic_loaded):
        b = np.logspace(7, 12, 20)
        np.testing.assert_allclose(ic.interpolator(b), ic_loaded.interpolator(b))

    def test_interpyields_matches(self, ic, ic_loaded):
        b = np.logspace(7, 12, 20)
        np.testing.assert_allclose(ic.interpyields(b), ic_loaded.interpyields(b))

    def test_load_replaces_in_place(self, ic, tmp_path_factory):
        """load() must populate the existing object, not return a new one."""
        path = str(tmp_path_factory.mktemp("ic_ref") / "psb")
        ic.save(path)
        ref = ic
        result = ic.load(path)
        assert ref is ic
        assert result is None


# ---------------------------------------------------------------------------
