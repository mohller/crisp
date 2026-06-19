"""
Test suite for the crisp package.

Run with:
    pip install crisp-py          # or: pip install -e .
    pip install pytest
    pytest tests/test_crisp.py -v
"""

import numpy as np
import pytest


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
# 7. Numpy compatibility (regression guard against trapz removal)
# ---------------------------------------------------------------------------

class TestNumpyCompat:
    def test_no_trapz_usage(self):
        """Guard: ensure deprecated np.trapz is not called anywhere in crisp."""
        import ast, os, crisp
        pkg_dir = os.path.dirname(crisp.__file__)
        violations = []
        for root, _, files in os.walk(pkg_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                src = open(fpath).read()
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    # catch np.trapz(...)
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "trapz"):
                        violations.append(f"{fpath}: np.trapz call found")
        assert not violations, "\n".join(violations)
