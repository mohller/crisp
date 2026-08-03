"""
Astrophysical source models: injection spectra and target photon fields
for InteractionCore.

`InteractionCore` (in `core.py`) needs two things to propagate a cosmic
ray population: an injection vector (how many nuclei of which species,
at which energy) and, when photomeson or photodisintegration rates are
computed from a source's own radiation field rather than the CMB or the
EBL, a target photon spectrum. This module supplies both, derived from
physical source parameters (luminosity, variability timescale, bulk
Lorentz factor, and so on) instead of being specified by hand.

`UHECRSourceModel` is the abstract base every model in this module
subclasses. It stores its input parameters as `pint` quantities, so
values carry units and are converted rather than silently misread, and
each derived property (radius, magnetic field, photon density) is
computed together with the symbolic `sympy` expression and the
substituted values that produced it, so `generate_report()` can print
the equation actually used, not just its numeric result. Some
parameters are only meaningfully defined in one reference frame
(comoving/jet, engine/central source, or observer at Earth); `kind` and
`native_frame` on each parameter's schema entry declare how, and
`get_parameter(name, frame=...)` converts between them on request.

Concrete models built on that base:

- `OneZoneISModel`: the classic single-zone GRB internal-shock model
  parametrized by observed quantities (photon luminosity, bulk Lorentz
  factor, variability timescale, redshift), in the NeuCosmA tradition
  (Huemmer et al. 2012).
- `AGNJetModel`: an AGN jet blob (FSRQ/blazar) with a BLR external
  photon field Doppler boosted into the blob frame.
- `InternalShockModel`, and its subclasses `PhotosphericDissipationModel`
  and `ICMARTModel`: the GRB prompt-emission jet models of De Lia and
  Tamborra (2024, arXiv:2406.14975), parametrized by microphysical
  energy fractions (eps_d, eps_e, eps_A, eps_B) of the engine's isotropic
  energy budget rather than by observed quantities directly. The three
  differ only in where the dissipation happens and what target photon
  field results: internal shocks with a Band spectrum, a dissipative
  photosphere with a three component spectrum, or magnetic reconnection
  (ICMART) at a fixed radius.
- `VariablePhotonSource`: wraps a source model whose target photon field
  changes over time, keeping one `InteractionCore` in memory and
  serializing past states to disk so they can be swapped back in.

A source model connects to the rest of the package through
`build_core(epsrange, xsec_model)`, which builds an
`InteractionCore_Source` from the model's `target_photons` and a cross
section model, and `injection_spectrum(...)`, which returns the
baryon-budget-normalized injection callable Q(E) for that core's species
and boost grid. `xsec_model` is any `photonuclear_cross_sections`
model implementing the `Cross_Section_Model` interface (`CRPropa_model`,
`PSB_model`, `SimProp_model`, `Model_Rack`, and so on); this module only
duck-types against that interface and does not import from
`photonuclear_cross_sections` directly. From there, `rates_by_interaction`,
`loss_rates`, `max_energy`, and `species_loss_rates` give diagnostics on
that source's own photon field, and `compute_temporal_response` /
`simulate_time_evolution` fold the injection history through the
propagated response to get densities as a function of time rather than
just of distance.

See the notebooks under `examples/` (for instance
`GRB_Jet_Composition.ipynb`) for worked constructions of these models
end to end.
"""

import pint
import numpy as np
import sympy as sp
import textwrap
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, ClassVar, Optional
from dataclasses import dataclass
from sympy import symbols, Eq, pi, Rational
from scipy.signal import convolve
import scipy.constants as physconst
from scipy.interpolate import interp1d
import astropy.units as umod


# Use ureg for the units used
ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"  # Compact pretty format

# Fundamental constants in SI (robust dimensional analysis)
c_SI = physconst.c * ureg.meter / ureg.second       # Speed of light
e_SI = physconst.e * ureg.coulomb                   # Elementary charge
m_p_SI = physconst.m_p * ureg.kilogram              # Proton mass
sigma_T_SI = 6.6524587321e-29 * ureg.meter**2       # Thomson cross section


# ============================================================================
# Helper function for astrophysical quantity formatting
# ============================================================================
def format_quantity(qty: pint.Quantity) -> str:
    """Format astrophysical quantities with appropriate unit scaling"""
    # Energy units
    if qty.dimensionality == ureg.eV.dimensionality:
        mag = qty.magnitude
        if mag >= 1e21: return f"{mag/1e21:.2f} ZeV"
        elif mag >= 1e18: return f"{mag/1e18:.2f} EeV"
        elif mag >= 1e15: return f"{mag/1e15:.2f} PeV"
        elif mag >= 1e12: return f"{mag/1e12:.2f} TeV"
        elif mag >= 1e9: return f"{mag/1e9:.2f} GeV"
        else: return f"{mag:.2f} eV"

    # Length units (cm-based for astrophysics)
    if qty.dimensionality == ureg.centimeter.dimensionality:
        mag = qty.magnitude
        if mag >= 3.086e24: return f"{mag/3.086e24:.2f} Mpc"
        elif mag >= 3.086e21: return f"{mag/3.086e21:.2f} kpc"
        elif mag >= 3.086e18: return f"{mag/3.086e18:.2f} pc"
        elif mag >= 1e16: return f"{mag/1e16:.2f} ×10¹⁶ cm"
        else: return f"{mag:.2e} cm"

    # Time units
    if qty.dimensionality == ureg.second.dimensionality:
        mag = qty.magnitude
        if mag >= 3.154e7: return f"{mag/3.154e7:.2f} yr"
        elif mag >= 86400: return f"{mag/86400:.2f} days"
        elif mag >= 3600: return f"{mag/3600:.2f} hr"
        elif mag >= 60: return f"{mag/60:.2f} min"
        else: return f"{mag:.2f} s"

    # Generic formatting
    mag_str = f"{qty.magnitude:.2e}" if abs(qty.magnitude) < 0.01 or abs(qty.magnitude) > 1e5 else f"{qty.magnitude:.2f}"
    try:
        unit_str = f"{qty.units:~P}".replace("dimensionless", "").strip()
    except:
        unit_str = str(qty.units)
    return f"{mag_str} {unit_str}".strip()


# Parameter class Definition
@dataclass(frozen=False)
class ParameterSchema:
    """Schema definition with symbolic variable and physical unit.

    kind / native_frame declare how the quantity transforms between the
    one-zone reference frames ('comoving' = jet/shock rest, primed;
    'engine' = central-engine/source rest, tilde; 'observer' = Earth,
    includes redshift): kind 'time' | 'rate' | 'energy' transform with the
    standard factors, 'invariant' does not transform, and None (default)
    means no rule is declared — get_parameter(frame=) raises for those."""
    name: str
    unit: pint.Unit
    description: str
    category: str
    symbol: sp.Symbol
    kind: Optional[str] = None
    native_frame: str = 'comoving'


# Base class for source models
class UHECRSourceModel(ABC):
    """Base class for UHECR source models with SI-based robust computations"""

    # Symbolic variables
    n_gamma, eta, n_p, R, d, B, epsilon_gamma = symbols('n_gamma eta n_p R d B epsilon_gamma')
    z, w, Gamma, t_var, t_dyn, V_iso, u_em = symbols('z w Gamma t_var t_dyn V_iso u_em')
    r_L, E_p_max, t_pgamma, t_syn, t_esc, L_gamma, L_nu, L_CR = symbols('r_L E_p_max t_pgamma t_syn t_esc L_gamma L_nu L_CR')
    e_sym, m_p_sym, sigma_T_sym, c_sym, eph_min, eph_max, eph_brk = symbols('e m_p sigma_T c , eph_min eph_max eph_brk')
    beta_sym, Z_sym, sigma_pgamma_sym, kappa_gamma_sym, U_B_sym = symbols('beta Z sigma_pgamma kappa_gamma U_B')
    gamma_p_sym, f_pi_sym, L_gamma_sym, eta_diff_sym = symbols('gamma_p f_pi L_gamma eta_diff')

    property_methods = {}

    SCHEMA: ClassVar[List[ParameterSchema]] = [
        # Inputs (astrophysical units for user convenience)
        ParameterSchema("redshift", ureg.dimensionless, "Redshift of the source", "input", z, kind='invariant'),
        ParameterSchema("variability_timescale", ureg.second, "Observed variability timescale", "input", t_var, kind='time', native_frame='observer'),
        ParameterSchema("bulk_lorentz_factor", ureg.dimensionless, "Bulk Lorentz factor of relativistic outflow", "input", Gamma, kind='invariant'),
        ParameterSchema("photon_luminosity", ureg.erg/ureg.second, "Photon luminosity", "input", L_gamma),
        ParameterSchema("distance", ureg.centimeter, "Distance from central engine", "input", d),
        ParameterSchema("shell_width", ureg.centimeter, "Radial width of interacting shells", "input", w),
        ParameterSchema("baryonic_loading", ureg.dimensionless, r"Baryon-to-photon energy density ratio", "input", eta, kind='invariant'),
        ParameterSchema("volume", ureg.centimeter**3, "Volume of radiating region", "input", V_iso),
        ParameterSchema("em_density", ureg.erg/ureg.centimeter**3, "Electro-magnetic energy density, assuming equipartition.", "input", u_em),
        ParameterSchema("magnetic_field", ureg.gauss, "Magnetic field strength", "input", B),

        ParameterSchema("photon_energy_min", ureg.GeV, "Target photon minimal energy", "input", eph_min, kind='energy'),
        ParameterSchema("photon_energy_max", ureg.GeV, "Target photon maximal energy", "input", eph_max, kind='energy'),
        ParameterSchema("photon_energy_brk", ureg.GeV, "Target photon break energy", "input", eph_brk, kind='energy'),
        
        ParameterSchema("photon_density", 1/ureg.centimeter**3, "Target photon number density", "input", n_gamma),
        ParameterSchema("proton_density", 1/ureg.centimeter**3, "Proton number density", "input", n_p),
        ParameterSchema("radius", ureg.centimeter, "Characteristic size of emission region", "input", R),
        ParameterSchema("photon_energy", ureg.eV, "Characteristic target photon energy", "input", epsilon_gamma, kind='energy'),
        ParameterSchema("expansion_timescale", ureg.second, "Dynamical/expansion timescale", "input", t_dyn, kind='time'),

        # Derived Properties
        ParameterSchema("proton_larmor_radius", ureg.centimeter, "Larmor radius: r_L = E_p/(eB)", "input", r_L),
        ParameterSchema("proton_energy_max", ureg.eV, "Maximum proton energy (Hillas criterion)", "input", E_p_max, kind='energy'),
        ParameterSchema("photopion_loss_timescale", ureg.second, "Photopion interaction timescale", "input", t_pgamma, kind='time'),
        ParameterSchema("synchrotron_loss_timescale", ureg.second, "Proton synchrotron loss timescale", "input", t_syn, kind='time'),
        ParameterSchema("escape_timescale", ureg.second, "Particle escape timescale", "input", t_esc, kind='time'),
        ParameterSchema("neutrino_luminosity", ureg.erg/ureg.second, "Neutrino luminosity from photopion interactions", "input", L_nu),  # NOW DEFINED
        ParameterSchema("cosmic_ray_luminosity", ureg.erg/ureg.second, "Total cosmic ray luminosity", "input", L_CR),  # NOW DEFINED
    ]

    _SCHEMA_MAP: ClassVar[Dict[str, ParameterSchema]] = {p.name: p for p in SCHEMA}

    def __init__(self, **inputs: Any):
        input_frames = inputs.pop('frames', None) or {}
        self._inputs: Dict[str, pint.Quantity] = {}
        for key, value in inputs.items():
            if key not in self._SCHEMA_MAP:
                valid = sorted(self._SCHEMA_MAP.keys())
                raise ValueError(f"Unknown parameter '{key}'. Valid: {valid}")

            schema = self._SCHEMA_MAP[key]
            if isinstance(value, pint.Quantity):
                self._inputs[key] = value.to(schema.unit)
            else:
                self._inputs[key] = value * schema.unit

        # inputs declared in another frame (frames={'name': 'observer', ...})
        # are converted to each parameter's native frame before any derived
        # property is computed — an explicit declaration instead of a
        # silent per-class convention
        for key, frm in input_frames.items():
            if key not in self._inputs:
                raise ValueError(f"frames= given for '{key}' but the "
                                 "parameter was not provided")
            schema = self._SCHEMA_MAP[key]
            self._inputs[key] = self._inputs[key] * self._frame_factor(
                schema.kind, frm, schema.native_frame, name=key)

        self._computed_properties: Dict[str, pint.Quantity] = {}
        self._symbolic_expressions: Dict[str, sp.Expr] = {}
        self._evaluation_mappings: Dict[str, Dict[sp.Symbol, pint.Quantity]] = {}

        self._compute_all_properties()

    def parameters(self) -> List[str]:
        return list(self._inputs.keys()) + list(self._computed_properties.keys())

    FRAMES = ('comoving', 'engine', 'observer')

    def _frame_factor(self, kind, frm, to, name=''):
        """Multiplicative factor taking a quantity of the given kind from
        frame `frm` to frame `to` in the standard one-zone bookkeeping: an
        engine-frame interval t~ appears comoving as t' = Gamma t~ and
        observed as t_obs = (1 + z) t~; energies transform inversely
        (E_obs = Gamma E' / (1 + z)); rates inversely to times."""
        for f in (frm, to):
            if f not in self.FRAMES:
                raise ValueError(f"unknown frame '{f}' (use one of "
                                 f"{self.FRAMES})")
        if frm == to or kind == 'invariant':
            return 1.0
        if kind not in ('time', 'rate', 'energy'):
            raise ValueError(
                f"no frame rule declared for parameter '{name}' "
                f"(kind={kind!r}); its value is defined in its native "
                "frame only")
        Gam = float(self._inputs['bulk_lorentz_factor'].m)
        zz = float(self._inputs['redshift'].m)
        # factors that bring a value of this kind TO the engine frame
        if kind == 'time':
            to_engine = {'engine': 1.0, 'comoving': 1.0 / Gam,
                         'observer': 1.0 / (1.0 + zz)}
        else:                             # 'rate' and 'energy' scale alike
            to_engine = {'engine': 1.0, 'comoving': Gam,
                         'observer': 1.0 + zz}
        return to_engine[frm] / to_engine[to]

    def get_parameter(self, name: str, frame: Optional[str] = None) -> pint.Quantity:
        """Access function for parameter values.

        frame=None returns the value in its native frame (the historical
        behavior, bit-exact); frame='comoving' | 'engine' | 'observer'
        converts quantities whose schema declares a kind ('time', 'rate',
        'energy' or 'invariant') and raises for those without a rule.
        """
        if name in self._SCHEMA_MAP:
            symbol_key = name
        elif name in [str(schema.symbol) for schema in self._SCHEMA_MAP.values()]:
            for key, schema in self._SCHEMA_MAP.items():
                if str(schema.symbol) == name:
                    symbol_key = key
                    break
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        if symbol_key in self._inputs:
            value = self._inputs[symbol_key]
        elif symbol_key in self._computed_properties:
            value = self._computed_properties[symbol_key]
        else:
            print('not found in any of the paramewters')
            return None

        if frame is None:
            return value
        schema = self._SCHEMA_MAP[symbol_key]
        return value * self._frame_factor(schema.kind, schema.native_frame,
                                          frame, name=symbol_key)

    @abstractmethod
    def _compute_radius(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        pass

    @abstractmethod
    def _compute_shell_width(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        pass
        
    @abstractmethod
    def _compute_volume(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        pass

    @abstractmethod
    def _compute_em_density(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        pass

    @abstractmethod
    def _compute_magnetic_field(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        pass

    def _compute_synchrotron_loss_timescale(self, species=(1, 1), E=None
                                            ) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        """Generalized nuclear synchrotron cooling timescale in the
        source's magnetic field, for any species (Z, A):

            t_syn = 3 m_A^3 / (4 sigma_T m_e^2 Z^4 c U_B gamma),
            gamma = E / m_A,   U_B = B^2 / 8 pi,

        i.e. E/(dE/dt) with the Thomson-scaled nuclear cross section
        sigma_eff = sigma_T Z^4 (m_e/m_A)^2 — the exact reciprocal of
        loss_rates' 'synchrotron' entry at the same species and energy.
        Defaults (no arguments, the schema/report path): a proton at
        1 EeV. E may be a pint energy or a number in GeV.
        """
        from scipy.constants import physical_constants, c, e
        from .data.nucleardecays import nuclear_mass_GeV

        Z, A = int(species[0]), int(species[1])
        if E is None:
            E_GeV = (1 * ureg.EeV).to(ureg.GeV).m
        elif isinstance(E, pint.Quantity):
            E_GeV = E.to(ureg.GeV).m
        else:
            E_GeV = float(E)

        sigma_T_cm2 = physical_constants['Thomson cross section'][0] * 1e4
        m_e = physical_constants[
            'electron mass energy equivalent in MeV'][0] * 1e-3   # GeV
        m_A = nuclear_mass_GeV(Z, A)                              # GeV
        gamma = E_GeV / m_A
        B_G = self.get_parameter('magnetic_field').to(ureg.gauss).m
        u_B = B_G**2 / (8 * float(pi.evalf()))                    # erg/cm^3
        rate = (4.0 / 3.0) * sigma_T_cm2 * Z**4 * (m_e / m_A)**2 \
            * (c * 1e2) * u_B * gamma / (m_A * e * 1e16)          # 1/s
        t_s = (1.0 / rate) * ureg.second

        m_A_sym, m_e_sym = symbols('m_A m_e')
        expr = 3 * m_A_sym / (4 * self.sigma_T_sym * self.Z_sym**4
                              * (m_e_sym / m_A_sym)**2 * self.c_sym
                              * self.U_B_sym * self.gamma_p_sym)
        return t_s, expr, {
            m_A_sym: m_A * ureg.GeV, m_e_sym: m_e * ureg.GeV,
            self.sigma_T_sym: sigma_T_SI, self.c_sym: c_SI,
            self.Z_sym: Z * ureg.dimensionless,
            self.U_B_sym: u_B * ureg.erg / ureg.cm**3,
            self.gamma_p_sym: gamma * ureg.dimensionless,
        }

    def _compute_escape_timescale(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        """Advective escape over one shell crossing, t_esc = w'/c — for
        spherical-blob models whose shell_width is the radius this is
        identically R/c."""
        w_val = self.get_parameter('shell_width')
        t_s = (w_val.to(ureg.meter) / c_SI).to(ureg.second)
        return t_s, self.w / self.c_sym, {self.w: w_val, self.c_sym: c_SI}
        
    def _compute_all_properties(self) -> None:

        for prop_name, compute_method in self.property_methods.items():
            try:
                quantity, expr, mapping = compute_method()
                expected_unit = self._SCHEMA_MAP[prop_name].unit
                if not quantity.check(expected_unit):
                    raise pint.DimensionalityError(quantity.units, expected_unit)

                self._computed_properties[prop_name] = quantity

                if not isinstance(expr, sp.Eq):
                    lhs_symbol = self._SCHEMA_MAP[prop_name].symbol
                    expr = Eq(lhs_symbol, expr)
                self._symbolic_expressions[prop_name] = expr
                self._evaluation_mappings[prop_name] = mapping
            except (KeyError, TypeError, ZeroDivisionError, pint.DimensionalityError, AttributeError) as e:
                print('error found', e)
                continue

    def _render_equation(self, expr: sp.Eq, mapping: Dict[sp.Symbol, pint.Quantity],
                        result_qty: pint.Quantity, max_width: int = 90) -> List[str]:
        lhs = expr.lhs
        rhs = expr.rhs
        symbolic_eq = f"{lhs} = {rhs}"

        # Skip fundamental constants in substitution display
        fundamental = {self.e_sym, self.m_p_sym, self.sigma_T_sym, self.c_sym,
                      self.beta_sym, self.Z_sym, self.sigma_pgamma_sym, self.kappa_gamma_sym}

        subs_lines = []
        for sym, qty in sorted(mapping.items(), key=lambda x: str(x[0])):
            if sym in fundamental:
                continue
            val_str = format_quantity(qty)
            subs_lines.append(f"    {sym} = {val_str}")

        lines = [f"  Symbolic: {symbolic_eq}"]
        if subs_lines:
            lines.append("  Substitutions:")
            lines.extend(subs_lines)
        lines.append(f"  Result:   {format_quantity(result_qty)}")
        return lines

    def generate_report(self, max_width: int = 100) -> str:
        lines = ["=" * max_width]
        lines.append(f"UHECR SOURCE MODEL REPORT: {self.__class__.__name__}")
        lines.append(f"Fireball Physics - Units: SI-based")
        lines.append("-" * max_width)
        lines.append("")

        # Inputs
        lines.append("INPUT PARAMETERS")
        lines.append("-" * max_width)
        lines.append(f"{'Parameter':<25} {'Value':<25} {'Description'}")
        lines.append("-" * max_width)

        input_params = [p for p in self.SCHEMA if p.category == 'input' and p.name in self._inputs]
        for param in sorted(input_params, key=lambda p: p.name):
            qty = self._inputs[param.name]
            val_str = format_quantity(qty)
            desc = textwrap.fill(param.description, width=max_width-55, subsequent_indent=' ' * 55)
            lines.append(f"{param.name:<25} {val_str:<25} {desc}")
        lines.append("")

        # Properties
        lines.append("DERIVED PROPERTIES")
        lines.append("-" * max_width)

        property_params = [p for p in self.SCHEMA if p.category == 'property' and p.name in self._computed_properties]

        if not property_params:
            lines.append("  No properties computed (insufficient inputs or computation error)")
        else:
            for param in sorted(property_params, key=lambda p: p.name):
                qty = self._computed_properties[param.name]
                expr = self._symbolic_expressions[param.name]
                mapping = self._evaluation_mappings[param.name]

                lines.append(f"\n -> {param.name.upper()}")
                lines.append(f"  Unit:        {param.unit:~P}")
                lines.append(f"  Description: {param.description}")
                lines.extend(self._render_equation(expr, mapping, qty, max_width))

        lines.append("\n" + "-" * max_width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        inputs_str = ", ".join(f"{k}={format_quantity(v).replace(' ', '')}"
                             for k, v in sorted(self._inputs.items()))
        return f"{self.__class__.__name__}({inputs_str})"

    def build_core(self, epsrange, xsec_model):
        """Build and store an InteractionCore_Source from self.target_photons.

        Arguments:
        ----------
        epsrange   : (e_min, e_max) photon energy range in GeV
        xsec_model : cross-section model instance (e.g. CRPropa_model)
        """
        from .core import InteractionCore_Source
        self.interaction_core = InteractionCore_Source(
            epsrange, self.target_photons, xsec_model=xsec_model
        )

    def rates_by_interaction(self, nucleus=None, interaction_core=None):
        """Interaction rates on the source's photon field decomposed by type
        (photodisintegration / photomeson), in Mpc^-1.

        Thin passthrough to InteractionCore.rates_by_interaction of the core
        given as argument — or the one stored by build_core /
        compute_temporal_response / simulate_time_evolution.
        """
        core = self._resolve_core(interaction_core)
        return core.rates_by_interaction(nucleus=nucleus)

    def _resolve_core(self, interaction_core=None):
        core = interaction_core if interaction_core is not None \
            else getattr(self, 'interaction_core', None)
        if core is None:
            raise ValueError('no interaction core available: pass interaction_core=, '
                             'or run build_core / compute_temporal_response first')
        return core

    def observed_energy(self, E_comoving):
        """Observer-frame energy of a comoving energy: E Gamma_bulk / (1 + z)."""
        return (np.asarray(E_comoving) * self.get_parameter('Gamma').m
                / (1 + self.get_parameter('redshift').m))

    def fluence_factor(self, cosmology=None) -> float:
        """Observer-frame fluence factor V' t_dur,obs (1+z)^2 / (4 pi d_L^2)
        [cm s]: converts a per-shell steady comoving production-rate density
        Q'(E (1+z)/Gamma) [GeV^-1 cm^-3 s^-1] into fluence at Earth
        (De Lia & Tamborra 2024, eq. 4.14).

        Consumes the schema parameters volume, duration and redshift;
        duration goes through the frame machinery (kind='time'), so any
        native frame — engine (InternalShockModel family) or observer
        (OneZoneISModel) — yields the observed duration.

        cosmology : astropy cosmology for d_L(z); default is the paper's
        flat LambdaCDM (H0 = 67.4 km/s/Mpc, Omega_M = 0.315).
        """
        from astropy.cosmology import FlatLambdaCDM

        if cosmology is None:
            cosmology = FlatLambdaCDM(H0=67.4, Om0=0.315)
        z_val = self.get_parameter('redshift').m
        d_L = cosmology.luminosity_distance(z_val).to('cm').value
        V_cm3 = self.get_parameter('volume').to(ureg.cm**3).m
        t_obs = self.get_parameter('duration',
                                   frame='observer').to(ureg.second).m
        return V_cm3 * t_obs * (1 + z_val)**2 / (4 * np.pi * d_L**2)

    def neutrino_fluence(self, E_nu, Q_nu_mu=None, Q_nu_e=None, detail=None,
                         rebin=1, theta12=33.5, cosmology=None):
        """Per-flavor neutrino fluence at Earth from the source's comoving
        production spectra, with averaged vacuum oscillations.

        The flavor transfer uses the theta_13 = 0, theta_23 = 45 deg
        averaged probabilities (De Lia & Tamborra 2024, eq. 4.14):
        P_ee = 1 - s/2, P_emu = P_etau = s/4, P_mumu = P_mutau = (4 - s)/8
        with s = sin^2(2 theta_12) — a unitary matrix, so the all-flavor
        fluence equals fluence_factor x (Q_mu + Q_e) exactly. The energy
        transform to the observer frame carries no extra Jacobian: the
        source's fluence_factor() (V' t_dur (1+z)^2 / 4 pi d_L^2) is
        constructed for per-shell steady comoving rates evaluated at
        E (1+z)/Gamma.

        Arguments:
        ----------
        E_nu    : comoving neutrino energy grid [GeV] (log-uniform), as
                  returned by InteractionCore.neutrino_production.
        Q_nu_mu : nu_mu + nubar_mu steady production rate per bin
                  [cm^-3 s^-1]; Q_nu_e likewise for nu_e + nubar_e (add
                  neutron_decay_neutrinos output here — beta decays feed
                  the electron flavor).
        detail  : ALTERNATIVE to Q_nu_mu/Q_nu_e — the nu/nubar-resolved
                  dict of neutrino_production's 'detail' ('nu_mu',
                  'nubar_mu', 'nu_e', 'nubar_e'; same per-bin rate units;
                  add neutron-decay antineutrinos to 'nubar_e'). The same
                  averaged matrix applies per CP sector, and the returned
                  dict carries all six species separately.
        rebin   : merge groups of `rebin` bins before differencing
                  (display smoothing of sparse comb spectra).
        theta12 : solar mixing angle [deg].
        cosmology : forwarded to fluence_factor.

        Returns:
        --------
        (E_obs, F) : observer-frame energies [GeV] and a dict of fluences
                     per comoving energy [GeV^-1 cm^-2] on that grid —
                     {'nu_mu','nu_e','nu_tau'} (each flavor + its
                     antineutrinos) for the summed input, or the six
                     species separately for detail= input.
        """
        E_nu = np.asarray(E_nu)
        s2 = np.sin(2 * np.radians(theta12))**2
        P_ee, P_emu, P_mumu = 1 - s2 / 2, s2 / 4, (4 - s2) / 8
        dln = np.log(E_nu[1] / E_nu[0]) * rebin
        nb = (len(E_nu) // rebin) * rebin
        reb = lambda v: np.asarray(v)[:nb].reshape(-1, rebin).sum(axis=1)
        E_r = np.exp(np.log(E_nu[:nb]).reshape(-1, rebin).mean(axis=1))
        ff = self.fluence_factor(cosmology=cosmology)
        if (detail is None) == (Q_nu_mu is None):
            raise ValueError('pass either Q_nu_mu and Q_nu_e, or detail=')
        if detail is not None:
            F = {}
            for bar in ('', 'bar'):
                Qmu = reb(detail[f'nu{bar}_mu']) / (E_r * dln)
                Qe = reb(detail[f'nu{bar}_e']) / (E_r * dln)
                F[f'nu{bar}_e'] = ff * (P_ee * Qe + P_emu * Qmu)
                F[f'nu{bar}_mu'] = ff * (P_emu * Qe + P_mumu * Qmu)
                F[f'nu{bar}_tau'] = F[f'nu{bar}_mu'].copy()
            return self.observed_energy(E_r), F
        Qmu, Qe = reb(Q_nu_mu) / (E_r * dln), reb(Q_nu_e) / (E_r * dln)
        F = {'nu_mu': ff * (P_mumu * Qmu + P_emu * Qe),
             'nu_e': ff * (P_emu * Qmu + P_ee * Qe),
             'nu_tau': ff * (P_mumu * Qmu + P_emu * Qe)}
        return self.observed_energy(E_r), F

    def acceleration_rate(self, species, E_GeV, eta_acc=1.0):
        """Bohm-like acceleration rate t_acc^-1 = eta_acc Z e B' c / E  [1/s],
        with the source's (equipartition) magnetic field."""
        from scipy.constants import c
        B_G = self.get_parameter('magnetic_field').to('gauss').m
        # Z e B c / E with E in GeV and B in gauss: Z c^2 1e-13 B / E_GeV [1/s]
        return eta_acc * species[0] * c**2 * 1e-13 * B_G / np.asarray(E_GeV)

    def loss_rates(self, species, interaction_core=None, include_pair=False,
                   include_ic=False, kappa=False):
        """Energy-loss / interaction rates of a species inside the source, on
        the core's boost grid [1/s]: photonuclear (photodisintegration +
        photomeson interaction rates on the source photons), nuclear
        synchrotron in the field B, and adiabatic Gamma c/R.

        include_pair adds the Bethe-Heitler pair-production energy-loss rate
        on the source photons ('pair' key, via continuous_losses.Bpp_generic)
        and includes it in 'total'; off by default so that the balance of
        max_energy is unchanged unless requested.

        kappa=True converts the photonuclear entries from interaction rates
        to energy-loss (cooling) rates by the exact inelasticity: for free
        nucleons 1 - <E_out>/E_in from the core's recoil kernels (the
        SOPHIA-table inelasticity, ~0.15 at the Delta to ~0.45 multipion);
        for nuclei <Delta A>/A from the core's channel tensor (GDR and the
        one-nucleon photomeson loss unified). This makes 'photomeson',
        'photodisintegration', 'photonuclear' and 'total' commensurable
        with the other entries, which are energy-loss rates by nature.
        Off by default: the interaction-rate convention answers
        "survives as this species" and keeps all outputs bit-exact.

        include_ic adds the inverse-Compton energy-loss rate on the source
        photons ('inverse_compton' key): the Thomson rate with the nuclear
        scaling sigma_T Z^4 (m_e/m_A)^2 — identical in form to synchrotron
        with u_B -> u_gamma, so at exact equipartition the two coincide in
        the Thomson regime — suppressed per photon energy by the
        Klein-Nishina factor (1 + b)^{-3/2}, b = 4 gamma eps / m_A
        (Moderski et al. 2005). Included in 'total' when requested;
        off by default (same convention as include_pair).

        Returns:
        --------
        dict with 'E' (comoving energies, GeV), 'photodisintegration',
        'photomeson', 'photonuclear', 'synchrotron', 'adiabatic'
        [, 'pair'], 'total'
        """
        from scipy.constants import c, e, parsec, physical_constants
        core = self._resolve_core(interaction_core)

        boosts = core.boosts
        E = core.energy_of_boost(species, boosts)
        c_cm = c * 1e2
        c_Mpc = c_cm / (parsec * 1e8)                       # Mpc/s

        dec = core.rates_by_interaction(nucleus=species)
        r_pdis = dec['photodisintegration'] * c_Mpc
        r_pm = dec['photomeson'] * c_Mpc
        if species[1] == 1 and hasattr(core, 'photomeson_rates_pn'):
            # free nucleons are not mothers of the rack's A->A' channels;
            # their photomeson interaction rate is the light-sector hook
            # (one array, p ~ n)
            r_pm = core.photomeson_rates_pn * c_Mpc

        sigma_T = physical_constants['Thomson cross section'][0] * 1e4    # cm^2
        m_e = physical_constants['electron mass energy equivalent in MeV'][0] * 1e-3
        m_A = core.energy_of_boost(species, 1.0)            # GeV
        GeV_erg = e * 1e16                                  # 1 GeV in erg
        # u_B from the model's own field (subclasses define B, e.g. by
        # equipartition) — the base makes no assumption beyond B itself
        B_G = self.get_parameter('magnetic_field').to('gauss').m
        u_B = B_G**2 / (8 * np.pi)                          # erg/cm^3
        sigma_eff = sigma_T * species[0]**4 * (m_e / m_A)**2
        r_syn = 4 / 3 * sigma_eff * c_cm * u_B * boosts / (m_A * GeV_erg)

        # comoving adiabatic/expansion rate Gamma c / R (De Lia & Tamborra
        # Eq. 3.3): the lab-frame expansion time R/c appears time-dilated
        # in the shell frame. Models without a bulk flow keep c/R.
        try:
            Gam_ad = float(self.get_parameter('bulk_lorentz_factor').to('').m)
        except Exception:
            Gam_ad = 1.0
        r_ad = np.full_like(E, Gam_ad * c_cm
                            / self.get_parameter('radius').to('cm').m)

        if kappa:
            if species[1] == 1:
                # free nucleon: 1 - <E_out>/E_in from the recoil kernels
                kern = getattr(core, 'photomeson_kernels', None)
                if not kern:
                    raise ValueError('kappa=True needs the charge-resolved '
                                     "kernels (photomeson='kernels')")
                s = 0 if species[0] == 1 else 1
                K_N = kern['p'][s] + kern['n'][s]          # (n_b, n_b)
                R_N = K_N.sum(axis=1)
                kept = (K_N @ boosts) / np.maximum(R_N * boosts, 1e-300)
                kap = np.where(R_N > 0, np.clip(1.0 - kept, 0.0, 1.0), 0.0)
            else:
                # nucleus: <Delta A>/A over the A-changing channels (the
                # tensor diagonal is the negative total rate — exclude it)
                i_sp = core.species.index(tuple(species))
                T = core.tensor[i_sp].copy()               # (n_sp, n_b)
                T[i_sp] = 0.0
                A_out = np.array([sp[1] for sp in core.species], float)
                R_A = T.sum(axis=0)
                dA = ((species[1] - A_out) @ T) / np.maximum(R_A, 1e-300)
                kap = np.where(R_A > 0, np.clip(dA, 0.0, None)
                               / species[1], 0.0)
            r_pdis = r_pdis * kap
            r_pm = r_pm * kap

        out = {'E': E, 'photodisintegration': r_pdis, 'photomeson': r_pm,
               'photonuclear': r_pdis + r_pm, 'synchrotron': r_syn,
               'adiabatic': r_ad, 'total': r_pdis + r_pm + r_syn + r_ad}

        if include_pair:
            from .continuous_losses import Bpp_generic
            tp = core.target_photons
            fields = tp if isinstance(tp, (list, tuple)) else (tp,)
            # adapter: our fields take GeV and return GeV^-1 cm^-3;
            # Bpp_generic wants eV -> eV^-1 m^-3 (x 1e-9 per eV, x 1e6 per m^3)
            dens = lambda e_eV, z: sum(f(e_eV * 1e-9) for f in fields) * 1e-3
            # Bpp_generic integrates target photons from m_e c^2/gamma up to
            # 1e4 eV: below gamma ~ m_e c^2 / 1e4 eV the grid inverts, so mask
            # those boosts (pair losses are negligible against adiabatic there)
            r_pair = np.zeros_like(E)
            valid = boosts > physical_constants['electron mass energy equivalent in MeV'][0] * 1e6 / 1e4
            if valid.any():
                r_pair[valid] = np.clip(
                    Bpp_generic(species[0], m_A / core.energy_of_boost((1, 1), 1.0),
                                boosts[valid], phot_dens=dens) * c_Mpc, 0.0, None)
            out['pair'] = r_pair
            out['total'] = out['total'] + r_pair

        if include_ic:
            tp = core.target_photons
            fields = tp if isinstance(tp, (list, tuple)) else (tp,)
            eg = np.logspace(-10, -1.5, 400)               # GeV
            ng = sum(f(eg) for f in fields)                # GeV^-1 cm^-3
            b_kn = 4.0 * boosts[:, None] * eg[None, :] / m_A
            u_eff = np.trapezoid(eg * ng * (1.0 + b_kn) ** -1.5, eg, axis=1)
            r_ic = 4 / 3 * sigma_eff * c_cm * (u_eff * GeV_erg) \
                * boosts / (m_A * GeV_erg)
            out['inverse_compton'] = r_ic
            out['total'] = out['total'] + r_ic

        return out

    def species_loss_rates(self, species_list, interaction_core=None,
                           include_pair=True, include_ic=False):
        """Continuous-loss rates of several species at once, converted to
        the [1/Mpc] convention and split into the coherent/dispersive drift
        arrays consumed by InteractionCore.species_evolution_boost_range
        (coherent_loss=, energy_loss=) and reprocessed_nucleons
        (energy_loss=): loops loss_rates over species_list and separates
        its 1/s entries into the adiabatic rate (species- and
        boost-independent by construction -- coherent, an exact rigid
        shift on the receiving end, methods paper Sect. 3.1) and
        synchrotron (+ pair, + inverse Compton if requested), which are
        boost-dependent even for a single species and so have no exact
        treatment (dispersive -- conservative CIC, Sect. 3.2). Neutral
        species (Z = 0) get zero dispersive rate without evaluating
        synchrotron/pair/IC: both vanish exactly for Z = 0, but pair's own
        integral is expensive to run just to multiply the result by zero.

        Arguments:
        ----------
        species_list : sequence of (Z, A) tuples, in the order the
                      caller's true_range columns are in.
        interaction_core, include_pair, include_ic : as in loss_rates.

        Returns:
        --------
        coherent : float [1/Mpc] (the adiabatic rate).
        dispersive : ndarray (n_boosts, len(species_list)) [1/Mpc], on the
                     core's boost grid.
        """
        from scipy.constants import c, parsec
        core = self._resolve_core(interaction_core)
        c_Mpc = c * 1e2 / (parsec * 1e8)

        coherent = None
        dispersive = np.zeros((len(core.boosts), len(species_list)))
        for i, sp in enumerate(species_list):
            lr = self.loss_rates(sp, core, include_pair=include_pair, include_ic=include_ic)
            if coherent is None:
                coherent = float(lr['adiabatic'][0]) / c_Mpc
            if sp[0] == 0:
                continue                   # neutral: synchrotron/pair/IC vanish exactly
            disp = lr['synchrotron'].copy()
            if include_pair and 'pair' in lr:
                disp = disp + lr['pair']
            if include_ic and 'inverse_compton' in lr:
                disp = disp + lr['inverse_compton']
            dispersive[:, i] = disp / c_Mpc

        return coherent, dispersive

    def max_energy(self, species, interaction_core=None, eta_acc=1.0,
                   kappa=False):
        """Comoving maximal energy E'_max [GeV]: where the acceleration rate
        equals the sum of all loss rates (cf. paper Appendix D). kappa=True
        balances against the photonuclear energy-loss rates instead of the
        interaction rates (see loss_rates): acceleration then competes with
        cooling, as in De Lia & Tamborra Fig. 1 — E'_max rises where
        photonuclear-limited."""
        rates = self.loss_rates(species, interaction_core, kappa=kappa)
        acc = self.acceleration_rate(species, rates['E'], eta_acc)
        below = acc < rates['total']
        if not below.any() or below.all():
            raise ValueError('no acceleration/loss crossing on the boost grid')
        ix = int(np.argmax(below))
        ratio = np.log(acc / rates['total'])
        return float(np.exp(np.interp(0.0, ratio[[ix, ix - 1]],
                                      np.log(rates['E'])[[ix, ix - 1]])))

    def effective_thickness(self, L=None, index=None):
        """Coherent-inhomogeneity change of variable for adiabatic cooling
        (methods paper Sect. 3.1), injection side.

        Adiabatic cooling at the comoving expansion rate Gamma c / R (cf.
        loss_rates' 'adiabatic' entry) is species- and boost-independent: a
        rigid drift b = r_ad / c = Gamma / R per unit path length in
        ln(gamma). It is therefore a coherent inhomogeneity — the
        homogeneous transport solution is reused unchanged, with the
        propagation variable remapped. For a power-law injection of
        spectral index -index, integrating the injection along the cooled
        characteristics turns a residence length L into the effective
        thickness

            delta(L) = (1 - exp(-(index - 1) b L)) / ((index - 1) b),

        the closed form of int_0^L exp(-(index-1) b s) ds — always < L, and
        -> L as b -> 0 or index -> 1.

        Arguments:
        ----------
        L     : residence length(s) [Mpc]; scalar or array. Ignored when
                index is None.
        index : spectral index k of the injection q ~ E^-k, or None.

        Returns:
        --------
        index=None : the drift b [1/Mpc] (plain float).
        else       : delta(L) [Mpc], same shape as L.
        """
        try:
            Gam_ad = float(self.get_parameter('bulk_lorentz_factor').to('').m)
        except Exception:
            Gam_ad = 1.0
        b = Gam_ad / self.get_parameter('radius').to('Mpc').m   # 1/Mpc

        if index is None:
            return b

        L = np.asarray(L, dtype=float)
        a = (index - 1.0) * b
        if a == 0.0:
            return L + 0.0
        return (1.0 - np.exp(-a * L)) / a

    def injection_spectrum(self, species=(26, 56), index=-2.0, E_max=None,
                           boost_range=(1e1, 1e8), interaction_core=None,
                           eta_acc=1.0, injection_time=None):
        """Baryon-budget-normalized injection spectrum of the source (cf.
        paper Appendix D): Q(E) = C' E^index exp[-(E/E_max)^2], with C' fixed
        by the energy argument

            int E Q(E) dE = baryonic_loading * em_density / t_inj,

        i.e. the baryon budget is injected over the episode t_inj —
        by default one crossing of the emission region, shell_width / c.
        E_max is computed from max_energy() when not given.

        Arguments beyond the spectral shape:
        ----------
        injection_time : episode duration [s]; default shell_width / c.

        Returns:
        --------
        q    : callable gamma -> injected number density per second and per
               ln(gamma) [cm^-3 s^-1], ready for convolve_with_injection /
               simulate_time_evolution (injection_density=) and, multiplied
               by the ladder's dln(gamma), for the production folds (weights=)
        info : dict with 'C' [GeV^(-index-1) cm^-3 s^-1], 'E_max' [GeV],
               'species', 'index'
        """
        from scipy.constants import c
        core = self._resolve_core(interaction_core)
        if E_max is None:
            E_max = self.max_energy(species, core, eta_acc)

        eta = self.get_parameter('baryonic_loading').m
        u = self.get_parameter('em_density').to(ureg.GeV / ureg.centimeter**3).m
        t_cross = injection_time if injection_time is not None \
            else self.get_parameter('shell_width').to('m').m / c    # s

        E_of = lambda g: core.energy_of_boost(species, np.asarray(g))
        lg = np.linspace(np.log(boost_range[0]), np.log(boost_range[1]), 600)
        En = E_of(np.exp(lg))
        norm = np.trapezoid(En**(index + 2) * np.exp(-(En / E_max)**2), lg)
        C = eta * u / (t_cross * norm)

        q = lambda g: C * E_of(g)**(index + 1) * np.exp(-(E_of(g) / E_max)**2)
        return q, {'C': C, 'E_max': E_max, 'species': species, 'index': index}

    def compute_temporal_response(self, interactions_core=None, nucinj=(26, 56), boosts=None, distance_grid=np.logspace(-3.5, .5, 50)):
        """Computes the temporal response of the nuclear densities for the source.

           Arguments:
           ----------
           interactions_core: InteractionCore_Source instance; defaults to self.interaction_core
           nucinj: the injected species, injected constantly over the variability timescale of the source.
           boosts: boost grid for the computation
           distance_grid: grid of distances for the computation, fraction relative to the total thickness of source
        """
        if interactions_core is None:
            interactions_core = self.interaction_core
        self.interaction_core = interactions_core

        Zinj, Ainj = 26, 56

        if boosts is None:
            boosts = interactions_core.boosts

        traversed_length = self.get_parameter('w').to('Mpc').m
        distances = distance_grid * traversed_length

        alpha, mr, tr, redtens = interactions_core.get_distribution_parameters(mass_lims=(Ainj, 0), injection_type=('only species', (Zinj, Ainj)), absorption_type=('only mass', [1]))
        spec_evol = interactions_core.species_evolution_boost_range(distances, alpha=alpha, mass_range=mr, boost_range=boosts, true_range=tr)

        self.boosts = boosts
        self.distances = distances
        self.spec_evol = spec_evol
    
    def convolve_with_injection(self, response_distances, response, timegridsize=1000,
                                injection_density=None):
        """Fold a per-injected-particle response with the source injection history.

        The temporal integral behind simulate_time_evolution, exposed for any
        per-particle quantity: particles are injected at a constant rate
        over the injection episode and travel at c, so a response X(L) sampled
        along the traversed distance accumulates in the source as

            n_X(t) = sum_t' Q(t') X(c (t - t')) dt'

        A cumulative per-particle yield (e.g. cumulated light-species, pion or
        neutrino production) gives a cumulative density in [X] cm^-3; a rate
        gives a rate density.

        Requires compute_temporal_response (or simulate_time_evolution with
        update_response=True) to have run, which fixes the injection episode.

        Arguments:
        ----------
        response_distances : distances [Mpc] where the response is sampled
        response : array (..., len(response_distances)), per injected particle
        timegridsize : resolution of the injection time grid
        injection_density : injected number density per second [cm^-3 s^-1]
            during the episode. Default (None) keeps the legacy convention of
            injecting the em_density magnitude as a number density — note that
            overstates the baryon content by a factor E_particle/loading: an
            injection consistent with the baryonic loading is
            baryonic_loading * em_density / E_particle(boost) with E in GeV.
            An array is broadcast against the leading axes of the response
            (e.g. one value per boost row).

        Returns:
        --------
        conv_time_grid : times [s] (same grid as simulate_time_evolution)
        convolved : array (..., len(conv_time_grid))
        """
        injection_time = self.distances[-1] / c_SI.to('Mpc/s').m
        tgrid = np.cumsum(injection_time / timegridsize * np.ones(timegridsize))

        regular = interp1d(response_distances, response,
                           bounds_error=False, fill_value=0)(c_SI.to('Mpc/s').m * tgrid)

        amp = np.asarray(self.get_parameter('em_density').m if injection_density is None
                         else injection_density)
        if amp.ndim:
            regular = regular * amp[..., None]   # per-row amplitudes
            amp = 1.0

        # constant injection over the episode
        Qinj = lambda t, tmax: amp * np.diff(tgrid)[0] * (1 - np.heaviside(t - tmax, 1))
        q0 = Qinj(tgrid, 1)

        # Convolution of injection and delta-injection evolution
        convolved = convolve(regular, q0.reshape((1,) * (np.ndim(regular) - 1) + (-1,)),
                             mode='full')
        conv_time_grid = np.append(tgrid - tgrid[0], tgrid[-1] + tgrid[:-1])
        return conv_time_grid, convolved

    def simulate_time_evolution(self, update_response=False, interactions_core=None, nucinj=(26, 56), boosts=None, distance_grid=np.logspace(-3.5, .5, 50), timegridsize=1000, injection_density=None):
        """Computes the temporal evolution of the nuclear densities for the source.

           Arguments:
           ----------
           update_response: is responso should be recomputed, default False
           interactions_core: instance of InteractionCore_Source
           nucinj: the injected species, injected constantly over the variability timescale of the source.
           boosts: boost grid for the computation
           distance_grid: grid of distances for the computation, fraction relative to the total thickness of source
           injection_density: injected number density per second [cm^-3 s^-1],
               scalar or one value per boost; see convolve_with_injection.
        """
        if update_response:
            self.compute_temporal_response(interactions_core, nucinj, boosts, distance_grid)

        conv_time_grid, convolved = self.convolve_with_injection(
            self.distances, np.permute_dims(self.spec_evol, (2, 0, 1)), timegridsize,
            injection_density=injection_density)

        self.conv_time_grid = conv_time_grid
        self.convolved = convolved


class OneZoneISModel(UHECRSourceModel):
    """One-zone internal-shock (colliding-shells) GRB prompt-emission model,
    parametrized by observed quantities (photon_luminosity, Gamma, t_var, z,
    band): collision radius R = 2 c Gamma^2 t_var / (1+z), comoving shell
    width c Gamma t_var / (1+z), photon density u'_gamma = L / (4 pi R^2
    Gamma^2 c) and equipartition B' = sqrt(8 pi u'_gamma) — the classic
    NeuCosmA-style one-zone setup (Huemmer et al. 2012; the per-collision
    physics of Bustamante et al. 2017 reduced to a single zone). See
    InternalShockModel for the energy-budget (epsilon-fraction)
    parametrization of De Lia & Tamborra 2024."""

    SCHEMA: ClassVar[List[ParameterSchema]] = (
        # Independent copies so category mutations stay local to this class
        [ParameterSchema(p.name, p.unit, p.description, p.category, p.symbol,
                         kind=p.kind, native_frame=p.native_frame)
         for p in UHECRSourceModel.SCHEMA]
        + [
            # observed T90: this model is parametrized by observer-frame
            # quantities, so fluence_factor's duration is native here
            ParameterSchema('duration', ureg.second,
                            'Burst duration T90 (observer frame)', 'input',
                            symbols('t_dur'), kind='time',
                            native_frame='observer'),
        ]
    )
    _SCHEMA_MAP: ClassVar[Dict[str, ParameterSchema]] = {p.name: p for p in SCHEMA}

    def __init__(self, **inputs: Any):
      
        self.property_methods = {
            'radius': self._compute_radius,
            'shell_width': self._compute_shell_width,
            'volume': self._compute_volume,
            'em_density': self._compute_em_density,
            'magnetic_field': self._compute_magnetic_field
        }

        for p in self.SCHEMA:
            if p.name in self.property_methods:
                p.category = 'property'

        super().__init__(**inputs)

        # computing target photon field
        from .background_photon_models import target_photons_spectrum

        target_photons_GRB = \
        target_photons_spectrum(self._inputs['photon_energy_min'].m,
                                self._inputs['photon_energy_max'].m,
                                self._inputs['photon_energy_brk'].m,
                                1, 2, normal=((self._inputs['photon_energy_min'].m, self._inputs['photon_energy_max'].m), 
                                self.get_parameter('em_density').m)) # density reduced for less disintegration

        self.target_photons = target_photons_GRB


    def _compute_radius(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        Gamma_val = self._inputs['bulk_lorentz_factor']
        tvar_val = self._inputs['variability_timescale']
        z_val = self._inputs['redshift']

        R_val = 2 * c_SI * Gamma_val**2 * tvar_val / (1 + z_val)
        R_km = R_val.to(ureg.km)

        return R_val, 2 * self.c_sym * self.Gamma**2 * self.t_var / (1 + self.z), {
            self.c_sym: c_SI, self.Gamma: Gamma_val, self.t_var: tvar_val, self.z: z_val
        }

    def _compute_shell_width(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        tvar_val = self._inputs['variability_timescale']
        Gamma_val = self._inputs['bulk_lorentz_factor']
        z_val = self._inputs['redshift']

        w_val = c_SI * Gamma_val * tvar_val / (1 + z_val)
        w_km = w_val.to(ureg.km)

        return w_km, self.c_sym * self.Gamma * self.t_var / (1 + self.z), {
            self.c_sym: c_SI, self.Gamma: Gamma_val, self.t_var: tvar_val, self.z: z_val
        }

    def _compute_volume(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        R_val = w_val = self.get_parameter('radius')
        w_val = self.get_parameter('shell_width')

        V_val = 4 * float(pi.evalf()) * R_val**2 * w_val
        V_m = V_val.to(ureg.m**3)

        return V_m, 4 * pi * self.R**2 * self.w, {
            self.R: R_val, self.w: w_val
        }

    def _compute_em_density(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        L_gamma_val = self._inputs['photon_luminosity']
        Gamma_val = self._inputs['bulk_lorentz_factor']
        V_val = self.get_parameter('volume')
        w_val = self.get_parameter('shell_width')

        u_em_val = L_gamma_val * w_val / (c_SI * Gamma_val**2 * V_val)
        u_em = u_em_val.to(ureg.GeV / ureg.cm**3)

        return u_em, self.L_gamma * self.w / (self.c_sym * self.Gamma**2 * self.V_iso), {
            self.c_sym: c_SI, self.L_gamma: L_gamma_val, self.Gamma: Gamma_val, self.w: w_val, self.V_iso: V_val
        }

    def _compute_magnetic_field(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        u_em_val = self.get_parameter('em_density')

        B_val = np.sqrt(8 * float(pi.evalf()) * u_em_val)
        B_G = B_val.to(ureg.gauss)

        return B_G, sp.sqrt(8 * pi * self.u_em), {
            self.u_em: u_em_val
        }

    # synchrotron / escape timescales: inherited from UHECRSourceModel
    # (generalized nuclear formula; advective w'/c)


# AGN jet blob model with BLR external photon field
class AGNJetModel(UHECRSourceModel):
    """AGN jet emission blob model (FSRQ/blazar) with BLR external photon field.

    Emission region: spherical blob moving with bulk Lorentz factor Γ.
    Target photon field: BLR photons (~10 eV) Doppler-boosted into the blob frame.
    """

    L_jet_sym, L_disk_sym, f_BLR_sym = symbols('L_jet L_disk f_BLR')

    SCHEMA: ClassVar[List[ParameterSchema]] = (
        # Independent copies so category mutations stay local to this class
        [ParameterSchema(p.name, p.unit, p.description, p.category, p.symbol,
                          kind=p.kind, native_frame=p.native_frame)
         for p in UHECRSourceModel.SCHEMA]
        + [
            ParameterSchema('jet_luminosity',  ureg.erg / ureg.second, 'Total jet kinetic + EM power',                     'input', L_jet_sym),
            ParameterSchema('disk_luminosity', ureg.erg / ureg.second, 'Accretion disk bolometric luminosity',             'input', L_disk_sym),
            ParameterSchema('blr_fraction',    ureg.dimensionless,     'Fraction of L_disk reprocessed by BLR (typ. 0.1)', 'input', f_BLR_sym),
        ]
    )
    _SCHEMA_MAP: ClassVar[Dict[str, ParameterSchema]] = {p.name: p for p in SCHEMA}

    def __init__(self, **inputs: Any):

        self.property_methods = {
            'radius':                     self._compute_radius,
            'shell_width':                self._compute_shell_width,
            'volume':                     self._compute_volume,
            'em_density':                 self._compute_em_density,
            'magnetic_field':             self._compute_magnetic_field,
            'synchrotron_loss_timescale': self._compute_synchrotron_loss_timescale,
            'escape_timescale':           self._compute_escape_timescale,
        }

        for p in self.SCHEMA:
            if p.name in self.property_methods:
                p.category = 'property'

        super().__init__(**inputs)

        self.target_photons = self._build_blr_spectrum()

    def _build_blr_spectrum(self):
        from .background_photon_models import target_photons_spectrum

        L_disk = self._inputs['disk_luminosity']
        f_BLR  = self._inputs['blr_fraction']
        Gamma  = self._inputs['bulk_lorentz_factor']
        L_BLR  = f_BLR * L_disk

        # Kaspi et al. (2007) BLR radius scaling: R_BLR = 0.1 pc × (L_disk / 10^46 erg/s)^0.5
        pc_cm = 3.086e18 * ureg.centimeter
        R_BLR = 0.1 * pc_cm * np.sqrt((L_disk / (1e46 * ureg.erg / ureg.second)).to(ureg.dimensionless).m)

        # Blob-frame BLR energy density boosted by Γ² (aberration of isotropic radiation field)
        U_BLR_prime = (Gamma**2 * L_BLR / (4 * float(pi.evalf()) * c_SI * R_BLR**2)).to(ureg.GeV / ureg.cm**3)

        Emin = self._inputs['photon_energy_min'].m
        Emax = self._inputs['photon_energy_max'].m
        Ebrk = self._inputs['photon_energy_brk'].m

        # si1=0 (flat below Lyα break), si2=1 (steep above) approximates quasi-monochromatic BLR
        return target_photons_spectrum(
            Emin, Emax, Ebrk,
            si1=0, si2=1,
            normal=((Emin, Emax), U_BLR_prime.m)
        )

    def _compute_radius(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        Gamma_val = self._inputs['bulk_lorentz_factor']
        tvar_val  = self._inputs['variability_timescale']
        z_val     = self._inputs['redshift']

        R_val = c_SI * Gamma_val**2 * tvar_val / (1 + z_val)
        R_cm  = R_val.to(ureg.centimeter)

        return R_cm, self.c_sym * self.Gamma**2 * self.t_var / (1 + self.z), {
            self.c_sym: c_SI, self.Gamma: Gamma_val, self.t_var: tvar_val, self.z: z_val
        }

    def _compute_shell_width(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        # Spherical blob: effective shell width equals blob radius
        R_val = self.get_parameter('radius')
        return R_val, self.R, {self.R: R_val}

    def _compute_volume(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        R_val = self.get_parameter('radius')
        V_val = (4 / 3) * float(pi.evalf()) * R_val**3
        V_cm3 = V_val.to(ureg.centimeter**3)

        return V_cm3, Rational(4, 3) * pi * self.R**3, {self.R: R_val}

    def _compute_em_density(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        L_jet_val = self._inputs['jet_luminosity']
        Gamma_val = self._inputs['bulk_lorentz_factor']
        R_val     = self.get_parameter('radius')

        # Comoving EM energy density; Γ⁴ = Γ² (Doppler) × Γ² (solid-angle boost)
        u_em_val = L_jet_val / (4 * float(pi.evalf()) * Gamma_val**4 * c_SI * R_val**2)
        u_em_GeV = u_em_val.to(ureg.GeV / ureg.cm**3)

        return u_em_GeV, self.L_jet_sym / (4 * pi * self.Gamma**4 * self.c_sym * self.R**2), {
            self.L_jet_sym: L_jet_val, self.Gamma: Gamma_val, self.c_sym: c_SI, self.R: R_val
        }

    def _compute_magnetic_field(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        u_em_val = self.get_parameter('em_density')
        B_val = np.sqrt(8 * float(pi.evalf()) * u_em_val)
        B_G   = B_val.to(ureg.gauss)

        return B_G, sp.sqrt(8 * pi * self.u_em), {self.u_em: u_em_val}

    # synchrotron / escape timescales: inherited from UHECRSourceModel —
    # the generalized nuclear formula reduces to this class's former
    # proton-correct expression at (1, 1); escape w'/c == R/c here since
    # the blob's shell_width is its radius


# GRB prompt-emission jet models of De Lia & Tamborra, JCAP 10 (2024) 054
# (arXiv:2406.14975). Engine-frame inputs (their tildes): iso_energy,
# duration, variability_timescale; comoving (primed) quantities are derived.
class InternalShockModel(UHECRSourceModel):
    """Internal-shock GRB prompt-emission model (De Lia & Tamborra 2024, Sec. 2.1).

    The burst is a train of n_shocks = t_dur/t_var identical shell collisions
    at R_IS = 2 Gamma^2 c t_var, each of comoving width w' = Gamma c t_var and
    volume V'_s = 4 pi R_IS^2 w', dissipating a comoving energy
    E'_iso,s = iso_energy / (Gamma n_shocks). Microphysics fractions:
    eps_d of the energy is dissipated, of which eps_e goes to photons
    (U_gamma = eps_d eps_e E'_iso,s / V'_s, their Eq. 2.3 normalization),
    eps_A to accelerated nuclei (their Eq. 2.2 budget U_A) and eps_B to the
    magnetic field, B' = sqrt(8 pi eps_B eps_d E'_iso,s / V'_s) (Eq. 2.1).

    Mapping onto the base machinery: em_density := U_gamma and
    baryonic_loading := eps_A/eps_e, so injection_spectrum(index=-k_index,
    injection_time=w'/c) closes on U_A, and loss_rates / max_energy /
    acceleration_rate work unchanged.

    Target photons: Band spectrum (Eq. 2.3) with alpha=-1.1, beta=-2.2,
    reference energy E'_0 = (1+z)/Gamma x 100 keV, peak from the Amati
    relation E~_p = 80 keV (eps_d eps_e iso_energy / 1e52 erg)^0.57 (engine
    frame; comoving = /Gamma), normalized to U_gamma over
    [photon_energy_min, photon_energy_max] (defaults 1e-9..1e-2 GeV comoving;
    the paper does not quote its window and the energy integral is
    peak-dominated).

    NOTE on frames: variability_timescale and duration are ENGINE-frame
    (central-engine) times, the paper's t~_var and t~_dur -- no (1+z) division
    as in OneZoneISModel; observer times are (1+z) larger.
    """

    E_iso_sym, t_dur_sym, N_sh_sym = symbols('E_iso t_dur N_shock')
    eps_d_sym, eps_e_sym, eps_A_sym, eps_B_sym, k_idx_sym = \
        symbols('epsilon_d epsilon_e epsilon_A epsilon_B k')

    SCHEMA: ClassVar[List[ParameterSchema]] = (
        # Independent copies so category mutations stay local to this class
        [ParameterSchema(p.name, p.unit, p.description, p.category, p.symbol,
                          kind=p.kind, native_frame=p.native_frame)
         for p in UHECRSourceModel.SCHEMA]
        + [
            ParameterSchema('iso_energy', ureg.erg,          'Isotropic-equivalent burst energy (engine frame)',        'input', E_iso_sym),
            ParameterSchema('duration',   ureg.second,       'Burst duration (engine frame)',                           'input', t_dur_sym, kind='time', native_frame='engine'),
            ParameterSchema('eps_d',      ureg.dimensionless, 'Fraction of jet energy dissipated',                      'input', eps_d_sym),
            ParameterSchema('eps_e',      ureg.dimensionless, 'Fraction of dissipated energy in electrons/photons',     'input', eps_e_sym),
            ParameterSchema('eps_A',      ureg.dimensionless, 'Fraction of dissipated energy in accelerated nuclei',    'input', eps_A_sym),
            ParameterSchema('eps_B',      ureg.dimensionless, 'Fraction of dissipated energy in the magnetic field',    'input', eps_B_sym),
            ParameterSchema('k_index',    ureg.dimensionless, 'Power-law index k of the accelerated-nuclei spectrum',   'input', k_idx_sym),
            ParameterSchema('n_shocks',   ureg.dimensionless, 'Number of emitting shells, t_dur/t_var',                 'input', N_sh_sym),
        ]
    )
    # this family consumes the paper's tilde (central-engine) times: the
    # copied base entry declares 'observer', so re-declare it here — pass
    # observer-frame values with frames={'variability_timescale': 'observer'}
    for _p in SCHEMA:
        if _p.name == 'variability_timescale':
            _p.native_frame = 'engine'
            _p.description = 'Variability timescale (engine frame)'
    del _p
    _SCHEMA_MAP: ClassVar[Dict[str, ParameterSchema]] = {p.name: p for p in SCHEMA}

    # Band shape parameters (Fermi-motivated, paper Sec. 2.1)
    band_alpha, band_beta = -1.1, -2.2

    def __init__(self, **inputs: Any):
        # subclasses may pre-set their own property_methods before super()
        if 'property_methods' not in self.__dict__:
            self.property_methods = {
                # ordered so each entry only needs inputs or earlier entries
                'n_shocks':          self._compute_n_shocks,
                'radius':            self._compute_radius,
                'shell_width':       self._compute_shell_width,
                'volume':            self._compute_volume,
                'em_density':        self._compute_em_density,
                'magnetic_field':    self._compute_magnetic_field,
                'baryonic_loading':  self._compute_baryonic_loading,
                'photon_energy_brk': self._compute_photon_peak,
            }
        for p in self.SCHEMA:
            if p.name in self.property_methods:
                p.category = 'property'

        inputs.setdefault('photon_energy_min', 1e-9 * ureg.GeV)
        inputs.setdefault('photon_energy_max', 1e-2 * ureg.GeV)
        super().__init__(**inputs)

        self.target_photons = self._build_photon_field()

    # -- shared helper -------------------------------------------------------
    def _shell_energy_density(self) -> pint.Quantity:
        """E'_iso,s / V'_s: comoving dissipatable energy density of one shell."""
        E_iso = self._inputs['iso_energy']
        Gamma = self._inputs['bulk_lorentz_factor']
        N_sh  = self.get_parameter('n_shocks')
        V_val = self.get_parameter('volume')
        return (E_iso / (Gamma * N_sh * V_val)).to(ureg.erg / ureg.cm**3)

    # -- schema properties ---------------------------------------------------
    def _compute_n_shocks(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        t_dur = self._inputs['duration']
        t_var = self._inputs['variability_timescale']
        N_val = (t_dur / t_var).to(ureg.dimensionless)

        return N_val, self.t_dur_sym / self.t_var, {
            self.t_dur_sym: t_dur, self.t_var: t_var
        }

    def _compute_radius(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        Gamma_val = self._inputs['bulk_lorentz_factor']
        tvar_val  = self._inputs['variability_timescale']

        R_val = 2 * c_SI * Gamma_val**2 * tvar_val
        R_cm  = R_val.to(ureg.centimeter)

        return R_cm, 2 * self.c_sym * self.Gamma**2 * self.t_var, {
            self.c_sym: c_SI, self.Gamma: Gamma_val, self.t_var: tvar_val
        }

    def _compute_shell_width(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        Gamma_val = self._inputs['bulk_lorentz_factor']
        tvar_val  = self._inputs['variability_timescale']

        w_val = (c_SI * Gamma_val * tvar_val).to(ureg.centimeter)

        return w_val, self.c_sym * self.Gamma * self.t_var, {
            self.c_sym: c_SI, self.Gamma: Gamma_val, self.t_var: tvar_val
        }

    def _compute_volume(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        R_val = self.get_parameter('radius')
        w_val = self.get_parameter('shell_width')

        V_val = (4 * float(pi.evalf()) * R_val**2 * w_val).to(ureg.centimeter**3)

        return V_val, 4 * pi * self.R**2 * self.w, {self.R: R_val, self.w: w_val}

    def _compute_em_density(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        eps_d = self._inputs['eps_d']
        eps_e = self._inputs['eps_e']

        u_val = (eps_d * eps_e * self._shell_energy_density()).to(ureg.GeV / ureg.cm**3)

        return u_val, self.eps_d_sym * self.eps_e_sym * self.E_iso_sym / (self.Gamma * self.N_sh_sym * self.V_iso), {
            self.eps_d_sym: eps_d, self.eps_e_sym: eps_e,
            self.E_iso_sym: self._inputs['iso_energy'],
            self.Gamma: self._inputs['bulk_lorentz_factor'],
            self.N_sh_sym: self.get_parameter('n_shocks'),
            self.V_iso: self.get_parameter('volume'),
        }

    def _compute_magnetic_field(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        eps_d = self._inputs['eps_d']
        eps_B = self._inputs['eps_B']

        u_B   = eps_B * eps_d * self._shell_energy_density()
        B_val = np.sqrt(8 * float(pi.evalf()) * u_B).to(ureg.gauss)

        return B_val, sp.sqrt(8 * pi * self.eps_B_sym * self.eps_d_sym * self.E_iso_sym / (self.Gamma * self.N_sh_sym * self.V_iso)), {
            self.eps_B_sym: eps_B, self.eps_d_sym: eps_d,
            self.E_iso_sym: self._inputs['iso_energy'],
            self.Gamma: self._inputs['bulk_lorentz_factor'],
            self.N_sh_sym: self.get_parameter('n_shocks'),
            self.V_iso: self.get_parameter('volume'),
        }

    def _compute_baryonic_loading(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        eps_A = self._inputs['eps_A']
        eps_e = self._inputs['eps_e']

        eta_val = (eps_A / eps_e).to(ureg.dimensionless)

        return eta_val, self.eps_A_sym / self.eps_e_sym, {
            self.eps_A_sym: eps_A, self.eps_e_sym: eps_e
        }

    def _compute_photon_peak(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        # Amati relation (engine frame), comoving peak = /Gamma
        eps_d = self._inputs['eps_d']
        eps_e = self._inputs['eps_e']
        E_iso = self._inputs['iso_energy']
        Gamma = self._inputs['bulk_lorentz_factor']

        E_gamma_iso = (eps_d * eps_e * E_iso / (1e52 * ureg.erg)).to(ureg.dimensionless).m
        Ep_engine   = 80. * ureg.keV * E_gamma_iso**0.57
        Ep_com      = (Ep_engine / Gamma).to(ureg.GeV)

        return Ep_com, sp.Float(8e-5) * (self.eps_d_sym * self.eps_e_sym * self.E_iso_sym / sp.Float(1e52))**sp.Float(0.57) / self.Gamma, {
            self.eps_d_sym: eps_d, self.eps_e_sym: eps_e,
            self.E_iso_sym: E_iso, self.Gamma: Gamma
        }

    # -- photon field --------------------------------------------------------
    def _band_reference_energy(self) -> float:
        """E'_0 = (1+z)/Gamma x 100 keV of Eq. (2.3), in GeV."""
        z_val = self._inputs['redshift'].m
        Gamma = self._inputs['bulk_lorentz_factor'].m
        return (1 + z_val) / Gamma * 1e-4

    def _build_photon_field(self):
        from .background_photon_models import band_photon_spectrum

        Emin = self._inputs['photon_energy_min'].to(ureg.GeV).m
        Emax = self._inputs['photon_energy_max'].to(ureg.GeV).m
        Ep   = self.get_parameter('photon_energy_brk').to(ureg.GeV).m
        U_g  = self.get_parameter('em_density').to(ureg.GeV / ureg.cm**3).m

        return band_photon_spectrum(Ep, self.band_alpha, self.band_beta,
                                    Emin=Emin, Emax=Emax,
                                    E0=self._band_reference_energy(),
                                    normal=((Emin, Emax), U_g))

    # -- fluence packaging (their Eq. 4.14) ----------------------------------
    # fluence_factor: inherited from UHECRSourceModel — this family's
    # engine-frame duration reaches the observer as (1+z) t_dur through
    # the schema's frame machinery.

    # synchrotron / escape timescales: inherited from UHECRSourceModel
    # (generalized nuclear formula; advective w'/c — identical to the
    # former local implementation)


class PhotosphericDissipationModel(InternalShockModel):
    """Dissipative-photosphere GRB model (De Lia & Tamborra 2024, Sec. 2.2).

    Same jet as InternalShockModel, but the target-photon field at R_IS is
    the three-component photospheric spectrum of their Eq. (2.4): a Band
    component released at the photosphere R_PH (its (R_PH/R_IS)^2 dilution
    cancels against the normalization at R_PH for equal shell widths, so it
    carries x_ph E'_iso,s/V'_s at R_IS), plus fast-cooling synchrotron (SYNC)
    and upscattered (UP) broken power laws carrying x_synch and x_up
    E'_iso,s/V'_s. Energy fractions x_ph = 0.2 (= eps_d), x_synch = 1.6e-4,
    x_up = 2e-3 follow their Sec. 2.2 (from Samuelsson et al.).

    The SYNC/UP break energies are not restated in the paper (their Ref. 33);
    they are parametrized relative to the Band peak E'_p as
    sync_breaks = (E'_p/30, E'_p/3) and up_breaks = (3 E'_p, 30 E'_p) by
    default -- both components carry <~ 1% of the photon energy, so results
    are insensitive to this choice. High-energy segment index (k+2)/2.
    """

    def __init__(self, x_ph=0.2, x_synch=1.6e-4, x_up=2e-3,
                 sync_breaks=None, up_breaks=None, **inputs: Any):
        self.x_ph, self.x_synch, self.x_up = x_ph, x_synch, x_up
        self.sync_breaks, self.up_breaks = sync_breaks, up_breaks
        super().__init__(**inputs)

    def photospheric_radius(self) -> pint.Quantity:
        """R_PH = sigma_T L~_iso / (4 pi Gamma^3 m_p c^3), their Sec. 2.2."""
        L_iso = (self._inputs['iso_energy'] / self._inputs['duration']).to(ureg.erg / ureg.second)
        Gamma = self._inputs['bulk_lorentz_factor']

        R_ph = sigma_T_SI * L_iso / (4 * float(pi.evalf()) * Gamma**3 * m_p_SI * c_SI**3)
        return R_ph.to(ureg.centimeter)

    def _build_photon_field(self):
        from .background_photon_models import (band_photon_spectrum,
                                               fastcooling_photon_spectrum)

        Emin = self._inputs['photon_energy_min'].to(ureg.GeV).m
        Emax = self._inputs['photon_energy_max'].to(ureg.GeV).m
        Ep   = self.get_parameter('photon_energy_brk').to(ureg.GeV).m
        U_0  = self._shell_energy_density().to(ureg.GeV / ureg.cm**3).m
        k    = self._inputs['k_index'].m
        sync = self.sync_breaks or (Ep / 30., Ep / 3.)
        up   = self.up_breaks or (3. * Ep, 30. * Ep)

        components = (
            band_photon_spectrum(Ep, self.band_alpha, self.band_beta,
                                 Emin=Emin, Emax=Emax,
                                 E0=self._band_reference_energy(),
                                 normal=((Emin, Emax), self.x_ph * U_0)),
            fastcooling_photon_spectrum(sync[0], sync[1], index_hi=(k + 2) / 2,
                                        Emin=Emin, Emax=Emax,
                                        normal=((Emin, Emax), self.x_synch * U_0)),
            fastcooling_photon_spectrum(up[0], up[1], index_hi=(k + 2) / 2,
                                        Emin=Emin, Emax=Emax,
                                        normal=((Emin, Emax), self.x_up * U_0)),
        )
        return lambda e: sum(f(e) for f in components)


class ICMARTModel(InternalShockModel):
    """ICMART (magnetic-reconnection) GRB model (De Lia & Tamborra 2024, Sec. 2.3).

    Same shell train and Band target photons as InternalShockModel, but the
    dissipation radius is a fixed input (their benchmark R = 1e15 cm) and the
    comoving field follows the jet magnetization sigma (their Eq. 2.6):

        B' = sqrt( 2 L~_iso sigma / ((sigma + 1) c Gamma^2 R^2) ),

    with L~_iso = iso_energy/duration. eps_B is not used (n/a in their
    Table 1). Defaults follow their Table 1 ICMART column: eps_d = 0.35,
    eps_e = 0.5, eps_A = 0.5, k_index = 2.0, sigma = 45, radius = 1e15 cm.
    """

    sigma_mag_sym = symbols('sigma')

    SCHEMA: ClassVar[List[ParameterSchema]] = (
        [ParameterSchema(p.name, p.unit, p.description, p.category, p.symbol,
                          kind=p.kind, native_frame=p.native_frame)
         for p in InternalShockModel.SCHEMA]
        + [
            ParameterSchema('sigma', ureg.dimensionless, 'Jet magnetization sigma', 'input', sigma_mag_sym),
        ]
    )
    _SCHEMA_MAP: ClassVar[Dict[str, ParameterSchema]] = {p.name: p for p in SCHEMA}

    def __init__(self, **inputs: Any):
        self.property_methods = {
            # no 'radius': the dissipation radius is a schema input here
            'n_shocks':          self._compute_n_shocks,
            'shell_width':       self._compute_shell_width,
            'volume':            self._compute_volume,
            'em_density':        self._compute_em_density,
            'magnetic_field':    self._compute_magnetic_field,
            'baryonic_loading':  self._compute_baryonic_loading,
            'photon_energy_brk': self._compute_photon_peak,
        }
        inputs.setdefault('eps_d', 0.35)
        inputs.setdefault('eps_e', 0.5)
        inputs.setdefault('eps_A', 0.5)
        inputs.setdefault('k_index', 2.0)
        inputs.setdefault('sigma', 45.)
        inputs.setdefault('radius', 1e15 * ureg.centimeter)
        super().__init__(**inputs)

    def _compute_magnetic_field(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        L_iso = (self._inputs['iso_energy'] / self._inputs['duration']).to(ureg.erg / ureg.second)
        sigma = self._inputs['sigma'].m
        Gamma = self._inputs['bulk_lorentz_factor']
        R_val = self.get_parameter('radius')

        B_val = np.sqrt(2 * L_iso * sigma / ((sigma + 1) * c_SI * Gamma**2 * R_val**2)).to(ureg.gauss)

        return B_val, sp.sqrt(2 * self.L_gamma * self.sigma_mag_sym / ((self.sigma_mag_sym + 1) * self.c_sym * self.Gamma**2 * self.R**2)), {
            self.L_gamma: L_iso, self.sigma_mag_sym: self._inputs['sigma'],
            self.c_sym: c_SI, self.Gamma: Gamma, self.R: R_val
        }


class VariablePhotonSource:
    """Source model with an evolving target photon field.

    Keeps one InteractionCore_Source in active memory. Each call to
    update() serializes the current core to disk and builds a new one
    from the supplied photon field. Past states can be swapped back in
    via recall(), which repopulates the active core in place.

    Arguments:
    ----------
    source_model : UHECRSourceModel instance with self.target_photons set
    epsrange     : (e_min, e_max) photon energy range in GeV
    xsec_model   : cross-section model instance (e.g. CRPropa_model)
    storage_dir  : directory where serialized cores are written
    """

    def __init__(self, source_model: UHECRSourceModel, epsrange, xsec_model, storage_dir: str):
        import os
        self.source_model = source_model
        self.epsrange     = epsrange
        self.xsec_model   = xsec_model
        self.storage_dir  = storage_dir
        self.history: List[dict] = []

        os.makedirs(storage_dir, exist_ok=True)

        source_model.build_core(epsrange, xsec_model)
        self.interaction_core = source_model.interaction_core

    def update(self, photon_field, label=None):
        """Replace the active photon field with a new one.

        Saves the current InteractionCore to disk, then builds a new one
        from photon_field and makes it the active core.

        Arguments:
        ----------
        photon_field : callable (energy in GeV) → photon number density
        label        : optional identifier for this state (e.g. timestamp)
        """
        import os
        from .core import InteractionCore_Source

        eps_grid    = np.logspace(np.log10(self.epsrange[0]), np.log10(self.epsrange[1]), 300)
        photon_vals = self.interaction_core.target_photons(eps_grid)

        idx  = len(self.history)
        path = os.path.join(self.storage_dir, f'state_{idx:04d}.npz')
        self.interaction_core.save(path)

        self.history.append({
            'label':           label,
            'path':            path,
            'photon_eps_grid': eps_grid,
            'photon_values':   photon_vals,
        })

        self.interaction_core = InteractionCore_Source(
            self.epsrange, photon_field, xsec_model=self.xsec_model
        )

    def recall(self, index: int, save_current: bool = True):
        """Swap the active core for a past state loaded from disk.

        Repopulates self.interaction_core in place — the existing object
        reference remains valid.

        Arguments:
        ----------
        index        : position in self.history (negative indexing supported)
        save_current : if True (default), the current active core is saved to
                       disk and appended to history before the swap, so it can
                       be recalled later. Set to False only if you are sure the
                       current state is already in history or is expendable.
        """
        import os

        if index >= len(self.history) or index < -len(self.history):
            raise IndexError(f"history index {index} out of range (len={len(self.history)})")

        if save_current:
            eps_grid    = np.logspace(np.log10(self.epsrange[0]), np.log10(self.epsrange[1]), 300)
            photon_vals = self.interaction_core.target_photons(eps_grid)

            idx  = len(self.history)
            path = os.path.join(self.storage_dir, f'state_{idx:04d}.npz')
            self.interaction_core.save(path)

            self.history.append({
                'label':           None,
                'path':            path,
                'photon_eps_grid': eps_grid,
                'photon_values':   photon_vals,
            })

        self.interaction_core.load(self.history[index]['path'])

    def compute_temporal_response(self, nucinj=(26, 56), boosts=None,
                                   distance_grid=np.logspace(-3.5, .5, 50)):
        """Compute the temporal response using the currently active core.

        Delegates to self.source_model.compute_temporal_response() with
        self.interaction_core. Results are stored on self.source_model
        as .boosts, .distances, and .spec_evol.

        Arguments:
        ----------
        nucinj        : injected species (Z, A)
        boosts        : boost grid; defaults to self.interaction_core.boosts
        distance_grid : fractional distance grid relative to source thickness
        """
        self.source_model.compute_temporal_response(
            interactions_core=self.interaction_core,
            nucinj=nucinj,
            boosts=boosts,
            distance_grid=distance_grid,
        )

    def simulate_time_evolution(self, update_response=False, nucinj=(26, 56),
                                boosts=None, distance_grid=np.logspace(-3.5, .5, 50),
                                timegridsize=1000):
        """Simulate time evolution using the currently active core.

        Delegates to self.source_model.simulate_time_evolution(). Results are
        stored on self.source_model as .conv_time_grid and .convolved.

        Arguments:
        ----------
        update_response : recompute temporal response before evolving
        nucinj          : injected species (Z, A)
        boosts          : boost grid; defaults to self.interaction_core.boosts
        distance_grid   : fractional distance grid relative to source thickness
        timegridsize    : number of time steps in the convolution
        """
        self.source_model.simulate_time_evolution(
            update_response=update_response,
            interactions_core=self.interaction_core,
            nucinj=nucinj,
            boosts=boosts,
            distance_grid=distance_grid,
            timegridsize=timegridsize,
        )

    def __len__(self):
        return len(self.history)

    def __repr__(self):
        n = len(self.history)
        label = self.history[-1]['label'] if n else None
        return (f"VariablePhotonSource(states_on_disk={n}, "
                f"last_label={label!r}, storage='{self.storage_dir}')")
