import pint
import numpy as np
import sympy as sp
import textwrap
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, ClassVar
from dataclasses import dataclass
from sympy import symbols, Eq, pi, Rational
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
    """Schema definition with symbolic variable and physical unit"""
    name: str
    unit: pint.Unit
    description: str
    category: str
    symbol: sp.Symbol


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
        ParameterSchema("redshift", ureg.dimensionless, "Redshift of the source", "input", z),
        ParameterSchema("variability_timescale", ureg.second, "Observed variability timescale", "input", t_var),
        ParameterSchema("bulk_lorentz_factor", ureg.dimensionless, "Bulk Lorentz factor of relativistic outflow", "input", Gamma),
        ParameterSchema("photon_luminosity", ureg.erg/ureg.second, "Photon luminosity", "input", L_gamma),
        ParameterSchema("distance", ureg.centimeter, "Distance from central engine", "input", d),
        ParameterSchema("shell_width", ureg.centimeter, "Radial width of interacting shells", "input", w),
        ParameterSchema("baryonic_loading", ureg.dimensionless, r"Baryon-to-photon energy density ratio", "input", eta),
        ParameterSchema("volume", ureg.centimeter**3, "Volume of radiating region", "input", V_iso),
        ParameterSchema("em_density", ureg.erg/ureg.centimeter**3, "Electro-magnetic energy density, assuming equipartition.", "input", u_em),
        ParameterSchema("magnetic_field", ureg.gauss, "Magnetic field strength", "input", B),

        ParameterSchema("photon_energy_min", ureg.GeV, "Target photon minimal energy", "input", eph_min),
        ParameterSchema("photon_energy_max", ureg.GeV, "Target photon maximal energy", "input", eph_max),
        ParameterSchema("photon_energy_brk", ureg.GeV, "Target photon break energy", "input", eph_brk),
        
        ParameterSchema("photon_density", 1/ureg.centimeter**3, "Target photon number density", "input", n_gamma),
        ParameterSchema("proton_density", 1/ureg.centimeter**3, "Proton number density", "input", n_p),
        ParameterSchema("radius", ureg.centimeter, "Characteristic size of emission region", "input", R),
        ParameterSchema("photon_energy", ureg.eV, "Characteristic target photon energy", "input", epsilon_gamma),
        ParameterSchema("expansion_timescale", ureg.second, "Dynamical/expansion timescale", "input", t_dyn),

        # Derived Properties
        ParameterSchema("proton_larmor_radius", ureg.centimeter, "Larmor radius: r_L = E_p/(eB)", "input", r_L),
        ParameterSchema("proton_energy_max", ureg.eV, "Maximum proton energy (Hillas criterion)", "input", E_p_max),
        ParameterSchema("photopion_loss_timescale", ureg.second, "Photopion interaction timescale", "input", t_pgamma),
        ParameterSchema("synchrotron_loss_timescale", ureg.second, "Proton synchrotron loss timescale", "input", t_syn),
        ParameterSchema("escape_timescale", ureg.second, "Particle escape timescale", "input", t_esc),
        ParameterSchema("neutrino_luminosity", ureg.erg/ureg.second, "Neutrino luminosity from photopion interactions", "input", L_nu),  # NOW DEFINED
        ParameterSchema("cosmic_ray_luminosity", ureg.erg/ureg.second, "Total cosmic ray luminosity", "input", L_CR),  # NOW DEFINED
    ]

    _SCHEMA_MAP: ClassVar[Dict[str, ParameterSchema]] = {p.name: p for p in SCHEMA}

    def __init__(self, **inputs: Any):
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

        self._computed_properties: Dict[str, pint.Quantity] = {}
        self._symbolic_expressions: Dict[str, sp.Expr] = {}
        self._evaluation_mappings: Dict[str, Dict[sp.Symbol, pint.Quantity]] = {}

        self._compute_all_properties()

    def parameters(self) -> List[str]:
        return list(self._inputs.keys()) + list(self._computed_properties.keys())

    def get_parameter(self, name: str) -> pint.Quantity:
        """Access function for parameter values
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
            return self._inputs[symbol_key]
        elif symbol_key in self._computed_properties:
            return self._computed_properties[symbol_key]
        else:
            print('not found in any of the paramewters')

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

    @abstractmethod
    def _compute_synchrotron_loss_timescale(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        pass
        
    @abstractmethod
    def _compute_escape_timescale(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        pass
        
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

    def compute_temporal_response(self, interactions_core, nucinj=(26, 56), boosts=None, distance_grid=np.logspace(-3.5, .5, 50)):
        """Computes the temporal response of the nuclear densities for the source.
    
           Arguments:
           ----------
           interactions_core: instance of InteractionCore_Source 
           nucinj: the injected species, injected constantly over the variability timescale of the source.
           boosts: boost grid for the computation
           distance_grid: grid of distances for the computation, fraction relative to the total thickness of source
        """
        Zinj, Ainj = 26, 56
    
        if boosts is None:
            boosts = interactions_core.boosts
    
        traversed_length = self.get_parameter('w').to('Mpc').m
        distances = distance_grid * traversed_length 
    
        alpha, mr, tr, redtens = interactions_core.get_distribution_parameters(mass_lims=(Ainj, 0), injection_type=('only species', (Zinj, Ainj)), absorption_type=('only mass', [1]))
        spec_evol = interactions_core.species_evolution_boost_range(distances, alpha=alpha, mass_range=mr, boost_range=boosts, true_range=tr)
    
        self.distances = distances
        self.spec_evol = spec_evol
    
    def simulate_time_evolution(self, update_response=False, interactions_core=None, nucinj=(26, 56), boosts=None, distance_grid=np.logspace(-3.5, .5, 50), timegridsize=1000):
        """Computes the temporal evolution of the nuclear densities for the source.
    
           Arguments:
           ----------
           update_response: is responso should be recomputed, default False
           interactions_core: instance of InteractionCore_Source 
           nucinj: the injected species, injected constantly over the variability timescale of the source.
           boosts: boost grid for the computation
           distance_grid: grid of distances for the computation, fraction relative to the total thickness of source
        """
        if update_response:
            self.compute_temporal_response(interactions_core, nucinj, boosts, distance_grid)
        
        injection_time = self.distances[-1] / c_SI.to('Mpc/s').m
        tgrid = np.cumsum(injection_time / timegridsize * np.ones(timegridsize))
    
        regular_spec_evol = interp1d(self.distances, np.permute_dims(self.spec_evol, (2, 0, 1)), 
                                     bounds_error=False, fill_value=0)(c_SI.to('Mpc/s').m * tgrid)
    
        # define constant injection function
        Qinj = lambda t, tmax: self.get_parameter('em_density').m * np.diff(tgrid)[0] * (1 - np.heaviside(t - tmax, 1))
        q0 = Qinj(tgrid, 1)
    
        # Convolution of injection and delta-injection evolution 
        convolved = convolve(regular_spec_evol, q0[None, None, :], mode='full')
        conv_time_grid = np.append(tgrid - tgrid[0], tgrid[-1] + tgrid[:-1])
    
        self.conv_time_grid = conv_time_grid
        self.convolved = convolved


# Example: GRB photospheric
class PhotosphericModel(UHECRSourceModel):
    """Photospheric emission from colliding shells"""

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
        from background_photon_models import target_photons_spectrum

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

    def _compute_synchrotron_loss_timescale(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        B_val = self._inputs['magnetic_field']
        E_p_val = 1 * ureg.EeV
        B_T = B_val * 1e-4 * ureg.tesla
        gamma_p_val = (E_p_val / (m_p_SI * c_SI**2)).to(ureg.dimensionless)
        U_B_val = (B_T**2 / (2 * ureg.mu0)).to(ureg.joule/ureg.meter**3)
        t_s = (6 * pi.evalf() * m_p_SI * c_SI) / (sigma_T_SI * c_SI * U_B_val * gamma_p_val)
        return t_s.to(ureg.second), (6 * pi * self.m_p_sym * self.c_sym) / (self.sigma_T_sym * self.c_sym * self.U_B_sym * self.gamma_p_sym), {
            self.m_p_sym: m_p_SI, self.c_sym: c_SI, self.sigma_T_sym: sigma_T_SI, self.U_B_sym: U_B_val, self.gamma_p_sym: gamma_p_val
        }

    def _compute_escape_timescale(self) -> Tuple[pint.Quantity, sp.Expr, Dict[sp.Symbol, pint.Quantity]]:
        R_val = self._inputs['radius']
        beta_val = 1.0 * ureg.dimensionless
        R_m = R_val * 1e-2 * ureg.meter
        t_s = R_m / (beta_val * c_SI)
        return t_s.to(ureg.second), self.R / (self.beta_sym * self.c_sym), {self.R: R_val, self.beta_sym: beta_val, self.c_sym: c_SI}
