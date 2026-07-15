"""Utility functions to model the total photonuclear cross section for any
nuclear species.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline

# Resolve the data directory relative to this package, regardless of install method
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

theta_plus = lambda z, eps : np.heaviside(eps - z, 1)
theta_minus = lambda z, eps : theta_plus(-z, -eps)

def get_particle_numbers(channel):
    """Extracts the info from the channel number in CRPropa's branching files
    The channel number is a number between 1 and 1000000 where the digits
    represents the amounts of different particles produced in an interaction.
    The channel number (CN) is as follows:
    CN = nN * 100000 +
        nP * 10000 +
        nH2 * 1000 +
        nH3 * 100 +
        nHe3 * 10 +
        nHe4 * 1
    nN   : Number of neutrons
    nP   : Number of protons
    nH2  : Number of deuterium
    nH3  : Number of tritium
    nHe3 : Number of helium three
    nHe4 : Number of helium four

    The function returns the values in the following order
    [nHe4, nHe3, nH3, nH2, nP, nN]
    """

    val = channel
    digits = []
    for _ in range(6):
        val, d = divmod(val, 10)
        digits.append(d)

    return digits

daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]

class Cross_Section_Model():
    # kind of interaction the model describes; photomeson models override this
    # so Model_Rack and InteractionCore can tell the groups apart
    interaction_type = 'photodisintegration'

    def __init__(self, *args, **kwargs):
        if 'erange' not in kwargs:
            self.erange = (10, 140) # in MeV
        else:
            self.erange = kwargs['erange'] # in MeV

        # filtering function, takes nucleus, returns True if it should be included
        if 'filter_nuclei' not in kwargs:
            def selection_function(nuc):
                return True
                
            self.filter_nuclei = selection_function
        else:
            self.filter_nuclei = kwargs['filter_nuclei']

    def cross_section(self, *args, **kwargs):
        # To be defined in each case
        pass

    def _mapped_remnant(self, Z, A, nloss):
        """Remnant (Zrem, Arem) after losing nloss nucleons, mapped onto nuclides
        the cascade can handle: the model nuclide of that mass when present,
        Be8/He5 (disintegrated further by nuclear decays) for the particle-unstable
        masses 8 and 5, He4 for the untabulated stable masses 6 and 7, and a free
        proton for mass 1. Returns None when no remnant exists.
        """
        Arem = int(A - nloss)
        if Arem < 1:
            return None
        elif Arem == 1:
            return (1, 1)
        elif Arem == 5:
            return (2, 5)
        elif Arem in (6, 7):
            return (2, 4)
        elif Arem == 8:
            return (4, 8)

        candidates = [Zr for Zr, Ar in self.nuclei if Ar == Arem]
        if not candidates:
            return None
        return (candidates[0], Arem)

    def _nloss_values_for_remnant(self, Z, A, rem):
        """All nucleon losses whose mapped remnant is rem (several nloss can
        share a remnant through the mass 6, 7 -> He4 mapping)."""
        return [nloss for nloss in range(1, min(16, A))
                if self._mapped_remnant(Z, A, nloss) == tuple(rem)]

    def cross_section_table(self, *args, nuclei_list=None, **kwargs):
        """Returns an array with cross sections of the species provided
           in nuclei_list, otherwise the full list of nuclei is used.
        """
        if 'eps' not in kwargs:
            eps = np.linspace(*self.erange, 100) # in MeV
        else:
            eps = kwargs['eps']

        if nuclei_list is None:
            nuclei_list = self.nuclei

        cross_section_table = np.vstack([self.cross_section(eps, *nuc) 
                                         for nuc in nuclei_list])
        return cross_section_table

    def channels_table(self, *args, nuclei_list=None, **kwargs):
        """Returns an array with cross sections of the species provided
           in nuclei_list, otherwise the full list of nuclei is used.
        """
        if 'eps' not in kwargs:
            eps = np.linspace(*self.erange, 100) # in MeV
        else:
            eps = kwargs['eps']

        if nuclei_list is None:
            nuclei_list = self.nuclei

        idcs = [self.nuclei.index(nuc) for nuc in nuclei_list]
        channels_list = [self.channels[idx] for idx in idcs]
        
        channels_table = []
        for nuc, channels in zip(nuclei_list, channels_list):
            for rem in channels:
                channels_table.append(self.cross_section(eps, *nuc, rem=rem))

        return np.vstack(channels_table)

    def energy_weighted_cross_section_table(self, *args, **kwargs):
        """Returns an array with energy weighted cross sections of the species 
           provided in nuclei_list, otherwise the full list of nuclei is used.
        """
        if 'eps' not in kwargs:
            eps = np.linspace(*self.erange, 100) # in MeV
        else:
            eps = kwargs['eps']

        cs_table = self.cross_section_table(*args, **kwargs)
        
        return 2 / eps**2 * cumulative_trapezoid(cs_table * eps, eps, initial=0)

    def energy_weighted_channels_table(self, *args, **kwargs):
        """Returns an array with energy weighted cross sections of the species 
           provided in nuclei_list, otherwise the full list of nuclei is used.
        """
        if 'eps' not in kwargs:
            eps = np.linspace(*self.erange, 100) # in MeV
        else:
            eps = kwargs['eps']

        ch_table = self.channels_table(*args, **kwargs)
        
        return 2 / eps**2 * cumulative_trapezoid(ch_table * eps, eps, initial=0)


class GDR_atlas(Cross_Section_Model):
    """Models the Giant Dipole Resonance of a large number of nuclei.
       Data and models obtained from https://www-nds.iaea.org/PSFdatabase/atlas-gdr.html
    """
    def __init__(self, *args, channel_set=None, **kwargs):
        Cross_Section_Model.__init__(self, *args, **kwargs)

        self.slo_filename = os.path.join(_DATA_DIR, 'gdr_parameters_exp&systematics/gdr-parameters_exp&systematics_slo.dat')
        self.slo_params = pd.read_fwf(self.slo_filename, widths=2*[4,] + 9*[9,] + [5,], header=3)
        self.slo_params.rename(columns={'#  Z':'Z'}, inplace=True)
        self.slo_params.fillna(0, inplace=True)

        self.smlo_filename = os.path.join(_DATA_DIR, 'gdr_parameters_exp&systematics/gdr-parameters_exp&systematics_smlo.dat')
        self.smlo_params = pd.read_fwf(self.smlo_filename, widths=2*[4,] + 9*[9,] + [5,], header=3)
        self.smlo_params.rename(columns={'#  Z':'Z'}, inplace=True)
        self.smlo_params.fillna(0, inplace=True)

        self.nuclei = [nuc for nuc in list(zip(self.slo_params.Z, self.slo_params.A)) if self.filter_nuclei(nuc)]
        self.channels = []

        if channel_set is None:
            for Z, A in self.nuclei:
                if A == 2:
                    channels = [(1, 1)]
                elif A == 3:
                    channels = [(1, 1), (1, 2)]
                elif A == 4:
                    channels = [(1, 2), (2, 3)]
                elif A == 9:
                    channels = [(2, 4)]
                elif A in range(10, 23):
                    channels = [(Z, A-nloss) for nloss in range(1, 7)]
                else:
                    channels = [(Z, A-nloss) for nloss in range(1, 16)]

                # quasi-deuteron channel: gamma + (np pair) -> n + p, remnant
                # (Z-1, A-2) — the QD strength routes here (charge-correct,
                # emits one proton and one neutron), not through the GDR
                # branchings over pure neutron loss
                qd_rem = (Z - 1, A - 2)
                if A >= 3 and 0 <= qd_rem[0] <= qd_rem[1] and qd_rem not in channels:
                    channels.append(qd_rem)

                self.channels.append(channels)

    def cross_section(self, eps, Z, A, nloss=None, rem=None, gdr_type='slo'):
        """Returns the cross section in mb, takes energy eps in MeV.

        Per channel, following the PSB energy-region structure (!! the
        branchings are not part of the GDR atlas !!):

        - eps < 30 MeV (the giant dipole proper): the Lorentzian strength
          goes to the exclusive 1n / 2n channels (0.8 / 0.2), as in PSB's
          low-energy component — mean mass loss ~1.2 per interaction;
        - eps >= 30 MeV: the PSB high-energy multiplicity branchings over
          the neutron-loss channels (mean ~4.3 nucleons for A > 22);
        - the Levinger quasi-deuteron part is carried at all energies by its
          physical n + p channel, remnant (Z-1, A-2).

        The channel sum reproduces total_cross_section identically in every
        energy region.
        """
        if nloss is None and rem is None:
            return self.total_cross_section(eps, Z, A, gdr_type=gdr_type)

        eps = np.asarray(eps, dtype=float)
        csec = np.zeros_like(eps)

        if rem is not None:
            if tuple(rem) == (Z - 1, A - 2):
                csec = csec + self.quasi_deuteron_cross_section(eps, Z, A)
            if rem[0] == Z:
                nloss = A - rem[1]

        if nloss is not None and 1 <= nloss <= 15:
            # branchings as in PSB (the >= 30 MeV multiplicity table)
            branchings = np.array([
                [.8,  .2,  0,   0,   0,    0,   0,    0,   0,    0,   0,    0,    0,    0,   0],
                [1.,   0,  0,   0,   0,    0,   0,    0,   0,    0,   0,    0,    0,    0,   0],
                [.1,  .3, .1,  .1,  .2,   .2,   0,    0,   0,    0,   0,    0,    0,    0,   0],
                [.1, .35, .1, .05, .15, .045, .04, .035, .03, .025, .02, .018, .015, .012, .01]
            ])
            # GDR region: exclusive 1n / 2n as in PSB's low-energy component
            gdr_row = branchings[0]

            if A in [3, 4]:
                f_lo, f_hi = branchings[0, nloss - 1], branchings[0, nloss - 1]
            elif A in [2, 9]:
                f_lo, f_hi = branchings[1, nloss - 1], branchings[1, nloss - 1]
            elif A in range(10, 23):
                f_lo, f_hi = gdr_row[nloss - 1], branchings[2, nloss - 1]
            elif A > 22:
                f_lo, f_hi = gdr_row[nloss - 1], branchings[3, nloss - 1]
            else:
                f_lo, f_hi = 0.0, 0.0

            f_i = np.where(eps < 30.0, f_lo, f_hi)
            csec = csec + self.gdr_cross_section(eps, Z, A, gdr_type=gdr_type) * f_i

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))

    def quasi_deuteron_cross_section(self, eps, Z, A):
        """Levinger quasi-deuteron cross section [mb], eps in MeV — the same
        formula for every nucleus:

            sigma_QD = L (N Z / A) sigma_d(eps) f(eps),   L = 6.5

        with the free-deuteron sigma_d = 61.2 (eps - 2.224)^(3/2) / eps^3 mb
        and the Pauli-blocking factor f of Chadwick et al., PRC 44, 814
        (1991) (polynomial for 20-140 MeV, exponential branches outside).
        Unwindowed — callers apply self.erange.
        """
        eps = np.asarray(eps, dtype=float)
        phi = np.where(eps < 140, np.where(eps < 20, np.exp(-73.3 / eps),
                      np.polyval([9.3537e-9, -3.4762e-6, 4.1222e-4, -9.8343e-3, 8.3714e-2], eps)),
                      np.exp(-24.2 / eps))

        return 397.8 * Z * (A - Z) / A * phi * \
            (np.sqrt(eps - 2.224, where=eps >= 2.224, out=np.zeros_like(eps)) / eps)**3

    def gdr_cross_section(self, eps, Z, A, gdr_type='slo'):
        """The giant-dipole (Lorentzian, atlas-parametrized) part of the cross
        section [mb], eps in MeV. Unwindowed — callers apply self.erange."""
        F_SLO = lambda G, E: 2 / np.pi * eps**2 * G / ((eps**2 - E**2)**2 + (eps*G)**2) if G and E else np.zeros_like(eps)
        F_SMLO = lambda G, E: 2 / np.pi * eps**2 * (G/E*eps) / ((eps**2 - E**2)**2 + (eps*(G/E*eps))**2) if G and E else np.zeros_like(eps)

        sgm_TRK = 60 * Z * (A - Z) / A # in MeV * mb
        sgm_GDR_fun = lambda S1, F1, S2, F2: sgm_TRK * (S1 * F1 + S2 * F2)

        if gdr_type == 'slo':
            params = self.slo_params[(self.slo_params['Z']==Z) & (self.slo_params['A']==A)]
            sgm_GDR = sgm_GDR_fun(params['S1'].values, F_SLO(*params[['Wr1', 'Er1']].values.flatten()),
                                  params['S2'].values, F_SLO(*params[['Wr2', 'E2']].values.flatten()))
        elif gdr_type == 'smlo':
            params = self.smlo_params[(self.smlo_params['Z']==Z) & (self.smlo_params['A']==A)]
            sgm_GDR = sgm_GDR_fun(params['S1'].values, F_SMLO(*params[['Wr1', 'Er1']].values.flatten()),
                                  params['S2'].values, F_SMLO(*params[['Wr2', 'E2']].values.flatten()))

        return np.nan_to_num(sgm_GDR)

    def total_cross_section(self, eps, Z, A, gdr_type='slo'):
        """Total photodisintegration cross section [mb], eps in MeV: the GDR
        Lorentzians plus the Levinger quasi-deuteron term (which the channel
        sum reproduces identically: GDR x branchings + the (Z-1, A-2) QD
        channel)."""
        csec = self.gdr_cross_section(eps, Z, A, gdr_type=gdr_type) \
            + self.quasi_deuteron_cross_section(eps, Z, A)

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))


class PSB_model(Cross_Section_Model):
    """Models the cross sections from the Puget Stecker Bredekamp 1976 paper
       Source: https://ui.adsabs.harvard.edu/abs/1976ApJ...205..638P/abstract
    """
    def __init__(self, *args, **kwargs):
        Cross_Section_Model.__init__(self, *args, **kwargs)

        self.PSB_filename = os.path.join(_DATA_DIR, 'PSB1976.csv')
        self.params = pd.read_csv(self.PSB_filename, header=1)
        self.params.fillna(0, inplace=True)

        self.nuclei = [nuc for nuc in list(zip(self.params.Z, self.params.A)) if self.filter_nuclei(nuc)]

        # One channel per distinct mapped remnant (Cross_Section_Model._mapped_remnant):
        # remnant masses absent from the table are not dropped but mapped to the real
        # particle-unstable nuclides (A=8 -> Be8, A=5 -> He5, disintegrated further by
        # nuclear decays), He4 for the untabulated stable A=6,7, and a free proton for
        # A=1. cross_section(rem=...) sums all nloss leading to the same remnant.
        self.channels = []
        for Z, A in self.nuclei:
            channels = []
            for nloss in range(1, min(16, A)):
                remnant = self._mapped_remnant(Z, A, nloss)
                if remnant is not None and remnant not in channels:
                    channels.append(remnant)
            self.channels.append(channels)

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        """The cross section as modeled in the reference to compute the
        interaction rates.
        """
        from scipy.special import erf
        params = self.params[np.logical_and(self.params['Z'] == Z, self.params['A'] == A)]

        if nloss is None:
            if rem is not None:
                # several nloss values can share a remnant (valley mapping)
                nloss_values = self._nloss_values_for_remnant(Z, A, rem)
                if not nloss_values:
                    return np.zeros_like(np.asarray(eps, dtype=float))
                return np.sum([self.cross_section(eps, Z, A, nloss=nl)
                               for nl in nloss_values], axis=0)
            else:
                return self.total_cross_section(eps, Z, A)

        zeta = float(params.iloc[0]['zeta'])
        Sigma_d = 59.8 * (A - Z) * Z / A # in MeV * mb

        csec = zeta * Sigma_d * theta_plus(30, eps) / 120 # applies for all nloss values
        f_i = float(params.iloc[0][f'{nloss}'])
        csec *= f_i

        if nloss in [1, 2]:
            eps0 = float(params.iloc[0][f'eps0{nloss}'])
            xi = float(params.iloc[0][f'xi{nloss}'])
            D = float(params.iloc[0][f'Delta{nloss}'])

            if D != 0:
                W = np.sqrt(np.pi/8) * (erf( (30 - eps0) / D * np.sqrt(2)) + erf( (eps0 - 2) / D * np.sqrt(2)))
                csec += 1/W * xi * Sigma_d / D * theta_plus(2, eps) * theta_minus(30, eps) * np.exp(-2 * ((eps - eps0) / D)**2)

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))

    def total_cross_section(self, eps, Z, A):
        """Cross section computed as the sum of all the exclusive cross sections
        of the channels of the given nucleus (Z, A)
        """
        channels = []
        for remnant in self.channels[self.nuclei.index((Z, A))]:
            channels.append(self.cross_section(eps, Z, A, rem=remnant))

        return np.sum(channels, axis=0)


class SimProp_model(Cross_Section_Model):
    """Models the cross sections in accordance with SimPropv2r4
       Source: https://iopscience.iop.org/article/10.1088/1475-7516/2017/11/009
    """
    def __init__(self, *args, filename=None, M=0, **kwargs):
        """Loads one of the models defined in the code

        Arguments:
        ----------
        filename: the file containing the data (e.g. xsect_BreitWigner_TALYS-1.0.txt) 
                  by default, assumes the PSB model is used.
        M: the input parameter used in SimProp for the given file (see publication).
        """
        Cross_Section_Model.__init__(self, *args, **kwargs)

        self.M = M

        if filename is None:
            if M in [0, 1]:
                self.M = 0
                filename = 'SimProp_models_M0_M1_M2.txt' # based on table from paper on SimPropv2.4
            elif M == 2:
                filename = 'xsect_BreitWigner_TALYS-1.6.txt' # based on table from paper on SimPropv2.4
            elif M == 3:
                filename = 'xsect_BreitWigner2_TALYS-1.6.txt'
            elif M == 4:
                filename = 'xsect_Gauss2_TALYS-restored.txt'

        self.filename = os.path.join(_DATA_DIR, filename)

        with open(self.filename) as file:
            num_species, eps_mid, eps_max = [float(val) for val in file.readline().split()]
        
        self.params = np.genfromtxt(self.filename, skip_header=1)
        self.eps_mid = eps_mid
        self.eps_max = eps_max

        if self.params.shape[0] != num_species:
            print('Warning: Number of species in file does not match number of parameter lines.')

        self.nuclei = [(int(Z), int(A)) for A, Z in self.params[:, :2] if self.filter_nuclei((Z, A))]
        self.nuclei.sort()

        self.branchings = np.array([
            [.8,  .2,  0,   0,   0,    0,   0,    0,   0,    0,   0,    0,    0,    0,   0],
            [1.,   0,  0,   0,   0,    0,   0,    0,   0,    0,   0,    0,    0,    0,   0],
            [.1,  .3, .1,  .1,  .2,   .2,   0,    0,   0,    0,   0,    0,    0,    0,   0],
            [.1, .35, .1, .05, .15, .045, .04, .035, .03, .025, .02, .018, .015, .012, .01]
        ])

        if M in [0, 1, 2]:
            # One channel per distinct remnant. nloss values whose remnant mass
            # is absent from the table are not dropped (that would lose channel
            # strength and leave dead ends, e.g. Be9): the remnant is the real
            # particle-unstable nuclide where nuclear decays disintegrate it
            # further (A=8 -> Be8, A=5 -> He5), He4 for the stable but
            # untabulated A=6,7 (Stecker-Salamon chain prescription), and a
            # free proton for A=1. Since several nloss can then share one
            # remnant, cross_section(rem=...) sums all nloss leading to it.
            self.channels = []
            for Z, A in self.nuclei:
                channels = []
                for nloss in range(1, min(16, A)):
                    remnant = self._mapped_remnant(Z, A, nloss)
                    if remnant is not None and remnant not in channels:
                        channels.append(remnant)
                self.channels.append(channels)
        elif M in [3, 4]:
            self.channels = [[(1, 1)]]
            for Z, A in self.nuclei[1:]:
                channels = [([Zr for Zr, Ar in self.nuclei if Ar == A-nloss][0], A-nloss) for nloss in [1, 4]
                            if [Zr for Zr, Ar in self.nuclei if Ar == A-nloss] != []]

                self.channels.append(channels)
            
            self.channels[self.nuclei.index((4, 9))] = [(1, 1), (2, 4)]

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        """The cross section as modeled in the reference to compute the
        interaction rates.
        """
        from scipy.special import erf

        if (nloss is None):
            if rem is not None:
                if self.M in [0, 1, 2]:
                    # several nloss values can share a remnant (valley mapping)
                    nloss_values = self._nloss_values_for_remnant(Z, A, rem)
                    if not nloss_values:
                        return np.zeros_like(np.asarray(eps, dtype=float))
                    return np.sum([self.cross_section(eps, Z, A, nloss=nl)
                                   for nl in nloss_values], axis=0)
                nloss = A - rem[1]
            else:
                return self.total_cross_section(eps, Z, A)

        if A in [3, 4]:
            f_i = self.branchings[0, nloss - 1]
        elif A in [2, 9]:
            f_i = self.branchings[1, nloss - 1]
        elif A in range(10, 23):
            f_i = self.branchings[2, nloss - 1]
        elif A in range(23, 57):
            f_i = self.branchings[3, nloss - 1]
        
        params = self.params[np.logical_and(self.params[:, 1] == Z, self.params[:, 0] == A)].flatten()[2:]

        if self.M in [0, 1, 2]:
            zeta = params[-1]
            Sigma_d = 60 * (A - Z) * Z / A # in MeV * mb
            csec = zeta * f_i * Sigma_d * theta_plus(self.eps_mid, eps) / (self.eps_max - self.eps_mid) # applies for all nloss values
            
            if nloss in [1, 2]:
                eps0 = params[2 + 3*(nloss-1)]
                epsmin = params[nloss-1]
                xi = params[3 + 3*(nloss-1)]
                D = params[4 + 3*(nloss-1)]
                
                if D != 0:
                    if self.M == 2:
                        csec += xi / (1 + ((eps - eps0) / D)**2)
                    else:
                        W = np.sqrt(np.pi) / 2 * D * (erf( (self.eps_max - eps0) / D) + erf( (eps0 - epsmin) / D))
                        csec += xi * Sigma_d / W * theta_plus(epsmin, eps) * theta_minus(self.eps_mid, eps) * np.exp(-((eps - eps0) / D)**2)
        elif self.M == 3:
            t_N, h1_N, x1_N, w1_N, h2_N, x2_N, w2_N, c_N, t_a, h1_a, x1_a, w1_a, h2_a, x2_a, w2_a, c_a = params
            
            m3comp = lambda h, x, w: h / (1 + ((eps - x) / w)**2)

            if nloss == 1:
                csec = np.where(np.logical_and(t_N <= eps, eps < self.eps_mid), m3comp(h1_N, x1_N, w1_N) + m3comp(h2_N, x2_N, w2_N), np.zeros_like(eps)) + \
                       np.where(np.logical_and(self.eps_mid <= eps, eps <= self.eps_max), c_N, np.zeros_like(eps))
            elif nloss == 4:
                csec = np.where(np.logical_and(t_a <= eps, eps < self.eps_mid), m3comp(h1_a, x1_a, w1_a) + m3comp(h2_a, x2_a, w2_a), np.zeros_like(eps)) + \
                       np.where(np.logical_and(self.eps_mid <= eps, eps <= self.eps_max), c_a, np.zeros_like(eps))
            else:
                csec = np.zeros_like(eps)

        elif self.M == 4:
            t_N, h1_N, x1_N, w1_N, c_N, t_a, h1_a, x1_a, w1_a, c_a = params

            m4comp = lambda h, x, w: h * np.exp(-(eps - x)**2 / w)

            if nloss == 1:
                csec = np.where(np.logical_and(t_N <= eps, eps < self.eps_mid), m4comp(h1_N, x1_N, w1_N), np.zeros_like(eps)) + \
                       np.where(np.logical_and(self.eps_mid <= eps, eps <= self.eps_max), c_N, np.zeros_like(eps))
            elif nloss == 4:
                csec = np.where(np.logical_and(t_a <= eps, eps < self.eps_mid), m4comp(h1_a, x1_a, w1_a), np.zeros_like(eps)) + \
                       np.where(np.logical_and(self.eps_mid <= eps, eps <= self.eps_max), c_a, np.zeros_like(eps))
            else:
                csec = np.zeros_like(eps)

        csec[eps > self.eps_max] = 0

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))
    
    def total_cross_section(self, eps, Z, A):
        """Cross section computed as the sum of all the exclusive cross sections
        of the channels of the given nucleus (Z, A)
        """
        channels = []
        for remnant in self.channels[self.nuclei.index((Z, A))]:
            channels.append(self.cross_section(eps, Z, A, rem=remnant))

        return np.sum(channels, axis=0)


class TabulatedDisintegration(Cross_Section_Model):
    """Photodisintegration model from user-supplied plain-text tables:
    total cross sections per mother plus a residual (multiplicity)
    distribution — the totals + multiplicities convention of e.g. NCMC/
    TALYS-style outputs. Exclusive channels are built as one channel per
    RESIDUAL nucleus with

        sigma_channel(eps) = sigma_tot(eps) * w_res(eps) / sum_res w(eps),

    the per-energy renormalization guaranteeing exactly one residual per
    interaction (the core's conservation accounting needs exclusive
    branches); the nucleon content of each channel follows from
    Delta Z / Delta N, boost-preserving, as for all photodisintegration
    channels.

    File formats (whitespace/comma separated, '#' comments):

    totals — first non-comment line: the energy grid in MeV, e.g.
        eps_MeV  e1 e2 ... en
    then one row per mother:
        Z  A  sigma_1 ... sigma_n         [mb]

    multiplicities — one row per (mother, residual):
        Z  A  Z_res  A_res  w             (energy-independent weight)   or
        Z  A  Z_res  A_res  w_1 ... w_n   (per-energy weights, on the
                                           SAME grid as the totals)
    Rows with A_res outside (0, A) are rejected; residual weights may be
    given unnormalized (relative branchings suffice).

    Arguments:
    ----------
    totals, multiplicities : file paths (or open file-like objects).
    erange : optional (MeV) window; defaults to the table's grid span.
    """
    interaction_type = 'photodisintegration'

    @staticmethod
    def _read_rows(source):
        text = source.read() if hasattr(source, 'read') else open(source).read()
        rows = []
        for line in text.splitlines():
            line = line.split('#')[0].replace(',', ' ').strip()
            if line:
                rows.append(line.split())
        return rows

    def __init__(self, *args, totals=None, multiplicities=None, **kwargs):
        rows = self._read_rows(totals)
        head = rows[0]
        try:
            float(head[0])
            self.eps = np.array(head, dtype=float)
        except ValueError:                      # leading label on the grid
            self.eps = np.array(head[1:], dtype=float)
        rows = rows[1:]
        if 'erange' not in kwargs:
            kwargs['erange'] = (self.eps.min(), self.eps.max())
        Cross_Section_Model.__init__(self, *args, **kwargs)

        n_e = len(self.eps)
        self.totals = {}
        for r in rows:
            Z, A = int(float(r[0])), int(float(r[1]))
            if not self.filter_nuclei((Z, A)):
                continue
            sig = np.array(r[2:], dtype=float)
            if len(sig) != n_e or np.any(sig < 0):
                raise ValueError(f'bad totals row for (Z, A) = ({Z}, {A})')
            self.totals[(Z, A)] = sig

        raw = {}
        for r in self._read_rows(multiplicities):
            Z, A, Zr, Ar = (int(float(v)) for v in r[:4])
            if (Z, A) not in self.totals:
                continue
            if not (0 < Ar < A) or not (0 <= Zr <= Z + (A - Ar)):
                raise ValueError(
                    f'unphysical residual ({Zr}, {Ar}) for mother ({Z}, {A})')
            w = np.array(r[4:], dtype=float)
            if len(w) == 1:
                w = np.full(n_e, float(w[0]))
            elif len(w) != n_e:
                raise ValueError(
                    f'multiplicity row for ({Z}, {A}) -> ({Zr}, {Ar}) must '
                    f'carry 1 or {n_e} weights')
            if np.any(w < 0):
                raise ValueError('negative multiplicity')
            raw.setdefault((Z, A), {})[(Zr, Ar)] = np.asarray(w, dtype=float)

        self.nuclei, self.channels, self._weights = [], [], {}
        for za, table in sorted(raw.items()):
            wsum = np.sum(list(table.values()), axis=0)
            if not np.any(wsum > 0):
                continue
            self.nuclei.append(za)
            rems = sorted(table)
            self.channels.append(rems)
            for rem in rems:
                self._weights[(za, rem)] = np.where(
                    wsum > 0, table[rem] / np.where(wsum > 0, wsum, 1.0), 0.0)

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        eps = np.asarray(eps, dtype=float)
        if (Z, A) not in self.totals:
            return np.zeros_like(eps)
        tot = np.interp(eps, self.eps, self.totals[(Z, A)],
                        left=0.0, right=0.0)
        window = (self.erange[0] <= eps) & (eps < self.erange[1])
        if rem is None and nloss is None:
            return np.where(window, tot, 0.0)
        csec = np.zeros_like(eps)
        for r, w in self._weights.items():
            if r[0] != (Z, A):
                continue
            if (rem is not None and tuple(rem) == r[1]) or \
                    (nloss is not None and A - r[1][1] == nloss):
                csec = csec + tot * np.interp(eps, self.eps, w,
                                              left=0.0, right=0.0)
        return np.where(window, csec, 0.0)


class Inclusive_model(Cross_Section_Model):
    """Photodisintegration model from user-supplied inclusive tables, loaded
    ONLY from explicitly given file paths (files may be renamed freely;
    nothing is auto-discovered or downloaded):

    egrid — one energy per line [MeV];
    nonel — rows `nucid sigma_1 .. sigma_n`, the total (nonelastic) cross
            section per nucleus [mb], nucid = 100 A + Z;
    incl  — rows `nucid_mother nucid_product sigma_1 .. sigma_n`, the
            INCLUSIVE cross section per (mother, product) pair [mb]. The
            inclusive convention contains the multiplicity:
            sigma_incl / sigma_nonel is the energy-dependent multiplicity
            of the product — the quantity that builds the light-particle
            yields (n, p, d, t, He3, He4).

    The tables satisfy the mass-closure identity

        sum_d A_d sigma_incl,d(eps) = A * sigma_nonel(eps)

    (verified on the TALYS tables to <~1.6%, exact for A <= 9; drip-line
    mothers can leave part of sigma_nonel unaccounted — those events
    produce no channel and are conservation-neutral no-ops). Allocation is
    bookkeeping only: each event's unique HEAVY survivor (2 A_res > A)
    becomes a cascade channel with its verbatim weight
    sigma_incl / sigma_nonel; events without one (unbound residual chains:
    C-12 -> 3 He4 through Be-8, Be-9 -> n + 2 He4, full dissociation, ...)
    are routed per energy to channels on the light species, heaviest
    first, capped by each species' own inclusive multiplicity. Lighter
    co-fragments (2 A_p <= A, not one of the six light species: He-5,
    Li-5, H-4, ...) are decomposed mass/charge-exactly into light species.
    The light-species inclusive cross sections — multiplicities included,
    minus the one-per-event channel allocation, plus the co-fragment
    content — are the boost-preserving light yields (light_yield_sigma);
    closure then follows from the identity with no Delta Z / Delta N
    inference. Survivors outside the tracked set resolve inside the
    interaction core (decay chains / same-A fallback).

    Arguments:
    ----------
    egrid, nonel, incl : file paths.
    max_mass : maximal mother mass A to track (user-facing species cap;
            filter_nuclei applies on top).
    cache : write/reuse a parsed .npz next to the incl file (the raw incl
            table is ~GB text; the cache makes reloads ~seconds).
    """
    interaction_type = 'photodisintegration'
    _LIGHT_IDS = {402: 0, 302: 1, 301: 2, 201: 3, 101: 4, 100: 5}

    def __init__(self, *args, egrid=None, nonel=None, incl=None,
                 max_mass=56, cache=True, **kwargs):
        self.eps = np.loadtxt(egrid)                       # MeV
        if 'erange' not in kwargs:
            kwargs['erange'] = (self.eps.min(), self.eps.max())
        Cross_Section_Model.__init__(self, *args, **kwargs)

        data = None
        cache_f = incl + f'.crisp3_A{int(max_mass)}.npz'
        stamp = f'{os.path.getmtime(nonel):.0f}_{os.path.getmtime(incl):.0f}'
        if cache and os.path.exists(cache_f):
            d = np.load(cache_f)
            if str(d['stamp']) == stamp:
                data = d
        if data is None:
            data = self._parse(nonel, incl, max_mass)
            data['stamp'] = stamp
            if cache:
                try:
                    np.savez_compressed(cache_f, **data)
                except OSError:
                    pass

        self.totals = {}
        for m, sig in zip(data['nonel_ids'], data['nonel_sig']):
            za = (int(m) % 100, int(m) // 100)
            self.totals[za] = np.asarray(sig, dtype=float)

        self.light_inclusive_sigma = {}
        raw = {}
        for (m, prod), sig in zip(data['incl_ids'], data['incl_sig']):
            mo = (int(m) % 100, int(m) // 100)
            if mo not in self.totals:
                continue
            if int(prod) in self._LIGHT_IDS:
                self.light_inclusive_sigma.setdefault(
                    mo, np.zeros((6, len(self.eps))))[
                    self._LIGHT_IDS[int(prod)]] = sig
            else:
                raw.setdefault(mo, {})[
                    (int(prod) % 100, int(prod) // 100)] = np.asarray(
                        sig, dtype=float)

        # channel weights are the product inclusive cross sections taken
        # VERBATIM (sigma_incl,d / sigma_nonel; multiplicities included).
        # The DEFICIT sigma_nonel - sum_res sigma_res over the HEAVY
        # survivors (A_res > A/2, at most one per event) consists of events
        # whose heaviest surviving product is a LIGHT species (unbound
        # residual chains: C-12 -> 3 He4 through Be-8, Be-9 -> n + 2 He4,
        # full dissociation to nucleons, ...): it is routed PER ENERGY to
        # channels on the light species, heaviest first, each capped by the
        # species' own inclusive multiplicity, and every allocation is
        # SUBTRACTED from that species' light-yield cross section — so mass
        # closure follows from the table identity
        # sum_d A_d sigma_d = A sigma_nonel with nothing inferred and
        # nothing clipped. Whatever the tables themselves leave unclosed
        # (drip-line mothers) stays unclosed here, faithfully.
        LIGHT_ZA = {0: (2, 4), 1: (2, 3), 2: (1, 3), 3: (1, 2),
                    4: (1, 1), 5: (0, 1)}
        self.nuclei, self.channels, self._weights = [], [], {}
        self._light_yield_sigma = {}
        mothers = sorted(set(raw) | set(self.light_inclusive_sigma),
                         key=lambda t: (t[1], t[0]))
        # deficit events may only be routed to light species that are
        # themselves tracked mothers (p and n always are): a channel on an
        # untracked species would silently drop from the core tensor
        mother_set = {za for za in mothers
                      if self.filter_nuclei(za) and za in self.totals}
        for za in mothers:
            if not self.filter_nuclei(za) or za not in self.totals:
                continue
            tot = self.totals[za]
            safe = np.where(tot > 0, tot, 1.0)
            table, mid = {}, {}
            for rem, sig in raw.get(za, {}).items():
                w = np.where(tot > 0, sig / safe, 0.0)
                if 2 * rem[1] > za[1]:      # the event's unique survivor
                    table[rem] = w
                else:                       # co-fragment (He-5, Li-5, ...)
                    mid[rem] = sig
            heavy = list(table)
            wsum = (np.sum([table[rem] for rem in heavy], axis=0) if heavy
                    else np.zeros_like(tot))
            deficit = np.clip(np.where(tot > 0, 1.0 - wsum, 0.0), 0.0, None)
            sig6 = self.light_inclusive_sigma.get(za)
            ly = (sig6.copy() if sig6 is not None
                  else np.zeros((6, len(self.eps))))
            for rem, sig in mid.items():   # mass/charge-exact light content
                for li, cnt in enumerate(self._light_decomposition(*rem)):
                    if cnt:
                        ly[li] = ly[li] + cnt * sig
            if np.any(deficit > 1e-6) and sig6 is not None:
                remaining = deficit
                for li in range(6):
                    if LIGHT_ZA[li][1] >= za[1]:
                        continue
                    if li < 4 and LIGHT_ZA[li] not in mother_set:
                        continue
                    cap = np.where(tot > 0, ly[li] / safe, 0.0)
                    alloc = np.minimum(remaining, cap)
                    if np.any(alloc > 1e-9):
                        rem = LIGHT_ZA[li]
                        table[rem] = table.get(rem, 0.0) + alloc
                        ly[li] = ly[li] - alloc * tot
                        remaining = remaining - alloc
                    if not np.any(remaining > 1e-9):
                        break
            # identity closure: whatever mass/charge the tables leave
            # unbalanced per accounted event (drip-line mothers, <~1.6%
            # elsewhere) is fixed so every channel row closes exactly,
            # A * sum_ch w = sum_ch A_ch w + yields: a table shortfall is
            # topped up with free nucleons, an overshoot is trimmed
            # proportionally across the light yields, and charge is closed
            # by a mass-neutral p <-> n transfer (clipped at yields >= 0)
            A_L6 = np.array([4., 3., 3., 2., 1., 1.])[:, None]
            Z_L6 = np.array([2., 2., 1., 1., 1., 0.])[:, None]
            if table:
                dest = np.sum(list(table.values()), axis=0)
                massH = sum(rem[1] * w for rem, w in table.items())
                chrgH = sum(rem[0] * w for rem, w in table.items())
            else:
                dest = massH = chrgH = np.zeros_like(tot)
            needL = np.clip(za[1] * dest - massH, 0.0, None) * tot
            needZ = np.clip(za[0] * dest - chrgH, 0.0, None) * tot
            massL = np.sum(A_L6 * ly, axis=0)
            ly = ly * np.where(massL > needL,
                               needL / np.where(massL > 0, massL, 1.0), 1.0)
            short = np.clip(needL - np.sum(A_L6 * ly, axis=0), 0.0, None)
            addp = np.clip(needZ - np.sum(Z_L6 * ly, axis=0), 0.0, short)
            ly[4] = ly[4] + addp
            ly[5] = ly[5] + short - addp
            transfer = np.clip(needZ - np.sum(Z_L6 * ly, axis=0),
                               -ly[4], ly[5])
            ly[4] = ly[4] + transfer
            ly[5] = ly[5] - transfer
            self._light_yield_sigma[za] = ly
            if not table:
                continue
            self.nuclei.append(za)
            rems = sorted(table)
            self.channels.append(rems)
            for rem in rems:
                self._weights[(za, rem)] = table[rem]
        assert len(self.nuclei) == len(self.channels)

    @staticmethod
    def _light_decomposition(Z, A):
        """Mass- and charge-exact greedy decomposition of a (Z, A) fragment
        into counts of the six light species [He4, He3, t, d, p, n] —
        matches the physical decay of the unbound co-fragments (He-5 ->
        He4 + n, Li-5 -> He4 + p, H-4 -> t + n, Li-4 -> He3 + p, ...);
        bound co-fragments (rare, tiny sigma) lose their identity only."""
        p, n = min(Z, A), max(A - Z, 0)
        c = [0, 0, 0, 0, 0, 0]
        while p >= 2 and n >= 2:
            c[0] += 1
            p, n = p - 2, n - 2
        if p >= 2 and n >= 1:
            c[1] += 1
            p, n = p - 2, n - 1
        elif p >= 1 and n >= 2:
            c[2] += 1
            p, n = p - 1, n - 2
        elif p >= 1 and n >= 1:
            c[3] += 1
            p, n = p - 1, n - 1
        c[4] += p
        c[5] += n
        return c

    def _parse(self, nonel, incl, max_mass):
        nonel_ids, nonel_sig = [], []
        with open(nonel) as f:
            for line in f:
                tok = line.split()
                m = int(float(tok[0]))
                if m // 100 <= max_mass:
                    nonel_ids.append(m)
                    nonel_sig.append(np.array(tok[1:], dtype=float))
        incl_ids, incl_sig = [], []
        with open(incl) as f:
            for line in f:
                tok = line.split(None, 2)
                m = int(float(tok[0]))
                if m // 100 > max_mass:
                    continue
                prod = int(float(tok[1]))
                A_m, A_p = m // 100, prod // 100
                # product == mother rows are (gamma, gamma') survival
                # channels (kept: the nucleus is NOT destroyed); product
                # id 0 is the emitted photon spectrum — a different
                # secondary type, excluded from the nucleon bookkeeping
                if not (prod in self._LIGHT_IDS or A_m >= A_p >= 2):
                    continue
                incl_ids.append((m, prod))
                incl_sig.append(np.array(line.split()[2:], dtype=float))
        return dict(nonel_ids=np.array(nonel_ids),
                    nonel_sig=np.array(nonel_sig),
                    incl_ids=np.array(incl_ids),
                    incl_sig=np.array(incl_sig))

    def light_yield_sigma(self, eps, Z, A):
        """Cross sections [mb] of the six light species
        [He4, He3, t, d, p, n] emitted as boost-preserving yields, on eps
        [MeV]: the table's inclusive cross sections (multiplicities
        included) minus the one-per-event channel allocation when a light
        species is itself an event's heaviest survivor — e.g.
        Be-9 -> n + 2 He4: one He4 is the channel remnant in the main
        tensor, sigma_incl carries multiplicity 2, so the yield here is
        exactly one He4 (plus the neutron, verbatim from the table).
        Returns (6, len(eps)); None when the mother is not in the tables.
        """
        eps = np.asarray(eps, dtype=float)
        ly = self._light_yield_sigma.get((Z, A))
        if ly is None:
            return None
        out = np.vstack([np.interp(eps, self.eps, row, left=0.0, right=0.0)
                         for row in ly])
        window = (self.erange[0] <= eps) & (eps < self.erange[1])
        return np.where(window[None, :], out, 0.0)

    def light_inclusive_multiplicity(self, Z, A):
        """Energy-dependent multiplicities sigma_incl/sigma_nonel of the six
        light species [He4, He3, t, d, p, n], shape (6, n_e) on self.eps —
        None when the tables carry no light data for this mother."""
        sig = self.light_inclusive_sigma.get((Z, A))
        if sig is None:
            return None
        tot = self.totals[(Z, A)]
        return np.where(tot > 0, sig / np.where(tot > 0, tot, 1.0), 0.0)

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        eps = np.asarray(eps, dtype=float)
        if (Z, A) not in self.totals:
            return np.zeros_like(eps)
        tot = np.interp(eps, self.eps, self.totals[(Z, A)],
                        left=0.0, right=0.0)
        window = (self.erange[0] <= eps) & (eps < self.erange[1])
        if rem is None and nloss is None:
            return np.where(window, tot, 0.0)
        # channel sigmas interpolate as ONE product w * sigma_nonel on the
        # native grid — interp(w) * interp(tot) would break the linear
        # mass-closure identity off-grid (a ~1% effect at the GDR peak)
        csec = np.zeros_like(eps)
        if rem is not None:
            w = self._weights.get(((Z, A), tuple(rem)))
            if w is not None:
                csec = np.interp(eps, self.eps, w * self.totals[(Z, A)],
                                 left=0.0, right=0.0)
        else:
            for (mo, r), w in self._weights.items():
                if mo == (Z, A) and A - r[1] == nloss:
                    csec = csec + np.interp(
                        eps, self.eps, w * self.totals[(Z, A)],
                        left=0.0, right=0.0)
        return np.where(window, csec, 0.0)


class CRPropa_model(Cross_Section_Model):
    """Loads the cross sections provided with CRPropa-data
       Source: https://iopscience.iop.org/article/10.1088/1475-7516/2017/11/009
    """
    def __init__(self, *args, path=None, **kwargs):
        """Loads the tabulated cross sections

        Arguments:
        ----------
        path: path to the cross section tables
        """
        Cross_Section_Model.__init__(self, *args, **kwargs)

        if np.any([name in path for name in ['PD_Talys1.8', 'PD_Talys1.9']]):
            self.tot_xsec_data = np.genfromtxt(os.path.join(path, 'xs_pd_sum.txt'))
            self.xsec_data = np.genfromtxt(os.path.join(path, 'xs_pd_thin.txt'))
        elif 'PD_external' in path:
            self.tot_xsec_data = np.genfromtxt(os.path.join(path, 'xs_sum.txt'))
            self.xsec_data = np.genfromtxt(os.path.join(path, 'xs_excl.txt'))
        else:
            self.tot_xsec_data = np.genfromtxt(os.path.join(path, 'xs_sum.txt'))
            self.xsec_data = np.genfromtxt(os.path.join(path, 'xs_thin.txt'))

        self.tot_xsec_data[:, 1] += self.tot_xsec_data[:, 0] # changing from (Z, N) to (Z, A)
        self.xsec_data[:, 1] += self.xsec_data[:, 0] # changing from (Z, N) to (Z, A)

        self.eps = np.genfromtxt(os.path.join(path, 'eps.txt'))
        self.isotopes = np.genfromtxt(os.path.join(path, 'isotopes.txt'))

        self.nuclei, self.channels = [], []
        for Z, A in [(Z, A) for Z, N, A in self.isotopes if self.filter_nuclei((Z, A))]:
            channels = self.xsec_data[np.argwhere(np.logical_and(self.xsec_data[:, 0] == Z, self.xsec_data[:, 1] == A)), 2]

            if np.any(channels):
                rem_list = []

                for channel in channels.flatten():
                    small_prods = np.array(get_particle_numbers(channel))

                    Zprod = small_prods.dot([Zd for Zd, _ in daughters])
                    Aprod = small_prods.dot([Ad for _, Ad in daughters])

                    rem_list.append((int(Z-Zprod), int(A-Aprod)))

                rem_list = sorted(list(set(rem_list)))
                self.channels.append(rem_list)
                self.nuclei.append((int(Z), int(A)))

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        """The cross section as modeled in the reference to compute the
        interaction rates.
        """
        csec = np.zeros_like(eps)

        if (nloss is None) and (rem is None):
            csec = self.total_cross_section(eps, Z, A)
        elif nloss is not None:
            csec = np.zeros_like(eps)

            for prod in self.channels[self.nuclei.index((Z, A))]:
                if nloss == prod[1]:
                    csec += self.cross_section(eps, Z, A, rem=prod)
        else:
            if rem in self.channels[self.nuclei.index((Z, A))]:
                channels = self.xsec_data[np.where(np.logical_and(self.xsec_data[:, 0] == Z, self.xsec_data[:, 1] == A))]

                for channel in channels:
                    small_prods = np.array(get_particle_numbers(channel[2]))
                    Zprod = small_prods.dot([Zd for Zd, _ in daughters])
                    Aprod = small_prods.dot([Ad for _, Ad in daughters])

                    if (Z-Zprod, A-Aprod) == rem:
                        csec += np.interp(eps, self.eps, channel[3:])
            else:
                csec = np.zeros_like(eps)

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))

    def total_cross_section(self, eps, Z, A):
        """Cross section computed as the sum of all the exclusive cross sections
        of the channels of the given nucleus (Z, A)
        """
        xs = self.tot_xsec_data[np.argwhere(np.logical_and(self.tot_xsec_data[:, 0] == Z, self.tot_xsec_data[:, 1] == A))].flatten()[2:]

        if len(xs) == 0:
            xs = np.zeros_like(self.eps)

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), np.interp(eps, self.eps, xs), np.zeros_like(eps))


def load_astrophomes(model='SingleParticleModel', path=None, auto_download=True,
                     channels=None, **model_kwargs):
    """Load a photomeson model class from the AstroPhoMes repository.

    Resolves the repository through crisp.data_download.get_astrophomes_path
    (ASTROPHOMES_PATH environment variable, explicit path, local cache, or
    GitHub download), performs the repository's import dance (its modules do
    ``from config import *``), and returns an instance of the requested model,
    ready for the Photomeson wrapper:

        xspm = Photomeson(pmm=load_astrophomes(), filter_nuclei=...)

    Arguments:
    ----------
    model : name of the model class in photomeson_lib.photomeson_models
            (e.g. 'SingleParticleModel', default; 'EmpiricalModel' for the
            improved mass scalings of Morejon et al. 2019).
    path : optional explicit repository path (else resolved as above).
    auto_download : download from GitHub when not found locally.
    channels : None keeps the model's own channel (multiplicity) table.
            'superposition' returns a hybrid: the requested model's CROSS
            SECTIONS (total and inclusive pions) with the SingleParticleModel
            A-1 channel structure — exclusive single-nucleon-loss channels
            compatible with the interaction core's per-channel conservation
            accounting. The EmpiricalModel's genuine table is INCLUSIVE
            (total multiplicity ~4 for Fe-56) and would overcount the
            destruction rate if fed to the core as exclusive branches.
            The hybrid also clips the (slightly negative) threshold artifact
            of the EmpiricalModel universal-function total cross section.
            'empirical' keeps the model's full fragment physics as exclusive
            channels the core can consume: one channel per RESIDUAL nucleus
            (the single heavy survivor per event, A_res >= max(2,
            ceil(A/2)); light fragments A <= 4 are excluded as channels and
            enter through the core's per-channel Delta Z / Delta N ejecta
            budget instead — deuterons and helium flattened to nucleons),
            with the residual weights renormalized to exactly one event so
            the summed channel rate equals the total. This carries the
            Morejon et al. (2019) mean mass loss (<Delta A> ~ 5.8 for
            Fe-56) instead of the superposition skeleton's Delta A = 1.
    **model_kwargs : forwarded to the model constructor.
    """
    import sys
    import importlib
    import importlib.util
    from .data_download import get_astrophomes_path

    repo = get_astrophomes_path(destination=path, auto_download=auto_download)

    # register the repository's config.py under the (generic) name its
    # internal `from config import *` expects, without leaving resolution
    # to whatever else sys.path might contain
    if 'config' not in sys.modules or not getattr(
            sys.modules['config'], '__file__', '').startswith(str(repo)):
        spec = importlib.util.spec_from_file_location(
            'config', os.path.join(repo, 'config.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules['config'] = module
        spec.loader.exec_module(module)

    if repo not in sys.path:
        sys.path.insert(0, repo)

    photomeson_models = importlib.import_module('photomeson_lib.photomeson_models')
    cls = getattr(photomeson_models, model)

    if channels == 'superposition':
        base_cs_nonel = cls.cs_nonel

        def cs_nonel_clipped(self, species):
            egrid, cgrid = base_cs_nonel(self, species)
            return egrid, np.clip(cgrid, 0.0, None)

        cls = type(model + '_A1Channels', (cls,), {
            '_fill_multiplicity':
                photomeson_models.SingleParticleModel._fill_multiplicity,
            'cs_nonel': cs_nonel_clipped,
        })
    elif channels == 'empirical':
        base_cs_nonel = cls.cs_nonel
        base_init = cls.__init__

        def cs_nonel_clipped(self, species):
            egrid, cgrid = base_cs_nonel(self, species)
            return egrid, np.clip(cgrid, 0.0, None)

        def init_residual(self, *a, **kw):
            base_init(self, *a, **kw)
            # residual-system channels: one heavy survivor per event; the
            # light fragments (A <= 4) are not channels — they flow through
            # the core's ejecta budget via each channel's Delta Z / Delta N
            keep = []
            for (m, prod) in self.incl_idcs:
                A_m, A_p = m // 100, prod // 100
                if A_m > A_p >= max(2, (A_m + 1) // 2):
                    keep.append((m, prod))
            # trim negligible residual cells (< 1e-3 of an event) before
            # renormalizing — fidelity loss < 0.1%, rack size ~x5 smaller
            keep = [k for k in keep
                    if float(np.asarray(self.multiplicity[k])) >= 1e-3]
            weight_sum = {}
            for (m, prod) in keep:
                w = float(np.asarray(self.multiplicity[(m, prod)]))
                weight_sum[m] = weight_sum.get(m, 0.0) + w
            for (m, prod) in keep:
                if weight_sum[m] > 0:
                    self.multiplicity[(m, prod)] = \
                        np.asarray(self.multiplicity[(m, prod)]) / weight_sum[m]
            self.incl_idcs = [k for k in keep if weight_sum[k[0]] > 0]

        cls = type(model + '_ResidualChannels', (cls,), {
            '__init__': init_residual,
            'cs_nonel': cs_nonel_clipped,
        })
    elif channels is not None:
        raise ValueError("channels must be None, 'superposition' or 'empirical'")

    return cls(**model_kwargs)


class Photomeson(Cross_Section_Model):
    interaction_type = 'photomeson'

    def __init__(self, *args, pmm=None, **kwargs):
        """Class to couple photomeson models.
        Requires a pmm argument that holds an instance of
        a photomeson model defined in AstroPhoMes.
        """
        nuc2id = lambda Z, A: A*100 + Z
        id2nuc = lambda nucid: (nucid % 100, nucid // 100)

        if 'erange' not in kwargs:
            kwargs['erange'] = (140, 1e9) # in MeV

        Cross_Section_Model.__init__(self, *args, **kwargs)

        self.pmm = pmm

        self.nuclei = [id2nuc(nid) for nid in self.pmm.nonel_idcs if self.filter_nuclei(id2nuc(nid))]

        # channels list EVERY inclusive daughter of the model: filter_nuclei
        # restricts only which mothers are tracked. Daughters that are not
        # tracked species are resolved by the interaction core at construction
        # (nuclear decay chains), like the photodisintegration models' raw
        # remnants — dropping them here would lose the interaction from this
        # model's accounting while cross_section still answers for the pair.
        self.channels = []
        for Z, A in self.nuclei:
            if A == 2:
                # AstroPhoMes has no inclusive deuteron data (nonel only):
                # a single breakup channel gamma + d -> p + n (+ pion)
                self.channels.append([(1, 1)])
                continue
            daughters = []
            for nid, pid in self.pmm.incl_idcs:
                if nid == nuc2id(Z, A) and id2nuc(pid) not in daughters:
                    daughters.append(id2nuc(pid))
            self.channels.append(daughters)

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        """The cross section adapted to the photomeson model.
        """
        nuc2id = lambda Z, A: A*100 + Z
        id2nuc = lambda nucid: (nucid % 100, nucid // 100)

        csec = np.zeros_like(eps)

        if A == 2:
            # nonel total carried by the single (1, 1) channel; zero for any
            # other requested remnant (a rack sums members per channel)
            if rem is None or tuple(rem) == (1, 1) or nloss == 1:
                csec = 1e-3 * np.interp(1e-3*eps, *self.pmm.cs_nonel(nuc2id(Z, A)))
            return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))

        if (nloss is None) and (rem is None):
            csec = 1e-3 * np.interp(1e-3*eps, *self.pmm.cs_nonel(nuc2id(Z, A)))
        elif nloss is not None:
            for (nid, pid) in self.pmm.incl_idcs:
                if id2nuc(nid)[1] - id2nuc(pid)[1] == nloss:
                    csec += self.cross_section(eps, Z, A, rem=id2nuc(pid))
        else: # rem is not None
            if (nuc2id(Z, A), nuc2id(*rem)) in self.pmm.incl_idcs:
                csec += 1e-3*np.interp(1e-3*eps, *self.pmm.cs_incl(nuc2id(Z, A), nuc2id(*rem)))

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))

    def fragment_yields(self, Z, A):
        """Per-event light-fragment content of a photomeson interaction on
        (Z, A), in the core's light-species order [He4, He3, t, d, p, n],
        plus the wide (struck-nucleon) charge split.

        The mix comes from the wrapped model's own inclusive fragment data
        (multiplicity[(mother, fragment)], i.e. sigma_incl/sigma_nonel per
        species); the per-mother totals are rescaled so that the fragments
        carry EXACTLY the <Delta A> and <Delta Z> implied by the (residual)
        channel table — the ~5% tail deficit of the raw tables is absorbed
        as a normalization (A via a global scale, Z via a p <-> n
        transfer). One nucleon per event is the struck one (wide recoil
        spectrum); its charge follows the free-nucleon mix, and the
        remainder is boost-preserving. Returns None when the model carries
        no usable fragment data for this mother (light mothers A < 12,
        where the residual/fragment factorization is ill-defined) — the
        caller falls back to the Delta Z / Delta N inference.

        Returns:
        --------
        dict with 'narrow' (6 floats, per-event boost-preserving yields),
        'wide_p', 'wide_n' (struck-nucleon charge split, wide_p + wide_n
        = 1) — or None.
        """
        nucid = A * 100 + Z
        mult = getattr(self.pmm, 'multiplicity', None)
        if mult is None or A < 12:
            return None
        FRAGS = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
        frag = np.array([float(np.asarray(mult.get((nucid, a * 100 + z), 0.0)))
                         for z, a in FRAGS])
        if frag.sum() <= 0:
            return None
        A_L = np.array([4., 3., 3., 2., 1., 1.])
        Z_L = np.array([2., 2., 1., 1., 1., 0.])
        # <Delta A>, <Delta Z> implied by the channel (residual) table
        res = [(k, float(np.asarray(mult[k]))) for k in self.pmm.incl_idcs
               if k[0] == nucid]
        if not res:
            return None
        dA = sum((A - k[1] // 100) * w for k, w in res)
        dZ = sum((Z - k[1] % 100) * w for k, w in res)
        # rescale the mix to carry dA exactly; repair the charge by a
        # p <-> n transfer (keeps A fixed)
        frag = frag * (dA / (A_L @ frag))
        shift = dZ - Z_L @ frag
        frag[4] = frag[4] + shift
        frag[5] = frag[5] - shift
        if frag[4] < 0 or frag[5] < 0:      # pathological cell: give up
            return None
        # the struck nucleon (wide): one per event, free-nucleon charge mix
        n_free = frag[4] + frag[5]
        if n_free < 1.0:
            return None
        wp = frag[4] / n_free
        narrow = frag.copy()
        narrow[4] -= wp
        narrow[5] -= (1.0 - wp)
        return {'narrow': narrow, 'wide_p': wp, 'wide_n': 1.0 - wp}

    def inclusive_cross_section(self, eps, Z, A, product):
        """Inclusive cross section [mb] for a light secondary of the wrapped
        photomeson model, on eps [MeV]: product in {2: pi+, 3: pi-, 4: pi0}
        (neucosma ids). Valid for any nucleus the model knows and for the
        free nucleons (1, 1)/(0, 1); same grid, unit and erange conventions
        as cross_section. This is the accessor the interaction core's
        photomeson_scaling='inclusive' factors are built from."""
        nucid = A * 100 + Z
        csec = 1e-3 * np.interp(1e-3 * np.asarray(eps, dtype=float),
                                *self.pmm.cs_incl(nucid, product))
        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]),
                        np.clip(csec, 0.0, None), 0.0)


class Photomeson_Superposition(Cross_Section_Model):
    """Superposition photomeson model: every nucleon interacts independently,
    sigma_A = Z sigma_p + N sigma_n (p ~ n), using the parametric photomeson
    cross section (cs_photomeson, which also applies the universal-function
    refinement for nuclei in the resonance region).

    Each interaction ejects the struck nucleon, so every nucleus has a single
    channel to the mass A-1 remnant (mapped onto the supplied nuclide table by
    Cross_Section_Model._mapped_remnant, like the photodisintegration
    single-nucleon channel). Drop it into a Model_Rack next to the
    photodisintegration models:

        Model_Rack(models=(pdis_model, Photomeson_Superposition(pdis_model.nuclei)))

    A=1 is excluded from the tracked nuclei (free nucleons are appended to the
    species separately and handled by the photomeson_rates_pn hook), but
    cross_section remains callable with (Z, A) = (1, 1) or (0, 1) for that hook.
    """
    interaction_type = 'photomeson'

    def __init__(self, nuclei, *args, **kwargs):
        """
        Arguments:
        ----------
        nuclei : nuclide list (Z, A) the model should cover, typically the
                 photodisintegration model's; A=1 entries are ignored.
        """
        if 'erange' not in kwargs:
            kwargs['erange'] = (145, 1e7)  # in MeV
        Cross_Section_Model.__init__(self, *args, **kwargs)

        self.nuclei = [(int(Z), int(A)) for Z, A in nuclei
                       if A > 1 and self.filter_nuclei((Z, A))]

        self.channels = []
        for Z, A in self.nuclei:
            remnant = self._mapped_remnant(Z, A, 1)
            self.channels.append([remnant] if remnant is not None else [])

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        """Superposition photomeson cross section in mb; eps in MeV.

        When asked for a specific remnant (rem=) or nucleon loss (nloss=),
        only this model's own channel (the mapped A-1 remnant) answers;
        other remnants get zero — essential inside a Model_Rack, whose
        cross_section sums the models for every requested channel.
        """
        eps = np.asarray(eps, dtype=float)

        if rem is not None or nloss is not None:
            if (Z, A) not in self.nuclei:
                return np.zeros_like(eps)
            own = self.channels[self.nuclei.index((Z, A))]
            asked = tuple(rem) if rem is not None else self._mapped_remnant(Z, A, nloss)
            if not own or asked != tuple(own[0]):
                return np.zeros_like(eps)

        csec = 1e27 * cs_photomeson(eps * 1e-3, A)   # cm2 -> mb
        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]),
                        csec, np.zeros_like(eps))

    def total_cross_section(self, eps, Z, A):
        return self.cross_section(eps, Z, A)


class Model_Rack(Cross_Section_Model):
    """A model holder that joins the cross sections of different models
    (photodisintegration and photomeson alike) into one interface."""
    def __init__(self, models=None, **kwargs):
        """Populates the model set

        Arguments:
        ----------
        models: list of models to be used.

        The rack's nuclei are the union of the models' nuclei, and each
        nucleus's channels are the union of the remnants over all models
        containing it (deduplicated, in order of first appearance).
        cross_section(rem=...) sums the contributions of every model, so a
        remnant shared between models (e.g. the photodisintegration and
        photomeson single-nucleon channels) carries the combined rate.
        """
        self.models = models

        nuclei = []
        for model in self.models:
            nuclei += model.nuclei

        self.nuclei = list(sorted(set(nuclei)))

        self.channels = []
        for nuc in self.nuclei:
            channels = []
            for model in self.models:
                if nuc in model.nuclei:
                    for remnant in model.channels[model.nuclei.index(nuc)]:
                        if tuple(remnant) not in channels:
                            channels.append(tuple(remnant))
            self.channels.append(channels)

    @property
    def photodisintegration_models(self):
        return [m for m in self.models
                if getattr(m, 'interaction_type', 'photodisintegration') == 'photodisintegration']

    @property
    def photomeson_models(self):
        return [m for m in self.models
                if getattr(m, 'interaction_type', 'photodisintegration') == 'photomeson']

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        csec = np.zeros_like(eps)

        for model in self.models:
            if (Z, A) in model.nuclei:
                csec += model.cross_section(eps, Z, A, nloss, rem)

        return csec

    def total_cross_section(self, eps, Z, A):
        csec = np.zeros_like(eps)

        for model in self.models:
            if (Z, A) in model.nuclei:
                csec += model.total_cross_section(eps, Z, A)

        return csec


def pgamma_components(eps_r):
    """The proton photomeson cross section of pgamma() decomposed into its
    physical contributions (Rachen PhD Thesis parametrization):

        'resonances' : nine baryon resonances (Delta(1232) ... Delta(1950))
        'direct'     : direct (t-channel) single-pion production
        'multipion'  : the multi-pion continuum

    Returns a dict of arrays in cm2 on eps_r [GeV]; their sum is pgamma(eps_r).
    Useful to attribute interaction rates and secondary production by process
    (cf. Huemmer et al. 2010, ApJ 721, 630)."""
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

    return {'resonances': mubarn_to_cm2 * resonances(eps_r),
            'direct':     mubarn_to_cm2 * direct(eps_r),
            'multipion':  mubarn_to_cm2 * multipion(eps_r)}


def pgamma(eps_r):
    """Photonuclear cross section in the energy range .1-1e4 GeV
    taken from Rachen PhD Thesis.

    Returns the cross section in cm2 (the sum of pgamma_components)."""
    comps = pgamma_components(eps_r)
    return comps['resonances'] + comps['direct'] + comps['multipion']


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
        cs *= 60. * N * Z / A / np.trapezoid(y, x)  # MeV * mb

    # print 60.*Z*N/A, np.trapezoid(cs, Evals)

    return cs


def get_interp_response_function(epsilon, cs):
    """Returns an interpolator with the response function

    Arguments:
    ----------
    epsilon : array, photon energies in nucleus rest frame desired units
    cs      : array, cross section values corresponding to the given energies
              in desired units
    """

    f = cumulative_trapezoid(epsilon * cs, x=epsilon, initial=0) / epsilon**2

    interp_f = InterpolatedUnivariateSpline(epsilon, f, ext=1)  # ext=1 to return
    # zeros outside of range

    return (epsilon[0], epsilon[-1]), interp_f


def universal_function(energy_grid):
    """Returns the universal function on a fixed energy range
    """
    from pickle import load as pickle_load
    from scipy.interpolate import UnivariateSpline

    with open(os.path.join(_DATA_DIR, 'universal-spline.pkl'), 'rb') as f:
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
    # pgamma is a fit valid in ~0.1-1e4 GeV; outside (notably below threshold)
    # its exponential pieces produce large negative artifacts -> clip
    cs_grid = np.clip(pgamma(Evals), 0, None)

    if A > 1:
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        path_to_file = os.path.join(_DATA_DIR, 'universal-spline.pkl')
        with open(path_to_file, 'rb') as f:
            tck = pickle_load(f, encoding='latin1')
    
        univ_spl = UnivariateSpline._from_tck(tck)
        
        idcs = np.argwhere((.2 < Evals) * (Evals < 1.9))  # selecting resonance regions
        if len(idcs):
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
