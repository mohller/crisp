"""Utility functions to model the total photonuclear cross section for any
nuclear species.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline
main_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

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

shortlived = [(2, 5), (3, 5), (5, 9)]
daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]

class Cross_Section_Model():
    def __init__(self, *args, **kwargs):
        if 'erange' not in kwargs:
            self.erange = (10, 140) # in MeV
        else:
            self.erange = kwargs['erange'] # in MeV

        # filtering function, takes nucleus, returns True if it should be included
        if 'filter_nuclei' not in kwargs:
            self.filter_nuclei = lambda nuc: True
        else:
            self.filter_nuclei = kwargs['filter_nuclei']

    def cross_section(self, *args, **kwargs):
        # To be defined in each case
        pass

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
        channels_list = [xsec_mr.channels[idx] for idx in idcs]
        
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
        
        return 2 / eps**2 * cumulative_trapezoid(cs_table, eps, initial=0)

    def energy_weighted_channels_table(self, *args, **kwargs):
        """Returns an array with energy weighted cross sections of the species 
           provided in nuclei_list, otherwise the full list of nuclei is used.
        """
        if 'eps' not in kwargs:
            eps = np.linspace(*self.erange, 100) # in MeV
        else:
            eps = kwargs['eps']

        ch_table = self.channels_table(*args, **kwargs)
        
        return 2 / eps**2 * cumulative_trapezoid(ch_table, eps, initial=0)


class GDR_atlas(Cross_Section_Model):
    """Models the Giant Dipole Resonance of a large number of nuclei.
       Data and models obtained from https://www-nds.iaea.org/PSFdatabase/atlas-gdr.html
    """
    def __init__(self, *args, channel_set=None, **kwargs):
        Cross_Section_Model.__init__(self, *args, **kwargs)

        self.slo_filename = os.path.join(main_path, 'data/gdr_parameters_exp&systematics/gdr-parameters_exp&systematics_slo.dat')
        self.slo_params = pd.read_fwf(self.slo_filename, widths=2*[4,] + 9*[9,] + [5,], header=3)
        self.slo_params.rename(columns={'#  Z':'Z'}, inplace=True)
        self.slo_params.fillna(0, inplace=True)

        self.smlo_filename = os.path.join(main_path, 'data/gdr_parameters_exp&systematics/gdr-parameters_exp&systematics_smlo.dat')
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

                self.channels.append(channels)

    def cross_section(self, eps, Z, A, nloss=None, rem=None, gdr_type='slo'):
        """Returns the cross section in mb, takes energy eps in MeV
           Works for individual channels, using the PSB coefficients.
           !! Not part of the GDR atlas !!
        """
        if nloss is None:
            if rem is not None:
                nloss = A - rem[1]
            else:
                return self.total_cross_section(eps, Z, A, gdr_type=gdr_type)

        # branchings as in PSB
        branchings = np.array([
            [.8,  .2,  0,   0,   0,    0,   0,    0,   0,    0,   0,    0,    0,    0,   0],
            [1.,   0,  0,   0,   0,    0,   0,    0,   0,    0,   0,    0,    0,    0,   0],
            [.1,  .3, .1,  .1,  .2,   .2,   0,    0,   0,    0,   0,    0,    0,    0,   0],
            [.1, .35, .1, .05, .15, .045, .04, .035, .03, .025, .02, .018, .015, .012, .01]
        ])

        if A in [3, 4]:
            f_i = branchings[0, nloss - 1]
        elif A in [2, 9]:
            f_i = branchings[1, nloss - 1]
        elif A in range(10, 23):
            f_i = branchings[2, nloss - 1]
        elif A > 22:
            f_i = branchings[3, nloss - 1]

        csec = self.total_cross_section(eps, Z, A, gdr_type=gdr_type) * f_i

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))

    def total_cross_section(self, eps, Z, A, gdr_type='slo'):
        """Cross section computed as the sum of all the exclusive cross sections
        of the channels of the given nucleus (Z, A)
        """
        phi = np.where(eps < 140, np.where(eps < 20, np.exp(-73.3 / eps),
                      np.polyval([9.3537e-9, -3.4762e-6, 4.1222e-4, -9.8343e-3, 8.3714e-2], eps)),
                      np.exp(-24.2 / eps))

        sgm_QD = 397.8 * Z * (A - Z) / A * phi * \
                 (np.sqrt(eps - 2.224, where=eps >= 2.224, out=np.zeros_like(eps)) / eps)**3

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

        csec = np.nan_to_num(sgm_GDR) + sgm_QD

        return np.where(np.logical_and(self.erange[0] <= eps, eps < self.erange[1]), csec, np.zeros_like(eps))


class PSB_model(Cross_Section_Model):
    """Models the cross sections from the Puget Stecker Bredekamp 1976 paper
       Source: https://ui.adsabs.harvard.edu/abs/1976ApJ...205..638P/abstract
    """
    def __init__(self, *args, **kwargs):
        Cross_Section_Model.__init__(self, *args, **kwargs)

        self.PSB_filename = os.path.join(main_path, 'data/PSB1976.csv')
        self.params = pd.read_csv(self.PSB_filename, header=1)
        self.params.fillna(0, inplace=True)

        self.nuclei = [nuc for nuc in list(zip(self.params.Z, self.params.A)) if self.filter_nuclei(nuc)]
        self.channels = []

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
                channels = [([Zr for Zr, Ar in self.nuclei if Ar == A-nloss][0], A-nloss) for nloss in range(1, 7)
                            if [Zr for Zr, Ar in self.nuclei if Ar == A-nloss] != []]
            elif A in range(23, 57):
                channels = [([Zr for Zr, Ar in self.nuclei if Ar == A-nloss][0], A-nloss) for nloss in range(1, 16)
                            if [Zr for Zr, Ar in self.nuclei if Ar == A-nloss] != []]

            self.channels.append(channels)

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        """The cross section as modeled in the reference to compute the
        interaction rates.
        """
        from scipy.special import erf
        params = self.params[np.logical_and(self.params['Z'] == Z, self.params['A'] == A)]

        if nloss is None:
            if rem is not None:
                nloss = A - rem[1]
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
        for _, Arem in self.channels[self.nuclei.index((Z, A))]:
            channels.append(self.cross_section(eps, Z, A, A-Arem))

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

        self.filename = os.path.join(main_path, 'data', filename)

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

        self.channels = [[(1, 1)]]
        if M in [0, 1, 2]:
            for Z, A in self.nuclei[1:]:
                channels = [([Zr for Zr, Ar in self.nuclei if Ar == A-nloss][0], A-nloss) for nloss in range(1, 16)
                            if [Zr for Zr, Ar in self.nuclei if Ar == A-nloss] != []]
                    
                self.channels.append(channels)
        elif M in [3, 4]:
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
            csec = zeta * f_i * Sigma_d * theta_plus(self.eps_mid, np.zeros_like(eps)) / (self.eps_max - self.eps_mid) # applies for all nloss values
            
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
        for _, Arem in self.channels[self.nuclei.index((Z, A))]:
            channels.append(self.cross_section(eps, Z, A, A-Arem))

        return np.sum(channels, axis=0)


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

                    if (Z-Zprod, A-Aprod) in shortlived:
                        rem_list.append((2, 4))
                    else:
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
              csec += self.cross_section(eps, Z, A, rem=prod)
        else:
            if rem in self.channels[self.nuclei.index((Z, A))]:
                channels = self.xsec_data[np.where(np.logical_and(self.xsec_data[:, 0] == Z, self.xsec_data[:, 1] == A))]

                for channel in channels:
                    small_prods = np.array(get_particle_numbers(channel[2]))
                    Zprod = small_prods.dot([Zd for Zd, _ in daughters])
                    Aprod = small_prods.dot([Ad for _, Ad in daughters])

                    if (Z-Zprod, A-Aprod) in [rem] + shortlived:
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


class Model_Rack(Cross_Section_Model):
    """A model holder that yields values from different models depending 
    on the nuclear species"""
    def __init__(self, models=None, **kwargs):
        """Populates the model set

        Arguments:
        ----------
        models: list of models to be used. 
        
        
        **Note**: The models are checked in the ordered given and if they contain 
        the requested species, then their corresponding cross section is given.
        """
        self.models = models
        
        nuclei = []
        for model in self.models:
            nuclei += model.nuclei

        self.nuclei = list(sorted(set(nuclei)))
        
        self.channels = []
        for nuc in self.nuclei:
            for model in self.models:
                if nuc in model.nuclei:
                    self.channels.append(model.channels[model.nuclei.index(nuc)])
                    break

    def cross_section(self, eps, Z, A, nloss=None, rem=None):
        for model in self.models:
            if (Z, A) in model.nuclei:
                return model.cross_section(eps, Z, A, nloss, rem)

        return np.zeros_like(eps)

    def total_cross_section(self, eps, Z, A):
        for model in self.models:
            if (Z, A) in model.nuclei:
                return model.total_cross_section(eps, Z, A)

        return np.zeros_like(eps)


def pgamma(eps_r):
    """Photonuclear cross section in the energy range .1-1e4 GeV
    taken from Rachen PhD Thesis. 

    Returns the cross section in cm2.
    """
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
    
    return mubarn_to_cm2 * (resonances(eps_r) + multipion(eps_r) + direct(eps_r))


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
        cs *= 60. * N * Z / A / np.trapz(y, x)  # MeV * mb

    # print 60.*Z*N/A, np.trapz(cs, Evals)

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

    with open('EXFOR_data/universal-spline.pkl', 'rb') as f:
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
    cs_grid = pgamma(Evals)

    if A > 1:
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        path_to_file = os.path.join(main_path, 'data/universal-spline.pkl')
        with open(path_to_file, 'rb') as f:
            tck = pickle_load(f, encoding='latin1')
    
        univ_spl = UnivariateSpline._from_tck(tck)
        
        idcs = np.argwhere((.2 < Evals) * (Evals < 1.9))  # selecting resonance regions
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
