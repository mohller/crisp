"""Production and testing the interaction matrices
"""

import logging
import os
import numpy as np
from math import factorial
from scipy.linalg import expm
from scipy.interpolate import interp1d
from .interaction_rates import cs_photomeson, interaction_rate_from_cross_section
from .UHECR_statistics import prepare_species_list
from astropy import units as u
from astropy.constants import c, m_p, m_n
c_in_Mpc_sec = c.to('Mpc/s').value
mp_in_GeV = (m_p * c**2).to('GeV')
mn_in_GeV = (m_n * c**2).to('GeV')
mb_to_cm2 = u.mbarn.to('cm^2')

# construction diagnostics (unmatched products, unresolved decay paths, ...)
# are logged at DEBUG level; enable with logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



def get_nucid(nuc):
    '''Utility function: returns neucos id from (Z, A) tuple
    '''
    return nuc[1] * 100 + nuc[0]

def get_ZA(nucid):
    '''Utility function: returns (Z, A) tuple from neucos id
    '''
    return (nucid % 100, nucid // 100)

def merge_marginal_rates(mrates1, mrates2):
    """Joining rates for different species
    """
    joined_rates = []
    all_products = set([(row[0], row[1]) for row in np.vstack([mrates1[:, :2], mrates2[:, :2]])])

    for Zp, Ap in all_products:
        jrate = np.zeros_like(mrates1[0, :])
        jrate[:2] = Zp, Ap
        
        nucidx = (mrates1[:, 0] == Zp) * (mrates1[:, 1] == Ap)
        if np.any(nucidx):
            jrate[2:] += mrates1[nucidx, 2:][0]
        
        nucidx = (mrates2[:, 0] == Zp) * (mrates2[:, 1] == Ap)
        if np.any(nucidx):
            jrate[2:] += mrates2[nucidx, 2:][0]

        joined_rates.append(jrate)

    return np.vstack(joined_rates)

def get_marginal_rates(nuclei, rates, boosts, branchings=None):
    """Makes a marginal rates matrix with branchings file from crpropa.
    
    If no branchings are provided, the returned matrix contains
    only rates for n or p emission with probabilities N/A and Z/A,
    and the corresponding remnants.
    """
    # He4, He3, H3, H2, p, n
    daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
    Zd = np.array([d[0] for d in daughters])
    Ad = np.array([d[1] for d in daughters])

    marginal_rates = []
    for k, spec in enumerate(nuclei):
        Z, A, N = spec[0], spec[1], spec[1]-spec[0]
        
        mrates_small = []
        mrates_large = []
        if branchings is None: # case for photopion
            total_rate = np.interp(boosts, 10**rates[:, 0], Z*rates[:, 1] + N*rates[:, 2])
            
            rates_large = np.zeros((2, 203))
            rates_large[0, :2] = Z - 1, A - 1
            rates_large[0, 2:] = total_rate * float(Z)/A
            rates_large[1, :2] = Z, A - 1
            rates_large[1, 2:] = total_rate * float(N)/A
            mrates_large.append(rates_large)
        elif branchings == 'minimal': # case for only one nucleon loss
            total_rate = rates[k]

            rates_large = np.zeros((2, 2 + len(boosts)))
            rates_large[0, :2] = Z - 1, A - 1
            rates_large[0, 2:] = total_rate * float(Z)/A
            rates_large[1, :2] = Z, A - 1
            rates_large[1, 2:] = total_rate * float(N)/A
            mrates_large.append(rates_large)
        else:
            # select all channels of a specific nucleus
            spec_branchings = branchings[(branchings[:, 0] == Z) * (branchings[:, 1] == N)]
        
            for br in spec_branchings:
                nprods = np.array(get_particle_numbers(int(br[2])))
                prods = np.array([int(np > 0) for np in nprods])

                # Creating remnant nucleus from channel
                Zrem, Arem = Z - Zd.dot(prods), A - Ad.dot(prods)

                if (Zrem, Arem) not in nuclei:
                    # Change remnant isomer. 
                    # This only affects produced protons and neutrons since
                    # the yields of other light particles do not change.
                    if (Zrem-1, Arem) in nuclei:
                        Zrem -= 1
                    elif (Zrem+1, Arem) in nuclei:
                        Zrem += 1
                    elif (Z == 3) and (A == 6):
                        # nprods = np.array(get_particle_numbers(110000))
                        # prods = np.array([int(np > 0) for np in nprods])
                        Zrem, Arem = 2, 4
                    else:
                        logger.debug('No suitable isomer found for remnant (%2d, %2d)', Zrem, Arem)
                
                # Largest fragment is not one of the small ones
                if np.any([(mr[0] == Zrem) and (mr[1] == Arem) for mr in mrates_large]):
                    idx = [j for j, mr in enumerate(mrates_large) if (mr[0] == Zrem) and (mr[1] == Arem)][0]
                    mrates_large[idx][2:] += rates[k, 2:] * br[3:]
                else:
                    rates_large = np.zeros(203)
                    rates_large[:2] = Zrem, Arem
                    rates_large[2:] = rates[k, 2:] * br[3:]
                    mrates_large.append(rates_large)
            
                if Arem <= 4:
                    all_rates_small = np.outer(prods, br[3:] * rates[k, 2:])
                    
                    for rs in all_rates_small:
                        if np.any(rs):
                            if np.any([(mr[0] == rs[0]) and (mr[1] == rs[1]) for mr in mrates_small]):
                                idx = [j for j, mr in enumerate(mrates_small) if (mr[0] == rs[0]) and (mr[1] == rs[1])][0]
                                mrates_small[idx][2:] += rs
                            else:
                                rates_small = np.zeros(203)
                                rates_small[:2] = rs[0], rs[1]
                                rates_small[2:] = rs
                                mrates_small.append(rates_small)
        
        mrates = mrates_large + mrates_small
        marginal_rates.append(np.vstack(mrates))
    
    return marginal_rates

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

def load_rates(filename):
    from pandas import read_csv, MultiIndex
    cols = [f'{i}' for i in range(201)]
    df_rates = read_csv(filename, comment='#', sep='\t', names=['Z', 'N'] + cols)

    df_rates.insert(1, 'A', df_rates['Z'] + df_rates['N'])
    df_rates.drop('N', axis=1, inplace=True)

    df_rates.index = MultiIndex.from_arrays(df_rates[['A', 'Z']].values.T)
    df_rates.sort_index(inplace=True)

    return df_rates

def load_branchings(filename):
    from pandas import concat, read_csv, MultiIndex
    cols = [f'{i}' for i in range(201)]
    df_brnch = read_csv(filename, comment='#', sep='\t', names=['Z', 'N', 'channel'] + cols)
    
    # Nuclei in file which have no decay implemented
    correction_channels = [
        # [(2, 5), (2, 4), [0, 0, 0, 0, 0, 1]],
        [(2, 6), (2, 4), [0, 0, 0, 0, 0, 2]],
        # [(3, 5), (2, 4), [0, 0, 0, 0, 1, 0]],
        [(4, 8), (2, 4), [1, 0, 0, 0, 0, 0]],
        # [(5, 9), (4, 9), [0, 0, 0, 0, 1, -1]],
        [(6, 9), (2, 4), [1, 0, 0, 0, 2, -1]],
        [(5, 12), (6, 12), [0, 0, 0, 0, -1, 1]],
        [(9, 16), (8, 16), [0, 0, 0, 0, 1, -1]],
        [(11, 20), (10, 20), [0, 0, 0, 0, 1, -1]],
        [(13, 31), (13, 30), [0, 0, 0, 0, 0, 1]],
        [(20, 39), (19, 39), [0, 0, 0, 0, 1, -1]],
        [(21, 42), (20, 42), [0, 0, 0, 0, 1, -1]],
        [(23, 46), (22, 46), [0, 0, 0, 0, 1, -1]],
        [(24, 47), (23, 47), [0, 0, 0, 0, 1, -1]],
        [(25, 50), (24, 50), [0, 0, 0, 0, 1, -1]],
    ]
    daughter_names = ['a', 'he3', 't', 'd', 'p', 'n']
    daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
    Ad = np.array([d[1] for d in daughters])
    Zd = np.array([d[0] for d in daughters])

    df_brnch.insert(1, 'A', df_brnch['Z'] + df_brnch['N'])
    df_brnch.drop('N', axis=1, inplace=True)
    
    channel_series = df_brnch.channel.apply(get_particle_numbers)
    df_brnch.drop('channel', axis=1, inplace=True)
    df_brnch.insert(2, 'channel', channel_series)

    Zr = df_brnch['channel'].apply(Zd.dot)
    Ar = df_brnch['channel'].apply(Ad.dot)

    df_brnch.insert(2, 'Ar', df_brnch['A'] - Ar)
    df_brnch.insert(2, 'Zr', df_brnch['Z'] - Zr)

    df_brnch.index = MultiIndex.from_arrays(df_brnch[['A', 'Z', 'Ar', 'Zr']].values.T)
    df_brnch.sort_index(inplace=True)

    # Replacing channels with dead ends
    for nuc0, nucr, prods in correction_channels:
        indices = df_brnch.loc[np.all(df_brnch[['Zr', 'Ar']] == nuc0, axis=1)].index

        for index in indices:
            subdataframe = df_brnch.loc[index]
            
            newindex = MultiIndex.from_arrays(np.atleast_2d([[*index[:2], *nucr[::-1]] for index in subdataframe.index]).T)

            subdataframe.index = newindex
            update_prods = lambda channel: list(np.array(channel) + np.array(prods))

            subdataframe.loc[newindex, ('Zr', 'Ar')] = nucr
            subdataframe.loc[newindex, 'channel'].apply(update_prods)

            concat([df_brnch, subdataframe])
        
        # Remove channels with dead ends
        df_brnch.drop(index=indices, inplace=True)

    # Splitting light products into individual columns
    channel_array = np.vstack(df_brnch['channel'].values)
    for k, cn in enumerate(daughter_names):
        df_brnch.insert(loc=3, column=cn, value=channel_array[:, k])
    df_brnch.drop(columns='channel', inplace=True)

    # Merging channels with the same heavy product
    merged_yields = []
    for col in daughter_names:
        df_brnch_no_channels = df_brnch.drop(columns=daughter_names)
        df_brnch_no_channels[cols] = df_brnch_no_channels.multiply(df_brnch[col].values, axis='index')[cols]
        merged_yields.append( df_brnch_no_channels )

    return df_brnch, merged_yields

def generate_photodisinteg_tables_from_cross_sections(cs_egrid, cs_array, target_photons, nboosts=41, boosts=None):
    """ Takes an array with cross sections and produces the lists of rates and light particle yields

        The cross sections should be in milibarn
    """
    from pandas import DataFrame, MultiIndex
    
    # He4, He3, H3, H2, p, n
    daughter_names = ['a', 'he3', 't', 'd', 'p', 'n']
    daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
    Zd = np.array([d[0] for d in daughters])
    Ad = np.array([d[1] for d in daughters])

    if boosts is None:
        boosts = np.logspace(5, 14, nboosts)
    
    cols = [f'{i}' for i in range(len(boosts))]
    
    all_nprods = np.vstack([get_particle_numbers(int(br_row[2])) for br_row in cs_array])
    Z, A = cs_array[:, 0], cs_array[:, 0] + cs_array[:, 1]
    Zrem, Arem = Z - all_nprods.dot(Zd), A - all_nprods.dot(Ad)

    all_rates = []
    for br_row in cs_array:
        Am = int(br_row[1]) + int(br_row[0])
        UHECR_SRFenergy = Am * boosts # in GeV

        cs_crpropa = br_row[3:] # in mb
        r_pdis = ir.interaction_rate_from_cross_section(UHECR_SRFenergy, Am,
                target_photons, cs_egrid, cs_crpropa*mb_to_cm2)  / c_in_Mpc_sec # 1 / Mpc
        
        all_rates.append(r_pdis)
        
    df_brnch_pdis = DataFrame(data=np.hstack([np.vstack([A, Z, Arem, Zrem]).T, all_nprods, np.vstack(np.abs(all_rates))]), index=MultiIndex.from_arrays(np.vstack([A, Z, Arem, Zrem])), columns=['A', 'Z', 'Ar', 'Zr'] + daughter_names + cols)
    df_rates_pdis = df_brnch_pdis.groupby(by=['A', 'Z']).sum()[cols]

    df_brnch_pdis[cols] = df_brnch_pdis.drop(columns=daughter_names).divide(df_rates_pdis.reindex(df_brnch_pdis.index, method='ffill'))[cols]
    df_brnch_pdis.fillna(0, inplace=True)

    # Merging channels with the same heavy product
    merged_yields = []
    for col in daughter_names:
        df_brnch_no_channels = df_brnch_pdis.drop(columns=daughter_names)
        df_brnch_no_channels[cols] = df_brnch_no_channels.multiply(df_brnch_pdis[col].values, axis='index')[cols]
        merged_yields.append( df_brnch_no_channels )
    
    return df_rates_pdis, df_brnch_pdis, merged_yields

def generate_photomeson_tables_from_cross_sections(nuclei, xsp, xsn, target_photons, nboosts=41, boosts=None):
    """ Takes an array with cross sections and produces the lists of rates and light particle yields

        The cross sections should be in milibarn
    """
    from pandas import DataFrame, MultiIndex
    
    # He4, He3, H3, H2, p, n
    daughter_names = ['a', 'he3', 't', 'd', 'p', 'n']
    daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]

    if boosts is None:
        boosts = np.logspace(5, 14, nboosts)
    else:
        nboosts = len(boosts)
    
    cols = [f'{i}' for i in range(len(boosts))]
    
    # Computing individual rates for proton and neutron
    pr_pmes = ir.interaction_rate_from_cross_section(boosts, 1, target_photons, 
                xsp[:, 0], xsp[:, 1]*mb_to_cm2)  / c_in_Mpc_sec # 1 / Mpc
    nr_pmes = ir.interaction_rate_from_cross_section(boosts, 1, target_photons, 
                xsn[:, 0], xsn[:, 1]*mb_to_cm2)  / c_in_Mpc_sec # 1 / Mpc

    pprates = np.zeros((len(nuclei), len(boosts)))
    for k, (Z, A) in enumerate(nuclei):
        pprates[k] = np.interp(boosts, boosts, Z * pr_pmes + (A-Z) * nr_pmes)
    
    nuc_cols = np.vstack([(A, Z) for Z, A in nuclei]) 
    df_rates_pmes = DataFrame(data=np.hstack([nuc_cols, pprates]), index=MultiIndex.from_arrays(nuc_cols.T), columns=['A', 'Z'] + cols)
    df_rates_pmes = df_rates_pmes.groupby(by=['A', 'Z']).sum()[cols]

    pmes_branchings = []
    pmes_marginal_yields = []
    for idx, (Z, A) in enumerate(nuclei):
        remnants = [(Z, A-1), (Z-1, A-1)]
        for br, (Zrem, Arem) in zip([(1-Z/A), Z/A], remnants):
            if (Zrem, Arem) in nuclei:
                pmes_branchings.append(np.hstack([A, Z, Arem, Zrem, br * pprates[idx]]))
                pmes_marginal_yields.append(np.hstack([A, Z, Arem, Zrem, 0, 0, 0, 0, Z-Zrem, A-Arem-Z+Zrem, br * np.ones(nboosts)]))

        if not np.all([rem in nuclei for rem in remnants]):
            if np.any([rem in nuclei for rem in remnants]):
                pmes_branchings[-1][4:] = pprates[idx]
                pmes_marginal_yields[-1][10:] = np.ones(nboosts)
            else:
                # No remnant in nuclei, add dummy channel with zeros
                pmes_branchings.append(np.hstack([A, Z, A-1, Z, np.zeros(nboosts)]))
                pmes_marginal_yields.append(np.hstack([A, Z, A-1, Z, 0, 0, 0, 0, 0, 0, np.zeros(nboosts)]))
    
    pmes_branchings = np.vstack(pmes_branchings)
    pmes_branchings = DataFrame(data=pmes_branchings, index=MultiIndex.from_arrays(pmes_branchings[:, :4].T), columns=['A', 'Z', 'Ar', 'Zr'] + cols)

    pmes_marginal_yields = np.vstack(pmes_marginal_yields)
    pmes_marginal_yields = DataFrame(data=pmes_marginal_yields, index=MultiIndex.from_arrays(pmes_marginal_yields[:, :4].T), columns=['A', 'Z', 'Ar', 'Zr'] + daughter_names + cols)

    # Merging channels with the same heavy product
    merged_yields = []
    for col in daughter_names:
        df_brnch_no_channels = pmes_marginal_yields.drop(columns=daughter_names)
        df_brnch_no_channels[cols] = df_brnch_no_channels.multiply(pmes_marginal_yields[col].values, axis='index')[cols]
        merged_yields.append( df_brnch_no_channels )
        
    return df_rates_pmes, pmes_branchings, merged_yields


def build_pion_prod_kernel(boosts, target_photons, inelasticity=None):
    """Compute the pion production kernel for proton and neutron parents.

    K[s, i, j] is the rate [Mpc⁻¹] at which a proton (s=0) or neutron (s=1) at
    Lorentz factor boosts[i] produces a pion at Lorentz factor boosts[j].

    The spread across pion boost bins arises from the energy-dependent mean
    inelasticity κ̄(ε'): photons at different rest-frame energies map to
    different pion boost bins even for a fixed parent boost.

    For a nucleus (Z, A) at boost boosts[i] the pion rate into boost bin j is:
        Z * K[0, i, j] + (A-Z) * K[1, i, j]

    Uses the parametric photomeson cross section (Rachen model, A=1).
    Proton and neutron are treated as equal at this level of approximation;
    supply separate callables via `inelasticity` to override.

    Arguments:
    ----------
    boosts         : 1-D array of Lorentz factors (parent boost grid)
    target_photons : callable, n_γ(ε) in GeV⁻¹ cm⁻³ with ε in GeV
    inelasticity   : None → default parametric κ̄(ε'), or callable κ̄(ε') with ε' in GeV

    Returns:
    --------
    K : ndarray, shape (2, n_boost, n_boost)
    """
    mp_GeV  = 0.939   # proton mass in GeV
    mpi_GeV = 0.140   # pion mass in GeV
    eps_th  = 0.145   # pion production threshold in nucleon rest frame (GeV)

    if inelasticity is None:
        def kappa(eps_prime):
            # SOPHIA-motivated: κ̄ ≈ 0.20 at threshold, rising to 0.50 at high ε'
            return np.minimum(0.2 + 0.07 * np.log10(np.maximum(eps_prime, eps_th) / eps_th), 0.5)
    else:
        kappa = inelasticity

    e_grid    = np.logspace(np.log10(eps_th), 4.0, 500)  # rest-frame photon energies (GeV)
    de        = np.gradient(e_grid)
    cs_vals   = cs_photomeson(e_grid, A=1)                # cm², proton ≈ neutron
    kappa_arr = kappa(e_grid)

    n_b = len(boosts)
    K   = np.zeros((2, n_b, n_b))

    for i, gamma in enumerate(boosts):
        eps_lab = e_grid / (2.0 * gamma)               # lab-frame photon energies (GeV)
        ng      = target_photons(eps_lab)               # GeV⁻¹ cm⁻³

        # differential rate contribution per rest-frame energy bin
        dR = (c_in_Mpc_sec / (2.0 * gamma)) * ng * cs_vals * de  # Mpc⁻¹

        # pion Lorentz factor produced at each rest-frame photon energy
        gamma_pi = kappa_arr * gamma * (mp_GeV / mpi_GeV)

        j_arr = np.searchsorted(boosts, gamma_pi) - 1
        valid = (j_arr >= 0) & (j_arr < n_b) & (dR > 0)

        np.add.at(K[0, i], j_arr[valid], dR[valid])
        np.add.at(K[1, i], j_arr[valid], dR[valid])   # proton ≈ neutron

    return K


def build_proton_recoil_kernel(boosts, target_photons, inelasticity=None,
                                branching_pp=2/3, branching_np=2/3):
    """Build the secondary-proton production kernel for photomeson interactions.

    K[s, i, j] is the rate [Mpc⁻¹] at which a proton (s=0) or neutron (s=1)
    at Lorentz factor boosts[i] produces a secondary proton at boosts[j].

    The secondary proton carries fraction (1−κ̄) of the parent energy:
        γ_p = (1 − κ̄(ε')) × Γ_parent

    so it always appears below the parent boost, spread across 1–4 bins by the
    energy-dependent inelasticity.

    Default branching fractions come from Δ(1232) isospin decomposition:
    - Δ⁺ (from p+γ) → p + π⁰ with probability 2/3  → branching_pp = 2/3
    - Δ⁰ (from n+γ) → p + π⁻ with probability 2/3  → branching_np = 2/3

    For a nucleus (Z, A) at boost boosts[i], secondary proton rate into boost j:
        Z * K[0, i, j] + (A-Z) * K[1, i, j]

    Uses the same parametric cross section and default inelasticity as
    build_pion_prod_kernel; both should be called with the same inelasticity
    to maintain energy conservation (pion energy + proton energy = parent energy).

    Arguments:
    ----------
    boosts         : 1-D array of Lorentz factors (parent boost grid)
    target_photons : callable, n_γ(ε) in GeV⁻¹ cm⁻³ with ε in GeV
    inelasticity   : None → default parametric κ̄(ε'), or callable κ̄(ε') in GeV
    branching_pp   : fraction of p+γ interactions yielding a secondary proton (default 2/3)
    branching_np   : fraction of n+γ interactions yielding a secondary proton (default 2/3)

    Returns:
    --------
    K : ndarray, shape (2, n_boost, n_boost)
    """
    eps_th = 0.145   # pion production threshold in nucleon rest frame (GeV)

    if inelasticity is None:
        def kappa(eps_prime):
            return np.minimum(0.2 + 0.07 * np.log10(np.maximum(eps_prime, eps_th) / eps_th), 0.5)
    else:
        kappa = inelasticity

    e_grid    = np.logspace(np.log10(eps_th), 4.0, 500)
    de        = np.gradient(e_grid)
    cs_vals   = cs_photomeson(e_grid, A=1)
    kappa_arr = kappa(e_grid)

    n_b = len(boosts)
    K   = np.zeros((2, n_b, n_b))

    for i, gamma in enumerate(boosts):
        eps_lab = e_grid / (2.0 * gamma)
        ng      = target_photons(eps_lab)

        dR = (c_in_Mpc_sec / (2.0 * gamma)) * ng * cs_vals * de   # Mpc⁻¹

        # secondary proton boost: (1-κ̄) × Γ  [mass unchanged — proton rest mass]
        gamma_p = (1.0 - kappa_arr) * gamma

        j_arr = np.searchsorted(boosts, gamma_p) - 1
        valid = (j_arr >= 0) & (j_arr < n_b) & (dR > 0)

        np.add.at(K[0, i], j_arr[valid], branching_pp * dR[valid])
        np.add.at(K[1, i], j_arr[valid], branching_np * dR[valid])

    return K


def fix_dead_end(product, rate):
    """Takes dead end nucleus (product) and computes the products of its
    disintegration and the corresponding rate.
    """
    import sys
    import os.path as path

    datapath = path.dirname(path.dirname(path.abspath(__file__)))
    sys.path.append(datapath)
    from data.nucleardecays import NuclearDataTable
    
    ndt = NuclearDataTable(path.join(datapath, 'data/nubase2016.txt'))
    decaydata = ndt.prepare_decay_table()

    final_products = [product, ]
    final_rate = rate

    prodid = get_nucid(product)
    if prodid in decaydata:
        if len(decaydata[prodid]['channels']) > 1:
            logger.debug('Number of possible channels larger than one. Please choose a suitable selection method.')
            return None

        additional = get_ZA(decaydata[prodid]['channels'][0][1])
        new_product = (product[0] - additional[0], product[1] - additional[1])
        
        decay_length = c_in_Mpc_sec * decaydata[prodid]['decay_time']
        inter_length = np.divide(1, rate, where=rate != 0)
        final_rate = np.divide(1, (inter_length + decay_length), where=(inter_length + decay_length) != 0)

        if get_nucid(new_product) in decaydata:
            final_products, final_rate = fix_dead_end(new_product, final_rate)
            final_products.append(additional)
        else:
            final_products = [additional, new_product]

    return final_products, final_rate

def generate_decay_tables(nuclei, nboosts=41, boosts=None):
    """ Generates the disintegration tables based on published data.
        The data employed here is available at 
        https://www.anl.gov/phy/atomic-mass-data-resources
    """
    import os.path as path
    from pandas import DataFrame, MultiIndex
    from data.nucleardecays import NuclearDataTable
    datapath = path.dirname(path.dirname(path.abspath(__file__)))
    ndt = NuclearDataTable(path.join(datapath, 'data/nubase2016.txt'))
    decaydata = ndt.prepare_decay_table()

    # He4, He3, H3, H2, p, n
    daughter_names = ['a', 'he3', 't', 'd', 'p', 'n']
    daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
    Zd = np.array([d[0] for d in daughters])
    Ad = np.array([d[1] for d in daughters])

    if boosts is None:
        boosts = np.logspace(5, 14, nboosts)

    cols = [f'{i}' for i in range(len(boosts))]
    decay_rates = np.zeros((len(nuclei), len(boosts)))

    for k, (Z, A) in enumerate(nuclei):
        nucid = A*100 + Z

        if nucid in decaydata:
            tau = decaydata[nucid]['decay_time'] # in seconds
            decay_rates[k] = 1/(boosts * tau * c_in_Mpc_sec)

    df_rates_pmes = DataFrame(data=np.hstack([np.vstack(nuclei), decay_rates]), index=MultiIndex.from_arrays(np.vstack(nuclei).T), columns=['Z', 'A'] + cols)

    # pmes_branchings = []
    # pmes_marginal_yields = []
    # for idx, (Z, A) in enumerate(nuclei):
    #     remnants = [(Z, A-1), (Z-1, A-1)]
    #     for br, (Zrem, Arem) in zip([(1-Z/A), Z/A], remnants):
    #         if (Zrem, Arem) in nuclei:
    #             pmes_branchings.append(np.hstack([A, Z, Arem, Zrem, br * pprates[idx]]))
    #             pmes_marginal_yields.append(np.hstack([A, Z, Arem, Zrem, 0, 0, 0, 0, Z-Zrem, A-Arem-Z+Zrem, br * np.ones(nboosts)]))

    #     if not np.all([rem in nuclei for rem in remnants]):
    #         if np.any([rem in nuclei for rem in remnants]):
    #             pmes_branchings[-1][4:] = pprates[idx]
    #             pmes_marginal_yields[-1][10:] = np.ones(nboosts)
    #         else:
    #             # No remnant in nuclei, add dummy channel with zeros
    #             pmes_branchings.append(np.hstack([A, Z, A-1, Z, np.zeros(nboosts)]))
    #             pmes_marginal_yields.append(np.hstack([A, Z, A-1, Z, 0, 0, 0, 0, 0, 0, np.zeros(nboosts)]))
    
    # pmes_branchings = np.vstack(pmes_branchings)
    # pmes_branchings = DataFrame(data=pmes_branchings, index=MultiIndex.from_arrays(pmes_branchings[:, :4].T), columns=['A', 'Z', 'Ar', 'Zr'] + cols)

    # pmes_marginal_yields = np.vstack(pmes_marginal_yields)
    # pmes_marginal_yields = DataFrame(data=pmes_marginal_yields, index=MultiIndex.from_arrays(pmes_marginal_yields[:, :4].T), columns=['A', 'Z', 'Ar', 'Zr'] + daughter_names + cols)

    # # Merging channels with the same heavy product
    # merged_yields = []
    # for col in daughter_names:
    #     df_brnch_no_channels = pmes_marginal_yields.drop(columns=daughter_names)
    #     df_brnch_no_channels[cols] = df_brnch_no_channels.multiply(pmes_marginal_yields[col].values, axis='index')[cols]
    #     merged_yields.append( df_brnch_no_channels )
        
    return df_rates_pmes#, pmes_branchings, merged_yields

class InteractionCore():
    """Base class to produce interaction matrices
    """

    def __init__(self, nuclear_decay_On=False, ftype=np.float64, decays=None,
                 xsec_model=None, target_photons=None, photomeson=None,
                 boosts=None, eps=None, masses='nubase'):
        """
        Arguments:
        ----------
        nuclear_decay_On : if True, load the nubase decay table automatically
                 (equivalent to decays=True).
        decays : a decay table as returned by NuclearDataTable.prepare_decay_table()
                 (see examples/Nuclear_Decays.ipynb), or True to auto-load it.
                 When given, channel products that are not tracked species are
                 disintegrated through their decay chain at construction time.
        xsec_model : a Cross_Section_Model instance (e.g. PSB_model(),
                 SimProp_model(M=2), CRPropa_model(path=...)). When given, the
                 core is built directly from it and the arguments below;
                 otherwise the subclass _construct_from_files() is used as
                 before.
        target_photons : photon field n_gamma(eps) in GeV^-1 cm^-3 with eps in
                 GeV, or a list/tuple of such callables (summed, e.g.
                 [cmb, ebl]). Default: the CMB. Only used with xsec_model.
        photomeson : None, or 'kernels' to attach the parametric photomeson
                 kernels as pion_prod_tensor and proton_recoil_tensor.
                 Only used with xsec_model.
        boosts : Lorentz-factor grid. Default: np.logspace(6, 14, 201).
        eps : photon-energy grid in GeV for the rate integrals.
                 Default: 1e-3 * np.logspace(-1, 2.1, 300).
        masses : nuclear masses used for energy <-> boost conversions and
                 species_masses: 'nubase' (real ground-state masses from the
                 nuclear data table), 'legacy' (A * 0.939 GeV), or a callable
                 m(Z, A) in GeV. The photodisintegration tensors are computed
                 on the boost grid and do not depend on this choice.
        """
        self.ftype = ftype

        if decays is True or (nuclear_decay_On and decays is None):
            from .data.nucleardecays import NuclearDataTable
            decays = NuclearDataTable().prepare_decay_table()
        self.decays = decays if isinstance(decays, dict) else None

        if masses == 'nubase':
            from .data.nucleardecays import nuclear_mass_GeV
            self._mass_fn = nuclear_mass_GeV
        elif masses == 'legacy':
            self._mass_fn = lambda Z, A: A * 0.939
        elif callable(masses):
            self._mass_fn = masses
        else:
            raise ValueError("masses must be 'nubase', 'legacy' or a callable m(Z, A)")

        if xsec_model is not None:
            from .background_photon_models import cmb_photon_density_GeVcm3

            self.xsec_model = xsec_model
            self.sim_model = xsec_model    # attribute name used by the older cores

            if target_photons is None:
                target_photons = cmb_photon_density_GeVcm3
            if isinstance(target_photons, (list, tuple)):
                fields = list(target_photons)
                target_photons = lambda e: np.sum([field(e) for field in fields], axis=0)
            self.target_photons = target_photons

            self.boosts = np.logspace(6, 14, 201) if boosts is None else np.asarray(boosts)
            self.eps = 1e-3 * np.logspace(-1, 2.1, 300) if eps is None else np.asarray(eps)

            self._construct_from_xsec_model()

            if photomeson == 'kernels':
                self.pion_prod_tensor     = build_pion_prod_kernel(self.boosts, self.target_photons)
                self.proton_recoil_tensor = build_proton_recoil_kernel(self.boosts, self.target_photons)
            elif photomeson is not None:
                raise NotImplementedError("photomeson must be None or 'kernels'")
        else:
            self._construct_from_files()

        self._genenerate_complete_matrices()

    def _construct_from_xsec_model(self):
        """Build rates, branchings and light yields from self.xsec_model and
        self.target_photons on the self.boosts / self.eps grids.

        Boost-native: the rate integral only depends on the Lorentz factor,
        so no nuclear mass enters the photodisintegration tensors.
        """
        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section_boosts

        boosts, eps = self.boosts, self.eps

        pdis_rates_all, branchings_all, mlyp, mlyn = [], [], [], []
        for (Z, A), products in zip(self.xsec_model.nuclei, self.xsec_model.channels):
            branchings, lyp, lyn = [], [], []

            for Zrem, Arem in products:
                cross_section = 1e-27 * self.xsec_model.cross_section(eps * 1e3, Z, A, rem=(Zrem, Arem)) # to cm2
                pdis_rates = interaction_rate_from_cross_section_boosts(boosts, self.target_photons, eps, cross_section)
                pdis_rates /= c / parsec / 1e6 # ito Mpc

                branchings.append(np.append([Zrem, Arem], pdis_rates))
                lyp.append(np.append([Zrem, Arem], (Z - Zrem) * pdis_rates))
                lyn.append(np.append([Zrem, Arem], (A - Z - Arem + Zrem) * pdis_rates))

            mlyp.append(np.vstack(lyp) if lyp else np.array([]))
            mlyn.append(np.vstack(lyn) if lyn else np.array([]))

            pdis_rates_all.append(np.sum(np.atleast_2d(branchings), axis=0)[2:])
            branchings_all.append(branchings)

        branchings_all = [np.vstack(br) for br in branchings_all]
        marginal_light_yields = [[np.atleast_2d(np.hstack([br[:, :2], np.zeros_like(br[:, 2:])])) for br in branchings_all] for _ in range(4)]
        marginal_light_yields.append(mlyp)
        marginal_light_yields.append(mlyn)

        self.nuclei = self.xsec_model.nuclei.copy()
        self.all_rates = np.vstack(pdis_rates_all)
        self.all_branchings = branchings_all
        self.marginal_light_yields = marginal_light_yields

    @property
    def species_masses(self):
        """Masses of self.species in GeV, per the masses= argument (default: nubase)."""
        if not hasattr(self, '_species_masses'):
            self._species_masses = np.array([self._mass_fn(Z, A) for Z, A in self.species])
        return self._species_masses

    def energy_of_boost(self, species, boost):
        """Total energy E = boost * m(Z, A) in GeV for the species (Z, A)."""
        Z, A = species
        return np.asarray(boost) * self._mass_fn(Z, A)

    def boost_of_energy(self, species, energy_GeV):
        """Lorentz factor E / m(Z, A) for the species (Z, A)."""
        Z, A = species
        return np.asarray(energy_GeV) / self._mass_fn(Z, A)

    def _construct_from_files():
        """Function to load the interaction data from given files
        and produce the core matrices. It should populate the 
        following class properties:
        - boosts : a grid of boosts corresponding to the grid of rates
        - nuclei : the nuclear species (Z, A) ordered by mass and charge
        - all_rates : a 2D array where in each row corresponds to 
        the grid of rates for the nucleus with the same row index in nuclei
        - all_branchings : a list where each element contains an 
        array with the number of rows matching the number of decays of the
        corresponding nucleus, the first two values of the row are the (Zd, Ad)
        charge and mass numbers of the product with the largest mass, and the 
        remaining values are the grid of rates for the corresponding cross
        section of that channel
        - marginal_light_yields : a list of six elements corresponding to the
        production of each of the light products [He4, He3, T, D, p, n] (in that 
        order) in each of the channels contained in all_branchings. Therefore, 
        in each of the six elements, there's a list with the same structure as 
        all_branchings (each element has an array with the production of the 
        species in the corresponding index in nuclei) but for each channel, 
        while the first two values of the row are the (Zd, Ad) charge and mass 
        numbers of the product with the largest mass, the grid contains the 
        product of the number of particles of the light product yielded in the 
        channel and the branching ratio of the channel.
        """
        pass

    def species_evolution_boost_range(self, L, alpha=None, mass_range=None, boost_range=None, true_range=None):
        """Returns the probabilities of each species at positions L for a range of boosts.
        If the distances are negative and in decreasing order, it's equivalent to back propagation. 

        Arguments:
        ----------
        L : a float or an array of distances at which the pdf will be evaluated
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        true_range : range of species not part of the absorption range (excluding indices for species that are part of the absorption state)
                     if none is given, the last species in mass_range is considered the absorption state.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        """

        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor = self.interpolator(boost_range)

        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]

        # make diagonal zero
        reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])]) 
        # recompute diagonal including absorption states
        reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)
        # reduce excluding absorption states
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]

        _, c, d = reduced_tensor.shape
        t_vs_boost = np.atleast_3d(reduced_tensor.sum(axis=1))
        bigLambda = np.append(np.append(reduced_tensor, np.swapaxes(t_vs_boost, 1, 2), axis=1), np.zeros((1, c+1, d)), axis=0)

        if type(L) is np.ndarray:
            # Per-boost sub-batches avoid a scipy batched-expm precision issue
            # that appears when matrices with widely different norms share a batch.
            expmatL = np.stack([
                expm(L[:, None, None] * bigLambda[:, :, b])
                for b in range(bigLambda.shape[-1])
            ])
        else:
            expmatL = expm(np.moveaxis(bigLambda * L, -1, 0))

        total = np.matmul(np.append(alpha[indices], 0), expmatL)

        return total
    
    def light_secondaries_production(self, L, alpha=None, mass_range=None, boost_range=None, true_range=None):
        """Returns the production of each light species at positions L for a range of boosts.

        Arguments:
        ----------
        L : a float or an array of distances at which the pdf will be evaluated
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        """
        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor = self.interpolator(boost_range)
        prod_mat = self.interpyields(boost_range)

        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]
            prod_mat = prod_mat[np.ix_(range(prod_mat.shape[0]), mass_range, mass_range, range(len(boost_range)))]

        # make diagonal zero
        reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])]) 
        # recompute diagonal including absorption states
        reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)
        # reduce excluding absorption states
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]
        prod_mat = prod_mat[np.ix_(range(6), indices, indices, range(len(boost_range)))]

        # Compute production
        LamYp = prod_mat * reduced_tensor[None, :, :, :] # production rate matrix, independent of distance

        d1, _, d3, d4 = LamYp.shape
        t_vs_boost = np.atleast_3d(LamYp.sum(axis=2))
        bigLamYp = np.append(np.append(np.moveaxis(LamYp, -1, 1), np.expand_dims(np.swapaxes(t_vs_boost, 1, 2), axis=3), axis=3), 
                             np.zeros((d1, d4, 1, d3+1)), axis=2)

        P = self.species_evolution_boost_range(L, alpha, mass_range, boost_range, true_range)
        
        production = np.sum(np.einsum('lmi, klij -> klmj', P, bigLamYp), axis=3)

        return production

    def _build_light_decay_matrix(self, boosts):
        """Build the 6×6 rate matrix (row-vector convention) for light particle decays.

        Only H3 (index 2) and n (index 5) undergo significant decay at cosmological scales:
            H3 → He3 + e⁻ + ν̄    τ₀ = 12.32 yr
            n  → p   + e⁻ + ν̄    τ₀ = 880 s
        He4, He3, H2, p are treated as stable.
        """
        tau_H3 = 12.32 * 365.25 * 24 * 3600  # seconds
        tau_n  = 880.0                          # seconds

        n_b = len(boosts)
        M = np.zeros((6, 6, n_b))

        lambda_H3 = 1.0 / (boosts * tau_H3 * c_in_Mpc_sec)
        lambda_n  = 1.0 / (boosts * tau_n  * c_in_Mpc_sec)

        # H3 (index 2) → He3 (index 1)
        M[2, 1, :] = +lambda_H3
        M[2, 2, :] = -lambda_H3

        # n (index 5) → p (index 4)
        M[5, 4, :] = +lambda_n
        M[5, 5, :] = -lambda_n

        return M

    def _build_light_interaction_matrix(self, boosts):
        """Build the 6×6 photodisintegration rate matrix for light species from self.all_branchings.

        Uses the same channel data already computed in _construct_from_files for the heavy cascade.
        He4, He3, H2 photodisintegration channels are extracted directly from self.all_branchings.
        H3 has no data in most models (only radioactive decay, handled by _build_light_decay_matrix).
        p and n are treated as stable here; photomeson can be added via self.photomeson_rates_pn.

        M is a counting matrix (not a probability matrix): one He4 → He3 + n event contributes to
        both M[He4, He3] and M[He4, n], so row sums are not zero. Nucleon-number-weighted row sums
        are zero (nucleon conservation).
        """
        LIGHT_ZA = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
        n_b = len(boosts)
        M = np.zeros((6, 6, n_b))

        for li, (Zi, Ai) in enumerate(LIGHT_ZA):
            if (Zi, Ai) not in self.nuclei:
                continue  # H3 absent from PSB; p/n added to species but not nuclei
            sp_idx = self.nuclei.index((Zi, Ai))
            channels = self.all_branchings[sp_idx]  # (n_channels, 2 + n_boosts)

            for ch in channels:
                Zrem, Arem = int(ch[0]), int(ch[1])
                raw_rates = ch[2:]
                rates = interp1d(self.boosts, raw_rates, kind='previous',
                                 bounds_error=False, fill_value=0.0)(boosts)

                # main daughter (if it is one of the 6 light species)
                if (Zrem, Arem) in LIGHT_ZA:
                    lj = LIGHT_ZA.index((Zrem, Arem))
                    M[li, lj, :] += rates

                # free protons emitted per event
                np_ = Zi - Zrem
                if np_ > 0:
                    M[li, 4, :] += np_ * rates

                # free neutrons emitted per event
                nn_ = (Ai - Zi) - (Arem - Zrem)
                if nn_ > 0:
                    M[li, 5, :] += nn_ * rates

                # diagonal: total outflow from species li
                M[li, li, :] -= rates

        # photomeson absorption for p (index 4) and n (index 5) if precomputed at construction
        if hasattr(self, 'photomeson_rates_pn'):
            lpm = interp1d(self.boosts, self.photomeson_rates_pn, kind='previous',
                           bounds_error=False, fill_value=0.0)(boosts)
            M[4, 4, :] -= lpm
            M[5, 5, :] -= lpm

        return M

    def _build_light_matrix(self, boosts):
        """Full 6×6 rate matrix for light species: radioactive decay + photodisintegration."""
        return self._build_light_decay_matrix(boosts) + self._build_light_interaction_matrix(boosts)

    def light_cascade_production(self, L, alpha=None, mass_range=None, boost_range=None, true_range=None):
        """Returns the distribution of light species at positions L for a range of boosts,
        evolving produced light particles (H3→He3, n→p) through their disintegration channels.

        Solves the coupled heavy + light ODE jointly via matrix exponential.
        The block rate matrix is [[Λ_heavy, Y, abs], [0, M_light, 0], [0, 0, 0]]
        where Y[i,k] is the total production rate of light species k from heavy species i.

        Output shape: (6, n_boosts, n_L) for array L, or (6, n_boosts) for scalar L.

        Arguments:
        ----------
        L : float or array of distances at which the distribution will be evaluated
        alpha : injection vector (must sum to one)
        mass_range : species indices to include. Must be provided together with true_range.
        boost_range : boost values to evaluate. Full grid by default.
        true_range : subset of mass_range that are not absorption states.
        """
        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor = self.interpolator(boost_range)   # (n_sp_full, n_sp_full, n_b)
        prod_mat = self.interpyields(boost_range)          # (6, n_sp_full, n_sp_full, n_b)

        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]
            prod_mat = prod_mat[np.ix_(range(prod_mat.shape[0]), mass_range, mass_range, range(len(boost_range)))]

        # make diagonal zero, then recompute it to absorb outflow to absorbed species
        reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])])
        reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)

        # restrict to true_range (non-absorbed) species
        indices = [mass_range.index(ival) for ival in true_range]

        n_sp = len(indices)
        n_b  = len(boost_range)

        # Production coupling: Y_block[i, k, b] = total production rate of light k from species i.
        # Sum over ALL daughters (including absorbed A=1 species) so that nucleon conservation holds
        # when combined with the absorbed probability column.  Restrict parents to indices, but
        # keep all daughters within mass_range before summing.
        Y_full  = prod_mat[np.ix_(range(6), indices, range(prod_mat.shape[2]), range(n_b))].sum(axis=2)
        Y_block = np.moveaxis(Y_full, 0, 1)                  # (n_sp, 6, n_b)

        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(n_b))]
        prod_mat = prod_mat[np.ix_(range(6), indices, indices, range(n_b))]

        # Light particle evolution matrix: radioactive decay + photodisintegration
        M_light = self._build_light_matrix(boost_range)  # (6, 6, n_b)

        # Outflow from each heavy species to absorption (negative values)
        heavy_abs = reduced_tensor.sum(axis=1)               # (n_sp, n_b)

        # Full augmented matrix: (n_sp + 7) × (n_sp + 7) × n_b
        # state ordering: [heavy_0..n_sp-1, light_0..5, absorbed]
        sz = n_sp + 7
        big_mat = np.zeros((sz, sz, n_b))

        big_mat[:n_sp,          :n_sp,          :] = reduced_tensor          # heavy↔heavy
        big_mat[:n_sp,          n_sp:n_sp+6,    :] = Y_block                 # heavy→light
        big_mat[:n_sp,          n_sp+6,         :] = heavy_abs               # heavy→absorbed
        big_mat[n_sp:n_sp+6,    n_sp:n_sp+6,    :] = M_light                 # light↔light cascade

        # Initial state: heavy injection, no light particles, no absorbed
        alpha_aug = np.append(np.append(alpha[indices], np.zeros(6)), 0.0)

        if type(L) is np.ndarray:
            # Per-boost sub-batches avoid a scipy batched-expm precision issue
            # that appears when matrices with widely different norms share a batch.
            expmatL = np.stack([
                expm(L[:, None, None] * big_mat[:, :, b])
                for b in range(big_mat.shape[-1])
            ])
            result  = np.matmul(alpha_aug, expmatL)          # (n_b, n_L, sz)
            light   = result[:, :, n_sp:n_sp+6]             # (n_b, n_L, 6)
            return np.moveaxis(light, -1, 0)                 # (6, n_b, n_L)
        else:
            expmatL = expm(np.moveaxis(big_mat * L, -1, 0))
            result  = np.matmul(alpha_aug, expmatL)          # (n_b, sz)
            light   = result[:, n_sp:n_sp+6]                 # (n_b, 6)
            return np.moveaxis(light, -1, 0)                 # (6, n_b)

    def light_production_cumulative(self, L, alpha=None, mass_range=None, boost_range=None,
                                    true_range=None, channel='total'):
        """Cumulative number of light particles produced up to L, separated by channel.

        Tracks, for each light species [He4, He3, H3, H2, p, n], how many particles
        were ever created up to distance L — a produced particle stays counted even
        if it is later destroyed or converted.  Three production channels are
        tallied separately:

        - 'emission'   : particles emitted by photodisintegration of the heavy cascade.
        - 'conversion' : particles created from other light particles (n→p decay,
                         H3→He3, photodisintegration of the bound secondaries).
        - 'leading'    : the heavy-cascade chain itself arriving at the species
                         (the leading remnant becoming He4/He3/H2 or a free nucleon).
        - 'total'      : sum of the three.

        All channels are exact matrix-exponential solutions of the same augmented
        ODE solved by light_cascade_production: the standing light populations
        evolve with the full light-block matrix, while accumulator columns
        integrate each production inflow without outflow.  The standing
        populations themselves are returned by light_cascade_production; the
        instantaneous heavy-cascade production rate by light_secondaries_production.

        Output shape: (6, n_boosts, n_L) for array L, or (6, n_boosts) for scalar L.

        Arguments:
        ----------
        L : float or array of distances at which the tallies are evaluated
        alpha : injection vector (must sum to one)
        mass_range : species indices to include. Must be provided together with true_range.
        boost_range : boost values to evaluate. Full grid by default.
        true_range : subset of mass_range that are not absorption states.
        channel : 'emission', 'conversion', 'leading' or 'total' (default).
        """
        if channel not in ('emission', 'conversion', 'leading', 'total'):
            raise ValueError(f"channel must be 'emission', 'conversion', 'leading' "
                             f"or 'total', got {channel!r}")
        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor = self.interpolator(boost_range)   # (n_sp_full, n_sp_full, n_b)
        prod_mat = self.interpyields(boost_range)          # (6, n_sp_full, n_sp_full, n_b)

        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]
            prod_mat = prod_mat[np.ix_(range(prod_mat.shape[0]), mass_range, mass_range, range(len(boost_range)))]

        # make diagonal zero, then recompute it to absorb outflow to absorbed species
        reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])])
        reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)

        indices = [mass_range.index(ival) for ival in true_range]
        n_sp = len(indices)
        n_b  = len(boost_range)

        # heavy → light emission rates (sum over ALL daughters, as in light_cascade_production)
        Y_full  = prod_mat[np.ix_(range(6), indices, range(prod_mat.shape[2]), range(n_b))].sum(axis=2)
        Y_block = np.moveaxis(Y_full, 0, 1)                  # (n_sp, 6, n_b)

        # heavy → leading arrival rates: mass_range-level tensor columns into each
        # light species, parents restricted to true_range, self-transitions excluded.
        # Absorbed species like (1,1)/(0,1) are handled by the same columns.
        LIGHT_ZA = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
        lead_block = np.zeros((n_sp, 6, n_b))
        for lk, za in enumerate(LIGHT_ZA):
            if za not in self.species:
                continue
            g = self.species.index(za)
            if g not in mass_range:
                continue
            col = mass_range.index(g)
            rates = reduced_tensor[indices, col, :].copy()   # (n_sp, n_b)
            if col in indices:
                rates[indices.index(col)] = 0.0              # exclude the (negative) self-entry
            lead_block[:, lk, :] = rates

        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(n_b))]
        heavy_abs = reduced_tensor.sum(axis=1)               # (n_sp, n_b)

        # light standing evolution and its production couplings (off-diagonal part)
        M_light = self._build_light_matrix(boost_range)      # (6, 6, n_b)
        M_conv  = M_light.copy()
        for k in range(6):
            M_conv[k, k, :] = 0.0                            # keep only production couplings

        # augmented state:
        # [heavy | light standing | acc_emission | acc_conversion | acc_leading | absorbed]
        sz  = n_sp + 25
        i_l, i_e, i_c, i_a = n_sp, n_sp + 6, n_sp + 12, n_sp + 18
        big_mat = np.zeros((sz, sz, n_b))
        big_mat[:n_sp,      :n_sp,      :] = reduced_tensor  # heavy ↔ heavy
        big_mat[:n_sp,      i_l:i_l+6,  :] = Y_block         # heavy → light standing
        big_mat[:n_sp,      i_e:i_e+6,  :] = Y_block         # heavy → emission tally
        big_mat[:n_sp,      i_a:i_a+6,  :] = lead_block      # heavy → leading tally
        big_mat[:n_sp,      n_sp+24,    :] = heavy_abs       # heavy → absorbed sink
        big_mat[i_l:i_l+6,  i_l:i_l+6,  :] = M_light         # light standing evolution
        big_mat[i_l:i_l+6,  i_c:i_c+6,  :] = M_conv          # light → conversion tally

        alpha_aug = np.append(alpha[indices], np.zeros(25))

        if type(L) is np.ndarray:
            # Per-boost sub-batches avoid a scipy batched-expm precision issue
            # that appears when matrices with widely different norms share a batch.
            expmatL = np.stack([
                expm(L[:, None, None] * big_mat[:, :, b])
                for b in range(big_mat.shape[-1])
            ])
            result  = np.matmul(alpha_aug, expmatL)          # (n_b, n_L, sz)
        else:
            expmatL = expm(np.moveaxis(big_mat * L, -1, 0))
            result  = np.matmul(alpha_aug, expmatL)          # (n_b, sz)

        if channel == 'total':
            tally = (result[..., i_e:i_e+6] + result[..., i_c:i_c+6]
                     + result[..., i_a:i_a+6])
        else:
            start = {'emission': i_e, 'conversion': i_c, 'leading': i_a}[channel]
            tally = result[..., start:start+6]

        return np.moveaxis(tally, -1, 0)                     # (6, n_b, n_L) or (6, n_b)

    def cdf_boost_range(self, L, alpha=None, mass_range=None, boost_range=None, true_range=None):
        """Returns the probability (cumulative) distribution values at positions L for a range of boosts

        Arguments:
        ----------
        L : a float or an array of distances at which the pdf will be evaluated
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        """

        if boost_range is None:
            boost_range = self.boosts       

        reduced_tensor = self.interpolator(boost_range)

        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]
        
        # make diagonal zero
        reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])]) 
        # recompute diagonal including absorption states
        reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)
        # reduce excluding absorption states
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]

        ones = np.ones_like(-np.moveaxis(reduced_tensor, -1, 0).dot(np.ones_like(alpha[indices])))

        if type(L) is np.ndarray:
            expmatL = np.stack([
                expm(L[:, None, None] * reduced_tensor[:, :, b])
                for b in range(reduced_tensor.shape[-1])
            ])
        else:
            expmatL = expm(np.moveaxis(reduced_tensor * L, -1, 0))

        if alpha.shape == ones.shape:
            total = 1 - np.matmul(np.matmul(alpha[indices], expmatL), ones)
        else:
            total = 1 - np.einsum('ijk,ik->ij', np.matmul(alpha[indices], expmatL), ones)

        return self.boosts, total

    def pdf_boost_range(self, L, alpha=None, mass_range=None, omega=None, boost_range=None, true_range=None):
        """Returns the probability density value at positions L for a range of boosts

        Arguments:
        ----------
        L : a float or an array of distances at which the pdf will be evaluated
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        omega : ending or production vector. By default is set to omega=-Te
        true_range : range of species not part of the absorption range (excluding indices for species that are part of the absorption state)
                     if none is given, the last species in mass_range is considered the absorption state.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        """

        if boost_range is None:
            boost_range = self.boosts       

        reduced_tensor = self.interpolator(boost_range)
        
        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]

        # if is a matrix
        if len(reduced_tensor[:, :, 0]) > 1:
            # make diagonal zero 
            reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])]) 
            # recompute diagonal including absorption states
            reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)

        # reduce excluding absorption states
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]

        if omega is None:
            omega = - np.moveaxis(reduced_tensor, -1, 0).dot(np.ones_like(alpha[indices]))

        if type(L) is np.ndarray:
            expmatL = np.stack([
                expm(L[:, None, None] * reduced_tensor[:, :, b])
                for b in range(reduced_tensor.shape[-1])
            ])
        else:
            expmatL = expm(np.moveaxis(reduced_tensor * L, -1, 0))

        if alpha.shape == omega.shape:
            total = np.matmul(np.matmul(alpha[indices], expmatL), omega)
        else:
            total = np.einsum('ijk,ik->ij', np.matmul(alpha[indices], expmatL), omega)

        return boost_range, total

    def pdf_moments_boost_range(self, alpha=None, mass_range=None, boost_range=None, true_range=None, degree=1):
        """Returns the moments for a range of boosts

        Arguments:
        ----------
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        true_range : range of species not part of the absorption range (excluding indices for species that are part of the absorption state)
                     if none is given, the last species in mass_range is considered the absorption state.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        degree : the order of the moment n as in  mu_n = E[X^n]
        """

        if boost_range is None:
            boost_range = self.boosts       

        reduced_tensor = self.interpolator(boost_range)
        
        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]

        # make diagonal zero
        reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])]) 
        # recompute diagonal including absorption states
        reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)
        # reduce excluding absorption states
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]

        inverse = np.linalg.inv(np.moveaxis(reduced_tensor, -1, 0))
        inverse_power = inverse**degree

        moment = factorial(degree) * (-1)**degree * np.matmul(np.matmul(alpha[indices], inverse_power), np.ones_like(alpha[indices]))

        return moment
    
    def pdf_variance_boost_range(self, alpha=None, mass_range=None, boost_range=None, true_range=None):
        """Returns the variance for a range of boosts

        Arguments:
        ----------
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        true_range : range of species not part of the absorption range (excluding indices for species that are part of the absorption state)
                     if none is given, the last species in mass_range is considered the absorption state.
        """

        if boost_range is None:
            boost_range = self.boosts       

        reduced_tensor = self.interpolator(boost_range)
        
        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]

        # make diagonal zero
        reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])]) 
        # recompute diagonal including absorption states
        reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)
        # reduce excluding absorption states
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]

        inverse = np.linalg.inv(np.moveaxis(reduced_tensor, -1, 0))

        momentum1 = -np.matmul(np.matmul(alpha[indices], inverse), np.ones_like(alpha[indices]))
        momentum2 = np.matmul(np.matmul(alpha[indices], inverse**2), np.ones_like(alpha[indices]))
        
        return 2*momentum2 - momentum1**2 

    def _genenerate_complete_matrices(self):
        """Generates, for each boost, a complete interaction matrix from the interaction tables
        """
        def ZA_ordinal(nuc):
            '''Useful to sort by mass and charge.
               Returns and ordinal that places (Z, A) values in the desired order.
            '''
            return nuc[1]*1000 + nuc[0]
        
        self.species = self.nuclei.copy()
        self.species.sort(key=ZA_ordinal, reverse=True)
        self.species += [(0, 1), (1, 1)]

        self._resolve_unstable_products()

        # generate interaction tensor by slices
        tensor = np.zeros((len(self.species), len(self.species), len(self.boosts)))
        for mom, nuc_branches in zip(self.nuclei, self.all_branchings):
            main_products = list(zip(nuc_branches[:, 0], nuc_branches[:, 1]))

            if np.any([prod not in self.species for prod in main_products]):
                logger.debug('For nucleus %s some products were not found.', mom)

            # Fix the channels with dead ends
            # if np.any([prod not in self.species for prod in main_products]):
            #     print(f'For nucleus {self.species[i]} some products were not found when creating light yields tensor.')

            #     for rowid, prod in enumerate(main_products):
            #         if prod not in self.species:
            #             new_prods, new_rates = fix_dead_end(prod, nuc_branches[rowid, 2:])

            #             # print(prod, new_prods, max(new_prods), np.any([nprod in main_products for nprod in new_prods]))
            #             if not np.any([nprod in main_products for nprod in new_prods]):
            #                 nuc_branches[rowid, :2] = max(new_prods)
            #                 nuc_branches[rowid, 2:] = new_rates
            
            try:
                i = self.species.index(mom)
            except:
                logger.debug('problem with nucleus %s: present in branchings but not in nuclei.', mom)
                continue

            for branch in nuc_branches:
                try:
                    j = self.species.index(tuple(branch[:2]))
                    tensor[i, j, :] += branch[2:]
                except:
                    logger.debug('problem in tensor with product %s of nucleus %s', branch[:2], mom)
                    continue

        tensor -= np.stack([np.diag(row) for row in tensor.sum(axis=1).T], axis=2)

        # generate light production tensors by slices
        ly_all_mats = []
        for light_yield in self.marginal_light_yields:
            ly_matrices = np.zeros((len(self.species), len(self.species), len(self.boosts)))
            for mom, nuc_branches in zip(self.nuclei, light_yield):
                # Row index by species lookup (like the heavy tensor above):
                # nuclei-list order is not always the reverse of species order.
                i = self.species.index(mom)
                main_products = list(zip(nuc_branches[:, 0], nuc_branches[:, 1]))
                
                # # Fix the channels with dead ends
                # if np.any([prod not in self.species for prod in main_products]):
                #     print(f'For nucleus {self.species[i]} some products were not found when creating light yields tensor.')
                    
                #     for rowid, prod in enumerate(main_products):
                #         if prod not in self.species:
                #             new_prods, new_rates = fix_dead_end(prod, nuc_branches[rowid, 2:])

                #             # print(prod, new_prods, max(new_prods), np.any([nprod in main_products for nprod in new_prods]))
                #             if not np.any([nprod in main_products for nprod in new_prods]):
                #                 nuc_branches[rowid, :2] = max(new_prods)
                #                 # nuc_branches[rowid, 2:] = new_rates # not rates, so new_rates is incorrect 

                for branch in nuc_branches:
                    try:
                        j = self.species.index(tuple(branch[:2]))
                        ly_matrices[i, j, :] += branch[2:]
                    except:
                        logger.debug('problem with product %s of nucleus %s', branch[:2], mom)
                        continue
            ly_all_mats.append( ly_matrices )

        # check that all rows add up to one!!!
        # np.all(np.isclose(np.einsum('ijk, j -> ik', tensor, np.ones(186)), 0))

        self.tensor = tensor.astype(self.ftype)
        self.light_prod_tensor = np.stack([lyt.astype(self.ftype) for lyt in ly_all_mats])
        self.interpolator = lambda boostval: interp1d(self.boosts, self.tensor, 'previous')(boostval)
        self.interpyields = lambda boostval: interp1d(self.boosts, self.light_prod_tensor, 'cubic')(boostval)

    def save(self, path):
        """Saves the data to a .npz file.

        Saves all arrays needed to reconstruct an instance with load().

        Arguments:
        ----------
        path : str or path-like — destination file (.npz appended if absent)
        """
        path = str(path)
        if not path.endswith('.npz'):
            path += '.npz'

        data = {
            'boosts':             self.boosts,
            'tensor':             self.tensor,
            'light_prod_tensor':  self.light_prod_tensor,
            'nuclei':             np.array(self.nuclei,  dtype=np.int32),
            'species':            np.array(self.species, dtype=np.int32),
            'n_light':            np.array([len(self.marginal_light_yields)], dtype=np.int32),
        }

        for i, br in enumerate(self.all_branchings):
            data[f'br_{i}'] = br

        for li, light_yields in enumerate(self.marginal_light_yields):
            for ni, arr in enumerate(light_yields):
                data[f'mly_{li}_{ni}'] = arr

        np.savez(path, **data)

    def load(self, path):
        """Populate an instance from a file saved with save().

        Replaces all computed attributes in place, bypassing
        _construct_from_files() and _generate_complete_matrices().
        The existing object reference remains valid after the call.

        Arguments:
        ----------
        path : str or path-like — source file (.npz appended if absent)
        """
        path = str(path)
        if not path.endswith('.npz'):
            path += '.npz'

        d = np.load(path, allow_pickle=False)

        self.boosts            = d['boosts']
        self.tensor            = d['tensor']
        self.light_prod_tensor = d['light_prod_tensor']
        self.ftype             = d['tensor'].dtype
        self.nuclei  = [tuple(row) for row in d['nuclei'].tolist()]
        self.species = [tuple(row) for row in d['species'].tolist()]

        n_nuc   = len(self.nuclei)
        n_light = int(d['n_light'][0])

        self.all_branchings = [d[f'br_{i}'] for i in range(n_nuc)]
        self.marginal_light_yields = [
            [d[f'mly_{li}_{ni}'] for ni in range(n_nuc)]
            for li in range(n_light)
        ]

        self.interpolator = lambda boostval: interp1d(self.boosts, self.tensor, 'previous')(boostval)
        self.interpyields = lambda boostval: interp1d(self.boosts, self.light_prod_tensor, 'cubic')(boostval)

    def get_distribution_parameters(self, mass_lims=(56, 11), injection_type=('only species', (26, 56)), absorption_type=('only mass', [54]), boost_range=None):
        """Produces the injection vector and mass_range required to
        produce the distribution of nuclei starting from a certain mass
        and producing mass lower than a minimum given value.

        Arguments:
        ----------
        species : The list of species (Z, A) that are included in the full nuclear cascade.
        mass_lims : a tuple (Amax, Amin) with the starting mass and the lower limit for mass
        injection type : (type, params) (str, dict) info specifying the injection. Possible values
                    - 'flat' : equal injection of all species included within the mass range mass_lims
                    - 'only mass' : equal injection of all species specified by a mass value in a list
                    - 'only species' : equal injection of all species specified as (Z, A)
        absorption type : (type, params) (str, dict) info specifying the absorbing state. Possible values
                    - 'only mass' : equal injection of all species specified by a mass value in a list
                    - 'only species' : equal injection of all species specified as (Z, A)
            
        """
        Amax, Amin = mass_lims
        
        mass_range = [k for k, spec in enumerate(self.species) if Amax >= spec[1] > Amin]
        alpha = np.ones(len(self.species))[mass_range]

        itype, iparams = injection_type
        atype, aparams = absorption_type

        if itype == 'only mass':
            masses = iparams
            indices = [k for k, idx in enumerate(mass_range) if self.species[idx][1] not in masses]
            alpha[indices] = 0
        elif itype == 'only species':
            species = iparams
            indices = [k for k, idx in enumerate(mass_range) if self.species[idx] != species]
            alpha[indices] = 0
        # renormalize injection vector
        alpha /= sum(alpha)

        if atype == 'only mass':
            masses = aparams
            arange = [k for k, idx in enumerate(mass_range) if self.species[idx][1] not in masses]
        elif atype == 'only species':
            species = aparams
            arange = [k for k, idx in enumerate(mass_range) if self.species[idx] not in species]
        elif atype == 'only charge':
            charges = aparams
            arange = [k for k, idx in enumerate(mass_range) if self.species[idx][0] not in charges]
        
        true_range = [idx for k, idx in enumerate(mass_range) if k in arange]

        # Computing matrices for the range of boosts provided
        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor = self.interpolator(boost_range)
        reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]

        # if it is a matrix
        if len(reduced_tensor[:, :, 0]) > 1:
            # make diagonal zero
            reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])])
            # recompute diagonal including absorption states
            reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)

        # reduce excluding absorption states
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]

        return alpha, mass_range, true_range, reduced_tensor

    def _check_tensor_balance(self):
        """Crosschecking integrity of tensor with different tests.
        """

        # Checking rows sum is zero
        if np.all(np.isclose(np.einsum('ijk, j -> ik', self.tensor, np.ones(len(self.species))), 0, rtol=1e-10)):
            print('The tensor row sum is null with relative tolerance 1e-10')
        else:
            print('The tensor row sum is not null with relative tolerance 1e-10')

        # Check for dead end species
        for i in range(len(self.tensor)):
            if np.all(np.isclose(self.tensor[i, i, :], 0, rtol=1e-10)):
                print('Dead end species found:', self.species[i])

    def _resolve_unstable_products(self):
        """Disintegrate channel products that are not tracked species through
        their nuclear decay chain, at construction time.

        Uses self.decays when provided (the nubase table); otherwise a builtin
        minimal table covering the particle-unstable valley nuclides that the
        cross-section models emit as remnants (Be8 -> 2 He4, He5 -> He4 + n,
        Li5 -> He4 + p), so channel strength is never silently dropped.

        Valid because the decays involved are many orders of magnitude faster
        than any propagation timescale, even time dilated. Branching rows are
        redirected to the final remnant of the chain (splitting the rate when
        a nuclide has several decay channels), the emitted light particles are
        added to marginal_light_yields, and beta decays only shift the remnant
        charge.
        """
        light_species = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
        light_daughters = {402: 0, 302: 1, 301: 2, 201: 3, 101: 4, 100: 5}  # nucid -> index above

        decays = self.decays if self.decays else {
            804: {'decay_time': 1.18e-16, 'channels': [[1.0, 402]]},   # Be8 -> He4 + He4
            502: {'decay_time': 1.01e-21, 'channels': [[1.0, 100]]},   # He5 -> He4 + n
            503: {'decay_time': 5.34e-22, 'channels': [[1.0, 101]]},   # Li5 -> He4 + p
        }

        def resolve(product, frac=1.0, depth=0):
            """List of (fraction, final remnant (Z, A), {light index: count}), or None."""
            Z, A = int(product[0]), int(product[1])
            if (Z, A) in self.species:
                return [(frac, (Z, A), {})]
            nucid = A * 100 + Z
            if depth > 10 or nucid not in decays:
                return None

            outcomes = []
            for channel in decays[nucid]['channels']:
                branching, daughters = channel[0], channel[1:]
                Zrem, Arem, counts = Z, A, {}
                for dau in daughters:
                    if dau == -1:    # beta-minus
                        Zrem += 1
                    elif dau == 1:   # beta-plus / electron capture
                        Zrem -= 1
                    else:
                        Zrem -= dau % 100
                        Arem -= dau // 100
                        if dau in light_daughters:
                            li = light_daughters[dau]
                            counts[li] = counts.get(li, 0) + 1

                if Arem == 0 and counts:
                    # fully disintegrated: promote the heaviest emitted
                    # particle to leading remnant
                    li = min(counts)
                    counts[li] -= 1
                    if counts[li] == 0:
                        del counts[li]
                    Zrem, Arem = light_species[li]

                sub = resolve((Zrem, Arem), frac * branching, depth + 1)
                if sub is None:
                    return None
                for sub_frac, sub_rem, sub_counts in sub:
                    merged = dict(counts)
                    for li, count in sub_counts.items():
                        merged[li] = merged.get(li, 0) + count
                    outcomes.append((sub_frac, sub_rem, merged))
            return outcomes

        for k in range(len(self.nuclei)):
            cache = {}

            def resolved(product):
                if product not in cache:
                    cache[product] = resolve(product)
                return cache[product]

            # redirect branching rows and collect the emitted light particles
            new_rows = []
            extra_yields = {}
            for row in self.all_branchings[k]:
                product = (int(row[0]), int(row[1]))
                if product in self.species:
                    new_rows.append(row)
                    continue
                outcomes = resolved(product)
                if outcomes is None:
                    logger.debug('No decay path found for product %s; channel left as is.', product)
                    new_rows.append(row)
                    continue
                for frac, remnant, counts in outcomes:
                    new_row = row.copy()
                    new_row[:2] = remnant
                    new_row[2:] = frac * row[2:]
                    new_rows.append(new_row)
                    for li, count in counts.items():
                        yield_row = row.copy()
                        yield_row[:2] = remnant
                        yield_row[2:] = count * frac * row[2:]
                        extra_yields.setdefault(li, []).append(yield_row)
            self.all_branchings[k] = np.vstack(new_rows)

            # relabel light-yield rows consistently and append the emitted particles
            for li, yields in enumerate(self.marginal_light_yields):
                arr = yields[k]
                rows = []
                for row in (np.atleast_2d(arr) if arr.size else []):
                    product = (int(row[0]), int(row[1]))
                    outcomes = resolved(product) if product not in self.species else None
                    if outcomes is None:
                        rows.append(row)
                        continue
                    for frac, remnant, counts in outcomes:
                        new_row = row.copy()
                        new_row[:2] = remnant
                        new_row[2:] = frac * row[2:]
                        rows.append(new_row)
                rows += extra_yields.get(li, [])
                if rows:
                    yields[k] = np.vstack(rows)


class InteractionCore_CRPropA(InteractionCore):
    """Producing interaction matrices from CRPropA interaction files 
    """

    def __init__(self, data_files=None):

        if data_files is None:
            # The CRPropa data directory must be provided by the user.
            # Download the CRPropa data from https://github.com/CRPropa/CRPropa3-data
            # and pass its path here, or set CRPROPA_DATA_PATH in the environment.
            crpropa_data = os.environ.get('CRPROPA_DATA_PATH', '')
            if not crpropa_data:
                raise ValueError(
                    "InteractionCore_CRPropA requires the CRPropa data directory.\n"
                    "Either pass data_files={'path': '/your/crpropa/data/', ...} or\n"
                    "set the environment variable CRPROPA_DATA_PATH=/your/crpropa/data/"
                )
            self.data_files = {
                'path' : crpropa_data,
                
                'photodisintegration': {
                    'rates_cmb' : 'Photodisintegration/rate_CMB.txt',
                    'rates_ebl' : 'Photodisintegration/rate_EBL_LopezSaldana21.txt',
                    'branchings_cmb' : 'Photodisintegration/branching_CMB.txt',
                    'branchings_ebl' : 'Photodisintegration/branching_EBL_LopezSaldana21.txt',
                },

                'photopionproduction': {
                    'rates_cmb' : 'PhotoPionProduction/rate_CMB.txt',
                } 
            }
        else:
            self.data_files = data_files
        
        InteractionCore.__init__(self)

    def _generate_photomeson_table(self, nuclei):
        """Creates photomeson data from tables assuming a superposition model.
        """
        from pandas import DataFrame, MultiIndex
        
        boosts = np.logspace(6, 14, 201)
        cols = [f'{i}' for i in range(201)]
        daughter_names = ['a', 'he3', 't', 'd', 'p', 'n']
        pp_rates = np.genfromtxt(os.path.join(self.data_files['path'], self.data_files['photopionproduction']['rates_cmb']))

        prates = np.interp(boosts, 10**pp_rates[:, 0], pp_rates[:, 1])
        nrates = np.interp(boosts, 10**pp_rates[:, 0], pp_rates[:, 2])

        zvals, avals = np.array([z for z, _ in nuclei]), np.array([a for _, a in nuclei])

        Z = np.repeat(np.atleast_2d(zvals).T, len(boosts), axis=1)
        A = np.repeat(np.atleast_2d(avals).T, len(boosts), axis=1)
        N = A - Z

        pprates = Z * prates + N * nrates

        df_rates_pmes = DataFrame(data=np.hstack([nuclei, pprates]), index=MultiIndex.from_arrays(np.array(nuclei).T), columns=['Z', 'A'] + cols)

        pmes_branchings = []
        pmes_marginal_yields = []
        for idx, (Z, A) in enumerate(nuclei):
            remnants = [(Z, A-1), (Z-1, A-1)]
            for br, (Zrem, Arem) in zip([(1-Z/A), Z/A], remnants):
                if (Zrem, Arem) in nuclei:
                    pmes_branchings.append(np.hstack([A, Z, Arem, Zrem, br * np.ones(201)]))
                    pmes_marginal_yields.append(np.hstack([A, Z, Arem, Zrem, 0, 0, 0, 0, Z-Zrem, A-Arem-Z+Zrem, br * np.ones(201)]))

            if not np.all([rem in nuclei for rem in remnants]):
                if np.any([rem in nuclei for rem in remnants]):
                    pmes_branchings[-1][4:] = np.ones(201)
                    pmes_marginal_yields[-1][10:] = np.ones(201)
                else:
                    # No remnant in nuclei, add dummy channel with zeros
                    pmes_branchings.append(np.hstack([A, Z, A-1, Z, np.zeros(201)]))
                    pmes_marginal_yields.append(np.hstack([A, Z, A-1, Z, 0, 0, 0, 0, 0, 0, np.zeros(201)]))

        
        pmes_branchings = np.vstack(pmes_branchings)
        pmes_branchings = DataFrame(data=pmes_branchings, index=MultiIndex.from_arrays(pmes_branchings[:, :4].T), columns=['A', 'Z', 'Ar', 'Zr'] + cols)

        pmes_marginal_yields = np.vstack(pmes_marginal_yields)
        pmes_marginal_yields = DataFrame(data=pmes_marginal_yields, index=MultiIndex.from_arrays(pmes_marginal_yields[:, :4].T), columns=['A', 'Z', 'Ar', 'Zr'] + daughter_names + cols)

        # Merging channels with the same heavy product
        merged_yields = []
        for col in daughter_names:
            df_brnch_no_channels = pmes_marginal_yields.drop(columns=daughter_names)
            df_brnch_no_channels[cols] = df_brnch_no_channels.multiply(pmes_marginal_yields[col].values, axis='index')[cols]
            merged_yields.append( df_brnch_no_channels )
            
        return df_rates_pmes, pmes_branchings, merged_yields

    def _construct_from_files(self):
        """CRPropA data is structured in different files depending on the 
        interaction and the photon field.
           This function loads the files and populates the fields:
           - boosts: the boost grid in which the data is given
           - nuclei: the list of nuclear species (Zi, Ai) contained in the files
           - all_rates: the interaction rates including all processes and CMB+EBL
           - all_branches: the marginal interaction rates including all processes and CMB+EBL
        """
        from pandas import DataFrame
        cols = [f'{i}' for i in range(201)]

        df_rates_cmb = load_rates(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['rates_cmb']))
        df_brnch_cmb, merged_yields_cmb = load_branchings(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['branchings_cmb']))

        df_rates_ebl = load_rates(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['rates_ebl']))
        df_brnch_ebl, merged_yields_ebl = load_branchings(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['branchings_ebl']))

        nuclei = [(z, a) for a, z in df_rates_cmb.index.values]
        df_rates_pmes, df_brnch_pmes, merged_yields_pmes = self._generate_photomeson_table(nuclei=[(0, 1), (1, 1)] + nuclei)

        df_rates = df_rates_cmb.groupby(by=['A', 'Z']).sum() + df_rates_ebl.groupby(by=['A', 'Z']).sum() + df_rates_pmes.groupby(by=['A', 'Z']).sum()

        # Computing photomeson rates by superposition
        pp_rates = np.genfromtxt(os.path.join(self.data_files['path'], self.data_files['photopionproduction']['rates_cmb']))
        pprates = np.zeros((len(nuclei), 201))
        boosts = np.logspace(6, 14, 201)
        for k in range(pprates.shape[0]):
            Z, A = nuclei[k]
            pprates[k] = np.interp(boosts, 10**pp_rates[:, 0], Z*pp_rates[:, 1] + (A-Z)*pp_rates[:, 2])

        df_brnch_cmb[cols] = df_brnch_cmb.multiply(df_rates_cmb.reindex(df_brnch_cmb.index, method='ffill'))[cols]
        merged_cmb = df_brnch_cmb.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_cmb = [np.hstack([np.vstack(merged_cmb.loc[nuc].index.values), merged_cmb.loc[nuc][cols].values]) for nuc in nuclei]

        df_brnch_ebl[cols] = df_brnch_ebl.multiply(df_rates_ebl.reindex(df_brnch_ebl.index, method='ffill'))[cols]
        merged_ebl = df_brnch_ebl.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_ebl = [np.hstack([np.vstack(merged_ebl.loc[nuc].index.values), merged_ebl.loc[nuc][cols].values]) for nuc in nuclei]

        df_brnch_pmes[cols] = df_brnch_pmes.multiply(df_rates_pmes.reindex(df_brnch_pmes.index, method='ffill'))[cols]
        merged_pmes = df_brnch_pmes.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_pmes = [np.hstack([np.vstack(merged_pmes.loc[nuc].index.values), merged_pmes.loc[nuc][cols].values]) for nuc in nuclei]

        all_branchings = []
        for mr1, mr2, mr3 in zip(allmr_cmb, allmr_ebl, allmr_pmes):
            mr12 = merge_marginal_rates(mr1, mr2)
            all_branchings.append(merge_marginal_rates(mr12, mr3))

        all_merged = []
        for mycmb, myebl, mypmes in zip(merged_yields_cmb, merged_yields_ebl, merged_yields_pmes):                    
            light_yield_cmb = mycmb[cols].multiply(df_rates_cmb.reindex(mycmb[cols].index, method='ffill'))
            light_yield_ebl = myebl[cols].multiply(df_rates_ebl.reindex(myebl[cols].index, method='ffill'))
            light_yield_pmes = mypmes[cols].multiply(df_rates_pmes.reindex(mypmes[cols].index, method='ffill'))

            merged_cols = (light_yield_cmb + light_yield_ebl).add(light_yield_pmes, fill_value=0)[cols]
            merged = DataFrame(data=np.hstack([np.vstack(merged_cols.index.values), merged_cols.values]), index=merged_cols.index, columns=['A', 'Z', 'Ar', 'Zr'] + cols)

            merged[cols] = merged.divide(df_rates.reindex(merged.index, method='ffill'))[cols]
            merged = merged.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
            all_merged.append([np.hstack([np.vstack(merged.loc[nuc].index.values), merged.loc[nuc][cols].values]) for nuc in nuclei])
    
        self.boosts = np.logspace(6, 14, 201)
        self.nuclei = nuclei
        self.all_rates = df_rates.values
        self.all_branchings = all_branchings
        self.marginal_light_yields = all_merged

        # Fix to avoid dead ends
        print('!!! Replacing some nuclei to avoid dead ends !!!')
        for i, branches in enumerate(self.all_branchings):
            for j, branch in enumerate(branches):
                if tuple(branch[:2]) in [(2, 5), (3, 5)]:
                    self.all_branchings[i][j, :2] = [2, 4]
                elif tuple(branch[:2]) in [(5, 9), (6, 9)]:
                    self.all_branchings[i][j, :2] = [4, 9]

    def check_data_consistency(self):
        """Verify that data is complete and numbers add up as expected
        """

        nuclei = self.nuclei

        allmr_cmb = get_marginal_rates(nuclei, rates_cmb, self.boosts, branchings_cmb) 
        allmr_ebl = get_marginal_rates(nuclei, rates_ebl, self.boosts, branchings_ebl) 
        allmr_ppi = get_marginal_rates(nuclei, pp_rates, self.boosts)
        allmr = [merge_marginal_rates(merge_marginal_rates(mr1, mr2), mr3) for mr1, mr2, mr3 in zip(allmr_cmb, allmr_ebl, allmr_ppi)]

        # TEST PHOTOPION: WORKING FINE
        # for k, (mr, nuc) in enumerate(zip(allmr_ppi, nuclei)):
        #     Z, A, N = nuc[0], nuc[1], nuc[1]-nuc[0]
        #     tr = np.interp(boosts, 10**pp_rates[:, 0], Z*pp_rates[:, 1] + N*pp_rates[:, 2])
        #     ratio = np.divide(np.sum(mr[:, 2:], axis=0), tr, where=tr>0)
        #     non_null_ratio = ratio[np.where(np.logical_not(np.isclose(ratio, 0)))]

        #     if np.any(np.logical_not(np.isclose(non_null_ratio, 1))):
        #         print(nuc)
        #         print(non_null_ratio)

        # TEST PHOTODIS CMB: WORKING FINE. (3,8) AND (5, 8) FAIL COMPLETELY 
        # for mr, tr, nuc in zip(allmr_cmb, rates_cmb, nuclei):
        #     ratio = np.divide(np.sum(mr[:, 2:], axis=0), tr[2:], where=tr[2:]>0)
        #     non_null_ratio = ratio[np.where(np.logical_not(np.isclose(ratio, 0)))]

        #     if np.any(np.logical_not(np.isclose(non_null_ratio, 1))):
        #         print(nuc)
        #         print(non_null_ratio)

        # TEST PHOTODIS EBL: MOST NUCLEI FAIL FOR THE LAST 57 BOOSTS. (3,8) AND (5, 8) FAIL COMPLETELY 
        # for mr, tr, nuc in zip(allmr_ebl, rates_ebl, nuclei):
        #     ratio = np.divide(np.sum(mr[:, 2:], axis=0), tr[2:], where=tr[2:]>0)
        #     non_null_ratio = ratio[np.where(np.logical_not(np.isclose(ratio, 0)))]

        #     # if np.any(np.logical_not(np.isclose(non_null_ratio, 1))):
        #     # if sum(np.logical_not(np.isclose(non_null_ratio, 1))) > 57:
        #     if np.any(np.logical_not(np.isclose(non_null_ratio[:-57], 1))):
        #         print(nuc)
        #         print(sum(np.logical_not(np.isclose(non_null_ratio, 1))))

        # TEST ALL RATES: WORKING FINE. (3,8) AND (5, 8) FAIL COMPLETELY, AS EXPECTED
        # for mr, tr, nuc in zip(allmr, all_rates, nuclei):
        #     ratio = np.divide(np.sum(mr[:, 2:], axis=0), tr, where=tr>0)
        #     non_null_ratio = ratio[np.where(np.logical_not(np.isclose(ratio, 0)))]

        #     if np.any(np.logical_not(np.isclose(non_null_ratio, 1))):
        #         print(nuc)
        #         print(non_null_ratio)

        #     print(nuc, ratio[boostidx])

        # CHECK BRANCHINGS CMB: WORKING FINE. (3,8) AND (5, 8) FAIL COMPLETELY, AS EXPECTED
        # for Z, N in set(zip(branchings_cmb[:, 0], branchings_cmb[:, 1])):
        #     brsum = np.sum(branchings_cmb[(branchings_cmb[:, 0]==Z) * (branchings_cmb[:, 1]==N)][:, 2:], axis=0)

        #     notnullbrsum = brsum[np.logical_not(np.isclose(brsum, 0))]

        #     if np.any(np.logical_not(np.isclose(notnullbrsum[1:], 1))):
        #         print(Z, N)
        #         print(notnullbrsum)

        # CHECK BRANCHINGS EBL: WORKING FINE. (3,8) AND (5, 8) FAIL COMPLETELY, AS EXPECTED
        # for Z, N in set(zip(branchings_ebl[:, 0], branchings_ebl[:, 1])):
        #     brsum = np.sum(branchings_ebl[(branchings_ebl[:, 0]==Z) * (branchings_ebl[:, 1]==N)][:, 2:], axis=0)

        #     notnullbrsum = brsum[np.logical_not(np.isclose(brsum, 0))]

        #     if np.any(np.logical_not(np.isclose(notnullbrsum[1:], 1))):
        #         print(Z, N)
        #         print(notnullbrsum)
            
        return None


class InteractionCore_CRPropA_CMB_pdis(InteractionCore_CRPropA):
    def _construct_from_files(self):
        """CRPropA data is structured in different files depending on the 
        interaction and the photon field.
        """
        cols = [f'{i}' for i in range(201)]

        df_rates_cmb = load_rates(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['rates_cmb']))
        df_brnch_cmb, merged_yields_cmb = load_branchings(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['branchings_cmb']))

        df_rates = df_rates_cmb.groupby(by=['A', 'Z']).sum()
        nuclei = [(z, a) for a, z in df_rates.index.values]

        df_brnch_cmb[cols] = df_brnch_cmb.multiply(df_rates_cmb.reindex(df_brnch_cmb.index, method='ffill'))[cols]
        merged_cmb = df_brnch_cmb.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_cmb = [np.hstack([np.vstack(merged_cmb.loc[nuc].index.values), merged_cmb.loc[nuc][cols].values]) for nuc in nuclei]

        all_merged = []
        for mycmb in merged_yields_cmb:
            merged = mycmb.copy()
                    
            light_yield_cmb = mycmb[cols].multiply(df_rates_cmb.reindex(mycmb[cols].index, method='ffill'))
            merged[cols] = light_yield_cmb[cols]
            merged[cols] = merged.divide(df_rates.reindex(merged.index, method='ffill'))[cols]
            merged = merged.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
            all_merged.append([np.hstack([np.vstack(merged.loc[nuc].index.values), merged.loc[nuc][cols].values]) for nuc in nuclei])
    
        self.boosts = np.logspace(6, 14, 201)
        self.nuclei = nuclei
        self.all_rates = df_rates.values
        self.all_branchings = allmr_cmb
        self.marginal_light_yields = all_merged


class InteractionCore_CRPropA_IRB_pdis(InteractionCore_CRPropA):
    def _construct_from_files(self):
        """CRPropA data is structured in different files depending on the 
        interaction and the photon field.
        """
        cols = [f'{i}' for i in range(201)]

        df_rates_ebl = load_rates(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['rates_ebl']))
        df_brnch_ebl, merged_yields_ebl = load_branchings(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['branchings_ebl']))

        df_rates = df_rates_ebl.groupby(by=['A', 'Z']).sum()
        nuclei = [(z, a) for a, z in df_rates.index.values]

        df_brnch_ebl[cols] = df_brnch_ebl.multiply(df_rates_ebl.reindex(df_brnch_ebl.index, method='ffill'))[cols]
        merged_ebl = df_brnch_ebl.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_ebl = [np.hstack([np.vstack(merged_ebl.loc[nuc].index.values), merged_ebl.loc[nuc][cols].values]) for nuc in nuclei]

        all_merged = []
        for myebl in merged_yields_ebl:
            merged = myebl.copy()
                    
            light_yield_ebl = myebl[cols].multiply(df_rates_ebl.reindex(myebl[cols].index, method='ffill'))
            merged[cols] = light_yield_ebl[cols]
            merged[cols] = merged.divide(df_rates.reindex(merged.index, method='ffill'))[cols]
            merged = merged.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
            all_merged.append([np.hstack([np.vstack(merged.loc[nuc].index.values), merged.loc[nuc][cols].values]) for nuc in nuclei])
    
        self.boosts = np.logspace(6, 14, 201)
        self.nuclei = nuclei
        self.all_rates = df_rates.values
        self.all_branchings = allmr_ebl
        self.marginal_light_yields = all_merged


class InteractionCore_CRPropA_pdis(InteractionCore_CRPropA):
    def _construct_from_files(self):
        """CRPropA data is structured in different files depending on the 
        interaction and the photon field.
        """
        cols = [f'{i}' for i in range(201)]

        df_rates_cmb = load_rates(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['rates_cmb']))
        df_brnch_cmb, merged_yields_cmb = load_branchings(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['branchings_cmb']))

        df_rates_ebl = load_rates(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['rates_ebl']))
        df_brnch_ebl, merged_yields_ebl = load_branchings(os.path.join(self.data_files['path'], self.data_files['photodisintegration']['branchings_ebl']))

        df_rates = df_rates_cmb.groupby(by=['A', 'Z']).sum() + df_rates_ebl.groupby(by=['A', 'Z']).sum()
        # nuclei = list(zip(df_rates['Z'], df_rates['A']))
        nuclei = [(z, a) for a, z in df_rates.index.values]

        df_brnch_cmb[cols] = df_brnch_cmb.multiply(df_rates_cmb.reindex(df_brnch_cmb.index, method='ffill'))[cols]
        merged_cmb = df_brnch_cmb.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_cmb = [np.hstack([np.vstack(merged_cmb.loc[nuc].index.values), merged_cmb.loc[nuc][cols].values]) for nuc in nuclei]

        df_brnch_ebl[cols] = df_brnch_ebl.multiply(df_rates_ebl.reindex(df_brnch_ebl.index, method='ffill'))[cols]
        merged_ebl = df_brnch_ebl.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_ebl = [np.hstack([np.vstack(merged_ebl.loc[nuc].index.values), merged_ebl.loc[nuc][cols].values]) for nuc in nuclei]

        all_merged = []
        for mycmb, myebl in zip(merged_yields_cmb, merged_yields_ebl):
            merged = mycmb.copy()
                    
            light_yield_cmb = mycmb[cols].multiply(df_rates_cmb.reindex(mycmb[cols].index, method='ffill'))
            light_yield_ebl = myebl[cols].multiply(df_rates_ebl.reindex(myebl[cols].index, method='ffill'))
            merged[cols] = ( light_yield_cmb + light_yield_ebl )[cols]
            merged[cols] = merged.divide(df_rates.reindex(merged.index, method='ffill'))[cols]
            merged = merged.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
            all_merged.append([np.hstack([np.vstack(merged.loc[nuc].index.values), merged.loc[nuc][cols].values]) for nuc in nuclei])
    
        all_branchings = []
        for mr1, mr2 in zip(allmr_cmb, allmr_ebl):
            all_branchings.append(merge_marginal_rates(mr1, mr2))

        self.boosts = np.logspace(6, 14, 201)
        self.nuclei = nuclei
        self.all_rates = df_rates.values
        self.all_branchings = all_branchings
        self.marginal_light_yields = all_merged


class InteractionCore_UHECR_Source_old(InteractionCore):
    """Producing interaction matrices from CRPropA interaction files.
    It requires files for photodisintegration and for photomeson.  
    """
    
    def __init__(self, data_directory, target_photon_spectrum):
        """Requires a string specifying the directory where CRPropa 
        cross section files are stored (argument data_directory) 
        """
        
        self._construct_from_files(data_directory, target_photon_spectrum)
        self._genenerate_complete_matrices()

    def generate_marginal_rates(self, target_photons, data_directory, remove_dead_ends=True):
        """Generate a marginal rates matrix with cross section files from crpropa.
        
        branchings is a 2d matrix where each row represents a disintegration channel.
        The first three columns contain Z, N and a channel representation 6-digit number,
        the remaining columns contain the cross section in mb for the channel as a function 
        of energy.
        """

        boosts = np.logspace(-1, 12)

        eps_crpropa = np.genfromtxt(data_directory + 'eps.txt') / 1e3 # in GeV
        branchings = np.genfromtxt(data_directory + 'xs_pd.txt')

        # He4, He3, H3, H2, p, n
        daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
        Zd = np.array([d[0] for d in daughters])
        Ad = np.array([d[1] for d in daughters])
        
        nuclei = daughters[2::-1] + [(int(Z), int(Z)+int(N)) for Z, N in list(sorted(set(zip(branchings[:, 0], branchings[:, 1]))))]

        marginal_rates = [[] for _ in nuclei]
        for count, br_row in enumerate(branchings[:, :]):
            Z, N, A = int(br_row[0]), int(br_row[1]), int(br_row[1])+int(br_row[0])
            UHECR_SRFenergy = A * boosts # in GeV
        
            nprods = np.array(get_particle_numbers(int(br_row[2])))
            prods = np.array([int(np > 0) for np in nprods])

            # Creating remnant nucleus from channel
            Zrem, Arem = Z - Zd.dot(prods), A - Ad.dot(prods)

            cs_crpropa = br_row[3:]
            r_pdis = ir.interaction_rate_from_cross_section(UHECR_SRFenergy, A,
                    target_photons, eps_crpropa, cs_crpropa*mb_to_cm2)  / c_in_Mpc_sec # 1/Mpc
                                                            
            if (Zrem, Arem) not in nuclei:
                # Change remnant isomer. 
                # This only affects produced protons and neutrons since
                # the yields of other light particles do not change.
                if (Zrem-1, Arem) in nuclei:
                    Zrem -= 1
                elif (Zrem+1, Arem) in nuclei:
                    Zrem += 1
                elif (Zrem-2, Arem) in nuclei:
                    Zrem -= 2
                elif (Zrem+2, Arem) in nuclei:
                    Zrem += 2
                elif (Zrem-1, Arem-1) in nuclei:
                    Zrem -= 1
                    Arem -= 1
                elif (Zrem-2, Arem-1) in nuclei:
                    Zrem -= 2
                    Arem -= 1
                elif (Z == 3) and (A == 6):
                    Zrem, Arem = 2, 4
                else:
                    print(f'No suitable isomer found for remnant ({Zrem:2d}, {Arem:2d}) with mother ({Z:2d}, {A:2d})')
                    Zrem, Arem = 0, 0
                    # continue
            
            nucidx = nuclei.index((Z, A))
            # Largest fragment is not one of the small ones
            if np.any([(mr[0] == Zrem) and (mr[1] == Arem) for mr in marginal_rates[nucidx]]):
                idx = [j for j, mr in enumerate(marginal_rates[nucidx]) if (mr[0] == Zrem) and (mr[1] == Arem)][0]
                marginal_rates[nucidx][idx][2:] += r_pdis
            else:
                rates_large = np.zeros(len(boosts) + 2)
                rates_large[:2] = Zrem, Arem
                rates_large[2:] = r_pdis
                marginal_rates[nucidx].append(rates_large)
            
        # Remove branchings leading to nuclei not included
        if remove_dead_ends:
            new_marginal_rates = []
            for k, mr in enumerate(marginal_rates):
                stacked_mr = np.array(mr)
                if len(stacked_mr.shape) > 1:
                    if np.any(stacked_mr[:, 1] == 0):
                        tot = stacked_mr.sum(axis=0)
                        new_mr = stacked_mr[np.where(stacked_mr[:, 1] != 0)]
                        new_tot = new_mr.sum(axis=0)
                        new_mr[:, 2:] *= np.divide(tot, new_tot, where=new_tot!=0, out=np.zeros_like(tot))[2:]

                        new_marginal_rates.append([mr_row for mr_row in new_mr])
                    else:
                        new_marginal_rates.append(mr)
                else:
                    new_marginal_rates.append(mr)
            marginal_rates = new_marginal_rates

        return nuclei, marginal_rates

    def _construct_from_files(self, data_directory, target_photons):
        """Using CRPROPA cross sections to produce the rates for a source
        of UHECR with a background photon field as a broken power law.

        CRPropA cross section file contains  is structured in different files depending on the 
        interaction and the photon field.
        """

        boosts = np.logspace(-1, 12)
        e_pmes = np.logspace(-1, 4, 100)  # in GeV

        nuclei, all_pdis_rates = self.generate_marginal_rates(target_photons, data_directory, False)

        all_rates, pdis_rates, pprates, all_branchings, allmr_pdis = [], [], [], [], []
        for nucidx, (_, A) in enumerate(nuclei):
            UHECR_SRFenergy = A * boosts # in GeV
            
            cs_pmes = ir.cs_photomeson(e_pmes, A) # in cm2
            r_pmes = ir.interaction_rate_from_cross_section(UHECR_SRFenergy, A,
                                                        target_photons, e_pmes, cs_pmes) / c_in_Mpc_sec # 1/Mpc
            # r_pmes = np.zeros_like(r_pmes)
            pprates.append(r_pmes) # 1/Mpc

            if A < 6:
                # 4He and below do not have photodis. channels
                all_rates.append(r_pmes)

                mrval = np.zeros((2, len(r_pmes) + 2))
                mrval[0, 0], mrval[0, 1] = 0, 1
                mrval[1, 0], mrval[1, 1] = 1, 1
                allmr_pdis.append(mrval)
                continue

            if len(all_pdis_rates[nucidx]) > 1:
                r_pdis = np.array(all_pdis_rates[nucidx]).sum(axis=0)[2:]
            else:
                r_pdis = np.array(all_pdis_rates[nucidx])[2:]

            pdis_rates.append(r_pdis)
            mr_pdis = [[chr[0], chr[1]] + list(chr[2:]) for chr in all_pdis_rates[nucidx]]

            if len(mr_pdis) > 1:
                allmr_pdis.append(np.vstack(mr_pdis))
            else:
                allmr_pdis.append(np.array(mr_pdis))
            
            total_rate = (r_pdis + r_pmes) # 1/Mpc
            all_rates.append(total_rate)

        all_rates = np.vstack(all_rates)
        allmr_phpi = get_marginal_rates(nuclei, pprates, boosts, 'minimal')
        
        for mr1, mr2 in zip(allmr_pdis, allmr_phpi):
            all_branchings.append(merge_marginal_rates(mr1, mr2))
        
        self.boosts = boosts 
        self.nuclei = nuclei
        self.all_rates = all_rates
        self.all_branchings = all_branchings


class InteractionCore_UHECR_Source(InteractionCore):
    """ Producing interaction matrices from CRPropA interaction files.
        It requires files for photodisintegration and for photomeson.  
    """
    
    def __init__(self, data_directory, target_photon_spectrum, boostfactor=None):
        """ Requires a string specifying the directory where CRPropa
            cross section files are stored (argument data_directory)
        """

        self._construct_from_files(data_directory, target_photon_spectrum, boostfactor)
        self._genenerate_complete_matrices()

    def _construct_from_files(self, data_directory, target_photons, boostfactor=None):
        """Using CRPROPA cross sections to produce the rates for a source
        of UHECR with a background photon field as a broken power law.

        CRPropA cross section file contains  is structured in different files depending on the 
        interaction and the photon field.
        """
        from pandas import DataFrame
        boosts = np.logspace(0, 12, 131)

        if boostfactor is not None:
            boosts *= boostfactor

        cols = [f'{i}' for i in range(len(boosts))]

        _pd_dir = 'PD_Talys1.8_Khan' if os.path.isdir(data_directory + 'PD_Talys1.8_Khan') else 'PD_Talys1.9'
        eps_crpropa = np.genfromtxt(data_directory + f'{_pd_dir}/eps.txt') / 1e3 # in GeV
        branchings = np.genfromtxt(data_directory + f'{_pd_dir}/xs_pd_thin.txt')
        df_rates_pdis, df_brnch_pdis, merged_yields_pdis = \
            generate_photodisinteg_tables_from_cross_sections(eps_crpropa, branchings, target_photons, boosts=boosts)
        
        nuclei = [(int(Z), int(A)) for A, Z in df_rates_pdis.index.values]

        xsp = np.genfromtxt(data_directory + 'PPP/xs_proton.txt')
        xsn = np.genfromtxt(data_directory + 'PPP/xs_neutron.txt')
        xsp[:, 1] *= 1e-3 # mubarn to mbarn
        xsn[:, 1] *= 1e-3 # mubarn to mbarn
        df_rates_pmes, df_brnch_pmes, merged_yields_pmes = \
            generate_photomeson_tables_from_cross_sections(nuclei, xsp, xsn, target_photons, boosts=boosts)

        df_rates = df_rates_pdis.groupby(by=['A', 'Z']).sum() + df_rates_pmes.groupby(by=['A', 'Z']).sum()

        df_brnch_pdis[cols] = df_brnch_pdis.multiply(df_rates_pdis.reindex(df_brnch_pdis.index, method='ffill'))[cols]
        merged_pdis = df_brnch_pdis.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_pdis = [np.hstack([np.vstack(merged_pdis.loc[nuc].index.values), merged_pdis.loc[nuc][cols].values]) for nuc in nuclei]

        # IMPORTANT: pmes_branches already are rescaled, following line is not needed
        # df_brnch_pmes[cols] = df_brnch_pmes.multiply(df_rates_pmes.reindex(df_brnch_pmes.index, method='ffill'))[cols]
        merged_pmes = df_brnch_pmes.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_pmes = [np.hstack([np.vstack(merged_pmes.loc[nuc].index.values), merged_pmes.loc[nuc][cols].values]) for nuc in nuclei]

        all_branchings = []
        for mr1, mr2 in zip(allmr_pdis, allmr_pmes):
            all_branchings.append(merge_marginal_rates(mr1, mr2))

        all_merged = []
        for mypdis, mypmes in zip(merged_yields_pdis, merged_yields_pmes):                    
            light_yield_pdis = mypdis[cols].multiply(df_rates_pdis.reindex(mypdis[cols].index, method='ffill'))
            light_yield_pmes = mypmes[cols].multiply(df_rates_pmes.reindex(mypmes[cols].index, method='ffill'))

            merged_cols = light_yield_pdis.add(light_yield_pmes, fill_value=0)[cols]
            merged = DataFrame(data=np.hstack([np.vstack(merged_cols.index.values), merged_cols.values]), index=merged_cols.index, columns=['A', 'Z', 'Ar', 'Zr'] + cols)

            merged[cols] = merged.divide(df_rates.reindex(merged.index, method='ffill'))[cols]
            merged = merged.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
            all_merged.append([np.hstack([np.vstack(merged.loc[nuc].index.values), merged.loc[nuc][cols].values]) for nuc in nuclei])

        self.boosts = boosts
        self.nuclei = nuclei
        self.all_rates = df_rates.values
        self.all_branchings = all_branchings
        self.marginal_light_yields = all_merged


class InteractionCore_Source(InteractionCore):
    """ Producing interaction matrices from CRPropA interaction files.
        Using cross section models for photodisintegration and for photomeson.
    """

    def __init__(self, epsrange, target_photon_spectrum, path=None, xsec_model=None, nuclear_decay_On=False):
        """ Based on a cross section model
        """

        if not (path is None):
            self.path = path
            from .photonuclear_cross_sections import CRPropa_model

            self.sim_model = CRPropa_model(path=self.path)
        elif not (xsec_model is None):
            self.sim_model = xsec_model
        else:
            raise ValueError('Error: no cross sections provided.')

        self.epsrange = epsrange
        self.target_photons = target_photon_spectrum

        InteractionCore.__init__(self, nuclear_decay_On)


    def _construct_from_files(self):
        """Using model cross sections to produce the rates for a source
        of UHECR with the background photon field provided.
        """
        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section

        boosts = np.logspace(0, 12, 131)
        eps = np.logspace(np.log10(self.epsrange[0]), np.log10(self.epsrange[1]), 300) # in GeV
        eps = np.logspace(-2, 6, 300) # in GeV, for the cross section

        pdis_rates_cmb, branchings_cmb, mlyp, mlyn = [], [], [], []
        for (Z, A), products in zip(self.sim_model.nuclei, self.sim_model.channels):
            branchings, lyp, lyn = [], [], []

            for Zrem, Arem in products:
                cross_section = 1e-27 * self.sim_model.cross_section(eps * 1e3, Z, A, rem=(Zrem, Arem)) # to cm2
                print()
                pdis_rates = interaction_rate_from_cross_section(boosts * A * .939, A, self.target_photons, eps, cross_section)
                pdis_rates /= c / parsec / 1e6 # ito Mpc

                branchings.append(np.append([Zrem, Arem], pdis_rates))
                lyp.append(np.append([Zrem, Arem], (Z - Zrem) * pdis_rates))
                lyn.append(np.append([Zrem, Arem], (A - Z - Arem + Zrem) * pdis_rates))

            if lyp != []:
                mlyp.append(np.vstack(lyp))
            else:
                mlyp.append(np.array([]))

            if lyn != []:
                mlyn.append(np.vstack(lyn))
            else:
                mlyn.append(np.array([]))

            pdis_rates_cmb.append(np.sum(np.atleast_2d((branchings)), axis=0)[2:])
            branchings_cmb.append(branchings)

        branchings_cmb = [np.vstack(br) for br in branchings_cmb]
        marginal_light_yields = [[np.atleast_2d(np.hstack([br[:, :2], np.zeros_like(br[:, 2:])])) for br in branchings_cmb] for _ in range(4)]
        marginal_light_yields.append(mlyp)
        marginal_light_yields.append(mlyn)

        self.boosts = boosts
        self.nuclei = self.sim_model.nuclei.copy()
        self.all_rates = np.vstack(pdis_rates_cmb)
        self.all_branchings = branchings_cmb
        self.marginal_light_yields = marginal_light_yields

        # adding photomeson rates
        # rates_pmes, allmr_pmes, merged_pmes = self._construct_photomeson_superposition()
        # self.all_rates += rates_pmes

        # new_branchings = [mr1 + mr2 for mr1, mr2 in zip(allmr_pmes, self.all_branchings)]
        # self.all_branchings = new_branchings

    def _construct_photomeson_superposition(self, data_directory=None, boostfactor=None):
        """Using CRPROPA cross sections to produce the rates for a source
        of UHECR with a background photon field as a broken power law.

        CRPropA cross section file contains  is structured in different files depending on the
        interaction and the photon field.
        """
        cols = [f'{i}' for i in range(len(self.boosts))]

        xsp = np.genfromtxt(data_directory + 'PPP/xs_proton.txt')
        xsn = np.genfromtxt(data_directory + 'PPP/xs_neutron.txt')
        xsp[:, 1] *= 1e-3 # mubarn to mbarn
        xsn[:, 1] *= 1e-3 # mubarn to mbarn

        df_rates_pmes, df_brnch_pmes, merged_yields_pmes = \
            generate_photomeson_tables_from_cross_sections(self.nuclei, xsp, xsn, self.target_photons, boosts=self.boosts)

        # preparing all rates
        rates_pmes = np.vstack([df_rates_pmes.loc[(nuc[1], nuc[0])] for nuc in self.nuclei])

        # preparing marginal rates
        new_mr = []
        for nuc, rate, branches in zip(self.nuclei, rates_pmes, self.all_branchings):
            branch_set = []
            for branch in branches:
                if (nuc[1], nuc[0], branch[1], branch[0]) in df_brnch_pmes.index:
                    branch_set.append(df_brnch_pmes.loc[(nuc[1], nuc[0], branch[1], branch[0])].values[2:])
                else:
                    branch_set.append(np.zeros(133))

            new_mr.append(np.vstack(branch_set))

        return rates_pmes, new_mr, merged_yields_pmes


class InteractionCore_PSB_CMB(InteractionCore):
    def _construct_from_files(self):
        """Based on PSB-model of nuclear cascades
        """
        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section
        from .background_photon_models import cmb_photon_density_GeVcm3
        from .photonuclear_cross_sections import PSB_model

        boosts = np.logspace(6, 14, 201)
        eps = 1e-3 * np.linspace(5, 50, 200) # in GeV

        psb_model = PSB_model()
        reversed_by_mass = psb_model.params.sort_values(by=['A', 'Z'], ascending=True)

        nuclei, pdis_rates_cmb, branchings_cmb, mlyp, mlyn = [], [], [], [], []
        for Z, A in zip(reversed_by_mass['Z'], reversed_by_mass['A']):
            nuclei.append((int(Z), int(A)))

            branchings, lyp, lyn = [], [], []
            for nloss in range(1, 16): # only up to 15 possible
                Arem = int(A - nloss)

                if Arem < 1:
                    continue
                elif Arem == 1:
                    Zrem = 1
                elif Arem in [5, 6, 7, 8]:
                    Arem, Zrem = 4, 2
                else:
                    Zrem = int(psb_model.params[psb_model.params['A'] == Arem]['Z'].iloc[0])

                cross_section = 1e-27 * psb_model.cross_section(eps * 1e3, Z, A, nloss) # to cm2
                pdis_rates = interaction_rate_from_cross_section(A*boosts, A, cmb_photon_density_GeVcm3, eps, cross_section)
                pdis_rates /= c / parsec / 1e6 # ito Mpc

                branchings.append(np.append([Zrem, Arem], pdis_rates))
                lyp.append(np.append([Zrem, Arem], (Z - Zrem) * pdis_rates))
                lyn.append(np.append([Zrem, Arem], (A - Z - Arem + Zrem) * pdis_rates))

            mlyp.append(np.vstack(lyp))
            mlyn.append(np.vstack(lyn))
            
            pdis_rates_cmb.append(np.sum(np.atleast_2d((branchings)), axis=0)[2:])
            branchings_cmb.append(branchings)

        branchings_cmb = [np.vstack(br) for br in branchings_cmb]
        marginal_light_yields = [[np.atleast_2d(np.hstack([br[:, :2], np.zeros_like(br[:, 2:])])) for br in branchings_cmb] for _ in range(4)]
        marginal_light_yields.append(mlyp)
        marginal_light_yields.append(mlyn)
            
        self.boosts = boosts 
        self.nuclei = nuclei
        self.all_rates = np.vstack(pdis_rates_cmb)
        self.all_branchings = branchings_cmb
        self.marginal_light_yields = marginal_light_yields


class InteractionCore_SimProp_CMB(InteractionCore):
    def __init__(self, M=1, nuclear_decay_On=False):
        self.M = M
        InteractionCore.__init__(self, nuclear_decay_On)

    def _construct_from_files(self):
        """Based on the cross section models implemented in SimProp v2.4
        """
        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section
        from .background_photon_models import cmb_photon_density_GeVcm3
        from .photonuclear_cross_sections import SimProp_model

        boosts = np.logspace(6, 14, 201)
        eps = 1e-3 * np.linspace(5, 50, 200) # in GeV

        sim_model = SimProp_model(M=self.M)

        pdis_rates_cmb, branchings_cmb, mlyp, mlyn = [], [], [], []
        for (Z, A), products in zip(sim_model.nuclei, sim_model.channels):
            branchings, lyp, lyn = [], [], []
            for Zrem, Arem in products:
                nloss = int(A - Arem)

                cross_section = 1e-27 * sim_model.cross_section(eps * 1e3, Z, A, nloss) # to cm2
                pdis_rates = interaction_rate_from_cross_section(A*boosts, A, cmb_photon_density_GeVcm3, eps, cross_section)
                pdis_rates /= c / parsec / 1e6 # ito Mpc

                branchings.append(np.append([Zrem, Arem], pdis_rates))
                lyp.append(np.append([Zrem, Arem], (Z - Zrem) * pdis_rates))
                lyn.append(np.append([Zrem, Arem], (A - Z - Arem + Zrem) * pdis_rates))

            mlyp.append(np.vstack(lyp))
            mlyn.append(np.vstack(lyn))
            
            pdis_rates_cmb.append(np.sum(np.atleast_2d((branchings)), axis=0)[2:])
            branchings_cmb.append(branchings)

        branchings_cmb = [np.vstack(br) for br in branchings_cmb]
        marginal_light_yields = [[np.atleast_2d(np.hstack([br[:, :2], np.zeros_like(br[:, 2:])])) for br in branchings_cmb] for _ in range(4)]
        marginal_light_yields.append(mlyp)
        marginal_light_yields.append(mlyn)
            
        self.boosts = boosts 
        self.nuclei = sim_model.nuclei.copy()
        self.all_rates = np.vstack(pdis_rates_cmb)
        self.all_branchings = branchings_cmb
        self.marginal_light_yields = marginal_light_yields


class InteractionCore_GDRA_CMB(InteractionCore):
    def _construct_from_files(self):
        """Based on the cross sections from the GDR atlas
        """
        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section
        from .background_photon_models import cmb_photon_density_GeVcm3
        from .photonuclear_cross_sections import GDR_atlas

        boosts = np.logspace(6, 14, 201)
        eps = 1e-3 * np.linspace(5, 50, 200) # in GeV

        sim_model = GDR_atlas()

        pdis_rates_cmb, branchings_cmb, mlyp, mlyn = [], [], [], []
        for (Z, A), products in zip(sim_model.nuclei, sim_model.channels):
            branchings, lyp, lyn = [], [], []
            for Zrem, Arem in products:
                nloss = int(A - Arem)

                cross_section = 1e-27 * sim_model.cross_section(eps * 1e3, Z, A, nloss) # to cm2
                pdis_rates = interaction_rate_from_cross_section(A*boosts, A, cmb_photon_density_GeVcm3, eps, cross_section)
                pdis_rates /= c / parsec / 1e6 # ito Mpc

                branchings.append(np.append([Zrem, Arem], pdis_rates))
                lyp.append(np.append([Zrem, Arem], (Z - Zrem) * pdis_rates))
                lyn.append(np.append([Zrem, Arem], (A - Z - Arem + Zrem) * pdis_rates))

            mlyp.append(np.vstack(lyp))
            mlyn.append(np.vstack(lyn))
            
            pdis_rates_cmb.append(np.sum(np.atleast_2d((branchings)), axis=0)[2:])
            branchings_cmb.append(branchings)

        branchings_cmb = [np.vstack(br) for br in branchings_cmb]
        marginal_light_yields = [[np.atleast_2d(np.hstack([br[:, :2], np.zeros_like(br[:, 2:])])) for br in branchings_cmb] for _ in range(4)]
        marginal_light_yields.append(mlyp)
        marginal_light_yields.append(mlyn)
            
        self.boosts = boosts 
        self.nuclei = sim_model.nuclei.copy()
        self.all_rates = np.vstack(pdis_rates_cmb)
        self.all_branchings = branchings_cmb
        self.marginal_light_yields = marginal_light_yields


class InteractionCore_CRPdata_CMB(InteractionCore):
    def __init__(self, path=None, nuclear_decay_On=False, xsec_model=None):
        if not (path is None):
          self.path = path
          from .photonuclear_cross_sections import CRPropa_model
          self.sim_model = CRPropa_model(path=self.path)
        elif not (xsec_model is None):
          self.sim_model = xsec_model
        else:
          raise ValueError('Error: no cross sections provided.')

        InteractionCore.__init__(self, nuclear_decay_On)

    def _construct_from_files(self):
        """Based on the cross sections data files used in CRPropa
        """
        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section
        from .background_photon_models import cmb_photon_density_GeVcm3

        boosts = np.logspace(6, 14, 201)
        eps = 1e-3 * np.logspace(-1, 2.1, 300) # in GeV

        pdis_rates_cmb, branchings_cmb, mlyp, mlyn = [], [], [], []
        for (Z, A), products in zip(self.sim_model.nuclei, self.sim_model.channels):
            branchings, lyp, lyn = [], [], []
            
            for Zrem, Arem in products:
                cross_section = 1e-27 * self.sim_model.cross_section(eps * 1e3, Z, A, rem=(Zrem, Arem)) # to cm2
                pdis_rates = interaction_rate_from_cross_section(boosts * A * .939, A, cmb_photon_density_GeVcm3, eps, cross_section)
                pdis_rates /= c / parsec / 1e6 # ito Mpc

                branchings.append(np.append([Zrem, Arem], pdis_rates))
                lyp.append(np.append([Zrem, Arem], (Z - Zrem) * pdis_rates))
                lyn.append(np.append([Zrem, Arem], (A - Z - Arem + Zrem) * pdis_rates))

            if lyp != []:
                mlyp.append(np.vstack(lyp))
            else:
                mlyp.append(np.array([]))
            
            if lyn != []:                
                mlyn.append(np.vstack(lyn))
            else:
                mlyn.append(np.array([]))
            
            pdis_rates_cmb.append(np.sum(np.atleast_2d((branchings)), axis=0)[2:])
            branchings_cmb.append(branchings)

        branchings_cmb = [np.vstack(br) for br in branchings_cmb]
        marginal_light_yields = [[np.atleast_2d(np.hstack([br[:, :2], np.zeros_like(br[:, 2:])])) for br in branchings_cmb] for _ in range(4)]
        marginal_light_yields.append(mlyp)
        marginal_light_yields.append(mlyn)
            
        self.boosts = boosts 
        self.nuclei = self.sim_model.nuclei.copy()
        self.all_rates = np.vstack(pdis_rates_cmb)
        self.all_branchings = branchings_cmb
        self.marginal_light_yields = marginal_light_yields


class InteractionCore_CRPdata_EBL(InteractionCore):
    def __init__(self, path=None, nuclear_decay_On=False, xsec_model=None, z=0):
        self.z = z
        if not (path is None):
          self.path = path
          from .photonuclear_cross_sections import CRPropa_model
          self.sim_model = CRPropa_model(path=self.path)
        elif not (xsec_model is None):
          self.sim_model = xsec_model
        else:
          raise ValueError('Error: no cross sections provided.')

        InteractionCore.__init__(self, nuclear_decay_On)

    def _construct_from_files(self):
        """Based on the cross sections data files used in CRPropa
        """
        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section
        from .background_photon_models import eblg_interp

        eblmodel = lambda energ: eblg_interp(energ * 1e9, self.z).flatten() * 1e3

        boosts = np.logspace(6, 14, 201)
        eps = 1e-3 * np.logspace(-1, 2.1, 300) # in GeV

        pdis_rates_cmb, branchings_cmb, mlyp, mlyn = [], [], [], []
        for (Z, A), products in zip(self.sim_model.nuclei, self.sim_model.channels):
            branchings, lyp, lyn = [], [], []
            
            for Zrem, Arem in products:
                cross_section = 1e-27 * self.sim_model.cross_section(eps * 1e3, Z, A, rem=(Zrem, Arem)) # to cm2
                pdis_rates = interaction_rate_from_cross_section(boosts * A * .939, A, eblmodel, eps, cross_section)
                pdis_rates /= c / parsec / 1e6 # ito Mpc

                branchings.append(np.append([Zrem, Arem], pdis_rates))
                lyp.append(np.append([Zrem, Arem], (Z - Zrem) * pdis_rates))
                lyn.append(np.append([Zrem, Arem], (A - Z - Arem + Zrem) * pdis_rates))

            if lyp != []:
                mlyp.append(np.vstack(lyp))
            else:
                mlyp.append(np.array([]))
            
            if lyn != []:                
                mlyn.append(np.vstack(lyn))
            else:
                mlyn.append(np.array([]))
            
            pdis_rates_cmb.append(np.sum(np.atleast_2d((branchings)), axis=0)[2:])
            branchings_cmb.append(branchings)

        branchings_cmb = [np.vstack(br) for br in branchings_cmb]
        marginal_light_yields = [[np.atleast_2d(np.hstack([br[:, :2], np.zeros_like(br[:, 2:])])) for br in branchings_cmb] for _ in range(4)]
        marginal_light_yields.append(mlyp)
        marginal_light_yields.append(mlyn)
            
        self.boosts = boosts 
        self.nuclei = self.sim_model.nuclei.copy()
        self.all_rates = np.vstack(pdis_rates_cmb)
        self.all_branchings = branchings_cmb
        self.marginal_light_yields = marginal_light_yields
