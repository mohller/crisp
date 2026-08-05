"""Nuclear cascade transport: the InteractionCore class.

This module defines `InteractionCore`, the class that assembles a cross
section model (see `photonuclear_cross_sections`), an optional photon
field, decay data, and photomeson kernels into interaction tensors, and
solves the resulting transport equation exactly as a matrix exponential.
Given an injected composition, `InteractionCore.species_evolution_boost_range`
returns the surviving and fragment nuclear population at any distance
from the source, and a family of related methods
(`pion_production`, `reprocessed_nucleons`, `neutrino_production`,
`light_cascade_production`, and others) fold that population against the
relevant secondary production kernel to return pions, nucleons,
neutrinos, or light nuclei produced along the way.

The same interaction tensors also support a second, complementary view
of the cascade: instead of "how much of the injected population is left
at distance L", `cdf_boost_range`/`pdf_boost_range` (and their moments,
`pdf_moments_boost_range`/`pdf_variance_boost_range`) give the closed
form probability distribution of the distance itself at which a nucleus
first interacts or is absorbed, i.e. a survival-analysis style
"distance until absorption" distribution for a given injected species
and boost.

See the package docstring (`crisp/__init__.py`, `import crisp; help(crisp)`)
for a minimal worked example and how this module fits with the rest of
the package.
"""

import logging
import os
import numpy as np
from math import factorial
from scipy.linalg import expm
from scipy.interpolate import interp1d
from .interaction_rates import (cs_photomeson, interaction_rate_from_cross_section,
                                exact_rates_for_sigma)
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


def build_photomeson_species_kernels(boosts, target_photons, inelasticity=None,
                                     mp_GeV=0.939, K_multipion=0.6):
    """Charge- and species-resolved photomeson production kernels, built as a
    sum over the interaction types of the Rachen parametrization
    (pgamma_components: resonances, direct, multi-pion) in the spirit of the
    simplified SOPHIA models of Huemmer et al. (2010, ApJ 721, 630).

    K[b][s, i, j] is the rate [Mpc⁻¹] at which a proton (s=0) or neutron
    (s=1) parent at boosts[i] produces secondary b at boosts[j], for
    b in {'pi+', 'pi-', 'pi0', 'p', 'n'}. For a nucleus (Z, A):
    Z * K[b][0] + (A-Z) * K[b][1].

    Per interaction type (multiplicities for proton parents; neutron parents
    are the isospin mirror pi+ <-> pi-, p <-> n):

      resonances : Delta isospin — pi0 2/3 (nucleon stays p), pi+ 1/3
                   (nucleon -> n); pion at kappa(eps') of the parent energy,
                   nucleon at 1 - kappa(eps')
      direct     : t-channel gamma p -> n pi+ — pi+ 1 (nucleon -> n)
      multi-pion : charge democracy — one pion of each charge at
                   x = K_multipion / 3 each; nucleon p/n 1/2 each at
                   1 - K_multipion

    Each interaction type is normalized per parent row to the exact
    isotropic-field rate of its component cross section
    (exact_rates_for_sigma), so the total nucleon output equals the total
    interaction rate exactly, while the pion count exceeds it where
    multi-pion production dominates — the multiplicity effect of
    Huemmer et al. The head-on mapping eps' = 2 Gamma eps only shapes the
    secondary spectra.

    Arguments:
    ----------
    boosts         : 1-D array of Lorentz factors (parent boost grid)
    target_photons : callable, n_γ(ε) in GeV⁻¹ cm⁻³ with ε in GeV
    inelasticity   : None → parametric κ̄(ε') (SOPHIA-motivated), or callable
    mp_GeV         : nucleon mass [GeV] used in the pion-boost mapping
    K_multipion    : total inelasticity of the multi-pion interaction type

    Returns:
    --------
    dict of ndarrays, shape (2, n_boost, n_boost) each
    """
    from .photonuclear_cross_sections import pgamma_components

    mpi_GeV = 0.140   # pion mass in GeV
    eps_th  = 0.145   # pion production threshold in nucleon rest frame (GeV)

    if inelasticity is None:
        def kappa(eps_prime):
            # SOPHIA-motivated: κ̄ ≈ 0.20 at threshold, rising to 0.50 at high ε'
            return np.minimum(0.2 + 0.07 * np.log10(np.maximum(eps_prime, eps_th) / eps_th), 0.5)
    else:
        kappa = inelasticity

    from scipy.constants import parsec
    Mpc_in_cm = parsec * 1e8   # cm per Mpc

    e_grid = np.logspace(np.log10(eps_th), 4.0, 500)   # rest-frame photon energies (GeV)
    de     = np.gradient(e_grid)
    comps  = pgamma_components(e_grid)                 # cm2 per interaction type
    kap    = kappa(e_grid)
    x_mp   = K_multipion / 3.0                         # per-pion fraction, multi-pion

    # per interaction type: sigma, pion fraction chi_pi(eps'), nucleon
    # fraction chi_N(eps'), multiplicities per species for a PROTON parent
    ones = np.ones_like(e_grid)
    ITS = {
        'res': (np.clip(comps['resonances'], 0, None), kap, 1.0 - kap,
                {'pi0': 2 / 3, 'pi+': 1 / 3, 'p': 2 / 3, 'n': 1 / 3}),
        'dir': (np.clip(comps['direct'], 0, None), kap, 1.0 - kap,
                {'pi+': 1.0, 'n': 1.0}),
        'mp':  (np.clip(comps['multipion'], 0, None), x_mp * ones,
                (1.0 - K_multipion) * ones,
                {'pi+': 1.0, 'pi-': 1.0, 'pi0': 1.0, 'p': 1 / 2, 'n': 1 / 2}),
    }
    MIRROR = {'pi+': 'pi-', 'pi-': 'pi+', 'pi0': 'pi0', 'p': 'n', 'n': 'p'}

    # exact per-IT interaction rates (one batched call)
    names = list(ITS)
    r_IT = exact_rates_for_sigma(boosts, target_photons, e_grid,
                                 np.vstack([1e27 * ITS[k][0] for k in names]))

    n_b = len(boosts)
    K = {b: np.zeros((2, n_b, n_b)) for b in MIRROR}
    logb = np.log(boosts)

    def deposit(row, gamma_sec, w):
        x = np.interp(np.log(gamma_sec), logb, np.arange(n_b),
                      left=-1.0, right=float(n_b - 1))
        valid = (x >= 0) & (w > 0)
        j0 = np.floor(x[valid]).astype(int)
        f = x[valid] - j0
        j1 = np.minimum(j0 + 1, n_b - 1)
        np.add.at(row, j0, (1.0 - f) * w[valid])
        np.add.at(row, j1, f * w[valid])

    for i, gamma in enumerate(boosts):
        eps_lab = e_grid / (2.0 * gamma)               # head-on mapping
        ng = target_photons(eps_lab)
        pref = (Mpc_in_cm / (2.0 * gamma)) * ng * de   # Mpc^-1 per sigma

        for k_IT, (sig, chi_pi, chi_N, mult) in zip(names, ITS.values()):
            dR = pref * sig
            total = dR[dR > 0].sum()
            if total <= 0.0:
                continue
            # interaction-count normalization to the exact isotropic rate
            scale = r_IT[names.index(k_IT), i] / total

            # one interaction-weighted deposit per placement, then distribute
            # into species by multiplicity (mirror for neutron parents)
            pion_row = np.zeros(n_b)
            nuc_row = np.zeros(n_b)
            deposit(pion_row, chi_pi * gamma * (mp_GeV / mpi_GeV), dR * scale)
            deposit(nuc_row, np.maximum(chi_N, 1e-12) * gamma, dR * scale)

            for b, M in mult.items():
                base = pion_row if b.startswith('pi') else nuc_row
                K[b][0, i] += M * base
                K[MIRROR[b]][1, i] += M * base

    return K


def build_photomeson_species_kernels_sophia(boosts, target_photons, pmm,
                                            mp_GeV=0.939):
    """Charge- and species-resolved photomeson kernels with the full SOPHIA
    x-distributions of an AstroPhoMes photomeson model (its redist_proton /
    redist_neutron tables): secondaries are spread over their complete
    energy-fraction distributions instead of the per-interaction-type mean
    placements of build_photomeson_species_kernels, and the neutron parent
    uses its own tables (no isospin-mirror assumption).

    Same output structure and rate convention: K[b][s, i, j] is the rate
    [Mpc^-1] at which parent species s (0 = p, 1 = n) at boosts[i] produces
    secondary b at boosts[j]; the total interaction count per parent row is
    normalized to the exact isotropic rate of the SOPHIA total cross
    section. Secondary multiplicities are the tables' own (nucleons: exactly
    one per interaction at the Delta, up to ~1.3 at very high eps_r from
    NNbar production; pions: the full multipion multiplicities). Also
    carries the strangeness channel absent from the interaction-type
    builder: 'K+' (table id 50, associated production, threshold
    eps_r ~ 1 GeV) and 'K-' (id 51, pair production, ~1.5 GeV).
    """
    from scipy.constants import parsec
    Mpc_in_cm = parsec * 1e8

    e_grid = np.asarray(pmm.egrid, dtype=float)        # eps_r [GeV]
    de = np.gradient(e_grid)
    xc = np.asarray(pmm.xcenters, dtype=float)
    xw = np.asarray(pmm.xwidths, dtype=float)

    mpi_GeV = 0.13957039
    mK_GeV = 0.493677
    PROD = {'pi+': 2, 'pi-': 3, 'pi0': 4, 'p': 101, 'n': 100,
            'K+': 50, 'K-': 51}
    m_sec = {'pi+': mpi_GeV, 'pi-': mpi_GeV, 'pi0': 0.1349768,
             'p': mp_GeV, 'n': mp_GeV, 'K+': mK_GeV, 'K-': mK_GeV}
    redists = (pmm.redist_proton, pmm.redist_neutron)
    sig_cm2 = (np.asarray(pmm.cs_proton_grid, dtype=float) * 1e-30,   # µb→cm²
               np.asarray(pmm.cs_neutron_grid, dtype=float) * 1e-30)

    # exact isotropic interaction rates per parent charge (µb → mb)
    r_ex = exact_rates_for_sigma(
        boosts, target_photons, e_grid,
        np.vstack([pmm.cs_proton_grid, pmm.cs_neutron_grid]) * 1e-3)

    # dN/dx dx per interaction, (n_e, n_x), per (parent charge, product)
    W = {(c, b): np.asarray(redists[c][PROD[b]], dtype=float) * xw[None, :]
         for c in (0, 1) for b in PROD}

    n_b = len(boosts)
    K = {b: np.zeros((2, n_b, n_b)) for b in PROD}

    for i, gamma in enumerate(boosts):
        eps_lab = e_grid / (2.0 * gamma)               # head-on mapping
        ng = target_photons(eps_lab)
        pref = (Mpc_in_cm / (2.0 * gamma)) * ng * de   # Mpc^-1 per sigma

        for c in (0, 1):
            dR = pref * sig_cm2[c]
            total = dR[dR > 0].sum()
            if total <= 0.0:
                continue
            dRs = dR * (r_ex[c, i] / total)            # exact-rate normalized
            for b in PROD:
                # collapse over eps_r first: the placement x-grid is shared
                wx = dRs @ W[(c, b)]                   # (n_x,) rate weights
                K[b][c, i] += deposit_log_cic(
                    boosts, xc * gamma * (mp_GeV / m_sec[b]), wx)

    return K


def build_pion_prod_kernel(boosts, target_photons, inelasticity=None, mp_GeV=0.939):
    """Charge-summed pion production kernel (pi+ + pi- + pi0), kept for
    backward compatibility — see build_photomeson_species_kernels for the
    charge- and species-resolved version and the interaction-type physics.

    K[s, i, j] is the rate [Mpc⁻¹] at which a proton (s=0) or neutron (s=1)
    at boosts[i] produces a pion (any charge) at boosts[j]; for a nucleus
    (Z, A): Z * K[0] + (A-Z) * K[1]. Row sums equal the exact interaction
    rate times the pion multiplicity of the mix (> 1 where multi-pion
    production dominates).
    """
    Ks = build_photomeson_species_kernels(boosts, target_photons,
                                          inelasticity=inelasticity, mp_GeV=mp_GeV)
    return Ks['pi+'] + Ks['pi-'] + Ks['pi0']


def build_proton_recoil_kernel(boosts, target_photons, inelasticity=None):
    """Secondary-nucleon (p + n) production kernel, kept for backward
    compatibility — see build_photomeson_species_kernels for the p/n-resolved
    version. One nucleon leaves every interaction, so the row sums equal the
    exact isotropic-field interaction rate identically.
    """
    Ks = build_photomeson_species_kernels(boosts, target_photons,
                                          inelasticity=inelasticity)
    return Ks['p'] + Ks['n']


def deposit_log_cic(boosts, gammas, weights):
    """Conservative cloud-in-cell deposit of weights at arbitrary boosts
    gammas onto the bins of a log-spaced boost grid (the kernels' placement
    scheme): weight below boosts[0] is dropped, weight above boosts[-1] goes
    to the top bin. Returns an array of len(boosts)."""
    boosts = np.asarray(boosts, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_b = len(boosts)
    row = np.zeros(n_b)
    x = np.interp(np.log(gammas), np.log(boosts), np.arange(n_b),
                  left=-1.0, right=float(n_b - 1))
    valid = (x >= 0) & (weights > 0)
    j0 = np.floor(x[valid]).astype(int)
    f = x[valid] - j0
    j1 = np.minimum(j0 + 1, n_b - 1)
    np.add.at(row, j0, (1.0 - f) * weights[valid])
    np.add.at(row, j1, f * weights[valid])
    return row


def shift_log_boost(boosts, state, shift_bins):
    """Shift a (n_boost, n_species) occupation array down in ln(gamma) by
    shift_bins, given directly in bin units: scalar (a coherent, species-
    and boost-independent drift, e.g. adiabatic cooling at Gamma c / R) ->
    exact gather (np.interp), no numerical diffusion; 1D per-bin array,
    shared by every species column (a dispersive drift evaluated at each
    source bin's own boost) -> conservative deposit via deposit_log_cic,
    applied once per column at gammas = boosts * exp(-shift_bins * dlnb)
    (the absolute-boost position equivalent to the bin-space shift on a
    log-uniform grid); 2D (n_boost, n_species) array, one shift per
    (bin, species) pair (e.g. synchrotron, genuinely different per
    species) -> the same conservative deposit, vectorized across all
    columns at once (bin-index space directly -- dst = bin_index -
    shift_bins -- the exact algebraic reduction of the 1D branch's
    gammas round-trip, skipped here since it would otherwise mean one
    deposit_log_cic call per species column). Counts pushed below the
    grid floor are dropped in every path (they leave the tracked
    window); a caller wanting a per-species-different SCALAR drift
    (e.g. Z^2/A-scaled pair losses, one number per species) calls this
    once per column with that column's own scalar shift_bins instead.
    """
    n_b = state.shape[0]
    x = np.arange(n_b, dtype=float)
    shift_bins = np.asarray(shift_bins, dtype=float)
    out = np.zeros_like(state)
    if shift_bins.ndim == 0:
        src = x + float(shift_bins)
        for c in range(state.shape[1]):
            out[:, c] = np.interp(src, x, state[:, c], left=0.0, right=0.0)
        return out
    if shift_bins.ndim == 2:
        if shift_bins.shape != state.shape:
            raise ValueError(f'2D shift_bins must match state shape {state.shape}, '
                             f'got {shift_bins.shape}')
        dst = x[:, None] - shift_bins                    # (n_b, n_col)
        j0 = np.floor(dst).astype(int)
        f = dst - j0
        col = np.broadcast_to(np.arange(state.shape[1]), state.shape)
        for jj, wt in ((j0, (1.0 - f) * state), (j0 + 1, f * state)):
            ok = (jj >= 0) & (jj < n_b)
            np.add.at(out, (jj[ok], col[ok]), wt[ok])
        return out
    dlnb = np.log(boosts[1] / boosts[0])
    gammas = np.asarray(boosts, dtype=float) * np.exp(-shift_bins * dlnb)
    for c in range(state.shape[1]):
        out[:, c] = deposit_log_cic(boosts, gammas, state[:, c])
    return out


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
    """Assembles interaction tensors from a cross section model, photon
    field, decay data, and photomeson kernels, and solves the resulting
    nuclear cascade transport equation exactly as a matrix exponential.

    Two families of methods are built on the same underlying tensors:
    `species_evolution_boost_range` and the secondary production methods
    on top of it (`pion_production`, `reprocessed_nucleons`,
    `neutrino_production`, `light_cascade_production`, and others) give
    the propagated population, or a secondary yield, at chosen
    distances; `cdf_boost_range`/`pdf_boost_range` (and their moments,
    `pdf_moments_boost_range`/`pdf_variance_boost_range`) give the
    complementary distance-until-absorption probability distribution for
    an injected species and boost.

    See `__init__` for the constructor arguments and a worked example,
    and the package docstring (`import crisp; help(crisp)`) for how this
    class fits with the rest of the package.
    """

    @staticmethod
    def _legacy_mass_GeV(Z, A):
        """masses='legacy' mass function (A * 0.939 GeV), kept for
        equivalence tests against the pre-consolidation classes. A
        staticmethod rather than a lambda so InteractionCore stays
        picklable."""
        return A * 0.939

    def __init__(self, nuclear_decay_On=False, ftype=np.float64, decays=None,
                 xsec_model=None, target_photons=None, photomeson=None,
                 photomeson_scaling=None, photomeson_spectra=None,
                 boosts=None, eps=None, masses='nubase', rate_method='fft'):
        """
        Parameters
        ----------
        nuclear_decay_On : if True, spontaneous decays of the TRACKED species
                 enter the main tensor as jump rates lambda(gamma) =
                 1/(gamma tau c) per branching (K40 -> Ca40 for beta-, ...;
                 emitted alpha/p/n/d/t are boost-preserving light yields,
                 beta leptons untracked). The nubase table is auto-loaded
                 when decays= is not given. Ultrafast particle emitters
                 (tau < 1 ns) stay construction-time resolved. The decay
                 component is reported as rates_by_interaction()['decay'].
                 Default False: decays affect only untracked products.
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
        photomeson : None, 'kernels' or 'superposition' (aliases). When active:
                 (a) the parametric spectra kernels are attached
                 (pion_prod_tensor, proton_recoil_tensor, enabling
                 pion/recoil/neutrino production); (b) photomeson interactions
                 enter the cascade rates — if xsec_model contains no
                 photomeson-group model (interaction_type == 'photomeson',
                 e.g. the AstroPhoMes Photomeson wrapper inside a Model_Rack),
                 a Photomeson_Superposition model over the same nuclei is
                 added automatically; (c) free p/n absorb at the photomeson
                 rate in the light sector (photomeson_rates_pn hook).
                 Only used with xsec_model.
        photomeson_scaling : None (default) or 'inclusive'. With 'inclusive',
                 the photomeson folds (pion / recoil / neutrino production)
                 rescale each nucleus row of the A = 1 kernels from the
                 superposition Z K_p + N K_n to the rack's photomeson-model
                 inclusive cross sections: per-parent-boost rate ratios
                 sigma_A / (Z sigma_p + N sigma_n) per pion charge (and the
                 nonelastic ratio for the recoil nucleons), e.g. the
                 A^alpha_pi(E) empirical scaling of the AstroPhoMes
                 EmpiricalModel (Morejon et al. 2019). Requires
                 photomeson='kernels' and a Photomeson(pmm=...) member in
                 xsec_model. The factors correct normalizations only — the
                 A = 1 kinematic shapes of the kernels are kept, and they are
                 not persisted by save().
        photomeson_spectra : None (default: the Rachen interaction-type
                 kernels with mean per-IT secondary placements) or an
                 AstroPhoMes photomeson model instance (load_astrophomes()):
                 the pion/nucleon kernels are then built from its full
                 SOPHIA x-distributions
                 (build_photomeson_species_kernels_sophia), charge-resolved
                 without the isospin mirror, and the free-nucleon rate hook
                 uses the SOPHIA total cross sections for consistency.
                 Requires photomeson='kernels'.
        boosts : Lorentz-factor grid. Default: np.logspace(6, 14, 201).
        eps : photon-energy grid in GeV for the rate integrals.
                 Default: 1e-3 * np.logspace(-1, 2.1, 300).
        masses : nuclear masses used for energy <-> boost conversions and
                 species_masses: 'nubase' (real ground-state masses from the
                 nuclear data table), 'legacy' (A * 0.939 GeV), or a callable
                 m(Z, A) in GeV. The photodisintegration tensors are computed
                 on the boost grid and do not depend on this choice.
        rate_method : 'fft' (default) computes all channel rates with a single
                 log-space convolution per construction (compute_rates), with
                 the analytic 1/y^2 continuation above the cross-section
                 support; 'direct' integrates each channel per boost
                 (interaction_rate_from_cross_section_boosts). The two agree
                 at the percent level below the boost where 2*Gamma*eps_peak
                 reaches the sigma support, above which 'direct' progressively
                 underestimates the rates. Only used with xsec_model.

        Examples
        --------
        The self-contained PSB cross sections need no external data
        download, so this constructs and uses a core in one step:

        >>> from crisp.core import InteractionCore
        >>> from crisp.photonuclear_cross_sections import PSB_model
        >>> core = InteractionCore(xsec_model=PSB_model())
        >>> len(core.species)
        53

        A core with photomeson kernels active (needed for
        `pion_production`/`neutrino_production`/`proton_recoil_production`)
        and the CMB as the default photon field:

        >>> core = InteractionCore(xsec_model=PSB_model(), photomeson='kernels')
        """
        self.ftype = ftype

        if decays is True or (nuclear_decay_On and decays is None):
            from .data.nucleardecays import NuclearDataTable
            decays = NuclearDataTable().prepare_decay_table()
        self.decays = decays if isinstance(decays, dict) else None
        self.nuclear_decay_On = bool(nuclear_decay_On)

        if masses == 'nubase':
            from .data.nucleardecays import nuclear_mass_GeV
            self._mass_fn = nuclear_mass_GeV
        elif masses == 'legacy':
            self._mass_fn = InteractionCore._legacy_mass_GeV
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
            if eps is not None:
                self.eps = np.asarray(eps)
            elif photomeson is not None:
                # the grid must also cover the photomeson region (0.145 - 1e4 GeV)
                self.eps = np.logspace(-4, 4, 650)
            else:
                self.eps = 1e-3 * np.logspace(-1, 2.1, 300)

            if rate_method not in ('fft', 'direct'):
                raise ValueError("rate_method must be 'fft' or 'direct'")
            self.rate_method = rate_method

            if photomeson is not None:
                if photomeson not in ('kernels', 'superposition'):
                    raise NotImplementedError(
                        "photomeson must be None, 'kernels' or 'superposition'; "
                        "richer photomeson models join the cascade as members of a "
                        "Model_Rack passed as xsec_model.")

                # secondary spectra (pion / recoil / neutrino production):
                # charge- and species-resolved kernels — per interaction type
                # with mean placements (Huemmer+10-style, default) or from
                # the full SOPHIA x-distributions of an AstroPhoMes model
                if photomeson_spectra is not None:
                    self.photomeson_kernels = build_photomeson_species_kernels_sophia(
                        self.boosts, self.target_photons, photomeson_spectra,
                        mp_GeV=self._mass_fn(1, 1))
                else:
                    self.photomeson_kernels = build_photomeson_species_kernels(
                        self.boosts, self.target_photons, mp_GeV=self._mass_fn(1, 1))
                Ks = self.photomeson_kernels
                self.pion_prod_tensor     = Ks['pi+'] + Ks['pi-'] + Ks['pi0']
                self.proton_recoil_tensor = Ks['p'] + Ks['n']

                # cascade rates: make sure a photomeson cross-section model is in
                # the rack — if the user did not supply one, add the superposition
                # model over the photodisintegration nuclei
                pm_models = self._collect_photomeson_models(xsec_model)
                if not pm_models:
                    from .photonuclear_cross_sections import Model_Rack, Photomeson_Superposition
                    pm = Photomeson_Superposition(xsec_model.nuclei)
                    xsec_model = Model_Rack(models=(xsec_model, pm))
                    self.xsec_model = xsec_model
                    self.sim_model = xsec_model
                    pm_models = [pm]

                # free p/n interact via the light-sector hook — with SOPHIA
                # spectra, from the same total cross sections as the kernels
                if photomeson_spectra is not None:
                    self.photomeson_rates_pn = exact_rates_for_sigma(
                        self.boosts, self.target_photons,
                        np.asarray(photomeson_spectra.egrid, dtype=float),
                        np.vstack([photomeson_spectra.cs_proton_grid,
                                   photomeson_spectra.cs_neutron_grid]) * 1e-3
                    ).mean(axis=0)
                else:
                    self.photomeson_rates_pn = self._photomeson_rates_A1(pm_models)

                if photomeson_scaling is not None:
                    if photomeson_scaling != 'inclusive':
                        raise ValueError("photomeson_scaling must be None or 'inclusive'")
                    carriers = [m for m in pm_models
                                if hasattr(m, 'inclusive_cross_section')]
                    if not carriers:
                        raise ValueError(
                            "photomeson_scaling='inclusive' needs a Photomeson(pmm=...) "
                            "member in xsec_model: the auto-added superposition model "
                            "carries no inclusive pion cross sections")
                    self._build_photomeson_fold_scaling(carriers[0])
            elif photomeson_scaling is not None:
                raise ValueError("photomeson_scaling requires photomeson='kernels'")
            elif photomeson_spectra is not None:
                raise ValueError("photomeson_spectra requires photomeson='kernels'")

            self._construct_from_xsec_model()
        else:
            self._construct_from_files()

        self._genenerate_complete_matrices()

    def _construct_from_xsec_model(self):
        """Build rates, branchings and light yields from self.xsec_model and
        self.target_photons on the self.boosts / self.eps grids.

        Boost-native: the rate integral only depends on the Lorentz factor,
        so no nuclear mass enters the photodisintegration tensors.

        With rate_method='fft' the rates of all channels of all nuclei are
        computed in one log-space convolution (compute_rates); with 'direct'
        each channel is integrated per boost.
        """
        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section_boosts

        boosts, eps = self.boosts, self.eps

        # photomeson channels eject their nucleon with a broad spectrum (the
        # recoil kernel carries the placement, cf. paper Sect. 2.4), so their
        # multiplicities are budgeted in self.photomeson_ejecta instead of the
        # boost-preserving light yields — the split needs each channel's
        # photomeson part separately from the total
        pm_models = self._collect_photomeson_models(self.xsec_model)

        if getattr(self, 'rate_method', 'fft') == 'fft':
            channel_rates = self._channel_rates_fft()
            channel_rates_pm = self._channel_rates_fft(models=pm_models) if pm_models else None

        pdis_rates_all, branchings_all, mlyp, mlyn = [], [], [], []
        mly_clusters = [[], [], [], []]        # He4, He3, t, d rows
        ejp_all, ejn_all = [], []
        row = 0
        for (Z, A), products in zip(self.xsec_model.nuclei, self.xsec_model.channels):
            branchings, lyp, lyn = [], [], []
            lyc = [[], [], [], []]
            ejp, ejn = np.zeros(len(boosts)), np.zeros(len(boosts))

            # boost-preserving LIGHT yields (He4, He3, t, d, p, n) of the
            # photodisintegration channels, from the model's inclusive
            # cross sections verbatim (multiplicities included, minus the
            # one-per-event channel allocation): the tables satisfy
            # sum_d A_d sigma_d = A sigma_nonel, so no Delta Z / Delta N
            # inference is needed — closure follows from the identity
            rates_ly = None
            for mdl in self._collect_pdis_models(self.xsec_model):
                if (Z, A) in getattr(mdl, 'nuclei', []) and \
                        hasattr(mdl, 'light_yield_sigma'):
                    sig6 = mdl.light_yield_sigma(eps * 1e3, Z, A)
                    if sig6 is not None:
                        rates_ly = self._rates_of_sigma_rows(sig6)
                    break

            # per-event fragment content of the photomeson interactions on
            # this mother, from the model's own inclusive fragment data
            # (sigma_incl/sigma_nonel): one struck nucleon goes to the wide
            # recoil budget, the rest is boost-preserving — clusters
            # included. None -> Delta Z / Delta N inference per channel.
            frag = None
            for m in pm_models:
                if (Z, A) in getattr(m, 'nuclei', []) and \
                        hasattr(m, 'fragment_yields'):
                    frag = m.fragment_yields(Z, A)
                    break

            # first pass: per-channel rates
            chan = []
            for Zrem, Arem in products:
                if getattr(self, 'rate_method', 'fft') == 'fft':
                    rates_total = channel_rates[row]
                    rates_pm = channel_rates_pm[row] if channel_rates_pm is not None else 0.0
                    row += 1
                else:
                    cross_section = 1e-27 * self.xsec_model.cross_section(eps * 1e3, Z, A, rem=(Zrem, Arem)) # to cm2
                    rates_total = interaction_rate_from_cross_section_boosts(boosts, self.target_photons, eps, cross_section)
                    rates_total /= c / parsec / 1e6 # ito Mpc
                    rates_pm = 0.0
                    if pm_models:
                        sig_pm = np.zeros_like(eps)
                        for m in pm_models:
                            if (Z, A) in m.nuclei:
                                sig_pm = sig_pm + np.asarray(m.cross_section(eps * 1e3, Z, A, rem=(Zrem, Arem)))
                        rates_pm = interaction_rate_from_cross_section_boosts(boosts, self.target_photons, eps, 1e-27 * sig_pm)
                        rates_pm /= c / parsec / 1e6
                rates_narrow = np.clip(rates_total - rates_pm, 0.0, None)
                chan.append((Zrem, Arem, rates_total, rates_pm, rates_narrow))

            for Zrem, Arem, rates_total, rates_pm, rates_narrow in chan:
                branchings.append(np.append([Zrem, Arem], rates_total))

                p_nar = (Z - Zrem) * rates_narrow
                n_nar = (A - Z - Arem + Zrem) * rates_narrow

                # wide (photomeson) side
                if frag is not None:
                    p_pm_narrow = frag['narrow'][4] * rates_pm
                    n_pm_narrow = frag['narrow'][5] * rates_pm
                    ejp = ejp + frag['wide_p'] * rates_pm
                    ejn = ejn + frag['wide_n'] * rates_pm
                    pm_clus = [frag['narrow'][ci] * rates_pm for ci in range(4)]
                else:
                    p_pm_narrow = np.zeros(len(boosts))
                    n_pm_narrow = np.zeros(len(boosts))
                    ejp = ejp + (Z - Zrem) * rates_pm
                    ejn = ejn + (A - Z - Arem + Zrem) * rates_pm
                    pm_clus = None

                if rates_ly is None:
                    lyp.append(np.append([Zrem, Arem], p_nar + p_pm_narrow))
                    lyn.append(np.append([Zrem, Arem], n_nar + n_pm_narrow))
                else:
                    # the table carries the complete light budget at MOTHER
                    # level (below): the per-channel rows here carry only
                    # the photomeson part
                    lyp.append(np.append([Zrem, Arem], p_pm_narrow))
                    lyn.append(np.append([Zrem, Arem], n_pm_narrow))
                for ci in range(4):
                    y = np.zeros(len(boosts))
                    if pm_clus is not None:
                        y = y + pm_clus[ci]
                    lyc[ci].append(np.append([Zrem, Arem], y))

            if rates_ly is not None and chan:
                # the six light-species yields verbatim from the model's
                # inclusive tables (multiplicity excess over the
                # one-per-event tensor allocation; Be-9: the second He4 +
                # one neutron, zero protons); labeled by the largest-rate
                # channel — the label indexes the light tensor, and a
                # marginal channel might not survive into the species list
                lci = int(np.argmax([np.sum(c[2]) for c in chan]))
                label = [chan[lci][0], chan[lci][1]]
                lyp.append(np.append(label, rates_ly[4]))
                lyn.append(np.append(label, rates_ly[5]))
                for ci in range(4):
                    lyc[ci].append(np.append(label, rates_ly[ci]))

            ejp_all.append(ejp)
            ejn_all.append(ejn)

            empty = np.zeros((0, 2 + len(boosts)))
            mlyp.append(np.vstack(lyp) if lyp else empty)
            mlyn.append(np.vstack(lyn) if lyn else empty)
            for ci in range(4):
                mly_clusters[ci].append(np.vstack(lyc[ci]) if lyc[ci] else empty)

            if branchings:
                pdis_rates_all.append(np.sum(np.atleast_2d(branchings), axis=0)[2:])
            else:
                pdis_rates_all.append(np.zeros(len(boosts)))
            branchings_all.append(branchings)

        # spontaneous nuclear decay of TRACKED species as jump rates
        # (nuclear_decay_On): lambda(gamma) = 1 / (gamma tau c) per decay
        # branching — the boost-diluted decay rate in Mpc^-1, added to the
        # corresponding row of the main tensor (K40 -> Ca40 for beta-,
        # ...). Emitted alpha/p/n/d/t are boost-preserving light yields;
        # beta decays shift Z only (leptons untracked, the documented Z
        # bookkeeping). Ultrafast particle emitters (tau < 1 ns: Be8, ...)
        # are excluded here — they resolve instantly at construction like
        # untracked products, and their diluted rates would dwarf every
        # interaction rate (numerically stiff, physically instantaneous).
        light_za6 = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
        light_ids6 = {402: 0, 302: 1, 301: 2, 201: 3, 101: 4, 100: 5}
        self.nuclear_decay_rates = np.zeros(
            (len(self.xsec_model.nuclei), len(boosts)))
        if self.nuclear_decay_On and self.decays:
            for k, (Z, A) in enumerate(self.xsec_model.nuclei):
                entry = self.decays.get(A * 100 + Z)
                if entry is None or entry['decay_time'] < 1e-9:
                    continue
                lam = 1.0 / (boosts * entry['decay_time'] * c_in_Mpc_sec)
                for channel in entry['channels']:
                    br, daughters = channel[0], channel[1:]
                    Zr, Ar, counts = Z, A, {}
                    for dau in daughters:
                        if dau == -1:                     # beta-minus
                            Zr += 1
                        elif dau == 1:                    # beta+ / EC
                            Zr -= 1
                        else:
                            Zr -= dau % 100
                            Ar -= dau // 100
                            if dau in light_ids6:
                                li = light_ids6[dau]
                                counts[li] = counts.get(li, 0) + 1
                    if Ar == 0 and counts:                # fully emitted
                        li = min(counts)
                        counts[li] -= 1
                        if counts[li] == 0:
                            del counts[li]
                        Zr, Ar = light_za6[li]
                    branchings_all[k].append(np.append([Zr, Ar], br * lam))
                    for li, cnt in counts.items():
                        row = np.append([Zr, Ar], cnt * br * lam)
                        if li == 4:
                            mlyp[k] = np.vstack([mlyp[k], row])
                        elif li == 5:
                            mlyn[k] = np.vstack([mlyn[k], row])
                        else:
                            mly_clusters[li][k] = np.vstack(
                                [mly_clusters[li][k], row])
                    self.nuclear_decay_rates[k] += br * lam

        branchings_all = [np.vstack(br) if br else np.zeros((0, 2 + len(boosts)))
                          for br in branchings_all]
        marginal_light_yields = list(mly_clusters)
        marginal_light_yields.append(mlyp)
        marginal_light_yields.append(mlyn)

        self.nuclei = self.xsec_model.nuclei.copy()
        self.all_rates = np.vstack(pdis_rates_all)
        self.all_branchings = branchings_all
        self.marginal_light_yields = marginal_light_yields
        # wide-spectrum nucleon budget per parent, rows in self.nuclei order
        # (counts only; proton_recoil_production carries the spectra)
        self.photomeson_ejecta = {'p': np.vstack(ejp_all), 'n': np.vstack(ejn_all)}

    def _collect_pdis_models(self, model):
        """Photodisintegration-group members of a (possibly nested) model."""
        if getattr(model, 'interaction_type', 'photodisintegration') == 'photodisintegration':
            return [model] if not getattr(model, 'models', None) else \
                [m for member in model.models
                 for m in self._collect_pdis_models(member)]
        members = getattr(model, 'models', None)
        if members:
            return [m for member in members
                    for m in self._collect_pdis_models(member)]
        return []

    def _collect_photomeson_models(self, model):
        """Photomeson-group members of a (possibly nested) cross-section model."""
        if getattr(model, 'interaction_type', 'photodisintegration') == 'photomeson':
            return [model]
        collected = []
        for member in getattr(model, 'models', None) or []:
            collected += self._collect_photomeson_models(member)
        return collected

    def _photomeson_rates_A1(self, pm_models):
        """Per-nucleon photomeson interaction rates on self.boosts [Mpc^-1],
        for the light-sector p/n absorption hook: the photomeson models' A=1
        total cross sections through the rate machinery, with the pion-kernel
        row sums as fallback."""
        eps_MeV = self.eps * 1e3
        rows = []
        for nuc in [(1, 1), (0, 1)]:
            sigma = np.zeros_like(eps_MeV)
            for model in pm_models:
                try:
                    sigma = sigma + np.asarray(model.cross_section(eps_MeV, *nuc))
                except Exception:
                    pass
            rows.append(sigma)

        if np.any(rows):
            rates = self._rates_of_sigma_rows(np.vstack(rows))
            return rates.mean(axis=0)          # p ~ n; the hook takes one array

        return self.pion_prod_tensor[0].sum(axis=1)

    def _build_photomeson_fold_scaling(self, pm):
        """Per-parent-boost rate-ratio factors that rescale the A = 1
        photomeson kernels in the folds from the superposition Z K_p + N K_n
        to the photomeson model's inclusive cross sections:

            F_c(A; Gamma) = R[sigma_c,A] / (Z R[sigma_c,p] + N R[sigma_c,n])

        per pion charge c (neucosma ids 2, 3, 4), plus 'pion' (charge-summed)
        and 'N' (nonelastic ratio — the recoil nucleon count follows the
        interaction rate, one nucleon per event, consistently with the
        cascade tensor and the photomeson_ejecta budget). Rates are computed
        on self.eps with the same machinery as the channel rates; where the
        superposition rate vanishes (sub-threshold) the factor is 1.

        Sets self.photomeson_fold_scaling {group: (n_mothers, n_boosts)} and
        self._pm_scaling_index {(Z, A): row}. Applied by _photomeson_fold
        via scaling_group=; not persisted by save()."""
        eps_MeV = self.eps * 1e3
        mothers = [(Z, A) for (Z, A) in pm.nuclei if A > 1]
        codes = (2, 3, 4)                              # pi+, pi-, pi0

        rows = []
        for Z, A in mothers:
            rows += [pm.inclusive_cross_section(eps_MeV, Z, A, c) for c in codes]
            rows.append(pm.cross_section(eps_MeV, Z, A))          # nonelastic
        for nuc in ((1, 1), (0, 1)):
            rows += [pm.inclusive_cross_section(eps_MeV, *nuc, c) for c in codes]
            rows.append(pm.cross_section(eps_MeV, *nuc))
        R = self._rates_of_sigma_rows(np.vstack(rows))            # (n_rows, n_b)

        n_m = len(mothers)
        R_m = R[:4 * n_m].reshape(n_m, 4, -1)
        R_p, R_n = R[4 * n_m:4 * n_m + 4], R[4 * n_m + 4:]        # (4, n_b)

        Z_m = np.array([z for (z, a) in mothers], dtype=float)
        N_m = np.array([a - z for (z, a) in mothers], dtype=float)
        den = Z_m[:, None, None] * R_p[None] + N_m[:, None, None] * R_n[None]

        # relative floor: sub-threshold rates are FFT-level denormals whose
        # ratios are numerical noise — keep the superposition (factor 1) there
        ok = den > 1e-12 * den.max(axis=-1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            F = np.where(ok, R_m / np.where(ok, den, 1.0), 1.0)
            pion_den = den[:, :3].sum(axis=1)
            ok_pi = pion_den > 1e-12 * pion_den.max(axis=-1, keepdims=True)
            F_pion = np.where(ok_pi,
                              R_m[:, :3].sum(axis=1)
                              / np.where(ok_pi, pion_den, 1.0), 1.0)

        self.photomeson_fold_scaling = {'pi+': F[:, 0], 'pi-': F[:, 1],
                                        'pi0': F[:, 2], 'N': F[:, 3],
                                        'pion': F_pion}
        self._pm_scaling_index = {nuc: i for i, nuc in enumerate(mothers)}

    def _channel_rates_fft(self, models=None):
        """Rates of every channel of every nucleus in one batched log-space
        convolution (see interaction_rates.compute_rates), in Mpc^-1.

        models : optional list of leaf models — restrict the cross sections to
            their contribution while keeping the rack's channel list (used to
            split channel rates by interaction type)."""
        eps_MeV = self.eps * 1e3

        if models is None:
            sigma_rows = np.vstack([
                self.xsec_model.cross_section(eps_MeV, Z, A, rem=(Zrem, Arem))   # mb
                for (Z, A), products in zip(self.xsec_model.nuclei, self.xsec_model.channels)
                for Zrem, Arem in products
            ])
        else:
            rows = []
            for (Z, A), products in zip(self.xsec_model.nuclei, self.xsec_model.channels):
                for rem in products:
                    sigma = np.zeros_like(eps_MeV)
                    for m in models:
                        if (Z, A) in m.nuclei:
                            sigma = sigma + np.asarray(m.cross_section(eps_MeV, Z, A, rem=tuple(rem)))
                    rows.append(sigma)
            sigma_rows = np.vstack(rows) if rows else np.zeros((0, len(eps_MeV)))
        return self._rates_of_sigma_rows(sigma_rows)

    def _rates_of_sigma_rows(self, sigma_rows):
        """Batched isotropic-field rates of cross-section rows sampled on
        self.eps (mb), on self.boosts, in Mpc^-1 (see exact_rates_for_sigma)."""
        return exact_rates_for_sigma(self.boosts, self.target_photons, self.eps, sigma_rows)

    def _collect_models_by_type(self, model):
        """Leaf cross-section models of a (possibly nested) model, grouped by
        their interaction_type ('photodisintegration', 'photomeson', ...)."""
        members = getattr(model, 'models', None)
        if members:
            groups = {}
            for member in members:
                for kind, leaves in self._collect_models_by_type(member).items():
                    groups.setdefault(kind, []).extend(leaves)
            return groups
        return {getattr(model, 'interaction_type', 'photodisintegration'): [model]}

    def conservation_imbalance(self):
        """Relative (A, Z) conservation imbalance of the assembled rate
        matrices: interaction tensor + boost-preserving light yields + the
        wide-spectrum photomeson-ejecta budget.

        Machine-level (~1e-14) for a consistent construction. Beta decays of
        resolved channel products shift the nuclear charge with the leptons
        untracked, so the Z imbalance reflects that physics on racks with
        proton-rich daughters while A must always balance.

        Returns
        -------
        tuple of float
            `(imbalance_A, imbalance_Z)`, each relative to `max(abs(tensor))`.
        """
        A_sp = np.array([s[1] for s in self.species], float)
        Z_sp = np.array([s[0] for s in self.species], float)
        A_L = np.array([4., 3., 3., 2., 1., 1.])
        Z_L = np.array([2., 2., 1., 1., 1., 0.])
        imbA = (np.einsum('j,ijb->ib', A_sp, self.tensor)
                + np.einsum('l,lijb->ib', A_L, self.light_prod_tensor))
        imbZ = (np.einsum('j,ijb->ib', Z_sp, self.tensor)
                + np.einsum('l,lijb->ib', Z_L, self.light_prod_tensor))
        ej = getattr(self, 'photomeson_ejecta', None)
        if ej is not None:
            for ni, nuc in enumerate(self.nuclei):
                si = self.species.index(tuple(nuc))
                imbA[si] += ej['p'][ni] + ej['n'][ni]
                imbZ[si] += ej['p'][ni]
        scale = np.abs(self.tensor).max()
        return np.abs(imbA).max() / scale, np.abs(imbZ).max() / scale

    def rates_by_interaction(self, nucleus=None):
        """Total interaction rates of each nucleus decomposed by interaction type.

        Re-evaluates the cross sections of the photodisintegration and the
        photomeson members of self.xsec_model separately through the same rate
        machinery used at construction, so the returned components sum to
        self.all_rates.  Useful to see which process drives the cascade of a
        given species on the core's photon field (free p/n absorption in the
        light sector is the separate self.photomeson_rates_pn attribute).

        Parameters
        ----------
        nucleus : tuple of int, optional
            `(Z, A)`. Restrict the output to that nucleus.

        Returns
        -------
        dict
            Maps interaction type to rates in Mpc^-1, arrays of shape
            `(n_nuclei, n_boosts)`, or `(n_boosts,)` when `nucleus` is
            given. Both `'photodisintegration'` and `'photomeson'` keys
            are always present (zeros when the model has no members of
            that type).
        """
        if getattr(self, 'xsec_model', None) is None:
            raise ValueError('rates_by_interaction needs a core constructed '
                             'from a cross-section model (xsec_model=)')

        from scipy.constants import c, parsec
        from .interaction_rates import interaction_rate_from_cross_section_boosts

        eps_MeV = self.eps * 1e3
        nuclei = [nucleus] if nucleus is not None else self.nuclei
        groups = self._collect_models_by_type(self.xsec_model)

        decomposed = {}
        for kind in sorted(set(groups) | {'photodisintegration', 'photomeson'}):
            sigma_rows = np.zeros((len(nuclei), len(eps_MeV)))
            for i, (Z, A) in enumerate(nuclei):
                for model in groups.get(kind, []):
                    if (Z, A) not in model.nuclei:
                        continue
                    for rem in model.channels[model.nuclei.index((Z, A))]:
                        sigma_rows[i] += model.cross_section(eps_MeV, Z, A, rem=tuple(rem))

            if getattr(self, 'rate_method', 'fft') == 'fft':
                rates = self._rates_of_sigma_rows(sigma_rows)
            else:
                rates = np.vstack([
                    interaction_rate_from_cross_section_boosts(
                        self.boosts, self.target_photons, self.eps, 1e-27 * row)
                    / (c / parsec / 1e6)
                    for row in sigma_rows])
            decomposed[kind] = rates[0] if nucleus is not None else rates

        # spontaneous-decay jump rates (nuclear_decay_On cores): reported
        # separately — 'photodisintegration' + 'photomeson' sum to
        # self.all_rates; the decay rows live in the tensor on top of that
        dec = getattr(self, 'nuclear_decay_rates', None)
        if dec is not None and np.any(dec):
            if nucleus is None:
                decomposed['decay'] = dec
            elif tuple(nucleus) in self.nuclei:
                # light-sector species ((1,1)/(0,1)) have no tensor row;
                # free-neutron decay lives in neutron_decay_neutrinos
                decomposed['decay'] = dec[self.nuclei.index(tuple(nucleus))]

        return decomposed

    def pion_production(self, L, alpha=None, mass_range=None, boost_range=None, true_range=None, P=None, weights=None,
                       method=None):
        """Cumulative pion spectrum produced by the heavy cascade at positions L.

        Solves the heavy cascade ODE for each parent boost and folds the result
        with self.pion_prod_tensor, integrating along L to give the total pion
        count in each boost bin of self.boosts between 0 and each element of L.

        Pions are treated as a single species (π+, π-, π0 lumped).  For a nucleus
        (Z, A) the contribution to pion boost bin j from parent boost bin i is
        Z * K[0, i, j] + (A-Z) * K[1, i, j] where K = self.pion_prod_tensor.

        Requires the photomeson kernels (construct with photomeson='kernels').

        Parameters
        ----------
        L : ndarray
            1-D array of distances [Mpc].
        alpha : ndarray
            Injection spectrum (must sum to 1), shape `(n_species_full,)`.
        mass_range : list of int
            Species indices, as returned by `get_distribution_parameters`.
        boost_range : ndarray, optional
            Parent boost values. Default `self.boosts`.
        true_range : list of int
            Non-absorbed subset of `mass_range`.
        P : ndarray, optional
            Precomputed heavy evolution, as returned by
            `species_evolution_boost_range` with the same arguments.
            Avoids re-solving the cascade ODE.
        weights : ndarray, optional
            Per-parent-boost injection weights, shape `(len(boost_range),)`,
            e.g. the injected number per slice, `dQ/dlnGamma * dlnGamma`.
            The parent slices are then summed as a weighted ladder total
            instead of unit injections.
        method : {None, 'exact', 'substep', 'auto'}, optional
            Passed straight through to `species_evolution_boost_range` when
            `P` is None (ignored otherwise). None (default) evaluates the
            heavy cascade exactly, matching this method's historical
            behavior; 'auto'/'substep' let a large species count over a
            wide `L` range trade a little precision for a lot of speed,
            the same tradeoff documented there.

        Returns
        -------
        ndarray
            `N_pion`, shape `(n_boost_pion, n_L)` with `n_boost_pion =
            len(self.boosts)`. `N_pion[j, l]` is the cumulative number of
            pions produced in boost bin `j` up to distance `L[l]`.

        Examples
        --------
        >>> import numpy as np
        >>> from crisp.core import InteractionCore
        >>> from crisp.photonuclear_cross_sections import PSB_model
        >>> core = InteractionCore(xsec_model=PSB_model(), photomeson='kernels')
        >>> alpha, mr, tr, _ = core.get_distribution_parameters(
        ...     mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        ...     absorption_type=('only mass', [1]))
        >>> L = np.array([10.0, 50.0, 200.0])  # Mpc
        >>> N_pion = core.pion_production(
        ...     L, alpha=alpha, mass_range=mr, boost_range=core.boosts[80:120],
        ...     true_range=tr)
        >>> N_pion.sum(axis=0)  # total pions produced by each distance
        array([0.        , 0.08670962, 0.09320204])
        """
        if not hasattr(self, 'pion_prod_tensor'):
            raise AttributeError('pion_prod_tensor not available; rebuild or load from file.')
        return self._photomeson_fold(self.pion_prod_tensor, L, alpha=alpha,
                                     mass_range=mass_range, boost_range=boost_range,
                                     true_range=true_range, P=P, weights=weights,
                                     scaling_group='pion', method=method)

    def photomeson_production(self, species, L, alpha=None, mass_range=None,
                              boost_range=None, true_range=None, P=None, weights=None,
                              cumulative=True, method=None):
        """Cumulative production of one photomeson secondary species along L:
        the charge/species-resolved kernel folded with the heavy cascade.

        Parameters
        ----------
        species : {'pi+', 'pi-', 'pi0', 'p', 'n'}
            Key into `self.photomeson_kernels`.
        cumulative : bool, optional
            True (default) integrates the production along `L`; False
            returns the local production rate per unit path instead of
            the `L`-integral.
        L, alpha, mass_range, boost_range, true_range, P, weights, method :
            As in `pion_production` (which this method mirrors, just for
            a chosen photomeson secondary instead of the lumped pion
            spectrum).

        Returns
        -------
        ndarray
            Same shape convention as `pion_production`'s return value.
        """
        group = {'pi+': 'pi+', 'pi-': 'pi-', 'pi0': 'pi0', 'p': 'N', 'n': 'N'}[species]
        return self._photomeson_fold(self.photomeson_kernels[species], L, alpha=alpha,
                                     mass_range=mass_range, boost_range=boost_range,
                                     true_range=true_range, P=P, weights=weights,
                                     scaling_group=group, cumulative=cumulative, method=method)

    def _photomeson_fold(self, kernel, L, alpha=None, mass_range=None,
                         boost_range=None, true_range=None, P=None, weights=None,
                         scaling_group=None, cumulative=True, method=None):
        """Fold a photomeson kernel (2, n_b, n_b) with the heavy cascade and
        integrate along L — the shared machinery of pion_production,
        proton_recoil_production and the per-charge neutrino folds.

        scaling_group : key into self.photomeson_fold_scaling ('pi+', 'pi-',
            'pi0', 'pion', 'N') applying the per-species inclusive rescaling
            of photomeson_scaling='inclusive'; silently a no-op on cores
            built without it (factor 1, bit-identical superposition).
        cumulative : True (default) integrates the production along L;
            False returns the local production RATE per unit path
            (n_b_sec, n_L) [same units per Mpc] — e.g. the wide-spectrum
            nucleon source term of reprocessed_nucleons.
        method : passed straight through to species_evolution_boost_range
            for the heavy-cascade solve when P is None; see that method's
            docstring (None/default -> exact, matching this method's
            historical behavior bit-for-bit)."""
        from scipy.integrate import cumulative_trapezoid

        if boost_range is None:
            boost_range = self.boosts

        n_b = len(boost_range)
        indices = [mass_range.index(ival) for ival in true_range]
        n_sp    = len(indices)
        L_arr   = np.asarray(L)

        if P is not None:
            heavy = np.asarray(P)[:, :, :n_sp]             # (n_b, n_L, n_sp)
        else:
            # heavy cascade ODE: the same (n_sp+1)-augmented exact system
            # species_evolution_boost_range solves, so delegate to it
            # directly (one place this math lives) instead of re-deriving
            # it here -- also picks up method='auto'/'substep' for free.
            P_full = self.species_evolution_boost_range(
                L_arr, alpha=alpha, mass_range=mass_range, boost_range=boost_range,
                true_range=true_range, method=method)
            heavy = P_full[:, :, :n_sp]                        # (n_b, n_L, n_sp)

        # --- fold with the kernel ---
        # interpolate kernel to boost_range on the parent-boost axis
        K = interp1d(self.boosts, kernel, axis=1, kind='linear',
                     bounds_error=False, fill_value=0.0)(boost_range)
        # K: (2, n_b, n_b_sec) where n_b_sec = len(self.boosts)

        # species composition
        Z_sp = np.array([self.species[true_range[k]][0] for k in range(n_sp)], dtype=float)
        A_sp = np.array([self.species[true_range[k]][1] for k in range(n_sp)], dtype=float)

        # K_sp[sp, b_parent, b_sec] = Z * K_p + (A-Z) * K_n
        K_sp = (Z_sp[:, None, None] * K[0][None]
                + (A_sp - Z_sp)[:, None, None] * K[1][None])  # (n_sp, n_b, n_b_sec)

        table = getattr(self, 'photomeson_fold_scaling', None)
        if scaling_group is not None and table is not None and scaling_group in table:
            rows = np.ones((n_sp, len(self.boosts)))
            for k in range(n_sp):
                ri = self._pm_scaling_index.get(tuple(self.species[true_range[k]]))
                if ri is not None:
                    rows[k] = table[scaling_group][ri]
            F_b = interp1d(self.boosts, rows, axis=1, kind='linear',
                           bounds_error=False, fill_value=1.0)(boost_range)
            K_sp = K_sp * F_b[:, :, None]

        if weights is not None:
            heavy = heavy * np.asarray(weights)[:, None, None]

        # production rate at each (L, b_sec): sum over parent boost and species
        rate = np.einsum('bls, sbj -> lj', heavy, K_sp)        # (n_L, n_b_sec)

        if not cumulative:
            return rate.T                                       # (n_b_sec, n_L)

        # integrate along L
        N_sec = cumulative_trapezoid(rate, L_arr, axis=0, initial=0.0)

        return N_sec.T                                          # (n_b_sec, n_L)

    def proton_recoil_production(self, L, alpha=None, mass_range=None,
                                  boost_range=None, true_range=None, P=None, weights=None):
        """Cumulative secondary-nucleon (p + n) spectrum from photomeson at
        positions L.

        Identical in structure to pion_production but uses
        self.proton_recoil_tensor (the p + n sum of the species kernels):
        secondaries are placed at γ_N = (1−κ) × Γ_parent, always below the
        parent boost, one nucleon per interaction. The p/n split is available
        through self.photomeson_kernels['p'] / ['n'] with _photomeson_fold.

        Requires the photomeson kernels (construct with photomeson='kernels').

        Parameters
        ----------
        L, alpha, mass_range, boost_range, true_range, P, weights :
            As in `pion_production`.

        Returns
        -------
        ndarray
            `N_nucleon`, shape `(n_boost, n_L)`: cumulative secondary
            nucleon count per boost bin of `self.boosts` up to `L[l]`.
        """
        if not hasattr(self, 'proton_recoil_tensor'):
            raise AttributeError('proton_recoil_tensor not available; rebuild or load from file.')
        return self._photomeson_fold(self.proton_recoil_tensor, L, alpha=alpha,
                                     mass_range=mass_range, boost_range=boost_range,
                                     true_range=true_range, P=P, weights=weights,
                                     scaling_group='N')

    def nucleon_transport_matrix(self):
        """The free-nucleon transport generator M [Mpc^-1] on the doubled
        boost grid, state = [p bins | n bins], row convention M[from, to]
        (evolve row vectors: n @ expm(M L)).

        Off-diagonal blocks are the recoil-kernel placements (struck nucleon
        at (1-kappa)Gamma per interaction type, charge-resolved); the
        diagonal removes the kernel row sums — which equal the exact
        interaction rate (one nucleon per event). Neutrons additionally
        beta-decay in flight, n -> p at the same boost (Q << E), with the
        mean lifetime from the core's nuclear decay table (nubase;
        fallback 880 s as in the light sector when no table was given).
        Every row still sums to zero, so nucleon number is conserved
        identically. Cached.

        Returns
        -------
        ndarray
            The generator `M`, shape `(2 * n_boost, 2 * n_boost)`.
        """
        if not hasattr(self, 'photomeson_kernels'):
            raise AttributeError("photomeson kernels not available; "
                                 "construct with photomeson='kernels'")
        if not hasattr(self, '_nucleon_transport_M'):
            Kp, Kn = self.photomeson_kernels['p'], self.photomeson_kernels['n']
            n_b = len(self.boosts)
            M = np.zeros((2 * n_b, 2 * n_b))       # [p bins | n bins]
            M[:n_b, :n_b] = Kp[0]                  # p parent -> secondary p
            M[:n_b, n_b:] = Kn[0]                  # p parent -> secondary n
            M[n_b:, :n_b] = Kp[1]                  # n parent -> secondary p
            M[n_b:, n_b:] = Kn[1]                  # n parent -> secondary n
            M -= np.diag(M.sum(axis=1))            # rows sum to zero exactly

            tau_n = 880.0                          # light-sector fallback [s]
            if isinstance(getattr(self, 'decays', None), dict):
                entry = self.decays.get(get_nucid((0, 1)))
                if entry is not None:
                    tau_n = float(entry['decay_time'])
            lam = 1.0 / (self.boosts * tau_n * c_in_Mpc_sec)   # 1/Mpc
            idx = np.arange(n_b)
            M[n_b + idx, n_b + idx] -= lam         # n decays away ...
            M[n_b + idx, idx] += lam               # ... into p at the same bin
            self._nucleon_transport_M = M
            self._nucleon_transport_lam = lam
            self._nucleon_transport_rate = np.maximum(
                (Kp[0] + Kn[0]).sum(axis=1), (Kp[1] + Kn[1]).sum(axis=1))
        return self._nucleon_transport_M

    def reprocessed_nucleons(self, L, injection=None, source=None,
                             energy_loss=None):
        """Multi-generation photomeson transport of free nucleons along the
        path. After every p/n + gamma -> pi + N' event, the struck nucleon
        re-enters at its recoil placement and keeps interacting until it
        degrades below the photomeson threshold, where the kernel rows
        vanish and the occupation freezes out. This is the nucleon
        reprocessing of De Lia and Tamborra (2024) and Biehl et al. (2018),
        replacing the one-generation exp(-tau) depletion.

        Solves dn/dx = n M + q with the transport generator of
        `nucleon_transport_matrix`, chained per L-interval. The source is
        treated exactly for its interval mean (augmented-matrix Duhamel),
        so the total nucleon number matches the trapezoid integral of the
        source at machine precision. (A piecewise-linear source needs one
        extra augmented row, not implemented.)

        Included beyond the interactions: neutron beta decay in flight
        (n -> p at the same boost, nubase mean lifetime; decay length
        gamma times 8.6e-12 Mpc, irrelevant inside a shell crossing but
        decisive over extragalactic distances), and, when `energy_loss` is
        given, continuous cooling (synchrotron or adiabatic) as a
        conservative upwind drift in log-boost. The
        `photomeson_scaling='inclusive'` factors are 1 for A = 1 parents,
        so scaled and unscaled cores transport identically.

        Parameters
        ----------
        L : ndarray
            Path positions [Mpc]. The state at `L[0]` is the injection.
        injection : ndarray, optional
            Standing spectrum at `L[0]`, shape `(n_b, 2)` for `[p, n]` on
            `self.boosts` (absolute or per-unit, in the caller's units).
        source : ndarray, optional
            Production-rate density along the path, shape `(n_b, 2)`
            constant or `(n_b, n_L, 2)`, per Mpc (e.g. from
            `cascade_nucleon_source`), integrated from `L[0]`.
        energy_loss : ndarray, optional
            Continuous fractional energy-loss rate `b(gamma) =
            -dln(gamma)/dx` [1/Mpc] on `self.boosts` (uniform log grid):
            shape `(n_b,)` applied to both species, or `(n_b, 2)` as
            `[p, n]`, e.g. (synchrotron + adiabatic)/c for protons,
            adiabatic/c only for neutrons, from a source model's
            `loss_rates`. Implemented as one-bin-down hops at rate
            `b / dln(gamma)`: nucleon number is conserved identically and
            the mean ln(gamma) drifts at exactly `-b` (first-order upwind,
            about one-bin numerical spreading).

        Returns
        -------
        ndarray
            `n`, shape `(n_b, n_L, 2)`: standing `[p, n]` spectra at each
            `L`.
        """
        M = self.nucleon_transport_matrix()
        lam = self._nucleon_transport_lam
        rate = self._nucleon_transport_rate
        L_arr = np.asarray(L, dtype=float)
        n_b, n_L, n2 = len(self.boosts), len(L_arr), 2 * len(self.boosts)
        idx = np.arange(n_b)

        b_loss = None
        dln_b = np.log(self.boosts[1] / self.boosts[0])
        if energy_loss is not None:
            b_loss = np.asarray(energy_loss, dtype=float)
            if b_loss.ndim == 1:
                b_loss = np.stack([b_loss, b_loss], axis=-1)     # (n_b, 2)

        # the decay/cooling-interaction interplay must be resolved wherever
        # both act (the charge label and energy at interaction time set the
        # pion output): sub-step so the fast rates satisfy r * dl <= 60 at
        # every relevant bin — for the drift only up to the highest OCCUPIED
        # bin (drift, decay and recoil all move downward, and the untouched
        # far-upper bins can carry enormous synchrotron rates)
        active = rate > 1e-6 * rate.max() if rate.max() > 0 else rate > 0
        lam_act = lam[active].max() if active.any() else 0.0
        if b_loss is not None:
            occ = np.zeros(n_b, dtype=bool)
            if injection is not None:
                occ |= np.asarray(injection).sum(axis=-1) > 0
            if source is not None:
                occ |= np.asarray(source).reshape(n_b, -1).sum(axis=1) > 0
            top = int(np.nonzero(occ)[0].max()) if occ.any() else n_b - 1
            lam_act = max(lam_act, b_loss[:top + 1].max() / dln_b)

        def interval_matrix(dl):
            # stiffness cap for the remaining (unoccupied / interaction-free)
            # bins, where only the downward transfer matters and ordering is
            # irrelevant: rates * dl reach ~1e13 at the grid extremes and
            # would inflate the expm scaling until row sums drift; capping
            # each loss+gain pair at 60/dl leaves the end state exact
            # (e^-60 ~ 1e-26) and keeps conservation at machine precision
            excess = np.clip(lam - 60.0 / dl, 0.0, None)
            Mk = M.copy()
            Mk[n_b + idx, n_b + idx] += excess
            Mk[n_b + idx, idx] -= excess
            if b_loss is not None:
                for s in range(2):
                    r = np.minimum(b_loss[:, s] / dln_b, 60.0 / dl)
                    r[0] = 0.0                 # freeze at the grid floor
                    o = s * n_b
                    Mk[o + idx, o + idx] -= r
                    Mk[o + idx[1:], o + idx[:-1]] += r[1:]
            return Mk

        y = np.zeros(n2) if injection is None \
            else np.asarray(injection, dtype=float).T.ravel().copy()
        out = np.empty((n_L, n2))
        out[0] = y

        if source is not None:
            q = np.asarray(source, dtype=float)
            if q.ndim == 2:                          # constant (n_b, 2)
                q = np.repeat(q[:, None, :], n_L, axis=1)
            q = np.moveaxis(q, -1, 0).reshape(n2, n_L)
            A = np.zeros((n2 + 1, n2 + 1))
            ya = np.append(y, 1.0)

        for k in range(n_L - 1):
            dL = L_arr[k + 1] - L_arr[k]
            m = max(1, int(np.ceil(dL * lam_act / 60.0))) if dL > 0 else 1
            dl = dL / m
            if source is None:
                E = expm(interval_matrix(dl) * dl)
                for _ in range(m):
                    y = y @ E
                out[k + 1] = y
            else:
                A[:n2, :n2] = interval_matrix(dl)
                A[n2, :n2] = 0.5 * (q[:, k] + q[:, k + 1])   # interval mean
                E = expm(A * dl)
                for _ in range(m):
                    ya = ya @ E
                out[k + 1] = ya[:n2]

        return np.transpose(out.reshape(n_L, 2, n_b), (2, 0, 1))

    def neutron_decay_neutrinos(self, neutrons):
        """Electron antineutrinos from neutron beta decay, n -> p e- nubar_e.
        Each neutron yields exactly one nubar_e with the boosted
        three-body decay spectrum (allowed shape, Q = 0.782 MeV, mean
        E_nu about 5.1e-4 E_n), deposited on the same neutrino energy
        grid as `neutrino_production` (E_nu = m_pi times `self.boosts`).

        The caller chooses the neutron census. For a source with
        advective escape, pass the standing neutron spectrum at the end
        of the crossing (they all decay in transit long before Earth,
        since the decay length gamma times 8.6e-12 Mpc dwarfs any shell
        but not the way to the observer), in the same units as the
        `neutrino_production` folds so the output adds directly to the
        'detail' nubar_e. For in-flight decays along a tracked path, pass
        the lambda-weighted path integral of the occupations instead.

        Antineutrinos from neutrons whose decay products fall below the
        grid floor (E_nu < m_pi, i.e. gamma_n less than about 300 at the
        mean fraction) are dropped with the deposit convention.

        Parameters
        ----------
        neutrons : ndarray
            Array whose first axis is the boost grid, e.g. `(n_b,)`
            escaping counts, or any `(n_b, ...)` stack.

        Returns
        -------
        ndarray
            `nubar_e`, the same shape with the boost axis mapped onto the
            neutrino energy grid (one antineutrino per neutron).
        """
        if not hasattr(self, '_neutron_decay_D'):
            Q, m_e, m_n = 0.782e-3, 0.511e-3, 0.93957     # GeV
            # rest-frame antineutrino spectrum (allowed beta shape)
            e0 = np.linspace(Q * 1e-4, Q, 400)
            E_e = Q + m_e - e0
            f0 = e0**2 * E_e * np.sqrt(np.clip(E_e**2 - m_e**2, 0.0, None))
            f0 /= np.trapezoid(f0, e0)
            # boosted isotropically (ultrarelativistic): x = E_nu / E_n in
            # (0, 2 e0max / m_n); dN/dx = int_{e0 > x m_n / 2} f0 m_n/(2 e0)
            x = np.linspace(2 * Q / m_n * 1e-4, 2 * Q / m_n, 400)
            dNdx = np.array([np.trapezoid(
                np.where(e0 > xx * m_n / 2.0, f0 * m_n / (2.0 * e0), 0.0), e0)
                for xx in x])
            w = dNdx * np.gradient(x)
            w /= w.sum()                                   # one nubar per decay

            mpi = 0.13957039
            E_grid = mpi * self.boosts                     # the chain's grid
            n_b = len(self.boosts)
            D = np.zeros((n_b, n_b))                       # (i_neutron, j_nu)
            for i, g in enumerate(self.boosts):
                D[i] = deposit_log_cic(E_grid, x * g * m_n, w)
            self._neutron_decay_D = D

        return np.tensordot(self._neutron_decay_D, np.asarray(neutrons),
                            axes=(0, 0))

    def cascade_nucleon_source(self, L, alpha=None, mass_range=None,
                               boost_range=None, true_range=None, P=None,
                               weights=None, LC=None):
        """Free-nucleon production rate of a heavy cascade on the
        `self.boosts` grid, the source term for `reprocessed_nucleons`.
        Three contributions:

        1. narrow: boost-preserving photodisintegration nucleons
           (`light_secondaries_production` rows p, n), deposited from the
           parent rungs onto the grid bins (CIC in log boost);
        2. conversion: standing light nuclei (He4, He3, H3, H2 of
           `light_cascade_production`) decaying or disintegrating into
           p, n (the light-matrix coupling), deposited like the narrow
           term;
        3. wide: photomeson-ejected nucleons with their recoil-kernel
           placement (charge-resolved folds, `scaling_group='N'`,
           consistent with the `photomeson_ejecta` budget and the
           inclusive scaling).

        Parameters
        ----------
        L, alpha, mass_range, boost_range, true_range, P :
            As in `pion_production`.
        weights : ndarray, optional
            Per-rung injection (`dQ/dlnGamma * dlnGamma`).
        LC : ndarray, optional
            Precomputed `light_cascade_production` result (same
            arguments), to avoid re-solving it.

        Returns
        -------
        ndarray
            `q`, shape `(n_b, n_L, 2)`: `[p, n]` production rate per Mpc.
        """
        if boost_range is None:
            boost_range = self.boosts
        L_arr = np.asarray(L, dtype=float)
        n_b, n_L = len(self.boosts), len(L_arr)
        w = np.ones(len(boost_range)) if weights is None \
            else np.asarray(weights, dtype=float)

        LS = self.light_secondaries_production(
            L_arr, alpha=alpha, mass_range=mass_range,
            boost_range=boost_range, true_range=true_range, P=P)

        if LC is None:
            LC = self.light_cascade_production(
                L_arr, alpha=alpha, mass_range=mass_range,
                boost_range=boost_range, true_range=true_range)
        M_light = self._build_light_matrix(boost_range)          # (6, 6, n_r)
        conv = np.einsum('mrl,msr->srl', np.asarray(LC)[:4],
                         M_light[:4, 4:6, :])                    # (2, n_r, n_L)

        # rung values onto the grid: each rung represents a continuum slice
        # of width dln(rung), so interpolate per-lnGamma densities between
        # rungs rather than depositing point deltas (which would comb a grid
        # finer than the ladder) — for the boost-preserving terms directly,
        # and for the wide term by folding the kernels with the
        # continuum-interpolated heavy occupations
        dln_r = np.gradient(np.log(np.asarray(boost_range, dtype=float)))
        dlnb = np.gradient(np.log(self.boosts))
        lnb, lnr = np.log(self.boosts), np.log(np.asarray(boost_range, float))

        heavy = np.asarray(P)[:, :, :len(true_range)] * (w / dln_r)[:, None, None]
        P_grid = interp1d(lnr, heavy, axis=0, kind='linear', bounds_error=False,
                          fill_value=0.0)(lnb) * dlnb[:, None, None]

        q = np.zeros((n_b, n_L, 2))
        for s in range(2):
            rho = (np.asarray(LS[4 + s]) + conv[s]) * (w / dln_r)[:, None]
            for l in range(n_L):
                q[:, l, s] = np.interp(lnb, lnr, rho[:, l],
                                       left=0.0, right=0.0) * dlnb
            q[:, :, s] += self._photomeson_fold(
                self.photomeson_kernels['p' if s == 0 else 'n'], L_arr,
                alpha=alpha, mass_range=mass_range, boost_range=self.boosts,
                true_range=true_range, P=P_grid, weights=None,
                scaling_group='N', cumulative=False)
        return q

    def photomeson_ejecta_production(self, L, alpha=None, mass_range=None,
                                     boost_range=None, true_range=None, P=None,
                                     weights=None, cumulative=False):
        """Production rate of the wide-spectrum photomeson nucleons along
        the cascade, per parent boost: the `self.photomeson_ejecta` budget
        folded with the cascade occupation like the light yields.

        Counts only (used e.g. by per-boost nucleon ledgers). The ejected
        nucleons' spectral placement is `proton_recoil_production`.

        Parameters
        ----------
        L, alpha, mass_range, boost_range, true_range, P :
            As in `light_secondaries_production`.
        weights : ndarray, optional
            As in `pion_production`.
        cumulative : bool, optional
            If True, integrate the rate along `L`. Default False.

        Returns
        -------
        ndarray
            `production`, shape `(2, n_boosts, n_L)`: `[p, n]` ejecta
            production rates per injected particle, per Mpc.
        """
        if boost_range is None:
            boost_range = self.boosts

        n_sp = len(true_range)
        ej = self.photomeson_ejecta
        rows = np.zeros((2, n_sp, len(self.boosts)))
        for k, t in enumerate(true_range):
            za = tuple(self.species[t])
            if za in self.nuclei:
                ni = self.nuclei.index(za)
                rows[0, k] = ej['p'][ni]
                rows[1, k] = ej['n'][ni]
        # 'previous', like interpolator/interpyields: rates must come from the
        # same boost bin as the destruction rates for consistent bookkeeping
        rows_b = interp1d(self.boosts, rows, kind='previous',
                          bounds_error=False, fill_value=0.0)(boost_range)

        if P is None:
            P = self.species_evolution_boost_range(L, alpha, mass_range, boost_range, true_range)
        heavy = np.asarray(P)[:, :, :n_sp]
        if weights is not None:
            heavy = heavy * np.asarray(weights)[:, None, None]

        production = np.einsum('bmi, kib -> kbm', heavy, rows_b)
        if cumulative:
            from scipy.integrate import cumulative_trapezoid
            production = cumulative_trapezoid(production, np.asarray(L), axis=-1, initial=0.0)
        return production

    def _fraction_matrix(self, fracs, weights):
        """Distribution matrix D[i, j]: the fraction of secondaries from
        energy bin i landing in bin j when the daughter carries fraction
        fracs[k] of the parent energy with probability weights[k] — a CIC
        deposit in the log-energy grid, shared by all decay steps."""
        n = len(self.boosts)
        logb = np.log(self.boosts)
        dl = logb[1] - logb[0]
        D = np.zeros((n, n))
        for frac, wt in zip(fracs, weights):
            if frac <= 0 or wt <= 0:
                continue
            x = np.arange(n) + np.log(frac) / dl
            j0 = np.floor(x).astype(int)
            f = x - j0
            for jj, ww in ((j0, (1 - f) * wt), (j0 + 1, f * wt)):
                valid = (jj >= 0) & (jj < n)
                D[np.arange(n)[valid], jj[valid]] += ww[valid]
        return D

    def _pion_decay_chain(self):
        """Cached decay matrices of the pi -> mu -> e chain with the muon
        helicity dependence of Lipari et al. (2007) as compiled in
        Huemmer et al. (2010), Eqs. (68)-(71).

        All matrices are (n_b, n_b) distributions on the E = m_pi * boosts
        energy grid, exactly count-normalized. Keys:
        'box' (direct nu from pi), 'muL'/'muR' (pi -> mu by helicity), and
        the muon-decay combinations 'P70p'/'P70m' (muon-flavor nu for
        h = +1 / -1) and 'P71p'/'P71m' (electron flavor).
        """
        if hasattr(self, '_nu_chain'):
            return self._nu_chain
        from scipy.constants import physical_constants

        mpi = 0.13957039   # charged pion mass, GeV (PDG)
        mmu = physical_constants['muon mass energy equivalent in MeV'][0] * 1e-3
        r = (mmu / mpi) ** 2

        # direct nu_mu: exact two-body box on [0, (1 - r) E_pi]
        E = mpi * self.boosts
        edges = np.concatenate([[E[0] ** 2 / E[1]], np.sqrt(E[:-1] * E[1:]),
                                [E[-1] ** 2 / E[-2]]])
        box_top = (1.0 - r) * E
        D_box = (np.minimum(edges[None, 1:], box_top[:, None])
                 - np.minimum(edges[None, :-1], box_top[:, None]))
        D_box = np.clip(D_box, 0.0, None) / box_top[:, None]

        # pi -> mu with helicity (Eqs. 68-69): x = E_mu / E_pi in [r, 1]
        x = np.linspace(r, 1.0, 400)
        dx = np.gradient(x)
        F_A = r * (1.0 - x) / ((1.0 - r) ** 2 * np.maximum(x, r))   # pi+ -> mu+_R
        F_B = (x - r) / ((1.0 - r) ** 2 * np.maximum(x, r))         # pi+ -> mu+_L
        wA, wB = F_A * dx, F_B * dx
        tot = wA.sum() + wB.sum()                                   # exactly one muon
        D_muR = self._fraction_matrix(x, wA / tot)
        D_muL = self._fraction_matrix(x, wB / tot)

        # mu -> nu with helicity (Eqs. 70-71): y = E_nu / E_mu in [0, 1]
        y = np.linspace(0.0, 1.0, 400)
        dy = np.gradient(y)
        P70 = 5 / 3 - 3 * y ** 2 + 4 * y ** 3 / 3
        Q70 = -1 / 3 + 3 * y ** 2 - 8 * y ** 3 / 3
        P71 = 2 - 6 * y ** 2 + 4 * y ** 3
        Q71 = 2 - 12 * y + 18 * y ** 2 - 8 * y ** 3

        def decay_mat(F):
            w = np.clip(F, 0.0, None) * dy
            return self._fraction_matrix(y, w / w.sum())            # one nu per muon

        self._nu_chain = {
            'r': r, 'mpi': mpi, 'mmu': mmu, 'box': D_box,
            'muL': D_muL, 'muR': D_muR,
            'P70p': decay_mat(P70 + Q70), 'P70m': decay_mat(P70 - Q70),
            'P71p': decay_mat(P71 + Q71), 'P71m': decay_mat(P71 - Q71),
        }
        return self._nu_chain

    def _kaon_decay_chain(self):
        """Cached decay matrices of the leading kaon mode K -> mu nu_mu
        (BR 0.636) — the channel Huemmer et al. (2010) track for kaons, the
        one producing the highest-energy neutrinos. Same two-body formalism
        as _pion_decay_chain with r -> (m_mu / m_K)^2 ~ 0.046, but the
        parent lives on E_K = m_K * boosts while all outputs land on the
        shared reporting grid E = m_pi * boosts: the mass ratio m_K / m_pi
        enters the energy placement explicitly, and the kaon muons are
        deposited on the shared muon grid so the pion chain's P70/P71
        matrices apply to them unchanged. Weight placed above the grid top
        is dropped (the usual grid-boundary convention).

        Keys: 'box' (direct nu_mu per decay), 'muL'/'muR' (K -> mu by
        helicity, one muon per decay) and the constants 'mK', 'rK',
        'tauK' (PDG rest-frame lifetime, s), 'BR_munu'.
        """
        if hasattr(self, '_nu_chain_K'):
            return self._nu_chain_K
        from scipy.constants import physical_constants

        mK = 0.493677      # charged kaon mass, GeV (PDG)
        mpi = 0.13957039
        mmu = physical_constants['muon mass energy equivalent in MeV'][0] * 1e-3
        rK = (mmu / mK) ** 2

        # direct nu_mu: exact two-body box on [0, (1 - rK) E_K], integrated
        # over the bin edges of the E = m_pi * boosts reporting grid
        E = mpi * self.boosts
        edges = np.concatenate([[E[0] ** 2 / E[1]], np.sqrt(E[:-1] * E[1:]),
                                [E[-1] ** 2 / E[-2]]])
        box_top = (1.0 - rK) * mK * self.boosts
        D_box = (np.minimum(edges[None, 1:], box_top[:, None])
                 - np.minimum(edges[None, :-1], box_top[:, None]))
        D_box = np.clip(D_box, 0.0, None) / box_top[:, None]

        # K -> mu with helicity (Eqs. 68-69 with r -> rK): x = E_mu / E_K
        x = np.linspace(rK, 1.0, 400)
        dx = np.gradient(x)
        F_A = rK * (1.0 - x) / ((1.0 - rK) ** 2 * np.maximum(x, rK))
        F_B = (x - rK) / ((1.0 - rK) ** 2 * np.maximum(x, rK))
        wA, wB = F_A * dx, F_B * dx
        tot = wA.sum() + wB.sum()                       # exactly one muon
        # fraction x * mK/mpi places E_mu = x E_K on the m_pi * boosts grid
        D_muR = self._fraction_matrix(x * (mK / mpi), wA / tot)
        D_muL = self._fraction_matrix(x * (mK / mpi), wB / tot)

        self._nu_chain_K = {
            'mK': mK, 'rK': rK, 'tauK': 1.2380e-8, 'BR_munu': 0.6356,
            'box': D_box, 'muL': D_muL, 'muR': D_muR,
        }
        return self._nu_chain_K

    def decay_before_cooling(self, E_GeV, m_GeV, tau0_s, B_gauss):
        """Closed-form probability that a secondary decays before synchrotron
        cooling degrades it: t_syn / (t_syn + t_dec), with t_dec = tau0 E/m
        and the single-particle synchrotron time in the field B. O(n) cost —
        no transport is solved; the cooled fraction is dropped (its energy
        goes to synchrotron photons), which is the no-migration
        approximation of the standard pi/mu cooling breaks."""
        from scipy.constants import c, e, physical_constants
        gamma = np.asarray(E_GeV) / m_GeV
        t_dec = tau0_s * gamma
        sigma_T = physical_constants['Thomson cross section'][0] * 1e4   # cm^2
        m_e = physical_constants['electron mass energy equivalent in MeV'][0] * 1e-3
        GeV_erg = e * 1e16
        u_B = B_gauss ** 2 / (8 * np.pi)                                 # erg/cm^3
        rate_syn = (4 / 3) * sigma_T * (m_e / m_GeV) ** 2 * (c * 1e2) \
            * u_B * gamma / (m_GeV * GeV_erg)
        return 1.0 / (1.0 + t_dec * rate_syn)

    def cooled_decay_matrix(self, E_grid, m_GeV, tau0_s, B_gauss):
        """Exact decay-energy distribution of a synchrotron-cooling
        secondary: S[i, j] is the probability that a particle injected in
        energy bin i decays in bin j. For cooling Edot = -a E^2 and decay
        rate m / (tau0 E), the fraction decaying below E has the closed
        form

            F(E | E0) = exp(-E_c^2 (1/E^2 - 1/E0^2)),   E <= E0,

        with E_c^2 = m / (2 tau0 a) = E_br^2 / 2 (E_br is the f = 1/2
        break of decay_before_cooling). F(E0 | E0) = 1 exactly, so the
        rows are stochastic by construction — cooling migrates energy,
        never number (the migration the drop treatment lacks: injections
        far above the break pile up at E_br / sqrt(3) instead of
        vanishing). Mass below the grid floor is dropped (exponentially
        small whenever the break is on-grid)."""
        E = np.asarray(E_grid, dtype=float)
        # invert the drop closed form at the top bin (largest 1 - f, best
        # conditioned): (1/f - 1) = (E/E_br)^2
        f0 = float(self.decay_before_cooling(E[-1], m_GeV, tau0_s, B_gauss))
        if f0 >= 1.0:               # break far above the grid: no cooling
            return np.eye(len(E))
        E_c2 = E[-1] ** 2 * f0 / (1.0 - f0) / 2.0
        edges = np.concatenate([[E[0] ** 2 / E[1]], np.sqrt(E[:-1] * E[1:]),
                                [E[-1] ** 2 / E[-2]]])
        lo = np.minimum(edges[None, :-1], E[:, None])
        hi = np.minimum(edges[None, 1:], E[:, None])
        S = (np.exp(E_c2 / E[:, None] ** 2 - E_c2 / hi ** 2)
             - np.exp(E_c2 / E[:, None] ** 2 - E_c2 / lo ** 2))
        return np.clip(S, 0.0, None)

    def neutrino_production(self, L=None, alpha=None, mass_range=None, boost_range=None,
                            true_range=None, N_pion=None, charged_fraction=1/3, P=None,
                            weights=None, B_gauss=None, kaons=False, cooling='drop'):
        """Neutrino yields from the decay of the photomeson pions and their muons,

            pi+ -> mu+ nu_mu,   mu+ -> e+ nu_e antinu_mu      (and c.c.)

        with the full decay spectra and the muon helicity dependence of
        Huemmer et al. (2010), Eqs. (68)-(71): the direct nu_mu is the exact
        two-body box, the muon is produced left/right-handed with the
        x-dependent probabilities of the pi -> mu kinematics, and the muon
        decays with the helicity-dependent distributions — their largest
        single correction to the neutrino spectra.

        Pion inputs: when N_pion is not given and the core carries the
        charge-resolved kernels (photomeson='kernels'), pi+ and pi- are
        folded separately (self.photomeson_kernels), making charged_fraction
        irrelevant; a lumped N_pion= input follows the legacy convention
        (charged_fraction of it charged, split evenly between pi+-).

        Optional synchrotron cooling before decay (`B_gauss`): pions and
        muons are suppressed per energy bin by the closed-form
        `decay_before_cooling` factors, the classic pi/mu spectral breaks
        at photospheric field strengths, at O(n_boosts) extra cost.

        The default cooling treatment ('drop') deletes the cooled
        fraction: the no-migration approximation, which discards real
        flux, since cooled secondaries do not vanish, they decay at lower
        energy. `cooling='migrate'` instead applies the exact cooled-decay
        transport (`cooled_decay_matrix`): every secondary decays
        somewhere, piling up at about E_br / sqrt(3) and filling the
        spectrum below the breaks, the treatment that matches kinetic
        codes (NeuCosmA) around and below the cooling breaks. Same
        omissions as 'drop': no escape or adiabatic term in the secondary
        kinetics (decay times are far shorter than the dynamical times at
        all relevant energies) and helicity labels preserved through
        cooling.

        Optional kaon component (`kaons=True`): K+ and K- from the SOPHIA
        tables (requires `photomeson_spectra` at construction, since the
        interaction-type kernels carry no strangeness channel), decaying
        through the leading mode K -> mu nu_mu (BR 0.636, the Huemmer et
        al. treatment); the kaon muons share the pion muon pipeline. Being
        3.5x heavier and shorter lived than pions, kaons largely evade the
        `B_gauss` cooling suppression and dominate the extreme end of the
        spectrum in strongly magnetized sources.

        Parameters
        ----------
        L, alpha, mass_range, boost_range, true_range, P, weights :
            Propagation arguments forwarded to the pion folds when
            `N_pion` is not given (requires `photomeson='kernels'`).
        N_pion : ndarray, optional
            Lumped pion yields `(n_boosts, n_L)`, legacy path.
        charged_fraction : float, optional
            Charged share of a lumped `N_pion` input only.
        B_gauss : float, optional
            Comoving magnetic field for the cooling factors. None (default)
            turns cooling off.
        kaons : {False, True, 'K+'}, optional
            Include the K -> mu nu_mu component. Default False, which
            needs the SOPHIA kernels. True carries both K+ and K-; 'K+'
            is the Huemmer et al. and NeuCosmA scope (K- dropped), so the
            kaon shoulder then exists only in the neutrino channels, the
            antineutrino shoulder vanishing (charge tagged: only
            K- -> mu- nubar_mu feeds antineutrinos directly).
        cooling : {'drop', 'migrate'}, optional
            'drop' (default) discards the cooled secondaries; 'migrate'
            transports them to their decay energies (requires `B_gauss`).

        Returns
        -------
        E_nu : ndarray
            Neutrino energy grid in GeV (`m_pi * self.boosts`).
        N_nu : dict
            Cumulative yields, shape of the pion input: `'nu_mu'`,
            `'nu_e'` (neutrino plus antineutrino totals, backward
            compatible) and `'detail'` with the separate `'nu_mu'`,
            `'nubar_mu'`, `'nu_e'`, `'nubar_e'` components.
        """
        if cooling not in ('drop', 'migrate'):
            raise ValueError("cooling must be 'drop' or 'migrate'")
        if cooling == 'migrate' and B_gauss is None:
            raise ValueError("cooling='migrate' needs B_gauss=")
        if kaons not in (False, True, 'K+'):
            raise ValueError("kaons must be False, True, or 'K+'")

        C = self._pion_decay_chain()

        if N_pion is not None:
            N_charged = charged_fraction * np.asarray(N_pion)
            N_plus = N_minus = 0.5 * N_charged
        elif getattr(self, 'photomeson_kernels', None) is not None:
            kw = dict(alpha=alpha, mass_range=mass_range, boost_range=boost_range,
                      true_range=true_range, P=P, weights=weights)
            N_plus = self._photomeson_fold(self.photomeson_kernels['pi+'], L,
                                           scaling_group='pi+', **kw)
            N_minus = self._photomeson_fold(self.photomeson_kernels['pi-'], L,
                                            scaling_group='pi-', **kw)
        else:
            N_lumped = self.pion_production(L, alpha=alpha, mass_range=mass_range,
                                            boost_range=boost_range,
                                            true_range=true_range, P=P, weights=weights)
            N_plus = N_minus = 0.5 * charged_fraction * np.asarray(N_lumped)

        if kaons:
            kernels = getattr(self, 'photomeson_kernels', None) or {}
            if 'K+' not in kernels:
                raise ValueError(
                    "kaons=True requires the SOPHIA kernels — build the core "
                    "with photomeson_spectra= (the interaction-type kernels "
                    "carry no strangeness channel)")
            CK = self._kaon_decay_chain()
            kwK = dict(alpha=alpha, mass_range=mass_range,
                       boost_range=boost_range, true_range=true_range,
                       P=P, weights=weights)
            NK_plus = self._photomeson_fold(kernels['K+'], L,
                                            scaling_group='K+', **kwK)
            # kaons='K+' is the Huemmer et al. / NeuCosmA scope (K- dropped
            # as ~2x suppressed): the direct nubar_mu kaon shoulder vanishes
            NK_minus = (self._photomeson_fold(kernels['K-'], L,
                                              scaling_group='K-', **kwK)
                        if kaons is True and 'K-' in kernels
                        else np.zeros_like(NK_plus))

        E_nu = C['mpi'] * self.boosts
        fold = lambda D, N: np.einsum('ij,il->jl', D, N)

        if B_gauss is not None and cooling == 'migrate':
            # exact cooled-decay transport: the secondaries are moved to
            # their decay energies instead of dropped (PDG lifetimes)
            S_pi = self.cooled_decay_matrix(E_nu, C['mpi'], 2.6033e-8, B_gauss)
            S_mu = self.cooled_decay_matrix(E_nu, C['mmu'], 2.1969811e-6, B_gauss)
            N_plus = fold(S_pi, N_plus)
            N_minus = fold(S_pi, N_minus)
            if kaons:
                S_K = self.cooled_decay_matrix(CK['mK'] * self.boosts,
                                               CK['mK'], CK['tauK'], B_gauss)
                NK_plus = fold(S_K, NK_plus)
                NK_minus = fold(S_K, NK_minus)
            f_mu = None
        elif B_gauss is not None:
            # PDG rest-frame lifetimes (not CODATA constants)
            f_pi = self.decay_before_cooling(E_nu, C['mpi'], 2.6033e-8, B_gauss)
            f_mu = self.decay_before_cooling(E_nu, C['mmu'], 2.1969811e-6, B_gauss)
            N_plus = f_pi[:, None] * N_plus
            N_minus = f_pi[:, None] * N_minus
            if kaons:
                f_K = self.decay_before_cooling(CK['mK'] * self.boosts,
                                                CK['mK'], CK['tauK'], B_gauss)
                NK_plus = f_K[:, None] * NK_plus
                NK_minus = f_K[:, None] * NK_minus
        else:
            f_mu = 1.0

        # muons per helicity (cooled before their decay when B is given);
        # CP mirror: the spectrum feeding mu+_R (Eq. 68) feeds mu-_L
        muR_p, muL_p = fold(C['muR'], N_plus), fold(C['muL'], N_plus)
        muL_m, muR_m = fold(C['muR'], N_minus), fold(C['muL'], N_minus)
        if kaons:
            bK = CK['BR_munu']
            muR_p += bK * fold(CK['muR'], NK_plus)
            muL_p += bK * fold(CK['muL'], NK_plus)
            muL_m += bK * fold(CK['muR'], NK_minus)
            muR_m += bK * fold(CK['muL'], NK_minus)
        if B_gauss is not None and cooling == 'migrate':
            muR_p, muL_p = fold(S_mu, muR_p), fold(S_mu, muL_p)
            muR_m, muL_m = fold(S_mu, muR_m), fold(S_mu, muL_m)
        elif B_gauss is not None:
            for mu in (muR_p, muL_p, muR_m, muL_m):
                mu *= f_mu[:, None] if np.ndim(f_mu) else f_mu

        detail = {
            'nu_mu':    fold(C['box'], N_plus) + fold(C['P70m'], muR_m) + fold(C['P70p'], muL_m),
            'nubar_mu': fold(C['box'], N_minus) + fold(C['P70p'], muR_p) + fold(C['P70m'], muL_p),
            'nu_e':     fold(C['P71p'], muR_p) + fold(C['P71m'], muL_p),
            'nubar_e':  fold(C['P71m'], muR_m) + fold(C['P71p'], muL_m),
        }
        if kaons:
            detail['nu_mu'] += bK * fold(CK['box'], NK_plus)
            detail['nubar_mu'] += bK * fold(CK['box'], NK_minus)
        N_nu = {'nu_mu': detail['nu_mu'] + detail['nubar_mu'],
                'nu_e': detail['nu_e'] + detail['nubar_e'],
                'detail': detail}
        return E_nu, N_nu

    @property
    def species_masses(self):
        """Masses of `self.species` in GeV, per the `masses` constructor
        argument (default 'nubase').

        Returns
        -------
        ndarray
            Shape `(len(self.species),)`.
        """
        if not hasattr(self, '_species_masses'):
            self._species_masses = np.array([self._mass_fn(Z, A) for Z, A in self.species])
        return self._species_masses

    def energy_of_boost(self, species, boost):
        """Total energy E = boost * m(Z, A) in GeV for the species (Z, A).

        Parameters
        ----------
        species : tuple of int
            `(Z, A)`.
        boost : float or ndarray
            Lorentz factor(s).

        Returns
        -------
        float or ndarray
            Energy in GeV, same shape as `boost`.

        Examples
        --------
        >>> from crisp.core import InteractionCore
        >>> from crisp.photonuclear_cross_sections import PSB_model
        >>> core = InteractionCore(xsec_model=PSB_model())
        >>> core.energy_of_boost((26, 56), 1e11)  # Fe-56 at Gamma = 1e11
        np.float64(5208977673560.206)
        """
        Z, A = species
        return np.asarray(boost) * self._mass_fn(Z, A)

    def boost_of_energy(self, species, energy_GeV):
        """Lorentz factor E / m(Z, A) for the species (Z, A). The inverse
        of `energy_of_boost`.

        Parameters
        ----------
        species : tuple of int
            `(Z, A)`.
        energy_GeV : float or ndarray
            Energy in GeV.

        Returns
        -------
        float or ndarray
            Lorentz factor(s), same shape as `energy_GeV`.
        """
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

    def _previous_index(self, boost_range):
        """Index of the nearest tabulated boost <= boost_range (the
        'previous'-kind lookup interp1d used to perform), without
        constructing an interp1d object every call. Replicates
        interp1d(kind='previous')'s bounds-checking exactly (raises
        ValueError out of range) so existing callers relying on that
        guard keep it."""
        x = np.asarray(boost_range, dtype=float)
        oob = (x < self.boosts[0]) | (x > self.boosts[-1])
        if np.any(oob):
            bad = x[oob][0]
            raise ValueError(
                f'A value ({bad}) in x_new is out of the interpolation '
                f'range [{self.boosts[0]}, {self.boosts[-1]}].')
        return np.clip(np.searchsorted(self.boosts, x, side='right') - 1,
                       0, len(self.boosts) - 1)

    def _tensor_at(self, boost_range, mass_range=None, which='tensor'):
        """self.tensor (or self.light_prod_tensor if which='light') at
        boost_range, optionally restricted to mass_range along both
        species axes (mass_range is only ever used with which='tensor':
        self.tensor is (n_species, n_species, n_boosts), but
        self.light_prod_tensor is (n_light, n_species, n_species,
        n_boosts) -- one extra leading axis -- so its species axes are
        never restricted here today). Restriction is FUSED into the
        lookup via one fancy-index when mass_range is given, so the full
        species axis is never materialized -- exact/bit-identical to
        calling self.interpolator(boost_range) and restricting
        afterward: the 'previous' lookup is independent per species
        pair, along the boost axis only, so restricting before or after
        it commutes. The boost axis is always LAST regardless of which
        tensor (indexed via an ellipsis, not a fixed axis position), so
        this is correct for both shapes."""
        idx = self._previous_index(boost_range)
        t = self.tensor if which == 'tensor' else self.light_prod_tensor
        if mass_range is not None:
            return t[np.ix_(mass_range, mass_range, idx)]
        return t[..., idx]

    def interpolator(self, boostval):
        """self.tensor at boostval ('previous'-step lookup along the
        boost axis, unrestricted). A real method rather than a
        per-instance lambda, so InteractionCore stays picklable."""
        return self._tensor_at(boostval)

    def interpyields(self, boostval):
        """self.light_prod_tensor at boostval. Must use the same
        'previous' lookup as interpolator: the destruction (tensor) and
        production (light-yield) rates have to come from the same boost
        bin, or nucleon conservation breaks at interpolated boosts."""
        return self._tensor_at(boostval, which='light')

    def _diagonal_fixed_tensor(self, boost_range, mass_range, guard_single=False):
        """Interaction tensor restricted to mass_range (if given), with the
        diagonal zeroed then recomputed as total outflow (row-sum, still
        over the full mass_range, so outflow to absorbed species is
        accounted for before those columns are ever dropped). Still
        mass_range-shaped, NOT yet restricted to true_range -- the first
        of two shared stages (see _restricted_generator), split out because
        light_production_cumulative needs this intermediate form for an
        extra computation (its 'leading arrival' rates) before applying
        its own true_range restriction.

        guard_single : if True, skip the diagonal zero/recompute steps
        entirely when the mass_range-restricted tensor has exactly one
        species (matches pdf_boost_range's and get_distribution_parameters's
        pre-existing special case for a degenerate 1x1 tensor; every other
        caller never had this guard, so it defaults off to preserve their
        exact prior behavior)."""
        reduced_tensor = self._tensor_at(boost_range, mass_range)

        if not guard_single or len(reduced_tensor[:, :, 0]) > 1:
            reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k]))
                                         for k in range(reduced_tensor.shape[-1])])
            reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)
        return reduced_tensor

    def _restricted_generator(self, boost_range, mass_range, true_range, guard_single=False):
        """_diagonal_fixed_tensor, further restricted to true_range (the
        non-absorbed rows/columns) -- shared preprocessing for
        light_cascade_production, cdf_boost_range, and pdf_boost_range
        (light_production_cumulative uses _diagonal_fixed_tensor directly
        instead -- see its own docstring note above; species_evolution_
        boost_range and _photomeson_fold no longer need this at all --
        the former IS the canonical exact-path implementation, the latter
        now delegates to it instead of re-deriving the same tensor).

        Returns (reduced_tensor, indices) where indices is
        [mass_range.index(v) for v in true_range] -- every caller needs
        this to index alpha/species lookups against the restricted array."""
        reduced_tensor = self._diagonal_fixed_tensor(boost_range, mass_range, guard_single=guard_single)
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]
        return reduced_tensor, indices

    @staticmethod
    def _expm_per_boost(M, L):
        """expm(L * M[:, :, b]) for every boost bin b, stacked into one
        (n_b, ...) array. Batched per boost bin rather than in one call
        across all of them -- avoids a scipy batched-expm precision issue
        that appears when different boost bins' matrices have widely
        different norms. Shared by _photomeson_fold [no longer -- see its
        own docstring], light_cascade_production, light_production_
        cumulative, cdf_boost_range, pdf_boost_range.

        MEMORY: for array L this returns and holds the FULL stack, shape
        (n_b, n_L, sz, sz) where sz is the matrix's own side (n_true+1 for
        cdf/pdf_boost_range's bare tensor, n_true+7 for
        light_cascade_production's heavy+light block, n_true+25 for
        light_production_cumulative's heavy+light+3-tally block) -- i.e.
        n_b * n_L * sz^2 * 8 bytes. At a few hundred species and a
        few-dozen-point L this is negligible; at the package's own
        ultra-heavy end (n_true ~ 1000+, as in
        Attenuation_Factor_UltraHeavy.ipynb's A<=208 network) this is the
        SAME class of blowup species_evolution_boost_range's exact path
        used to have before it was changed (this session) to reduce with
        alpha one boost at a time instead of materializing this stack --
        that fix was NOT propagated to these four callers. None of them
        are exercised at that scale by any notebook in this repo today,
        so nothing is currently broken, but a caller reaching for
        light_cascade_production/light_production_cumulative/
        cdf_boost_range/pdf_boost_range at a comparable species count and
        L-grid density should expect the same wall in the same place, not
        yet fixed here."""
        n_b = M.shape[-1]
        if type(L) is np.ndarray:
            return np.stack([expm(L[:, None, None] * M[:, :, b]) for b in range(n_b)])
        return expm(np.moveaxis(M * L, -1, 0))

    @staticmethod
    def _stiffness_rate(reduced_tensor):
        """Peak interaction rate [1/Mpc] driving both the exact path's
        expm scaling-and-squaring cost and the substep path's step count
        -- restricted to boost bins whose rate isn't negligible relative
        to the peak (far-tail boost bins can carry spuriously huge
        tabulated/extrapolated rates, per reprocessed_nucleons)."""
        n_true, _, n_b = reduced_tensor.shape
        diag_rate = np.stack([-np.diag(reduced_tensor[:, :, b]) for b in range(n_b)], axis=1)
        rate_per_boost = diag_rate.max(axis=0) if n_true else np.zeros(n_b)
        rate_max = rate_per_boost.max() if n_b else 0.0
        return rate_per_boost[rate_per_boost > 1e-6 * rate_max].max() if rate_max > 0 else 0.0

    @staticmethod
    def _exact_is_cheaper(n_true, L, lam_act):
        """Cost-proxy comparison used by method='auto' for a zero-drift
        propagation: direct per-distance expm (the 'exact' path) vs. a
        chained-substep evaluation (the 'substep' path), both solving the
        identical equation. expm's cost is dominated by ~log2(L_max*rate)
        scaling-and-squaring doublings, each an O(n_true^3) matrix-matrix
        multiply, paid once per requested distance; the substep path's
        cost is dominated by its step count (sized to resolve every
        requested checkpoint under resolve_checkpoints), each a cheap
        O(n_true^2) matrix-vector multiply. Back propagation (any
        negative L) always forces exact -- the substep path has no such
        implementation."""
        L_arr = np.atleast_1d(np.asarray(L, dtype=float))
        if L_arr.size == 0 or np.any(L_arr < 0):
            return True
        L_max = float(np.max(L_arr))
        if L_max <= 0.0 or lam_act <= 0.0:
            return True

        n_L = L_arr.size
        s_max = max(1, int(np.ceil(np.log2(max(L_max * lam_act, 1.0)))))
        cost_exact = n_L * s_max * n_true

        m_stiff = max(1, int(np.ceil(L_max * lam_act / 60.0)))
        if n_L > 1:
            gaps = np.diff(np.sort(L_arr))
            gaps = gaps[gaps > 0]
            min_gap = gaps.min() if gaps.size else L_max
            m_res = max(1, int(np.ceil(L_max / min_gap)))
        else:
            m_res = 1
        m_total = max(m_stiff, m_res)

        return cost_exact <= m_total

    def species_evolution_boost_range(self, L, alpha=None, mass_range=None, boost_range=None,
                                      true_range=None, coherent_loss=None, energy_loss=None,
                                      resolve_checkpoints=None, method=None):
        """The core transport call: returns the probability (or population)
        of each species at positions L for a range of boosts. If the
        distances are negative and in decreasing order, this is
        equivalent to back propagation.

        Parameters
        ----------
        L : float or ndarray
            Distance(s) [Mpc] at which the population will be evaluated.
        alpha : ndarray
            Injection vector (sum of entries must equal one).
        mass_range : list of int, optional
            Species to be included in the matrix. If None, all species
            are included.
        true_range : list of int
            Species not part of the absorption range (excludes indices
            for species that are part of the absorption state). If none
            is given, the last species in `mass_range` is considered the
            absorption state.
        boost_range : ndarray, optional
            The boost grid to evaluate on. The full grid by default
            (None).
        coherent_loss : float, optional
            Scalar continuous-loss rate `b = -dln(gamma)/dx` [1/Mpc],
            species- and boost-independent (e.g. adiabatic cooling
            Gamma/R). Applied every sub-step as an exact rigid shift in
            ln(gamma) (`shift_log_boost`'s scalar branch, no numerical
            diffusion), the coherent-inhomogeneity treatment of the
            methods paper Sect. 3.1.
        energy_loss : ndarray, optional
            Continuous fractional energy-loss rate on `boost_range`
            (uniform log grid): shape `(n_b,)` applied to every
            `true_range` species, or `(n_b, n_true)` per species (e.g.
            synchrotron and/or pair losses, both boost-dependent even for
            a single species, so no exact treatment exists). Applied
            every sub-step via conservative cloud-in-cell
            (`shift_log_boost`'s array branch), the dispersive-
            inhomogeneity treatment of Sect. 3.2.
        resolve_checkpoints : bool, optional
            Only meaningful for a sub-stepped evaluation (`coherent_loss`
            or `energy_loss` active, or `method='auto'`/`'substep'`
            choosing the sub-stepped path for a zero-drift case, see
            `method` below). The sub-step count is normally chosen from
            tensor/rate stiffness alone, independent of how many or how
            closely spaced the requested `L` values are: fine when only
            a handful of checkpoints are requested, but it means
            closely-spaced checkpoints can round onto the same sub-step
            position (a quantization artifact, visible as a staircase if
            the returned values are read off as a smooth function of
            `L`). True additionally floors the sub-step spacing at the
            tightest gap between requested checkpoints, so every one
            resolves to its own position, at a modest cost increase.
            Default None: resolves to True when the sub-stepped path is
            standing in for an exact, zero-drift evaluation (accuracy
            matters, since nothing else asked for approximate output),
            and to False, the original and cheaper behavior, when a
            genuine `coherent_loss`/`energy_loss` was requested.
        method : {None, 'exact', 'substep', 'auto'}, optional
            Only meaningful when `coherent_loss`/`energy_loss` represent
            no real continuous loss (`coherent_loss` in `(None, 0.0)` and
            `energy_loss` is None). With an actual loss rate active, the
            sub-stepped path is the only option regardless of this
            argument (raises `ValueError` only if `method='exact'` is
            explicitly requested there, since no closed form exists for
            L-dependent drift; the default, None, never raises this, it
            simply defers to the sub-stepped path whenever real drift is
            active, the same as before this argument existed). For the
            zero-drift case, both the direct 'exact' expm-per-distance
            evaluation and a chained-substep evaluation (see
            `_species_evolution_with_drift`) solve the identical
            equation: they differ only in cost, which can favor either
            one depending on species count, `L_max`, tensor stiffness,
            and how many distances are requested (see
            `_exact_is_cheaper`). 'exact' is cheap for small to moderate
            species counts and `L_max * rate` products, but its cost
            grows with the number of scipy `expm` scaling-and-squaring
            doublings needed, which can make it dramatically slower than
            'substep' at large species counts evaluated over a wide
            distance range (measured directly: about 20x for a
            1184-species network over 200 Mpc). None resolves to 'exact'
            in the zero-drift case, so that every other method built on
            top of this one (several duplicate its exact-path math inline
            rather than calling it, e.g. `light_cascade_production`,
            `light_production_cumulative`) keeps agreeing with it to
            machine precision by default; pass `method='auto'` explicitly
            to let this call estimate both costs and pick the cheaper one
            (or `'substep'` to force it), accepting that the result will
            then only agree with those other methods' internal
            computations to substep-level precision, not machine
            precision (falls back to 'exact' for back propagation either
            way, which 'substep' cannot do).

        When either `coherent_loss` or `energy_loss` is a genuine nonzero
        rate, `L` must be a non-negative scalar or a non-negative,
        ascending array (forward propagation only; back propagation with
        active continuous losses is not implemented), and the method
        switches from the single exact `expm` per boost bin to a chain of
        sub-steps (interaction, then drift) between L=0 and each requested
        position, with the sub-step count chosen automatically from the
        stiffness of the tensor and the supplied rates (the same
        convention as `reprocessed_nucleons`: every rate times sub-step
        stays bounded). With `coherent_loss=None`, `energy_loss=None`,
        and `method='exact'` (or 'auto' resolving to 'exact'), this path
        is unchanged and bit-identical to before this argument existed.

        Injection with losses active: without drift, boost bins never
        mix, so broadcasting one species-only alpha to every bin and
        reading off row b is exactly equivalent to injecting alpha at b
        in isolation, and that equivalence is what lets the no-loss path
        solve every boost bin's independent problem in one batched call.
        Drift breaks it: `shift_log_boost` mixes neighboring bins, so a
        uniformly-broadcast alpha stops meaning "the answer for injection
        at this bin" and instead means "the state from injecting at every
        bin at once, summed in by the drift", rarely what's wanted. Pass
        alpha as a `(n_boost_range, n_true)` array (matching
        `mass_range`'s `true_range` columns) for the general,
        boost-resolved case (e.g. injection confined to a limited boost
        window); the plain `(n_true,)` vector remains valid as the
        special case of a genuinely uniform-in-boost injection, broadcast
        internally to every bin.

        Returns
        -------
        ndarray
            `P`, the population/probability array. Shape
            `(n_boost, n_L, n_true)` for array `L` (or
            `(n_boost, n_true)` for scalar `L`) when `coherent_loss` and
            `energy_loss` are both inactive; one extra "absorbed" column
            (`n_true + 1`) when the sub-stepped path is used (drop it
            with `P[..., :len(true_range)]` if not needed, as this
            module's own callers do).

        Examples
        --------
        >>> import numpy as np
        >>> from crisp.core import InteractionCore
        >>> from crisp.photonuclear_cross_sections import PSB_model
        >>> core = InteractionCore(xsec_model=PSB_model())
        >>> alpha, mr, tr, _ = core.get_distribution_parameters(
        ...     mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        ...     absorption_type=('only mass', []))
        >>> distances = np.array([1.0, 10.0, 50.0, 200.0])  # Mpc
        >>> P = core.species_evolution_boost_range(
        ...     distances, alpha, mr, core.boosts, tr)
        >>> i_fe = tr.index(core.species.index((26, 56)))
        >>> P[82, :, i_fe]  # Fe-56's own surviving fraction near 99 EeV
        array([0.99645193, 0.96508049, 0.83717775, 0.49121398])
        """

        if boost_range is None:
            boost_range = self.boosts
        boost_range = np.asarray(boost_range, dtype=float)

        reduced_tensor = self._tensor_at(boost_range, mass_range)

        # make diagonal zero
        reduced_tensor -= np.dstack([np.diag(np.diag(reduced_tensor[:, :, k])) for k in range(reduced_tensor.shape[-1])])
        # recompute diagonal including absorption states
        reduced_tensor -= np.stack([np.diag(row) for row in reduced_tensor.sum(axis=1).T], axis=2)
        # reduce excluding absorption states
        indices = [mass_range.index(ival) for ival in true_range]
        reduced_tensor = reduced_tensor[np.ix_(indices, indices, range(len(boost_range)))]

        if method is not None and method not in ('auto', 'exact', 'substep'):
            raise ValueError("method must be 'auto', 'exact', 'substep', or None")

        zero_drift = (coherent_loss is None or coherent_loss == 0.0) and energy_loss is None

        if method == 'exact' and not zero_drift:
            # only trips on an EXPLICIT method='exact' -- the default
            # (None) never raises this; it just defers to the sub-stepped
            # path below whenever real drift is active, exactly as before
            # this argument existed.
            raise ValueError("method='exact' has no closed form for an "
                             "active coherent_loss/energy_loss; use "
                             "method='substep' or 'auto', or leave method "
                             "unset (defaults to the sub-stepped path when "
                             "drift is active)")

        if not zero_drift:
            use_exact = False           # only the sub-stepped path can apply real drift
        elif method is None or method == 'exact':
            use_exact = True            # unset default (and explicit 'exact'): always exact
        elif method == 'substep':
            use_exact = False
        else:                            # method == 'auto'
            lam_act = self._stiffness_rate(reduced_tensor)
            use_exact = self._exact_is_cheaper(reduced_tensor.shape[0], L, lam_act)

        if use_exact:
            _, c, d = reduced_tensor.shape
            t_vs_boost = np.atleast_3d(reduced_tensor.sum(axis=1))
            bigLambda = np.append(np.append(reduced_tensor, np.swapaxes(t_vs_boost, 1, 2), axis=1), np.zeros((1, c+1, d)), axis=0)

            alpha_arr = np.asarray(alpha, dtype=float)
            if alpha_arr.ndim == 1:
                # single composition, broadcast identically to every boost
                # bin -- exact since bins never mix without drift, so this
                # is equivalent to solving each bin's injection in isolation.
                vec_full = np.append(alpha_arr[indices], 0.0)          # (c+1,)
            else:
                # boost-resolved composition: row b must be propagated with
                # boost b's own transfer matrix, not a single shared vector
                # -- indexing alpha_arr's boost axis with species indices
                # (what a naive np.matmul would do here) is wrong.
                vec_full = np.zeros((alpha_arr.shape[0], c + 1))
                vec_full[:, :c] = alpha_arr[:, indices]                # (n_boost, c+1)

            is_1d = alpha_arr.ndim == 1
            array_L = type(L) is np.ndarray
            # Reduce with alpha ONE BOOST AT A TIME: stacking every boost's
            # own (n_L, c+1, c+1) exponential before reducing (the previous
            # form of this code) peaks at n_boost times a single boost's
            # array -- for a large species count (e.g. this package's own
            # A<=208 ultra-heavy network, c ~ 1180) and a several-dozen-point
            # distance grid, that stack alone is tens of GB even though any
            # one boost's batch is well under 1 GB. Per-boost sub-batching
            # over L also avoids a scipy batched-expm precision issue that
            # appears when matrices with widely different norms (as
            # different boost bins routinely have) share one batch.
            out_shape = (d, len(L), c + 1) if array_L else (d, c + 1)
            total = np.empty(out_shape)
            for b in range(d):
                Eb = expm(L[:, None, None] * bigLambda[:, :, b]) if array_L \
                    else expm(bigLambda[:, :, b] * L)
                vb = vec_full if is_1d else vec_full[b]
                total[b] = vb @ Eb

            return total

        # sub-stepped path: either a genuine coherent_loss/energy_loss was
        # requested, or method chose (explicitly or via 'auto''s cost
        # estimate) to evaluate a zero-drift propagation this way instead.
        # Default resolve_checkpoints accordingly: standing in for an
        # exact evaluation should actually resolve every checkpoint (True);
        # a real drift request keeps the original, cheaper default (False).
        if resolve_checkpoints is None:
            resolve_checkpoints = zero_drift

        alpha_arr = np.asarray(alpha, dtype=float)
        alpha_idx = alpha_arr[indices] if alpha_arr.ndim == 1 else alpha_arr[:, indices]
        return self._species_evolution_with_drift(
            reduced_tensor, boost_range, L, alpha_idx,
            coherent_loss=coherent_loss, energy_loss=energy_loss,
            resolve_checkpoints=resolve_checkpoints)

    def _species_evolution_with_drift(self, reduced_tensor, boost_range, L, alpha0,
                                      coherent_loss=None, energy_loss=None,
                                      resolve_checkpoints=False):
        """Substepped companion to species_evolution_boost_range for
        coherent_loss / energy_loss != None -- see its docstring.

        reduced_tensor : (n_true, n_true, n_b) generator, already restricted
                     to true_range (as built by species_evolution_boost_range
                     before the exact-path augmentation).
        alpha0 : injection on true_range: (n_true,) broadcast to every boost
                     bin, or (n_b, n_true) already boost-resolved (see the
                     public method's docstring for why this matters once
                     drift is active).

        Interactions are exact per sub-step (one expm per boost bin of the
        augmented (n_true+1, n_true+1) generator); the absorbed (n_true-th)
        slot only receives inflow and does not itself drift. Losses are
        applied between interaction sub-steps via shift_log_boost: exactly
        for coherent_loss, conservatively (CIC) for energy_loss.

        The sub-step grid is chosen ONCE for the whole path [0, L[-1]] and
        the per-boost expm precomputed once from it (a single set of n_b
        dense (n_true+1)^2 exponentials), independent of how many, or how
        irregularly spaced, checkpoints are requested: state is recorded
        into the output whenever the running position passes a requested L
        (the same 'advance until past checkpoint' pattern used to fix the
        time-resolution bug this generalizes). Sizing the grid per
        REQUESTED INTERVAL instead (recomputing the expm set every time)
        was tried and is why this method exists in its current form: with
        a log-spaced checkpoint grid, essentially every interval has a
        distinct spacing, so a per-interval expm cache never hits and the
        (n_true+1)^2 exponentials get recomputed hundreds of times over --
        catastrophic at n_true ~ 500. `resolve_checkpoints=True` is a
        middle ground: still ONE grid, ONE precomputed expm set, just sized
        finer (a resolution floor on top of the stiffness floor) so checkpoints
        don't round onto shared positions -- not a re-derivation per interval.
        """
        n_true, _, n_b = reduced_tensor.shape
        dlnb = np.log(boost_range[1] / boost_range[0])
        alpha0 = np.asarray(alpha0, dtype=float)
        inj = np.broadcast_to(alpha0, (n_b, n_true)) if alpha0.ndim == 1 else alpha0
        if inj.shape != (n_b, n_true):
            raise ValueError(f'alpha must have shape ({n_true},) or '
                             f'({n_b}, {n_true}), got {alpha0.shape}')

        scalar_L = not isinstance(L, np.ndarray)
        L_in = np.array([float(L)]) if scalar_L else np.asarray(L, dtype=float)
        if L_in.ndim != 1 or L_in.size == 0 or L_in[0] < 0 or np.any(np.diff(L_in) < 0):
            raise ValueError('coherent_loss/energy_loss require a non-negative, '
                             'ascending L (forward propagation only)')
        # integration always starts at the injection, L = 0 (matching the
        # no-loss path's convention); prepend it as a checkpoint if the
        # caller's own grid does not start there, and drop it again on return
        prepend = L_in[0] > 0.0
        L_arr = np.concatenate([[0.0], L_in]) if prepend else L_in
        L_max = float(L_arr[-1])

        e_loss = None if energy_loss is None else np.asarray(energy_loss, dtype=float)
        c_loss = None if coherent_loss is None else float(coherent_loss)

        n_L = len(L_arr)
        out = np.zeros((n_b, n_L, n_true + 1))
        state = np.zeros((n_b, n_true + 1))
        state[:, :n_true] = inj
        out[:, 0, :] = state
        k_rec = 1
        while k_rec < n_L and L_arr[k_rec] <= 0.0:
            out[:, k_rec, :] = state
            k_rec += 1

        if L_max > 0.0 and k_rec < n_L:
            # stiffness estimate for the sub-step count, mirroring
            # reprocessed_nucleons: cap every rate * sub-step at <= 60 so
            # expm's internal scaling stays well conditioned. Far-tail
            # boost bins can carry spuriously huge tabulated/extrapolated
            # rates (as noted in reprocessed_nucleons), so restrict to
            # bins whose rate is not negligible relative to the peak
            # before taking the max.
            diag_rate = np.stack([-np.diag(reduced_tensor[:, :, b]) for b in range(n_b)], axis=1)
            rate_per_boost = diag_rate.max(axis=0) if n_true else np.zeros(n_b)
            rate_max = rate_per_boost.max() if n_b else 0.0
            lam_act = rate_per_boost[rate_per_boost > 1e-6 * rate_max].max() if rate_max > 0 else 0.0
            if c_loss is not None:
                lam_act = max(lam_act, abs(c_loss) / dlnb)
            if e_loss is not None:
                lam_act = max(lam_act, np.abs(e_loss).max() / dlnb)

            m_stiff = max(1, int(np.ceil(L_max * lam_act / 60.0))) if lam_act > 0 else 1
            if resolve_checkpoints and n_L > 1:
                # resolution floor: also require the sub-step spacing to be
                # no coarser than the tightest gap between requested
                # checkpoints, so no two round onto the same position. This
                # is independent of the stiffness floor above -- take the
                # finer (larger m_total) of the two.
                gaps = np.diff(L_arr)
                gaps = gaps[gaps > 0]
                min_gap = gaps.min() if gaps.size else L_max
                m_res = max(1, int(np.ceil(L_max / min_gap)))
                m_total = max(m_stiff, m_res)
            else:
                m_total = m_stiff
            dl = L_max / m_total

            E_steps = np.empty((n_b, n_true + 1, n_true + 1))
            for b in range(n_b):
                Mb = np.zeros((n_true + 1, n_true + 1))
                Mb[:n_true, :n_true] = reduced_tensor[:, :, b]
                Mb[:n_true, n_true] = -reduced_tensor[:, :, b].sum(axis=1)
                E_steps[b] = expm(Mb * dl)

            shift_c = None if c_loss is None else c_loss * dl / dlnb
            if e_loss is None:
                shift_e = None
            elif e_loss.ndim == 1:
                shift_e = e_loss * dl / dlnb                        # (n_b,)
            else:
                shift_e = e_loss * (dl / dlnb)                      # (n_b, n_true)

            l_now = 0.0
            for _ in range(m_total):
                state = np.einsum('bi,bij->bj', state, E_steps)
                live = state[:, :n_true]
                if shift_c is not None:
                    live = shift_log_boost(boost_range, live, shift_c)
                if shift_e is not None:
                    live = shift_log_boost(boost_range, live, shift_e)
                state[:, :n_true] = live
                l_now += dl
                while k_rec < n_L and l_now >= L_arr[k_rec] * (1 - 1e-9):
                    out[:, k_rec, :] = state
                    k_rec += 1

        while k_rec < n_L:
            out[:, k_rec, :] = state
            k_rec += 1

        if scalar_L:
            return out[:, -1, :]
        return out[:, 1:, :] if prepend else out

    def light_secondaries_production(self, L, alpha=None, mass_range=None, boost_range=None, true_range=None, P=None, cumulative=False):
        """Returns the production rate of each light species at positions L for a
        range of boosts, in units of number per injected nucleus per Mpc.
        With cumulative=True the rate is integrated along L (trapezoid), giving
        the cumulative counts up to each distance.

        The light-yield tensor already contains the channel rates (yields are
        stored as multiplicity x rate), so the rate of light species k is the
        A-weighted fold of the heavy-cascade occupation with the yield rows,
        summed over ALL destination channels (including those into the
        absorbed states). Integrating the returned rate along L reproduces the
        cumulative 'emission' tally of light_production_cumulative.

        Parameters
        ----------
        L : float or ndarray
            Distance(s) [Mpc] at which the rate will be evaluated.
        alpha : ndarray
            Injection vector (sum of entries must equal one).
        mass_range : list of int, optional
            Species to be included in the matrix. If None, all species
            are included.
        boost_range : ndarray, optional
            Boost values to evaluate. The whole grid by default (None).
        P : ndarray, optional
            Precomputed heavy evolution, as returned by
            `species_evolution_boost_range` with the same arguments.
            Avoids re-solving the cascade ODE.
        cumulative : bool, optional
            If True, integrate the rate along `L` (trapezoid), giving the
            cumulative counts up to each distance. Default False.

        Returns
        -------
        ndarray
            `production`, shape `(6, n_boosts, n_L)`: `[He4, He3, H3, H2,
            p, n]`.
        """
        if boost_range is None:
            boost_range = self.boosts

        prod_mat = self.interpyields(boost_range)   # multiplicity x rate, /Mpc

        if mass_range is not None:
            prod_mat = prod_mat[np.ix_(range(prod_mat.shape[0]), mass_range, mass_range, range(len(boost_range)))]

        indices = [mass_range.index(ival) for ival in true_range]

        # total yield rate of each light species from parent i: summed over all
        # destination channels j (dropping any j loses the yields of channels
        # into the absorbed states)
        Y_tot = prod_mat[np.ix_(range(6), indices, range(prod_mat.shape[2]),
                                range(len(boost_range)))].sum(axis=2)   # (6, n_sp, n_b)

        if P is None:
            P = self.species_evolution_boost_range(L, alpha, mass_range, boost_range, true_range)

        # production rate of light species k at (boost b, distance m)
        production = np.einsum('bmi, kib -> kbm', P[:, :, :len(indices)], Y_tot)

        if cumulative:
            from scipy.integrate import cumulative_trapezoid
            production = cumulative_trapezoid(production, np.asarray(L), axis=-1, initial=0.0)
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
        The block rate matrix is [[Lambda_heavy, Y, abs], [0, M_light, 0],
        [0, 0, 0]], where Y[i,k] is the total production rate of light
        species k from heavy species i.

        Parameters
        ----------
        L : float or ndarray
            Distance(s) [Mpc] at which the distribution will be evaluated.
        alpha : ndarray
            Injection vector (must sum to one).
        mass_range : list of int
            Species indices to include. Must be provided together with
            `true_range`.
        boost_range : ndarray, optional
            Boost values to evaluate. Full grid by default.
        true_range : list of int
            Subset of `mass_range` that are not absorption states.

        Returns
        -------
        ndarray
            Shape `(6, n_boosts, n_L)` for array `L`, or `(6, n_boosts)`
            for scalar `L`.

        Notes
        -----
        Memory: solved via `_expm_per_boost` on an `(n_true+7)x(n_true+7)`
        block (heavy plus 6 light species plus absorbed), with the full
        stack held for array `L`. See `_expm_per_boost`'s own docstring
        for the scaling this implies at large `n_true`.
        """
        if boost_range is None:
            boost_range = self.boosts

        prod_mat = self.interpyields(boost_range)          # (6, n_sp_full, n_sp_full, n_b)
        if mass_range is not None:
            prod_mat = prod_mat[np.ix_(range(prod_mat.shape[0]), mass_range, mass_range, range(len(boost_range)))]

        reduced_tensor, indices = self._restricted_generator(boost_range, mass_range, true_range)

        n_sp = len(indices)
        n_b  = len(boost_range)

        # Production coupling: Y_block[i, k, b] = total production rate of light k from species i.
        # Sum over ALL daughters (including absorbed A=1 species) so that nucleon conservation holds
        # when combined with the absorbed probability column.  Restrict parents to indices, but
        # keep all daughters within mass_range before summing.
        Y_full  = prod_mat[np.ix_(range(6), indices, range(prod_mat.shape[2]), range(n_b))].sum(axis=2)
        Y_block = np.moveaxis(Y_full, 0, 1)                  # (n_sp, 6, n_b)

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

        expmatL = self._expm_per_boost(big_mat, L)
        if type(L) is np.ndarray:
            result  = np.matmul(alpha_aug, expmatL)          # (n_b, n_L, sz)
            light   = result[:, :, n_sp:n_sp+6]             # (n_b, n_L, 6)
        else:
            result  = np.matmul(alpha_aug, expmatL)          # (n_b, sz)
            light   = result[:, n_sp:n_sp+6]                 # (n_b, 6)
        return np.moveaxis(light, -1, 0)                     # (6, n_b, n_L) or (6, n_b)

    def light_production_cumulative(self, L, alpha=None, mass_range=None, boost_range=None,
                                    true_range=None, channel='total'):
        """Cumulative number of light particles produced up to L, separated
        by channel.

        Tracks, for each light species [He4, He3, H3, H2, p, n], how many
        particles were ever created up to distance L (a produced particle
        stays counted even if it is later destroyed or converted). Three
        production channels are tallied separately:

        - 'emission': particles emitted by photodisintegration of the
          heavy cascade.
        - 'conversion': particles created from other light particles
          (n to p decay, H3 to He3, photodisintegration of the bound
          secondaries).
        - 'leading': the heavy-cascade chain itself arriving at the
          species (the leading remnant becoming He4/He3/H2 or a free
          nucleon).
        - 'total': sum of the three.

        All channels are exact matrix-exponential solutions of the same
        augmented ODE solved by `light_cascade_production`: the standing
        light populations evolve with the full light-block matrix, while
        accumulator columns integrate each production inflow without
        outflow. The standing populations themselves are returned by
        `light_cascade_production`; the instantaneous heavy-cascade
        production rate by `light_secondaries_production`.

        Parameters
        ----------
        L : float or ndarray
            Distance(s) [Mpc] at which the tallies are evaluated.
        alpha : ndarray
            Injection vector (must sum to one).
        mass_range : list of int
            Species indices to include. Must be provided together with
            `true_range`.
        boost_range : ndarray, optional
            Boost values to evaluate. Full grid by default.
        true_range : list of int
            Subset of `mass_range` that are not absorption states.
        channel : {'emission', 'conversion', 'leading', 'total'}, optional
            Default 'total'.

        Returns
        -------
        ndarray
            Shape `(6, n_boosts, n_L)` for array `L`, or `(6, n_boosts)`
            for scalar `L`.

        Notes
        -----
        Memory: the largest of this file's augmented systems, solved via
        `_expm_per_boost` on an `(n_true+25)x(n_true+25)` block (heavy
        plus 6 light plus 3x6 accumulator tallies plus absorbed), with
        the full stack held for array `L`. Hits the same large-`n_true`
        memory wall `_expm_per_boost`'s docstring describes, at a smaller
        `n_true` than any other method here (its block is the biggest of
        the four sharing that helper).
        """
        if channel not in ('emission', 'conversion', 'leading', 'total'):
            raise ValueError(f"channel must be 'emission', 'conversion', 'leading' "
                             f"or 'total', got {channel!r}")
        if boost_range is None:
            boost_range = self.boosts

        prod_mat = self.interpyields(boost_range)          # (6, n_sp_full, n_sp_full, n_b)
        if mass_range is not None:
            prod_mat = prod_mat[np.ix_(range(prod_mat.shape[0]), mass_range, mass_range, range(len(boost_range)))]

        # mass_range-shaped, diagonal-fixed, but not yet true_range-restricted --
        # the lead_block computation below needs it in this intermediate form.
        reduced_tensor = self._diagonal_fixed_tensor(boost_range, mass_range)

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

        expmatL = self._expm_per_boost(big_mat, L)
        if type(L) is np.ndarray:
            result  = np.matmul(alpha_aug, expmatL)          # (n_b, n_L, sz)
        else:
            result  = np.matmul(alpha_aug, expmatL)          # (n_b, sz)

        if channel == 'total':
            tally = (result[..., i_e:i_e+6] + result[..., i_c:i_c+6]
                     + result[..., i_a:i_a+6])
        else:
            start = {'emission': i_e, 'conversion': i_c, 'leading': i_a}[channel]
            tally = result[..., start:start+6]

        return np.moveaxis(tally, -1, 0)                     # (6, n_b, n_L) or (6, n_b)

    def cdf_boost_range(self, L, alpha=None, mass_range=None, boost_range=None, true_range=None):
        """The cumulative distance-until-absorption distribution: the
        probability that an injected nucleus has already interacted or
        been absorbed by distance L, for a range of boosts.

        Parameters
        ----------
        L : float or ndarray
            Distance(s) [Mpc] at which the cdf will be evaluated. In
            practice always called with an array; see the Notes below.
        alpha : ndarray
            Injection vector (sum of entries must equal one).
        mass_range : list of int, optional
            Species to be included in the matrix. If None, all species
            are included.
        boost_range : ndarray, optional
            The boost grid to evaluate on. The full grid by default
            (None).

        Returns
        -------
        boosts : ndarray
            `self.boosts`.
        total : ndarray
            The cdf value at each requested boost and distance.

        Notes
        -----
        Memory: solved via `_expm_per_boost` on the bare `(n_true)x(n_true)`
        tensor (no absorption augmentation, the smallest of this file's
        systems), with the full stack held for array `L`. See
        `_expm_per_boost`'s own docstring.

        Known, pre-existing bug, not fixed here: the scalar-`L` branch
        (the `else` of `if alpha.shape == ones.shape`) reduces `expmatL`
        to a 2-D result, but the einsum pattern used is the one for array
        `L`'s 3-D result, so it raises a shape `ValueError` for a 1-D
        `alpha` and scalar `L`. Confirmed via git history that this
        einsum branching itself predates and was never touched by any
        refactor; never exercised by any notebook or test in this repo,
        which only ever call this with array `L`.
        """

        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor, indices = self._restricted_generator(boost_range, mass_range, true_range)

        ones = np.ones_like(-np.moveaxis(reduced_tensor, -1, 0).dot(np.ones_like(alpha[indices])))

        expmatL = self._expm_per_boost(reduced_tensor, L)

        if alpha.shape == ones.shape:
            total = 1 - np.matmul(np.matmul(alpha[indices], expmatL), ones)
        else:
            total = 1 - np.einsum('ijk,ik->ij', np.matmul(alpha[indices], expmatL), ones)

        return self.boosts, total

    def pdf_boost_range(self, L, alpha=None, mass_range=None, omega=None, boost_range=None, true_range=None):
        """The distance-until-absorption probability density at positions
        L for a range of boosts: the complement of `cdf_boost_range`.

        Parameters
        ----------
        L : float or ndarray
            Distance(s) [Mpc] at which the pdf will be evaluated. In
            practice always called with an array; see the Notes below.
        alpha : ndarray
            Injection vector (sum of entries must equal one).
        mass_range : list of int, optional
            Species to be included in the matrix. If None, all species
            are included.
        omega : ndarray, optional
            Ending or production vector. By default set to `omega = -T e`
            (the row-sum of the restricted tensor).
        true_range : list of int
            Species not part of the absorption range (excludes indices
            for species that are part of the absorption state). If none
            is given, the last species in `mass_range` is considered the
            absorption state.
        boost_range : ndarray, optional
            The boost grid to evaluate on. The full grid by default
            (None).

        Returns
        -------
        boost_range : ndarray
            The boost grid actually used.
        total : ndarray
            The pdf value at each requested boost and distance.

        Notes
        -----
        Memory: same profile as `cdf_boost_range` (bare `(n_true)x(n_true)`
        tensor, full per-boost stack for array `L`), see
        `_expm_per_boost`. Shares the same pre-existing, unfixed
        scalar-`L`/1-D-`alpha` shape bug described there too.
        """

        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor, indices = self._restricted_generator(
            boost_range, mass_range, true_range, guard_single=True)

        if omega is None:
            omega = - np.moveaxis(reduced_tensor, -1, 0).dot(np.ones_like(alpha[indices]))

        expmatL = self._expm_per_boost(reduced_tensor, L)

        if alpha.shape == omega.shape:
            total = np.matmul(np.matmul(alpha[indices], expmatL), omega)
        else:
            total = np.einsum('ijk,ik->ij', np.matmul(alpha[indices], expmatL), omega)

        return boost_range, total

    def pdf_moments_boost_range(self, alpha=None, mass_range=None, boost_range=None, true_range=None, degree=1):
        """The moments of the distance-until-absorption distribution
        (`pdf_boost_range`) for a range of boosts.

        Parameters
        ----------
        alpha : ndarray
            Injection vector (sum of entries must equal one).
        mass_range : list of int, optional
            Species to be included in the matrix. If None, all species
            are included.
        true_range : list of int
            Species not part of the absorption range (excludes indices
            for species that are part of the absorption state). If none
            is given, the last species in `mass_range` is considered the
            absorption state.
        boost_range : ndarray, optional
            The boost grid to evaluate on. The full grid by default
            (None).
        degree : int, optional
            The order n of the moment, as in `mu_n = E[X^n]`. Default 1
            (the mean).

        Returns
        -------
        ndarray
            The n-th moment at each boost.
        """

        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor = self._tensor_at(boost_range, mass_range)

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
        """The variance of the distance-until-absorption distribution
        (`pdf_boost_range`) for a range of boosts.

        Parameters
        ----------
        alpha : ndarray
            Injection vector (sum of entries must equal one).
        mass_range : list of int, optional
            Species to be included in the matrix. If None, all species
            are included.
        boost_range : ndarray, optional
            The boost grid to evaluate on. The full grid by default
            (None).
        true_range : list of int
            Species not part of the absorption range (excludes indices
            for species that are part of the absorption state). If none
            is given, the last species in `mass_range` is considered the
            absorption state.

        Returns
        -------
        ndarray
            The variance at each boost.
        """

        if boost_range is None:
            boost_range = self.boosts

        reduced_tensor = self._tensor_at(boost_range, mass_range)

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

        # channel products that are stable (no decay data) but untracked and
        # without a same-A tracked stand-in become INERT tracked species —
        # they accumulate without interacting (e.g. Be-9 remnants when the
        # table carries no Be-9 mother data). Dropping them would leak the
        # channel's mass from the conservation accounting.
        known_dec = self.decays if self.decays else {
            804: None, 502: None, 503: None, 202: None, 200: None}
        valley_A = {5, 6, 7, 8}
        light6 = {(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)}
        inert = set()
        for rows in self.all_branchings:
            for row in np.atleast_2d(rows):
                if not len(row):
                    continue
                Zp, Ap = int(row[0]), int(row[1])
                if Ap <= 1 or (Zp, Ap) in light6 \
                        or (Zp, Ap) in self.species or (Zp, Ap) in inert:
                    continue
                if (Ap * 100 + Zp) in known_dec:
                    continue
                if any(s[1] == Ap for s in self.species):
                    continue
                if Ap in valley_A:
                    continue
                inert.add((Zp, Ap))
        if inert:
            logger.debug('inert species added for untracked stable '
                         'products: %s', sorted(inert))
            self.species += sorted(inert, key=ZA_ordinal, reverse=True)
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

    def save(self, path):
        """Saves the data to a .npz file.

        Saves all arrays needed to reconstruct an instance with `load()`.
        Construction-time secondaries machinery is NOT persisted (same
        policy as the photomeson kernels): `photomeson_kernels`,
        `pion_prod_tensor`, `proton_recoil_tensor`, `photomeson_rates_pn`,
        and the `photomeson_scaling='inclusive'` factor tables. Rebuild
        from the `xsec_model` to use the production folds.

        Parameters
        ----------
        path : str or path-like
            Destination file (`.npz` appended if absent).
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

        ej = getattr(self, 'photomeson_ejecta', None)
        if ej is not None:
            data['pm_ejecta_p'] = ej['p']
            data['pm_ejecta_n'] = ej['n']

        np.savez(path, **data)

    def load(self, path):
        """Populate an instance from a file saved with `save()`.

        Replaces all computed attributes in place, bypassing
        `_construct_from_files()` and `_generate_complete_matrices()`.
        The existing object reference remains valid after the call.

        Parameters
        ----------
        path : str or path-like
            Source file (`.npz` appended if absent).
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

        if 'pm_ejecta_p' in d:
            self.photomeson_ejecta = {'p': d['pm_ejecta_p'], 'n': d['pm_ejecta_n']}

    def get_distribution_parameters(self, mass_lims=(56, 11), injection_type=('only species', (26, 56)), absorption_type=('only mass', [54]), boost_range=None):
        """Builds the injection vector `alpha`, `mass_range`, and
        `true_range` arguments that `species_evolution_boost_range` and
        the other propagation methods need, for a cascade starting from
        a chosen mass range and absorbing below a chosen threshold.

        Parameters
        ----------
        mass_lims : tuple of int, optional
            `(Amax, Amin)`, the starting mass and the lower limit for
            mass. Species with `Amin < A <= Amax` are included in
            `mass_range`. Default `(56, 11)`.
        injection_type : tuple, optional
            `(type, params)` specifying the injection. `type` is one of:

            - `'flat'`: equal injection of every species in `mass_lims`.
            - `'only mass'`: equal injection of the species whose mass
              is in the list `params`.
            - `'only species'`: injection concentrated on the single
              species `(Z, A) = params`.

            Default `('only species', (26, 56))` (Fe-56 only).
        absorption_type : tuple, optional
            `(type, params)` specifying the absorbing state, the species
            excluded from `true_range`. `type` is one of `'only mass'`,
            `'only species'` (as above), or `'only charge'` (species
            whose charge is in the list `params`). Default
            `('only mass', [54])`.
        boost_range : ndarray, optional
            The boost grid to build the reduced tensor on. The full grid
            by default (None).

        Returns
        -------
        alpha : ndarray
            The injection vector, normalized to sum to 1, indexed like
            `mass_range`.
        mass_range : list of int
            Indices into `self.species` for every species between
            `mass_lims`.
        true_range : list of int
            The subset of `mass_range` not excluded by `absorption_type`
            (the non-absorbed species).
        reduced_tensor : ndarray
            The interaction tensor restricted to `true_range`, on
            `boost_range`.

        Examples
        --------
        >>> from crisp.core import InteractionCore
        >>> from crisp.photonuclear_cross_sections import PSB_model
        >>> core = InteractionCore(xsec_model=PSB_model())
        >>> alpha, mass_range, true_range, reduced_tensor = core.get_distribution_parameters(
        ...     mass_lims=(56, 0), injection_type=('only species', (26, 56)),
        ...     absorption_type=('only mass', [1]))
        >>> alpha.sum()
        np.float64(1.0)
        >>> len(mass_range), len(true_range)
        (53, 51)
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

        reduced_tensor = self._tensor_at(boost_range, mass_range)

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

        builtin = {
            804: {'decay_time': 1.18e-16, 'channels': [[1.0, 402]]},   # Be8 -> He4 + He4
            502: {'decay_time': 1.01e-21, 'channels': [[1.0, 100]]},   # He5 -> He4 + n
            503: {'decay_time': 5.34e-22, 'channels': [[1.0, 101]]},   # Li5 -> He4 + p
            202: {'decay_time': 1e-22,    'channels': [[1.0, 101]]},   # 2p  -> p + p
            200: {'decay_time': 1e-22,    'channels': [[1.0, 100]]},   # 2n  -> n + n
        }
        # the builtin particle-emitters stay available as a FALLBACK even
        # with a decay table provided: nubase carries no entries for these
        # unbound nuclides, and losing them dead-ends whole decay chains
        decays = {**builtin, **self.decays} if self.decays else builtin

        def resolve(product, frac=1.0, depth=0):
            """List of (fraction, final remnant (Z, A), {light index: count}), or None."""
            Z, A = int(product[0]), int(product[1])
            if (Z, A) in self.species:
                return [(frac, (Z, A), {})]
            nucid = A * 100 + Z
            if depth > 10:
                return None
            if nucid not in decays:
                # fallback ladder for products without decay data, so channel
                # strength is never silently dropped: a same-A tracked species
                # stands in for the beta descendant (charge shift only, like
                # the beta branch below); particle-unstable masses outside the
                # table map onto the valley (Stecker-Salamon prescription)
                same_A = [s for s in self.species if s[1] == A and s != (Z, A)]
                if same_A:
                    Zn = min(same_A, key=lambda s: abs(s[0] - Z))
                    return resolve(Zn, frac, depth + 1)
                valley = {5: (2, 5), 6: (2, 4), 7: (2, 4), 8: (4, 8)}
                if A in valley and valley[A] != (Z, A):
                    Zv, Av = valley[A]
                    # emit the leftover nucleons of the A -> valley step so
                    # the redirect conserves mass (charge best-effort)
                    n_p = int(np.clip(Z - Zv, 0, A - Av))
                    n_n = (A - Av) - n_p
                    counts0 = {}
                    if n_p:
                        counts0[4] = n_p
                    if n_n:
                        counts0[5] = n_n
                    sub = resolve(valley[A], frac, depth + 1)
                    if sub is None:
                        return None
                    out = []
                    for sub_frac, sub_rem, sub_counts in sub:
                        merged = dict(counts0)
                        for li, cnt in sub_counts.items():
                            merged[li] = merged.get(li, 0) + cnt
                        out.append((sub_frac, sub_rem, merged))
                    return out
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
            if new_rows:
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


class InteractionCore_Source(InteractionCore):
    """Deprecated wrapper. Construct the base class directly instead:

        InteractionCore(xsec_model=..., target_photons=...,
                        photomeson='kernels', boosts=..., eps=...)

    Preserves the historical constructor signature and grids
    (boosts logspace(0, 12, 131), eps logspace(-2, 6, 300) GeV).
    """

    def __init__(self, epsrange, target_photon_spectrum, path=None, xsec_model=None,
                 nuclear_decay_On=False, decays=None):
        import warnings
        warnings.warn("InteractionCore_Source is deprecated; construct "
                      "InteractionCore(xsec_model=..., target_photons=..., "
                      "photomeson='kernels', ...) instead.",
                      DeprecationWarning, stacklevel=2)

        if path is not None:
            from .photonuclear_cross_sections import CRPropa_model
            self.path = path
            xsec_model = CRPropa_model(path=path)
        elif xsec_model is None:
            raise ValueError('Error: no cross sections provided.')

        self.epsrange = epsrange   # kept for backward compatibility (unused)

        InteractionCore.__init__(self, nuclear_decay_On=nuclear_decay_On, decays=decays,
                                 xsec_model=xsec_model,
                                 target_photons=target_photon_spectrum,
                                 photomeson='kernels',
                                 boosts=np.logspace(0, 12, 131),
                                 eps=np.logspace(-2, 6, 300))


class InteractionCore_PSB_CMB(InteractionCore):
    """Deprecated wrapper. Construct the base class directly instead:

        InteractionCore(xsec_model=PSB_model(), ...)

    Results differ slightly from the historical class: the unified path uses
    the boost-native rate convention (the old constructor shifted the boost
    grid by 1/0.939), the default 0.1-126 MeV photon-energy grid (the old one
    stopped at 50 MeV), and nuclear decays for remnants on the unstable masses
    (Be8 -> 2 He4), which gives PSB a secondary-He4 emission channel.
    """

    def __init__(self, nuclear_decay_On=False, ftype=np.float64, decays=None):
        import warnings
        warnings.warn("InteractionCore_PSB_CMB is deprecated; construct "
                      "InteractionCore(xsec_model=PSB_model(), ...) instead.",
                      DeprecationWarning, stacklevel=2)
        from .photonuclear_cross_sections import PSB_model
        InteractionCore.__init__(self, nuclear_decay_On=nuclear_decay_On, ftype=ftype,
                                 decays=decays, xsec_model=PSB_model())


class InteractionCore_SimProp_CMB(InteractionCore):
    """Deprecated wrapper. Construct the base class directly instead:

        InteractionCore(xsec_model=SimProp_model(M=...), ...)

    See InteractionCore_PSB_CMB for the (small) differences with respect to
    the historical class.
    """

    def __init__(self, M=1, nuclear_decay_On=False, decays=None, ftype=np.float64):
        import warnings
        warnings.warn("InteractionCore_SimProp_CMB is deprecated; construct "
                      "InteractionCore(xsec_model=SimProp_model(M=...), ...) instead.",
                      DeprecationWarning, stacklevel=2)
        from .photonuclear_cross_sections import SimProp_model
        self.M = M
        InteractionCore.__init__(self, nuclear_decay_On=nuclear_decay_On, ftype=ftype,
                                 decays=decays, xsec_model=SimProp_model(M=M))


class InteractionCore_CRPdata_CMB(InteractionCore):
    """Deprecated wrapper. Construct the base class directly instead:

        InteractionCore(xsec_model=...)          # the CMB is the default photon field
    """

    def __init__(self, path=None, nuclear_decay_On=False, xsec_model=None, decays=None):
        import warnings
        warnings.warn("InteractionCore_CRPdata_CMB is deprecated; construct "
                      "InteractionCore(xsec_model=...) instead.",
                      DeprecationWarning, stacklevel=2)

        if path is not None:
            from .photonuclear_cross_sections import CRPropa_model
            self.path = path
            xsec_model = CRPropa_model(path=path)
        elif xsec_model is None:
            raise ValueError('Error: no cross sections provided.')

        InteractionCore.__init__(self, nuclear_decay_On=nuclear_decay_On, decays=decays,
                                 xsec_model=xsec_model)


class InteractionCore_CRPdata_EBL(InteractionCore):
    """Deprecated wrapper. Construct the base class directly instead:

        InteractionCore(xsec_model=..., target_photons=<EBL field>)
    """

    def __init__(self, path=None, nuclear_decay_On=False, xsec_model=None, z=0, decays=None):
        import warnings
        warnings.warn("InteractionCore_CRPdata_EBL is deprecated; construct "
                      "InteractionCore(xsec_model=..., target_photons=<EBL field>) "
                      "instead.", DeprecationWarning, stacklevel=2)

        if path is not None:
            from .photonuclear_cross_sections import CRPropa_model
            self.path = path
            xsec_model = CRPropa_model(path=path)
        elif xsec_model is None:
            raise ValueError('Error: no cross sections provided.')

        from .background_photon_models import eblg_interp
        self.z = z
        ebl_field = lambda energ: eblg_interp(energ * 1e9, z).flatten() * 1e3

        InteractionCore.__init__(self, nuclear_decay_On=nuclear_decay_On, decays=decays,
                                 xsec_model=xsec_model, target_photons=ebl_field)


