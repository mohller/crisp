"""Production and testing the interaction matrices
"""

import os
import numpy as np
from scipy.linalg import expm
from scipy.interpolate import interp1d
from UHECR_statistics import prepare_species_list

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
        elif branchings is 'minimal': # case for only one nucleon loss
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
                        print()
                        Zrem, Arem = 2, 4
                    else:
                        print(f'No suitable isomer found for remnant ({Zrem:2d}, {Arem:2d})')
                
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
    df_rates = read_csv(filename, header=1, sep='\t', names=['Z', 'N'] + cols)

    df_rates.insert(1, 'A', df_rates['Z'] + df_rates['N'])
    df_rates.drop('N', axis=1, inplace=True)

    df_rates.index = MultiIndex.from_arrays(df_rates[['A', 'Z']].values.T)
    df_rates.sort_index(inplace=True)

    return df_rates

def load_branchings(filename):
    from pandas import read_csv, MultiIndex
    cols = [f'{i}' for i in range(201)]
    df_brnch = read_csv(filename, header=1, sep='\t', names=['Z', 'N', 'channel'] + cols)
    
    # Nuclei in file which have no decay implemented
    correction_channels = [
        [(2, 5), (2, 4), [0, 0, 0, 0, 0, 1]],
        [(2, 6), (2, 4), [0, 0, 0, 0, 0, 2]],
        [(3, 5), (2, 4), [0, 0, 0, 0, 1, 0]],
        [(4, 8), (2, 4), [1, 0, 0, 0, 0, 0]],
        [(5, 9), (4, 9), [0, 0, 0, 0, 1, -1]],
        [(5, 12), (6, 12), [0, 0, 0, 0, -1, 1]],
        [(9, 16), (8, 16), [0, 0, 0, 0, 1, -1]],
        [(11, 20), (10, 20), [0, 0, 0, 0, 1, -1]],
        [(13, 31), (13, 30), [0, 0, 0, 0, 0, 1]],
        [(20, 39), (19, 39), [0, 0, 0, 0, 1, -1]],
        [(21, 42), (20, 42), [0, 0, 0, 0, 1, -1]],
        [(24, 47), (23, 47), [0, 0, 0, 0, 1, -1]],
    ]
    daughter_names = ['a', 'he3', 't', 'd', 'p', 'n']
    daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
    Ad = np.array([d[1] for d in daughters])
    Zd = np.array([d[0] for d in daughters])

    df_brnch.insert(1, 'A', df_brnch['Z'] + df_brnch['N'])
    df_brnch.drop('N', axis=1, inplace=True)

    df_brnch.loc[:, 'channel'] = df_brnch.channel.apply(get_particle_numbers)
    Zr = df_brnch['channel'].apply(Zd.dot)
    Ar = df_brnch['channel'].apply(Ad.dot)

    df_brnch.insert(2, 'Ar', df_brnch['A'] - Ar)
    df_brnch.insert(2, 'Zr', df_brnch['Z'] - Zr)

    df_brnch.index = MultiIndex.from_arrays(df_brnch[['A', 'Z', 'Ar', 'Zr']].values.T)

    # Replacing channels with dead ends
    for nuc0, nucr, prods in correction_channels:
        new_prods = np.vstack(df_brnch[np.all(df_brnch[['Zr', 'Ar']] == nuc0, axis=1)]['channel']) + np.array(prods)
        df_brnch.loc[np.all(df_brnch[['Zr', 'Ar']] == nuc0, axis=1), ['channel']] = [[list(row)] for row in new_prods]
        df_brnch.loc[np.all(df_brnch[['Zr', 'Ar']] == nuc0, axis=1), ['Zr', 'Ar']] = nucr

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
    import interaction_rates as ir
    from astropy.constants import c
    c_in_Mpc_sec = c.to('Mpc/s').value
    mb_to_cm2 = 1e-27
    
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

        cs_crpropa = br_row[3:]
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
    import interaction_rates as ir
    from astropy.constants import c
    c_in_Mpc_sec = c.to('Mpc/s').value
    mb_to_cm2 = 1e-27
    
    # He4, He3, H3, H2, p, n
    daughter_names = ['a', 'he3', 't', 'd', 'p', 'n']
    daughters = [(2, 4), (2, 3), (1, 3), (1, 2), (1, 1), (0, 1)]
    Zd = np.array([d[0] for d in daughters])
    Ad = np.array([d[1] for d in daughters])

    if boosts is None:
        boosts = np.logspace(5, 14, nboosts)
    
    cols = [f'{i}' for i in range(len(boosts))]
    
    # Computing individual rates for proton and neutron
    pr_pmes = ir.interaction_rate_from_cross_section(boosts, 1, target_photons, 
                xsp[:, 0], xsp[:, 0]*mb_to_cm2*1e-3)  / c_in_Mpc_sec # 1 / Mpc
    nr_pmes = ir.interaction_rate_from_cross_section(boosts, 1, target_photons, 
                xsn[:, 0], xsn[:, 0]*mb_to_cm2*1e-3)  / c_in_Mpc_sec # 1 / Mpc

    pprates = np.zeros((len(nuclei), len(boosts)))
    for k, (Z, A) in enumerate(nuclei):
        pprates[k] = np.interp(boosts, boosts/A, Z * pr_pmes + (A-Z) * nr_pmes)
    print(np.hstack([np.vstack(nuclei), pprates]).shape)
    df_rates_pmes = DataFrame(data=np.hstack([np.vstack(nuclei), pprates]), index=MultiIndex.from_arrays(np.vstack(nuclei).T), columns=['Z', 'A'] + cols)

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

class InteractionCore():
    """Base class to produce interaction matrices
    """

    def __init__(self):
        self._construct_from_files()
        self._genenerate_complete_matrices()

    def _construct_from_files():
        """Function to load the interaction data from given files
        and produce the core matrices.
        """
        pass

    def species_evolution_boost_range(self, L, alpha=None, mass_range=None, boost_range=None):
        """Returns the probabilities of each species at positions L for a range of boosts.
        If the distances are negative and in decreasing order, it's equivalent to back propagation. 

        Arguments:
        ----------
        L : a float or an array of distances at which the pdf will be evaluated
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        """

        if boost_range is None:
            boost_range = self.boosts

        if mass_range is None:
            reduced_tensor = self.interpolator(boost_range)
        else:
            reduced_tensor = self.interpolator(boost_range)
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]

        if type(L) is np.ndarray:
            expmatL = expm(np.moveaxis(L[:, None, None, None] * reduced_tensor, -1, 0))
        else:
            expmatL = expm(np.moveaxis(reduced_tensor * L, -1, 0))

        total = np.matmul(alpha, expmatL)

        return total
    
    def light_secondaries_production(self, L, alpha=None, mass_range=None, boost_range=None):
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
        
        P = self.species_evolution_boost_range(L, alpha, mass_range, boost_range)

        LamYp = prod_mat * reduced_tensor # production rate matrix, independent of distance

        production = np.sum(np.einsum('lmi, kijl -> klmj', P, LamYp), axis=3)

        return production

    def cdf_boost_range(self, L, alpha=None, mass_range=None, boost_range=None):
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

        if mass_range is None:
            reduced_tensor = self.interpolator(boost_range)
        else:
            reduced_tensor = self.interpolator(boost_range)
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]
        
        ones = np.ones_like(- np.moveaxis(reduced_tensor, -1, 0).dot(np.ones_like(alpha)))

        if type(L) is np.ndarray:
            expmatL = expm(np.moveaxis(L[:, None, None, None] * reduced_tensor, -1, 0))
        else:
            expmatL = expm(np.moveaxis(reduced_tensor * L, -1, 0))

        if alpha.shape == ones.shape:
            total = 1 - np.matmul(np.matmul(alpha, expmatL), ones)
        else:
            total = 1 - np.einsum('ijk,ik->ij', np.matmul(alpha, expmatL), ones)

        return self.boosts, total

    def pdf_boost_range(self, L, alpha=None, mass_range=None, omega=None, boost_range=None, true_range=None):
        """Returns the probability density value at positions L for a range of boosts

        Arguments:
        ----------
        L : a float or an array of distances at which the pdf will be evaluated
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        omega : ending or production vector. By default is set to omega=-Te
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
        reduced_tensor = reduced_tensor[np.ix_(true_range, true_range, range(len(boost_range)))]

        if omega is None:
            omega = - np.moveaxis(reduced_tensor, -1, 0).dot(np.ones_like(alpha[true_range]))

        if type(L) is np.ndarray:
            expmatL = expm(np.moveaxis(L[:, None, None, None] * reduced_tensor, -1, 0))
        else:
            expmatL = expm(np.moveaxis(reduced_tensor * L, -1, 0))

        if alpha.shape == omega.shape:
            total = np.matmul(np.matmul(alpha, expmatL), omega)
        else:
            total = np.einsum('ijk,ik->ij', np.matmul(alpha[true_range], expmatL), omega)

        return boost_range, total

    def pdf_moments_boost_range(self, alpha=None, mass_range=None, boost_range=None, degree=1):
        """Returns the moments for a range of boosts

        Arguments:
        ----------
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        """

        if boost_range is None:
            boost_range = self.boosts       

        reduced_tensor = self.interpolator(boost_range)
        
        if mass_range is not None:
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]

        inverse = np.linalg.inv(np.moveaxis(reduced_tensor, -1, 0))
        inverse_power = inverse**degree

        moment = np.math.factorial(degree) * (-1)**degree * np.matmul(np.matmul(alpha, inverse_power), np.ones_like(alpha))

        return moment
    
    def pdf_variance_boost_range(self, alpha=None, mass_range=None, boost_range=None):
        """Returns the variance for a range of boosts

        Arguments:
        ----------
        alpha : injection vector (sum of entries must equal one).
        mass_range : species to be included in the matrix. If None, all species are included.
        boost_range : A two element variable with the limits minimum and maximum. The whole range by default (None). 
        """

        if boost_range is None:
            boost_range = self.boosts       

        if mass_range is None:
            reduced_tensor = self.interpolator(boost_range)
        else:
            reduced_tensor = self.interpolator(boost_range)
            reduced_tensor = reduced_tensor[np.ix_(mass_range, mass_range, range(len(boost_range)))]

        inverse = np.linalg.inv(np.moveaxis(reduced_tensor, -1, 0))

        momentum1 = -np.matmul(np.matmul(alpha, inverse), np.ones_like(alpha))
        momentum2 = np.matmul(np.matmul(alpha, inverse**2), np.ones_like(alpha))
        
        return 2*momentum2 - momentum1**2 

    def _genenerate_complete_matrices(self):
        """Generates, for each boost, a complete interaction matrix from the interaction tables
        """
        Zmax, Amax = self.nuclei[-1]  # largest species

        species = prepare_species_list(self.nuclei, Zmax, Amax, Amax-1)
        self.species = species

        # generate interaction tensor by slices
        tensor = np.zeros((len(species)+2, len(species)+2, len(self.boosts)))
        for i, nuc_branches in enumerate(self.all_branchings[::-1]):
            for branch in nuc_branches:
                try:
                    j = (self.species + [(1, 1), (0, 1)]).index(tuple(branch[:2]))
                    # j = (ic1.species).index(tuple(branch[:2]))
                    tensor[i, j, :] = branch[2:]
                except:
                    print('problem with product', branch[:2], 'of nucleus', self.species[i])
                    continue

        tensor -= np.stack([np.diag(row) for row in tensor.sum(axis=1).T], axis=2)

        # generate light production tensors by slices
        ly_all_mats = []
        for light_yield in self.marginal_light_yields:
            ly_matrices = np.zeros((len(species)+2, len(species)+2, len(self.boosts)))
            for i, nuc_branches in enumerate(light_yield[::-1]):
                for branch in nuc_branches:
                    try:
                        j = (self.species + [(1, 1), (0, 1)]).index(tuple(branch[:2]))
                        ly_matrices[i, j, :] = branch[2:]
                    except:
                        print('problem with product', branch[:2], 'of nucleus', self.species[i])
                        continue
            ly_all_mats.append( ly_matrices )

        # check that all rows add up to one!!!
        # np.all(np.isclose(np.einsum('ijk, j -> ik', tensor, np.ones(186)), 0))

        self.tensor = tensor
        self.light_prod_tensor = np.stack(ly_all_mats)
        self.interpolator = interp1d(self.boosts, self.tensor, 'cubic')
        self.interpyields = interp1d(self.boosts, self.light_prod_tensor, 'cubic')


class InteractionCore_CRPropA(InteractionCore):
    """Producing interaction matrices from CRPropA interaction files 
    """

    def __init__(self, data_files=None):

        if data_files is None:
            self.data_files = {
                'path' : os.path.abspath('GitProjects/CRPropa3/data-2021-07-30/data/'),
                
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
        
        pp_rates = np.genfromtxt(os.path.join(self.data_files['path'], self.data_files['photopionproduction']['rates_cmb']))
        pprates = np.zeros((len(nuclei), 201))
        boosts = np.logspace(6, 14, 201)
        cols = [f'{i}' for i in range(201)]
        daughter_names = ['a', 'he3', 't', 'd', 'p', 'n']

        for k in range(pprates.shape[0]):
            Z, A = nuclei[k]
            pprates[k] = np.interp(boosts, 1./A * 10**pp_rates[:, 0], Z*pp_rates[:, 1] + (A-Z)*pp_rates[:, 2])

        df_rates_pmes = DataFrame(data=np.hstack([nuclei, pprates]), index=MultiIndex.from_arrays(np.array(nuclei).T), columns=['Z', 'A'] + cols)

        pmes_branchings = []
        pmes_marginal_yields = []
        for idx, (Z, A) in enumerate(nuclei):
            remnants = [(Z, A-1), (Z-1, A-1)]
            for br, (Zrem, Arem) in zip([(1-Z/A), Z/A], remnants):
                if (Zrem, Arem) in nuclei:
                    pmes_branchings.append(np.hstack([A, Z, Arem, Zrem, br * pprates[idx]]))
                    pmes_marginal_yields.append(np.hstack([A, Z, Arem, Zrem, 0, 0, 0, 0, Z-Zrem, A-Arem-Z+Zrem, br * np.ones(201)]))

            if not np.all([rem in nuclei for rem in remnants]):
                if np.any([rem in nuclei for rem in remnants]):
                    pmes_branchings[-1][4:] = pprates[idx]
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
        import interaction_rates as ir
        from astropy.constants import c
        c_in_Mpc_sec = c.to('Mpc/s').value
        mb_to_cm2 = 1e-27

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
        import interaction_rates as ir
        from astropy.constants import c
        c_in_Mpc_sec = c.to('Mpc/s').value

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
    
    def __init__(self, data_directory, target_photon_spectrum):
        """ Requires a string specifying the directory where CRPropa 
            cross section files are stored (argument data_directory) 
        """
        
        self._construct_from_files(data_directory, target_photon_spectrum)
        self._genenerate_complete_matrices()

    def _construct_from_files(self, data_directory, target_photons):
        """Using CRPROPA cross sections to produce the rates for a source
        of UHECR with a background photon field as a broken power law.

        CRPropA cross section file contains  is structured in different files depending on the 
        interaction and the photon field.
        """
        from pandas import DataFrame
        boosts = np.logspace(3, 12, 41)
        cols = [f'{i}' for i in range(len(boosts))]

        eps_crpropa = np.genfromtxt(data_directory + 'PD_Talys1.8_Khan/eps.txt') / 1e3 # in GeV
        branchings = np.genfromtxt(data_directory + 'PD_Talys1.8_Khan/xs_pd.txt')
        df_rates_pdis, df_brnch_pdis, merged_yields_pdis = \
            generate_photodisinteg_tables_from_cross_sections(eps_crpropa, branchings, target_photons, boosts=boosts)
        
        nuclei = [(int(Z), int(A)) for A, Z in df_rates_pdis.index.values]

        xsp = np.genfromtxt(data_directory + 'PPP/xs_proton.txt')
        xsn = np.genfromtxt(data_directory + 'PPP/xs_neutron.txt')
        df_rates_pmes, df_brnch_pmes, merged_yields_pmes = \
            generate_photomeson_tables_from_cross_sections(nuclei, xsp, xsn, target_photons, boosts=boosts)

        df_rates = df_rates_pdis.groupby(by=['A', 'Z']).sum() + df_rates_pmes.groupby(by=['A', 'Z']).sum()

        df_brnch_pdis[cols] = df_brnch_pdis.multiply(df_rates_pdis.reindex(df_brnch_pdis.index, method='ffill'))[cols]
        merged_pdis = df_brnch_pdis.groupby(by=['Z', 'A', 'Zr', 'Ar']).sum()
        allmr_pdis = [np.hstack([np.vstack(merged_pdis.loc[nuc].index.values), merged_pdis.loc[nuc][cols].values]) for nuc in nuclei]

        df_brnch_pmes[cols] = df_brnch_pmes.multiply(df_rates_pmes.reindex(df_brnch_pmes.index, method='ffill'))[cols]
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


class InteractionCore_PSB_CMB(InteractionCore):
    def _construct_from_files(self):
        """Based on PSB-model of nuclear cascades
        """
        from scipy.constants import c, parsec
        from interaction_rates import interaction_rate_from_cross_section
        from background_photon_models import cmb_photon_density_GeVcm3
        from photonuclear_cross_sections import PSB_model

        boosts = np.logspace(6, 14, 201)
        eps = 1e-3 * np.linspace(5, 50, 200) # in GeV

        psb_model = PSB_model()
        reversed_by_mass = psb_model.params.sort_values(by=['A', 'Z'], ascending=True)

        nuclei, pdis_rates_cmb, branchings_cmb = [], [], []
        for Z, A in zip(reversed_by_mass['Z'], reversed_by_mass['A']):
            nuclei.append((int(Z), int(A)))

            branchings = []
            for nloss in range(1, 16): # only up to 15 possible
                Arem = int(A - nloss)

                if Arem < 1:
                    continue
                elif Arem == 1:
                    Zrem = 1
                elif Arem in [5, 6, 7, 8]:
                    Arem, Zrem = 4, 2
                else:
                    Zrem = int(psb_model.params[psb_model.params['A'] == Arem]['Z'])

                cross_section = 1e-27 * psb_model.cross_section(eps * 1e3, Z, A, nloss) # to cm2
                pdis_rates = interaction_rate_from_cross_section(A*boosts, A, cmb_photon_density_GeVcm3, eps, cross_section)
                pdis_rates /= c / parsec / 1e6 # ito Mpc

                branchings.append(np.append([Zrem, Arem], pdis_rates))
            
            pdis_rates_cmb.append(np.sum(np.atleast_2d((branchings)), axis=0)[2:])
            branchings_cmb.append(branchings)
            
        self.boosts = boosts 
        self.nuclei = nuclei
        self.all_rates = pdis_rates_cmb
        self.all_branchings = branchings_cmb