"""Canonical serial-cascade benchmarks and general matrix-exponential
disintegration distributions, in the terms of the methods paper (Sect. 2:
regular serial cascades, irregular serial cascades, and the general
concurrent-cascade matrix-exponential form).

A regular serial cascade (RSeC) is the paper's idealized reference case:
each interaction produces exactly one daughter (a serial, not branching,
chain), and the interaction rate scales exactly with mass,
lambda_A = A * lambda_1. Under that assumption the distance-until-
absorption distribution for losing k nucleons has the closed binomial
form of the paper's Eq. 7. `transition_matrix` builds the corresponding
tridiagonal RSeC rate matrix directly (state l has outgoing rate
(A - l) * lam); `transition_matrix_gaussian_variation` perturbs those
rates with a log-Gaussian factor to generate irregular serial cascade
(ISeC) instances, quantifying how much a realistic (non-regular) cascade
deviates from the canonical benchmark.

`pdf`, `cdf`, and `momenta` evaluate any such matrix-exponential
distribution (the paper's general concurrent-cascade form, phi exp(Lambda
L) ...): given a transition matrix and an injection vector, they are the
generic machinery both the canonical RSeC/ISeC matrices above and a full,
realistic disintegration network share. `ME` wraps a (matrix, injection,
ejection) triple as a `scipy.stats.rv_continuous` for convenient
sampling/fitting. `complete_matrix` builds that full, realistic transition
matrix from an actual cross-section-derived rate and branching table
(one-nucleon-loss transitions with the true, non-regular branchings),
and `create_distribution` / `create_distribution_crpropa` wrap it into an
`ME` distribution, i.e. the realistic counterpart to compare against the
canonical RSeC/ISeC benchmark. `reduce_matrix` collapses such a matrix
down to its equivalent mass-only serial chain (Horvath and Telek's
theorem). `prepare_species_list` (also imported directly by `core.py`)
enumerates the species reachable from an injected nucleus by losing up
to `nloss` nucleons, and `get_injection_parameters` builds the matching
injection vector.

`InteractionCore.cdf_boost_range`/`pdf_boost_range` solve the same kind
of distance-until-absorption problem directly from the full interaction
tensor, for arbitrary (non-serial) cascades; this module's closed-form
RSeC/ISeC solutions remain the paper's benchmark for how close a
realistic cross-section model comes to the regular-cascade idealization.
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats import rv_continuous

def recurs_spec(Z, A, nloss=1):
    """Recursive yield of daughters
    """
    yield (Z, A)

    for l in range(1, nloss+1):
        for k in range(l+1):
            if (A-l < 3 * (Z-k)) and (A-l > 1.3 * (Z-k)):
                yield (Z-k, A-l)

def prepare_species_list(nuclei, Zinj=26, Ainj=56, nloss=2, mass_range=False, random_chain=False):
    """Returns the list of species included in all
    possible disintegration chains that begin at species
    (Ainj, Zinj) and end at any species with a number of
    nucleons less given by nloss.

    When mass_range is True, all species with mass from Ainj up to and
    including Ainj-nloss are included, regardless of whether they lie on
    an actual disintegration chain.

    When random_chain is True, a single random decreasing-mass chain is
    returned instead of every reachable species: starting at (Zinj, Ainj),
    each step loses one proton or one neutron (chosen at random, subject
    to landing on a nuclide present in `nuclei`) until `nloss` nucleons
    have been shed.

    Parameters
    ----------
    nuclei : list of tuple of int
        The (Z, A) nuclides available to build chains from (species
        outside this list are never selected).
    Zinj, Ainj : int, optional
        Charge and mass number of the injected nucleus. Default (26, 56).
    nloss : int, optional
        Maximum number of nucleons lost along a chain. Default 2.
    mass_range : bool, optional
        Return the flat mass window described above instead of walking
        the disintegration chains. Default False.
    random_chain : bool, optional
        Return one random chain instead of every reachable species.
        Default False.

    Returns
    -------
    list of tuple of int
        The selected (Z, A) species.
    """
    if random_chain:
        selected = [(Zinj, Ainj)]
        for _ in range(nloss):
            ploss = np.random.randint(0, 2) # 0 or 1
            nuc = (selected[-1][0] - ploss, selected[-1][1] - 1)
            if nuc in nuclei:
                selected.append(nuc)
            else:
                new_nuc = (selected[-1][0] - np.abs(ploss - 1), nuc[1])
                if new_nuc in nuclei:
                    selected.append(new_nuc)
    elif mass_range:
        selected = [nuc for nuc in nuclei if nuc[1] in range(Ainj - nloss, Ainj+1)]
    else:
        selected = [nuc for nuc in recurs_spec(Zinj, Ainj, nloss) if nuc in nuclei]

    return selected

def complete_matrix(nuclei, all_rates, all_branchings, species_list=None, idx=None):
    """Returns the full matrix corresponding to
    the list of species and rates included in crpropa
    
    Only one nucleon losses are included. The decay from
    one nucleus to the next one is the interaction length
    for losing one nucleon, weighted by the relative number
    of protons (neutrons) when losing a proton (neutron).

    idx denotes the index of boost to be used.

    Parameters
    ----------
    nuclei : list of tuple of int
        Full list of (Z, A) nuclides that `all_rates`/`all_branchings`
        are indexed against (CRPropa-style tables).
    all_rates : array_like
        Per-nuclide interaction rate on the boost grid, indexed like
        `nuclei`.
    all_branchings : list
        Per-nuclide list of branchings to daughters (He4, He3, H3, H2, p,
        n, and others), indexed like `nuclei`.
    species_list : list of tuple of int, optional
        The (Z, A) species to include as matrix states; default the
        result of `prepare_species_list()` with its own defaults.
    idx : int, optional
        Index into the boost grid to build the matrix at. Default 0.

    Returns
    -------
    ndarray
        The (len(species_list), len(species_list)) transition matrix.
    """
    if species_list is None:
        species_list = prepare_species_list()

    if idx is None:
        idx = 0 # boost index for which to build matrix

    no_species = len(species_list)
    Tmatrix = np.zeros((no_species, no_species))
    
    # Generate interactions for only one nucleon loss
    for i, nuc2 in enumerate(species_list):
        nucid = np.argwhere([nuc2 == nuc for nuc in nuclei])[0][0]

        Tmatrix[i, i] = - all_rates[nucid][idx]

        nucid = np.argwhere([nuc2 == nuc for nuc in nuclei])[0][0]
        brans = all_branchings[nucid] # branchings to He4, He3, H3, H2, p, n and others
        
        for br in brans:
            Zd, Ad = br[0], br[1] # daughter values

            dauidx = np.argwhere([(spec[0] == Zd) and (spec[1] == Ad) for spec in species_list])

            if dauidx.size > 0:
                Tmatrix[i, dauidx[0][0]] = br[2+idx]
    
        if not np.isclose(sum(Tmatrix[i, :]), 0): # if row is incomplete
            remainder = -Tmatrix[i, :].dot(np.ones(len(species_list)))
            
            for j in range(len(species_list) - i):
                if Tmatrix[i, -j-1] > 0:
                    # Adding remainder of decay rate into the last element
                    Tmatrix[i, -j-1] += remainder
                    break

    return Tmatrix

def create_distribution(nuclei, all_rates, all_branchings, Zinj=None, Ainj=None, nloss=None, idx=None):
    """Return the matrix-exponential distance-until-absorption
    distribution for a nucleus (Zinj, Ainj) losing up to nloss nucleons,
    built from `complete_matrix`.

    Parameters
    ----------
    nuclei : list of tuple of int
        Full list of (Z, A) nuclides that `all_rates`/`all_branchings`
        are indexed against (CRPropa-style tables).
    all_rates : array_like
        Per-nuclide interaction rate on the boost grid, indexed like
        `nuclei`.
    all_branchings : list
        Per-nuclide list of branchings to daughters, indexed like
        `nuclei`; see `complete_matrix`.
    Zinj, Ainj : int
        Charge and mass number of the injected nucleus.
    nloss : int
        Number of nucleons lost that defines absorption (species with
        A = Ainj - nloss are the absorbing states).
    idx : int, optional
        Index into the boost grid to build the matrix at. Default 0.

    Returns
    -------
    thedist : ME
        The matrix-exponential distribution (a `scipy.stats.rv_continuous`).
    species : list of tuple of int
        The (Z, A) species in `thedist.matrix`'s state order.
    """
    species = prepare_species_list(nuclei, Zinj, Ainj, nloss)
    Tmatrix = complete_matrix(nuclei, all_rates, all_branchings, species, idx=idx)

    alpha = np.zeros(len(Tmatrix))
    alpha[0] = 1 # injecting only iron, or the heaviest element
    
    # Add injections to fictitious absorbing state
    eta = np.zeros(len(Tmatrix))
    for i, spec in enumerate(species):
        if spec[1] == Ainj - nloss:
            eta[i] = -Tmatrix[i, i]

    thedist = ME(matrix=Tmatrix, injection=alpha, ejection=eta)

    return thedist, species

def create_distribution_crpropa(nuclei, boost, all_rates, all_branchings, Zinj=None, Ainj=None, nloss=None, idx=None):
    """Return the matrix-exponential distance-until-absorption
    distribution for a nucleus (Zinj, Ainj) losing up to nloss nucleons.
    Same construction as `create_distribution`, kept as a separate entry
    point for CRPropa-sourced rate/branching tables.

    Parameters
    ----------
    nuclei : list of tuple of int
        Full list of (Z, A) nuclides that `all_rates`/`all_branchings`
        are indexed against.
    boost : unused
        Accepted for signature compatibility with CRPropa-style callers;
        not read by this function (`idx` selects the boost row instead).
    all_rates : array_like
        Per-nuclide interaction rate on the boost grid, indexed like
        `nuclei`.
    all_branchings : list
        Per-nuclide list of branchings to daughters, indexed like
        `nuclei`; see `complete_matrix`.
    Zinj, Ainj : int
        Charge and mass number of the injected nucleus.
    nloss : int
        Number of nucleons lost that defines absorption (species with
        A = Ainj - nloss are the absorbing states).
    idx : int, optional
        Index into the boost grid to build the matrix at. Default 0.

    Returns
    -------
    thedist : ME
        The matrix-exponential distribution (a `scipy.stats.rv_continuous`).
    species : list of tuple of int
        The (Z, A) species in `thedist.matrix`'s state order.
    """
    species = prepare_species_list(nuclei, Zinj, Ainj, nloss)
    Tmatrix = complete_matrix(nuclei, all_rates, all_branchings, species, idx=idx)

    alpha = np.zeros(len(Tmatrix))
    alpha[0] = 1 # injecting only iron, or the heaviest element
    
    # Add injections to fictitious absorbing state
    eta = np.zeros(len(Tmatrix))
    for i, spec in enumerate(species):
        if spec[1] == Ainj - nloss:
            eta[i] = -Tmatrix[i, i]

    thedist = ME(matrix=Tmatrix, injection=alpha, ejection=eta)

    return thedist, species

def reduce_matrix(thedist, species):
    """Constructs the equivalent PH distribution containing 
    only the masses as a CTMC. Based on theorem 1 in Horvath, Telek.
    The mean of nuclei of equal mass is used. 
    """
    Avals = list(sorted(set([A for _, A in species if A>0]), reverse=True))
    Tred = np.zeros((len(Avals), len(Avals)))
    
    for k, A in enumerate(Avals):
        Tred[k, k] = np.mean([thedist.matrix[l, l]
            for l, spec in enumerate(species) if spec[1] == A])
        
        if k + 1 < len(Avals):
            Tred[k, k+1] = -Tred[k, k]

    # Tred[-1, -1] = thedist.matrix[-1, -1]

    print(Tred.diagonal())

    return Tred
    
def transition_matrix(A, k=1, lam=1):
    """Returns the T matrix for a PH distribution
    where all phases are related as the rates of UHECRs
    nuclei proportional to the mass A. The number of nucleons
    to be lost is k and the interaction per nucleon is lam.
    """
    T = np.zeros((k, k))

    for l in range(k-1):
        T[l, l] = -(A - l) * lam
        T[l, l+1] = -T[l, l]

    T[k-1, k-1] = -(A - k + 1) * lam

    return T

def transition_matrix_gaussian_variation(A, k=1, lam=1, sigma=1):
    """Returns the T matrix for a PH distribution
    where all phases are related as the rates of UHECRs
    nuclei proportional to the mass A, and randomly perturbed
    by a factor distributed as a log-gaussian. The number of 
    nucleons to be lost is k and the interaction per nucleon 
    is lam.
    """
    T = transition_matrix(A, k, lam)

    perturbation = 10**np.random.normal(0, .45, T.shape[0])
    perturbed_T = T * perturbation[:, None]

    return perturbed_T

def get_injection_parameters(species, mass_lims=(56, 11), injection_type=('flat', None)):
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
    """
    Amax, Amin = mass_lims
    
    mass_range = np.array([k for k, spec in enumerate(species) if Amax >= spec[1] > Amin])
    alpha = np.ones(len(species))[mass_range]

    itype, params = injection_type

    if itype == 'flat':
        alpha /= sum(alpha)
    elif itype == 'only mass':
        masses = params
        indices = np.array([k for k, idx in enumerate(mass_range) if species[idx][1] not in masses])
        alpha[indices] = 0
        alpha /= sum(alpha)
    elif itype == 'only species':
        species = params
        indices = np.array([k for k, idx in enumerate(mass_range) if species[idx] != species])
        alpha[indices] = 0
        alpha /= sum(alpha)

    return alpha, mass_range

def pdf(x, T, alpha=None, eta=None):
    """Returns the distribution function of a matrix-exponential distribution 
    evaluated in x.
    The transition matrix T reflects the transition probabilities between phases.
    Arguments
    =========
    x: Positions of evaluation in units of a common lambda (x = L*lam1)
    T: matrix of the distribution. Interaction constants in units of a common lambda.
    alpha: starting vector. By default only the first nucleus is injected.
    eta: ending vector. By default is computed as Te with e vector of ones.
    """
    dim = T.shape[0]

    if alpha is None:
        alpha = np.zeros((1, dim))
        # inject only first element
        alpha[0, 0] = 1

    if eta is None:
        eta = -np.matmul(T, np.ones_like(alpha))

    pdf_nonvectorized = lambda t: np.dot(np.dot(alpha, expm(T * t)), eta)
    pdf_vectorized = np.vectorize(pdf_nonvectorized)

    return pdf_vectorized(x)

def cdf(x, T, alpha=None):
    """Returns the cumulative distribution function of a PH distribution evaluated 
    in x. The transition matrix T reflects the transition probabilities between 
    phases.
    Arguments
    =========
    x: Positions of evaluation in units of a common lambda (x = L*lam1)
    T: matrix of the distribution. Interaction constants in units of a common lambda.
    alpha: starting vector. By default only the first nucleus is injected.
    eta: ending vector. By default is computed as Te with e vector of ones.
    """
    dim = T.shape[0]

    if alpha is None:
        alpha = np.zeros((1, dim))
        alpha[0, 0] = 1

    v_ones = np.ones((dim, 1))

    pdf_nonvectorized = lambda t: 1 - np.dot(np.dot(alpha, expm(T * t)), v_ones)
    pdf_vectorized = np.vectorize(pdf_nonvectorized)

    return pdf_vectorized(x)

def momenta(T, alpha=None):
    """Returns the mean and standard deviation of a matrix-exponential distribution
    The transition matrix T reflects the transition probabilities between phases.
    Arguments
    =========
    T: matrix of the distribution
    alpha: starting vector. By default only the first nucleus is injected.
    Returns
    """
    dim = T.shape[0]

    if alpha is None:
        alpha = np.zeros((1, dim))
        alpha[0, 0] = 1

    invT = np.linalg.inv(T)
    mean = -alpha.dot(invT.dot(np.ones_like(alpha).T))
    variance = 2*alpha.dot(np.linalg.matrix_power(invT, 2).dot(np.ones_like(alpha).T)) - mean**2

    return mean[0][0], np.sqrt(variance)[0][0]

class ME(rv_continuous):
    """Probability distribution of the disintegration of nuclei
    over a distance in units of a mean disintegration distance.

    The distribution is a Matrix Exponential, and the characteristic
    matrix contains the interaction constants related to the production 
    of sequential nuclei with lighter masses the longer the propagation.
    """
    def __init__(self, matrix, injection, ejection, momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, extradoc=None, seed=None):
        """Takes the interaction matrix and injection vector to produce 
        cdf and pdf., and related quantities.
        """
        super().__init__(momtype, a, b, xtol, badvalue, name, longname, shapes, extradoc, seed)

        self.injection = injection
        self.ejection = ejection
        self.matrix = matrix

    def _pdf(self, x):
        return pdf(x, self.matrix, alpha=self.injection, eta=self.ejection)

    def _cdf(self, x):
        return cdf(x, self.matrix, alpha=self.injection)