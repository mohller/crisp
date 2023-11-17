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
    including Ainj-k are included.

    When random_chain is true, a random sequence of decreasing mass nuclei
    is returned, where they  
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

    Arguments:
    ---------
    nuclei : (list) nuclei to be included in the disintegration matrix, specified as (Z, A)
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
    """Return pdf for starting a given species and lose a given number of nucleons

    Arguments:
    ---------
    nuclei : (list) nuclei to be included in the disintegration matrix, specified as (Z, A)
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
    """Return pdf for starting a given species and lose a given number of nucleons. Adapted for crpropa data.

    Arguments:
    ---------
    nuclei : (list) nuclei to be included in the disintegration matrix, specified as (Z, A)
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
        masses = range(*mass_lims)
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
        alpha[0, 0] = 1

    if eta is None:
        eta = np.zeros((dim, 1))
        eta[-1, 0] = -T[-1, -1]

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