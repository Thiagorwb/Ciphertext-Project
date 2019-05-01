import numpy as np
from util import *
from math import log

def decode(ciphertext, has_breakpoint):
    """ input: ciphertext, list of chars
       has_breakpoint boolean

       output: plaintext, decoded; list of chars
    """
    # initialization and reading
    alphabet = _read_csv("data", "alphabet")[0]
    transitions = _read_csv("data", "letter_transition_matrix")
    letterp = _read_csv("data", "letter_probabilities")[0]

    # constants
    m = len(alphabet)
    niters = 100
    nstop = 20

    # precomputation of count matrix, the sufficient statistic
    textvector = to_index(ciphertext)
    firstindex = textvector[0]
    ciphermatrix = count_matrix(textvector, m)

    # decifer
    pcipherindex = decodeMC(niters, nstop, ciphermatrix, firstindex, transitions, letterp, has_breakpoint)
    pcipher = to_text(pcipherindex)

    # decode
    d = dict(zip(alphabet, range(len(alphabet))))
    plaintext = ''.join(pcipher[d[ltr]] for ltr in ciphertext)

    return plaintext

def count_matrix(textvector, size):
    """ input: numpy array of letter indices in [size], and size

    computes the matrix of counts, a sufficient statistic for the model

    output: list of lists, where element ij is the count of times i is preceded by j"""

    matrix = np.zeros((size, size))

    for j in range(len(textvector)-1):
        matrix[textvector[j+1], textvector[j]] += 1

    return matrix 



def decodeMC(niters, nstop, ciphermatrix, firstindex, transitions, letterp, has_breakpoint=False):
    """ input: ciphermatrix, a matrixij list of lists of counts when i is preceded by j in ciphertext
            firstindex, first letter in chain
            alphabet, list of chars
            transitions, list of lists of transition probabilities
            letterp, list of letter probabilities
            has_breakpoint, boolean

      output: list of chars, the most likely permutation; possibly a list of likelihoods """

   # initialize parameters, counters, iteration counter t and repetition counter s
    m = len(letterp)
    t = 0
    s = 0

   # initialize permutation
    permutation = np.random.permutation(range(m))

    while t < niters and s < nstop:

      # edge in chain
        a, b = np.random.randint(0, m + 1, 2)
        new_permutation = permutation.copy()
        new_permutation[a], new_permutation[b] = new_permutation[b],new_permutation[a]

      # log likelihood ratio
        ratiolog = computeloglikelihood(ciphermatrix, firstindex, new_permutation, transitions, letterp) - computeloglikelihood(ciphermatrix, firstindex, permutation, transitions, letterp)

      # accept probability
        x = np.random.random()

        if ratiolog > math.log(x):
            permutation = new_permutation
        else: 
            s+= 1

        t += 1

    return permutation


def computeloglikelihood(ciphermatrix, firstindex, permutation, transitions, letterp):
    """ input: ciphermatrix, np.array count matrix; firstindex, int of first char in chain;
            permutation, list of ints; transitions, list of list of probabilities;
            letterp, list of probabilities


      output: log likelihood 

    """

   # initialization
    m = len(permutation)

   # computes f^-1, the inverse permutation
    inverse_permutation = np.zeros(m)
    for i in range(m):
        inverse_permutation[permutation[i]] = i

   # loglikelihood of initial char in chain
    value = math.log(letterp[inverse_permutation[firstindex]])

   # computation of conditional log likelihood
    for i in range(m):
        for j in range(m):
            value += ciphermatrix[i][j]*math.log(transitions[inverse_permutation[i]][inverse_permutation[j]])

    return value