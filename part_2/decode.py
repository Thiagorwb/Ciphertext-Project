import numpy as np
import util
from math import log
import statistics
import decodep1
import riffleshuffle
import hungarian

def decode(ciphertext, has_breakpoint):
    """ 
    input: ciphertext, list of chars; has_breakpoint, boolean
    output: plaintext string, decoded and statistics tuple of loglikelihoods and boolean acceptance
    """


    # initialization and reading
    alphabet = util._read_csv("data", "alphabet")[0]
    transitions = util._read_csv("data", "letter_transition_matrix")
    letterp = util._read_csv("data", "letter_probabilities")[0]

    # constants
    m = len(alphabet)
    niters = 3000
    nstop = 1000

    # precomputation of count matrix, the sufficient statistic
    textvector = util.to_index(ciphertext, alphabet)

    #MCMC
    permutation = simpledecode(niters, nstop, textvector, transitions, letterp)

    #decode
    plaintext = decrypt(permutation, textvector, alphabet)

    print("here are the lengths", len(ciphertext), len(plaintext))

    return plaintext





def simpledecode(niters, nstop, textvector, transitions, letterp):
    """ given ciphermatrix and transition probabilities, returns most likely cipher permutation"""

    # definitions
    m = len(letterp)
    ciphermatrix = statistics.count_matrix(textvector, m)
    logtransitions = logtransitionmatrix(transitions)

    #initialize MCMC
    t = 0
    s = 0
    permutation = initialize(textvector, letterp)
    logl = loglikelihood(ciphermatrix, logtransitions, permutation)

    #MCMC
    while t < niters and s < nstop:

        # step
        new_permutation = genpermutation(permutation)
        new_logl = loglikelihood(ciphermatrix, logtransitions, new_permutation)
        ratiolog = new_logl - logl

        # accept probability
        x = np.random.random()

        if ratiolog > log(x):
            permutation = new_permutation
            logl = new_logl
            s = 0
        else: 
            s+= 1

        t += 1

    return permutation



def initialize(textvector, letterp):
    """ 
    Returns initial permutation numpy array assigned by linear assignment. 
    input is lists textvector of integer indices, alphabet of chars, and letter probabilities

    """

    m = len(letterp)

    empiricalfrequencies = statistics.empiricalfrequency(textvector, m)

    cost_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            cost_matrix[i, j] = (float(letterp[i]) - empiricalfrequencies[j])**2

    return hungarian.linear_sum_assignment(cost_matrix)[1]

def genpermutation(permutation):
    """ given numpy array of integers permutation, returns riffle-shuffled permutation"""
    return riffleshuffle.shuffle(permutation)

def loglikelihood(ciphermatrix, logtransitions, permutation):
    """ Returns loglikelihood of given permutation, given precomputed ciphermatrix and log of transition matrix"""

    # initialization 
    m = len(permutation)

    # generate permuted ciphermatrix
    permutedmatrix = np.zeros_like(ciphermatrix)
    for i in range(m):
        for j in range(m):
            permutedmatrix[permutation[i], permutation[j]] = ciphermatrix[i, j]

    # compute loglikelihood
    value = 0.0
    for i in range(m):
        for j in range(m):
            value += permutedmatrix[i, j]*float(logtransitions[i, j])

    return value

def logtransitionmatrix(transitions):
    logmatrix = np.zeros_like(transitions)
    m = len(logmatrix)
    for i in range(m):
        for j in range(m):
            logmatrix[i, j] = log(float(transitions[i, j])) if float(transitions[i, j])!=0 else -float("inf")
    return logmatrix

def decrypt(permutation, textvector, alphabet):
    pcipher = util.to_text(permutation, alphabet)
    plaintext = ''.join(pcipher[index] for index in textvector)
    return plaintext

