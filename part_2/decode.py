import numpy as np
import util
from math import log
import statistics
import riffleshuffle
import hungarian
from threading import Thread
import time
import matplotlib as plt


def decode(ciphertext, has_breakpoint = False, niters = 5000, N = 5, threaded = True):
    """ 
    input: ciphertext, string; has_breakpoint, boolean
    output: plaintext string, decoded and statistics tuple of loglikelihoods and boolean acceptance
    """

    # initialization and reading
    alphabet = util._read_csv("data", "alphabet")[0]
    transitions = util._read_csv("data", "letter_transition_matrix")
    letterp = util._read_csv("data", "letter_probabilities")[0]

    # constants
    m = len(alphabet)
    nstop = 300

    #indexation and logtransition array
    textvector = util.to_index(ciphertext, alphabet)
    logtransitions = logtransitionmatrix(transitions)

    # breakpoint decode
    if has_breakpoint:
        permutations, logl, position = FindOptimalX(3000, nstop, textvector, logtransitions, letterp, N, threaded)
        plaintext = breakdecrypt(permutations[0], permutations[1], position, textvector, alphabet) 

    else:
        permutation, logl = SimpleDecode(niters, nstop, textvector, logtransitions, letterp,  threaded, N)
        plaintext = decrypt(permutation, textvector, alphabet)

    return plaintext

def EnsembleMCMC(niters, nstop, ciphermatrix, empiricalfrequencies, logtransitions, letterp, N):
    """
    input: integer number of iterations and termination, textvector array, letterp array, and integer number of threads
    output: float maxlogl and permutation array of integers
    """

    #Parallelization
    results = [{} for i in range(N)]
    threads = []

    for i in range(N):
        process = Thread(target = MCMC, args = [niters, nstop, ciphermatrix, empiricalfrequencies, logtransitions, letterp,  True, results, i])
        threads.append(process)
        process.start()

    for process in threads:
        process.join()

    # Finding maximum
    maxpermutation, maxlogl = results[0]
    for (permutation, logl) in results:
        if logl > maxlogl:
            maxlogl = logl
            maxpermutation = permutation

    return permutation, maxlogl


def BreakDecode(niters, nstop, textvector, logtransitions, letterp, N, threaded, x):
    """ Function returns list [f1, f2, x] of cipher functions and approximate breakpoint position x
        Brute Force Search in large intervals then fine grained
    """

    n = len(textvector)

    p1, ll1 = SimpleDecode(niters, nstop, textvector[:x], logtransitions, letterp, N, threaded)
    p2, ll2 = SimpleDecode(niters, nstop, textvector[x:], logtransitions, letterp, N, threaded)

    
    return (p1, p2), ll1 + ll2




def FindOptimalX(niters, nstop, textvector, logtransitions, letterp, N, threaded):
    """ Binary Search on loglikelihoods of given splits"""

    n = len(textvector)
    brute = n/100


    mx = n/2
    permutations = None

    upper = n
    lower = 0

    while (not permutations) or upper - lower >= brute:


        permutations, mll = BreakDecode(niters, nstop, textvector, logtransitions, letterp, N, threaded, mx)
        lx = mx - brute
        leftpermutations, lmll = BreakDecode(niters, nstop, textvector, logtransitions, letterp, N, threaded, lx)
        #rightpermutations, rmll = BreakDecode(niters, nstop, textvector, logtransitions, letterp, N, threaded, rx)

        if mll > lmll:
            lower = mx
        else:
            upper = mx

        mx = (lower + upper)/2


    return permutations, mll, mx




def SimpleDecode(niters, nstop, textvector, logtransitions, letterp,  N, threaded = False):    
    """Conditions on parallelization and calls Markov Chain Methods
        Returns permutation array, loglikelihood value"""

    #statistics of text
    ciphermatrix = statistics.count_matrix(textvector, len(letterp))
    empiricalfrequencies = statistics.empiricalfrequency(textvector, len(letterp))

    #parallelization and repetition
    if not threaded and N == 1:
        return MCMC(niters, nstop, ciphermatrix, empiricalfrequencies, logtransitions, letterp)
    elif N > 1 and threaded:
        return EnsembleMCMC(niters, nstop, ciphermatrix, empiricalfrequencies, logtransitions, letterp, N)
    else: 
        mp, maxlogl = MCMC(niters, nstop, ciphermatrix, empiricalfrequencies, logtransitions, letterp)
        for i in range(N):
            p, logl  = MCMC(niters, nstop, ciphermatrix, empiricalfrequencies, logtransitions, letterp)
            if maxlogl < logl:
                maxlogl = logl
                mp = p

    return mp, maxlogl

def MCMC(niters, nstop, ciphermatrix, empiricalfrequencies, logtransitions, letterp, threaded = False, results = None, i = None):
    """ given ciphermatrix and transition probabilities, returns most likely cipher permutation"""

    # definitions,  precomputation of count matrix, the sufficient statistic
    m = len(letterp)

    #initialize MCMC using Linear assignment
    permutation = Initialize(empiricalfrequencies, letterp)
    logl = loglikelihood(ciphermatrix, logtransitions, permutation)
    t = 0
    s = 0
    
    # initialize statistics
    maximumlogl = logl
    optimumpermutation = permutation

    #MCMC
    c = 0
    while t < niters and s < nstop:

        # step
        new_permutation = swap_permute(permutation)
        new_logl = loglikelihood(ciphermatrix, logtransitions, new_permutation)
        ratiolog = new_logl - logl

        # update optima
        if new_logl > maximumlogl:
            maximumlogl = new_logl
            optimumpermutation = permutation

        # accept probability
        x = np.random.random()

        if ratiolog > log(x):
            permutation = new_permutation
            logl = new_logl
            s = 0
            c+=1
        else: 
            s+= 1

        t += 1

    if threaded:
        results[i] = [optimumpermutation, maximumlogl]
        return

    else:
        return (optimumpermutation, maximumlogl)

def Initialize(empiricalfrequencies, letterp):
    """ 
    Returns initial permutation numpy array assigned by linear assignment. 
    input is lists textvector of integer indices, alphabet of chars, and letter probabilities

    """

    m = len(letterp)

    cost_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            cost_matrix[i, j] = (float(letterp[i]) - empiricalfrequencies[j])**2

    return hungarian.linear_sum_assignment(cost_matrix)[1]

def rifflepermute(permutation):
    """ given numpy array of integers permutation, returns riffle-shuffled permutation"""
    return riffleshuffle.shuffle(permutation)

def swap_permute(permutation):
    # returns permutation array by swapping 2 positions
    m = len(permutation)
    a, b = np.random.randint(0, m, 2)
    new_permutation = permutation.copy()
    new_permutation[a], new_permutation[b] = new_permutation[b],new_permutation[a]
    return new_permutation

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
            value += permutedmatrix[i, j]*float(logtransitions[i, j]) if permutedmatrix[i, j] != 0 else 0

    return value

def logtransitionmatrix(transitions):
    logmatrix = np.zeros_like(transitions)
    m = len(logmatrix)
    for i in range(m):
        for j in range(m):
            logmatrix[i, j] = log(float(transitions[i, j])) if float(transitions[i, j])!=0.0 else -2000
    return logmatrix

def decrypt(permutation, textvector, alphabet):
    pcipher = util.to_text(permutation, alphabet)
    plaintext = ''.join(pcipher[index] for index in textvector)
    return plaintext

def breakdecrypt(p1,p2,x, textvector, alphabet):
    s1 = decrypt(p1, textvector[:x], alphabet)
    s2 = decrypt(p2, textvector[x:], alphabet)
    return s1+s2


if __name__ == '__main__':

    start = time.time()

    # import text
    ciphertext = util._read_text(None, "test_ciphertext_breakpoint")
    plaintext = util._read_text(None, "test_plaintext")
    alphabet = util._read_csv("data", "alphabet")[0]
    #textvector = util.to_index(ciphertext, alphabet)
    #letterp = util._read_csv("data", "letter_probabilities")[0]


   
    ################################

    # Decode
    computedtext = decode(ciphertext, True)

    
    end = time.time()
    print "evaluation time is", end - start

    print "final accuracy is ", statistics.accuracy(plaintext, computedtext)
    

    




