import numpy as np
import util
from math import log
import statistics
import decodep1
import riffleshuffle
import hungarian
from threading import Thread
import time
import matplotlib as plt


def decode(ciphertext, N, has_breakpoint = None):
    """ 
    input: ciphertext, string; has_breakpoint, boolean
    output: plaintext string, decoded and statistics tuple of loglikelihoods and boolean acceptance
    """

    # initialization and reading
    alphabet = util._read_csv("data", "alphabet")[0]
    transitions = util._read_csv("data", "letter_transition_matrix")
    letterp = util._read_csv("data", "letter_probabilities")[0]

    #Number of threads

    # constants
    m = len(alphabet)
    niters = 4000
    nstop = 100

    #indexation
    textvector = util.to_index(ciphertext, alphabet)

    #compute logs and perturb 0 values
    logtransitions = logtransitionmatrix(transitions)

    
    #ensemble MCMC
    results = [{} for i in range(N)]
    threads = [] 
    for i in range(N):
        # print "starting thread " + str(i)
        process = Thread(target = simpledecode, args = [niters, nstop, textvector, logtransitions, letterp,  results, i, True])
        threads.append(process)
        process.start()

    for process in threads:
        process.join()
    
    optlogl = results[0][1]
    optimumpermutation = results[0][0]
    for (permutation, logl) in results:
        if logl > optlogl:
            optlogl = logl
            optimumpermutation = permutation
  
    """
    
    # single pass
    optimumpermutation, optlogl = simpledecode(niters, nstop, textvector, transitions, letterp)
    for i in range(10):
        p, logl  = simpledecode(niters, nstop, textvector, transitions, letterp)
        if optlogl < logl:
            optlogl = logl
            optimumpermutation = p

    """

    print optlogl


    # decode
    plaintext = decrypt(optimumpermutation, textvector, alphabet)

    return plaintext

def breakpointdecode(niters, nstop, textvector, logtransitions, letterp, results = None,  i = None,  multithread = False):
    """ Function returns list [f1, f2, x] of cipher functions and approximate breakpoint position x through brute force iteration"""
    n = len(textvector)

    imax = 0
    maxll, p0 = simpledecode(niters, nstop, textvector, transitions, letterp, results = None,  i = None,  multithread = False)
    for i in range(n/100, n, n/100):

        # definitions,  precomputation of count matrix, the sufficient statistic
        m = len(letterp)
        ciphermatrix = statistics.count_matrix(textvector[:i], m)







        p1, ll1 = simpledecode(niters, nstop, textvector[:i], logtransitions, letterp, results = None,  i = None,  multithread = False)
        p2, ll2 = simpledecode(niters, nstop, textvector[i:], logtransitions, letterp, results = None,  i = None,  multithread = False)
        
        if maxll < ll1 + ll2:
            maxll = ll1 + ll2
            p0 = (p1, p2)

    return maxll, p0, imax










def simpledecode(niters, nstop, textvector, logtransitions, letterp, results = None,  i = None,  multithread = False):
    """ given ciphermatrix and transition probabilities, returns most likely cipher permutation"""

    # definitions,  precomputation of count matrix, the sufficient statistic
    m = len(letterp)
    ciphermatrix = statistics.count_matrix(textvector, m)

    #initialize MCMC using Linear assignment
    t = 0
    s = 0
    permutation = initialize(textvector, letterp)
    logl = loglikelihood(ciphermatrix, logtransitions, permutation)

    # initialize statistics
    maximumlogl = logl
    optimumpermutation = permutation

    #MCMC
    c = 0
    while t < niters and s < nstop:

        # step
        new_permutation = genpermutationsimple(permutation)
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

    if multithread:
        results[i] = (optimumpermutation, maximumlogl)
        return 

    else:
        return (optimumpermutation, maximumlogl)

def genpermutationsimple(permutation):
    # returns permutation by swapping 2 positions
    m = len(permutation)
    a, b = np.random.randint(0, m, 2)
    new_permutation = permutation.copy()
    new_permutation[a], new_permutation[b] = new_permutation[b],new_permutation[a]
    return new_permutation

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

def genpermutationsimple(permutation):
    # returns permutation by swapping 2 positions
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
            logmatrix[i, j] = log(float(transitions[i, j])) if float(transitions[i, j])!=0.0 else -1000
    return logmatrix

def decrypt(permutation, textvector, alphabet):
    pcipher = util.to_text(permutation, alphabet)
    plaintext = ''.join(pcipher[index] for index in textvector)
    return plaintext

if __name__ == '__main__':

    start = time.time()

    # import text
    ciphertext = util._read_text(None, "test_ciphertext")
    plaintext = util._read_text(None, "test_plaintext")
    alphabet = util._read_csv("data", "alphabet")[0]    

    """
    # parallelization accuracy test

    accuracies = []
    for i in range(2, 45, 5):
        computedtext = decode(ciphertext, i, False)
        accuracies.append(statistics.accuracy(plaintext, computedtext))

    fig, ax = plt.subplots()
    ax.plot(range(1, 35, 5), accuracies)
    ax.set(xlabel = "Number of threads", ylabel = "Accuracy")
    fig.savefig("threads.png")


    """

    # Decode
    computedtext= decode(ciphertext, False)

    # Correctness, number of mistakes
    if len(computedtext)!=len(ciphertext):
        print "length FAIL"

    end = time.time()

    print("evaluation time is", end - start)

    print "accuracy is ", statistics.accuracy(plaintext, computedtext)
    """

    




