import numpy as np
import util
from math import log
import statistics

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
    firstindex = textvector[0]
    ciphermatrix = count_matrix(textvector, m)

    # decifer, and statistics tuple
    permutation, data = decodeMC(niters, nstop, ciphermatrix, firstindex, transitions, letterp, has_breakpoint)

    # decode
    plaintext = decrypt(permutation, textvector, alphabet)

    return plaintext, data

def count_matrix(textvector, size):
    """ input: numpy array of letter indices in [size], and size

    computes the matrix of counts, a sufficient statistic for the model

    output: list of lists, where element ij is the count of times i is preceded by j"""

    matrix = np.zeros((size, size))

    for j in range(len(textvector)-1):
        matrix[textvector[j+1]][textvector[j]] += 1

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

   # initialize statistics, list of loglikelihoods, boolean list of acceptances, and stored list of permutations
    loglist = []
    alist = []
    permutation_list = []

   # initialize permutation
    notvalid = True
    while notvalid:

        permutation = np.random.permutation(range(m))
        
        if computeloglikelihood(ciphermatrix, firstindex, permutation, transitions, letterp) != -float("inf"):
            notvalid = False

    while t < niters and s < nstop:

        permutation_list.append(permutation)

        # traverse edge in chain
        a, b = np.random.randint(0, m, 2)
        new_permutation = permutation.copy()
        new_permutation[a], new_permutation[b] = new_permutation[b],new_permutation[a]

        # log likelihood ratio
        loglnew = computeloglikelihood(ciphermatrix, firstindex, new_permutation, transitions, letterp)
        loglold = computeloglikelihood(ciphermatrix, firstindex, permutation, transitions, letterp)
        ratiolog = loglnew - loglold
        loglist.append(loglold)

        # accept probability
        x = np.random.random()

        if ratiolog > log(x):
            permutation = new_permutation
            s = 0
            alist.append(1)
        else: 
            s+= 1
            alist.append(0)

        t += 1

    return permutation, (loglist, alist, permutation_list)

def computeloglikelihood(ciphermatrix, firstindex, permutation, transitions, letterp):
    """ input: ciphermatrix, np.array count matrix; firstindex, int of first char in chain;
            permutation, list of ints; transitions, list of list of probabilities;
            letterp, list of probabilities


      output: log likelihood 

    """

   # initialization
    m = len(permutation)

   # loglikelihood of initial char in chain
    v = float(letterp[int(permutation[firstindex])])
    value = log(v)

   # computation of conditional log likelihood
    for i in range(m):
        for j in range(m):
            v = float(transitions[int(permutation[i])][int(permutation[j])])

            if v == 0 and ciphermatrix[i][j]!= 0:
                value = -float("inf")
                break
            elif v != 0:
                value += ciphermatrix[i][j]*log(v)

    return value

def decrypt(permutation, textvector, alphabet):
    pcipher = util.to_text(permutation, alphabet)
    plaintext = ''.join(pcipher[index] for index in textvector)
    return plaintext

def accuracy(plaintext, computedtext):
    # computes accuracy
    c = 0
    for i in range(len(plaintext)):
        if plaintext[i]!=computedtext[i]:
            c+= 1

    return float(c)/len(plaintext)

def accuracyplot(permutation_list, plaintext, textvector, alphabet):
    # plots accuracy during the step evolution

    data = accuracydata(permutation_list, plaintext, textvector, alphabet)

    statistics.plainplot("Iteration", "accuracy", "Accuracy.png", data)

def accuracydata(permutation_list, plaintext, textvector, alphabet):
    # initialization
    accuracies = []

    # decryption and accuracy
    for permutation in permutation_list:
        computedtext = decrypt(permutation, textvector, alphabet)
        accuracies.append(accuracy(plaintext, computedtext))

    accuracies.reverse()

    return accuracies

def lengthexperiment(number, plaintext, textvector, alphabet):

    n = len(plaintext)
    accuracies = []

    for index in range(1, number+1):
        size = index * n//number

        computedtext, data = decode(ciphertext[:size], False)
        (loglist, acceptances, permutation_list) = data

        accuracies.append(accuracydata(permutation_list, plaintext, textvector, alphabet))

    statistics.multipleplot("Iteration", "Accuracies", "sizes", accuracies, n)



if __name__ == '__main__':

    # import text
    ciphertext = util._read_text(None, "test_ciphertext")
    plaintext = util._read_text(None, "test_plaintext")
    alphabet = util._read_csv("data", "alphabet")[0]
    textvector = util.to_index(ciphertext, alphabet)

    computedtext, data = decode(ciphertext, False)
    (loglist, acceptances, permutation_list) = data
    newloglist = []
    for logl in loglist:
        newloglist.append(logl/(log(2)*len(plaintext)))

    statistics.plainplot("Iteration", "Log likelihood per symbol [bits]", "bitplot.png", newloglist)

    """

    # Decode
    computedtext, data = decode(ciphertext, False)
    (loglist, acceptances, permutation_list) = data

    # Correctness, number of mistakes
    if len(computedtext)!=len(ciphertext):
        print "length FAIL"

    c = 0
    for i in range(len(plaintext)):
        if plaintext[i]!=computedtext[i]:
            c += 1

    print ("the final number of errors is ", c)

    # Plots and statistics
    accuracyplot(permutation_list, plaintext, textvector, alphabet)
    statistics.acceptanceplot(acceptances[:3000], 100)
    statistics.likelihoodplot(loglist[:3000])

    """








