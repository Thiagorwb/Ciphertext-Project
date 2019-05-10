"""
Plots and statistics

"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import decode

def acceptanceplot(acceptance, T):
    """input: boolean list acceptance, integer window size T"""

    # construct average acceptance
    avgaccept = np.zeros(len(acceptance) - T)

    for i in range(len(avgaccept)):
        avgaccept[i] = sum(acceptance[i:i+T])/float(T)
    
    # plot initialization
    fig, ax = plt.subplots()
    ax.plot(range(len(avgaccept)), avgaccept)
    ax.set(xlabel = "iteration", ylabel = "Acceptance Ratio")

    plt.show()

    # exposition
    fig.savefig("Acceptance.png")


def likelihoodplot(loglist):
    # plot initialization
    fig, ax = plt.subplots()
    ax.plot(range(len(loglist)), loglist)
    ax.set(xlabel = "iteration", ylabel = "Log Likelihood")

    # exposition
    fig.savefig("logl.png")

def plainplot(xaxisstring, yaxisstring, titlestring, data):
    # plain plot
    fig, ax = plt.subplots()
    ax.plot(range(len(data)), data)
    ax.set(xlabel = xaxisstring, ylabel = yaxisstring)
    fig.savefig(titlestring)

def multipleplot(xaxisstring, yaxisstring, titlestring, data, size):
    number = len(data)
    print(number, size)
    fig, ax = plt.subplots()
    for i in range(1, number+1):
        ax.plot(range(len(data[0])), data[i-1], label = str(size * i/number))
        ax.legend(loc='upper left')

    ax.set(xlabel = xaxisstring, ylabel = yaxisstring)
    plt.show()
    fig.savefig(titlestring)

def empiricalfrequency(textvector, m):
    """ function returns numpy array of relative frequencies of each letter-index, 
        inputs:  textvector list and size of alphabet m"""

    frequencies = np.zeros(m)

    for i in range(len(textvector)):
        frequencies[textvector[i]] += 1.0/float(len(textvector))

    return frequencies

def count_matrix(textvector, m):
    """ input: numpy array of letter indices in [m], and size

    computes the matrix of counts, a sufficient statistic for the model

    output: list of lists, where element ij is the count of times i is preceded by j"""

    matrix = np.zeros((m, m))

    for j in range(len(textvector)-1):
        matrix[textvector[j+1]][textvector[j]] += 1

    return matrix 

def accuracy(plaintext, computedtext):

    if len(plaintext)!=len(computedtext):
        print "Length Fail"
        return 0
    # computes accuracy
    c = 0
    for i in range(len(plaintext)):
        if plaintext[i]==computedtext[i]:
            c+= 1

    return float(c)/len(plaintext)

def parallizationaccuracyplot(ciphertext, plaintext, has_breakpoint = False, npoints = 5, maximum = 35):

    accuracies = []
    for i in range(2, maximum, npoints):
        computedtext = decode.decode(ciphertext, i, has_breakpoint)
        accuracies.append(statistics.accuracy(plaintext, computedtext))

    fig, ax = plt.subplots()
    ax.plot(range(1, 35, 5), accuracies)
    ax.set(xlabel = "Number of threads", ylabel = "Accuracy")
    fig.savefig("threads.png")

















