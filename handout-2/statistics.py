"""
Plots and statistics

"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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














