# Ciphertext-Project

This is the repo for my 6.437 project, a MCMC method for ciphertext decryption. Here I describe my implementation and relevant improvements.

## The MCMC

This is an outline of the algorithmic description used. A full version can be found in the writeup.

I model the markov chain as a graph where the vertices are the possible ciphers, alphabet permutations, and the edges are between permutations that differ in exactly 2 elements. We precompute the ciphertext into a ciphermatrix, a count of the number of transitions of letters i to j in the text that is a sufficient statistic for this model. This allows us to quickly compute the log likelihoods. We generate the next permutation by picking 2 random indices, and swapping those positions in the permutation. The log acceptance probability is given by the difference between log likelihoods of each permutation. 

I furthermore implemented subroutines in case the cipher has a breakpoint, i.e. at some unique position x the plaintext starts being encoded with a different cipher. We find said position x by brute force search, running the MCMC algorithm on both partitions of the text with some large step to find a sum of loglikelihoods maxima, the fine grain search in the 2 adjacent steps. 

## Improvements
Here I describe the algorithmic and performance improvements used.

### 1. Linear Assignment Initialization

Initializing the MCMC permutation uniformly at random is particularly inefficient, since the initial log likelihood could be infinite, or very low. However, if we initialize the permutation via linear assignment, we can guarantee a ~4.5% accuracy initial state that can be generated in time .01s, much faster than the overall runtime.
In particular, define the cost matrix C_{ij} = |P_{i}-E_{j}|, P_{i} the a priori probability of symbol i, E_{j} the empirical frequency of symbol j in the text.
We can use the Hungarian algorithm to solve this in polynomial time in the size of the alphabet, and I used the hungarian alg. implementation available by scikit-learn on github.
A short algorithmic description follows:

Given cost matrix C
* Subtract the smallest entry in each row from the entries in said rows
* Next, do the same on the columns
* Cover the 0's in the resulting matrix using the smallest subset of rows and columns of the matrix
* If the size of the subset is the number of rows of the matrix, then we are done. If not,
* Find the smallest entry not covered by any line. Subtract it from each uncovered row, and add it to each covered column. Repeat #3

### 2. Ensemble MCMC

The MCMC sampling is a biased random walk on a graph. As such, inevitably this system can be caught in local probability maxima during the walk. A solution to this is to run multiple MCMC's in parallel, and return the maximum loglikelihood value among them. Moreover, this allows a considerable time speedup, in contrast to running sequencially the same MCMC that amount of times. 

### 3. Parallel Brute Force Search

When brute force searching for the most probable break position, we partition the search range into some number of regions s.t. each region is run in parallel.

## Additional Attempts

### 1. Riffle Shuffle Markov Chain

I tried to model the markov chain as riffle shuffling, i.e. in the Gilbert-Shannon-Reeds model of Card Shuffling. Although generating each individual permutation is more complex, i.e. O(n) complexity as opposed to the original O(1), the overall required number of iterations to mix is considerably lower, O(log n) as opposed to naive O(n^2logn).
The full algorithmic description follows:

* Partition the permutation list in 2 at some point k chosen from a binomial distribution. 
* With probability proportional to the size of the partition, take a card from each partition to construct the next permutation, until one of the partitions terminates.

However, the difference in likelihoods among 2 adjacent states was very large, and so the step acceptance rates were too low for this scheme to be faster than the simple swap.

