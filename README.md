# Ciphertext-Project

This is the repo for my 6.437 project, a MCMC method for ciphertext decryption. Here I describe my implementation and relevant improvements.

## Improvements
Here I describe the algorithmic and computational improvements used.

### 1. Riffle Shuffle Markov Chain

I model the markov chain as riffle shuffling, i.e. in the Gilbert-Shannon-Reeds model of Card Shuffling. Although generating each individual permutation is more complex, i.e. O(n) complexity as opposed to the original O(1), the overall required number of iterations to mix is considerably lower, O(log n) as opposed to naive O(n^2logn).
The full algorithmic description follows:

* Partition the permutation list in 2 at some point k chosen from a binomial distribution. 
* With probability proportional to the size of the partition, take a card from each partition to construct the next permutation, until one of the partitions terminates.

### 2. Linear Assignment Initialization

Initializing the MCMC permutation uniformly at random is particularly inefficient, and can be considerably increased to a value ~.3 via linear assignment. 
In particular, define the cost matrix C_{ij} = |P_{i}-E_{j}|, P_{i} the a priori probability of symbol i, E_{j} the empirical frequency of symbol j in the text.
We can use the Hungarian algorithm to solve this in polynomial time in the size of the alphabet. 
A short algorithmic description follows:

Given cost matrix C
* Subtract the smallest entry in each row from the entries in said rows
* Next, do the same on the columns
* Cover the 0's in the resulting matrix using the smallest subset of rows and columns of the matrix
* If the size of the subset is the number of rows of the matrix, then we are done. If not,
* Find the smallest entry not covered by any line. Subtract it from each uncovered row, and add it to each covered column. Repeat #3

### 3. Majority Voting
 
Instead of outputting the maximum likelihood permutation, let us output a weighted permutation biased by the random walk.


### 4. Ensemble MCMC

The MCMC sampling is a biased random walk on a graph. As such, inevitably this system can be caught in local probability maxima during the walk. 
A solution to this is to run multiple MCMC's in parallel. In this case, with 10 processors.



