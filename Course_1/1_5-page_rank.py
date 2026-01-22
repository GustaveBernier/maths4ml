# Before we begin, let's load the libraries.
import numpy as np
import numpy.linalg as la
from assets.PageRankFunctions import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# GRADED FUNCTION
# Complete this function to provide the PageRank for an arbitrarily sized internet.
# I.e. the principal eigenvector of the damped system, using the power iteration method.
# (Normalisation doesn't matter here)
# The functions inputs are the linkMatrix, and d the damping parameter - as defined in this worksheet.
# (The damping parameter, d, will be set by the function - no need to set this yourself.)
def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1-d)/n * np.ones([n, n]) # np.ones() is the J matrix, with ones for each entry.
    r = 100 * np.ones(n) / n # Sets up this vector (n entries of 1/n × 100 each)
    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01 :
        lastR = r
        r = M @ r
        i += 1

    print(str(i) + " iterations to convergence.")  
    print(r)  
    return r

L = generate_internet(100)
d = 0.9
r = pageRank(L, d)
plt.bar(np.arange(r.shape[0]), r)
plt.show()