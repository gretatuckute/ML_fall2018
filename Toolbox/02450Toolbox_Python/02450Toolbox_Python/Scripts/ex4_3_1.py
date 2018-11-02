# exercise 4.3.1

from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
import numpy as np


# Number of samples
N = 200

# Mean
mu = 17

# Standard deviation
s = 2

# Number of bins in histogram
nbins = 100

# Generate samples from the Normal distribution
X = np.random.normal(mu,s,N).T 
# or equally:
X = np.random.randn(N).T * s + mu

# Plot the samples and histogram
figure(figsize=(12,4))
title('Normal distribution')
subplot(1,2,1)
plot(X,'.')
subplot(1,3,3)
hist(X, bins=nbins)
show()

print('Ran Exercise 4.3.1')