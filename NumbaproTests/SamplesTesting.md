#Numbapro performance tests

Testing GPU: GeForce GTX 870M

After the installation of Numbapro, we digged into its performance optimization and tried different ways of accelerating.

Generally, there are three methods to improve the python computing ability. First of all, using `vectorize` to automatically accelerate is a very simple but useful way, which converts a scalar implementatioin to a vectorized program. Besides, Numbapro provides CUDA libraries Host API, like cuRAND random number generator that performs high quality GPU-accelerated random number generation 8X faster than typical CPU only code. Finally, it is also possible to write CUDA directly in Python code with the interaction to CUDA driver API supported by Numbapro.

## Monte Carlo Option Pricer via Numpy

The following is a general Numpy example to implement Monte Carlo Option Pricer and its performance, here is part of the code:

```python

import numpy as np                         # numpy namespace
from timeit import default_timer as timer  # for timing
from matplotlib import pyplot              # for plotting
import math

def step_numpy(dt, prices, c0, c1, noises):
    return prices * np.exp(c0 * dt + c1 * noises)

def mc_numpy(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in xrange(1, paths.shape[1]):   # for each time step
        prices = paths[:, j - 1]          # last prices
        # gaussian noises for simulation
        noises = np.random.normal(0., 1., prices.size)
        # simulate
        paths[:, j] = step_numpy(dt, prices, c0, c1, noises)
```

<img src="https://raw.githubusercontent.com/shaowei-su/CSC453/master/NumbaproTests/1.png" width="400px" height="188px">




