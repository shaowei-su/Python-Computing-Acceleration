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

 # stock parameter

StockPrice = 20.83
StrikePrice = 21.50
Volatility = 0.021
InterestRate = 0.20
Maturity = 5. / 12.

# monte-carlo parameter 

NumPath = 3000000
NumStep = 100

# plotting
MAX_PATH_IN_PLOT = 50

def driver(pricer, do_plot=False):
    paths = np.zeros((NumPath, NumStep + 1), order='F')
    paths[:, 0] = StockPrice
    DT = Maturity / NumStep

    ts = timer()
    pricer(paths, DT, InterestRate, Volatility)
    te = timer()
    elapsed = te - ts

    ST = paths[:, -1]
    PaidOff = np.maximum(paths[:, -1] - StrikePrice, 0)
    print 'Result'
    fmt = '%20s: %s'
    print fmt % ('stock price', np.mean(ST))
    print fmt % ('standard error', np.std(ST) / np.sqrt(NumPath))
    print fmt % ('paid off', np.mean(PaidOff))
    optionprice = np.mean(PaidOff) * np.exp(-InterestRate * Maturity)
    print fmt % ('option price', optionprice)

    print 'Performance'
    NumCompute = NumPath * NumStep
    print fmt % ('Mstep/second', '%.2f' % (NumCompute / elapsed / 1e6))
    print fmt % ('time elapsed', '%.3fs' % (te - ts))

    if do_plot:
        pathct = min(NumPath, MAX_PATH_IN_PLOT)
        for i in xrange(pathct):
            pyplot.plot(paths[i])
        print 'Plotting %d/%d paths' % (pathct, NumPath)
        pyplot.show()
    return elapsed

numpy_time = driver(mc_numpy, do_plot=True)