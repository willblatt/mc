# https://www.johndcook.com/blog/distributions_scipy/

from scipy import stats
import numpy as np


def montecarlo(model, n_trials, *dists, **scalars):
    rand_variates = [dist.rvs(size=n_trials) for dist in dists]
    vals = model(*rand_variates, **scalars)

    percentiles = [0.005,
                   0.010,
                   0.020,
                   0.100,
                   01.00,
                   10.00,
                   50.00,
                   90.00,
                   99.00,
                   99.90,
                   99.98,
                   99.99,
                   99.995]

    # Calculate individual sensitivities for each distribution
    rank_corr_coefs = [stats.spearmanr(var, vals)[0]**2 for var in rand_variates]
    sensitivities = rank_corr_coefs/np.sum(rank_corr_coefs)
    scores = [stats.scoreatpercentile(vals, per) for per in percentiles]

    print('Sensitivities:')
    for d, s in zip(dists, sensitivities):
        print('{: <10}: {:.3f}'.format(d.name, s))

    print('---')

    print('Percentiles:')
    for s, v in zip(percentiles, scores):
        print('{:6.3f}: {:7.6f}'.format(s, v))


def volume(l, w, h, **scalars):
    return l*w*h


if __name__ == '__main__':
    l = stats.norm(1, 0.01)
    l.name = 'length'

    w = stats.norm(1, 0.01)
    w.name = 'width'

    h = stats.norm(1, 0.01)
    h.name = 'height'

    n_trials = 1000000

    montecarlo(volume, n_trials, l, w, h, density=4, color='brown')

