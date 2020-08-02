from glob import glob
import pandas as pd
import numpy as np
import re
import os 
from numpy import genfromtxt

# Base folders, change if needed
ROOT = '/home/erichanslee/Look-Ahead'
COLORS = ['r', 'b', 'g', 'm', 'k', 'y']


# MIN OF SYNTHETIC FUNCTIONS:
MINIMA = {'ackley' : 0, 'rastrigin' : 0, 'branin': 0.397887, 'sixhump': -1.0316, 'levy': 0}

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Load only one blackbox at a time
def load_data(optimizers, blackbox, seeds=None):
    if seeds is None:
        seeds = ['*']
    if optimizers is None:
        optimizers = ['*']
    data = {}
    for opt in optimizers:
        files = []
        for seed in seeds:
            files += glob(
                '{}/results/{}/{}/{}.csv'.format(
                    ROOT, opt, blackbox, seed
                )
            )
            # need this additional line because I saved auxillary data with the 'aux' modified, and we don't want to load that
            files = [f for f in files if 'aux' not in f]
        data[opt] = np.array([load_data_file(f) for f in files])
    return data


def get_best(result):
    mmin = result[0]
    for i in range(1, len(result)):
        if result[i] < mmin:
            mmin = result[i]
        result[i] = mmin
    return result


def load_data_file(filename):
    """loads a single file into a DataFrame"""
    regexp = '^.*/results/([^/]+)/([^/]+)/([^/]+).csv$'
    optimizer, blackbox, seed = re.match(regexp, filename).groups()
    f = ROOT + '/results/{}/{}/{}.csv'.format(optimizer, blackbox, seed)
    result = np.genfromtxt(f, delimiter=',')
    return get_best(result)


def plot_runs(ax, optimizers, blackbox, colors=None, markers=None, skip=1, offset=0, offset_end=-1, mean=True):
    if colors is None:
        colors = COLORS
    if markers is None:
        markers = ['*' for _ in range(len(optimizers))]

    d = load_data(optimizers, blackbox)
    lines = []
    for opt, color, marker in zip(optimizers, colors, markers):

        if opt in MINIMA.keys():
            mmin = MINIMA[opt]
        else:
            mmin = 0

        runs = d[opt]
        if mean:
            m = np.mean(runs, axis=0)
        else:
            m = np.median(runs, axis=0)
        std = np.std(runs, axis=0)
        t = np.arange(len(m))[offset:offset_end:skip]
        m = m[offset:offset_end:skip] - mmin
        std_err = std[offset:offset_end:skip] / np.sqrt(m.shape[0])
        line = ax.plot(t, m, '--{}'.format(color), label=opt)
        lines = lines + line
        ax.fill_between(t, m-std_err, m+std_err, color=color, alpha=0.2)

    # ax.set_yscale('log')
    return lines
