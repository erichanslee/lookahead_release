"""
NOTE 1: all test functions designed to be evaluated on cube [0.1]^d
Make sure domains are slighly uneven (b/c the minimum is often in
the exact middle of the domain for these test functions)

NOTE 2: most test functions' implementations copied directly either from
SFU Virtual Library of Simulation Experiments:
https://www.sfu.ca/~ssurjano/index.html

or from pySOT:
https://github.com/dme65/pySOT/blob/master/pySOT/optimization_problems.py
"""

import numpy as np

def from_unit_box(x, lb, ub):
    return lb + (ub - lb) * x

# Evaluated from [-5, 10]^2
def rosenbrock(x):
    lb = np.full((2,), -5)
    ub = np.full((2,), 10)
    x = from_unit_box(x, lb, ub)
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 1. - x1
    b = x2 - x1*x1
    return a*a + b*b*100.

# Evaluated by default from [-4, 5]^d
def ackley(x, lb=None, ub=None):
    n, d = x.shape
    if lb is None or ub is None:
        lb = np.full((d,), -4)
        ub = np.full((d,), 5)
    x = from_unit_box(x, lb, ub)
    return -20.0 * np.exp(-0.2*np.sqrt(np.sum(x**2, axis=1) / d)) - \
        np.exp(np.sum(np.cos(2.0*np.pi*x), axis=1) / d) + 20 + np.exp(1)


# Evaluated by default from [-2.5, 3]^d
def rastrigin(x, lb=None, ub=None):
    n, d = x.shape
    if lb is None or ub is None:
        lb = np.full((d,), -2.5)
        ub = np.full((d,), 3)
    x = from_unit_box(x, lb, ub)
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


# Evaluated by default from [-1.8, 2.2]^2
def sixhump(x, lb=None, ub=None):
    if x.shape[1] != 2:
        raise Exception('Dimension must be 2')
    d = 2
    if lb is None or ub is None:
        lb = np.full((d,), -1.8)
        ub = np.full((d,),  2.2)
    x = from_unit_box(x, lb, ub)
    return (4.0 - 2.1*x[:, 0]**2 + (x[:, 0]**4)/3.0)*x[:, 0]**2 + \
        x[:, 0]*x[:, 1] + (-4 + 4*x[:, 1]**2) * x[:, 1]**2

# Evaluated from [-5, 10] x [0, 15]
def branin(x, lb=None, ub=None):
    if x.shape[1] != 2:
        raise Exception('Dimension must be 2')
    d = 2
    if lb is None or ub is None:
        lb = np.full((d,), 0)
        ub = np.full((d,), 0)
        lb[0] = -5
        lb[1] = 0
        ub[0] = 10
        ub[1] = 15
    x = from_unit_box(x, lb, ub)
    x1 = x[:,0]
    x2 = x[:,1]
    t = 1 / (8 * np.pi)
    s = 10
    r = 6
    c = 5 / np.pi
    b = 5.1 / (4 * np.pi ** 2)
    a = 1
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s


def hartman3(x, lb=None, ub=None):
    if x.shape[1] != 3:
        raise Exception('Dimension must be 3')
    d = 3
    if lb is None or ub is None:
        lb = np.full((d,), 0)
        ub = np.full((d,), 1)
    x = from_unit_box(x, lb, ub)
    alpha = np.array([1, 1.2, 3, 3.2])
    A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0],
                 [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
    P = np.array([[0.3689, 0.1170, 0.2673],
                 [0.4699, 0.4387, 0.747],
                 [0.1091, 0.8732, 0.5547],
                 [0.0381, 0.5743, 0.8828]])
    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(3):
            xj = x[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner += Aij * ((xj-Pij) ** 2)
        outer += alpha[ii] * np.exp(-inner)
    return -outer


def hartman4(x, lb=None, ub=None):
    if x.shape[1] != 4:
        raise Exception('Dimension must be 4')
    d = 4
    if lb is None or ub is None:
        lb = np.full((d,), 0)
        ub = np.full((d,), 1)
    x = from_unit_box(x, lb, ub)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
        ])
    P = 1e-4 * np.array([
       [1312, 1696, 5569, 124, 8283, 5886],
       [2329, 4135, 8307, 3736, 1004, 9991],
       [2348, 1451, 3522, 2883, 3047, 6650],
       [4047, 8828, 8732, 5743, 1091, 381]
       ])
    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(4):
            xj = x[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * ((xj-Pij)**2)
        outer += alpha[ii] * np.exp(-inner)
    return (1.1 - outer) / 0.839


def levy(x, lb=None, ub=None):
    d = x.shape[1]
    if lb is None or ub is None:
        lb = -5*np.full((d,), 1)
        ub = 5*np.full((d,), 1)
    x = from_unit_box(x, lb, ub)
    w = 1 + (x - 1.0) / 4.0
    return np.sin(np.pi*w[:, 0]) ** 2 + \
        np.sum((w[:, 1:d-1]-1)**2 * (1 + 10*np.sin(np.pi*w[:, 1:d-1]+1)**2), axis=1) + \
        (w[:, d-1] - 1)**2 * (1 + np.sin(2*np.pi*w[:, d-1])**2)


def hartman6(x, lb=None, ub=None):
    if x.shape[1] != 6:
        raise Exception('Dimension must be 6')
    d = 6
    if lb is None or ub is None:
        lb = np.full((d,), 0)
        ub = np.full((d,), 1)
    x = from_unit_box(x, lb, ub)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10.0, 3.0,  17.0, 3.5,  1.7,  8.0],
                  [0.05, 10.0, 17.0, 0.1,  8.0,  14.0],
                  [3.0,  3.5,  1.7,  10.0, 17.0, 8.0],
                  [17.0, 8.0,  0.05, 10.0, 0.1,  14.0]])
    P = 1e-4 * np.array([[1312.0, 1696.0, 5569.0, 124.0,  8283.0, 5886.0],
                         [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                         [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                         [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]])
    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(6):
            xj = x[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner += Aij * ((xj - Pij) ** 2)
        outer += alpha[ii] * np.exp(-inner)
    return -outer
