import numpy as np
import time

# Takes min of run information
def process_run(yhist):
    for i in range(1, yhist.shape[0]):
        yhist[i] = np.min(yhist[0:i+1])
    return yhist

def filter_cand_points(acq_function, cand_points):
    vals = acq_function.evaluate_at_point_list(cand_points)

    # calculate particular information about vals
    median_vals = np.median(vals)
    return cand_points[vals > median_vals, :]

# creates a reshaped meshgrid matrix of size n^d x d
# where n = size_per_dimension
def unit_grid_points(size_per_dimension, d):
    a = np.linspace(0, 1, size_per_dimension)
    A = (a,) * d
    return np.dstack(np.meshgrid(*A)).ravel('F').reshape(len(A), -1).T

# Generates uniform grid of size_per_dimension x size_per_dimesion,
# reshaped into proper dimensions
def unit_grid_points2(size_per_dimension):
    x = np.linspace(0, 1, size_per_dimension)
    X1, X2 = np.meshgrid(x, x)
    X1vec = np.ndarray.flatten(X1)
    X2vec = np.ndarray.flatten(X2)
    X = np.stack((X1vec, X2vec), axis=1)
    return X, X1, X2

# A check if points are inside unit cube or not
def in_unit_cube(points):
    if(points.shape[1] > 2):
        raise Exception('Only used when d < 3')

    if(points.shape[1] == 2):
        return all(points[:,0] >= 0) \
        and all(points[:,0] <= 1) \
        and all(points[:,1] >= 0) \
        and all(points[:,1] <= 1)
    elif(points.shape[1] == 1):
        return all(points[:,0] >= 0) \
        and all(points[:,0] <= 1)

# A simple normpdf function if
# we don't want to use the scipy one
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(x-mean)**2/(2*var))
    return num/denom

def slhd(num_pts, dim, seed=None):
    # Symmetric Latin Hypercube initial design
    # Implementation is found in PySOT:
    #
    # D. Eriksson and D. Bindel, PySOT,
    # https://github.com/dme65/pySOT
    # Fix the seed if necessary
    if seed is not None:
        np.random.seed(seed)

    # Generate a one-dimensional array based on sample number
    points = np.zeros([num_pts, dim])
    points[:, 0] = np.arange(1, num_pts+1)

    # Get the last index of the row in the top half of the hypercube
    middleind = num_pts // 2

    # special manipulation if odd number of rows
    if num_pts % 2 == 1:
        points[middleind, :] = middleind + 1

    # Generate the top half of the hypercube matrix
    for j in range(1, dim):
        for i in range(middleind):
            if np.random.random() < 0.5:
                points[i, j] = num_pts - i
            else:
                points[i, j] = i + 1
        np.random.shuffle(points[:middleind, j])

    # Generate the bottom half of the hypercube matrix
    for i in range(middleind, num_pts):
        points[i, :] = num_pts + 1 - points[num_pts - 1 - i, :]

    # Get new random seed:
    t = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(t) % 2**32)

    return (points - 1) / (num_pts - 1)  # Map to [0, 1]^d
