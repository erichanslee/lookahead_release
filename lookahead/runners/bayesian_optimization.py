import numpy as np
from lookahead.model.gaussian_process import GaussianProcessSimple as GaussianProcess
import os

class BayesianOptimization(object):
    def __init__(self, search_space):
        self.opt_name = 'abstract'
        self.gaussian_process = None
        self.search_space = search_space

    def run(self, f, seed, budget_minus_initialization, initialization_duration=5):

        # Warm start with 5 points, with fixed random seed
        np.random.seed(seed)
        d = len(self.search_space.domain_bounds)
        xhist = np.random.rand(initialization_duration, d)
        yhist = f(xhist)
        self.gaussian_process = GaussianProcess(xhist, yhist)
        self.gaussian_process.train()

        while budget_minus_initialization > 0:

            # Get next sample point
            xsample = self.get_next_point()
            ysample = f(xsample)
            xhist = np.vstack((xhist, xsample))
            yhist = np.append(yhist, ysample)
            self.gaussian_process = GaussianProcess(xhist, yhist)
            self.gaussian_process.train()
            budget_minus_initialization -= 1

        xhist, yhist = self.gaussian_process.get_historical_data()
        self.save_bo_run(yhist, str(f.__name__), seed)

    def get_next_point(self):
        # To be implemented by each acquisition function
        pass

    def save_bo_run(self, yhist, objective_name, seed):
        seed = str(seed)
        """
        Saves run to the folder ~/Look-Ahead/results/optimizer_name/objective_name/seed.csv
        """
        base_path = os.path.expanduser('~') + '/Look-Ahead/results/'
        # Make paths if necessary
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path + self.opt_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + objective_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        run_name = path + str(seed) + '.csv'

        # Save data as csv to path
        np.savetxt(run_name, yhist, delimiter=',')
