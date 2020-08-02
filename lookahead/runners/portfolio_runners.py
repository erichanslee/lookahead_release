import numpy as np
from lookahead.runners.bayesian_optimization import BayesianOptimization
from lookahead.acquisitions.rollout_portfolio import RolloutPortfolio, RolloutPortfolioEI
from lookahead.model.gaussian_process import GaussianProcessSimple as GaussianProcess
import os
import csv

class PortfolioRunner(BayesianOptimization):
    """
    A  runner class for Portfolio, Diagnostic because it outputs auxillary data.
    """
    def __init__(self, search_space, horizon):
        super().__init__(search_space)
        self.horizon = horizon
        self.opt_name = 'portfolio' + str(horizon)

    def run(self, f, seed, budget_minus_initialization, initialization_duration=5):


        acquisition_chosen_all = []

        # Warm start with 5 points, with fixed random seed
        np.random.seed(seed)
        d = len(self.search_space.domain_bounds)
        xhist = np.random.rand(5, d)
        yhist = f(xhist)
        self.gaussian_process = GaussianProcess(xhist, yhist)
        self.gaussian_process.train()

        while budget_minus_initialization > 0:

            # Get next sample point
            xsample, acquisition_chosen = self.get_next_point()
            acquisition_chosen_all.append(acquisition_chosen)
            ysample = f(xsample)
            xhist = np.vstack((xhist, xsample))
            yhist = np.append(yhist, ysample)
            self.gaussian_process = GaussianProcess(xhist, yhist)
            self.gaussian_process.train()
            budget_minus_initialization -= 1

        xhist, yhist = self.gaussian_process.get_historical_data()

        # Save BOTH acquisitions chosen as well as run information
        self.save_auxillary_data(acquisition_chosen_all, str(f.__name__), seed)
        self.save_bo_run(yhist, str(f.__name__), seed)


    def get_next_point(self):
        # To be implemented by each acquisition function
        pr = RolloutPortfolio(self.gaussian_process, self.search_space, self.horizon)
        return pr.next_point()


    def save_auxillary_data(self, acquisition_chosen_all, objective_name, seed):
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
        run_name = path + 'portfolio_aux' + str(seed) + '.csv'

        # Save data as csv to path
        with open(run_name, 'w') as file:
            writer = csv.writer(file, delimiter='\n')
            writer.writerow(acquisition_chosen_all)

class PortfolioEIRunner(BayesianOptimization):
    """
    A  runner class for Portfolio, Diagnostic because it outputs auxillary data.
    """
    def __init__(self, search_space, horizon):
        super().__init__(search_space)
        self.horizon = horizon
        self.opt_name = 'portfolio_ei' + str(horizon)

    def run(self, f, seed, budget_minus_initialization, initialization_duration=5):
        acquisition_chosen_all = []

        # Warm start with 5 points, with fixed random seed
        np.random.seed(seed)
        d = len(self.search_space.domain_bounds)
        xhist = np.random.rand(5, d)
        yhist = f(xhist)
        self.gaussian_process = GaussianProcess(xhist, yhist)
        self.gaussian_process.train()

        while budget_minus_initialization > 0:

            # Get next sample point
            xsample, acquisition_chosen = self.get_next_point()
            acquisition_chosen_all.append(acquisition_chosen)
            ysample = f(xsample)
            xhist = np.vstack((xhist, xsample))
            yhist = np.append(yhist, ysample)
            self.gaussian_process = GaussianProcess(xhist, yhist)
            self.gaussian_process.train()
            budget_minus_initialization -= 1

        # Save BOTH acquisitions chosen as well as run information
        xhist, yhist = self.gaussian_process.get_historical_data()
        self.save_auxillary_data(acquisition_chosen_all, str(f.__name__), seed)
        self.save_bo_run(yhist, str(f.__name__), seed)


    def get_next_point(self):
        # To be implemented by each acquisition function
        pr = RolloutPortfolioEI(self.gaussian_process, self.search_space, self.horizon)
        return pr.next_point()


    def save_auxillary_data(self, acquisition_chosen_all, objective_name, seed):
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
        run_name = path + 'portfolio_aux' + str(seed) + '.csv'

        # Save data as csv to path
        with open(run_name, 'w') as file:
            writer = csv.writer(file, delimiter='\n')
            writer.writerow(acquisition_chosen_all)
