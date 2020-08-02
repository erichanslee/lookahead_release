from lookahead.acquisitions.expected_improvement import ExpectedImprovement
from lookahead.model.gaussian_process import GaussianProcessSimple

import numpy as np

class BayesOptAcquisitionOptimizer:
    """
    Optimizes acquisition function w/ bayesian optimization
    """

    def __init__(self, gaussian_process, acquisition_handle, opt_domain):
        self.gaussian_process = gaussian_process
        self.acquisition_handle = acquisition_handle
        self.opt_domain = opt_domain

    def get_initial_design(self):

        # maximize the expected improvement first
        ei = ExpectedImprovement(self.gaussian_process, self.opt_domain)
        x_ei_opt = ei.next_point()
        X = self.opt_domain.generate_quasi_random_points_in_domain(5)
        X = np.vstack([X, x_ei_opt])
        return X

    def get_sample_point(self):
        """
        Get next sample point through BO, using 100d BO iterations. We want to maximize, meaning we have to multiple by
        negative when calling the handle
        """
        budget = self.gaussian_process.d*5
        initial_design = self.get_initial_design()
        X = initial_design
        Y = -1*self.acquisition_handle(X)

        for i in range(budget):

            # Normalize Y
            Y_train = np.copy(Y)
            Y_train = (Y_train - np.mean(Y_train)) / np.std(Y_train)
            gp = GaussianProcessSimple(X, Y_train)
            gp.train()

            ei = ExpectedImprovement(gp, self.opt_domain)
            xx = ei.next_point()
            yy = -1*self.acquisition_handle(xx)
            X = np.vstack([X, xx])
            Y = np.concatenate([Y, yy])

        idx = np.argmin(Y)
        return X[[idx], :]

