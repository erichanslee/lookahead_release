# Separate wrapper for Gaussian process using
# MOE's GaussianProcess class as the base class (for internal use).

import numpy as np
import scipy as sp
from scipy.linalg import solve_triangular, cho_solve, cholesky
from lookahead.model.domain import ClosedInterval
from lookahead.model.historical_data import HistoricalData
from lookahead.model.domain import TensorProductDomain
from lookahead.model.covariance import C4RadialMatern as Matern
from lookahead.model._gaussian_process import GaussianProcess
from lookahead.model.parametrization import GaussianProcessLogMarginalLikelihood as GPlml
from lookahead.model.scalar_optimization import LBFGSBOptimizer as lbfgs_opt
from lookahead.model.scalar_optimization import MultistartMaximizer as ms_opt

# Updates Cholesky by adding a
# set of rows/columns, with
# data in c12, c22
def cho_addcol(L1, c12, c22):

    # Assume L1 is in scipy-compliant format
    L1 = L1[0]
    n = L1.shape[0]
    arg_bool = L1[1]
    k = c12.shape[1]
    L = np.zeros((n+k,n+k))
    L12 = solve_triangular(L1, c12, lower=True)
    if k == 1:
        L22 = np.sqrt(c22 - sp.linalg.norm(L12) ** 2)
    else:
        L22 = cholesky(c22 - np.dot(L12.T, L12))
    L[0:n, 0:n] = L1
    if k > 1:
        L[n:, 0:n] = L12.T
        L[n:, n:] = L22.T
    else:
        L[n, 0:n] = L12[:, 0]
        L[n, n] = L22
    return [L, True]

class GaussianProcessSimple(GaussianProcess):
    """
    A less complicated implementation of a Gaussian process with a Matern 5/2 kernel, to be used for BO.
    """
    def __init__(self, xtrain, ytrain):

        self.ymean = np.mean(ytrain)
        self.tikh = 1e-6
        ytrain = np.copy(ytrain) - self.ymean
        self.best_value = np.min(ytrain) # update best value
        _, self.d = xtrain.shape
        hd = HistoricalData(dim=self.d)
        hd.append_historical_data(
            points_sampled=xtrain,
            points_sampled_value=ytrain,
            points_sampled_noise_variance=self.tikh * np.ones(ytrain.shape),
        )
        params_init = np.ones((self.d+1))
        cov = Matern(params_init)

        super().__init__(
            covariance=cov,
            historical_data=hd,
            tikhonov_param=self.tikh,
        )

    def set_hypers(self, params):
        self.covariance.set_hyperparameters(params)
        super().build_precomputed_data()

    # Train hyperparameters
    def train(self):
        log_marginal_likelihood = GPlml(
            covariance=self.covariance,
            historical_data=self.historical_data,
            log_domain=True,
        )
        hp_domain = TensorProductDomain([ClosedInterval(-7, 3)] + [ClosedInterval(-3, 4)] * self.d)
        solver = ms_opt(lbfgs_opt(hp_domain, log_marginal_likelihood), num_multistarts=4)
        self.covariance.set_hyperparameters(np.exp(solver.optimize()))
        super().build_precomputed_data()

    def mean(self, xx):
        return self.ymean + self.compute_mean_of_points(xx)

    # Only difference between predict and mean is that
    # predict returns an (n,1) array instead of (n) array
    def predict(self, xx):
        temp = self.ymean + self.compute_mean_of_points(xx)
        temp = temp[:, np.newaxis]
        return temp

    def variance(self, xx):
        return self.compute_variance_of_points(xx)

    def add_points(self, XX, YY, retrain=False):
        YY = YY - self.ymean
        self.best_value = min(np.min(YY), self.best_value)  # update best value
        num_points = XX.shape[0]

        for i in range(num_points):
            self.historical_data.append_historical_data(
                points_sampled=XX[[i],:],
                points_sampled_value=YY[[i]], points_sampled_noise_variance=np.array([1e-6])
                )
        if retrain:
            self.train()
        else:
            super().build_precomputed_data()

    # Sample from posterior
    def sample(self, num_samples, points_to_sample):
        return super().draw_posterior_samples_of_points(num_samples, points_to_sample) + self.ymean

    # Update Cholesky factorization with a row/column
    def chol_update(self, XX, YY):
        K12 = self.covariance.build_kernel_matrix(
            XX, self.points_sampled
            )
        K22 = self.covariance.build_kernel_matrix(
            XX, XX
            ) + self.tikh*np.identity(XX.shape[0])
        YY = YY - self.ymean
        self.best_value = min(np.min(YY), self.best_value) # update best value
        self.historical_data.append_historical_data(
            points_sampled=XX,
            points_sampled_value=YY, points_sampled_noise_variance=1e-6*np.ones(YY.shape)
            )
        self.K_chol = cho_addcol(self.K_chol, K12, K22)
        self.K_inv_y = cho_solve(self.K_chol, self.points_sampled_value)

    # Return historical data in the form of two arrays xtrain, ytrain
    def get_historical_data(self):
        return (
            self.historical_data.points_sampled,
            self.ymean + self.historical_data.points_sampled_value,
        )

    # Sample single point, should be faster than self.sample()
    def sample_single(self, point_to_sample, random_gaussian_normal_sample=None):
        mu, var = self.mean_variance_single(point_to_sample)
        if random_gaussian_normal_sample is None:
            return np.random.normal(mu, np.sqrt(var), 1)
        else:
            mu, var = self.mean_variance_single(point_to_sample)
            return np.atleast_1d(mu + np.sqrt(var)*random_gaussian_normal_sample)

    # Compute means and variance at single point, should be vaster than self.variance()
    # returns in order mu, sigma (mean and variance respectively)
    def mean_variance_single(self, point_to_sample):
        Kx = self.covariance.build_kernel_matrix(
            self.historical_data.points_sampled, points_to_sample=point_to_sample)
        Kxx = self.covariance.build_kernel_matrix(
            point_to_sample, points_to_sample=point_to_sample
            )
        mu = np.dot(Kx, self.K_inv_y) + self.ymean
        Q = solve_triangular(
            self.K_chol[0],
            Kx.T,
            lower=self.K_chol[1],
            overwrite_b=True,
        )
        return mu.item(), Kxx.item() - np.sum(Q ** 2, axis=0).item()

    def components(self, points_to_evaluate):
        mean, var, grad_mean, grad_var = self.compute_mean_variance_grad_of_points(points_to_evaluate)
        sqrt_var = np.sqrt(var)
        grad_sqrt_var = .5 * grad_var / sqrt_var[:, np.newaxis]
        z = (self.best_value - mean) / sqrt_var
        cdf_z = sp.stats.norm.cdf(z)
        pdf_z = sp.stats.norm.pdf(z)
        return z, sqrt_var, cdf_z, pdf_z, grad_mean, grad_var, grad_sqrt_var
