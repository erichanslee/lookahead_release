import numpy as np
from scipy.stats import norm
from lookahead.acquisitions.acquisition_function_interface import AcquisitionFunctionInterface


class ProbabilityImprovement(AcquisitionFunctionInterface):

    def __init__(self, gaussian_process, opt_domain):
        super().__init__(gaussian_process, opt_domain)

    def evaluate_at_point_list(self, points_to_evaluate):
        mu = self.gaussian_process.mean(points_to_evaluate)
        sigma2 = self.gaussian_process.variance(points_to_evaluate)
        sigma = np.sqrt(sigma2)
        xhist, yhist = self.gaussian_process.get_historical_data()
        fmin = np.min(yhist)
        z = (fmin - mu)/sigma
        cdf_z = norm.cdf(z)
        return cdf_z
