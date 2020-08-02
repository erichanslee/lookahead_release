import numpy as np
from scipy.stats import norm
from lookahead.acquisitions.acquisition_function_interface import AcquisitionFunctionInterface

class ExpectedImprovement(AcquisitionFunctionInterface):

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
        pdf_z = norm.pdf(z)
        return sigma * np.fmax(0.0, z * cdf_z + pdf_z)

    def evaluate_grad_at_point_list(self, points_to_evaluate):
        return self._evaluate_grad_at_point_list_core(self.gaussian_process.components(points_to_evaluate))

    def _evaluate_at_point_list_core(self, core_components):
        z, sqrt_var, cdf_z, pdf_z, _, _, _ = core_components
        return sqrt_var * np.fmax(0.0, z * cdf_z + pdf_z)

    def _evaluate_grad_at_point_list_core(self, core_components):
        _, _, cdf_z, pdf_z, grad_mean, _, grad_sqrt_var = core_components
        return grad_sqrt_var * pdf_z[:, np.newaxis] - grad_mean * cdf_z[:, np.newaxis]

    def joint_function_gradient_eval(self, points_to_evaluate):
        points_to_evaluate = np.atleast_2d(points_to_evaluate)
        core_components = self.gaussian_process.components(points_to_evaluate)
        ei = self._evaluate_at_point_list_core(core_components)
        ei_grad = self._evaluate_grad_at_point_list_core(core_components)[0, :]
        return -ei, -ei_grad

    def next_point(self):
        return self.next_point_grad()
