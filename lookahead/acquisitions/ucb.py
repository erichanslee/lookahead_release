import numpy as np
from lookahead.acquisitions.acquisition_function_interface import AcquisitionFunctionInterface
BASE_KAPPA = 1

class UpperConfidenceBound(AcquisitionFunctionInterface):
    def __init__(self, gaussian_process, opt_domain, **kwargs):
        super().__init__(gaussian_process, opt_domain)
        self.kappa = kwargs.get('base_kappa', BASE_KAPPA)

    def evaluate_at_point_list(self, points_to_evaluate):
        mean, variance = self.gaussian_process.compute_mean_and_variance_of_points(points_to_evaluate)
        return -mean - self.kappa * np.sqrt(variance)

    def joint_function_gradient_eval(self, points_to_evaluate):
        points_to_evaluate = np.atleast_2d(points_to_evaluate)
        mean, sqrt_var, _, _, grad_mean, _, grad_sqrt_var  = self.gaussian_process.components(points_to_evaluate)
        ucb = -mean + self.kappa * sqrt_var
        ucb_grad = -grad_mean - self.kappa * grad_sqrt_var
        return -ucb, -ucb_grad
