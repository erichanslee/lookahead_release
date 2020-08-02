# -*- coding: utf-8 -*-
import numpy as np
from lookahead.acquisitions.acquisition_function_interface import AcquisitionFunctionInterface
MINIMUM_VARIANCE_KG = 2.2250738585072014e-308  # np.finfo(np.float64).tiny

KG_MC_ITERATIONS = 1000
KG_DOMAIN_NUM_SAMPLES = 666


class KnowledgeGradient(AcquisitionFunctionInterface):
  """
  Implementation of Knowledge Gradient (see, e.g., Frazier et al 2008)
  """
  def __init__(self, gaussian_process, opt_domain):
    super().__init__(gaussian_process, opt_domain)
    eval_domain_points = opt_domain.generate_quasi_random_points_in_domain(KG_DOMAIN_NUM_SAMPLES)
    self.num_mc_iterations = eval_domain_points.shape[0]
    xhist, yhist = self.gaussian_process.get_historical_data()
    self.best_value = np.min(yhist)
    self.eval_domain_points = eval_domain_points
    self.eval_domain_points_mean_estimate = self.gaussian_process.compute_mean_of_points(self.eval_domain_points)

  @property
  def differentiable(self):
    return False

  def evaluate_at_point_list(self, points_to_evaluate):
    num_points = len(points_to_evaluate)
    sigma_star = np.sqrt(self.gaussian_process.compute_variance_of_points(points_to_evaluate))
    pred_stdev = self.gaussian_process.cross_correlation_for_samples(points_to_evaluate, self.eval_domain_points)
    curr_best = self.best_value
    std_norm_rv = np.random.normal(size=(self.num_mc_iterations, num_points))

    scaled_norm_rvs = std_norm_rv[:, np.newaxis] * (pred_stdev / sigma_star)
    post_estimate = self.eval_domain_points_mean_estimate[:, np.newaxis] + scaled_norm_rvs
    # min applied for each mc sample over the eval_domain_points, then mean over all MC samples
    value_kg = curr_best - np.mean(np.min(post_estimate, axis=1), axis=0)
    return value_kg

  def compute_acquisition_function(self, point_to_sample):
    return self.evaluate_at_point_list(np.atleast_2d(point_to_sample))[0]

  def compute_grad_acquisition_function(self, point_to_sample):
    raise NotImplementedError('The gradient of the KG is not implemented yet')

  def next_point(self):
    return self.next_point_grid()
