import copy
import numpy

from scipy.linalg import solve_triangular, cho_factor, cho_solve
from lookahead.model.covariance import CovarianceInterface, DifferentiableCovariance
from lookahead.model.historical_data import HistoricalData
from lookahead.model.utils import compute_cholesky_for_gp_sampling

MINIMUM_STD_DEV_GRAD_CHOLESKY = numpy.finfo(numpy.float64).eps
MINIMUM_VARIANCE_GRAD_EI = 150 * MINIMUM_STD_DEV_GRAD_CHOLESKY ** 2
MINIMUM_KRIGING_VARIANCE = 1e-100  # Just something really small

class GaussianProcessDataInterface(object):

  def __init__(self, covariance, historical_data):
    assert isinstance(covariance, CovarianceInterface)
    assert isinstance(historical_data, HistoricalData)
    assert covariance.dim == historical_data.dim
    self.covariance = copy.deepcopy(covariance)
    self.historical_data = copy.deepcopy(historical_data)
    assert self.num_sampled > 0

  @property
  def dim(self):
    return self.historical_data.dim

  @property
  def num_sampled(self):
    return self.historical_data.num_sampled

  @property
  def differentiable(self):
    return isinstance(self.covariance, DifferentiableCovariance)

  @property
  def points_sampled(self):
    return self.historical_data.points_sampled

  @property
  def points_sampled_value(self):
    return self.historical_data.points_sampled_value

  @property
  def points_sampled_noise_variance(self):
    return self.historical_data.points_sampled_noise_variance


class GaussianProcess(GaussianProcessDataInterface):
  def __init__(self, covariance, historical_data, tikhonov_param=None):
    super().__init__(
      covariance=covariance,
      historical_data=historical_data,
    )
    self.tikhonov_param = tikhonov_param

    self.K_chol = None
    self.K_inv_y = None

    self.build_precomputed_data()

  def build_precomputed_data(self):
    if self.num_sampled == 0:
      self.K_chol = numpy.array([])
      self.K_inv_y = numpy.array([])
    else:
      if self.tikhonov_param is not None:
        noise_diag_vector = numpy.full(self.num_sampled, self.tikhonov_param)
      else:
        noise_diag_vector = self.points_sampled_noise_variance
      kernel_matrix = self.covariance.build_kernel_matrix(
        self.points_sampled,
        noise_variance=noise_diag_vector,
      )
      self.K_chol = cho_factor(kernel_matrix, lower=True, overwrite_a=True)
      self.K_inv_y = cho_solve(self.K_chol, self.points_sampled_value)

  def _compute_core_posterior_components(self, points_to_sample, option):
    K_eval = grad_K_eval = cardinal_functions_at_points_to_sample = None
    if option in ('K_eval', 'all'):
      K_eval = self.covariance.build_kernel_matrix(self.points_sampled, points_to_sample=points_to_sample)
    if option in ('grad_K_eval', 'all'):
      grad_K_eval = self.covariance.build_kernel_grad_tensor(self.points_sampled, points_to_sample=points_to_sample)
    if option == 'all':
      cardinal_functions_at_points_to_sample = cho_solve(self.K_chol, K_eval.T).T
    return K_eval, grad_K_eval, cardinal_functions_at_points_to_sample

  def compute_mean_of_points(self, points_to_sample):
    K_eval, _, _ = self._compute_core_posterior_components(points_to_sample, 'K_eval')
    return self._compute_mean_of_points(points_to_sample, K_eval)

  def _compute_mean_of_points(self, points_to_sample, K_eval):
    return numpy.dot(K_eval, self.K_inv_y)

  def compute_variance_of_points(self, points_to_sample):
    K_eval, _, _ = self._compute_core_posterior_components(points_to_sample, 'K_eval')
    return self._compute_variance_of_points(points_to_sample, K_eval)

  def _compute_variance_of_points(self, points_to_sample, K_eval, cardinal_functions_at_points_to_sample=None):
    K_x_x_array = self.covariance.covariance(points_to_sample, points_to_sample)
    if cardinal_functions_at_points_to_sample is None:
      V = solve_triangular(
        self.K_chol[0],
        K_eval.T,
        lower=self.K_chol[1],
        overwrite_b=True,
      )
      schur_complement_component = numpy.sum(V ** 2, axis=0)
    else:
      schur_complement_component = numpy.sum(K_eval * cardinal_functions_at_points_to_sample, axis=1)
    return numpy.fmax(MINIMUM_KRIGING_VARIANCE, K_x_x_array - schur_complement_component)

  def compute_mean_and_variance_of_points(self, points_to_sample):
    K_eval, _, _ = self._compute_core_posterior_components(points_to_sample, 'K_eval')
    mean = self._compute_mean_of_points(points_to_sample, K_eval)
    var = self._compute_variance_of_points(points_to_sample, K_eval)
    return mean, var

  def compute_grad_mean_of_points(self, points_to_sample):
    _, grad_K_eval, _ = self._compute_core_posterior_components(points_to_sample, 'grad_K_eval')
    return self._compute_grad_mean_of_points(points_to_sample, grad_K_eval)

  def _compute_grad_mean_of_points(self, points_to_sample, grad_K_eval):
    return numpy.einsum('ijk, j', grad_K_eval, self.K_inv_y)

  def compute_grad_variance_of_points(self, points_to_sample):
    _, grad_K_eval, cardinal_functions = self._compute_core_posterior_components(points_to_sample, 'all')
    return self._compute_grad_variance_of_points(grad_K_eval, cardinal_functions)

  def _compute_grad_variance_of_points(self, grad_K_eval, cardinal_functions_at_points_to_sample):
    if not self.covariance.translation_invariant:
      raise NotImplementedError('Not yet ready for general kernels.')

    return -2 * numpy.sum(grad_K_eval * cardinal_functions_at_points_to_sample[:, :, numpy.newaxis], axis=1)

  def compute_mean_variance_grad_of_points(self, points_to_sample):
    K_eval, grad_K_eval, cardinal_functions = self._compute_core_posterior_components(points_to_sample, 'all')

    mean = self._compute_mean_of_points(points_to_sample, K_eval)
    var = self._compute_variance_of_points(points_to_sample, K_eval, cardinal_functions)
    grad_mean = self._compute_grad_mean_of_points(points_to_sample, grad_K_eval)
    grad_var = self._compute_grad_variance_of_points(grad_K_eval, cardinal_functions)

    return mean, var, grad_mean, grad_var

  def compute_covariance_of_points(self, points_to_sample):
    K_eval_var = self.covariance.build_kernel_matrix(points_to_sample)
    if self.num_sampled == 0:
      return numpy.diag(numpy.diag(K_eval_var))

    K_eval = self.covariance.build_kernel_matrix(self.points_sampled, points_to_sample=points_to_sample)
    V = solve_triangular(
      self.K_chol[0],
      K_eval.T,
      lower=self.K_chol[1],
      overwrite_b=True,
    )

    return K_eval_var - numpy.dot(V.T, V)

  def cross_correlation_for_samples(self, points_being_sampled, eval_domain_points):
    all_k_x_vectors = self.covariance.build_kernel_matrix(self.points_sampled, points_being_sampled)
    all_k_xp_vectors = self.covariance.build_kernel_matrix(self.points_sampled, eval_domain_points)

    V_k = solve_triangular(
      self.K_chol[0],
      all_k_x_vectors.T,
      lower=self.K_chol[1],
      overwrite_b=True,
    )
    V_kprime = solve_triangular(
      self.K_chol[0],
      all_k_xp_vectors.T,
      lower=self.K_chol[1],
      overwrite_b=True,
    )
    K_x_xprime = self.covariance.build_kernel_matrix(points_being_sampled, eval_domain_points)
    cross_corr = K_x_xprime - numpy.dot(V_kprime.T, V_k)

    return cross_corr

  def draw_posterior_samples_of_points(self, num_samples, points_to_sample):
    mean = self.compute_mean_of_points(points_to_sample)
    cov = self.compute_covariance_of_points(points_to_sample)
    L = compute_cholesky_for_gp_sampling(cov)

    # z_samples is an array with shape (num_points, num_samples)
    z_samples = numpy.atleast_2d(numpy.random.normal(size=(len(mean), num_samples)))
    return mean[numpy.newaxis, :] + numpy.transpose(numpy.dot(L, z_samples))

  def draw_posterior_samples(self, num_samples):
    return self.draw_posterior_samples_of_points(num_samples, self.points_sampled)
