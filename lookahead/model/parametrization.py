import numpy
import scipy.linalg

from lookahead.model._gaussian_process import GaussianProcess, GaussianProcessDataInterface

DEFAULT_TIKHONOV_PARAMETER = 1.0e-10


class GaussianProcessLogMarginalLikelihood(GaussianProcessDataInterface):
  def __init__(
    self,
    covariance,
    historical_data,
    log_domain=False,
    tikhonov_parameter=DEFAULT_TIKHONOV_PARAMETER,
  ):
    super().__init__(covariance, historical_data)
    self.log_domain = log_domain
    self.tikhonov_parameter = tikhonov_parameter
    self.gp = GaussianProcess(self.covariance, self.historical_data, tikhonov_param=self.tikhonov_parameter)

  @property
  def num_hyperparameters(self):
    return self.covariance.num_hyperparameters

  @property
  def tikhonov_param(self):
    return self.gp.tikhonov_param

  def get_hyperparameters(self):
    hyperparameters = self.covariance.hyperparameters
    return numpy.log(hyperparameters) if self.log_domain else hyperparameters

  def set_hyperparameters(self, hyperparameters):
    hp_linear_domain = numpy.exp(hyperparameters) if self.log_domain else hyperparameters
    # Don't pass the noise term in covariance.hyperparameters since we pass it as a separate param
    self.covariance.hyperparameters = hp_linear_domain[:self.dim + 1]
    self.gp = GaussianProcess(self.covariance, self.historical_data, self.tikhonov_parameter)

  hyperparameters = property(get_hyperparameters, set_hyperparameters)
  current_point = hyperparameters

  def compute_objective_function(self):
    y_Pb = self.gp.points_sampled_value
    Kinvy_Pb = self.gp.K_inv_y
    L = self.gp.K_chol[0]
    log_likelihood = numpy.dot(y_Pb, Kinvy_Pb) + 2 * numpy.sum(numpy.log(L.diagonal()))
    return -log_likelihood

  def compute_grad_objective_function(self):
    grad_hyperparameter_cov_tensor = self.covariance.build_kernel_hparam_grad_tensor(self.points_sampled)
    grad_log_marginal = numpy.empty(self.num_hyperparameters)
    Kinvy_Pb = self.gp.K_inv_y
    K_chol = self.gp.K_chol
    for k in range(self.num_hyperparameters):
      dK = grad_hyperparameter_cov_tensor[:, :, k]
      dKKinvy_Pb = numpy.dot(dK, Kinvy_Pb)
      grad_log_marginal[k] = -numpy.dot(Kinvy_Pb, dKKinvy_Pb)
      grad_log_marginal[k] += numpy.trace(scipy.linalg.cho_solve(K_chol, dK, overwrite_b=True))

    log_scaling = numpy.exp(self.hyperparameters) if self.log_domain else 1.0
    return -grad_log_marginal * log_scaling
