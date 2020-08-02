import numpy
import scipy.optimize

MAXIMUM_REPRESENTABLE_FINITE_FLOAT = numpy.finfo(numpy.float64).max


class MultistartMaximizer(object):
  def __init__(self, optimizer, num_multistarts=1, log_sample=False):
    assert not isinstance(optimizer, MultistartMaximizer)
    self.optimizer = optimizer
    assert num_multistarts >= 1
    self.num_multistarts = num_multistarts
    self.log_sample = log_sample

  def optimize(self, **kwargs):
    all_starts = self.optimizer.domain.generate_quasi_random_points_in_domain(self.num_multistarts, self.log_sample)

    best_point = None
    best_function_value = -numpy.inf
    for point in all_starts:
      try:
        self.optimizer.objective_function.current_point = point
        self.optimizer.optimize(**kwargs)
      except numpy.linalg.LinAlgError:
        function_value = float('nan')
        success = False
      else:
        # The negation here is required because the optimizer decorator has already negated the value
        function_value = -self.optimizer.optimization_results.fun
        success = self.optimizer.optimization_results.success

      end_point = self.optimizer.objective_function.current_point
      if not self.optimizer.domain.check_point_acceptable(end_point):
        function_value = float('nan')
        success = False

      if best_point is None or (success and function_value > best_function_value):
        if best_point is None and not success:
          best_point = point
          continue
        best_point = end_point
        best_function_value = function_value if not numpy.isnan(function_value) else best_function_value

    return best_point


class LBFGSBOptimizer(object):
  def __init__(self, domain, optimizable, approx_grad=False):
    self.domain = domain
    self.objective_function = optimizable
    self.optimization_results = None
    self.approx_grad = approx_grad
    assert self.objective_function.differentiable or self.approx_grad

  @property
  def dim(self):
    return self.domain.dim

  def _domain_as_array(self):
    return numpy.array([(interval.min, interval.max) for interval in self.domain.get_bounding_box()])

  def joint_function_gradient_eval(self, **kwargs):
    def decorated(point):
      if numpy.any(numpy.isnan(point)):
        return numpy.inf, numpy.zeros((self.dim,))

      self.objective_function.current_point = point
      value = -self.objective_function.compute_objective_function(**kwargs)
      gradient = -self.objective_function.compute_grad_objective_function(**kwargs)
      assert numpy.isfinite(value) and gradient.shape == (self.dim, )
      return value, gradient

    return decorated

  def _scipy_decorator(self, func, **kwargs):
    def decorated(point):
      self.objective_function.current_point = point
      return -func(**kwargs)

    return decorated

  def optimize(self, **kwargs):
    self.optimization_results = self._optimize_core(**kwargs)
    point = self.optimization_results.x
    self.objective_function.current_point = point

  def _optimize_core(self, **kwargs):
    options = {'eps': 1.0e-8, 'gtol': 1.0e-4, 'maxcor': 10, 'maxfun': 15000, 'ftol': 1e-4}
    if self.approx_grad:
      return scipy.optimize.minimize(
        fun=self._scipy_decorator(self.objective_function.compute_objective_function, **kwargs),
        x0=self.objective_function.current_point.flatten(),
        method='L-BFGS-B',
        bounds=self._domain_as_array(),
        options=options,
      )
    else:
      options.pop('eps')
      return scipy.optimize.minimize(
        fun=self.joint_function_gradient_eval(**kwargs),
        x0=self.objective_function.current_point.flatten(),
        method='L-BFGS-B',
        jac=True,
        bounds=self._domain_as_array(),
        options=options,
      )
