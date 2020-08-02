import numpy
from abc import ABCMeta
from scipy.spatial.distance import squareform, pdist, cdist


class CovarianceInterface(object, metaclass=ABCMeta):
  process_variance = None

  @property
  def num_hyperparameters(self):
    raise NotImplementedError()

  @property
  def dim(self):
    raise NotImplementedError()

  @property
  def translation_invariant(self):
    return NotImplemented

  def get_hyperparameters(self):
    raise NotImplementedError()

  def set_hyperparameters(self, hyperparameters):
    raise NotImplementedError()

  hyperparameters = property(get_hyperparameters, set_hyperparameters)

  def _covariance(self, x, z):
    raise NotImplementedError

  def covariance(self, x, z):
    assert len(x.shape) == len(z.shape) == 2
    n, d = x.shape
    assert n == z.shape[0]
    assert self.dim == d == z.shape[1]

    covariance_vector = self._covariance(x, z)
    assert len(covariance_vector) == n

    return self.process_variance * covariance_vector

  def _build_kernel_matrix(self, points_sampled, points_to_sample=None):
    raise NotImplementedError

  def build_kernel_matrix(self, points_sampled, points_to_sample=None, noise_variance=None):
    kernel_matrix = self.process_variance * self._build_kernel_matrix(points_sampled, points_to_sample)

    if noise_variance is not None:
      nx, nz = kernel_matrix.shape
      assert nx == nz  # Or else there should be no noise_variance term because it would be meaningless
      kernel_matrix.flat[::nx + 1] += noise_variance
    return kernel_matrix


class DifferentiableCovariance(CovarianceInterface):

  def _grad_covariance(self, x, z):
    raise NotImplementedError

  def grad_covariance(self, x, z):
    assert len(x.shape) == len(z.shape) == 2
    n, d = x.shape
    assert n == z.shape[0]
    assert self.dim == d == z.shape[1]

    grad_covariance_vector = self._grad_covariance(x, z)
    assert grad_covariance_vector.shape == (n, d)

    return self.process_variance * grad_covariance_vector

  def _hyperparameter_grad_covariance_without_process_variance(self, x, z):
    raise NotImplementedError

  def hyperparameter_grad_covariance(self, x, z):
    assert len(x.shape) == len(z.shape) == 2
    n, d = x.shape
    assert n == z.shape[0]
    assert self.dim == d == z.shape[1]

    hyperparameter_grad_covariance = numpy.empty((n, self.num_hyperparameters))
    hyperparameter_grad_covariance[:, 0] = self._covariance(x, z)
    hyperparameter_grad_covariance[:, 1:] = (
      self.process_variance * self._hyperparameter_grad_covariance_without_process_variance(x, z)
    )

    return hyperparameter_grad_covariance

  def _build_kernel_grad_tensor(self, points_sampled, points_to_sample=None):
    raise NotImplementedError()

  def build_kernel_grad_tensor(self, points_sampled, points_to_sample=None):
    return self.process_variance * self._build_kernel_grad_tensor(points_sampled, points_to_sample)

  def _build_kernel_hparam_grad_tensor_without_process_variance(self, points_sampled, points_to_sample=None):
    raise NotImplementedError()

  def build_kernel_hparam_grad_tensor(self, points_sampled, points_to_sample=None):
    n_cols, _ = points_sampled.shape
    n_rows = n_cols if points_to_sample is None else len(points_to_sample)

    kg_tensor = numpy.empty((n_rows, n_cols, self.num_hyperparameters))
    kg_tensor[:, :, 0] = self._build_kernel_matrix(points_sampled, points_to_sample)
    kg_tensor[:, :, 1:] = (
      self.process_variance *
      self._build_kernel_hparam_grad_tensor_without_process_variance(points_sampled, points_to_sample)
    )

    return kg_tensor


class RadialCovariance(CovarianceInterface):
  def __init__(self, hyperparameters):
    self._hyperparameters = None
    self._length_scales = None
    self._length_scales_squared = None
    self._length_scales_cubed = None
    self.process_variance = None
    self.set_hyperparameters(hyperparameters)

  def __str__(self):
    return f'{self.__class__.__name__}_{self.dim}({self.hyperparameters})'

  def check_hyperparameters_are_valid(self, new_hyperparameters):
    new_hyperparameters = numpy.asarray(new_hyperparameters, dtype=float)
    assert len(new_hyperparameters.shape) == 1, f'Hyperparameters should be in 1D array, not {new_hyperparameters}'
    assert numpy.all(new_hyperparameters > 0), f'For {self.__class__.__name__}, all hyperparameters must be positive'
    return new_hyperparameters

  @property
  def num_hyperparameters(self):
    return self._hyperparameters.size

  @property
  def dim(self):
    return len(self._length_scales)

  @property
  def translation_invariant(self):
    return True

  def get_hyperparameters(self):
    return numpy.copy(self._hyperparameters)

  def set_hyperparameters(self, hyperparameters):
    self._hyperparameters = self.check_hyperparameters_are_valid(hyperparameters)
    self.process_variance = self._hyperparameters[0]
    self._length_scales = numpy.copy(self._hyperparameters[1:])
    self._length_scales_squared = self._length_scales ** 2
    self._length_scales_cubed = self._length_scales ** 3

  hyperparameters = property(get_hyperparameters, set_hyperparameters)

  def eval_radial_kernel(self, distance_matrix_squared):
    raise NotImplementedError()

  def _distance_between_points(self, data, eval_points):
    data_shape = data.shape
    eval_shape = eval_points.shape
    if len(data_shape) != 2 or len(eval_shape) != 2:
      raise ValueError(f'Points must be a 2D array: data.shape = {data_shape}, eval_points.shape = {eval_shape}')
    elif data_shape != eval_shape:
      raise ValueError(f'Data size {data_shape}, Eval size {eval_shape}')
    elif data_shape[1] != self.dim:
      raise ValueError(f'Points dimension {data_shape[1]}, Covariance dimension {self.dim}')

    diff_vecs = eval_points - data
    r = numpy.sqrt(numpy.sum(numpy.power(diff_vecs / self._length_scales, 2), axis=1))
    return r, diff_vecs

  def _build_distance_matrix_squared(
    self,
    data,
    eval_points=None,
    build_diff_matrices=False,
  ):
    if eval_points is None:
      return self._build_symmetric_distance_matrix_squared(data, build_diff_matrices)
    else:
      return self._build_nonsymmetric_distance_matrix_squared(data, eval_points, build_diff_matrices)

  def _build_symmetric_distance_matrix_squared(self, data, build_diff_matrices):
    diff_mats = None
    dist_mat_sq = squareform(pdist(data / self._length_scales[numpy.newaxis, :], 'sqeuclidean'))
    if build_diff_matrices:
      diff_mats = data[:, numpy.newaxis, :] - data[numpy.newaxis, :, :]
    return dist_mat_sq, diff_mats

  def _build_nonsymmetric_distance_matrix_squared(self, data, eval_points, build_diff_matrices):
    diff_mats = None
    x = eval_points / self._length_scales[numpy.newaxis, :]
    z = data / self._length_scales[numpy.newaxis, :]
    dist_mat_sq = cdist(x, z) ** 2
    if build_diff_matrices:
      diff_mats = eval_points[:, numpy.newaxis, :] - data[numpy.newaxis, :, :]
    return dist_mat_sq, diff_mats

  def _build_kernel_matrix(self, points_sampled, points_to_sample=None):
    return self.eval_radial_kernel(self._build_distance_matrix_squared(points_sampled, points_to_sample)[0])


class DifferentiableRadialCovariance(DifferentiableCovariance, RadialCovariance):

  def eval_radial_kernel_grad(self, distance_matrix_squared, difference_matrix):
    raise NotImplementedError()

  def eval_radial_kernel_hparam_grad(self, distance_matrix_squared, difference_matrix):
    raise NotImplementedError()

  def _build_kernel_grad_tensor(self, points_sampled, points_to_sample=None):
    dm_sq, diff_mats = self._build_distance_matrix_squared(points_sampled, points_to_sample, True)
    return self.eval_radial_kernel_grad(dm_sq, diff_mats)

  def _build_kernel_hparam_grad_tensor_without_process_variance(self, points_sampled, points_to_sample=None):
    dm_sq, diff_mats = self._build_distance_matrix_squared(points_sampled, points_to_sample, True)
    return self.eval_radial_kernel_hparam_grad(dm_sq, diff_mats)


def _scale_difference_matrix(scale, difference_matrix):
  return scale[:, :, numpy.newaxis] * difference_matrix


class C4RadialMatern(DifferentiableRadialCovariance):

  def __init__(self, hyperparameters):
    super().__init__(hyperparameters)

  def eval_radial_kernel(self, distance_matrix_squared):
    r = numpy.sqrt(distance_matrix_squared)
    return (1 + r + 1.0 / 3.0 * distance_matrix_squared) * numpy.exp(-r)

  def eval_radial_kernel_grad(self, distance_matrix_squared, difference_matrix):
    r = numpy.sqrt(distance_matrix_squared)
    return _scale_difference_matrix(
      -(1.0 / 3.0) * (1 + r) * numpy.exp(-r),
      difference_matrix
    ) / self._length_scales_squared

  def eval_radial_kernel_hparam_grad(self, distance_matrix_squared, difference_matrix):
    r = numpy.sqrt(distance_matrix_squared)
    return _scale_difference_matrix(
      (1.0 / 3.0) * (1 + r) * numpy.exp(-r),
      (difference_matrix ** 2)
    ) / self._length_scales_cubed

  def _covariance(self, x, z):
    r, _ = self._distance_between_points(z, x)
    return (1 + r + 1.0 / 3.0 * r ** 2) * numpy.exp(-r)

  def _grad_covariance(self, x, z):
    r, dm = self._distance_between_points(z, x)
    r_2d = r[:, numpy.newaxis]
    return -(1.0 / 3.0) * (1 + r_2d) * numpy.exp(-r_2d) * dm / self._length_scales_squared

  def _hyperparameter_grad_covariance_without_process_variance(self, x, z):
    r, dm = self._distance_between_points(z, x)
    r_2d = r[:, numpy.newaxis]
    return (1.0 / 3.0) * (1 + r_2d) * numpy.exp(-r_2d) * (dm ** 2) / self._length_scales_cubed
