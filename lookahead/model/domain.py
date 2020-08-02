from collections import namedtuple
import copy
import numpy
import qmcpy


def generate_sobol_points(num_points, domain_bounds):
  distribution = qmcpy.Sobol(dimension=len(domain_bounds))
  pts01 = distribution.gen_samples(n=2 ** numpy.ceil(numpy.log2(num_points)))[:num_points]
  pts_scale = numpy.array([domain.length for domain in domain_bounds])
  pts_min = numpy.array([domain.min for domain in domain_bounds])
  return pts_min + pts_scale * pts01


class ClosedInterval(namedtuple('BaseInterval', ('min', 'max'))):
  __slots__ = ()

  def __repr__(self):
    return f"{self.__class__.__name__}({self.min},{self.max})"

  def __contains__(self, key):
    return self.min <= key <= self.max

  def __nonzero__(self):
    return self.min <= self.max

  def __bool__(self):
    return self.__nonzero__()

  def __eq__(self, other):
    return (
      self.__class__ == other.__class__ and
      self.min == other.min and
      self.max == other.max
    )

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash((self.min, self.max))

  @property
  def length(self):
    return self.max - self.min

  def is_inside(self, value):
    return self.__contains__(value)

  def is_valid(self):
    return self.max >= self.min


class TensorProductDomain(object):
  def __init__(self, domain_bounds):
    """Construct a TensorProductDomain with the specified bounds defined using
    a list of ClosedInterval objects.
    """
    self.domain_bounds = copy.deepcopy(domain_bounds)

    for interval in self.domain_bounds:
      if not interval.is_valid():
        raise ValueError('Tensor product region is EMPTY.')

  def __repr__(self):
    return f'TensorProductDomain({self.domain_bounds})'

  @property
  def dim(self):
    return len(self.domain_bounds)

  def check_point_inside(self, point):
    return all([interval.is_inside(point[i]) for i, interval in enumerate(self.domain_bounds)])

  def check_point_acceptable(self, point):
    assert len(point) == self.dim
    return self.check_point_inside(point)

  def get_bounding_box(self):
    return copy.copy(self.domain_bounds)

  def generate_quasi_random_points_in_domain(self, num_points, log_sample=False):
    r"""Generate quasi-random points in the domain.

    :param num_points: max number of points to generate
    :type num_points: int >= 0
    :param log_sample: sample logarithmically spaced points
    :type log_sample: bool
    :return: uniform random sampling of points from the domain
    :rtype: array of float64 with shape (num_points, dim)

    """
    domain_bounds = self.domain_bounds
    if log_sample:
      domain_bounds = [ClosedInterval(numpy.log(a), numpy.log(b)) for (a, b) in domain_bounds]

    points = generate_sobol_points(num_points, domain_bounds)
    return numpy.exp(points) if log_sample else points
