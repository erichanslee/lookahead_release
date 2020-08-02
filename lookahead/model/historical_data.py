import pprint
import numpy

class SamplePoint(object):
  def __init__(self, point, value=None, noise_variance=0.0):
    if not (noise_variance >= 0.0 and numpy.isfinite(noise_variance)):
      raise ValueError(f'noise_variance = {noise_variance} must be non-negative and finite!')
    if value is None or numpy.isinf(value):
      raise ValueError(f'value = {value} must be finite (nan allowed)!')
    if any(~numpy.isfinite(point)):
      raise ValueError(f'point = {point} must be finite!')

    self.point = numpy.copy(point)
    self.value = value
    self.noise_variance = noise_variance

  @classmethod
  def from_dict(cls, d):
    return SamplePoint(d['point'], d['value'], d['value_var'])

  @property
  def as_tuple(self):
    return self.point, self.value, self.noise_variance

  def __getitem__(self, item):
    if item == 'point':
      return self.point
    elif item == 'value':
      return self.value
    elif item == 'value_var':
      return self.noise_variance
    else:
      raise ValueError(f"key {item} not recognized, must be one of ('point', 'value', 'value_var')")

  def __repr__(self):
    """Pretty print this object as a dict."""
    return pprint.pformat(dict(zip(('point', 'value', 'value_var'), self.as_tuple)))

  def json_payload(self):
    return {
      'point': list(self.point),  # json needs a list
      'value': self.value,
      'value_var': self.noise_variance,
    }


class HistoricalData(object):
  @staticmethod
  def convert_list_of_sample_points_to_arrays(sample_points):
    if len(sample_points) == 0:
      return numpy.array([[]]), numpy.array([]), numpy.array([])
    return tuple(numpy.array(v) for v in zip(*[p.as_tuple for p in sample_points]))

  def __init__(self, dim, sample_points=None):
    if sample_points is None:
      sample_points = []

    self.dim = dim
    self.points_sampled = numpy.empty((0, self.dim))
    self.points_sampled_value = numpy.empty(0)
    self.points_sampled_noise_variance = numpy.empty(0)

    self.append_sample_points(sample_points)

  def __str__(self, pretty_print=True):
    """String representation of this HistoricalData object.
    """
    if pretty_print:
      return pprint.pformat(self.to_list_of_sample_points())
    return '\n'.join([
      repr(self.points_sampled),
      repr(self.points_sampled_value),
      repr(self.points_sampled_noise_variance)
    ])

  def append_lies(self, points_being_sampled, lie_value, lie_value_var):
    self.append_sample_points([SamplePoint(point, lie_value, lie_value_var) for point in points_being_sampled])

  def append_historical_data(self, points_sampled, points_sampled_value, points_sampled_noise_variance):
    """Append lists of points_sampled, their values, and their noise variances to the data members of this class.
    """
    if points_sampled.size == 0:
      return

    assert len(points_sampled.shape) == 2
    assert len(points_sampled_value.shape) == len(points_sampled_noise_variance.shape) == 1
    assert len(points_sampled) == len(points_sampled_value) == len(points_sampled_noise_variance)
    assert points_sampled.shape[1] == self.dim

    self.points_sampled = numpy.append(self.points_sampled, points_sampled, axis=0)
    self.points_sampled_value = numpy.append(self.points_sampled_value, points_sampled_value)
    self.points_sampled_noise_variance = numpy.append(self.points_sampled_noise_variance, points_sampled_noise_variance)

  def append_sample_points(self, sample_points):
    self.append_historical_data(*self.convert_list_of_sample_points_to_arrays(sample_points))

  def to_list_of_sample_points(self):
    return [
      SamplePoint(*a) for a in zip(self.points_sampled, self.points_sampled_value, self.points_sampled_noise_variance)
    ]

  @property
  def num_sampled(self):
    """Return the number of sampled points."""
    return self.points_sampled.shape[0]
