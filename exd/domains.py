"""
  Harness to manage optimisation domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=arguments-differ

import numpy as np
from numbers import Number

class Domain(object):
  """ Domain class. An abstract class which implements domains. """

  def get_type(self):
    """ Returns the type of the domain. """
    raise NotImplementedError('Implement in a child class.')

  def get_dim(self):
    """ Returns the dimension of the space. """
    raise NotImplementedError('Implement in a child class.')

  def is_a_member(self, point):
    """ Returns True if point is a member of this domain. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def members_are_equal(cls, point_1, point_2):
    """ Compares two members and returns True if they are the same. """
    return point_1 == point_2

  def compute_distance(self, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    raise NotImplementedError('Implement in a child class.')

  def __str__(self):
    """ Returns a string representation. """
    raise NotImplementedError('Implement in a child class.')


# Universal Domain ----------
class UniversalDomain(Domain):
  """ A universal domian. Everything is a part of this.
      Used mostly in instances where the domain is not critical for lazy coding.
  """

  def get_type(self):
    """ Returns the type of the domain. """
    return 'universal'

  def get_dim(self):
    """ Return the dimensions. """
    return None

  def is_a_member(self, _):
    """ Returns true if point is in the domain. """
    return True

  @classmethod
  def compute_distance(cls, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    raise ValueError('Distance not defined for Universal Domain.')

  def __str__(self):
    """ Returns a string representation. """
    return 'Universal Domain'


# Euclidean spaces ---------
class EuclideanDomain(Domain):
  """ Domain for Euclidean spaces. """

  def __init__(self, bounds):
    """ Constructor. """
    self.bounds = np.array(bounds)
    self.diameter = np.linalg.norm(self.bounds[:, 1] - self.bounds[:, 0])
    self.dim = len(bounds)
    super(EuclideanDomain, self).__init__()

  def get_type(self):
    """ Returns the type of the domain. """
    return 'euclidean'

  def get_dim(self):
    """ Return the dimensions. """
    return self.dim

  def is_a_member(self, point):
    """ Returns true if point is in the domain. """
    return is_within_bounds(self.bounds, point)

  def members_are_equal(self, point_1, point_2):
    """ Compares two members and returns True if they are the same. """
    return self.compute_distance(point_1, point_2) < 1e-8 * self.diameter

  @classmethod
  def compute_distance(cls, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    return np.linalg.norm(np.array(point_1) - np.array(point_2))

  def __str__(self):
    """ Returns a string representation. """
    return 'Euclidean: %s'%(_get_bounds_as_str(self.bounds))


# Integral spaces ------------
class IntegralDomain(Domain):
  """ Domain for vector valued integers. """

  def __init__(self, bounds):
    """ Constructor. """
    self.bounds = np.array(bounds, dtype=np.int)
    self.diameter = np.linalg.norm(self.bounds[:, 1] - self.bounds[:, 0])
    self.dim = len(bounds)
    super(IntegralDomain, self).__init__()

  def get_type(self):
    """ Returns the type of the domain. """
    return 'integral'

  def get_dim(self):
    """ Return the dimensions. """
    return self.dim

  def is_a_member(self, point):
    """ Returns true if point is in the domain. """
    are_ints = [isinstance(x, (int, np.int, np.int64)) for x in point]
    return all(are_ints) and is_within_bounds(self.bounds, point)

  def members_are_equal(self, point_1, point_2):
    """ Compares two members and returns True if they are the same. """
    return self.compute_distance(point_1, point_2) < 1e-8 * self.diameter

  @classmethod
  def compute_distance(cls, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    return np.linalg.norm(np.array(point_1) - np.array(point_2))

  def __str__(self):
    """ Returns a string representation. """
    return 'Integral: %s'%(_get_bounds_as_str(self.bounds))


# Discrete spaces -------------
class DiscreteDomain(Domain):
  """ A Domain for discrete objects. """

  def __init__(self, list_of_items):
    """ Constructor. """
    self.list_of_items = list_of_items
    self.size = len(list_of_items)
    super(DiscreteDomain, self).__init__()

  def get_type(self):
    """ Returns the type of the domain. """
    return 'discrete'

  def get_dim(self):
    """ Return the dimensions. """
    return 1

  def is_a_member(self, point):
    """ Returns true if point is in the domain. """
    return point in self.list_of_items

  @classmethod
  def _get_disc_domain_type(cls):
    """ Prefix for __str__. Can be overridden by a child class. """
    return "Disc"

  @classmethod
  def compute_distance(cls, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    return float(point_1 == point_2)

  def __str__(self):
    """ Returns a string representation. """
    base_str = '%s(%d)'%(self._get_disc_domain_type(), self.size)
    if self.size < 4:
      return '%s: %s'%(base_str, self.list_of_items)
    return base_str


class DiscreteNumericDomain(DiscreteDomain):
  """ A domain for discrete objects all of which are numeric. """

  def __init__(self, list_of_items):
    """ Constructor. """
    if not all_items_are_numeric(list_of_items):
      raise ValueError('list_of_items must be a list of numbers.')
    super(DiscreteNumericDomain, self).__init__(list_of_items)

  def get_type(self):
    """ Returns the type of the domain. """
    return 'discrete_numeric'

  def _get_disc_domain_type(self):
    """ Prefix for __str__. Can be overridden by a child class. """
    return "DiscNum"

  @classmethod
  def compute_distance(cls, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    return abs(point_1 - point_2)

  def is_a_member(self, point):
    """ Returns true if point is in the domain. """
    return discrete_numeric_element_is_in_list(point, self.list_of_items)


# A product of discrete spaces -----------------------------------------------------
class ProdDiscreteDomain(Domain):
  """ A product of discrete objects. """

  def __init__(self, list_of_list_of_items):
    """ Constructor. """
    self.list_of_list_of_items = list_of_list_of_items
    self.dim = len(list_of_list_of_items)
    self.size = np.prod([len(loi) for loi in list_of_list_of_items])

  def get_type(self):
    """ Returns the type of the domain. """
    return 'prod_discrete'

  def get_dim(self):
    """ Return the dimensions. """
    return self.dim

  def is_a_member(self, point):
    """ Returns true if point is in the domain. """
    if not hasattr(point, '__iter__') or len(point) != self.dim:
      return False
    ret = [elem in loi for elem, loi in zip(point, self.list_of_list_of_items)]
    return all(ret)

  def members_are_equal(self, point_1, point_2):
    """ Compares two members and returns True if they are the same. """
    elems_are_equal = [point_1[i] == point_2[i] for i in range(self.dim)]
    return all(elems_are_equal)

  @classmethod
  def _get_prod_disc_domain_type(cls):
    """ Prefix for __str__. Can be overridden by a child class. """
    return "ProdDisc"

  @classmethod
  def compute_distance(cls, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    return float(sum([elem_1 == elem_2 for (elem_1, elem_2) in zip(point_1, point_2)]))

  def __str__(self):
    """ Returns a string representation. """
    return '%s(d=%d,size=%d)'%(self._get_prod_disc_domain_type(), self.dim, self.size)


class ProdDiscreteNumericDomain(ProdDiscreteDomain):
  """ A product of discrete numeric objects. """

  def __init__(self, list_of_list_of_items):
    """ Constructor. """
    if not all_lists_of_items_are_numeric(list_of_list_of_items):
      raise ValueError('list_of_list_of_items must of a list where each element is ' +
                       'a list of numeric objects.')
    super(ProdDiscreteNumericDomain, self).__init__(list_of_list_of_items)

  def get_type(self):
    """ Returns the type of the domain. """
    return 'prod_discrete_numeric'

  def is_a_member(self, point):
    """ Returns True if point is in the domain. """
    if not hasattr(point, '__iter__') or len(point) != self.dim:
      return False
    ret = [discrete_numeric_element_is_in_list(elem, loi)
           for elem, loi in zip(point, self.list_of_list_of_items)]
    return all(ret)

  @classmethod
  def compute_distance(cls, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    return np.linalg.norm(np.array(point_1) - np.array(point_2))

  @classmethod
  def _get_prod_disc_domain_type(cls):
    """ Prefix for __str__. Can be overridden by a child class. """
    return "ProdDiscNum"


# Compound Domains ------------------------------------------
# Implementing a series of domains derived from the above

class CartesianProductDomain(Domain):
  """ The cartesian product of several domains. """

  def __init__(self, list_of_domains):
    """ Constructor.
        list_of_domains is a list of domain objects.
        An element in this domain is represented by a list whose ith element
        belongs to list_of_domains[i].
    """
    self.list_of_domains = list_of_domains
    self.num_domains = len(list_of_domains)
    try:
      self.dim = sum([dom.get_dim() for dom in self.list_of_domains])
    except TypeError:
      self.dim = None

  def get_type(self):
    """ Returns the type of the domain. """
    return 'cartesian_product'

  def get_dim(self):
    """ Returns the dimension. """
    return self.dim

  def is_a_member(self, point):
    """ Returns true if the point is in the domain. """
    if not hasattr(point, '__iter__') or len(point) != self.num_domains:
      return False
    for dom_pt, dom in zip(point, self.list_of_domains):
      if not dom.is_a_member(dom_pt): # check if each element is in the respective domain.
        return False
    return True

  def members_are_equal(self, point_1, point_2):
    """ Compares two members and returns True if they are the same. """
    for i, dom in enumerate(self.list_of_domains):
      if not dom.members_are_equal(point_1[i], point_2[i]):
        return False
    return True

  def compute_distance(self, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    return sum([dom.compute_distance(elem_1, elem_2) for (elem_1, elem_2, dom) in
                zip(point_1, point_2, self.list_of_domains)])

  def __str__(self):
    """ Returns a string representation of the domain. """
    list_of_domains_str = ', '.join([str(dom) for dom in self.list_of_domains])
    return 'CartProd(N=%d,d=%d)::[%s]'%(self.num_domains, self.dim,
                                        list_of_domains_str)


# Utilities we will need for the above ------------------------------------------
def is_within_bounds(bounds, point):
  """ Returns true if point is within bounds. point is a d-array and bounds is a
      dx2 array. bounds is expected to be an np.array object.
  """
  point = np.array(point)
  if point.shape != (bounds.shape[0],):
    return False
  above_lb = np.all((point - bounds[:, 0] >= 0))
  below_ub = np.all((bounds[:, 1] - point >= 0))
  return above_lb * below_ub

def _get_bounds_as_str(bounds):
  """ returns a string representation of bounds. """
  bounds_list = [list(b) for b in bounds]
  return str(bounds_list)

def all_items_are_numeric(list_of_items):
  """ Returns true if all items in the list are numeric. """
  for elem in list_of_items:
    if not isinstance(elem, Number):
      return False
  return True

def all_lists_of_items_are_numeric(list_of_list_of_items):
  """ Returns true if all lists in list_of_list_of_items are numeric. """
  for elem in list_of_list_of_items:
    if not all_items_are_numeric(elem):
      return False
  return True

def discrete_numeric_element_is_in_list(elem, list_of_num_elements, tol=1e-8):
  """ Returns True if elem is in list_of_num_elements. Writing this separately due to
      precision issues with Python.
  """
  if not isinstance(elem, Number):
    return False
  # Iterate through the list and check if the element exists within tolerance.
  for list_elem in list_of_num_elements:
    if abs(elem - list_elem) < tol:
      return True
  return False

