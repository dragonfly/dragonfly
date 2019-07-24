"""
  Harness to manage optimisation domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=arguments-differ

import numpy as np
from numbers import Number
from scipy.spatial.distance import cdist

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
    _check_if_valid_euc_int_bounds(bounds)
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
    _check_if_valid_euc_int_bounds(bounds)
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
    dist = self.compute_distance(point_1, point_2)
    return dist == 0 or dist < 1e-8 * self.diameter

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
    return float(point_1 != point_2)

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


class DiscreteEuclideanDomain(DiscreteDomain):
  """ Domain for Discrete Euclidean spaces. """

  def __init__(self, list_of_items):
    """ Constructor. """
    list_of_items = np.array(list_of_items)
    self.dim = list_of_items.shape[1]
    self.size = len(list_of_items)
    self.diameter = np.sqrt(self.dim) * (list_of_items.max() - list_of_items.min())
    super(DiscreteEuclideanDomain, self).__init__(list_of_items)

  def get_type(self):
    """ Returns the type of the domain. """
    return 'discrete_euclidean'

  def _get_disc_domain_type(self):
    """ Prefix for __str__. Can be overridden by a child class. """
    return "DiscEuc"

  def get_dim(self):
    """ Return the dimensions. """
    return self.dim

  @classmethod
  def compute_distance(cls, point_1, point_2):
    """ Computes the distance between point_1 and point_2. """
    return np.linalg.norm(np.array(point_1) - np.array(point_2))

  def is_a_member(self, point):
    """ Returns true if point is in the domain. """
    # Naively find the nearest point in the domain
    return cdist([point], self.list_of_items).min() < 1e-8 * self.diameter

  def members_are_equal(self, point_1, point_2):
    """ Compares two members and returns True if they are the same. """
    return self.compute_distance(point_1, point_2) < 1e-8 * self.diameter


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
    return float(sum([elem_1 != elem_2 for (elem_1, elem_2) in zip(point_1, point_2)]))

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

  def __init__(self, list_of_domains, domain_info=None):
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
    # Domain info
    self.domain_info = domain_info
    self._has_constraints = False
    if self.domain_info is not None:
      from .cp_domain_utils import get_raw_point_from_processed_point
      self.raw_name_ordering = self.domain_info.config_orderings.raw_name_ordering
      self.get_raw_point = lambda x: get_raw_point_from_processed_point(x,
                             self, self.domain_info.config_orderings.index_ordering,
                             self.domain_info.config_orderings.dim_ordering)
      if hasattr(self.domain_info, 'config_file') and \
        self.domain_info.config_file is not None:
        import os
        self.config_file = self.domain_info.config_file
        self.config_file_dir = os.path.dirname(os.path.abspath(os.path.realpath(
                                               self.domain_info.config_file)))
      if hasattr(self.domain_info, 'constraints'):
        self._has_constraints = True
        self._constraint_eval_set_up()

  def _constraint_eval_set_up(self):
    """ Set up for evaluating constraints. """
    from importlib import import_module
    import sys
    from ..utils.general_utils import evaluate_strings_with_given_variables
    self.str_constraint_evaluator = evaluate_strings_with_given_variables
    self.domain_constraints = self.domain_info.constraints
    self.num_domain_constraints = len(self.domain_constraints)
    # Separate the constraints into different types
    self.eval_as_pyfile_idxs = [idx for idx in range(self.num_domain_constraints) if
      isinstance(self.domain_constraints[idx]['constraint'], str) and
      self.domain_constraints[idx]['constraint'].endswith('.py')]
    self.eval_as_str_idxs = [idx for idx in range(self.num_domain_constraints) if
      isinstance(self.domain_constraints[idx]['constraint'], str) and
      idx not in self.eval_as_pyfile_idxs]
    self.eval_as_pyfunc_idxs = [idx for idx in range(self.num_domain_constraints) if
      hasattr(self.domain_constraints[idx]['constraint'], '__call__')]
    # Save constraints here
    self.pyfunc_constraints = [self.domain_constraints[idx]['constraint'] for idx
                              in self.eval_as_pyfunc_idxs]
    self.str_constraints = [self.domain_constraints[idx]['constraint'] for idx in
                            self.eval_as_str_idxs]
    # pyfile constraints
    self.pyfile_constraints = []
    if len(self.eval_as_pyfile_idxs) > 0:
      if not hasattr(self, 'config_file_dir'):
        raise ValueError('Constraints can be specified in a python file only when'
                         ' using a configuration file.')
      # This is relevant only if the domain is loaded via a configuration file.
      pyfile_modules = [self.domain_constraints[idx]['constraint'] for idx
                        in self.eval_as_pyfile_idxs]
      sys.path.append(self.config_file_dir)
      for pfm_file_name in pyfile_modules:
        pfm = pfm_file_name.split('.')[0]
        constraint_source_module = import_module(pfm, self.config_file_dir)
        self.pyfile_constraints.append(constraint_source_module.constraint)
      sys.path.remove(self.config_file_dir)

  def get_type(self):
    """ Returns the type of the domain. """
    return 'cartesian_product'

  def has_constraints(self):
    """ Returns True if the domain has constraints. """
    return self._has_constraints

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
    # Now check if the constraints are satisfied
    if not self.constraints_are_satisfied(point):
      return False
    return True

  def _evaluate_all_constraints(self, raw_point, name_to_pt_dict):
    """ Evaluates all constraints. """
    # Evaluate all constraints
    ret_str_all = self.str_constraint_evaluator(self.str_constraints, name_to_pt_dict)
    ret_pyfile_all = [elem(raw_point) for elem in self.pyfile_constraints]
    ret_pyfunc_all = [elem(raw_point) for elem in self.pyfunc_constraints]
    # Merge results
    ret_all = [None] * self.num_domain_constraints
    for str_idx, orig_idx in enumerate(self.eval_as_str_idxs):
      ret_all[orig_idx] = ret_str_all[str_idx]
    for pyfile_idx, orig_idx in enumerate(self.eval_as_pyfile_idxs):
      ret_all[orig_idx] = ret_pyfile_all[pyfile_idx]
    for pyfunc_idx, orig_idx in enumerate(self.eval_as_pyfunc_idxs):
      ret_all[orig_idx] = ret_pyfunc_all[pyfunc_idx]
    return ret_all

  def constraints_are_satisfied(self, point):
    """ Checks if the constraints are satisfied. """
    if hasattr(self, 'domain_constraints') and self.domain_constraints is not None:
      raw_point = self.get_raw_point(point)
      name_to_pt_dict = {k:v for (k, v) in zip(self.raw_name_ordering, raw_point)}
      ret_all = self._evaluate_all_constraints(raw_point, name_to_pt_dict)
      for idx, elem in enumerate(ret_all):
        if not isinstance(elem, (bool, np.bool, np.bool_)):
          raise ValueError(
            'Constraint %d:%s (%s) returned %s. It should return type bool.'%(idx,
            self.domain_constraints[idx][0], self.domain_constraints[idx][1], str(elem)))
      return all(ret_all)
    else:
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
    ret1 = 'CartProd(N=%d,d=%d)::[%s]'%(self.num_domains, self.dim,
                                        list_of_domains_str)
    if self.has_constraints():
      constraints_as_list_of_strs = ['%s: %s'%(elem['name'], elem['constraint'])
                                     for elem in self.domain_constraints]
      constraints_as_str = ', '.join(constraints_as_list_of_strs)
      ret2 = ',  Constraints:: %s'%(constraints_as_str)
    else:
      ret2 = ''
    return ret1 + ret2


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

def _check_if_valid_euc_int_bounds(bounds):
  """ Checks if the bounds are valid. """
  for bd in bounds:
    if bd[0] > bd[1]:
      raise ValueError('Given bound %s is not in [lower_bound, upper_bound] format.'%(
                        bd))

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

