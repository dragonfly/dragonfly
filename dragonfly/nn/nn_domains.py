"""
  Implements classes and methods to define domains for neural networks.
  The NNConstraintChecker class and its subclasses are used to check if a neural network
  architecture satisfies certain constriants. This is mostly needed to constrain the
  search space in NASBOT.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

import numpy as np
from copy import copy
# Local
from ..exd.domains import Domain


class NNConstraintChecker(object):
  """ A class for checking if a neural network satisfies constraints. """

  def __init__(self, max_num_layers, min_num_layers, max_mass, min_mass,
               max_in_degree, max_out_degree, max_num_edges,
               max_num_units_per_layer, min_num_units_per_layer):
    """ Constructor. """
    super(NNConstraintChecker, self).__init__()
    self.max_num_layers = max_num_layers
    self.min_num_layers = min_num_layers
    self.max_mass = max_mass
    self.min_mass = min_mass
    self.max_in_degree = max_in_degree
    self.max_out_degree = max_out_degree
    self.max_num_edges = max_num_edges
    self.max_num_units_per_layer = max_num_units_per_layer
    self.min_num_units_per_layer = min_num_units_per_layer
    self.constraint_names = ['max_num_layers', 'min_num_layers', 'max_mass',
      'min_mass', 'max_in_degree', 'max_out_degree', 'max_num_edges',
      'max_num_units_per_layer', 'min_num_units_per_layer']

  def __call__(self, nn, *args, **kwargs):
    """ Checks if the constraints are satisfied for the given nn. """
    return self.constraints_are_satisfied(nn, *args, **kwargs)

  def constraints_are_satisfied(self, nn, return_violation=False):
    """ Checks if the neural network nn satisfies the constraints. If return_violation
        is True, it returns a string representing the violation. """
    violation = ''
    if not self._check_leq_constraint(len(nn.layer_labels), self.max_num_layers):
      violation = 'too_many_layers'
    elif not self._check_geq_constraint(len(nn.layer_labels), self.min_num_layers):
      violation = 'too_few_layers'
    elif not self._check_leq_constraint(nn.get_total_mass(), self.max_mass):
      violation = 'too_much_mass'
    elif not self._check_geq_constraint(nn.get_total_mass(), self.min_mass):
      violation = 'too_little_mass'
    elif not self._check_leq_constraint(nn.get_out_degrees().max(), self.max_out_degree):
      violation = 'large_max_out_degree'
    elif not self._check_leq_constraint(nn.get_in_degrees().max(), self.max_in_degree):
      violation = 'large_max_in_degree'
    elif not self._check_leq_constraint(nn.conn_mat.sum(), self.max_num_edges):
      violation = 'too_many_edges'
    elif not self._check_leq_constraint(
                              self._finite_max_or_min(nn.num_units_in_each_layer, 1),
                              self.max_num_units_per_layer):
      violation = 'max_units_per_layer_exceeded'
    elif not self._check_geq_constraint(
                              self._finite_max_or_min(nn.num_units_in_each_layer, 0),
                              self.min_num_units_per_layer):
      violation = 'min_units_per_layer_not_exceeded'
    else:
      violation = self._child_constraints_are_satisfied(nn)
    return violation if return_violation else (violation == '')

  @classmethod
  def _check_leq_constraint(cls, value, bound):
    """ Returns true if bound is None or if value is less than or equal to bound. """
    return bound is None or (value <= bound)

  @classmethod
  def _check_geq_constraint(cls, value, bound):
    """ Returns true if bound is None or if value is greater than or equal to bound. """
    return bound is None or (value >= bound)

  def _child_constraints_are_satisfied(self, nn):
    """ Checks if the constraints of the child class are satisfied. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _finite_max_or_min(cls, iterable, is_max):
    """ Returns the max ignorning Nones, nans and infs. """
    finite_vals = [x for x in iterable if x is not None and np.isfinite(x)]
    return max(finite_vals) if is_max else min(finite_vals)


class CNNConstraintChecker(NNConstraintChecker):
  """ A class for checking if a CNN satisfies constraints. """

  def __init__(self, max_num_layers, min_num_layers, max_mass, min_mass,
               max_in_degree, max_out_degree, max_num_edges,
               max_num_units_per_layer, min_num_units_per_layer,
               max_num_2strides=None):
    """ Constructor.
      max_num_2strides is the maximum number of 2-strides (either via pooling or conv
      operations) that the image can go through in the network.
    """
    super(CNNConstraintChecker, self).__init__(
      max_num_layers, min_num_layers, max_mass, min_mass,
      max_in_degree, max_out_degree, max_num_edges,
      max_num_units_per_layer, min_num_units_per_layer)
    self.max_num_2strides = max_num_2strides
    self.constraint_names.append('max_num_2strides')

  def _child_constraints_are_satisfied(self, nn):
    """ Checks if the constraints of the child class are satisfied. """
    img_inv_sizes = [piis for piis in nn.post_img_inv_sizes if piis != 'x']
    max_post_img_inv_sizes = None if self.max_num_2strides is None \
                                  else 2**self.max_num_2strides
    violation = ''
    if not self._check_leq_constraint(max(img_inv_sizes), max_post_img_inv_sizes):
      violation = 'too_many_2strides'
    return violation


class MLPConstraintChecker(NNConstraintChecker):
  """ A class for checking if a MLP satisfies constraints. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(MLPConstraintChecker, self).__init__(*args, **kwargs)

  def _child_constraints_are_satisfied(self, nn):
    """ Checks if the constraints of the child class are satisfied. """
    return ''


# An NN Domain class -----------------------------------------------------------------
class NNDomain(Domain):
  """ Domain for Neural Network architectures. """

  def __init__(self, nn_type, constraint_checker=None):
    """ Constructor. """
    self.nn_type = nn_type
    self.constraint_checker = constraint_checker
    super(NNDomain, self).__init__()

  def get_type(self):
    """ Returns type of the domain. """
    return "neural_network"

  def get_dim(self):
    """ Return dimension. """
    return 1

  def is_a_member(self, point):
    """ Returns true if point is in the domain. """
    if not self.nn_type == point.nn_class:
      return False
    else:
      return self.constraint_checker(point)

  @classmethod
  def members_are_equal(cls, point_1, point_2):
    """ Returns true if they are equal. """
    return neural_nets_are_equal(point_1, point_2)

  def __str__(self):
    """ Returns a string representation. """
    cc_attrs = {key:getattr(self.constraint_checker, key) for
                key in self.constraint_checker.constraint_names}
    return 'NN(%s):%s'%(self.nn_type, cc_attrs)


def neural_nets_are_equal(net1, net2):
  """ Returns true if both net1 and net2 are equal.
  """
  is_true = True
  for key in net1.__dict__.keys():
    val1 = net1.__dict__[key]
    val2 = net2.__dict__[key]
    is_true = True
    if isinstance(val1, dict):
      for val_key in val1.keys():
        is_true = is_true and np.all(val1[val_key] == val2[val_key])
    elif hasattr(val1, '__iter__'):
      is_true = is_true and np.all(val1 == val2)
    else:
      is_true = is_true and val1 == val2
    if not is_true: # break here if necessary
      return is_true
  return is_true


# An API to return an NN Domain using the constraints --------------------------------
def get_nn_domain_from_constraints(nn_type, constraint_dict):
  """ nn_type is the type of the network.
      See CNNConstraintChecker, MLPConstraintChecker, NNConstraintChecker constructors
      for args and kwargs.
  """
  constraint_dict = copy(constraint_dict)
  # Specify the mandatory and optional key values
  mandatory_keys = ['max_num_layers', 'max_mass']
  optional_key_vals = [('min_num_layers', 5),
                       ('min_mass', 0),
                       ('max_out_degree', np.inf),
                       ('max_in_degree', np.inf),
                       ('max_num_edges', np.inf),
                       ('max_num_units_per_layer', 10001),
                       ('min_num_units_per_layer', 5),
                      ]
  if nn_type.startswith('cnn'):
    optional_key_vals += [('max_num_2strides', np.inf)]
  # Check if the mandatory keys exist in constraint_dict
  for mkey in mandatory_keys:
    if mkey not in constraint_dict.keys():
      raise ValueError('Must specify keys %s in constraint_dict.'%(
                       ', '.join(mandatory_keys)))
  # If an optional key does not exist, then add it
  for okey, oval in optional_key_vals:
    if okey not in constraint_dict.keys():
      constraint_dict[okey] = oval
  # Specify the constructor
  if nn_type.startswith('cnn'):
    cc_constructor = CNNConstraintChecker
  elif nn_type.startswith('mlp'):
    cc_constructor = MLPConstraintChecker
  else:
    raise ValueError('Unknown nn_type: %s.'%(nn_type))
  # Now create constraint checker object
  cc_attributes = mandatory_keys + [okv[0] for okv in optional_key_vals]
  constraint_dict_to_pass = {key: constraint_dict[key] for key in cc_attributes}
  constraint_checker = cc_constructor(**constraint_dict_to_pass)
  # Create Domain and return
  return NNDomain(nn_type, constraint_checker)

