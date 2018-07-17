"""
  Harness to manage optimisation domains.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=arguments-differ
# pylint: disable=abstract-class-not-used
# pylint: disable=abstract-class-little-used

from builtins import object
import numpy as np
# Local
from gp.kernel import SEKernel
from utils.oper_utils import random_maximise, direct_ft_maximise, pdoo_maximise


_EUCLIDEAN_DFLT_OPT_METHOD = 'direct'
_NN_DFLT_OPT_METHOD = 'ga'


class Domain(object):
  """ Domain class. An abstract class which implements domains. """

  def __init__(self, dflt_domain_opt_method):
    """ Constructor. """
    super(Domain, self).__init__()
    self.dflt_domain_opt_method = dflt_domain_opt_method

  def maximise_obj(self, opt_method, obj, num_evals, *args, **kwargs):
    """ Optimises the objective and returns it. """
    if opt_method == 'dflt_domain_opt_method':
      opt_method = self.dflt_domain_opt_method
    return self._child_maximise_obj(opt_method, obj, num_evals, *args, **kwargs)

  def _child_maximise_obj(self, opt_method, obj, num_evals, *args, **kwargs):
    """ Child class implementation for optimising an objective. """
    raise NotImplementedError('Implement in a child class.')

  def get_default_kernel(self, *args, **kwargs):
    """ Get the default kernel for this domain. """
    raise NotImplementedError('Implement in a child class.')

  def get_type(self):
    """ Returns the type of the domain. """
    raise NotImplementedError('Implement in a child class.')

  def get_dim(self):
    """ Returns the dimension of the space. """
    raise NotImplementedError('Implement in a child class.')

  def is_a_member(self, point):
    """ Returns True if point is a member of this domain. """
    raise NotImplementedError('Implement in a child class.')


# Euclidean spaces -------------------------------------------------------------------
class EuclideanDomain(Domain):
  """ Domain for Euclidean spaces. """

  def __init__(self, bounds):
    """ Constructor. """
    self.bounds = np.array(bounds)
    self.dim = len(bounds)
    super(EuclideanDomain, self).__init__('rand')

  def _child_maximise_obj(self, opt_method, obj, num_evals):
    """ Child class implementation for optimising an objective. """
    if opt_method == 'rand':
      return self._rand_maximise_obj(obj, num_evals)
    elif opt_method == 'direct':
      return self._direct_maximise_obj(obj, num_evals)
    elif opt_method == 'pdoo':
      return self._pdoo_maximise_obj(obj, num_evals)
    else:
      raise ValueError('Unknown opt_method=%s for EuclideanDomain'%(opt_method))

  def _rand_maximise_obj(self, obj, num_evals):
    """ Maximise with random evaluations. """
    if num_evals is None:
      lead_const = 10 * min(5, self.dim)**2
      num_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)), 2000, 3e4)
    opt_val, opt_pt, _ = random_maximise(obj, self.bounds, num_evals)
    return opt_val, opt_pt

  def _direct_maximise_obj(self, obj, num_evals):
    """ Maximise with direct. """
    if num_evals is None:
      lead_const = 10 * min(5, self.dim)**2
      num_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)), 2000, 3e4)
    opt_val, opt_pt, _ = direct_ft_maximise(obj, self.bounds, num_evals)
    return opt_val, opt_pt

  def _pdoo_maximise_obj(self, obj, num_evals):
    """ Maximise with direct. """
    if num_evals is None:
      lead_const = 10 * min(5, self.dim)**2
      num_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)), 2000, 3e4)
    opt_val, opt_pt, _ = pdoo_maximise(obj, self.bounds, num_evals)
    return opt_val, opt_pt

  def get_default_kernel(self, range_Y):
    """ Returns the default (SE) kernel. """
    return SEKernel(self.dim, range_Y/4.0, dim_bandwidths=0.05*np.sqrt(self.dim))

  def get_type(self):
    """ Returns the type of the domain. """
    return 'euclidean'

  def get_dim(self):
    """ Return the dimensions. """
    return self.dim

  def is_a_member(self, point):
    """ Returns true if point is in the domain. """
    point = np.array(point)
    above_lb = np.all((point - self.bounds[:, 0] >= 0))
    below_ub = np.all((self.bounds[:, 1] - point >= 0))
    return above_lb * below_ub

