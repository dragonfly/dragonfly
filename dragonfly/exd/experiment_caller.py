"""
  Harness for calling experiments including Multi-fidelity.
  -- kandasamy@cs.cmu.edu
"""
# Convention: fidel/fidel_space always comes before point/domain --KK

# pylint: disable=abstract-class-little-used
# pylint: disable=abstract-method
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

from __future__ import print_function

from argparse import Namespace
import numpy as np
import pickle
# Local imports
from .cp_domain_utils import load_config_file, sample_from_cp_domain, \
                                get_processed_func_from_raw_func_via_config, \
                                get_processed_func_from_raw_func_for_cp_domain
from .domains import EuclideanDomain
from .exd_utils import EVAL_ERROR_CODE
from ..utils.general_utils import map_to_cube, map_to_bounds

_FIDEL_TOL = 1e-2


class CalledMultiFidelOnSingleFidelCaller(Exception):
  """ An exception to handle calling multi-fidelity experiments on single-fidelity
      callers.
  """
  def __init__(self, exp_caller):
    """ Constructor. """
    err_msg = ('ExperimentCaller %s is not a multi-fidelity caller. ' + \
               'Please use eval_single or eval_multiple.')%(str(exp_caller))
    super(CalledMultiFidelOnSingleFidelCaller, self).__init__(err_msg)


# Base class for Experiments ============================================================
class ExperimentCaller(object):
  """ Function Caller class. """

  # Methods needed for construction and set up ------------------------------------------
  def __init__(self, experiment, domain, descr='',
               noise_type='no_noise', noise_scale=None,
               fidel_space=None, fidel_cost_func=None, fidel_to_opt=None,
               config=None):
    """ Constructor. """
    self.experiment = experiment
    self.domain = domain
    self.descr = descr
    self.config = config
    self._set_up_noise(noise_type, noise_scale)
    self._mf_set_up(fidel_space, fidel_cost_func, fidel_to_opt)

  def _set_up_noise(self, noise_type, noise_scale):
    """ Sets up noise. """
    self.noise_type = noise_type
    if noise_type == 'no_noise':
      self._is_noisy = False
      self.noise_scale = None
      self.noise_adder_single = None
      self.noise_adder_multiple = None
    else:
      self._is_noisy = True
      self.noise_scale = noise_scale
      self.noise_type = noise_type
      self._set_up_noisy_evals()

  def _set_up_noisy_evals(self):
    """ Set up noisy evaluations. """
    raise NotImplementedError('Implement this in a noisy evaluator.')

  def _mf_set_up(self, fidel_space, fidel_cost_func, fidel_to_opt):
    """ Sets up multi-fidelity. """
    if any([elem is None for elem in [fidel_space, fidel_cost_func, fidel_to_opt]]):
      # If any of these are None, assert that all of them are. Otherwise through an
      # error.
      try:
        assert fidel_space is None
        assert fidel_cost_func is None
        assert fidel_to_opt is None
      except AssertionError:
        raise ValueError('Either all fidel_space, fidel_cost_func, and fidel_to_opt' +
                         'should be None or should be not-None')
      self._is_mf = False
    else:
      self.fidel_space = fidel_space
      self.fidel_cost_func = fidel_cost_func
      self.fidel_to_opt = fidel_to_opt
      self._is_mf = True

  def is_noisy(self):
    """ Returns true if noisy. """
    return self._is_noisy

  def is_mf(self):
    """ Returns True if this is multi-fidelity. """
    return self._is_mf

  def is_fidel_to_opt(self, fidel):
    """ Returns True if fidel is the fidel_to_opt. Naively tests for equality, but can
        be overridden by a child class. """
    if not self.is_mf():
      raise CalledMultiFidelOnSingleFidelCaller()
    return self.fidel_space.members_are_equal(fidel, self.fidel_to_opt)

  # Evaluation --------------------------------------------------------------------------
  def get_noisy_value(self, true_val):
    """ Gets the noisy evaluation. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _pickle_dump_to_result_file(cls, qinfo, file_handle):
    """ Writes to the pickle file. """
    pickle.dump(qinfo, file_handle)

  def _eval_single_common_wrap_up(self, true_val, qinfo, noisy, caller_eval_cost):
    """ Wraps up the evaluation by adding noise and adding info to qinfo.
        Common to both single and mutli-fidelity eval experiments.
    """
    if true_val == EVAL_ERROR_CODE:
      val = EVAL_ERROR_CODE
    elif noisy and self.is_noisy():
      val = self.get_noisy_value(true_val)
    else:
      val = true_val
    # put everything into qinfo
    qinfo = Namespace() if qinfo is None else qinfo
    qinfo.true_val = true_val
    qinfo.val = val
    qinfo.caller_eval_cost = caller_eval_cost
    # Writing to a file
    if hasattr(qinfo, 'result_file') and qinfo.result_file is not None:
      if isinstance(qinfo.result_file, str):
        file_handle = open(qinfo.result_file, 'wb')
        self._pickle_dump_to_result_file(qinfo, file_handle)
        file_handle.close()
      elif isinstance(qinfo.result_file, file):
        self._pickle_dump_to_result_file(qinfo, qinfo.result_file)
      else:
        raise ValueError('qinfo.result_file should a string or file object.')
    return val, qinfo

  def _get_true_val_from_experiment_at_point(self, point):
    """ Returns the true value from the experiment. Can be overridden by child class if
        experiment is represented differently. """
    assert self.domain.is_a_member(point)
    return self.experiment(point)

  def _get_true_val_from_experiment_at_fidel_point(self, fidel, point):
    """ Returns the true value from the experiment. Can be overridden by child class if
        experiment is represented differently. """
    assert self.fidel_space.is_a_member(fidel)
    assert self.domain.is_a_member(point)
    return self.experiment(fidel, point)

  # Single fidelity evaluations --------------------------------------------
  def eval_single(self, point, qinfo=None, noisy=True):
    """ Evaluates experiment at a single point point. If the experiment_caller is noisy by
        default, we can obtain a noiseless evaluation by setting noisy to be False.
    """
    if self.is_mf(): # if multi-fidelity call the experiment at fidel_to_opt
      return self.eval_at_fidel_single(self.fidel_to_opt, point, qinfo, noisy)
    else:
      if qinfo is None:
        qinfo = Namespace()
      true_val = self._get_true_val_from_experiment_at_point(point)
      qinfo.point = point
      val, qinfo = self._eval_single_common_wrap_up(true_val, qinfo, noisy, None)
      return val, qinfo

  def eval_multiple(self, points, qinfos=None, noisy=True):
    """ Evaluates multiple points. If the experiment_caller is noisy by
        default, we can obtain a noiseless evaluation by setting noisy to be False.
    """
    qinfos = [None] * len(points) if qinfos is None else qinfos
    ret_vals = []
    ret_qinfos = []
    for i in range(len(points)):
      val, qinfo = self.eval_single(points[i], qinfos[i], noisy)
      ret_vals.append(val)
      ret_qinfos.append(qinfo)
    return ret_vals, ret_qinfos

  # Multi-fidelity evaluations -------------------------------------------------
  def eval_at_fidel_single(self, fidel, point, qinfo=None, noisy=True):
    """ Evaluates experiment at a single (fidel, point). If the experiment_caller is
        noisy by default, we can obtain a noiseless evaluation by setting noisy to be
        False.
    """
    if not self.is_mf():
      raise CalledMultiFidelOnSingleFidelCaller(self)
    if qinfo is None:
      qinfo = Namespace()
    true_val = self._get_true_val_from_experiment_at_fidel_point(fidel, point)
    cost_at_fidel = self.fidel_cost_func(fidel)
    qinfo.fidel = fidel
    qinfo.point = point
    qinfo.cost_at_fidel = cost_at_fidel
    val, qinfo = self._eval_single_common_wrap_up(true_val, qinfo, noisy,
                                                  cost_at_fidel)
    return val, qinfo

  def eval_at_fidel_multiple(self, fidels, points, qinfos=None, noisy=True):
    """ Evaluates experiment at a multiple (fidel, point) pairs.
        If the experiment_caller is noisy by
        default, we can obtain a noiseless evaluation by setting noisy to be False.
    """
    qinfos = [None] * len(points) if qinfos is None else qinfos
    ret_vals = []
    ret_qinfos = []
    for i in range(len(points)):
      val, qinfo = self.eval_at_fidel_single(fidels[i], points[i], qinfos[i], noisy)
      ret_vals.append(val)
      ret_qinfos.append(qinfo)
    return ret_vals, ret_qinfos

  # Eval from qinfo
  def eval_from_qinfo(self, qinfo, *args, **kwargs):
    """ Evaluates from a qinfo object. Returns the qinfo. """
    if not hasattr(qinfo, 'fidel'):
      _, qinfo = self.eval_single(qinfo.point, qinfo, *args, **kwargs)
      return qinfo
    else:
      _, qinfo = self.eval_at_fidel_single(qinfo.fidel, qinfo.point, qinfo,
                                            *args, **kwargs)
      return qinfo

  # Cost -----------------------------------------------------------------------------
  def _get_true_cost_from_fidel_cost_func_at_fidel(self, fidel):
    """ Returns the true value from the experiment. Can be overridden by child class if
        fidel_cost_func is represented differently. """
    return float(self.fidel_cost_func(fidel))

  def cost_single(self, fidel):
    """ Returns the cost at a single fidelity. """
    return self._get_true_cost_from_fidel_cost_func_at_fidel(fidel)

  def cost_multiple(self, fidels):
    """ Returns the cost at multiple fidelities. """
    return [self._get_true_cost_from_fidel_cost_func_at_fidel(fidel) for fidel in fidels]

  def cost_ratio_single(self, fidel_numerator, fidel_denominator=None):
    """ Returns the cost ratio. If fidel_denominator is None, we set it to be
        fidel_to_opt.
    """
    if fidel_denominator is None:
      fidel_denominator = self.fidel_to_opt
    return self.cost_single(fidel_numerator) / self.cost_single(fidel_denominator)

  def cost_ratio_multiple(self, fidels_numerator, fidel_denominator=None):
    """ Returns the cost ratio for multiple fidels. If fidel_denominator is None,
        we set it to be fidel_to_opt.
    """
    if fidel_denominator is None:
      fidel_denominator = self.fidel_to_opt
    numerator_costs = self.cost_multiple(fidels_numerator)
    denom_cost = self.cost_single(fidel_denominator)
    ret = [x/denom_cost for x in numerator_costs]
    return ret

  # Other methods ---------------------------------------------------------------------
  def get_candidate_fidels(self, domain_point, filter_by_cost=True, *args, **kwargs):
    """ Returns candidate fidelities at domain_point.
        If filter_by_cost is true returns only those for which cost is larger than
        fidel_to_opt.
    """
    if not self.is_mf():
      raise CalledMultiFidelOnSingleFidelCaller(self)
    candidate_fidels = self._child_get_candidate_fidels(domain_point, filter_by_cost,
                                                        *args, **kwargs)
    candidate_fidels.append(self.fidel_to_opt)
    return candidate_fidels

  def _child_get_candidate_fidels(self, domain_point, filter_by_cost=True,
                                  *args, **kwargs):
    """ Returns candidate fidelities at domain_point for the child class.
        If filter_by_cost is true returns only those for which cost is larger than
        fidel_to_opt.
    """
    raise NotImplementedError('Implement in a child class.')

  def get_candidate_fidels_and_cost_ratios(self, domain_point, filter_by_cost=True,
                                           *args, **kwargs):
    """ Returns candidate fidelities and the cost ratios.
        If filter_by_cost is true returns only those for which cost is larger than
        fidel_to_opt.
    """
    candidates = self.get_candidate_fidels(domain_point, filter_by_cost=False,
                                           add_fidel_to_opt=False, *args, **kwargs)
    fidel_cost_ratios = self.cost_ratio_multiple(candidates)
    if filter_by_cost:
      filtered_idxs = np.where(np.array(fidel_cost_ratios) < 1.0)[0]
      candidates = [candidates[idx] for idx in filtered_idxs]
      fidel_cost_ratios = [fidel_cost_ratios[idx] for idx in filtered_idxs]
      # But re-add fidel_to_opt.
      candidates.append(self.fidel_to_opt)
      fidel_cost_ratios.append(1.0)
    return candidates, fidel_cost_ratios

  def get_information_gap(self, fidels):
    """ Returns the information gap w.r.t fidel_to_opt. """
    raise NotImplementedError('Implement in a child class.')


# A class for evaluation multiple functions ==========================================
# That is, the output of each function is a real number.
class MultiFunctionCaller(ExperimentCaller):
  """ An Experiment Caller specifically for evaluating a collection of functions. """

  def __init__(self, funcs, domain, descr='',
               argmax=None, maxval=None, argmin=None, minval=None,
               noise_type='no_noise', noise_scale=None,
               fidel_space=None, fidel_cost_func=None, fidel_to_opt=None,
               *args, **kwargs):
    """ Constructor. """
    self.funcs = funcs
    self.argmax = argmax
    self.maxval = maxval
    self.argmin = argmin
    self.minval = minval
    experiment_caller = self._get_experiment_caller()
    super(MultiFunctionCaller, self).__init__(experiment_caller, domain, descr,
      noise_type=noise_type, noise_scale=noise_scale,
      fidel_space=fidel_space, fidel_cost_func=fidel_cost_func, fidel_to_opt=fidel_to_opt,
      *args, **kwargs)

  def _get_experiment_caller(self):
    """ Returns the experiment caller. If funcs is a list, then the function caller
        returns a list of element for each function in funcs.
        Otherwise, it just returns self.funcs which is assumed to be callable.
    """
    if isinstance(self.funcs, list):
      self._has_many_functions = True
      self.num_funcs = len(self.funcs)
      return lambda *_args, **_kwargs: [float(f(*_args, **_kwargs)) for f in self.funcs]
    elif isinstance(self.funcs, tuple) and len(self.funcs) == 2:
      self._has_many_functions = True
      self.num_funcs = self.funcs[1]
      return lambda *_args, **_kwargs: [float(val) for val
                                        in self.funcs[0](*_args, **_kwargs)]
    else:
      self._has_many_functions = False
      return lambda *_args, **_kwargs: float(self.funcs(*_args, **_kwargs))

  def _set_up_noisy_evals(self):
    """ Sets up noisy evaluations. """
    if self.noise_type == 'gauss':
      self.noise_adder_single = lambda: self.noise_scale * np.random.normal()
      self.noise_adder_multiple = \
        lambda num_samples: self.noise_scale * np.random.normal((num_samples,))
    elif self.noise_type == 'uniform':
      self.noise_adder_single = lambda: self.noise_scale * (np.random.random() - 0.5)
      self.noise_adder_multiple = \
        lambda num_samples: self.noise_scale * (np.random.normal((num_samples,)) - 0.5)
    else:
      raise NotImplementedError(('Not implemented %s noise yet')%(self.noise_type))

  def get_noisy_value(self, true_val):
    """ Gets the noisy evaluation. """
    if self._has_many_functions:
      return [tv + self.noise_adder_single() for tv in true_val]
    else:
      return true_val + self.noise_adder_single()


# A function caller where the domain is Euclidean ======================================
class EuclideanMultiFunctionCaller(MultiFunctionCaller):
  """ An experiment caller on Euclidean spaces. """

  def __init__(self, funcs, raw_domain, descr='', vectorised=False,
               to_normalise_domain=True,
               raw_argmax=None, maxval=None, raw_argmin=None, minval=None,
               noise_type='no_noise', noise_scale=None,
               raw_fidel_space=None, fidel_cost_func=None, raw_fidel_to_opt=None,
               *args, **kwargs):
    """ Constructor. """
    # pylint: disable=too-many-arguments
    # Prelims
    if hasattr(raw_domain, '__iter__'):
      raw_domain = EuclideanDomain(raw_domain)
    if hasattr(raw_fidel_space, '__iter__'):
      raw_fidel_space = EuclideanDomain(raw_fidel_space)
    self.vectorised = vectorised
    self.domain_is_normalised = to_normalise_domain
    # Set domain and and argmax/argmin
    self.raw_domain = raw_domain
    self.raw_argmax = raw_argmax
    argmax = None if raw_argmax is None else self.get_normalised_domain_coords(raw_argmax)
    self.raw_argmin = raw_argmin
    argmin = None if raw_argmin is None else self.get_normalised_domain_coords(raw_argmin)
    domain = EuclideanDomain([[0, 1]] * raw_domain.dim) if to_normalise_domain \
             else raw_domain
    # Set fidel_space
    if raw_fidel_space is not None:
      self.raw_fidel_space = raw_fidel_space
      self.raw_fidel_to_opt = raw_fidel_to_opt
      fidel_space = EuclideanDomain([[0, 1]] * raw_fidel_space.dim) \
                    if to_normalise_domain else raw_fidel_space
      fidel_to_opt = self.get_normalised_fidel_coords(raw_fidel_to_opt)
      self.fidel_space_diam = np.linalg.norm(
        fidel_space.bounds[:, 1] - fidel_space.bounds[:, 0])
    else:
      fidel_space = None
      fidel_to_opt = None
    # Now call the super constructor
    super(EuclideanMultiFunctionCaller, self).__init__(funcs=funcs, domain=domain,
                 descr=descr, argmax=argmax, maxval=maxval, argmin=argmin, minval=minval,
                 noise_type=noise_type, noise_scale=noise_scale,
                 fidel_space=fidel_space, fidel_cost_func=fidel_cost_func,
                 fidel_to_opt=fidel_to_opt, *args, **kwargs)

  def is_fidel_to_opt(self, fidel):
    """ Returns True if fidel is the fidel_to_opt. """
    return np.linalg.norm(fidel - self.fidel_to_opt) < _FIDEL_TOL * self.fidel_space_diam

  # Methods required for normalising coordinates -----------------------------------------
  def get_normalised_fidel_coords(self, Z):
    """ Maps points in the original fidelity space to the cube. """
    if self.domain_is_normalised:
      return map_to_cube(Z, self.raw_fidel_space.bounds)
    else:
      return Z

  def get_normalised_domain_coords(self, X):
    """ Maps points in the original domain to the cube. """
    if self.domain_is_normalised:
      return map_to_cube(X, self.raw_domain.bounds)
    else:
      return X

  def get_normalised_fidel_domain_coords(self, Z, X):
    """ Maps points in the original space to the cube. """
    ret_Z = None if Z is None else self.get_normalised_fidel_coords(Z)
    ret_X = None if X is None else self.get_normalised_domain_coords(X)
    return ret_Z, ret_X

  def get_raw_fidel_coords(self, Z):
    """ Maps points from the fidelity space cube to the original space. """
    if self.domain_is_normalised:
      return map_to_bounds(Z, self.raw_fidel_space.bounds)
    else:
      return Z

  def get_raw_domain_coords(self, X):
    """ Maps points from the domain cube to the original space. """
    if self.domain_is_normalised:
      return map_to_bounds(X, self.raw_domain.bounds)
    else:
      return X

  def get_raw_fidel_domain_coords(self, Z, X):
    """ Maps points from the cube to the original spaces. """
    ret_Z = None if Z is None else self.get_raw_fidel_coords(Z)
    ret_X = None if X is None else self.get_raw_domain_coords(X)
    return ret_Z, ret_X

  # Override _get_true_val_from_experiment_at_point and
  # _get_true_val_from_experiment_at_fidel_point
  # so as to account for normalisation of the domain and/or fidel_space
  def _get_true_val_from_experiment_at_point(self, point):
    """ Evaluates experiment by first unnormalising point. """
    raw_dom_coords = self.get_raw_domain_coords(point)
    assert self.raw_domain.is_a_member(raw_dom_coords)
    if self.vectorised:
      raw_dom_coords = raw_dom_coords.reshape((-1, 1))
    return self.experiment(raw_dom_coords)

  def _get_true_val_from_experiment_at_fidel_point(self, fidel, point):
    """ Evaluates experiment by first unnormalising point. """
    raw_fidel_coords = self.get_raw_fidel_coords(fidel)
    assert self.raw_fidel_space.is_a_member(raw_fidel_coords)
    raw_dom_coords = self.get_raw_domain_coords(point)
    assert self.raw_domain.is_a_member(raw_dom_coords)
    if self.vectorised:
      raw_dom_coords = raw_dom_coords.reshape((-1, 1))
      raw_fidel_coords = raw_fidel_coords.reshape((-1, 1))
    return self.experiment(self.get_raw_fidel_coords(fidel),
                           self.get_raw_domain_coords(point))

  def _get_true_cost_from_fidel_cost_func_at_fidel(self, fidel):
    """ Evaluates fidel_cost_func by unnormalising fidel. """
    raw_fidel_coords = self.get_raw_fidel_coords(fidel)
    assert self.raw_fidel_space.is_a_member(raw_fidel_coords)
    if self.vectorised:
      raw_fidel_coords = raw_fidel_coords.reshape((-1, 1))
    return float(self.fidel_cost_func(raw_fidel_coords))

  def _child_get_candidate_fidels(self, domain_point, filter_by_cost=True,
                                  *args, **kwargs):
    """ Returns candidate fidelities at domain_point.
        If filter_by_cost is true returns only those for which cost is larger than
        fidel_to_opt.
    """
    if self.fidel_space.dim == 1:
      norm_candidates = np.linspace(0, 1, 100).reshape((-1, 1))
    elif self.fidel_space.dim == 2:
      num_per_dim = 25
      norm_candidates = (np.indices((num_per_dim, num_per_dim)).reshape(2, -1).T + 0.5) \
                        / float(num_per_dim)
    elif self.fidel_space.dim == 3:
      num_per_dim = 10
      cand_1 = (np.indices((num_per_dim, num_per_dim, num_per_dim)).reshape(3, -1).T
                + 0.5) / float(num_per_dim)
      cand_2 = np.random.random((1000, self.fidel_space.dim))
      norm_candidates = np.vstack((cand_1, cand_2))
    else:
      norm_candidates = np.random.random((4000, self.fidel_space.dim))
    # Now unnormalise if necessary
    if self.domain_is_normalised:
      candidates = norm_candidates
    else:
      candidates = map_to_bounds(candidates, self.fidel_space.bounds)
    if filter_by_cost:
      fidel_costs = self.cost_multiple(candidates)
      filtered_idxs = np.where(np.array(fidel_costs) <
                               self.cost_single(self.fidel_to_opt))[0]
      candidates = candidates[filtered_idxs, :]
    # Finally, always add the highest fidelity
    candidates = list(candidates)
    return candidates

  def get_information_gap(self, fidels):
    """ Returns distances to fidel_to_opt. """
    if not self.is_mf():
      raise CalledMultiFidelOnSingleFidelCaller(self)
    return [np.linalg.norm(fidel - self.fidel_to_opt)/self.fidel_space_diam \
            for fidel in fidels]


class CPMultiFunctionCaller(MultiFunctionCaller):
  """ Cartesian product multi-function caller. """

  def __init__(self, funcs, domain, descr='', raw_funcs=None, domain_orderings=None,
               argmax=None, maxval=None, argmin=None, minval=None,
               noise_type='no_noise', noise_scale=None,
               fidel_space=None, fidel_cost_func=None, fidel_to_opt=None,
               fidel_space_orderings=None, *args, **kwargs):
    """ Constructor. """
    self.raw_funcs = raw_funcs
    self.domain_orderings = domain_orderings
    self.fidel_space_orderings = fidel_space_orderings
    super(CPMultiFunctionCaller, self).__init__(funcs, domain, descr,
      argmax=argmax, maxval=maxval, argmin=argmin, minval=minval,
      noise_type=noise_type, noise_scale=noise_scale,
      fidel_space=fidel_space, fidel_cost_func=fidel_cost_func, fidel_to_opt=fidel_to_opt,
      *args, **kwargs)
    self._set_up_point_reconfiguration()

  def _set_up_point_reconfiguration(self):
    """ Sets up reconfiguring points if domain orderings etc. are not None. """
    if self.domain_orderings is not None or self.fidel_space_orderings is not None:
      from .cp_domain_utils import get_raw_point_from_processed_point
    if self.domain_orderings is not None:
      self.get_raw_domain_point_from_processed = \
        lambda pt: get_raw_point_from_processed_point(pt, self.domain,
                     self.domain_orderings.index_ordering,
                     self.domain_orderings.dim_ordering)
    if self.fidel_space_orderings is not None:
      self.get_raw_fidel_from_processed = \
        lambda pt: get_raw_point_from_processed_point(pt, self.fidel_space,
                     self.fidel_space_orderings.index_ordering,
                     self.fidel_space_orderings.dim_ordering)

  def _child_get_candidate_fidels(self, domain_point, filter_by_cost=True,
                                  *args, **kwargs):
    """ Returns candidate fidelities at domain_point for the child class.
        If filter_by_cost is true returns only those for which cost is larger than
        fidel_to_opt.
    """
    num_samples = np.clip(100 * self.fidel_space.get_dim(), 100, 8000)
    return sample_from_cp_domain(self.fidel_space, num_samples,
                                 euclidean_sample_type='latin_hc',
                                 integral_sample_type='latin_hc')

  def get_information_gap(self, fidels):
    """ Returns the information gap to fidel_to_opt. """
    if not self.is_mf():
      raise CalledMultiFidelOnSingleFidelCaller(self)
    return [self.fidel_space.compute_distance(self.fidel_to_opt, fidel) for fidel
            in fidels]


# A wrapper on top of MultiFunctionCaller for single function callers ==================
class FunctionCaller(MultiFunctionCaller):
  """ A function caller on Euclidean spaces. """

  def __init__(self, func, *args, **kwargs):
    """ Constructor. """
    self.func = func
    super(FunctionCaller, self).__init__(func, *args, **kwargs)


class EuclideanFunctionCaller(EuclideanMultiFunctionCaller):
  """ An experiment caller on Euclidean spaces. """

  def __init__(self, func, *args, **kwargs):
    """ Constructor. """
    self.func = func
    super(EuclideanFunctionCaller, self).__init__(func, *args, **kwargs)


class CPFunctionCaller(CPMultiFunctionCaller):
  """ Cartesian product function caller. """

  def __init__(self, func, domain, descr='', raw_func=None, *args, **kwargs):
    """ Constructor. """
    self.func = func
    self.raw_func = raw_func
    super(CPFunctionCaller, self).__init__(func, domain, descr, raw_func, *args, **kwargs)


# An API to obtain a MultiFunctionCaller on a CPDomain from cp_domain ==================
def get_multifunction_caller_from_config(raw_funcs, domain_config_file, descr='',
                                         raw_fidel_cost_func=None, **kwargs):
  """ Returns a multi-function caller from the raw functions and configuration file. """
  # pylint: disable=no-member
  # pylint: disable=maybe-no-member
  if isinstance(domain_config_file, str):
    config = load_config_file(domain_config_file)
  else:
    config = domain_config_file
  if isinstance(raw_funcs, (list, tuple)):
    is_multi_function = True
  else:
    is_multi_function = False
    raw_funcs = [raw_funcs]
  funcs = [get_processed_func_from_raw_func_via_config(rf, config) for rf in raw_funcs]
  # Check multi-fidelity
  if hasattr(config, 'fidel_space') and config.fidel_space is not None:
    fidel_cost_func = get_processed_func_from_raw_func_for_cp_domain(raw_fidel_cost_func,
      config.fidel_space, config.fidel_space_orderings.index_ordering,
      config.fidel_space_orderings.dim_ordering)
    fidel_space_orderings = config.fidel_space_orderings
    fidel_to_opt = config.fidel_to_opt
    fidel_space = config.fidel_space
  else:
    fidel_cost_func = None
    fidel_space_orderings = None
    fidel_to_opt = None
    fidel_space = None
  # Now return
  if is_multi_function:
    return CPMultiFunctionCaller(funcs, config.domain, descr, raw_funcs,
             config.domain_orderings, fidel_space=fidel_space,
             fidel_cost_func=fidel_cost_func, fidel_to_opt=fidel_to_opt,
             fidel_space_orderings=fidel_space_orderings, config=config, **kwargs)
  else:
    return CPFunctionCaller(funcs[0], config.domain, descr, raw_funcs[0],
             config.domain_orderings, fidel_space=fidel_space,
             fidel_cost_func=fidel_cost_func, fidel_to_opt=fidel_to_opt,
             fidel_space_orderings=fidel_space_orderings, config=config, **kwargs)

