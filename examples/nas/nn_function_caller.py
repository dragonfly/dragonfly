"""
  A function caller to work with MLPs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used
# pylint: disable=no-member
# pylint: disable=signature-differs

from copy import deepcopy
import numpy as np
from time import sleep
# Local imports
from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.exd.exd_core import EVAL_ERROR_CODE
from dragonfly.utils.reporters import get_reporter

_DEBUG_ERROR_PROB = 0.1
# _DEBUG_ERROR_PROB = 0.0


def _get_cpfc_args_from_config(config):
  """ Return arguments as a dict. """
  # pylint: disable=maybe-no-member
  if isinstance(config, str):
    from dragonfly.exd.cp_domain_utils import load_config_file
    config = load_config_file(config)
  ret = {'domain':config.domain,
         'domain_orderings':config.domain_orderings,
         'fidel_space':config.fidel_space,
         'fidel_to_opt':config.fidel_to_opt,
         'fidel_space_orderings':config.fidel_space_orderings}
  return ret


class NNFunctionCaller(CPFunctionCaller):
  """ Function Caller for NN evaluations. """

  def __init__(self, config, train_params, descr='', debug_mode=False,
               reporter='silent'):
    """ Constructor for train params. """
    constructor_args = _get_cpfc_args_from_config(config)
    super(NNFunctionCaller, self).__init__(None, descr=descr,
            fidel_cost_func=self._fidel_cost, **constructor_args)
    self.train_params = deepcopy(train_params)
    self.debug_mode = debug_mode
    self.reporter = get_reporter(reporter)

  @classmethod
  def is_mf(cls):
    """ Returns True if Multi-fidelity. """
    return True

  def _fidel_cost(self, fidel):
    """ Get fidel cost. """
    raw_fidel = self.get_raw_fidel_from_processed(fidel)
    return self._raw_fidel_cost(raw_fidel)

  @classmethod
  def _raw_fidel_cost(cls, raw_fidel):
    """ Fidelity cost function. """
    # The first (and only) fidelity argument is the number of batch loops, and the cost
    # is linear in the number of loops. The multiplication by a scalar does not change
    # relative costs.
    return raw_fidel[0]/100.0

  def eval_at_fidel_single(self, fidel, point, qinfo, noisy=False):
    """ Evaluate the function here. This method overrides eval_at_fidel_single
        from dragonfly.exd.experiment_caller.py but we need information in qinfo
        to carry out the NN evaluation.
    """
    # Dragonfly stores them in a different format to the configuration specifed.
    # First convert them back to the "raw" format.
    raw_fidel = self.get_raw_fidel_from_processed(fidel)
    raw_point = self.get_raw_domain_point_from_processed(point)
    # Obtain true value
    true_val = self._func_wrapper(raw_fidel, raw_point, qinfo)
    cost_at_fidel = self._raw_fidel_cost(raw_fidel)
    # Prep qinfo
    qinfo.fidel = fidel
    qinfo.point = point
    qinfo.cost_at_fidel = cost_at_fidel
    val, qinfo = self._eval_single_common_wrap_up(true_val, qinfo, noisy,
                                                  cost_at_fidel)
    return val, qinfo

  def _func_wrapper(self, raw_fidel, raw_point, qinfo):
    """ Evaluates the function here - mostly a wrapper to decide between
        the synthetic function vs the real function. """
    # pylint: disable=broad-except
    if self.debug_mode:
      ret = self._eval_synthetic_function(raw_point)
    else:
      try:
        ret = self._eval_validation_score(raw_fidel, raw_point, qinfo)
      except Exception as exc:
        self.reporter.writeln('Exception when evaluating %s at fidel %s: %s.'%(
                              str(raw_fidel), str(raw_point), exc))
        ret = EVAL_ERROR_CODE
    return ret

  @classmethod
  def _eval_synthetic_function(cls, raw_point):
    """ Evaluates the synthetic function. """
    result = len(str(raw_point)) + np.random.random()
    sleep_time = 2 + 3 * np.random.random()
    sleep(sleep_time)
    if np.random.random() < _DEBUG_ERROR_PROB:
      # For debugging, return an error code with small probability
      return EVAL_ERROR_CODE
    else:
      return result

  def _eval_validation_score(self, raw_fidel, raw_point, qinfo):
    """ Evaluates the validation score. """
    # Design your API here. You can use self.training_params to store anything
    # additional you need.
    raise NotImplementedError('Implement this for specific application.')

