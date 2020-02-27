"""
  Implements some instances of a random optimiser.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from argparse import Namespace
import numpy as np
# Local imports
from ..exd.cp_domain_utils import get_processed_func_from_raw_func_for_cp_domain, \
                                load_cp_domain_from_config_file
from ..exd import domains
from ..exd.exd_utils import get_euclidean_initial_qinfos, get_cp_domain_initial_qinfos
from ..exd.exd_core import mf_exd_args
from ..exd.experiment_caller import CPFunctionCaller
from ..exd.cp_domain_utils import sample_from_cp_domain
from ..exd.worker_manager import SyntheticWorkerManager
from .blackbox_optimiser import BlackboxOptimiser, blackbox_opt_args, \
                                   CalledMFOptimiserWithSFCaller
from ..utils.option_handler import load_options
from ..utils.reporters import get_reporter
from ..utils.general_utils import map_to_bounds

random_optimiser_args = blackbox_opt_args
euclidean_random_optimiser_args = random_optimiser_args
cp_random_optimiser_args = random_optimiser_args
mf_euclidean_random_optimiser_args = euclidean_random_optimiser_args + mf_exd_args
mf_cp_random_optimiser_args = cp_random_optimiser_args + mf_exd_args


def random_sample_from_cp_domain_wrapper(num_pts, domain, reporter):
  """ A wrapper to sample from a domain. """
  ret = []
  num_pts_to_request = num_pts
  num_tries = 0
  while len(ret) < num_pts:
    ret.extend(sample_from_cp_domain(domain, num_pts_to_request,
                                     verbose_constraint_satisfaction=False))
    num_pts_to_request *= 2
    num_tries += 1
    if len(ret) == 0:
      if num_tries % 10 == 0:
        error_msg = ('Could not randomly sample from %s domain despite %d tries with ' +
                     'up to %d candidates.')%(domain, num_tries, num_pts_to_request)
        reporter.writeln(error_msg)
      if num_tries >= 51:
        error_msg = ('Could not randomly sample from domain %s despite %d tries with up' +
                     ' to %d candidates. Quitting now')%(domain, num_tries,
                                                         num_pts_to_request)
        raise ValueError(error_msg)
  return ret[:num_pts]


# Base class for Random Optimisation -----------------------------------------------
class RandomOptimiser(BlackboxOptimiser):
  """ A class which optimises using random evaluations. """
  #pylint: disable=attribute-defined-outside-init
  #pylint: disable=abstract-method

  # Constructor.
  def __init__(self, func_caller, worker_manager=None, options=None, 
               reporter=None, ask_tell_mode=False):
    """ Constructor. """
    options = load_options(random_optimiser_args, partial_options=options)
    super(RandomOptimiser, self).__init__(func_caller, worker_manager, model=None,
                                          options=options, reporter=reporter, 
                                          ask_tell_mode=ask_tell_mode)

  def _opt_method_set_up(self):
    """ Any set up specific to otptimisation. """
    pass

  def _get_method_str(self):
    """ Returns a string describing the method. """
    return 'rand'

  def _add_data_to_model(self, qinfos):
    """ Adds data to model. """
    pass

  def _child_build_new_model(self):
    """ Builds a new model. """
    pass

# Random optimiser for Euclidean spaces --------------------------------------------
class EuclideanRandomOptimiser(RandomOptimiser):
  """ A class which optimises in Euclidean spaces using random evaluations. """

  def __init__(self, func_caller, worker_manager=None, options=None, 
               reporter=None, ask_tell_mode=False):
    options = load_options(euclidean_random_optimiser_args, partial_options=options)
    super(EuclideanRandomOptimiser, self).__init__(func_caller, worker_manager,
                                                   options=options, reporter=reporter, 
                                                   ask_tell_mode=ask_tell_mode)
  
  def is_an_mf_method(self):
    """ Returns False since this is not a MF method. """
    return False

  def _determine_next_query(self):
    """ Determines the next query. """
    qinfo = Namespace(point=map_to_bounds(np.random.random(self.domain.dim),
                                          self.domain.bounds))
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of queries. """
    qinfos = [self._determine_next_query() for _ in range(batch_size)]
    return qinfos

  def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
    """ Returns initial qinfos. """
    return get_euclidean_initial_qinfos(self.options.init_method, num_init_evals,
                                        self.domain.bounds)

# Multi-fidelity Random Optimiser for Euclidean Spaces -------------------------------
class MFEuclideanRandomOptimiser(RandomOptimiser):
  """ A class which optimises in Euclidean spaces using random evaluations and
      multi-fidelity.
  """

  # Constructor.
  def __init__(self, func_caller, worker_manager=None, call_fidel_to_opt_prob=0.25,
               ask_tell_mode=False, *args, **kwargs):
    """ Constructor.
        call_fidel_to_opt_prob is the probability with which we will choose
        fidel_to_opt as the fidel.
    """
    options = load_options(mf_euclidean_random_optimiser_args, partial_options=kwargs.pop("options", None))
    super(MFEuclideanRandomOptimiser, self).__init__(func_caller, worker_manager,
                                                     options=options,
                                                     ask_tell_mode=ask_tell_mode,
                                                     *args, **kwargs)
    self.call_fidel_to_opt_prob = call_fidel_to_opt_prob
    if not func_caller.is_mf():
      raise CalledMFOptimiserWithSFCaller(self, func_caller)

  def is_an_mf_method(self):
    """ Returns Truee since this is a MF method. """
    return True

  def _determine_next_query(self):
    """ Determines the next query. """
    # An internal function which returns the next fidelity.
    def _get_next_fidel():
      """ Returns the next fidelity. """
      if np.random.random() <= self.call_fidel_to_opt_prob:
        return self.func_caller.fidel_to_opt
      else:
        return np.random.random(self.fidel_space.dim)
    # Create and return qinfo
    qinfo = Namespace(point=np.random.random(self.domain.dim),
                      fidel=_get_next_fidel())
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of queries. """
    qinfos = [self._determine_next_query() for _ in range(batch_size)]
    return qinfos

  def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
    """ Returns initial qinfos. """
    return get_euclidean_initial_qinfos(self.options.init_method, num_init_evals,
             self.domain.bounds, self.options.fidel_init_method, self.fidel_space.bounds,
             self.func_caller.fidel_to_opt,
             self.options.init_set_to_fidel_to_opt_with_prob)


# A random optimiser in Cartesian product spaces --------------------------------------
class CPRandomOptimiser(RandomOptimiser):
  """ A random optimiser for cartesian product domains. """
  def __init__(self, func_caller, worker_manager=None, options=None, 
              reporter=None, ask_tell_mode=False):
    options = load_options(cp_random_optimiser_args, partial_options=options)
    super(CPRandomOptimiser, self).__init__(func_caller, worker_manager,
                                            options=options, reporter=reporter, 
                                            ask_tell_mode=ask_tell_mode)

  def is_an_mf_method(self):
    """ Returns False since it is not a False method. """
    return False

  def _determine_next_query(self):
    """ Determines the next query. """
    qinfo = Namespace(
      point=random_sample_from_cp_domain_wrapper(1, self.domain, self.reporter)[0])
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of queries. """
    qinfos = [self._determine_next_query() for _ in range(batch_size)]
    return qinfos

  def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
    """ Returns initial qinfos. """
    return get_cp_domain_initial_qinfos(self.domain, num_init_evals,
                                        dom_euclidean_sample_type='latin_hc',
                                        dom_integral_sample_type='latin_hc',
                                        dom_nn_sample_type='rand', *args, **kwargs)


class MFCPRandomOptimiser(RandomOptimiser):
  """ A class which optimises in Cartesian product spaces using random evaluations. """

  def __init__(self, func_caller, worker_manager, call_fidel_to_opt_prob=0.25,
               *args, **kwargs):
    """ Constructor. """
    super(MFCPRandomOptimiser, self).__init__(func_caller, worker_manager,
                                              *args, **kwargs)
    self.call_fidel_to_opt_prob = call_fidel_to_opt_prob
    if not func_caller.is_mf():
      raise CalledMFOptimiserWithSFCaller(self, func_caller)

  def is_an_mf_method(self):
    """ Returns False since it is not a False method. """
    return True

  def _determine_next_query(self):
    """ Determines next query. """
    # An internal function which returns the next fidelity.
    def _get_next_fidel():
      """ Returns the next fidelity. """
      if np.random.random() <= self.call_fidel_to_opt_prob:
        return self.func_caller.fidel_to_opt
      else:
        return random_sample_from_cp_domain_wrapper(1, self.fidel_space, self.reporter)[0]
    # Create and return qinfo
    qinfo = Namespace(
      point=random_sample_from_cp_domain_wrapper(1, self.domain, self.reporter)[0],
      fidel=_get_next_fidel())
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of queries. """
    qinfos = [self._determine_next_query() for _ in range(batch_size)]
    return qinfos

  def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
    """ Returns initial qinfos. """
    return get_cp_domain_initial_qinfos(self.domain, num_init_evals,
             self.fidel_space, self.func_caller.fidel_to_opt,
             set_to_fidel_to_opt_with_prob=self.call_fidel_to_opt_prob,
             dom_euclidean_sample_type='latin_hc',
             dom_integral_sample_type='latin_hc',
             dom_nn_sample_type='rand',
             fidel_space_euclidean_sample_type='latin_hc',
             fidel_space_integral_sample_type='latin_hc',
             fidel_space_nn_sample_type='rand', *args, **kwargs)


# APIs for random optimisation ===========================================================

# An API for single fidelity optimisation
def random_optimiser_from_func_caller(func_caller, worker_manager, max_capital,
                                      mode='asy', options=None, reporter='default'):
  """ Creates an appropriate RandomOptimiser Object and optimises the function. """
  reporter = get_reporter(reporter)
  if isinstance(func_caller.domain, domains.EuclideanDomain):
    optimiser_constructor = EuclideanRandomOptimiser
    dflt_list_of_options = euclidean_random_optimiser_args
  elif isinstance(func_caller.domain, domains.CartesianProductDomain):
    optimiser_constructor = CPRandomOptimiser
    dflt_list_of_options = cp_random_optimiser_args
  else:
    raise ValueError('Random optimiser not implemented for domain of type %s.'%(
                     type(func_caller.domain)))
  # Load options and modify where necessary
  options = load_options(dflt_list_of_options, partial_options=options)
  options.mode = mode
  from ..exd.worker_manager import RealWorkerManager, SyntheticWorkerManager
  if isinstance(worker_manager, RealWorkerManager):
    options.capital_type = 'realtime'
  elif isinstance(worker_manager, SyntheticWorkerManager):
    options.capital_type = 'return_value'
  # Create optimiser
  optimiser = optimiser_constructor(func_caller, worker_manager, options, reporter)
  # optimise and return
  return optimiser.optimise(max_capital)

def cp_random_optimiser_from_raw_args(raw_func, domain_config_file, *args, **kwargs):
  """ A random optimiser on Cartesian product spaces. """
  # pylint: disable=no-member
  cp_dom, orderings = load_cp_domain_from_config_file(domain_config_file)
  proc_func = get_processed_func_from_raw_func_for_cp_domain(
                raw_func, cp_dom, orderings.index_ordering, orderings.dim_ordering)
  func_caller = CPFunctionCaller(proc_func, cp_dom, raw_func=raw_func,
                                 domain_orderings=orderings)
  return random_optimiser_from_func_caller(func_caller, *args, **kwargs)


# An API for multi-fidelity optimisation
def mf_random_optimiser_from_func_caller(func_caller, worker_manager, max_capital,
                                         mode='asy', options=None, reporter='default',
                                         *args, **kwargs):
  """ Creates a MF EuclideanRandomOptimiser Object and optimises the function. """
  reporter = get_reporter(reporter)
  if isinstance(func_caller.domain, domains.EuclideanDomain) and \
     isinstance(func_caller.fidel_space, domains.EuclideanDomain):
    optimiser_constructor = MFEuclideanRandomOptimiser
    dflt_list_of_options = mf_euclidean_random_optimiser_args
  elif isinstance(func_caller.domain, domains.CartesianProductDomain) and \
    isinstance(func_caller.fidel_space, domains.CartesianProductDomain):
    optimiser_constructor = MFCPRandomOptimiser
    dflt_list_of_options = mf_cp_random_optimiser_args
  else:
    raise ValueError(('MF Random optimiser not implemented for (domain, fidel_space) '
                      + 'of types (%s, %s).')%(
                     type(func_caller.domain), type(func_caller.fidel_space)))
  # Load options
  options = load_options(dflt_list_of_options, partial_options=options)
  options.mode = mode
  # Create optimiser
  optimiser = optimiser_constructor(func_caller, worker_manager, options=options,
                                    reporter=reporter, *args, **kwargs)
  # optimise and return
  return optimiser.optimise(max_capital)

