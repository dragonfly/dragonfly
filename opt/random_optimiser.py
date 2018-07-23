"""
  Implements some instances of a random optimiser.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from argparse import Namespace
import numpy as np
# Local imports
import exd.domains as domains
from exd.exd_utils import get_euclidean_initial_qinfos
from exd.exd_core import mf_exd_args
from opt.blackbox_optimiser import BlackboxOptimiser, blackbox_opt_args, \
                                   CalledMFOptimiserWithSFCaller
from utils.option_handler import load_options
from utils.reporters import get_reporter
from utils.general_utils import map_to_bounds

random_optimiser_args = blackbox_opt_args
euclidean_random_optimiser_args = random_optimiser_args
mf_euclidean_random_optimiser_args = euclidean_random_optimiser_args + mf_exd_args


# Base class for Random Optimisation -----------------------------------------------
class RandomOptimiser(BlackboxOptimiser):
  """ A class which optimises using random evaluations. """
  #pylint: disable=attribute-defined-outside-init
  #pylint: disable=abstract-method

  # Constructor.
  def __init__(self, func_caller, worker_manager, options=None, reporter=None):
    """ Constructor. """
    self.reporter = get_reporter(reporter)
    if options is None:
      options = load_options(random_optimiser_args, reporter=reporter)
    super(RandomOptimiser, self).__init__(func_caller, worker_manager, model=None,
                                          options=options, reporter=self.reporter)

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

  def _get_initial_qinfos(self, num_init_evals):
    """ Returns initial qinfos. """
    return get_euclidean_initial_qinfos(self.options.init_method, num_init_evals,
                                        self.domain.bounds)


# Multi-fidelity Random Optimiser for Euclidean Spaces -------------------------------
class MFEuclideanRandomOptimiser(RandomOptimiser):
  """ A class which optimises in Euclidean spaces using random evaluations and
      multi-fidelity.
  """

  def is_an_mf_method(self):
    """ Returns Truee since this is a MF method. """
    return True

  # Constructor.
  def __init__(self, func_caller, worker_manager, call_fidel_to_opt_prob=0.25,
               *args, **kwargs):
    """ Constructor.
        call_fidel_to_opt_prob is the probability with which we will choose
        fidel_to_opt as the fidel.
    """
    super(MFEuclideanRandomOptimiser, self).__init__(func_caller, worker_manager,
                                                     *args, **kwargs)
    self.call_fidel_to_opt_prob = call_fidel_to_opt_prob
    if not func_caller.is_mf():
      raise CalledMFOptimiserWithSFCaller(self, func_caller)

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

  def _get_initial_qinfos(self, num_init_evals):
    """ Returns initial qinfos. """
    return get_euclidean_initial_qinfos(self.options.init_method, num_init_evals,
             self.domain.bounds, self.options.fidel_init_method, self.fidel_space.bounds,
             self.func_caller.fidel_to_opt,
             self.options.init_set_to_fidel_to_opt_with_prob)


# APIs for random optimisation ===========================================================

# An API for single fidelity optimisation
def random_optimiser_from_func_caller(func_caller, worker_manager, max_capital, mode,
                                      options=None, reporter='default'):
  """ Creates a EuclideanRandomOptimiser Object and optimises the function. """
  reporter = get_reporter(reporter)
  if isinstance(func_caller.domain, domains.EuclideanDomain):
    optimiser_constructor = EuclideanRandomOptimiser
    dflt_list_of_options = euclidean_random_optimiser_args
  else:
    raise ValueError('Random optimiser not implemented for domain of type %s.'%(
                     type(func_caller.domain)))
  # Load options
  if options is None:
    options = load_options(dflt_list_of_options)
  options.mode = mode
  # Create optimiser
  optimiser = optimiser_constructor(func_caller, worker_manager, options, reporter)
  # optimise and return
  return optimiser.optimise(max_capital)

# An API for multi-fidelity optimisation
def mf_random_optimiser_from_func_caller(func_caller, worker_manager, max_capital, mode,
                                         options=None, reporter='default',
                                         *args, **kwargs):
  """ Creates a MF EuclideanRandomOptimiser Object and optimises the function. """
  reporter = get_reporter(reporter)
  if isinstance(func_caller.domain, domains.EuclideanDomain) and \
     isinstance(func_caller.fidel_space, domains.EuclideanDomain):
    optimiser_constructor = MFEuclideanRandomOptimiser
    dflt_list_of_options = mf_euclidean_random_optimiser_args
  else:
    raise ValueError(('MF Random optimiser not implemented for (domain, fidel_space) '
                      + 'of types (%s, %s).')%(
                     type(func_caller.domain), type(func_caller.fidel_space)))
  # Load options
  if options is None:
    options = load_options(dflt_list_of_options)
  options.mode = mode
  # Create optimiser
  optimiser = optimiser_constructor(func_caller, worker_manager, options=options,
                                    reporter=reporter, *args, **kwargs)
  # optimise and return
  return optimiser.optimise(max_capital)

