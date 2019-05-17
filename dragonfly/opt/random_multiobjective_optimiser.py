"""
  Implements some instances of a random optimiser for multi-objective optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from argparse import Namespace
import numpy as np
# Local
from .random_optimiser import random_sample_from_cp_domain_wrapper
from ..exd.cp_domain_utils import get_processed_func_from_raw_func_for_cp_domain, \
                                load_cp_domain_from_config_file
from ..exd import domains
from ..exd.exd_utils import get_euclidean_initial_qinfos, get_cp_domain_initial_qinfos
from ..exd.experiment_caller import CPMultiFunctionCaller
from .multiobjective_optimiser import MultiObjectiveOptimiser, \
                                         multiobjective_opt_args
from ..utils.option_handler import load_options
from ..utils.reporters import get_reporter
from ..utils.general_utils import map_to_bounds


random_multiobjective_optimiser_args = multiobjective_opt_args
euclidean_random_multiobjective_optimiser_args = random_multiobjective_optimiser_args
cp_random_multiobjective_optimiser_args = random_multiobjective_optimiser_args


class RandomMultiObjectiveOptimiser(MultiObjectiveOptimiser):
  """ Optimises multiple objectives with random evaluations. """
  #pylint: disable=abstract-method

  def __init__(self, multi_func_caller, worker_manager, options=None, reporter=None):
    """ Constructor. """
    reporter = get_reporter(reporter)
    if options is None:
      options = load_options(random_multiobjective_optimiser_args, reporter=reporter)
    super(RandomMultiObjectiveOptimiser, self).__init__(multi_func_caller, worker_manager,
                                           model=None, options=options, reporter=reporter)

  def _multi_opt_method_set_up(self):
    """ Any set up specific to the method for multi-objective optimisation. """
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


class EuclideanRandomMultiObjectiveOptimiser(RandomMultiObjectiveOptimiser):
  """ A class which optimises for multiple objectives in Euclidean spaces using random
      evaluations.
  """

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


class CPRandomMultiObjectiveOptimiser(RandomMultiObjectiveOptimiser):
  """ A class which optimises for multiple objectives in Cartesian product spaces using
      random evaluations.
  """

  def is_an_mf_method(self):
    """ Returns False since it is not a False method. """
    return False

  def _determine_next_query(self):
    """ Determines the next query. """
    qinfo = Namespace(point=random_sample_from_cp_domain_wrapper(1, self.domain,
                                                                 self.reporter)[0])
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


# APIs -----------------------------------------------------------------------------------
def random_multiobjective_optimisation_from_multi_func_caller(multi_func_caller,
    worker_manager, max_capital, mode, options=None, reporter='default'):
  """ Creates an appropriate MultiObjectiveOptimiser object and optimises the functions
      in multi_func_caller.
  """
  reporter = get_reporter(reporter)
  if isinstance(multi_func_caller.domain, domains.EuclideanDomain):
    moo_constructor = EuclideanRandomMultiObjectiveOptimiser
    dflt_list_of_options = euclidean_random_multiobjective_optimiser_args
  elif isinstance(multi_func_caller.domain, domains.CartesianProductDomain):
    moo_constructor = CPRandomMultiObjectiveOptimiser
    dflt_list_of_options = cp_random_multiobjective_optimiser_args
  else:
    raise ValueError('Random optimiser not implemented for domain of type %s.'%(
                     type(multi_func_caller.domain)))
  # Load options
  if options is None:
    options = load_options(dflt_list_of_options)
  options.mode = mode
  # Create Optimiser
  moo_optimiser = moo_constructor(multi_func_caller, worker_manager, options, reporter)
  # optimise and return
  return moo_optimiser.optimise(max_capital)


def cp_random_multiobjective_optimisation_from_raw_args(raw_funcs,
    domain_config_file, *args, **kwargs):
  """ A random multi-objective optimiser on CP spaces. """
  # pylint: disable=no-member
  cp_dom, orderings = load_cp_domain_from_config_file(domain_config_file)
  proc_funcs = [get_processed_func_from_raw_func_for_cp_domain(
                  rf, cp_dom, orderings.index_ordering, orderings.dim_ordering)
                for rf in raw_funcs]
  multi_func_caller = CPMultiFunctionCaller(proc_funcs, cp_dom, raw_funcs=raw_funcs,
                                            domain_orderings=orderings)
  return random_multiobjective_optimisation_from_multi_func_caller(multi_func_caller,
                                                                   *args, **kwargs)



