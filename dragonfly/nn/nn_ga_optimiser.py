"""
  A GA optimiser for Neural Networks.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from argparse import Namespace
import numpy as np
# Local imports
from .nn_opt_utils import get_initial_pool
from ..opt.ga_optimiser import GAOptimiser, ga_opt_args
from ..utils.option_handler import load_options
from ..utils.reporters import get_reporter


nn_ga_opt_args = ga_opt_args


class NNGAOptimiser(GAOptimiser):
  """ Optimising functions defined on Neural Networks. """
  # pylint: disable=abstract-method

  def _get_initial_qinfos(self, num_init_evals):
    """ Initial qinfos. """
    ret = get_initial_pool(self.domain.nn_type)
    np.random.shuffle(ret)
    ret = ret[:num_init_evals]
    return [Namespace(point=x) for x in ret]

# APIs
# ======================================================================================
def nn_ga_optimise_from_args(func_caller, worker_manager, max_capital, mode, mutation_op,
                             crossover_op=None, options=None, reporter='default'):
  """ GA optimisation from args. """
  if options is None:
    reporter = get_reporter(reporter)
    options = load_options(ga_opt_args, reporter=reporter)
  options.mode = mode
  return (NNGAOptimiser(func_caller, worker_manager, mutation_op, crossover_op,
                        options, reporter)).optimise(max_capital)

