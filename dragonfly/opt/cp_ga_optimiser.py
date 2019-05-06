"""
  A GA Optimiser for Cartesian Product Domains.
  We will use this mostly for optimising the acquisition in Dragonfly.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from copy import copy
import numpy as np
from scipy.spatial.distance import cdist
# Local imports
from ..exd.cp_domain_utils import get_processed_func_from_raw_func_for_cp_domain, \
                            load_cp_domain_from_config_file
from ..exd.experiment_caller import ExperimentCaller, CPFunctionCaller
from ..exd.exd_utils import get_cp_domain_initial_qinfos
from .ga_optimiser import GAOptimiser, ga_opt_args
from ..utils.general_utils import project_to_bounds
from ..utils.option_handler import load_options


cpga_opt_args = ga_opt_args


# Define Mutation Operators for each space individually
def _get_gauss_perturbation(x, bounds, sigmas=None):
  """ Returns the Gaussian perturbation. """
  if sigmas is None:
    sigmas = [(b[1] - b[0])/10 for b in bounds]
  epsilon = np.random.normal(scale=sigmas)
  return project_to_bounds(np.array(x) + epsilon, bounds)

def _return_ndarray_with_type(x, ret):
  """ Returns after checking type. """
  if isinstance(x, np.ndarray):
    return ret
  else:
    return type(x)(ret)

def euclidean_gauss_mutation(x, bounds, sigmas=None):
  """ Defines a Euclidean Mutation. """
  ret = _get_gauss_perturbation(x, bounds, sigmas)
  return _return_ndarray_with_type(x, ret)

def discrete_euclidean_mutation(x, list_of_items, uniform_prob=0.2):
  """ Makes a change depending on the vector values. """
  # cdist requires 2d input
  dists = cdist([x], list_of_items)[0]
  # Exponentiate and normalise to get the probabilities.
  unnorm_diff_probs = np.exp(-dists)
  sample_diff_probs = unnorm_diff_probs / unnorm_diff_probs.sum()
  # Change the distribution to interplate between it and uniform
  n = sample_diff_probs.shape[0]
  unif = np.full(sample_diff_probs.shape, 1. / n)
  p = (1 - uniform_prob) * sample_diff_probs + uniform_prob * unif
  # Now draw the samples
  idxs = np.arange(n)
  idx = np.random.choice(idxs, p=p)
  ret = list_of_items[idx]
  return _return_ndarray_with_type(x, ret)

def integral_gauss_mutation(x, bounds, sigmas=None):
  """ Defines a Euclidean Mutation. """
  ret = _get_gauss_perturbation(x, bounds, sigmas)
  ret = ret.round().astype(np.int)
  return _return_ndarray_with_type(x, ret)

def prod_discrete_random_mutation(x, list_of_list_of_items):
  """ Randomly chooses an index and changes it to something else from
      list_of_list_of_items. """
  ret = [copy(elem) for elem in x]
  change_idx = np.random.choice(len(x))
  change_list = copy(list_of_list_of_items[change_idx])
  change_list.remove(x[change_idx])
  change_val = np.random.choice(change_list)
  ret[change_idx] = change_val
  return ret

def discrete_random_mutation(x, list_of_items):
  """ Randomly chooses an index and changes it to something else from
      list_of_items. """
  ret = prod_discrete_random_mutation([x], [list_of_items])
  return ret[0]

def prod_discrete_numeric_exp_mutation(x, list_of_list_of_items, uniform_prob=0.2):
  """ Makes a change depending on the numerical values. """
  diff_vals = [np.abs(loi - x[idx]) for idx, loi in enumerate(list_of_list_of_items)]
  # Exponentiate and normalise to get the probabilities.
  unnorm_diff_probs = [np.exp(-elem) for elem in diff_vals]
  sample_diff_probs = [elem/elem.sum() for elem in unnorm_diff_probs]
  # Now change the distribution so that we pick one randomly with prob uniform_prob.
  sample_probs = [(1 - uniform_prob) * elem +
                  uniform_prob * np.ones((len(elem),)) / float(len(elem))
                  for elem in sample_diff_probs]
  # Now draw the samples
  ret = [np.random.choice(loi, p=p) for (loi, p) in
         zip(list_of_list_of_items, sample_probs)]
  return ret

def discrete_numeric_exp_mutation(x, list_of_items):
  """ Randomly chooses an index and changes it to something else from
      list_of_items. """
  ret = prod_discrete_numeric_exp_mutation([x], [list_of_items])
  return ret[0]


def get_default_mutation_op(dom):
  """ Returns the default mutation operator for the domain. """
  if dom.get_type() == 'euclidean':
    return lambda x: euclidean_gauss_mutation(x, dom.bounds)
  elif dom.get_type() == 'integral':
    return lambda x: integral_gauss_mutation(x, dom.bounds)
  elif dom.get_type() == 'discrete':
    return lambda x: discrete_random_mutation(x, dom.list_of_items)
  elif dom.get_type() == 'prod_discrete':
    return lambda x: prod_discrete_random_mutation(x, dom.list_of_list_of_items)
  elif dom.get_type() == 'discrete_numeric':
    return lambda x: discrete_numeric_exp_mutation(x, dom.list_of_items)
  elif dom.get_type() == 'prod_discrete_numeric':
    return lambda x: prod_discrete_numeric_exp_mutation(x, dom.list_of_list_of_items)
  elif dom.get_type() == 'discrete_euclidean':
    return lambda x: discrete_euclidean_mutation(x, dom.list_of_items)
  elif dom.get_type() == 'neural_network':
    from ..nn.nn_modifiers import get_single_nn_mutation_op
    return get_single_nn_mutation_op(dom, [0.5, 0.25, 0.125, 0.075, 0.05])
  else:
    raise ValueError('No default mutation implemented for domain type %s.'%(
                     dom.get_type()))


class CPGAOptimiser(GAOptimiser):
  """ A GA Optimiser for Cartesian Product Domains. """

  def __init__(self, func_caller, worker_manager, single_mutation_ops=None,
               single_crossover_ops=None, options=None, reporter=None):
    """ Constructor. """
    options = load_options(cpga_opt_args, partial_options=options)
    super(CPGAOptimiser, self).__init__(func_caller, worker_manager,
      mutation_op=self._mutation_op, crossover_op=self._crossover_op,
      options=options, reporter=reporter)
    self._set_up_single_mutation_ops(single_mutation_ops)
    self._set_up_single_crossover_ops(single_crossover_ops)

  def _set_up_single_mutation_ops(self, single_mutation_ops):
    """ Set up mutation operations. """
    if single_mutation_ops is None:
      single_mutation_ops = [None] * self.domain.num_domains
    for idx, dom in enumerate(self.domain.list_of_domains):
      if single_mutation_ops[idx] is None:
        single_mutation_ops[idx] = get_default_mutation_op(dom)
    self.single_mutation_ops = single_mutation_ops

  def _set_up_single_crossover_ops(self, crossover_ops):
    """ Set up cross-over operations. """
    # pylint: disable=unused-argument
    self.crossover_ops = crossover_ops

  def _mutation_op(self, X, num_mutations):
    """ The mutation operator for the product domain. """
    if hasattr(num_mutations, '__iter__'):
      num_mutations_for_each_x = num_mutations
    else:
      choices_for_each_mutation = np.random.choice(len(X), num_mutations, replace=True)
      num_mutations_for_each_x = [np.sum(choices_for_each_mutation == i) for i
                                  in range(len(X))]
    ret = []
    # Now extend
    for idx in range(len(X)):
      ret.extend(self._get_mutation_for_single_x(X[idx],
                 num_mutations_for_each_x[idx]))
    np.random.shuffle(ret)
    return ret

  def _get_mutation_for_single_x(self, x, num_mutations):
    """ Gets the mutation for single x. """
    ret = []
    for _ in range(num_mutations):
      curr_mutation = []
      for idx, elem in enumerate(x):
        curr_mutation.append(self.single_mutation_ops[idx](elem))
      ret.append(curr_mutation)
    return ret

  def _crossover_op(self):
    """ The crossover operation for the product domain. """
    raise NotImplementedError('Not implemented cross over operation yet.')

  # Define a function to obtain the initial qinfos =======================================
  def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
    """ Gets num_init_evals initial points. """
    return get_cp_domain_initial_qinfos(self.domain, num_init_evals,
                                        dom_euclidean_sample_type='latin_hc',
                                        dom_integral_sample_type='latin_hc',
                                        dom_nn_sample_type='rand', *args, **kwargs)


# APIs =================================================================================
def cp_ga_optimiser_from_proc_args(func_caller, cp_domain, worker_manager, max_capital,
                                   mode='asy', orderings=None, single_mutation_ops=None,
                                   single_crossover_ops=None, options=None,
                                   reporter=None):
  """ A GA optimiser on Cartesian product space from the function caller. """
  if not isinstance(func_caller, ExperimentCaller):
    func_caller = CPFunctionCaller(func_caller, cp_domain, domain_orderings=orderings)
  options = load_options(cpga_opt_args, partial_options=options)
  options.mode = mode
  from ..exd.worker_manager import RealWorkerManager, SyntheticWorkerManager
  if isinstance(worker_manager, RealWorkerManager):
    options.capital_type = 'realtime'
  elif isinstance(worker_manager, SyntheticWorkerManager):
    options.capital_type = 'return_value'
  return (CPGAOptimiser(func_caller, worker_manager,
                        single_mutation_ops=single_mutation_ops,
                        single_crossover_ops=single_crossover_ops,
                        options=options, reporter=reporter)).optimise(max_capital)


def cp_ga_optimiser_from_raw_args(raw_func, domain_config_file, worker_manager,
                                  max_capital, mode='asy', single_mutation_ops=None,
                                  single_crossover_ops=None, options=None,
                                  reporter='default'):
  """ Optimise a function from raw args such as the raw function and the configuration
      file. """
  # pylint: disable=no-member
  cp_dom, orderings = load_cp_domain_from_config_file(domain_config_file)
  proc_func = get_processed_func_from_raw_func_for_cp_domain(
                raw_func, cp_dom, orderings.index_ordering, orderings.dim_ordering)
  func_caller = CPFunctionCaller(proc_func, cp_dom, raw_func=raw_func,
                                 domain_orderings=orderings)
  return cp_ga_optimiser_from_proc_args(func_caller, cp_dom, worker_manager, max_capital,
                                        mode, orderings, single_mutation_ops,
                                        single_crossover_ops, options, reporter)

