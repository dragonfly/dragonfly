"""
  Main APIs and command line tool for GP Bandit Optimisation.
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu

  Usage:
  python dragonfly.py --config <config file in .json or .pb format>
    --options <options file>
"""

# pylint: disable=relative-import
# pylint: disable=invalid-name
# pylint: disable=maybe-no-member
# pylint: disable=no-member

from __future__ import print_function
import os
import imp
# Local
from exd.cp_domain_utils import get_raw_from_processed_via_config, load_config_file, \
                                get_processed_func_from_raw_func_for_cp_domain_fidelity, \
                                get_processed_func_from_raw_func_for_cp_domain
from exd.domains import EuclideanDomain
from exd.exd_utils import get_unique_list_of_option_args
from exd.experiment_caller import EuclideanFunctionCaller, CPFunctionCaller
from exd.worker_manager import SyntheticWorkerManager
from opt.gp_bandit import get_all_euc_gp_bandit_args, get_all_cp_gp_bandit_args, \
                          get_all_mf_euc_gp_bandit_args, get_all_mf_cp_gp_bandit_args, \
                          gpb_from_func_caller
from utils.option_handler import get_option_specs, load_options

dragonfly_args = [ \
  get_option_specs('config', False, None, 'Path to the json or pb config file. '),
  get_option_specs('options', False, None, 'Path to the options file. '),
  get_option_specs('max_capital', False, -1.0,
                   'Maximum capital to be used in the experiment. '),
  get_option_specs('budget', False, -1.0, \
      'The budget of evaluations. If max_capital is none, will use this as max_capital.'),
                 ]


def _preprocess_multifidelity_arguments(fidel_space, domain, func, fidel_cost_func,
                                        fidel_to_opt, config):
  """ Preprocess fidel_space, domain arguments and configuration file. """
  # Preprocess config argument
  proc_func = func
  converted_cp_to_euclidean = False
  if isinstance(config, str):
    config = load_config_file(config)
  if fidel_space is None:
    fidel_space = config.fidel_space
  if domain is None:
    domain = config.domain
  if fidel_to_opt is None:
    fidel_to_opt = config.fidel_to_opt
  # The function
  if config is not None:
    proc_func = get_processed_func_from_raw_func_for_cp_domain_fidelity(func, config)
    proc_fidel_cost_func = get_processed_func_from_raw_func_for_cp_domain(fidel_cost_func,
      config.fidel_space, config.fidel_space_orderings.index_ordering,
      config.fidel_space_orderings.dim_ordering)
  else:
    proc_func = func
    proc_fidel_cost_func = fidel_cost_func
  ret_func = proc_func
  ret_fidel_cost_func = proc_fidel_cost_func
  # Preprocess domain argument
  if isinstance(fidel_space, (list, tuple)) and isinstance(domain, (list, tuple)):
    domain = EuclideanDomain(domain)
    fidel_space = EuclideanDomain(fidel_space)
  elif fidel_space.get_type() == 'euclidean' and domain.get_type() == 'euclidean':
    pass
  elif fidel_space.get_type() == 'cartesian_product' and \
       domain.get_type() == 'cartesian_product':
    if fidel_space.num_domains == 1 and \
         fidel_space.list_of_domains[0].get_type() == 'euclidean' and \
       domain.num_domains == 1 and domain.list_of_domains[0].get_type() == 'euclidean':
      # Change the fidelity space
      fidel_space = fidel_space.list_of_domains[0]
      config.fidel_space_orderings.dim_ordering = \
        config.fidel_space_orderings.dim_ordering[0]
      config.fidel_space_orderings.index_ordering = \
        config.fidel_space_orderings.index_ordering[0]
      config.fidel_space_orderings.kernel_ordering = \
        config.fidel_space_orderings.kernel_ordering[0]
      config.fidel_space_orderings.name_ordering = \
        config.fidel_space_orderings.name_ordering[0]
      config.fidel_to_opt = config.fidel_to_opt[0]
      fidel_to_opt = fidel_to_opt[0]
      # Change the domain
      domain = domain.list_of_domains[0]
      config.domain_orderings.dim_ordering = config.domain_orderings.dim_ordering[0]
      config.domain_orderings.index_ordering = config.domain_orderings.index_ordering[0]
      config.domain_orderings.kernel_ordering = config.domain_orderings.kernel_ordering[0]
      config.domain_orderings.name_ordering = config.domain_orderings.name_ordering[0]
      # Add to config
      config.fidel_space = fidel_space
      config.domain = domain
      converted_cp_to_euclidean = True
      # Functions
      ret_func = lambda z, x: proc_func([z], [x])
      ret_fidel_cost_func = lambda z: proc_fidel_cost_func([z])
  else:
    raise ValueError('fidel_space and domain should be either both instances of ' +
                     'EuclideanDomain or both CartesianProductDomain.')
  return (fidel_space, domain, ret_func, ret_fidel_cost_func, fidel_to_opt, config,
          converted_cp_to_euclidean)


def _preprocess_arguments(domain, func, config):
  """ Preprocess domain arguments and configuration file. """
  # Preprocess config argument
  converted_cp_to_euclidean = False
  if isinstance(config, str):
    config = load_config_file(config)
  if domain is None:
    domain = config.domain
  # The function
  if config is not None:
    proc_func = get_processed_func_from_raw_func_for_cp_domain(func, config.domain,
      config.domain_orderings.index_ordering, config.domain_orderings.dim_ordering)
  else:
    proc_func = func
  ret_func = proc_func
  # Preprocess domain argument
  if isinstance(domain, (list, tuple)):
    domain = EuclideanDomain(domain)
  elif domain.get_type() == 'euclidean':
    pass
  elif domain.get_type() == 'cartesian_product':
    if domain.num_domains == 1 and domain.list_of_domains[0].get_type() == 'euclidean':
      domain = domain.list_of_domains[0]
      config.domain_orderings.dim_ordering = config.domain_orderings.dim_ordering[0]
      config.domain_orderings.index_ordering = config.domain_orderings.index_ordering[0]
      config.domain_orderings.kernel_ordering = config.domain_orderings.kernel_ordering[0]
      config.domain_orderings.name_ordering = config.domain_orderings.name_ordering[0]
      config.domain = domain
      converted_cp_to_euclidean = True
      # The function
      ret_func = lambda x: proc_func([x])
  else:
    raise ValueError('domain should be an instance of EuclideanDomain or ' +
                     'CartesianProductDomain.')
  return domain, ret_func, config, converted_cp_to_euclidean


def maximise_multifidelity_function(func, fidel_space, domain, fidel_to_opt,
  fidel_cost_func, max_capital, config=None, options=None):
  """
    Maximises a multi-fidelity function 'func' over the domain 'domain' and fidelity
    space 'fidel_space'.
    Inputs:
      func: The function to be maximised. Takes two arguments func(z, x) where z is a
            member of the fidelity space and x is a member of the domain.
      fidel_space: The fidelity space from which the approximations are obtained.
                   Should be an instance of the Domain class in exd/domains.py.
                   If of the form [[l1, u1], [l2, u2], ...] where li < ui, then we will
                   create a Euclidean domain with lower bounds li and upper bounds
                   ui along each dimension.
      domain: The domain over which the function should be maximised, should be an
              instance of the Domain class in exd/domains.py.
              If domain is a list of the form [[l1, u1], [l2, u2], ...] where li < ui,
              then we will create a Euclidean domain with lower bounds li and upper bounds
              ui along each dimension.
      fidel_to_opt: The point at the fidelity space at which we wish to maximise func.
      max_capital: The maximum capital (budget) available for optimisation.
      config: Contains configuration parameters that are typically returned by
              exd.cp_domain_utils.load_config_file. config can be None only if domain
              is a EuclideanDomain object.
      options: Additional hyper-parameters for optimisation.
      * Alternatively, domain and fidelity space could be None if config is either a
        path_name to a configuration file or has configuration parameters.
    Returns:
      opt_val: The maximum value found during the optimisation procdure.
      opt_pt: The corresponding optimum point.
      history: A record of the optimisation procedure which include the point evaluated
               and the values at each time step.
  """
  # Preprocess domain and config arguments
  raw_func = func
  fidel_space, domain, func, fidel_cost_func, fidel_to_opt, config, _ = \
    _preprocess_multifidelity_arguments(fidel_space, domain, func, fidel_cost_func,
                                        fidel_to_opt, config)
  # Load arguments and function caller
  if fidel_space.get_type() == 'euclidean' and domain.get_type() == 'euclidean':
    func_caller = EuclideanFunctionCaller(func, domain, vectorised=False,
                    raw_fidel_space=fidel_space, fidel_cost_func=fidel_cost_func,
                    raw_fidel_to_opt=fidel_to_opt)
  else:
    func_caller = CPFunctionCaller(func, domain, '', raw_func=raw_func,
      domain_orderings=config.domain_orderings, fidel_space=fidel_space,
      fidel_cost_func=fidel_cost_func, fidel_to_opt=fidel_to_opt,
      fidel_space_orderings=config.fidel_space_orderings)

  # Create worker manager
  worker_manager = SyntheticWorkerManager(num_workers=1)
  # Optimise function here -----------------------------------------------------------
  opt_val, opt_pt, history = gpb_from_func_caller(func_caller, worker_manager,
                                     max_capital, is_mf=True, options=options)
  # Post processing
  if domain.get_type() == 'euclidean' and config is None:
    opt_pt = func_caller.get_raw_domain_coords(opt_pt)
    history.curr_opt_points = [func_caller.get_raw_domain_coords(pt)
                               for pt in history.curr_opt_points]
    history.query_points = [func_caller.get_raw_domain_coords(pt)
                            for pt in history.query_points]
  else:
    def _get_raw_from_processed_for_mf(fidel, pt):
      """ Returns raw point from processed point by accounting for the fact that a
          point could be None in the multi-fidelity setting. """
      if fidel is None or pt is None:
        return None, None
      else:
        return get_raw_from_processed_via_config((fidel, pt), config)
    # Now re-write curr_opt_points
    opt_pt = _get_raw_from_processed_for_mf(fidel_to_opt, opt_pt)[1]
    history.curr_opt_points_raw = [_get_raw_from_processed_for_mf(fidel_to_opt, pt)[1]
                                   for pt in history.curr_opt_points]
    query_fidel_points_raw = [_get_raw_from_processed_for_mf(fidel, pt)
      for fidel, pt in zip(history.query_fidels, history.query_points)]
    history.query_fidels = [zx[0] for zx in query_fidel_points_raw]
    history.query_points = [zx[1] for zx in query_fidel_points_raw]
  return opt_val, opt_pt, history


def maximise_function(func, domain, max_capital, config=None, options=None):
  """
    Maximises a function 'func' over the domain 'domain'.
    Inputs:
      func: The function to be maximised.
      domain: The domain over which the function should be maximised, should be an
              instance of the Domain class in exd/domains.py.
              If domain is a list of the form [[l1, u1], [l2, u2], ...] where li < ui,
              then we will create a Euclidean domain with lower bounds li and upper bounds
              ui along each dimension.
      max_capital: The maximum capital (budget) available for optimisation.
      config: Contains configuration parameters that are typically returned by
              exd.cp_domain_utils.load_config_file. config can be None only if domain
              is a EuclideanDomain object.
      options: Additional hyper-parameters for optimisation.
      * Alternatively, domain could be None if config is either a path_name to a
        configuration file or has configuration parameters.
    Returns:
      opt_val: The maximum value found during the optimisatio procdure.
      opt_pt: The corresponding optimum point.
      history: A record of the optimisation procedure which include the point evaluated
               and the values at each time step.
  """
  # Preprocess domain and config arguments
  raw_func = func
  domain, func, config, _ = _preprocess_arguments(domain, func, config)
  # Load arguments depending on domain type
  if domain.get_type() == 'euclidean':
    func_caller = EuclideanFunctionCaller(func, domain, vectorised=False)
  else:
    func_caller = CPFunctionCaller(func, domain, raw_func=raw_func,
                    domain_orderings=config.domain_orderings)
  # Create worker manager and function caller
  worker_manager = SyntheticWorkerManager(num_workers=1)
  # Optimise function here -----------------------------------------------------------
  opt_val, opt_pt, history = gpb_from_func_caller(func_caller, worker_manager,
                               max_capital, is_mf=False, options=options)
  # Post processing
  if domain.get_type() == 'euclidean' and config is None:
    opt_pt = func_caller.get_raw_domain_coords(opt_pt)
    history.curr_opt_points = [func_caller.get_raw_domain_coords(pt)
                               for pt in history.curr_opt_points]
    history.query_points = [func_caller.get_raw_domain_coords(pt)
                            for pt in history.query_points]
  else:
    opt_pt = get_raw_from_processed_via_config(opt_pt, config)
    history.curr_opt_points_raw = [get_raw_from_processed_via_config(pt, config)
                                   for pt in history.curr_opt_points]
    history.query_points_raw = [get_raw_from_processed_via_config(pt, config)
                                for pt in history.query_points]
  return opt_val, opt_pt, history


def main():
  """
    Maximizes a function given a config file containing the hyperparameters and the
    corresponding domain bounds.
  """
  # First load the domain and the objective
  all_args = dragonfly_args + get_all_euc_gp_bandit_args() + get_all_cp_gp_bandit_args() \
             + get_all_mf_euc_gp_bandit_args() + get_all_mf_cp_gp_bandit_args()
  all_args = get_unique_list_of_option_args(all_args)
  options = load_options(all_args, cmd_line=True)
  config = load_config_file(options.config)
  if hasattr(config, 'fidel_space'):
    is_mf = True
  else:
    is_mf = False
  objective_file_name = config.name
  expt_dir = os.path.dirname(os.path.abspath(os.path.realpath(options.config)))
  if not os.path.exists(expt_dir):
    raise ValueError("Experiment directory does not exist.")
  objective = imp.load_source(objective_file_name,
                              os.path.join(expt_dir, objective_file_name + '.py'))
#   func = objective.obj
#   if is_mf:
#     cost_func = objective.cost
#   # Preprocess domain and config arguments
#   if is_mf:
#     cost_func = objective.cost
#     fidel_space, domain, config, converted_cp_to_euclidean = \
#       _preprocess_multifidelity_arguments(config.fidel_space, config.domain, config)
#   else:
#     domain, config, converted_cp_to_euclidean = \
#       _preprocess_domain_and_config(config.domain, config)
  # Set capital
  options.capital_type = 'return_value'
  if options.budget < 0:
    budget = options.max_capital
  else:
    budget = options.budget
  if budget < 0:
    raise ValueError('Specify the budget via argument budget or max_capital.')
  options.max_capital = budget

  # Call optimiser
  if is_mf:
    print('Performing optimisation on fidel_space: %s, domain %s.'%(
          config.fidel_space, config.domain))
    opt_val, opt_pt, history = maximise_multifidelity_function(objective.obj,
      domain=None, fidel_space=None, fidel_to_opt=config.fidel_to_opt,
      fidel_cost_func=objective.cost, max_capital=options.max_capital, config=config,
      options=options)
  else:
    print('Performing optimisation on domain %s.'%(config.domain))
    opt_val, opt_pt, history = maximise_function(objective.obj, domain=None,
      max_capital=options.max_capital, config=config, options=options)
  print('Optimum Value in %d evals: %0.4f'%(len(history.curr_opt_points), opt_val))
  print('Optimum Point: %s.'%(opt_pt))


if __name__ == '__main__':
  main()

