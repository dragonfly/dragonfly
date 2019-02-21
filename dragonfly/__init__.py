"""
  Dragonfly: An open source library for scalable and robust Bayesian optimisation.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=relative-import
# pylint: disable=invalid-name
# pylint: disable=maybe-no-member
# pylint: disable=no-member

from __future__ import print_function
# Local
from dragonfly.exd.cp_domain_utils import get_raw_from_processed_via_config, load_config_file, \
                                get_processed_func_from_raw_func_for_cp_domain_fidelity, \
                                get_processed_func_from_raw_func_for_cp_domain
from dragonfly.exd.domains import EuclideanDomain
from dragonfly.exd.experiment_caller import EuclideanFunctionCaller, CPFunctionCaller, \
                                  EuclideanMultiFunctionCaller, CPMultiFunctionCaller
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.opt.gp_bandit import gpb_from_func_caller
from dragonfly.opt.multiobjective_gp_bandit import multiobjective_gpb_from_multi_func_caller


def _preprocess_multifidelity_arguments(fidel_space, domain, funcs, fidel_cost_func,
                                        fidel_to_opt, config):
  """ Preprocess fidel_space, domain arguments and configuration file. """
  # Preprocess config argument
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
    proc_funcs = [get_processed_func_from_raw_func_for_cp_domain_fidelity(f, config)
                  for f in funcs]
    proc_fidel_cost_func = get_processed_func_from_raw_func_for_cp_domain(fidel_cost_func,
      config.fidel_space, config.fidel_space_orderings.index_ordering,
      config.fidel_space_orderings.dim_ordering)
  else:
    proc_funcs = funcs
    proc_fidel_cost_func = fidel_cost_func
  ret_funcs = proc_funcs
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
      def _get_ret_func_from_proc_func_for_euc_domains(_proc_func):
        """ Get function to return. """
        return lambda z, x: _proc_func([z], [x])
      ret_funcs = [_get_ret_func_from_proc_func_for_euc_domains(pf) for pf in proc_funcs]
      ret_fidel_cost_func = lambda z: proc_fidel_cost_func([z])
  else:
    raise ValueError('fidel_space and domain should be either both instances of ' +
                     'EuclideanDomain or both CartesianProductDomain.')
  return (fidel_space, domain, ret_funcs, ret_fidel_cost_func, fidel_to_opt, config,
          converted_cp_to_euclidean)


def _preprocess_arguments(domain, funcs, config):
  """ Preprocess domain arguments and configuration file. """
  # Preprocess config argument
  converted_cp_to_euclidean = False
  if isinstance(config, str):
    config = load_config_file(config)
  if domain is None:
    domain = config.domain
  # The function
  if config is not None:
    proc_funcs = [get_processed_func_from_raw_func_for_cp_domain(f, config.domain,
      config.domain_orderings.index_ordering, config.domain_orderings.dim_ordering)
      for f in funcs]
  else:
    proc_funcs = funcs
  ret_funcs = proc_funcs
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
      def _get_ret_func_from_proc_func_for_euc_domains(_proc_func):
        """ Get function to return. """
        return lambda x: _proc_func([x])
      ret_funcs = [_get_ret_func_from_proc_func_for_euc_domains(pf) for pf in proc_funcs]
  else:
    raise ValueError('domain should be an instance of EuclideanDomain or ' +
                     'CartesianProductDomain.')
  return domain, ret_funcs, config, converted_cp_to_euclidean


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
  fidel_space, domain, preproc_func_list, fidel_cost_func, fidel_to_opt, config, _ = \
    _preprocess_multifidelity_arguments(fidel_space, domain, [func], fidel_cost_func,
                                        fidel_to_opt, config)
  func = preproc_func_list[0]
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
  domain, preproc_func_list, config, _ = _preprocess_arguments(domain, [func], config)
  func = preproc_func_list[0]
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


def _post_process_history_for_minimisation(history):
  """ Post processes history for minimisation. """
  # Negate past queries
  history.query_vals = [-qv for qv in history.query_vals]
  history.curr_opt_vals = [-cov for cov in history.curr_opt_vals]
  history.curr_true_opt_vals = [-cov for cov in history.curr_true_opt_vals]
  return history


def minimise_function(func, *args, **kwargs):
  """
    Minimises a function func over domain domain. See maximise_function for a description
    of the arguments. All arguments are the same except for func, which should now be
    minimised.
  """
  func_to_max = lambda x: -func(x)
  max_val, opt_pt, history = maximise_function(func_to_max, *args, **kwargs)
  min_val = - max_val
  history = _post_process_history_for_minimisation(history)
  return min_val, opt_pt, history


def minimise_multifidelity_function(func, *args, **kwargs):
  """
    Minimises a multifidelity function func over domain domain. See
    maximise_multifidelity_function for a description of the arguments. All arguments are
    the same except for func, which should now be minimised.
  """
  func_to_max = lambda x, z: -func(x, z)
  max_val, opt_pt, history = maximise_multifidelity_function(func_to_max, *args, **kwargs)
  min_val = - max_val
  history = _post_process_history_for_minimisation(history)
  return min_val, opt_pt, history


def multiobjective_maximise_functions(funcs, domain, max_capital,
                                      config=None, options=None):
  """
    Co-optimises the functions 'funcs' over the domain 'domain'.
    Inputs:
      funcs: The functions to be co-optimised (maximised).
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
      pareto_values: The pareto optimal values found during the optimisation procdure.
      pareto_points: The corresponding pareto optimum points in the domain.
      history: A record of the optimisation procedure which include the point evaluated
               and the values at each time step.
  """
  # Preprocess domain and config arguments
  raw_funcs = funcs
  domain, funcs, config, _ = _preprocess_arguments(domain, funcs, config)
  # Load arguments depending on domain type
  if domain.get_type() == 'euclidean':
    multi_func_caller = EuclideanMultiFunctionCaller(funcs, domain, vectorised=False)
  else:
    multi_func_caller = CPMultiFunctionCaller(funcs, domain, raw_funcs=raw_funcs,
                          domain_orderings=config.domain_orderings)
  # Create worker manager and function caller
  worker_manager = SyntheticWorkerManager(num_workers=1)
  # Optimise function here -----------------------------------------------------------
  pareto_values, pareto_points, history = multiobjective_gpb_from_multi_func_caller(
             multi_func_caller, worker_manager, max_capital, is_mf=False, options=options)
  # Post processing
  if domain.get_type() == 'euclidean' and config is None:
    pareto_points = [multi_func_caller.get_raw_domain_coords(pt) for pt in pareto_points]
    history.query_points = [multi_func_caller.get_raw_domain_coords(pt)
                            for pt in history.query_points]
  else:
    pareto_points = [get_raw_from_processed_via_config(pt, config) for
                     pt in pareto_points]
    history.query_points_raw = [get_raw_from_processed_via_config(pt, config)
                                for pt in history.query_points]
  return pareto_values, pareto_points, history


def multiobjective_minimise_functions(funcs, *args, **kwargs):
  """
    Minimises the functions funcs over domain domain. See
    multiobjective_maximise_functions for a description of the arguments. All arguments
    are the same except for funcs, which should now be minimised.
  """
  def _get_func_to_max(_func):
    """ Returns a function to be maximised. """
    return lambda x: - _func(x)
  funcs_to_max = [_get_func_to_max(f) for f in funcs]
  pareto_max_values, pareto_points, history = multiobjective_maximise_functions(
                                                funcs_to_max, *args, **kwargs)
  # Post process history
  def _negate_array(arr):
    """ Negates an array. """
    return [-elem for elem in arr]
  pareto_min_values = _negate_array(pareto_max_values)
  history.query_vals = [_negate_array(qv) for qv in history.query_vals]
  history.curr_pareto_vals = [_negate_array(cpv) for cpv in history.curr_pareto_vals]
  history.curr_true_pareto_vals = [_negate_array(ctpv) for ctpv in
                                   history.curr_true_pareto_vals]
  return pareto_min_values, pareto_points, history

