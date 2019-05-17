"""
  Dragonfly: An open source library for scalable and robust Bayesian optimisation.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=relative-import
# pylint: disable=invalid-name
# pylint: disable=maybe-no-member
# pylint: disable=no-member

# Local
from .api_utils import get_worker_manager_from_type, \
                       load_options_for_method, \
                       post_process_history_for_minimisation, \
                       preprocess_arguments, \
                       preprocess_multifidelity_arguments, \
                       preprocess_options_for_gp_bandits
from ..exd.experiment_caller import EuclideanFunctionCaller, CPFunctionCaller
from ..exd.cp_domain_utils import get_raw_from_processed_via_config
from ..opt.cp_ga_optimiser import cp_ga_optimiser_from_proc_args
from ..opt.gp_bandit import gpb_from_func_caller
from ..opt.random_optimiser import random_optimiser_from_func_caller, \
                                   mf_random_optimiser_from_func_caller
from ..utils.oper_utils import direct_ft_maximise
from ..utils.doo import pdoo_maximise_from_args


def maximise_multifidelity_function(func, fidel_space, domain, fidel_to_opt,
  fidel_cost_func, max_capital, opt_method='bo',
  worker_manager='default', num_workers=1, capital_type='return_value',
  config=None, options=None, reporter='default'):
  # pylint: disable=too-many-arguments
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
      max_capital: The maximum capital (time budget or number of evaluations) available
                   for optimisation.
      opt_method: The method used for optimisation. Could be one of bo or rand.
                  Default is bo. bo - Bayesian optimisation, rand - Random search.
      worker_manager: Should be an instance of WorkerManager (see exd/worker_manager.py)
                      or a string with one of the following values
                      {'default', 'synthetic', 'multiprocessing', 'schedulint'}.
      num_workers: The number of parallel workers (i.e. number of evaluations to carry
                   out in parallel).
      capital_type: The type of capital. Should be one of 'return_value' or 'realtime'.
                    Default is return_value which indicates we will use the value returned
                    by fidel_cost_func. If realtime, we will use wall clock time.
      config: Either a configuration file or or parameters returned by
              exd.cp_domain_utils.load_config_file. config can be None only if domain
              is a EuclideanDomain object.
      options: Additional hyperparameters for optimisation, as a namespace.
      reporter: A stream to print progress made during optimisation, or one of the
                following strings 'default', 'silent'. If 'silent', then it suppresses
                all outputs. If 'default', writes to stdout.
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
  (fidel_space, domain, preproc_func_list, fidel_cost_func, fidel_to_opt, config,
   converted_cp_to_euclidean) = \
    preprocess_multifidelity_arguments(fidel_space, domain, [func], fidel_cost_func,
                                       fidel_to_opt, config)
  func = preproc_func_list[0]
  # Load arguments and function caller
  if fidel_space.get_type() == 'euclidean' and domain.get_type() == 'euclidean':
    func_caller = EuclideanFunctionCaller(func, domain, vectorised=False,
                    raw_fidel_space=fidel_space, fidel_cost_func=fidel_cost_func,
                    raw_fidel_to_opt=fidel_to_opt, config=config)
  else:
    func_caller = CPFunctionCaller(func, domain, '', raw_func=raw_func,
      domain_orderings=config.domain_orderings, fidel_space=fidel_space,
      fidel_cost_func=fidel_cost_func, fidel_to_opt=fidel_to_opt,
      fidel_space_orderings=config.fidel_space_orderings, config=config)
  # load options
  options = load_options_for_method(opt_method, 'mfopt', domain, capital_type, options)
  # Create worker manager
  worker_manager = get_worker_manager_from_type(num_workers=num_workers,
                     worker_manager_type=worker_manager, capital_type=capital_type)

  # Select method here -----------------------------------------------------------
  if opt_method == 'bo':
    options = preprocess_options_for_gp_bandits(options, config, 'mfopt',
                                                converted_cp_to_euclidean)
    opt_val, opt_pt, history = gpb_from_func_caller(func_caller, worker_manager,
                                       max_capital, is_mf=True, options=options,
                                       reporter=reporter)
  elif opt_method == 'rand':
    opt_val, opt_pt, history = mf_random_optimiser_from_func_caller(func_caller,
                worker_manager, max_capital, options=options, reporter=reporter)

  # Post processing
  if domain.get_type() == 'euclidean' and config is None:
    opt_pt = func_caller.get_raw_domain_coords(opt_pt)
    history.curr_opt_points = [func_caller.get_raw_domain_coords(pt)
                               for pt in history.curr_opt_points]
    history.query_points = [func_caller.get_raw_domain_coords(pt)
                            for pt in history.query_points]
    history.query_fidels = [func_caller.get_raw_fidel_coords(fidel)
                            for fidel in history.query_fidels]
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


def maximise_function(func, domain, max_capital, opt_method='bo',
                      worker_manager='default', num_workers=1, capital_type='num_evals',
                      config=None, options=None, reporter='default'):
  """
    Maximises a function 'func' over the domain 'domain'.
    Inputs:
      func: The function to be maximised.
      domain: The domain over which the function should be maximised, should be an
              instance of the Domain class in exd/domains.py.
              If domain is a list of the form [[l1, u1], [l2, u2], ...] where li < ui,
              then we will create a Euclidean domain with lower bounds li and upper bounds
              ui along each dimension.
      max_capital: The maximum capital (time budget or number of evaluations) available
                   for optimisation.
      opt_method: The method used for optimisation. Could be one of bo, rand, ga, ea,
                  direct, or pdoo. Default is bo.
                  bo - Bayesian optimisation, ea/ga: Evolutionary algorithm,
                  rand - Random search, direct: Dividing Rectangles, pdoo: PDOO
      worker_manager: Should be an instance of WorkerManager (see exd/worker_manager.py)
                      or a string with one of the following values
                      {'default', 'synthetic', 'multiprocessing', 'scheduling'}.
      num_workers: The number of parallel workers (i.e. number of evaluations to carry
                   out in parallel).
      capital_type: The type of capital. Should be one of 'return_value' or 'realtime'.
                    Default is return_value which indicates we will use the value returned
                    by fidel_cost_func. If realtime, we will use wall clock time.
      config: Contains configuration parameters that are typically returned by
              exd.cp_domain_utils.load_config_file. config can be None only if domain
              is a EuclideanDomain object.
      options: Additional hyper-parameters for optimisation, as a namespace.
      reporter: A stream to print progress made during optimisation, or one of the
                following strings 'default', 'silent'. If 'silent', then it suppresses
                all outputs. If 'default', writes to stdout.
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
  domain, preproc_func_list, config, converted_cp_to_euclidean = \
    preprocess_arguments(domain, [func], config)
  func = preproc_func_list[0]
  # Load arguments depending on domain type
  if domain.get_type() == 'euclidean':
    func_caller = EuclideanFunctionCaller(func, domain, vectorised=False, config=config)
  else:
    func_caller = CPFunctionCaller(func, domain, raw_func=raw_func,
                    domain_orderings=config.domain_orderings, config=config)
  # load options
  options = load_options_for_method(opt_method, 'opt', domain, capital_type, options)
  # Create worker manager and function caller
  worker_manager = get_worker_manager_from_type(num_workers=num_workers,
                     worker_manager_type=worker_manager, capital_type=capital_type)
  # Optimise function here -----------------------------------------------------------
  if opt_method == 'bo':
    options = preprocess_options_for_gp_bandits(options, config, 'opt',
                                                converted_cp_to_euclidean)
    opt_val, opt_pt, history = gpb_from_func_caller(func_caller, worker_manager,
      max_capital, is_mf=False, options=options, reporter=reporter)
  elif opt_method in ['ga', 'ea']:
    opt_val, opt_pt, history = cp_ga_optimiser_from_proc_args(func_caller, domain,
      worker_manager, max_capital, options=options, reporter=reporter)
  elif opt_method == 'rand':
    opt_val, opt_pt, history = random_optimiser_from_func_caller(func_caller,
      worker_manager, max_capital, options=options, reporter=reporter)
  elif opt_method == 'direct':
    opt_val, opt_pt, history = direct_ft_maximise(func, domain.bounds, int(max_capital),
                                                  return_history=True)
  elif opt_method == 'pdoo':
    opt_val, opt_pt, history = pdoo_maximise_from_args(func, domain.bounds,
                                     int(max_capital), return_history=True)
  # Post processing
  if domain.get_type() == 'euclidean' and config is None:
    if opt_method not in ['direct', 'pdoo']:
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


def minimise_function(func, *args, **kwargs):
  """
    Minimises a function func over domain domain. See maximise_function for a description
    of the arguments. All arguments are the same except for func, which should now be
    minimised.
  """
  func_to_max = lambda x: -func(x)
  max_val, opt_pt, history = maximise_function(func_to_max, *args, **kwargs)
  min_val = - max_val
  history = post_process_history_for_minimisation(history)
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
  history = post_process_history_for_minimisation(history)
  return min_val, opt_pt, history


# Alternative spelling
maximize_function = maximise_function
maximize_multifidelity_function = maximise_multifidelity_function
minimize_function = minimise_function
minimize_multifidelity_function = minimise_multifidelity_function

