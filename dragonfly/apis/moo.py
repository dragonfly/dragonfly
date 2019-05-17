"""
  APIs for Multi-objective Optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=maybe-no-member


from .api_utils import get_worker_manager_from_type, \
                       load_options_for_method, \
                       preprocess_arguments, \
                       preprocess_options_for_gp_bandits
from ..exd.experiment_caller import EuclideanMultiFunctionCaller, CPMultiFunctionCaller
from ..exd.cp_domain_utils import get_raw_from_processed_via_config
from ..opt.multiobjective_gp_bandit import multiobjective_gpb_from_multi_func_caller
from ..opt.random_multiobjective_optimiser import \
                                 random_multiobjective_optimisation_from_multi_func_caller


_FUNC_FORMAT_ERR_MSG = ('funcs should either be a list of functions or a tuple (F, n) ' +
                        'where F returns a list of values and n is the number of ' +
                        'objectives.')


def multiobjective_maximise_functions(funcs, domain, max_capital, opt_method='bo',
        worker_manager='default', num_workers=1, capital_type='num_evals',
        config=None, options=None, reporter='default'):
  """
    Co-optimises the functions 'funcs' over the domain 'domain'.
    Inputs:
      funcs: The functions to be co-optimised (maximised).
      domain: The domain over which the function should be maximised, should be an
              instance of the Domain class in exd/domains.py.
              If domain is a list of the form [[l1, u1], [l2, u2], ...] where li < ui,
              then we will create a Euclidean domain with lower bounds li and upper bounds
              ui along each dimension.
      max_capital: The maximum capital (time budget or number of evaluations) available
                   for optimisation.
      opt_method: The method used for optimisation. Could be one of bo or rand.
                  Default is bo. bo: Bayesian optimisation, rand: Random search.
      worker_manager: Should be an instance of WorkerManager (see exd/worker_manager.py)
                      or a string with one of the following values
                      {'default', 'synthetic', 'multiprocessing', 'schedulint'}.
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
      pareto_values: The pareto optimal values found during the optimisation procdure.
      pareto_points: The corresponding pareto optimum points in the domain.
      history: A record of the optimisation procedure which include the point evaluated
               and the values at each time step.
  """
  # Preprocess domain and config arguments
  if isinstance(funcs, tuple) and len(funcs) == 2:
    raw_funcs = funcs[0]
    domain, _mfc_funcs_arg_0, config, converted_cp_to_euclidean = \
        preprocess_arguments(domain, [raw_funcs], config)
    mfc_funcs_arg = (_mfc_funcs_arg_0[0], funcs[1])
  elif isinstance(funcs, list):
    raw_funcs = funcs
    domain, mfc_funcs_arg, config, converted_cp_to_euclidean = \
        preprocess_arguments(domain, funcs, config)
  else:
    raise ValueError(_FUNC_FORMAT_ERR_MSG)
  # Load arguments depending on domain type
  if domain.get_type() == 'euclidean':
    multi_func_caller = EuclideanMultiFunctionCaller(mfc_funcs_arg, domain,
                                                     vectorised=False, config=config)
  else:
    multi_func_caller = CPMultiFunctionCaller(mfc_funcs_arg, domain, raw_funcs=raw_funcs,
                          domain_orderings=config.domain_orderings, config=config)
  # load options
  options = load_options_for_method(opt_method, 'moo', domain, capital_type, options)
  # Create worker manager and function caller
  worker_manager = get_worker_manager_from_type(num_workers=num_workers,
                     worker_manager_type=worker_manager, capital_type=capital_type)
  # Select method here -------------------------------------------------------------------
  if opt_method == 'bo':
    options = preprocess_options_for_gp_bandits(options, config, 'moo',
                                                converted_cp_to_euclidean)
    pareto_values, pareto_points, history = multiobjective_gpb_from_multi_func_caller(
                                        multi_func_caller, worker_manager, max_capital,
                                        is_mf=False, options=options, reporter=reporter)
  elif opt_method == 'rand':
    pareto_values, pareto_points, history = \
      random_multiobjective_optimisation_from_multi_func_caller(multi_func_caller,
        worker_manager, max_capital, mode='asy', options=options, reporter=reporter)

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
  # The following function returns the negative of a single function
  def _get_func_to_max(_func):
    """ Returns a function to be maximised. """
    return lambda x: - _func(x)
  # The following function returns the negative of multiple functions
  def _get_funcs_to_max(_funcs):
    """ Returns funcs to max. """
    if isinstance(_funcs, tuple) and len(_funcs) == 2:
      neg_funcs = lambda *args, **kwargs: [-val for val in _funcs[0](*args, **kwargs)]
      funcs_to_max = (neg_funcs, _funcs[1])
    elif isinstance(_funcs, list):
      funcs_to_max = [_get_func_to_max(f) for f in funcs]
    else:
      raise ValueError(_FUNC_FORMAT_ERR_MSG)
    return funcs_to_max
  # Transform the functions and call the maximiser
  funcs_to_max = _get_funcs_to_max(funcs)
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


# Alternative spelling
multiobjective_maximize_functions = multiobjective_maximise_functions
multiobjective_minimize_functions = multiobjective_minimise_functions

