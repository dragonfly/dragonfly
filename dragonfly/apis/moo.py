"""
  APIs for Multi-objective Optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=maybe-no-member


from .api_utils import preprocess_arguments, get_worker_manager_from_capital_type, \
                       load_options_for_method
from ..exd.experiment_caller import EuclideanMultiFunctionCaller, CPMultiFunctionCaller
from ..exd.cp_domain_utils import get_raw_from_processed_via_config
from ..opt.multiobjective_gp_bandit import multiobjective_gpb_from_multi_func_caller
from ..opt.random_multiobjective_optimiser import \
                                 random_multiobjective_optimisation_from_multi_func_caller


def multiobjective_maximise_functions(funcs, domain, max_capital,
                                      capital_type='num_evals', opt_method='bo',
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
      capital_type: The type of capital. Should be one of 'return_value' or 'realtime'.
                    Default is return_value which indicates we will use the value returned
                    by fidel_cost_func. If realtime, we will use wall clock time.
      opt_method: The method used for optimisation. Could be one of bo, rand, ga, ea,
                  direct, or pdoo. Default is bo.
                  bo - Bayesian optimisation, ea/ga: Evolutionary algorithm,
                  rand - Random search, direct: Dividing Rectangles, pdoo: PDOO
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
  raw_funcs = funcs
  domain, funcs, config, _ = preprocess_arguments(domain, funcs, config)
  # Load arguments depending on domain type
  if domain.get_type() == 'euclidean':
    multi_func_caller = EuclideanMultiFunctionCaller(funcs, domain, vectorised=False)
  else:
    multi_func_caller = CPMultiFunctionCaller(funcs, domain, raw_funcs=raw_funcs,
                          domain_orderings=config.domain_orderings)
  # load options
  options = load_options_for_method(opt_method, 'moo', domain, capital_type, options)
  # Create worker manager and function caller
  worker_manager = get_worker_manager_from_capital_type(capital_type)

  # Select method here -------------------------------------------------------------------
  if opt_method == 'bo':
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


# Alternative spelling
multiobjective_maximize_functions = multiobjective_maximise_functions
multiobjective_minimize_functions = multiobjective_minimise_functions

