"""
  A collection of utilities needed for the APIs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=maybe-no-member
# pylint: disable=no-member
# pylint: disable=bad-indentation
# pylint: disable=bad-continuation


from argparse import Namespace
import numpy as np
# Local imports
from ..exd.cp_domain_utils import load_config_file, \
                                get_processed_func_from_raw_func_for_cp_domain_fidelity, \
                                get_processed_func_from_raw_func_for_cp_domain
from ..exd.domains import EuclideanDomain
from ..exd.worker_manager import MultiProcessingWorkerManager, SyntheticWorkerManager, \
                                 AbstractWorkerManager
from ..opt.ga_optimiser import ga_opt_args
from ..opt.gp_bandit import get_all_euc_gp_bandit_args, \
                            get_all_cp_gp_bandit_args, get_all_mf_euc_gp_bandit_args, \
                            get_all_mf_cp_gp_bandit_args
from ..opt.random_optimiser import euclidean_random_optimiser_args, \
                                   mf_euclidean_random_optimiser_args, \
                                   cp_random_optimiser_args, mf_cp_random_optimiser_args
from ..opt.multiobjective_gp_bandit import get_all_euc_moo_gp_bandit_args, \
                                           get_all_cp_moo_gp_bandit_args
from ..opt.random_multiobjective_optimiser import \
              euclidean_random_multiobjective_optimiser_args, \
              cp_random_multiobjective_optimiser_args
from ..utils.general_utils import map_to_bounds
from ..utils.option_handler import load_options


def get_worker_manager_from_type(num_workers=1, worker_manager_type='default',
                                 capital_type=None, tmp_dir=None, *args, **kwargs):
  """ Get worker manager. """
  # If worker_manager_type is already a WorkerManager instance, --------------------------
  # then simply return this.
  if isinstance(worker_manager_type, AbstractWorkerManager):
    return worker_manager_type
  # Assign worker manager type depending on the capital_type -----------------------------
  # if worker_manager_type is default
  if worker_manager_type == 'default':
    if capital_type in ['return_value', 'num_evals']:
      _wm_type = 'synthetic'
    elif capital_type == 'realtime':
      _wm_type = 'multiprocessing'
    else:
      raise ValueError(('If worker_manager_type is default, capital_type should be one ' +
                        'return_value, num_evals, or multiprocessing. Given : %s.')%(
                            capital_type))
  else:
    _wm_type = worker_manager_type
  # Now create the worker manager --------------------------------------------------------
  if _wm_type == 'synthetic':
    return SyntheticWorkerManager(num_workers=num_workers, *args, **kwargs)
  elif _wm_type == 'multiprocessing':
    if tmp_dir is None:
      from datetime import datetime
      tmp_dir = './tmp_%s'%(datetime.now().strftime('%m%d_%H%M%S'))
    return MultiProcessingWorkerManager(worker_ids=num_workers, tmp_dir=tmp_dir)
  elif _wm_type == 'scheduling':
    raise NotImplementedError('Not Implemented a SchedulingWorkerManager yet!')
  else:
    raise ValueError('Unknown worker_manager_type: %s.'%(_wm_type))


def _raise_load_options_not_supported_error(method, prob, domain_type, capital_type):
  """ Raises an exception if the combination of method/prob/domain are not supported. """
  # problem
  prob_str_dict = {'opt': 'optimisation', 'moo': 'multi-objective optimisation',
                   'mfopt': 'multi-fidelity optimisation'}
  if prob not in prob_str_dict.keys():
    raise ValueError('Unknown problem %s.'%(prob))
  prob_str = prob_str_dict[prob]
  # method
  meth_str_dict = {'bo': 'Bayesian-Optimisation', 'ea': 'Evolutionary-Algorithms',
                   'ga': 'Evolutionary-Algorithms', 'direct': 'DiRect', 'pdoo': 'PDOO'}
  if method not in meth_str_dict.keys():
    raise ValueError('Unknown method %s.'%(method))
  meth_str = meth_str_dict[method]
  # domain
  if domain_type not in ['euclidean', 'cartesian_product']:
    dom_err_msg = 'We do not support directly optimising on %s spaces. Consider ' + \
                  'wrapping your domain with a CartesianProduct Domain. See ' + \
                  'dragonfly/exd/domains.py.'
    raise ValueError(dom_err_msg)
  err_msg = ('Draogonfly not support %s problems with %s on %s domains with capital ' +
             'type %s.')%(prob_str, meth_str, domain_type, capital_type)
  raise ValueError(err_msg)


def load_options_for_method(method, prob, domain, capital_type, options=None):
  """ Loads and returns the options for the problem. """
  # pylint: disable=too-many-branches
  # Both ea and ga can be used to specify evolutionary algorithms
  method = 'ga' if method == 'ea' else method
  # Now load options according to the case
  case_sel = (method, prob, domain.get_type())
  if case_sel == ('ga', 'opt', 'cartesian_product'):
    opt_options = load_options(ga_opt_args)
  elif case_sel == ('bo', 'opt', 'euclidean'):
    opt_options = load_options(get_all_euc_gp_bandit_args())
  elif case_sel == ('bo', 'opt', 'cartesian_product'):
    opt_options = load_options(get_all_cp_gp_bandit_args())
  elif (case_sel == ('direct', 'opt', 'euclidean') or \
        case_sel == ('pdoo', 'opt', 'euclidean')) and \
       capital_type in ['return_value', 'num_evals']:
    opt_options = Namespace()
  elif case_sel == ('rand', 'opt', 'euclidean'):
    opt_options = load_options(euclidean_random_optimiser_args)
  elif case_sel == ('rand', 'opt', 'cartesian_product'):
    opt_options = load_options(cp_random_optimiser_args)
  elif case_sel == ('rand', 'mfopt', 'euclidean'):
    opt_options = load_options(mf_euclidean_random_optimiser_args)
  elif case_sel == ('rand', 'mfopt', 'cartesian_product'):
    opt_options = load_options(mf_cp_random_optimiser_args)
  elif case_sel == ('bo', 'mfopt', 'euclidean'):
    opt_options = load_options(get_all_mf_euc_gp_bandit_args())
  elif case_sel == ('bo', 'mfopt', 'cartesian_product'):
    opt_options = load_options(get_all_mf_cp_gp_bandit_args())
  elif case_sel == ('bo', 'moo', 'euclidean'):
    opt_options = load_options(get_all_euc_moo_gp_bandit_args())
  elif case_sel == ('bo', 'moo', 'cartesian_product'):
    opt_options = load_options(get_all_cp_moo_gp_bandit_args())
  elif case_sel == ('rand', 'moo', 'euclidean'):
    opt_options = load_options(euclidean_random_multiobjective_optimiser_args)
  elif case_sel == ('rand', 'moo', 'cartesian_product'):
    opt_options = load_options(cp_random_multiobjective_optimiser_args)
  else:
    _raise_load_options_not_supported_error(method, prob, domain.get_type(), capital_type)
  # Now look at any options that have been passed in already and add them to opt_options
  options = Namespace() if options is None else options
  for attr in vars(options):
    setattr(opt_options, attr, getattr(options, attr))
  # Also add capital type
  opt_options.capital_type = 'return_value' if capital_type == 'num_evals' \
                             else capital_type
  return opt_options


def preprocess_multifidelity_arguments(fidel_space, domain, funcs, fidel_cost_func,
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
        ((not hasattr(fidel_space, 'domain_constraints')) or \
        fidel_space.domain_constraints is None) and \
        domain.num_domains == 1 and domain.list_of_domains[0].get_type() == 'euclidean' and \
        ((not hasattr(domain, 'domain_constraints')) or domain.domain_constraints is None):
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


def preprocess_arguments(domain, funcs, config):
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
    if domain.num_domains == 1 and domain.list_of_domains[0].get_type() == 'euclidean' and \
       ((not hasattr(domain, 'domain_constraints')) or domain.domain_constraints is None):
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


def post_process_history_for_minimisation(history):
  """ Post processes history for minimisation. """
  # Negate past queries
  history.query_vals = [-qv for qv in history.query_vals]
  history.curr_opt_vals = [-cov for cov in history.curr_opt_vals]
  history.curr_true_opt_vals = [-cov for cov in history.curr_true_opt_vals]
  return history


def preprocess_options_for_gp_bandits(options, config, prob, converted_cp_to_euclidean):
  """ Preprocesses the options for GP Bandits. """
  # pylint: disable=too-many-branches
  # pylint: disable=too-many-locals
  options = Namespace(**vars(options)) # Make a new copy
  # We will call this internal function repeatedly below =================================
  def _get_gpb_prior_mean_from_unproc(prior_mean_unproc_given, prior_mean_given,
        config, prob, converted_cp_to_euclidean):
    """ Return the prior mean for a single point from arguments. """
    if prior_mean_given is not None:
      gpb_prior_mean = prior_mean_given # If the processed version is given, use that.
    elif prior_mean_unproc_given is None:
      gpb_prior_mean = None
    elif prob in ['opt', 'moo']:
      if config.domain.get_type() == 'euclidean' and not converted_cp_to_euclidean:
        # In this case, both the processed and unprocessed versions are the same!
        gpb_prior_mean_single = prior_mean_unproc_given
      else:
        gpb_pms = get_processed_func_from_raw_func_for_cp_domain(
                       prior_mean_unproc_given, config.domain,
                       config.domain_orderings.index_ordering,
                       config.domain_orderings.dim_ordering)
        if config.domain.get_type() == 'euclidean' and converted_cp_to_euclidean:
          # TODO (KK): this is a bit of a clumsy fix. Check if this can be neatened.
          gpb_prior_mean_single = lambda x, *args, **kwargs: gpb_pms(
              map_to_bounds(x, config.domain.bounds), *args, **kwargs)
        else:
          gpb_prior_mean_single = gpb_pms
      gpb_prior_mean = lambda X, *args, **kwargs: np.array(
        [gpb_prior_mean_single(x, *args, **kwargs) for x in X])
    elif prob in ['mfopt', 'mfmoo']:
      if config.fidel_space.get_type() == 'euclidean' \
        and config.domain.get_type() == 'euclidean' and not converted_cp_to_euclidean:
        # In this case, both the processed and unprocessed versions are the same!
        gpb_prior_mean_mf_single = prior_mean_unproc_given
      else:
        gpb_pmmfs = get_processed_func_from_raw_func_for_cp_domain_fidelity(
                        prior_mean_unproc_given, config)
        if config.domain.get_type() == 'euclidean' and converted_cp_to_euclidean:
          gpb_prior_mean_mf_single = lambda z, x, *args, **kwargs: gpb_pmmfs(
              map_to_bounds(z, config.fidel_space.bounds),
              map_to_bounds(x, config.domain.bounds),
              *args, **kwargs)
        else:
          gpb_prior_mean_mf_single = gpb_pmmfs
      gpb_prior_mean = lambda ZX, *args, **kwargs: np.array(
        [gpb_prior_mean_mf_single(z, x, *args, **kwargs) for (z, x) in ZX])
    else:
      raise ValueError('Unrecognised problem type (prob): %s.'%(prob))
    return gpb_prior_mean

  # 1. Prior mean for single objective ===================================================
  if hasattr(options, 'gp_prior_mean'):
    prior_mean_given = options.gpb_prior_mean if hasattr(options, 'gpb_prior_mean') \
                       else None
    options.gpb_prior_mean = _get_gpb_prior_mean_from_unproc(
                               options.gp_prior_mean, prior_mean_given,
                               config, prob, converted_cp_to_euclidean)
  # 2. A custom kernel for single objective ==============================================
  if hasattr(options, 'gpb_prior_kernel_unproc') and \
     options.gpb_prior_kernel_unproc is not None:
    raise NotImplementedError('Not Implemented custom kernels yet!')
  # 3. Prior mean for multi-objective ====================================================
  if hasattr(options, 'gps_prior_means') and \
    options.gps_prior_means is not None:
    if not hasattr(options.gps_prior_means, '__iter__'):
      raise ValueError('gps_prior_means should be a list or tuple of ' +
                       'callable objects!')
    moo_prior_means_given = options.moo_gpb_prior_means \
                              if hasattr(options, 'moo_gpb_prior_means') else None
    if moo_prior_means_given is None:
      moo_prior_means_given = [None] * len(options.gps_prior_means)
    options.moo_gpb_prior_means = []
    for (prior_mean_unproc_given, prior_mean_given) in \
      zip(options.gps_prior_means, moo_prior_means_given):
      options.moo_gpb_prior_means.append(
        _get_gpb_prior_mean_from_unproc(prior_mean_unproc_given, prior_mean_given,
                                        config, prob, converted_cp_to_euclidean))
  # 4. Custom kernel for multi objective =================================================
  if hasattr(options, 'moo_gpb_prior_kernels_unproc') and \
     options.moo_gpb_prior_kernels_unproc is not None:
    raise NotImplementedError('Not Implemented custom kernels yet!')
  # Return options =======================================================================
  return options

