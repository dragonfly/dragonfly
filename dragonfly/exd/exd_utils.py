"""
  Some generic utilities for Experiment Design.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=unbalanced-tuple-unpacking

from argparse import Namespace
import numpy as np
# Local imports
from .cp_domain_utils import get_processed_from_raw_via_config, \
                             get_raw_from_processed_via_config, \
                             sample_from_cp_domain
from ..utils.general_utils import map_to_bounds, flatten_list_of_lists
from ..utils.oper_utils import direct_ft_maximise, latin_hc_sampling, pdoo_maximise, \
                               random_maximise

# Define constants
EVAL_ERROR_CODE = 'eval_error_250320181729'

# For initialisation
# ========================================================================================
def random_sampling_cts(dim, num_samples):
  """ Just picks uniformly random samples from a  dim-dimensional space. """
  return np.random.random((num_samples, dim))

def random_sampling_kmeans_cts(dim, num_samples):
  """ Picks a large number of points uniformly at random and then runs k-means to
      select num_samples points. """
  try:
    from sklearn.cluster import KMeans
    num_candidates = np.clip(100*(dim**2), 4*num_samples, 20*num_samples)
    candidates = random_sampling_cts(dim, num_candidates)
    centres = KMeans(n_clusters=num_samples).fit(candidates)
    return centres.cluster_centers_
  except ImportError:
    return random_sampling_cts(dim, num_samples)

def random_sampling_discrete(param_values, num_samples):
  """ Picks uniformly random samples from a discrete set of values.
      param_values is a list of lists where each internal list is the set of values
      for each parameter. """
  num_params = len(param_values)
  ret_vals_for_each_param = []
  for i in range(num_params):
    rand_idxs = np.random.randint(0, len(param_values[i]), size=(num_samples,))
    ret_vals_for_each_param.append([param_values[i][idx] for idx in rand_idxs])
  # Now put them into an num_samples x num_params array
  ret = []
  for j in range(num_samples):
    curr_sample = [ret_vals_for_each_param[param_idx][j] for param_idx in
                                                             range(num_params)]
    ret.append(curr_sample)
  return ret


# A wrapper to get initial points ------------------------------------------
def get_euclidean_initial_points(init_method, num_samples, domain_bounds):
  """ Gets the initial set of points for a Euclidean space depending on the init_method.
  """
  dim = len(domain_bounds)
  if init_method == 'rand':
    ret = random_sampling_cts(dim, num_samples)
  elif init_method == 'rand_kmeans':
    ret = random_sampling_kmeans_cts(dim, num_samples)
  elif init_method == 'latin_hc':
    ret = latin_hc_sampling(dim, num_samples)
  else:
    raise ValueError('Unknown init method %s.'%(init_method))
  return map_to_bounds(ret, domain_bounds)


def _process_fidel_for_initialisation(fidel, fidel_to_opt, set_to_fidel_to_opt_with_prob):
  """ Returns fidel to fidel_to_opt with probability set_to_fidel_to_opt_with_prob. """
  if np.random.random() < set_to_fidel_to_opt_with_prob:
    return fidel_to_opt
  else:
    return fidel

def get_euclidean_initial_fidels(init_method, num_samples, fidel_space_bounds,
                                 fidel_to_opt, set_to_fidel_to_opt_with_prob=None):
  """ Returns an initial set of fidels for Euclidean spaces depending on the init_method.
      sets the fidel to fidel_to_opt with probability set_to_fidel_to_opt_with_prob.
  """
  # An internal function to process fidelity
  set_to_fidel_to_opt_with_prob = 0.0 if set_to_fidel_to_opt_with_prob is None else \
                                  set_to_fidel_to_opt_with_prob
  init_fidels = get_euclidean_initial_points(init_method, num_samples, fidel_space_bounds)
  return [_process_fidel_for_initialisation(fidel, fidel_to_opt,
                                            set_to_fidel_to_opt_with_prob)
          for fidel in init_fidels]


def get_cp_domain_initial_fidels(fidel_space, num_samples, fidel_to_opt,
                                 set_to_fidel_to_opt_with_prob,
                                 euclidean_sample_type='latin_hc',
                                 integral_sample_type='latin_hc',
                                 nn_sample_type='rand', *args, **kwargs):
  """ Function to return the initial fidelities for CP Domain. """
  set_to_fidel_to_opt_with_prob = 0.0 if set_to_fidel_to_opt_with_prob is None else \
                                  set_to_fidel_to_opt_with_prob
  init_fidels = sample_from_cp_domain(fidel_space, num_samples,
                              euclidean_sample_type=euclidean_sample_type,
                              integral_sample_type=integral_sample_type,
                              nn_sample_type=nn_sample_type, *args, **kwargs)
  return [_process_fidel_for_initialisation(fidel, fidel_to_opt,
                                            set_to_fidel_to_opt_with_prob)
          for fidel in init_fidels]


def get_euclidean_initial_qinfos(domain_init_method, num_samples, domain_bounds,
                                 fidel_init_method=None, fidel_space_bounds=None,
                                 fidel_to_opt=None, set_to_fidel_to_opt_with_prob=None,
                                 *args, **kwargs):
  """ Returns the initial points in qinfo Namespaces. """
  # pylint: disable=unused-argument
  init_points = get_euclidean_initial_points(domain_init_method, num_samples,
                                             domain_bounds)
  if fidel_space_bounds is None:
    return [Namespace(point=init_point) for init_point in init_points]
  else:
    init_fidels = get_euclidean_initial_fidels(fidel_init_method, num_samples,
                   fidel_space_bounds, fidel_to_opt, set_to_fidel_to_opt_with_prob)
    return [Namespace(point=ipt, fidel=ifl) for (ipt, ifl)
            in zip(init_points, init_fidels)]


def get_cp_domain_initial_qinfos(domain, num_samples, fidel_space=None, fidel_to_opt=None,
                                 set_to_fidel_to_opt_with_prob=None,
                                 dom_euclidean_sample_type='latin_hc',
                                 dom_integral_sample_type='latin_hc',
                                 dom_nn_sample_type='rand',
                                 fidel_space_euclidean_sample_type='latin_hc',
                                 fidel_space_integral_sample_type='latin_hc',
                                 fidel_space_nn_sample_type='rand',
                                 *args, **kwargs):
  """ Get initial qinfos in Cartesian product domain. """
  # pylint: disable=too-many-arguments
  ret_dom_pts = sample_from_cp_domain(domain, num_samples,
                                      euclidean_sample_type=dom_euclidean_sample_type,
                                      integral_sample_type=dom_integral_sample_type,
                                      nn_sample_type=dom_nn_sample_type, *args, **kwargs)
  if fidel_space is None:
    ret_dom_pts = ret_dom_pts[:num_samples]
    return [Namespace(point=x) for x in ret_dom_pts]
  else:
    ret_fidels = get_cp_domain_initial_fidels(fidel_space, num_samples, fidel_to_opt,
                   set_to_fidel_to_opt_with_prob,
                   euclidean_sample_type=fidel_space_euclidean_sample_type,
                   integral_sample_type=fidel_space_integral_sample_type,
                   nn_sample_type=fidel_space_nn_sample_type, *args, **kwargs)
    return [Namespace(point=ipt, fidel=ifl) for (ipt, ifl) in
            zip(ret_dom_pts, ret_fidels)]

# Maximise with method here.
# Some wrappers for all methods
def maximise_with_method(method, obj, domain, max_evals, return_history=False,
                         *args, **kwargs):
  """ A wrapper which optimises obj over the domain domain. """
  # If the method itself is a function, just use it.
  if hasattr(method, '__call__'):
    return method(obj, domain, max_evals, return_history, *args, **kwargs)
  # The common use case is that the method is a string.
  if domain.get_type() == 'euclidean':
    return maximise_with_method_on_euclidean_domain(method, obj, domain.bounds, max_evals,
                                                    return_history, *args, **kwargs)
  elif domain.get_type() == 'integral':
    return maximise_with_method_on_integral_domain(method, obj, domain.bounds, max_evals,
                                                   domain.get_dim(), return_history,
                                                   *args, **kwargs)
  elif domain.get_type() == 'prod_discrete':
    return maximise_with_method_on_prod_discrete_domain(method, obj, domain, max_evals,
                                                        return_history, *args, **kwargs)
  elif domain.get_type() == 'cartesian_product':
    return maximise_with_method_on_cp_domain(method, obj, domain, max_evals,
                                             return_history, *args, **kwargs)
  else:
    raise ValueError('Unknown domain type %s.'%(domain.get_type()))


def maximise_with_method_on_euclidean_domain(method, obj, bounds, max_evals, dim,
                                             return_history=False, *args, **kwargs):
  """ A wrapper for euclidean spaces which calls one of the functions below based on the
      method. """
  if method.lower().startswith('rand'):
    max_val, max_pt, history = \
      random_maximise(obj, bounds, max_evals, return_history, *args, **kwargs)
  elif method.lower().startswith('direct') and dim <= 60:
    max_val, max_pt, history = \
      direct_ft_maximise(obj, bounds, max_evals, return_history, *args, **kwargs)
  elif method.lower().startswith('pdoo') or (method.lower().startswith('direct')
                                             and dim > 60):
    max_val, max_pt, history = \
      pdoo_maximise(obj, bounds, max_evals, *args, **kwargs)
  else:
    raise ValueError('Unknown maximisation method: %s.'%(method))
  if return_history:
    return max_val, max_pt, history
  else:
    return max_val, max_pt

def maximise_with_method_on_integral_domain(method, obj, bounds, max_evals,
                                            return_history=False, *args, **kwargs):
  """ A wrapper for integral spaces which calls one of the functions below based on the
      method. """
  # pylint: disable=unused-argument
  raise NotImplementedError('Not implemented integral domain optimisers yet.')


def maximise_with_method_on_prod_discrete_domain(method, obj, domain, max_evals,
                                                 return_history=False, *args, **kwargs):
  """ A wrapper for discrete spaces which calls one of the functions below based on the
      method. """
  # pylint: disable=unused-argument
  raise NotImplementedError('Not implemented discrete domain optimisers yet.')


def maximise_with_method_on_product_euclidean_spaces(method, obj, list_of_euc_domains,
                                                     max_evals, return_history=False,
                                                     *args, **kwargs):
  """ Maximises an objective with method on a product Euclidean space. """
  def _regroup_flattended_point(pt, _dom_dims, _cum_dims):
    """ Regroups a flattened point into a list of lists. """
    ret = [pt[cd:cd+d] for (cd, d) in zip(_cum_dims, _dom_dims)]
    assert len(ret) == len(_dom_dims) # Check if dimensions match.
    return ret
  dom_dims = [dom.dim for dom in list_of_euc_domains]
  cum_dims = list(np.cumsum(dom_dims))
  cum_dims = [0] + cum_dims[:-1]
  euc_dom_bounds = flatten_list_of_lists([dom.bounds for dom in list_of_euc_domains])
  modified_obj = lambda x: obj(_regroup_flattended_point(x, dom_dims, cum_dims))
  opt_result = maximise_with_method_on_euclidean_domain(method, modified_obj,
    euc_dom_bounds, max_evals, return_history, *args, **kwargs)
  if return_history:
    max_val, max_pt, history = opt_result
  else:
    max_val, max_pt = opt_result
  # regroup the maximum point
  regrouped_max_pt = _regroup_flattended_point(max_pt, dom_dims, cum_dims)
  if return_history:
    return max_val, regrouped_max_pt, history
  else:
    return max_val, regrouped_max_pt


def _rand_maximise_vectorised_objective_in_cp_domain(obj, domain, max_evals,
  return_history=False):
  """ Maximises a vectorised function in Cartesian product spaces.
      Mostly used for TS style acquisitions in BO.
  """
  rand_samples = []
  num_sample_tries = 0
  while not (len(rand_samples) >= max_evals or
             (len(rand_samples) > 0 and num_sample_tries >= 5)):
    rand_samples.extend(sample_from_cp_domain(domain, int(max_evals),
                                              verbose_constraint_satisfaction=False))
    num_sample_tries += 1
    if len(rand_samples) == 0 and num_sample_tries % 10 == 0:
      from warnings import warn
      warn(('Sampling from domain failed despite %d attempts -- will ' +
            'continue trying but consider reparametrising your domain if ' +
            'this problem persists.')%(num_sample_tries))
  # Compute objective on sampled points.
  rand_values = [obj(x) for x in rand_samples]
  max_idx = np.argmax(rand_values)
  max_pt = rand_samples[max_idx]
  max_val = rand_values[max_idx]
  if return_history:
    history = Namespace(query_points=rand_samples,
                        query_vals=rand_values)
    return max_val, max_pt, history
  else:
    return max_val, max_pt


def maximise_with_method_on_cp_domain(method, obj, domain, max_evals,
                                      return_history=False, *args, **kwargs):
  """ A wrapper for maximising an objective on a CartesianProductDomain. """
  # pylint: disable=too-many-locals
  # pylint: disable=too-many-branches
  if method.lower().startswith(('direct', 'pdoo')): # ------------------------------------
    # If the domain consists entirely of Euclidean parts, use PDOO or DiRect
    return maximise_with_method_on_product_euclidean_spaces(method, obj,
      domain.list_of_domains, max_evals, return_history, *args, **kwargs)
  elif method.lower() == 'rand': # -------------------------------------------------------
    # Use a random optimiser
    return _rand_maximise_vectorised_objective_in_cp_domain(obj, domain, max_evals,
                                                            return_history)
  elif method.lower().startswith('ga'): # ------------------------------------------------
    # First decide if there is a follow up Euclidean opt method
    ga_opt_methods = method.lower().split('-')
    euc_domains = [dom for dom in domain.list_of_domains if dom.get_type() == 'euclidean']
    if len(ga_opt_methods) == 2 and len(euc_domains) > 0:
      to_follow_up_with_euc_opt = True
      euc_opt_method = ga_opt_methods[1]
      euc_domain_idxs = [idx for (idx, dom) in enumerate(domain.list_of_domains) if
                         dom.get_type() == 'euclidean']
    else:
      to_follow_up_with_euc_opt = False
    # Run a GA optimiser ---------------------------------------
    from .worker_manager import SyntheticWorkerManager
    from .experiment_caller import CPFunctionCaller
    from ..opt.cp_ga_optimiser import cp_ga_optimiser_from_proc_args
    obj_in_func_caller = CPFunctionCaller(obj, domain, domain_orderings=None)
    worker_manager = SyntheticWorkerManager(1, time_distro='const')
    ga_max_val, ga_max_pt, ga_history = cp_ga_optimiser_from_proc_args(obj_in_func_caller,
      domain, worker_manager, max_evals, mode='asy', options=None, reporter='silent')
    # Finished running a GA optimiser --------------------------
    if to_follow_up_with_euc_opt:
      # If you need to do an additional optimisation on the Euclidean part, do so.
      def _swap_indices_at_point(swap_pt, orig_pt, swap_idxs):
        """ Swap indices at point swap_pt. """
        ret = orig_pt[:]
        for sidx, spt in zip(swap_idxs, swap_pt):
          ret[sidx] = spt
        return ret
      def _get_swapped_obj(_obj, _orig_pt, _swap_idxs):
        """ Returns an objective which swaps in new points at _swap_idxs. """
        return lambda x: _obj(_swap_indices_at_point(x, _orig_pt, _swap_idxs))
      swapped_obj = _get_swapped_obj(obj, ga_max_pt, euc_domain_idxs)
      euc_max_val, euc_max_pt = \
        maximise_with_method_on_product_euclidean_spaces(euc_opt_method, swapped_obj,
        euc_domains, max_evals, return_history=False)
      if euc_max_val > ga_max_val:
        max_val = euc_max_val
        history = ga_history
        max_pt = _swap_indices_at_point(euc_max_pt, ga_max_pt, euc_domain_idxs)
      else:
        max_val, max_pt, history = (ga_max_val, ga_max_pt, ga_history)
    else:
      max_val, max_pt, history = (ga_max_val, ga_max_pt, ga_history)
    # return
    if return_history:
      return max_val, max_pt, history
    else:
      return max_val, max_pt
  else:
    raise NotImplementedError('Not implemented yet!')


# Miscellaneous
# ===============================================================================
def get_unique_list_of_option_args(all_args):
  """ Returns a unique list of option args. """
  ret = []
  ret_names = []
  for arg in all_args:
    if arg['name'] not in ret_names:
      ret.append(arg)
      ret_names.append(arg['name'])
  return ret

# For saving and loading data ---------------------------------------------------
def preprocess_loaded_data_for_domain(loaded_data, experiment_caller):
  """ Preprocesses loaded data. """
  if hasattr(experiment_caller, 'config') and experiment_caller.config is not None:
    config = experiment_caller.config
    if ('config_points' in loaded_data) and (not 'points' in loaded_data):
      loaded_data['points'] = [get_processed_from_raw_via_config(cpt, config) for cpt in
                               loaded_data['config_points']]
    if ('config_fidels' in loaded_data) and (not 'fidels' in loaded_data):
      loaded_data['fidels'] = [get_processed_from_raw_via_config(cf, config) for cf in
                               loaded_data['config_fidels']]
  return loaded_data

def postprocess_data_to_save_for_domain(data_to_save, experiment_caller):
  """ Post process loaded data. """
  if hasattr(experiment_caller, 'config') and experiment_caller.config is not None:
    config = experiment_caller.config
    if 'points' in data_to_save:
      try:
        data_to_save['config_points'] = [get_raw_from_processed_via_config(pt, config)
                                        for pt in data_to_save['points']]
      except:
        pass
    if 'fidels' in data_to_save:
      try:
        data_to_save['config_fidels'] = [get_raw_from_processed_via_config(fidel, config)
                                        for pt in data_to_save['fidels']]
      except:
        pass
  return data_to_save

