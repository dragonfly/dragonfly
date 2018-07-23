"""
  Some generic utilities for Experiment Design.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

from argparse import Namespace
import numpy as np
# Local imports
from utils.general_utils import map_to_bounds

# Define constants
EVAL_ERROR_CODE = 'eval_error_250320181729'

# For initialisation
# ========================================================================================
def latin_hc_indices(dim, num_samples):
  """ Obtains indices for Latin Hyper-cube sampling. """
  index_set = [list(range(num_samples))] * dim
  lhs_indices = []
  for i in range(num_samples):
    curr_idx_idx = np.random.randint(num_samples-i, size=dim)
    curr_idx = [index_set[j][curr_idx_idx[j]] for j in range(dim)]
    index_set = [index_set[j][:curr_idx_idx[j]] + index_set[j][curr_idx_idx[j]+1:]
                 for j in range(dim)]
    lhs_indices.append(curr_idx)
  return lhs_indices

def latin_hc_sampling(dim, num_samples):
  """ Latin Hyper-cube sampling in the unit hyper-cube. """
  if num_samples == 0:
    return np.zeros((0, dim))
  elif num_samples == 1:
    return 0.5 * np.ones((1, dim))
  lhs_lower_boundaries = (np.linspace(0, 1, num_samples+1)[:num_samples]).reshape(1, -1)
  width = lhs_lower_boundaries[0][1] - lhs_lower_boundaries[0][0]
  lhs_lower_boundaries = np.repeat(lhs_lower_boundaries, dim, axis=0).T
  lhs_indices = latin_hc_indices(dim, num_samples)
  lhs_sample_boundaries = []
  for i in range(num_samples):
    curr_idx = lhs_indices[i]
    curr_sample_boundaries = [lhs_lower_boundaries[curr_idx[j]][j] for j in range(dim)]
    lhs_sample_boundaries.append(curr_sample_boundaries)
  lhs_sample_boundaries = np.array(lhs_sample_boundaries)
  uni_random_width = width * np.random.random((num_samples, dim))
  lhs_samples = lhs_sample_boundaries + uni_random_width
  return lhs_samples

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

def get_euclidean_initial_fidels(init_method, num_samples, fidel_space_bounds,
                                 fidel_to_opt, set_to_fidel_to_opt_with_prob=None):
  """ Returns an initial set of fidels for Euclidean spaces depending on the init_method.
      sets the fidel to fidel_to_opt with probability set_to_fidel_to_opt_with_prob.
  """
  # An internal function to process fidelity
  set_to_fidel_to_opt_with_prob = 0.0 if set_to_fidel_to_opt_with_prob is None else \
                                  set_to_fidel_to_opt_with_prob
  def _process_fidel(fidel):
    """ Returns fidel to fidel_to_opt with probability set_to_fidel_to_opt_with_prob. """
    if np.random.random() < set_to_fidel_to_opt_with_prob:
      return fidel_to_opt
    else:
      return fidel
  init_fidels = get_euclidean_initial_points(init_method, num_samples, fidel_space_bounds)
  return [_process_fidel(fidel) for fidel in init_fidels]

def get_euclidean_initial_qinfos(domain_init_method, num_samples, domain_bounds,
                                 fidel_init_method=None, fidel_space_bounds=None,
                                 fidel_to_opt=None, set_to_fidel_to_opt_with_prob=None):
  """ Returns the initial points in qinfo Namespaces. """
  init_points = get_euclidean_initial_points(domain_init_method, num_samples,
                                             domain_bounds)
  if fidel_space_bounds is None:
    return [Namespace(point=init_point) for init_point in init_points]
  else:
    init_fidels = get_euclidean_initial_fidels(fidel_init_method, num_samples,
                   fidel_space_bounds, fidel_to_opt, set_to_fidel_to_opt_with_prob)
    return [Namespace(point=ipt, fidel=ifl) for (ipt, ifl)
            in zip(init_points, init_fidels)]

