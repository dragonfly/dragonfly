"""
  A collection of very generic python utilities.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import print_function
from __future__ import division

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np
from scipy.sparse import dok_matrix
from scipy.linalg import solve_triangular


# Some very generic utility functions
def map_to_cube(pts, bounds):
  """ Maps bounds to [0,1]^d and returns the representation in the cube. """
  return (pts - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])


def map_to_bounds(pts, bounds):
  """ Given a point in [0,1]^d, returns the representation in the original space. """
  return pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def get_sublist_from_indices(orig_list, idxs):
  """ Returns a sublist from the indices. orig_list can be anthing that can be
      indexed and idxs are a list of indices. """
  return [orig_list[idx] for idx in idxs]


def project_to_bounds(x, bounds):
  """ Projects to bounds. """
  bounds = np.array(bounds)
  ret = np.clip(x, bounds[:, 0], bounds[:, 1])
  return ret


def get_idxs_from_list_of_lists(list_of_lists, idx):
  """ Returns a list of objects corresponding to index idx from a list of lists. """
  return [elem[idx] for elem in list_of_lists]


def transpose_list_of_lists(list_of_lists):
  """ Transposes a list of lists. """
  return [list(i) for i in zip(*list_of_lists)]


def compute_average_sq_prediction_error(Y1, Y2):
  """ Returns the average prediction error. """
  return np.linalg.norm(np.array(Y1) - np.array(Y2))**2 / len(Y1)


def dist_squared(X1, X2):
  """ If X1 is n1xd and X2 is n2xd, this returns an n1xn2 matrix where the (i,j)th
      entry is the squared distance between X1(i,:) and X2(j,:).
  """
  n1, dim1 = X1.shape
  n2, dim2 = X2.shape
  if dim1 != dim2:
    raise ValueError('Second dimension of X1 and X2 should be equal.')
  dist_sq = (np.outer(np.ones(n1), (X2**2).sum(axis=1))
             + np.outer((X1**2).sum(axis=1), np.ones(n2))
             - 2*X1.dot(X2.T))
  dist_sq = np.clip(dist_sq, 0.0, np.inf)
  return dist_sq


def pareto_dominates(u, v, tolerance=None):
  """ Returns true if u >= v elementwise, and at least
      one of the elements is not an equality.
  """
  u = np.asarray(u, dtype=np.float32)
  v = np.asarray(v, dtype=np.float32)
  if tolerance is None:
    return np.all(u >= v) and not np.all(u == v)
  else:
    return np.all(u >= v) and not np.linalg.norm(u - v) <= tolerance


def update_pareto_set(vals, points, new_val, new_point):
  """ Updates the current pareto optimal values and points with the
      new value and new point. vals is assummed to be a non-dominated
      set. The returned updated vals and points is guaranteed to
      be non-dominated provided the original values were
      non-dominated.
  """
  num_points = len(points)
  new_vals = []
  new_points = []
  for i in range(num_points):
    # If new_val does not dominate vals[i], keep it
    if not pareto_dominates(new_val, vals[i]):
      new_vals.append(vals[i])
      new_points.append(points[i])
  # Check if new_vals is not dominated by any vals[i].
  # If so, then keep it.
  dominated = False
  for i in range(num_points):
    if pareto_dominates(vals[i], new_val):
      dominated = True
      break
  if not dominated:
    new_points.append(new_point)
    new_vals.append(new_val)
  return new_vals, new_points


def pairwise_hamming_kernel(X1, X2, weights=None):
  """ Computes the pairwise hamming kernels between X1 and X2.
  """
  # An internal function to compute weighted distances between an array and a vector.
  def _compute_dist_between_2D_1D_array(arr_2d, arr_1d, wts):
    """ Returns distance between 2D and 1D array. """
    return (np.equal(arr_2d, arr_1d) * wts).sum(axis=1)
  # An internal function to compute weighted distances between a large array and
  # a small array
  def _compute_dist_between_large_small_array(arr_large, arr_small, wts):
    """ Returns distances between large and small arrays. """
    ret = np.zeros((arr_large.shape[0], arr_small.shape[0]))
    for idx, small_vec in enumerate(arr_small):
      ret[:, idx] = _compute_dist_between_2D_1D_array(arr_large, small_vec, wts)
    return ret
  # ---------------------------------------------------------------------------------
  if len(X1) == 0 and len(X2) == 0:
    return np.zeros((0, 0))
  elif len(X1) == 0:
    return np.zeros((0, len(X2)))
  elif len(X2) == 0:
    return np.zeros((len(X1), 0))
  else:
    if weights is None:
      dim = len(X1[0])
      weights = np.ones((dim, ))/float(dim)
    X1 = np.array(X1, dtype=np.object)
    X2 = np.array(X2, dtype=np.object)
    n1, _ = X1.shape
    n2, _ = X2.shape
    if n2 < n1:
      return _compute_dist_between_large_small_array(X2, X1, weights).T
    else:
      return _compute_dist_between_large_small_array(X1, X2, weights)


# Some linear algebraic utilities
def project_symmetric_to_psd_cone(M, is_symmetric=True, epsilon=0):
  """ Projects the symmetric matrix M to the PSD cone. """
  if is_symmetric:
    try:
      eigvals, eigvecs = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
      print('LinAlgError encountered with eigh. Defaulting to eig.')
      eigvals, eigvecs = np.linalg.eig(M)
      eigvals = np.real(eigvals)
      eigvecs = np.real(eigvecs)
  else:
    eigvals, eigvecs = np.linalg.eig(M)
  clipped_eigvals = np.clip(eigvals, epsilon, np.inf)
  return (eigvecs * clipped_eigvals).dot(eigvecs.T)


def stable_cholesky(M, add_to_diag_till_psd=True):
  """ Returns L, a 'stable' cholesky decomposition of M. L is lower triangular and
      satisfies L*L' = M.
      Sometimes nominally psd matrices are not psd due to numerical issues. By adding a
      small value to the diagonal we can make it psd. This is what this function does.
      Use this iff you know that K should be psd. We do not check for errors.
  """
  _printed_warning = False
  if M.size == 0:
    return M # if you pass an empty array then just return it.
  try:
    # First try taking the Cholesky decomposition.
    L = np.linalg.cholesky(M)
  except np.linalg.linalg.LinAlgError as e:
    # If it doesn't work, then try adding diagonal noise.
    if not add_to_diag_till_psd:
      raise e
    diag_noise_power = -11
    max_M = np.diag(M).max()
    diag_noise = np.diag(M).max() * 1e-11
    chol_decomp_succ = False
    while not chol_decomp_succ:
      try:
        diag_noise = (10 ** diag_noise_power) * max_M
        L = np.linalg.cholesky(M + diag_noise * np.eye(M.shape[0]))
        chol_decomp_succ = True
      except np.linalg.linalg.LinAlgError:
        if diag_noise_power > -9 and not _printed_warning:
          from warnings import warn
          warn(('Could not compute Cholesky decomposition despite adding %0.4f to the '
                'diagonal. This is likely because the M is not positive semi-definite.')%(
               (10**diag_noise_power) * max_M))
          _printed_warning = True
        diag_noise_power += 1
      if diag_noise_power >= 5:
        raise ValueError(('Could not compute Cholesky decomposition despite adding' +
                          ' %0.4f to the diagonal. This is likely because the M is not ' +
                          'positive semi-definite or has infinities/nans.')%(diag_noise))
  return L


# Solving triangular matrices ------------------------------------------------------------
def _solve_triangular_common(A, b, lower):
  """ Solves Ax=b when A is a triangular matrix. """
  if A.size == 0 and b.shape[0] == 0:
    return np.zeros((b.shape))
  else:
    return solve_triangular(A, b, lower=lower)

def solve_lower_triangular(A, b):
  """ Solves Ax=b when A is lower triangular. """
  return _solve_triangular_common(A, b, lower=True)

def solve_upper_triangular(A, b):
  """ Solves Ax=b when A is upper triangular. """
  return _solve_triangular_common(A, b, lower=False)


def draw_gaussian_samples(num_samples, mu, K):
  """ Draws num_samples samples from a Gaussian distribution with mean mu and
      covariance K.
  """
  num_pts = len(mu)
  L = stable_cholesky(K)
  U = np.random.normal(size=(num_pts, num_samples))
  V = L.dot(U).T + mu
  return V


# Executing a Pythong string -------------------------------------------------------
def evaluate_strings_with_given_variables(strs_to_execute, variable_dict=None):
  """ Executes a list of python strings and returns the results as a list.
      variable_dict is a dictionary mapping strings to values which may be used in
      str_to_execute.
  """
  if variable_dict is None:
    variable_dict = {}
  if not isinstance(strs_to_execute, list):
    got_list_of_constraints = False
    strs_to_execute = [strs_to_execute]
  else:
    got_list_of_constraints = True
  # Now set local variables
  for key, value in variable_dict.items():
    locals()[key] = value
  ret = []
  for elem in strs_to_execute:
    curr_result = eval(elem)
    ret.append(curr_result)
  if got_list_of_constraints:
    return ret
  else:
    return ret[0]

# Matrix/Array/List utilities ------------------------------------------------------
def get_nonzero_indices_in_vector(vec):
  """ Returns the nonzero indices in the vector vec. """
  if not isinstance(vec, np.ndarray):
    vec = np.asarray(vec.todense()).ravel()
  ret, = vec.nonzero()
  return ret

def reorder_list_or_array(M, ordering):
  """ Reorders a list or array like object. """
  if isinstance(M, list):
    return [M[i] for i in ordering]
  else:
    return M[ordering]

def reorder_list(L, ordering):
  """ reorders a list. """
  return reorder_list_or_array(L, ordering)

def get_original_order_from_reordered_list(L, ordering):
  """ Returns the original order from a reordered list. """
  ret = [None] * len(ordering)
  for orig_idx, ordered_idx in enumerate(ordering):
    ret[ordered_idx] = L[orig_idx]
  return ret

def flatten_nested_lists(L):
  """ Each element of L could be a list or any object other than a list. This function
      returns a list of non-list objects, where the internal lists or lists_of_lists
      have been 'flattened'.
  """
  ret = []
  for elem in L:
    if isinstance(elem, list):
      ret.extend(flatten_nested_lists(elem))
    else:
      ret.append(elem)
  return ret

def flatten_list_of_objects_and_lists(L):
  """ Each element of L could be a non-list object or a list of non-objects. This function
      returns a list of non-list objects, where the internal lists have been 'flattened'.
  """
  ret = []
  for elem in L:
    if isinstance(elem, list):
      ret.extend(elem)
    else:
      ret.append(elem)
  return ret

def flatten_list_of_objects_and_iterables(L):
  """ Each element of L could be a non-list object or a list of non-objects. This function
      returns a list of non-list objects, where the internal lists have been 'flattened'.
  """
  ret = []
  for elem in L:
    if hasattr(elem, '__iter__') and not isinstance(elem, str):
      ret.extend(elem)
    else:
      ret.append(elem)
  return ret

def flatten_list_of_lists(L):
  """ Flattens a list of lists to return a single list of objects. """
  return [item for sublist in L for item in sublist]

def reorder_rows_and_cols_in_matrix(M, ordering):
  """ Reorders the rows and columns in matrix M. """
  array_type = type(M)
  if array_type == dok_matrix: # Check if a sparse matrix to convert to array
    M = np.asarray(M.todense())
  elif array_type == list:
    M = np.array(M)
  # Now do the reordering
  M = M[:, ordering][ordering]
  # Convert back
  if array_type == dok_matrix: # Check if a sparse matrix for return
    M = dok_matrix(M)
  elif array_type == list:
    M = [list(m) for m in M]
  return M

def _set_coords_to_val(A, coords, val):
  """ Sets the indices in matrix A to value. """
  for coord in coords:
    A[coord[0], coord[1]] = val

def get_dok_mat_with_set_coords(n, coords):
  """ Returns a sparse 0 matrix with the coordinates in coords set to 1. """
  A = dok_matrix((n, n))
  _set_coords_to_val(A, coords, 1)
  return A

def block_augment_array(A, B, C, D):
  """ Given a n1xn2 array A, an n1xn3 array B, an n4xn5 array C, and a
      n4x(n2 + n3 - n5) array D, this returns (n1+n4)x(n2+n3) array of the form
      [A, B; C; D].
  """
  AB = np.hstack((A, B))
  CD = np.hstack((C, D))
  return np.vstack((AB, CD))


# For sampling based on fitness values -------------------------------------------------
# We are using them in the GA and BO algorithms.
def get_exp_probs_from_fitness(fitness_vals, scaling_param=None, scaling_const=None):
  """ Returns sampling probabilities from fitness values; the fitness values are
      exponentiated and used as probabilities.
  """
  fitness_vals = np.array(fitness_vals)
  if scaling_param is None:
    scaling_const = scaling_const if scaling_const is not None else 0.5
    scaling_param = scaling_const * (fitness_vals.std() + 0.0001)
  mean_param = fitness_vals.mean()
  exp_probs = np.exp((fitness_vals - mean_param)/scaling_param)
  return exp_probs/exp_probs.sum()

def sample_according_to_exp_probs(fitness_vals, num_samples, replace=False,
                                  scaling_param=None, scaling_const=None,
                                  sample_uniformly_if_fail=False):
  """ Samples after exponentiating the fitness values. """
  exp_probs = get_exp_probs_from_fitness(fitness_vals, scaling_param, scaling_const)
  if (not np.isfinite(exp_probs.sum())) and sample_uniformly_if_fail:
    exp_probs = np.ones((len(fitness_vals),)) / float(len(fitness_vals))
  return np.random.choice(len(fitness_vals), num_samples, p=exp_probs, replace=replace)

