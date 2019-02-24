"""
  Demo on the Shrunk Additive Least Squares Approximations method (SALSA) for high
  dimensional regression.
  -- kandasamy@cs.cmu.edu

  SALSA is Kernel Ridge Regression with special kind of kernel structure.
  We tune for the following parameters in the method.
    - Kernel type: {se, matern0.5, matern1.5, matern2.5}
    - Additive Order: An integer in (1, d) where d is the dimension.
    - Kernel scale: float
    - Bandwidths for each dimension: float
    - L2 Regularisation penalty: float

  If you use this experiment, please cite the following paper.
    - Kandasamy K, Yu Y, "Additive Approximations in High Dimensional Nonparametric
      Regression via the SALSA", International Conference on Machine Learning, 2016.
"""

# pylint: disable=invalid-name

from __future__ import print_function
import numpy as np
from scipy.linalg import solve_triangular


# Kernels defined here -------------------------------------------------------------------
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


class Kernel(object):
  """ A kernel class. """

  def __init__(self):
    """ Constructor. """
    super(Kernel, self).__init__()
    self.hyperparams = {}

  def __call__(self, X1, X2=None):
    """ Evaluates the kernel by calling evaluate. """
    return self.evaluate(X1, X2)

  def evaluate(self, X1, X2=None):
    """ Evaluates kernel values between X1 and X2 and returns an n1xn2 kernel matrix.
        This is a wrapper for _child_evaluate.
    """
    X2 = X1 if X2 is None else X2
    if len(X1) == 0 or len(X2) == 0:
      return np.zeros((len(X1), len(X2)))
    return self._child_evaluate(X1, X2)

  def _child_evaluate(self, X1, X2):
    """ Evaluates kernel values between X1 and X2 and returns an n1xn2 kernel matrix.
        This is to be implemented in a child kernel.
    """
    raise NotImplementedError('Implement in a child class.')

  def add_hyperparams(self, **kwargs):
    """ Set additional hyperparameters here. """
    for key, value in kwargs.items():
      self.hyperparams[key] = value


class SEKernel(Kernel):
  """ Squared exponential kernel. """

  def __init__(self, dim, scale=None, dim_bandwidths=None):
    """ Constructor. dim is the dimension. """
    super(SEKernel, self).__init__()
    self.dim = dim
    self.set_se_hyperparams(scale, dim_bandwidths)

  def set_dim_bandwidths(self, dim_bandwidths):
    """ Sets the bandwidth for each dimension. """
    if dim_bandwidths is not None:
      if len(dim_bandwidths) != self.dim:
        raise ValueError('Dimension of dim_bandwidths should be the same as dimension.')
      dim_bandwidths = np.array(dim_bandwidths).T
    self.add_hyperparams(dim_bandwidths=dim_bandwidths)

  def set_single_bandwidth(self, bandwidth):
    """ Sets the bandwidth of all dimensions to be the same value. """
    dim_bandwidths = None if bandwidth is None else [bandwidth] * self.dim
    self.set_dim_bandwidths(dim_bandwidths)

  def set_scale(self, scale):
    """ Sets the scale parameter for the kernel. """
    self.add_hyperparams(scale=scale)

  def set_se_hyperparams(self, scale, dim_bandwidths):
    """ Sets both the scale and the dimension bandwidths for the SE kernel. """
    self.set_scale(scale)
    if hasattr(dim_bandwidths, '__len__'):
      self.set_dim_bandwidths(dim_bandwidths)
    else:
      self.set_single_bandwidth(dim_bandwidths)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the SE kernel between X1 and X2 and returns the gram matrix. """
    scaled_X1 = self.get_scaled_repr(X1)
    scaled_X2 = self.get_scaled_repr(X2)
    dist_sq = dist_squared(scaled_X1, scaled_X2)
    K = self.hyperparams['scale'] * np.exp(-dist_sq/2)
    return K

  def get_scaled_repr(self, X):
    """ Returns the scaled version of an input by the bandwidths. """
    return X/self.hyperparams['dim_bandwidths']


class MaternKernel(Kernel):
  """ The Matern class of kernels. """
  # pylint: disable=abstract-method

  def __init__(self, dim, nu=None, scale=None, dim_bandwidths=None):
    """ Constructor. dim is the dimension. """
    super(MaternKernel, self).__init__()
    self.dim = dim
    self.p = None
    self.norm_constant = None
    self.set_matern_hyperparams(nu, scale, dim_bandwidths)

  def set_matern_hyperparams(self, nu, scale, dim_bandwidths):
    """ Sets the parameters of the matern kernel. """
    if nu%1 != 0.5:
      raise ValueError('Matern kernel: nu has to be p + 0.5 where p is an integer.')
    self.add_hyperparams(nu=nu)
    self.add_hyperparams(scale=scale)
    dim_bandwidths = dim_bandwidths if hasattr(dim_bandwidths, '__len__') else \
                     [dim_bandwidths] * self.dim
    dim_bandwidths = np.array(dim_bandwidths).T
    self.add_hyperparams(dim_bandwidths=dim_bandwidths)
    self.p = int(nu)
    self.norm_constant = 1.0 / self._eval_kernel_values_unnormalised(0)

  def get_scaled_repr(self, X):
    """ Returns scaled versions of the input by the bandwidths. """
    return X/self.hyperparams['dim_bandwidths']

  def _eval_kernel_values_unnormalised(self, dist):
    """ Given the distances, evaluates the unnormalised kernel value. """
    unnorm_K = np.zeros(dist.shape) if isinstance(dist, np.ndarray) else 0
    for i in range(self.p+1):
      coeff = np.math.factorial(self.p+i)/(np.math.factorial(i) *
                                           np.math.factorial(self.p-i))
      mult_mat = (np.sqrt(8 * self.hyperparams['nu']) * dist)
      unnorm_K += coeff * (mult_mat)**(self.p - i)
    # Finally multiply by the leading exponential
    unnorm_K *= (np.math.gamma(self.p+1) / np.math.gamma(2*self.p+1) *
                 np.exp(-np.sqrt(2 * self.hyperparams['nu']) * dist))
    return unnorm_K

  def _child_evaluate(self, X1, X2):
    """ Evaluates the SE kernel between X1 and X2 and returns the gram matrix. """
    scaled_X1 = self.get_scaled_repr(X1)
    scaled_X2 = self.get_scaled_repr(X2)
    dist = np.sqrt(dist_squared(scaled_X1, scaled_X2))
    unnorm_kernel_vals = self._eval_kernel_values_unnormalised(dist)
    K = self.hyperparams['scale'] * self.norm_constant * unnorm_kernel_vals
    return K


class ESPKernel(Kernel):
  """ Implements the ESP kernel from Kandasamy, Yu 2016, 'Additive Approximations in High
      Dimensional Nonparametric Regression. """

  def __init__(self, scale, order, kernel_list):
    super(ESPKernel, self).__init__()
    self.dim = len(kernel_list)
    self.kernel_list = kernel_list
    self.add_hyperparams(scale=scale, order=order)

    if self.dim < 0:
      raise ValueError("dim cannot not be negative.")
    if order > self.dim:
      raise ValueError("order must be less than or equal to dim.")
    if order < 1:
      raise ValueError("order must be an integer between 1 and dim.")

  def _child_evaluate(self, X1, X2):
    "computes the gram matrix"
    # pylint: disable=too-many-locals
    order = self.hyperparams["order"]
    kernel_list = self.kernel_list
    scale = self.hyperparams["scale"]
    if isinstance(X1, list):
      X1 = np.array(X1)
    if isinstance(X2, list):
      X2 = np.array(X2)
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    kernel_matrices = []
    for i in range(self.dim):
      kernel = kernel_list[i]
      kernel_matrices.append(kernel(X1[:, i:i+1], X2[:, i:i+1]))
    power_sum = [np.zeros((n1, n2)) for _ in range(order + 1)]
    ones = np.ones((n1, n2))
    power_sum[0] = ones
    for i in range(1, order + 1):
      for matrix in kernel_matrices:
        power_sum[i] += matrix ** i
    # elementary symmetric polynomials
    esp = [np.zeros((n1, n2)) for _ in range(order + 1)]
    esp[0] = ones
    for m in range(1, order + 1):
      for i in range(1, m + 1):
        esp[m] += ((-1) ** (i - 1)) * esp[m - i] * power_sum[i]
      esp[m] /= m
    return scale * esp[order]


class ESPKernelSE(ESPKernel):
  """Implements ESP kernel with SE kernel for each dimension"""
  def __init__(self, dim, scale, order, dim_bandwidths):
    kernel_list = []
    for i in range(dim):
      kernel_list.append(SEKernel(1, 1.0, np.asscalar(dim_bandwidths[i])))
    super(ESPKernelSE, self).__init__(scale, order, kernel_list)


class ESPKernelMatern(ESPKernel):
  """Implements ESP kernel with Matern kernel for each dimension"""
  def __init__(self, dim, nu, scale, order, dim_bandwidths):
    kernel_list = []
    for i in range(dim):
      kernel_list.append(MaternKernel(1, nu[i], 1.0, np.asscalar(dim_bandwidths[i])))
    super(ESPKernelMatern, self).__init__(scale, order, kernel_list)


# SALSA class defined here ---------------------------------------------------------------
def _check_feature_label_lengths_and_format(X, Y):
  """ Checks if the length of X and Y are the same. """
  if len(X) != len(Y):
    raise ValueError('Length of X (' + len(X) + ') and Y (' + \
      len(Y) + ') do not match.')


# Solving triangular matrices -----------------------
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


def stable_cholesky(M, add_to_diag_till_psd=True):
  """ Returns L, a 'stable' cholesky decomposition of M. L is lower triangular and
      satisfies L*L' = M.
      Sometimes nominally psd matrices are not psd due to numerical issues. By adding a
      small value to the diagonal we can make it psd. This is what this function does.
      Use this iff you know that K should be psd. We do not check for errors.
  """
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
        L = np.linalg.cholesky(M +
            ((10**diag_noise_power) * max_M)  * np.eye(M.shape[0]))
        chol_decomp_succ = True
      except np.linalg.linalg.LinAlgError:
        diag_noise_power += 1
      if diag_noise_power >= 5:
        print('**************** Cholesky failed: Added diag noise = %e'%(diag_noise))
  return L


class SALSA(object):
  """ Implements the ESP kernel from Kandasamy, Yu 2016, 'Additive Approximations in High
      Dimensional Nonparametric Regression. """

  def __init__(self, X, Y, kernel, mean_func, l2_pen_coeff):
    """ Constructor. """
    super(SALSA, self).__init__()
    _check_feature_label_lengths_and_format(X, Y)
    self.set_data(X, Y)
    self.kernel = kernel
    self.mean_func = mean_func
    self.l2_pen_coeff = l2_pen_coeff
    # Initialise other attributes we will need.
    self.L = None
    self.alpha = None
    # Compute parameters needed for test time.
    self._compute_estimation_parameters()

  def set_data(self, X, Y):
    """ Sets the data to X and Y. """
    self.X = list(X)
    self.Y = list(Y)
    self.num_tr_data = len(self.Y)

  def _compute_estimation_parameters(self):
    """ Computes parameters that can be used during test time. """
    K_trtr_wo_noise = self.kernel(self.X, self.X)
    K_trtr_w_noise = K_trtr_wo_noise + \
                     self.l2_pen_coeff * np.eye(K_trtr_wo_noise.shape[0])
    self.L = stable_cholesky(K_trtr_w_noise)
    Y_centred = self.Y - self.mean_func(self.X)
    self.alpha = solve_upper_triangular(self.L.T,
                                        solve_lower_triangular(self.L, Y_centred))

  def eval(self, X_test):
    """ Evaluates SALSA on X_test.
    """
    test_mean = self.mean_func(X_test)
    K_tetr = self.kernel(X_test, self.X)
    predictions = test_mean + K_tetr.dot(self.alpha)
    return predictions


# Training and Validation -----------------------------------------------------------
def get_salsa_kernel_from_params(kernel_type, add_order, kernel_scale, bandwidths,
                                 problem_dim):
  """ Returns the kernel for SALSA. """
  if kernel_type == 'se':
    return ESPKernelSE(problem_dim, kernel_scale, add_order, bandwidths)
  elif kernel_type.startswith('matern'):
    nu = float(kernel_type[-3:])
    nu_vals = [nu] * problem_dim
    return ESPKernelMatern(problem_dim, nu_vals, kernel_scale, add_order, bandwidths)
  else:
    raise ValueError('Unknown kernel type %s.'%(kernel_type))


def get_salsa_estimator_from_data_and_hyperparams(X_tr, Y_tr, kernel_type, add_order,
                                                  kernel_scale, bandwidths, l2_reg):
  """ Returns an estimator using the data. """
  problem_dim = np.array(X_tr).shape[1]
  kernel = get_salsa_kernel_from_params(kernel_type, add_order, kernel_scale,
                                        bandwidths, problem_dim)
  def _get_mean_func(_mean_func_const_value):
    """ Returns the mean function from the constant value. """
    return lambda x: np.array([_mean_func_const_value] * len(x))
  mean_func = _get_mean_func(np.median(Y_tr))
  salsa_obj = SALSA(X_tr, Y_tr, kernel, mean_func, l2_reg)
  return salsa_obj.eval


def salsa_train_and_validate(X_tr, Y_tr, X_va, Y_va, kernel_type, add_order,
                             kernel_scale, bandwidths, l2_reg,
                             num_tr_data_to_use=None, num_va_data_to_use=None,
                             shuffle_data_when_using_a_subset=True):
  """ Train and return validation error for SALSA. """
  # pylint: disable=too-many-arguments
  # Prelims
  if num_tr_data_to_use is None:
    num_tr_data_to_use = len(X_tr)
  if num_va_data_to_use is None:
    num_va_data_to_use = len(X_va)
  num_tr_data_to_use = int(num_tr_data_to_use)
  num_va_data_to_use = int(num_va_data_to_use)
  if num_tr_data_to_use < len(X_tr) and shuffle_data_when_using_a_subset:
    X_tr = np.copy(X_tr)
    np.random.shuffle(X_tr)
  if num_va_data_to_use < len(X_va) and shuffle_data_when_using_a_subset:
    X_va = np.copy(X_va)
    np.random.shuffle(X_va)
  # Get relevant subsets
  X_tr = X_tr[:num_tr_data_to_use]
  Y_tr = Y_tr[:num_tr_data_to_use]
  X_va = X_va[:num_va_data_to_use]
  Y_va = Y_va[:num_va_data_to_use]
  # Get estimator
  salsa_estimator = get_salsa_estimator_from_data_and_hyperparams(X_tr, Y_tr,
                      kernel_type, add_order, kernel_scale, bandwidths, l2_reg)
  valid_predictions = salsa_estimator(X_va)
  valid_diffs = (valid_predictions - Y_va) ** 2
  avg_valid_err = valid_diffs.mean()
  print('Trained with %d data, and validated with %d data. err=%0.4f'%(
        len(X_tr), len(X_va), avg_valid_err))
  return avg_valid_err

