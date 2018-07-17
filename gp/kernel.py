"""
  Implements various kernels.
  -- kandasamy@cs.cmu.edu
  -- kvysyara@andrew.cmu.edu
"""

from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=abstract-method

from builtins import zip
from builtins import object
import numpy as np
# Local imports
from utils.ancillary_utils import get_list_of_floats_as_str
from utils.general_utils import dist_squared


# Utilities -------
def _get_se_matern_scale_bw_strs(kern):
  """ Gets descriptions of the bandwidths and scales for the SE and matern kernels. """
  if kern.dim > 6:
    bw_str = 'avg-bw: %0.4f'%(kern.hyperparams['dim_bandwidths'].mean())
  else:
    bw_str = 'bws:[' + ' '.join(['%0.2f'%(dbw) for dbw in
                                 kern.hyperparams['dim_bandwidths']]) + ']'
  scale_str = 'sc:%0.4f'%(kern.hyperparams['scale'])
  return scale_str, bw_str


class Kernel(object):
  """ A kernel class. """

  def __init__(self):
    """ Constructor. """
    super(Kernel, self).__init__()
    self.hyperparams = {}

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    raise NotImplementedError('Implement in a child class.')

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

  def evaluate_from_dists(self, dists):
    """ Evaluates the kernel from pairwise distances. """
    raise NotImplementedError('Implement in a child class.')

  def set_hyperparams(self, **kwargs):
    """ Set hyperparameters here. """
    self.hyperparams = kwargs

  def add_hyperparams(self, **kwargs):
    """ Set additional hyperparameters here. """
    for key, value in kwargs.items():
      self.hyperparams[key] = value

  def get_effective_norm(self, X, order=None, *args, **kwargs):
    """ Gets the effective norm scaled by bandwidths. """
    raise NotImplementedError('Implement in a child class.')

  def compute_std_slack(self, X1, X2):
    """ Computes a bound on the maximum standard deviation between X1 and X2. """
    raise NotImplementedError('Implement in a child class.')

  def change_smoothness(self, factor):
    """ Decreases smoothness by the factor given. """
    raise NotImplementedError('Implement in a child class.')

  def gradient(self, param, X1, X2=None, *args):
    """ Computes gradient of kernel w.r.t the param """
    if len(X1) == 0 or len(X2) == 0:
      return np.zeros((len(X1), len(X2)))
    X2 = X1 if X2 is None else X2
    return self._child_gradient(param, X1, X2, *args)

  def _child_gradient(self, param, X1, X2, param_num=None):
    """ Computes gradient of kernel w.r.t the param """
    raise NotImplementedError('Implement in a child class.')

  def __str__(self):
    """ Returns a string representation of the kernel. """
    return '%s:: %s'%(type(self), str(self.hyperparams))


class SEKernel(Kernel):
  """ Squared exponential kernel. """

  def __init__(self, dim, scale=None, dim_bandwidths=None):
    """ Constructor. dim is the dimension. """
    super(SEKernel, self).__init__()
    self.dim = dim
    self.set_se_hyperparams(scale, dim_bandwidths)

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return True

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

  def get_effective_norm(self, X, order=None, is_single=True):
    """ Gets the effective norm. That is the norm of X scaled by bandwidths. """
    # pylint: disable=arguments-differ
    scaled_X = self.get_scaled_repr(X)
    if is_single:
      return np.linalg.norm(scaled_X, ord=order)
    else:
      return np.array([np.linalg.norm(sx, ord=order) for sx in scaled_X])

  def compute_std_slack(self, X1, X2):
    """ Computes a bound on the maximum standard deviation diff between X1 and X2. """
    k_12 = np.array([float(self.evaluate(X1[i].reshape(1, -1), X2[i].reshape(1, -1)))
                     for i in range(len(X1))])
    return np.sqrt(self.hyperparams['scale'] - k_12)

  def change_smoothness(self, factor):
    """ Decreases smoothness by the given factor. """
    self.hyperparams['dim_bandwidths'] *= factor

  def _child_gradient(self, param, X1, X2, param_num=None):
    """ Computes gradient of kernel w.r.t the param """
    scaled_X1 = self.get_scaled_repr(X1)
    scaled_X2 = self.get_scaled_repr(X2)
    dist_sq = dist_squared(scaled_X1, scaled_X2)
    if param == 'scale':
      return self.hyperparams['scale'] * np.exp(-dist_sq/2)
    elif param == 'same_dim_bandwidths':
      dist_sq_dv = dist_sq/(self.hyperparams['dim_bandwidths'][0, 0])
      return self.hyperparams['scale'] * (np.multiply(dist_sq_dv, np.exp(-dist_sq/2)))
    else:
      dim_X1 = np.expand_dims(scaled_X1[:, param_num], axis=1)
      dim_X2 = np.expand_dims(scaled_X2[:, param_num], axis=1)
      dim_sq = dist_squared(dim_X1, dim_X2)
      dim_sq = dim_sq/(self.hyperparams['dim_bandwidths'][0, param_num])
      return self.hyperparams['scale'] * (np.multiply(dim_sq, np.exp(-dist_sq/2)))

  def __str__(self):
    """ Returns a string representation of the kernel. """
    scale_str, bw_str = _get_se_matern_scale_bw_strs(self)
    return 'SE: ' + scale_str + ' ' + bw_str


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

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return True

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

  def _eval_grad_kernel_values_unnormalised(self, dist, dist_dv):
    """ Given the distances, evaluates the gradient of unnormalised kernel value. """
    unnorm_K = np.zeros(dist.shape) if isinstance(dist, np.ndarray) else 0
    unnorm_K_dv = np.zeros(dist_dv.shape) if isinstance(dist_dv, np.ndarray) else 0
    for i in range(self.p+1):
      coeff = np.math.factorial(self.p+i)/(np.math.factorial(i) *
                                           np.math.factorial(self.p-i))
      mult_mat = (np.sqrt(8 * self.hyperparams['nu']) * dist)
      unnorm_K += coeff * (mult_mat)**(self.p - i)
      if self.p - i > 0:
        unnorm_K_dv += (np.sqrt(8 * self.hyperparams['nu'])) * (self.p - i) * \
                       coeff * ((mult_mat)**(self.p - i - 1)) * dist_dv
    # Finally multiply by the leading exponential
    unnorm_K_dv *= (np.math.gamma(self.p+1) / np.math.gamma(2*self.p+1) *
                    np.exp(-np.sqrt(2 * self.hyperparams['nu']) * dist))
    unnorm_K *= (np.math.gamma(self.p+1) / np.math.gamma(2*self.p+1) *
                 np.exp(-np.sqrt(2 * self.hyperparams['nu']) * dist) *
                 (-np.sqrt(2 * self.hyperparams['nu']) * dist_dv))
    return unnorm_K + unnorm_K_dv

  def _child_evaluate(self, X1, X2):
    """ Evaluates the SE kernel between X1 and X2 and returns the gram matrix. """
    scaled_X1 = self.get_scaled_repr(X1)
    scaled_X2 = self.get_scaled_repr(X2)
    dist = np.sqrt(dist_squared(scaled_X1, scaled_X2))
    unnorm_kernel_vals = self._eval_kernel_values_unnormalised(dist)
    K = self.hyperparams['scale'] * self.norm_constant * unnorm_kernel_vals
    return K

  def _child_gradient(self, param, X1, X2, param_num=None):
    """ Computes gradient of kernel w.r.t the param. """
    scaled_X1 = self.get_scaled_repr(X1)
    scaled_X2 = self.get_scaled_repr(X2)
    dist = np.sqrt(dist_squared(scaled_X1, scaled_X2))
    if param == 'scale':
      unnorm_kernel_vals = self._eval_kernel_values_unnormalised(dist)
      return self.hyperparams['scale'] * self.norm_constant * unnorm_kernel_vals
    elif param == 'same_dim_bandwidths':
      dist_dv = -(dist/(self.hyperparams['dim_bandwidths'][0, 0]))
      unnorm_kernel_vals = self._eval_grad_kernel_values_unnormalised(dist, dist_dv)
      return self.hyperparams['scale'] * self.norm_constant * unnorm_kernel_vals
    else:
      dim_X1 = np.expand_dims(scaled_X1[:, param_num], axis=1)
      dim_X2 = np.expand_dims(scaled_X2[:, param_num], axis=1)
      np.fill_diagonal(dist, 1.0)
      dist_dv = 1/dist
      np.fill_diagonal(dist, 0.0)
      dim_sq = dist_squared(dim_X1, dim_X2)
      dist_dv *= -(dim_sq/(self.hyperparams['dim_bandwidths'][0, param_num]))
      unnorm_kernel_vals = self._eval_grad_kernel_values_unnormalised(dist, dist_dv)
      return self.hyperparams['scale'] * self.norm_constant * unnorm_kernel_vals

  def __str__(self):
    """ Returns a string representation of the kernel. """
    scale_str, bw_str = _get_se_matern_scale_bw_strs(self)
    nu_str = 'nu=%0.1f'%(self.hyperparams['nu'])
    return 'Matern: ' + nu_str + ' ' + scale_str + ' ' + bw_str


class PolyKernel(Kernel):
  """ The polynomial kernel. """
  # pylint: disable=abstract-method

  def __init__(self, dim, order, scale, dim_scalings=None):
    """ Constructor. """
    super(PolyKernel, self).__init__()
    self.dim = dim
    self.set_poly_hyperparams(order, scale, dim_scalings)

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return True

  def set_order(self, order):
    """ Sets the order of the polynomial. """
    self.add_hyperparams(order=order)

  def set_scale(self, scale):
    """ Sets the scale of the kernel. """
    self.add_hyperparams(scale=scale)

  def set_dim_scalings(self, dim_scalings):
    """ Sets the scaling for each dimension in the polynomial kernel. This will be a
        dim+1 dimensional vector.
    """
    if dim_scalings is not None:
      if len(dim_scalings) != self.dim:
        raise ValueError('Dimension of dim_scalings should be dim.')
      dim_scalings = np.array(dim_scalings)
    self.add_hyperparams(dim_scalings=dim_scalings)

  def set_single_scaling(self, scaling):
    """ Sets the same scaling for all dimensions. """
    if scaling is None:
      self.set_dim_scalings(None)
    else:
      self.set_dim_scalings([scaling] * self.dim)

  def set_poly_hyperparams(self, order, scale, dim_scalings):
    """Sets the hyper parameters. """
    self.set_order(order)
    self.set_scale(scale)
    if hasattr(dim_scalings, '__len__'):
      self.set_dim_scalings(dim_scalings)
    else:
      self.set_single_scaling(dim_scalings)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the polynomial kernel and returns and the gram matrix. """
    X1 = X1 * self.hyperparams['dim_scalings']
    X2 = X2 * self.hyperparams['dim_scalings']
    K = self.hyperparams['scale'] * ((X1.dot(X2.T) + 1)**self.hyperparams['order'])
    return K

  def __str__(self):
    """ Returns a string representation of the kernel. """
    dim_scalings_str = ','.join(['%0.2f'%(elem) for elem in
                                 self.hyperparams['dim_scalings']])
    return 'Poly: d=%d, scale=%0.2f, %s'%(self.hyperparams['order'], \
            self.hyperparams['scale'], dim_scalings_str)


class ExpDecayKernel(Kernel):
  """ A kernel for exponentially decaying functions. Taken from
      Freeze-Thaw Bayesian Optimization, Swersky et al.
  """
  # TODO: write unit tests for ExpDecay Kernel

  def __init__(self, dim, scale=None, offset=None, powers=None):
    """ Constructor. """
    super(ExpDecayKernel, self).__init__()
    self.dim = dim
    if not hasattr(powers, '__iter__'):
      powers = [powers] * dim
    self.set_hyperparams(scale=scale, offset=offset, powers=powers)

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return True

  def _child_evaluate(self, X1, X2):
    """ Evaluates the child kernel between X1 and X2 and returns the gram matrix. """
    if isinstance(X1, list):
      X1 = np.array(X1)
    if isinstance(X2, list):
      X2 = np.array(X2)
    powers = self.hyperparams['powers']
    ret = self.hyperparams['scale'] * np.ones((X1.shape[0], X2.shape[0]))
    for dim_idx in range(self.dim):
      dim_X1 = X1[:, dim_idx]
      dim_X2 = X2[:, dim_idx]
      dim_kernel = 1 / (1 + np.add.outer(dim_X1, dim_X2)) ** powers[dim_idx]
      ret *= dim_kernel
    ret += self.hyperparams['offset']
    return ret

  def __str__(self):
    """ Return a string representation. """
    return 'ExpDec: sc=%0.3f, offset=%0.3f, pow=%s'%(self.hyperparams['scale'], \
        self.hyperparams['offset'], get_list_of_floats_as_str(self.hyperparams['powers']))


# Derivative Kernels ====================================================================
class AdditiveKernel(Kernel):
  """Implements an Additive kernel on Euclidean Spaces with non-overlapping groups. """
  # pylint: disable=abstract-method

  def __init__(self, scale, kernel_list, groupings):
    """ Constructor:
        kernel_list is a list of kernel objects.
        groupings is a list of list first dimension is the group second dimension is the
        set of dimensions in the group.
    """
    if len(kernel_list) != len(groupings):
      raise ValueError("number of kernels do not correspond to number of groups.")
    super(AdditiveKernel, self).__init__()
    self.kernel_list = kernel_list
    self.groupings = groupings
    self.add_hyperparams(scale=scale)
    self.dim = sum([kern.dim for kern in self.kernel_list])

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. Returns True if each kernel is is_guaranteed_psd()"""
    return all([kern.is_guaranteed_psd() for kern in self.kernel_list])

  def _child_evaluate(self, X1, X2):
    "computes the gram matrix"
    # Convert them to arrays so that we can index the groups out.
    X1 = np.array(X1)
    X2 = np.array(X2)
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    result = np.zeros((n1, n2))
    for kernel, group in zip(self.kernel_list, self.groupings):
      result += kernel(X1[:, group], X2[:, group])
    return self.hyperparams['scale'] * result

  def __str__(self):
    """ Returns a string representation of the kernel. """
    kernels_str_list = ['%s(%s)'%(grp, kern) for (grp, kern) in
                        zip(self.groupings, self.kernel_list)]
    kernels_str = ', '.join(kernels_str_list)
    return 'ADD scale=%0.2f, '%(self.hyperparams['scale']) + kernels_str


class DomainProductKernel(Kernel):
  """ Implements a product kernel which takes the product of all kernels in
      the kernel list.
  """

  def __init__(self, scale, kernel_list):
    """ kernel_list is a list of kernel objects.
        We will assume that when computing the kernel, each X is a list of secondary
        lists where the j'th in a secondary list corresponds to the j'th kernel.
    """
    super(DomainProductKernel, self).__init__()
    self.kernel_list = kernel_list
    self.add_hyperparams(scale=scale)

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. Returns True if each kernel is is_guaranteed_psd()"""
    return all([kern.is_guaranteed_psd() for kern in self.kernel_list])

  @classmethod
  def _get_idxs_from_list_of_lists(cls, list_of_lists, idx):
    """ Returns a list of objects. """
    return [elem[idx] for elem in list_of_lists]

  def _child_evaluate(self, X1, X2):
    """ Computes the Gram Matrix. """
    n1 = len(X1)
    n2 = len(X2)
    result = self.hyperparams['scale'] * np.ones((n1, n2))
    for idx, kern in enumerate(self.kernel_list):
      curr_X1 = self._get_idxs_from_list_of_lists(X1, idx)
      curr_X2 = self._get_idxs_from_list_of_lists(X2, idx)
      result *= kern(curr_X1, curr_X2)
    return result

  def __str__(self):
    """ Returns a string representation of the kernel. """
    kernels_str = ', '.join([str(kern) for kern in self.kernel_list])
    return 'DomProd scale=%0.2f, '%(self.hyperparams['scale']) + kernels_str


class CoordinateProductKernel(Kernel):
  """ Implements a coordinatewise product kernel. """
  # pylint: disable=abstract-method

  def __init__(self, dim, scale, kernel_list=None, coordinate_list=None):
    """ Constructor.
        kernel_list is a list of n Kernel objects. coordinate_list is a list of n lists
        each indicating the coordinates each kernel in kernel_list should be applied to.
    """
    super(CoordinateProductKernel, self).__init__()
    self.dim = dim
    self.add_hyperparams(scale=scale)
    self.kernel_list = kernel_list
    self.coordinate_list = coordinate_list

  def set_kernel_list(self, kernel_list):
    """ Sets a new list of kernels. """
    self.kernel_list = kernel_list

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. Returns True if each kernel is is_guaranteed_psd()"""
    return all([kern.is_guaranteed_psd() for kern in self.kernel_list])

  def set_new_kernel(self, kernel_idx, new_kernel):
    """ Sets new_kernel to kernel_list[kernel_idx]. """
    self.kernel_list[kernel_idx] = new_kernel

  def set_kernel_hyperparams(self, kernel_idx, **kwargs):
    """ Sets the hyper-parameters for kernel_list[kernel_idx]. """
    self.kernel_list[kernel_idx].set_hyperparams(**kwargs)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the combined kernel. """
    n1 = len(X1)
    n2 = len(X2)
    X1 = np.array(X1)
    X2 = np.array(X2)
    K = self.hyperparams['scale'] * np.ones((n1, n2))
    for idx, kernel in enumerate(self.kernel_list):
      X1_curr = X1[:, self.coordinate_list[idx]]
      X2_curr = X2[:, self.coordinate_list[idx]]
      K *= kernel(X1_curr, X2_curr)
    return K

  def __str__(self):
    """ Returns a string representation of the kernel. """
    kernels_str_list = ['%s(%s)'%(grp, kern) for (grp, kern) in
                        zip(self.coordinate_list, self.kernel_list)]
    kernels_str = ', '.join(kernels_str_list)
    return 'CoordProd scale=%0.2f, '%(self.hyperparams['scale']) + kernels_str


class ExpSumOfDistsKernel(Kernel):
  """ Given a function that returns a list of distances d1, d2, ... dk, this kernel
      takes the form exp(beta1*d1^p + beta2*d2^p + ... + betak*dk^p. """
  # pylint: disable=abstract-method

  def __init__(self, dist_computer, betas, scale, powers=1, num_dists=None,
               dist_is_hilbertian=False):
    """ Constructor.
          trans_dist_computer: Given two lists of networks X1 and X2, trans_dist_computer
            is a function which returns a list of n1xn2 matrices where ni=len(Xi).
    """
    super(ExpSumOfDistsKernel, self).__init__()
    self.num_dists = num_dists if num_dists is not None else len(betas)
    self.dist_computer = dist_computer
    betas = betas if hasattr(betas, '__iter__') else [betas] * self.num_dists
    powers = powers if hasattr(powers, '__iter__') else [powers] * self.num_dists
    self.add_hyperparams(betas=np.array(betas), powers=np.array(powers), scale=scale)
    self.dist_is_hilbertian = dist_is_hilbertian

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return self.dist_is_hilbertian

  def evaluate_from_dists(self, list_of_dists):
    """ Evaluates the kernel from pairwise distances. """
    sum_of_dists = _compute_raised_scaled_sum(list_of_dists, self.hyperparams['betas'], \
                                                            self.hyperparams['powers'])
    return self.hyperparams['scale'] * np.exp(-sum_of_dists)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the kernel between X1 and X2. """
    list_of_dists = self.dist_computer(X1, X2)
    return self.evaluate_from_dists(list_of_dists)


class SumOfExpSumOfDistsKernel(Kernel):
  """ Given a function that returns a list of distances d1, d2, ... dk, this kernel
      takes the form alpha_1*exp(beta11*d1^p + ... + beta1k*d2^p) +
      alpha_2 * exp(beta_21 + ... + beta2k*dk^p) + ... .
  """

  def __init__(self, dist_computer, alphas, groups, betas, powers, num_dists=None,
               dist_is_hilbertian=False):
    super(SumOfExpSumOfDistsKernel, self).__init__()
    self.num_dists = num_dists if num_dists is not None else len(betas)
    self.dist_computer = dist_computer
    assert len(alphas) == len(groups)
    betas = betas if hasattr(betas, '__iter__') else [betas] * self.num_dists
    powers = powers if hasattr(powers, '__iter__') else [powers] * self.num_dists
    self.add_hyperparams(betas=np.array(betas), powers=np.array(powers), alphas=alphas,
                         groups=groups)
    self.dist_is_hilbertian = dist_is_hilbertian

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return self.dist_is_hilbertian

  def evaluate_from_dists(self, list_of_dists):
    """ Evaluates the kernel from pairwise distances. """
    individual_kernels = []
    for gi, group in enumerate(self.hyperparams['groups']):
      curr_list_of_dists = [list_of_dists[i] for i in group]
      curr_betas = [self.hyperparams['betas'][i] for i in group]
      curr_powers = [self.hyperparams['powers'][i] for i in group]
      curr_sum_of_dists = _compute_raised_scaled_sum(curr_list_of_dists,
                                                     curr_betas, curr_powers)
      individual_kernels.append(self.hyperparams['alphas'][gi] *
                                np.exp(-curr_sum_of_dists))
    return sum(individual_kernels)

  def _child_evaluate(self, X1, X2):
    list_of_dists = self.dist_computer(X1, X2)
    return self.evaluate_from_dists(list_of_dists)


# Ancillary functions for the ExpSumOfDists and SumOfExpSumOfDistsKernel classes ---------
def _compute_raised_scaled_sum(dist_arrays, betas, powers):
  """ Returns the distances raised to the powers and scaled by betas. """
  sum_of_dists = np.zeros(dist_arrays[0].shape)
  for idx, curr_dist in enumerate(dist_arrays):
    if powers[idx] == 1:
      raised_curr_dists = curr_dist
    else:
      raised_curr_dists = curr_dist ** powers[idx]
    sum_of_dists += betas[idx] * raised_curr_dists
  return sum_of_dists

