"""
  Implements some standard synthetic functions in Euclidean spaces.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import division

# pylint: disable=invalid-name

import numpy as np
# Local imports
from .general_utils import map_to_cube
from ..exd.experiment_caller import EuclideanFunctionCaller


# Hartmann Functions ---------------------------------------------------------------------
def hartmann(x, alpha, A, P, max_val=np.inf):
  """ Computes the hartmann function for any given A and P. """
  log_sum_terms = (A * (P - x)**2).sum(axis=1)
  return min(max_val, alpha.dot(np.exp(-log_sum_terms)))

def _get_hartmann_data(domain_dim):
  """ Returns A and P for the 3D hartmann function. """
  # pylint: disable=bad-whitespace
  if domain_dim == 3: # 3D hartmann function
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]], dtype=np.float64)
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [ 381, 5743, 8828]], dtype=np.float64)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    domain = [[0, 1]] * 3
    opt_pt = np.array([0.114614, 0.555649, 0.852547])
    max_val = 3.86278
  elif domain_dim == 6: # 6D hartmann function
    A = np.array([[  10,   3,   17, 3.5, 1.7,  8],
                  [0.05,  10,   17, 0.1,   8, 14],
                  [   3, 3.5,  1.7,  10,  17,  8],
                  [  17,   8, 0.05,  10, 0.1, 14]], dtype=np.float64)
    P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091,  381]], dtype=np.float64)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    domain = [[0, 1]] * 6
    opt_pt = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    max_val = 3.322368
  else:
    raise NotImplementedError('Only implemented in 3 and 6 dimensions.')
  return A, P, alpha, opt_pt, domain, max_val

def get_mf_hartmann_function_data(fidel_dim, domain_dim):
  """ Returns a function f(z, x). z refers to the fidelity and x is the point in the
      domain. """
  A, P, alpha, opt_pt, domain_bounds, max_val = _get_hartmann_data(domain_dim)
  # This is how much we will perturb the alphas
  delta = np.array([0.1] * fidel_dim + [0] * (4-fidel_dim))
  # Define a wrapper for the mf objective
  def mf_hart_obj(z, x):
    """ Wrapper for the mf hartmann objective. z is fidelity and x is domain. """
    assert len(z) == fidel_dim
    z_extended = np.append(z, [0] * (4-fidel_dim))
    alpha_z = alpha - (1 - z_extended) * delta
    return hartmann(x, alpha_z, A, P, max_val)
  # Define a wrapper for the sf objective
  def hart_obj(x):
    """ Wrapper for the hartmann objective. z is fidelity and x is domain. """
    return hartmann(x, alpha, A, P, max_val)
  # Define the optimum fidelity and the fidelity bounds
  fidel_to_opt = np.ones(fidel_dim)
  fidel_bounds = [[0, 1]] * fidel_dim
  opt_val = hart_obj(opt_pt)
  return mf_hart_obj, hart_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds


# Currin Exponential Function ------------------------------------------------------------
def currin_exp(x, alpha):
  """ Computes the currin exponential function. """
  x1 = x[0]
  x2 = x[1]
  val_1 = 1 - alpha * np.exp(-1/(2 * x2))
  val_2 = (2300*x1**3 + 1900*x1**2 + 2092*x1 + 60) / (100*x1**3 + 500*x1**2 + 4*x1 + 20)
  return val_1 * val_2

def get_mf_currin_exp_function_data():
  """ Returns the multi-fidelity currin exponential function with d=6 and p=2. """
  opt_val = 13.7986850
  # A wrapper for the MF objective
  def mf_currin_exp_obj(z, x):
    """ Wrapper for the MF currin objective. """
    alpha_z = 1 - 0.1 * z
    return min(opt_val, currin_exp(x, alpha_z))
  fidel_to_opt = np.array([1])
  opt_pt = None
  # A wrapper for the SF objective
  def sf_currin_exp_obj(x):
    """ Wrapper for the single fidelity currin objective. """
    return min(opt_val, currin_exp(x, fidel_to_opt))
  fidel_bounds = np.array([[0, 1]])
  domain_bounds = np.array([[0, 1], [0, 1]])
  return mf_currin_exp_obj, sf_currin_exp_obj, opt_pt, opt_val, fidel_to_opt, \
         fidel_bounds, domain_bounds


# Branin Function ------------------------------------------------------------------------
def branin_function(x, a, b, c, r, s, t):
  """ Computes the Branin function. """
  x1 = x[0]
  x2 = x[1]
  neg_ret = float(a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)
  return -neg_ret

def branin_function_alpha(x, alpha, a, r, s):
  """ Alternative form for the branin function. """
  return branin_function(x, a, alpha[0], alpha[1], r, s, alpha[2])

def get_mf_branin_function(fidel_dim):
  """ Returns the Branin function as a multifidelity function. """
  a0 = 1
  b0 = 5.1/(4*np.pi**2)
  c0 = 5/np.pi
  r0 = 6
  s0 = 10
  t0 = 1/(8*np.pi)
  alpha = np.array([b0, c0, t0])
  # Define delta
  delta = [0.01, 0.1, -0.005]
  delta = np.array(delta[0:fidel_dim] + [0] * (3 - fidel_dim))
  # A wrapper for the MF objective
  def mf_branin_obj(z, x):
    """ Wrapper for the MF Branin objective. """
    assert len(z) == fidel_dim
    z_extended = np.append(z, [0] * (3-fidel_dim))
    alpha_z = alpha - (1 - z_extended) * delta
    return branin_function_alpha(x, alpha_z, a0, r0, s0)
  # A wrapper for the SF objective
  def sf_branin_obj(x):
    """ Wrapper for the SF Branin objective. """
    return branin_function(x, a0, b0, c0, r0, s0, t0)
  # Other data
  fidel_to_opt = np.ones((fidel_dim))
  fidel_bounds = [[0, 1]] * fidel_dim
  opt_pt = np.array([np.pi, 2.275])
  opt_val = sf_branin_obj(opt_pt)
  domain_bounds = [[-5, 10], [0, 15]]
  return mf_branin_obj, sf_branin_obj, opt_pt, opt_val, fidel_to_opt, \
         fidel_bounds, domain_bounds

# Borehole Function ----------------------------------------------------------------------
def borehole_function(x, z, max_val):
  """ Computes the Bore Hole function. """
  # pylint: disable=bad-whitespace
  rw = x[0]
  r  = x[1]
  Tu = x[2]
  Hu = x[3]
  Tl = x[4]
  Hl = x[5]
  L  = x[6]
  Kw = x[7]
  # Compute high fidelity function
  frac2 = 2*L*Tu/(np.log(r/rw) * rw**2 * Kw)
  f2 = min(max_val, 2 * np.pi * Tu * (Hu - Hl)/(np.log(r/rw) * (1 + frac2 + Tu/Tl)))
  # Compute low fidelity function
  f1 = 5 * Tu * (Hu - Hl)/(np.log(r/rw) * (1.5 + frac2 + Tu/Tl))
  # Compute final output
  return float(f2*z + f1*(1-z))

def get_mf_borehole_function():
  """ Gets the MF BoreHole function. """
  opt_val = 309.523221
  opt_pt = None
  mf_borehole_function = lambda z, x: borehole_function(x, z, opt_val)
  domain_bounds = [[0.05, 0.15],
                   [100, 50000],
                   [63070, 115600],
                   [990, 1110],
                   [63.1, 116],
                   [700, 820],
                   [1120, 1680],
                   [9855, 12045]]
  fidel_bounds = [[0, 1]]
  fidel_to_opt = np.array([1])
  # A wrapper for the single fidelity function
  def sf_borehole_obj(x):
    """ A wrapper for the single fidelity objective. """
    return borehole_function(x, fidel_to_opt, opt_val)
  # return
  return mf_borehole_function, sf_borehole_obj, opt_pt, opt_val, fidel_to_opt, \
         fidel_bounds, domain_bounds

# Park Functions ----------------------------------------------------------------------
def park1(x, max_val):
  """ Computes the park1 function. """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  x4 = x[3]
  ret1 = (x1/2) * (np.sqrt(1 + (x2 + x3**2)*x4/(x1**2)) - 1)
  ret2 = (x1 + 3*x4) * np.exp(1 + np.sin(x3))
  return min(ret1 + ret2, max_val)

def get_mf_park1_function():
  """ Gets the MF Park1 function. """
  opt_val = 25.5872304
  opt_pt = None
  sf_park1_obj = lambda x: park1(x, opt_val)
  domain_bounds = [[0, 1]] * 4
  return None, sf_park1_obj, opt_pt, opt_val, None, None, domain_bounds

def park2(x, max_val):
  """ Comutes the park2 function """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  x4 = x[3]
  ret = (2.0/3.0) * np.exp(x1 + x2) - x4*np.sin(x3) + x3
  return min(ret, max_val)

def get_mf_park2_function():
  """ Gets the MF Park2 function. """
  opt_val = 5.925698
  opt_pt = None
  sf_park2_obj = lambda x: park2(x, opt_val)
  domain_bounds = [[0, 1]] * 4
  return None, sf_park2_obj, opt_pt, opt_val, None, None, domain_bounds

# A cost function for MF evaluations -------------------------------------------
def get_mf_cost_function(fidel_bounds):
  """ Returns the cost function. fidel_bounds are the bounds for the fidelity space. """
  fidel_dim = len(fidel_bounds)
  if fidel_dim == 1:
    fidel_powers = [2]
  elif fidel_dim == 2:
    fidel_powers = [3, 2]
  elif fidel_dim == 3:
    fidel_powers = [3, 2, 1.5]
  else:
    fidel_powers = [3] + list(np.linspace(2, 1.2, fidel_dim-1))
  # Define the normalised
  def _unnorm_cost_function(norm_z):
    """ The cost function with normalised coordinates. """
    return np.power(norm_z, fidel_powers).sum()
  max_unnorm_cost = _unnorm_cost_function(np.ones(fidel_dim))
  fidel_bounds = np.array(fidel_bounds)
  def _norm_cost_function(z):
    """ Normalised cost function. """
    return 0.1 + 0.9 * (_unnorm_cost_function(map_to_cube(np.array(z), fidel_bounds)) /
                        max_unnorm_cost)
  return _norm_cost_function

def get_high_dim_function(domain_dim, group_dim, mf_obj, sf_obj):
  """ Constructs a higher dimensional functions. """
  num_groups = int(domain_dim/group_dim)
  def mf_obj_high_dim(z, x):
    """ Evaluates the higher dimensional function. """
    ret = mf_obj(z, x[0:group_dim])
    for j in range(1, num_groups):
      ret += sf_obj(x[j*group_dim: (j+1)*group_dim])
    return ret
  def sf_obj_high_dim(x):
    """ Evaluates the higher dimensional function. """
    ret = 0
    for j in range(num_groups):
      ret += sf_obj(x[j*group_dim: (j+1)*group_dim])
    return ret
  return mf_obj_high_dim, sf_obj_high_dim, num_groups

def get_high_dim_function_data(func_name, fidel_dim=None):
  """ Gets a high dimensional function data from the description. """
  fidel_dim_to_pass = 1 if fidel_dim is None else fidel_dim
  segments = func_name.split('-')
  domain_dim = int(segments[1])
  mf_obj, sf_obj, _, _, fidel_to_opt, fidel_bounds, domain_bounds = \
                               get_function_data(segments[0], fidel_dim=fidel_dim_to_pass)
  group_dim = len(domain_bounds)
  mf_obj_high_dim, sf_obj_high_dim, num_groups = get_high_dim_function(domain_dim, \
                                                               group_dim, mf_obj, sf_obj)
  high_d_domain_bounds = np.tile(np.array(domain_bounds).T, num_groups+1).T[0:domain_dim]

  return mf_obj_high_dim, sf_obj_high_dim, None, None, fidel_to_opt, \
         fidel_bounds, high_d_domain_bounds

def get_function_data(func_name, domain_dim=None, fidel_dim=None, noise_type='no_noise',
                      noise_scale=None):
  """ Gets a function data from the description. """
  # pylint: disable=too-many-branches
  fidel_dim_to_pass = 1 if fidel_dim is None else fidel_dim
  if func_name == 'hartmann':
    if domain_dim is None:
      domain_dim = 6
    if fidel_dim is not None and fidel_dim > 4:
      raise ValueError(('For the hartmann functions, fidel_dim has to be 4 or less. ' +
                        'Given: %s.')%(fidel_dim))
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_mf_hartmann_function_data(fidel_dim_to_pass, domain_dim)
  elif func_name == 'hartmann3':
    return get_function_data('hartmann', 3, fidel_dim, noise_type, noise_scale)
  elif func_name == 'hartmann6':
    return get_function_data('hartmann', 6, fidel_dim, noise_type, noise_scale)
  elif func_name == 'branin':
    if domain_dim != 2 and domain_dim is not None:
      raise ValueError(('For the branin function, domain_dim has to be 2 or None. ' +
                        'Given: %s.')%(domain_dim))
    if fidel_dim is not None and fidel_dim > 3:
      raise ValueError(('For the branin function, fidel_dim has to be 3 or less. ' +
                        'Given: %s.')%(fidel_dim))
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_mf_branin_function(fidel_dim_to_pass)
  elif func_name == 'borehole':
    if domain_dim != 8 and domain_dim is not None:
      raise ValueError(('For the borehole function, domain_dim has to be 8 or None. ' +
                        'Given: %s.')%(domain_dim))
    if fidel_dim != 1 and fidel_dim is not None:
      raise ValueError(('For the borehole function, fidel_dim has to be 1 or None. ' +
                        'Given: %s.')%(fidel_dim))
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_mf_borehole_function()
  elif func_name == 'park1':
    if domain_dim != 4 and domain_dim is not None:
      raise ValueError(('For the park1 function, domain_dim has to be 4 or None. ' +
                        'Given: %s.')%(domain_dim))
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_mf_park1_function()
  elif func_name == 'park2':
    if domain_dim != 4 and domain_dim is not None:
      raise ValueError(('For the park2 function, domain_dim has to be 4 or None. ' +
                        'Given: %s.')%(domain_dim))
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_mf_park2_function()
  else:
    raise ValueError('Unknwon func_name: %s.'%(func_name))

  return mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds

# An API that returns a function caller from the description --------------------------
def get_syn_func_caller(func_name, domain_dim=None, fidel_dim=None,
                        noise_type='no_noise', noise_scale=None,
                        to_normalise_domain=True):
  """ Returns a FunctionCaller object from the function name. """
  func_name = func_name.lower()
  funcs = ['hartmann', 'hartmann6', 'hartmann3', 'branin', 'borehole', 'park1', 'park2']
  if func_name in funcs:
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_function_data(func_name, domain_dim, fidel_dim, noise_type, noise_scale)
  else:
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_high_dim_function_data(func_name, fidel_dim)

  # Now return
  if fidel_dim is None:
    return EuclideanFunctionCaller(sf_obj, domain_bounds, descr=func_name, \
                          vectorised=False, to_normalise_domain=to_normalise_domain, \
                          raw_argmax=opt_pt, maxval=opt_val, noise_type=noise_type, \
                          noise_scale=noise_scale)
  else:
    fidel_cost_func = get_mf_cost_function(fidel_bounds)
    return EuclideanFunctionCaller(mf_obj, raw_domain=domain_bounds, descr=func_name, \
                          vectorised=False, to_normalise_domain=to_normalise_domain, \
                          raw_argmax=opt_pt, maxval=opt_val, noise_type=noise_type, \
                          noise_scale=noise_scale, fidel_cost_func=fidel_cost_func, \
                          raw_fidel_space=fidel_bounds, raw_fidel_to_opt=fidel_to_opt)

# An API that returns function data from the description --------------------------
def get_syn_function(func_name, noise_type='no_noise', noise_scale=None):
  """ Returns a synthetic function from the function name along with its
      optimum point, optimum value and the domain_bounds. sf_obj is the
      concerned synthetic function.
  """
  func_name = func_name.lower()
  funcs = ['hartmann', 'hartmann6', 'hartmann3', 'branin', 'borehole', 'park1', 'park2']
  if func_name in funcs:
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_function_data(func_name, noise_type=noise_type, noise_scale=noise_scale)
  else:
    mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds = \
      get_high_dim_function_data(func_name)

  return mf_obj, sf_obj, opt_pt, opt_val, fidel_to_opt, fidel_bounds, domain_bounds
