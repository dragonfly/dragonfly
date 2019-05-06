"""
  Implements GPs for Euclidean spaces.
  -- kandasamy@cs.cmu.edu
  -- kvysyara@andrew.cmu.edu
"""

from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-member

from argparse import Namespace
import numpy as np
# Local imports
from . import gp_core, mf_gp
from . import kernel as gp_kernel
from ..utils.ancillary_utils import get_list_of_floats_as_str
from ..utils.general_utils import get_sublist_from_indices, map_to_bounds
from ..utils.option_handler import get_option_specs, load_options
from ..utils.oper_utils import random_sample_from_discrete_domain
from ..utils.reporters import get_reporter

_DFLT_KERNEL_TYPE = 'matern'

# Some basic parameters for Euclidean GPs.
basic_euc_gp_args = [ \
  get_option_specs('kernel_type', False, 'default',
                   'Specify type of kernel. This depends on the application.'),
  get_option_specs('use_same_bandwidth', False, False,
                   ('If true, will use same bandwidth on all dimensions. Applies only '
                    'when kernel_type is se or matern. Default=False.')), \
  ]
# Parameters for the SE kernel.
se_gp_args = [ \
  ]
# Parameters for the matern kernel
matern_gp_args = [ \
  get_option_specs('matern_nu', False, 2.5, \
    ('Specify the nu value for the matern kernel. If negative, will fit.')),
                 ]
# Parameters for the Polynomial kernel.
poly_gp_args = [ \
  get_option_specs('use_same_scalings', False, False,
                   'If true uses same scalings on all dimensions. Default is False.'),
  get_option_specs('poly_order', False, 1,
                   'Order of the polynomial to be used. Default is 1 (linear kernel).')
               ]
# Parameters for an additive kernel
add_gp_args = [ \
  get_option_specs('use_additive_gp', False, False,
                   'Whether or not to use an additive GP. '),
  get_option_specs('add_max_group_size', False, 6,
                   'The maximum number of groups in the additive grouping. '),
  get_option_specs('add_grouping_criterion', False, 'randomised_ml',
                   'Specify the grouping algorithm, should be one of {randomised_ml}'),
  get_option_specs('num_groups_per_group_size', False, -1,
                   'The number of groups to try per group size.'),
  get_option_specs('add_group_size_criterion', False, 'sampled',
                   'Specify how to pick the group size, should be one of {max, sampled}.')
              ]
# Parameters for an esp kernel
esp_gp_args = [ \
  get_option_specs('esp_order', False, -1,
                   'Order of the esp kernel. '),
  get_option_specs('esp_kernel_type', False, 'se',
                   'Specify type of kernel. This depends on the application.'),
  get_option_specs('esp_matern_nu', False, -1.0, \
                   ('Specify the nu value for matern kernel. If negative, will fit.')),
              ]
# All parameters for a basic GP
euclidean_gp_args = gp_core.mandatory_gp_args + basic_euc_gp_args + se_gp_args + \
                    matern_gp_args + poly_gp_args + add_gp_args + esp_gp_args

# Hyper-parameters for Euclidean Multi-fidelity GPs
basic_mf_euc_gp_args = [ \
  # Fidelity kernel ------------------------------------------------------------
    get_option_specs('fidel_kernel_type', False, 'se', \
      'Type of kernel for the fidelity space. Should be se, matern, poly or expdecay'),
    # Secondary parameters for the fidelity kernel
    get_option_specs('fidel_matern_nu', False, 2.5, \
      ('Specify the nu value for the matern kernel. If negative, will fit.')),
    get_option_specs('fidel_use_same_bandwidth', False, False, \
      ('If true, will use same bandwidth on all fidelity dimensions. Applies only when ' \
      'fidel_kernel_type is se or matern. Default=False.')),
    get_option_specs('fidel_use_same_scalings', False, False, \
      ('If true, will use same scaling on all fidelity dimensions. Applies only when ' \
      'fidel_kernel_type is poly. Default=False.')),
    get_option_specs('fidel_poly_order', False, -1, \
      ('Order of the polynomial for fidelity kernel. Default = -1 (means will tune)')),
    # Domain kernel --------------------------------------------------------------
    get_option_specs('domain_kernel_type', False, 'se',
                     'Type of kernel for the domain space. Should be se, matern or poly'),
    # Secondary parameters for the domain kernel
    get_option_specs('domain_matern_nu', False, 2.5, \
      ('Specify the nu value for the matern kernel. If negative, will fit.')),
    get_option_specs('domain_use_same_bandwidth', False, False, \
      ('If true, will use same bandwidth on all domain dimensions. Applies only when ' \
      'domain_kernel_type is se or matern. Default=False.')),
    get_option_specs('domain_use_same_scalings', False, False, \
      ('If true, will use same scaling on all domainity dimensions. Applies only when ' \
      'domain_kernel_type is poly. Default=False.')),
    get_option_specs('domain_poly_order', False, -1, \
      ('Order of the polynomial for domainity kernel. Default = -1 (means will fit)')),
    # Additive models for the domain kernel
    get_option_specs('domain_use_additive_gp', False, False,
                     'Whether or not to use an additive GP. '),
    get_option_specs('domain_add_max_group_size', False, 6,
                     'The maximum number of groups in the additive grouping. '),
    get_option_specs('domain_add_grouping_criterion', False, 'randomised_ml',
                     'Specify the grouping algorithm, should be one of {randomised_ml}'),
    get_option_specs('domain_num_groups_per_group_size', False, -1,
                     'The number of groups to try per group size.'),
    get_option_specs('domain_add_group_size_criterion', False, 'sampled', \
      'Specify how to pick the group size, should be one of {max, sampled}.'), \
    get_option_specs('domain_esp_order', False, -1,
                     'Order of the esp kernel. '),
    get_option_specs('domain_esp_kernel_type', False, 'se',
                     'Specify type of kernel. This depends on the application.'),
    get_option_specs('domain_esp_matern_nu', False, -1.0, \
                     ('Specify the nu value for matern kernel. If negative, will fit.')),
    get_option_specs('fidel_esp_order', False, -1,
                     'Order of the esp kernel. '),
    get_option_specs('fidel_esp_kernel_type', False, 'se',
                     'Specify type of kernel. This depends on the application.'),
    get_option_specs('fidel_esp_matern_nu', False, -1.0, \
                     ('Specify the nu value for matern kernel. If negative, will fit.')),\
  ]
# Define this which includes mandatory_gp_args and basic_gp_args
euclidean_mf_gp_args = gp_core.mandatory_gp_args + basic_mf_euc_gp_args

# Part I: EuclideanGP and EuclideanGPFitter
# ======================================================================================
class EuclideanGP(gp_core.GP):
  """ euclidean GP factory """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, X, Y, kernel, mean_func, noise_var,
               kernel_hyperparams=None, build_posterior=True, reporter=None):
    """
    X, Y: data
    kern: could be an object or one of the following strings, 'se', 'poly', 'matern'
    kernel_hyperparams: dictionary specifying the hyper-parameters for the kernel.
          'se' : 'dim', 'scale' (optional), 'dim_bandwidths' (optional)
          'poly' : 'dim', 'order', 'scale', 'dim_scalings' (optional)
          'matern' : 'dim', 'nu' (optional), 'scale' (optional),
                     'dim_bandwidths' (optional)
    """
    if isinstance(kernel, str):
      kernel = self._get_kernel_from_type(kernel, kernel_hyperparams)
    super(EuclideanGP, self).__init__(X, Y, kernel, mean_func, noise_var,
                                      build_posterior, reporter)

  @classmethod
  def _get_kernel_from_type(cls, kernel_type, kernel_hyperparams):
    """ Get different euclidean kernels based on kernel_type"""
    if kernel_type in ['se']:
      return gp_kernel.SEKernel(kernel_hyperparams['dim'], kernel_hyperparams['scale'],
                                kernel_hyperparams['dim_bandwidths'])
    elif kernel_type in ['poly']:
      return gp_kernel.PolyKernel(kernel_hyperparams['dim'], kernel_hyperparams['order'],
                                  kernel_hyperparams['scale'],
                                  kernel_hyperparams['dim_scalings'])
    elif kernel_type in ['matern']:
      return gp_kernel.MaternKernel(kernel_hyperparams['dim'],
                                    kernel_hyperparams['nu'], kernel_hyperparams['scale'],
                                    kernel_hyperparams['dim_bandwidths'])
    elif kernel_type in ['esp']:
      return gp_kernel.ESPKernelSE(kernel_hyperparams['dim'], kernel_hyperparams['scale'],
                                   kernel_hyperparams['order'],
                                   kernel_hyperparams['dim_bandwidths'])
    else:
      raise ValueError('Cannot construct kernel from kernel_type %s.' % (kernel_type))

  def _child_str(self):
    """ String representation for child GP. """
    ke_str = self._get_kernel_str(self.kernel)
    dim = 0 if len(self.X) == 0 else len(self.X[0])
    mean_str = 'mu(0)=%0.3f'%(self.mean_func([np.zeros(dim,)])[0])
    ret = 'scale: %0.3f, %s, %s' % (self.kernel.hyperparams['scale'], ke_str, mean_str)
    return ret

  @classmethod
  def _get_kernel_str(cls, kern):
    """ Gets a string format of the kernel depending on whether it is SE/Poly."""
    if isinstance(kern, gp_kernel.AdditiveKernel):
      return str(kern)
    if isinstance(kern, gp_kernel.SEKernel) or isinstance(kern, gp_kernel.MaternKernel):
      hp_name = 'dim_bandwidths'
      kern_name = 'se' if isinstance(kern, gp_kernel.SEKernel) else \
        'matern(%0.1f)' % (kern.hyperparams['nu'])
    elif isinstance(kern, gp_kernel.PolyKernel):
      hp_name = 'dim_scalings'
      kern_name = 'poly'
    else:  # Return an empty string.
      return ''
    if kern.dim > 6:
      ret = '%0.4f(avg)' % (kern.hyperparams[hp_name].mean())
    else:
      ret = get_list_of_floats_as_str(kern.hyperparams[hp_name])
    ret = kern_name + '-' + ret
    return ret


class EuclideanGPFitter(gp_core.GPFitter):
  """ Fits a GP by tuning the kernel hyper-params. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, X, Y, options=None, reporter=None):
    """ Constructor. """
    self.dim = len(X[0])
    reporter = get_reporter(reporter)
    options = load_options(euclidean_gp_args, partial_options=options)
    super(EuclideanGPFitter, self).__init__(X, Y, options, reporter)

  def _child_set_up(self):
    """ Sets parameters for GPFitter. """
    # IMPORTANT: Keep this order when tuning for the hyper-parameters.
    # Continuous: Mean value, GP noise, kernel scale, then the remaining Kernel params.
    # Discrete: add_group_size, then the remaining kernel params.
    # Check args - so that we don't have to keep doing this all the time
    if self.options.kernel_type not in ['se', 'matern', 'poly', 'esp', 'default']:
      raise ValueError('Unknown kernel_type. Should be either se, matern or poly.')
    if self.options.noise_var_type not in ['tune', 'label', 'value']:
      raise ValueError('Unknown noise_var_type. Should be either tune, label or value.')
    if self.options.mean_func_type not in ['mean', 'median', 'const', 'zero', 'tune']:
      raise ValueError('Unknown mean_func_type. Should be mean/median/const/zero/tune.')
    # Set kernel type
    if self.options.kernel_type == 'default':
      self.kernel_type = _DFLT_KERNEL_TYPE
    else:
      self.kernel_type = self.options.kernel_type
    # 1 & 2: mean value and noise variance - done in parent class.
    # 3. Kernel parameters
    if self.kernel_type == 'se':
      self._se_kernel_set_up()
    elif self.kernel_type == 'matern':
      self._matern_kernel_set_up()
    elif self.kernel_type in ['poly']:
      self._poly_kernel_set_up()
    elif self.kernel_type == 'esp':
      self._esp_kernel_set_up()
    # 4. Additive grouping
    if self.options.use_additive_gp and self.kernel_type != 'esp':
      self.add_group_size_idx_in_dscr_hp_vals = len(self.dscr_hp_vals)
      self.add_max_group_size = min(self.options.add_max_group_size, self.dim)
      self.dscr_hp_vals.append([x+1 for x in range(self.add_max_group_size)])
      self.param_order.append(["additive_grp", "dscr"])
    elif self.kernel_type == 'esp' and self.options.esp_order == -1:
      self.dscr_hp_vals.append(list(range(1, max(self.dim, self.options.esp_order) + 1)))
      self.param_order.append(["esp_order", "dscr"])

  def _se_kernel_set_up(self):
    """ Set up for the SE kernel. """
    # Scale
    self.scale_log_bounds = [np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)]
    self.param_order.append(["scale", "cts"])
    # Bandwidths
    X_std_norm = np.linalg.norm(self.X, 'fro') + 1e-4
    single_bandwidth_log_bounds = [np.log(0.01 * X_std_norm), np.log(10 * X_std_norm)]
    if self.options.use_same_bandwidth:
      self.bandwidth_log_bounds = [single_bandwidth_log_bounds]
      self.param_order.append(["same_dim_bandwidths", "cts"])
    else:
      self.bandwidth_log_bounds = [single_bandwidth_log_bounds] * self.dim
      for _ in range(self.dim):
        self.param_order.append(["dim_bandwidths", "cts"])
    self.cts_hp_bounds += [self.scale_log_bounds] + self.bandwidth_log_bounds

  def _matern_kernel_set_up(self):
    """ Set up for the Matern kernel. """
    # Set up scale and bandwidth - which is identical to the SE kernel.
    self._se_kernel_set_up()
    # Set up optimisation values for the nu parameter.
    if self.options.matern_nu < 0:
      self.dscr_hp_vals.append([0.5, 1.5, 2.5])
      self.param_order.append(["nu", "dscr"])

  def _poly_kernel_set_up(self):
    """ Set up for the Poly kernel. """
    raise NotImplementedError('Not implemented Poly kernel yet.')

  def _esp_kernel_set_up(self):
    """ Set up for the ESP kernel. """
    if self.options.esp_kernel_type not in ['se', 'matern']:
      raise NotImplementedError('Not implemented yet.')
    # Scale
    self.scale_log_bounds = [np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)]
    self.param_order.append(["scale", "cts"])
    # Bandwidths
    X_std_norm = np.linalg.norm(self.X, 'fro') + 1e-4
    single_bandwidth_log_bounds = [np.log(0.01 * X_std_norm), np.log(10 * X_std_norm)]
    self.bandwidth_log_bounds = [single_bandwidth_log_bounds] * self.dim
    for _ in range(self.dim):
      self.param_order.append(["dim_bandwidths", "cts"])
    self.cts_hp_bounds += [self.scale_log_bounds] + self.bandwidth_log_bounds
    if self.options.esp_kernel_type == 'matern' and self.options.esp_matern_nu < 0:
      self.dscr_hp_vals.append([0.5, 1.5, 2.5])
      self.param_order.append(["nu", "dscr"])

  def _prep_init_kernel_hyperparams(self, kernel_type):
    """ Wrapper to pack the kernel hyper-parameters into a dictionary. """
    return prep_euclidean_integral_kernel_hyperparams(kernel_type, self.options, self.dim)

  def _optimise_cts_hps_for_given_dscr_hps(self, given_dscr_hps):
    """ Optimises the continuous hyper-parameters for the given discrete hyper-params.
    """
    if not self.options.use_additive_gp:
      return super(EuclideanGPFitter, self)._optimise_cts_hps_for_given_dscr_hps( \
                                                                           given_dscr_hps)
    else:
      return optimise_cts_hps_for_given_dscr_hps_in_add_model(given_dscr_hps, \
        self.options.num_groups_per_group_size, self.dim, self.hp_tune_max_evals, \
        self.cts_hp_optimise, self._tuning_objective)

  def _sample_cts_dscr_hps_for_rand_exp_sampling(self):
    """ Samples continous and discrete hyper-parameters for rand_exp_sampling. """
    if not self.options.use_additive_gp:
      return super(EuclideanGPFitter, self)._sample_cts_dscr_hps_for_rand_exp_sampling()
    else:
      return sample_cts_dscr_hps_for_rand_exp_sampling_in_add_model( \
        self.hp_tune_max_evals, self.cts_hp_bounds, self.dim, self.dscr_hp_vals, \
        self.add_group_size_idx_in_dscr_hp_vals, self._tuning_objective)

  def _child_build_gp(self, mean_func, noise_var, gp_cts_hps, gp_dscr_hps,
                      other_gp_params=None, *args, **kwargs):
    """ Builds the GP. """
    # Domain kernel --------------------------------------
    kernel_hyperparams = self._prep_init_kernel_hyperparams(self.kernel_type)
    add_gp_groupings = None
    if self.options.use_additive_gp:
      gp_dscr_hps = gp_dscr_hps[:-1] # The first element is the group size
      add_gp_groupings = other_gp_params.add_gp_groupings
    kernel, gp_cts_hps, gp_dscr_hps = \
      get_euclidean_integral_gp_kernel(self.kernel_type, kernel_hyperparams, gp_cts_hps,
                                       gp_dscr_hps, self.options.use_same_bandwidth,
                                       add_gp_groupings, self.options.esp_kernel_type)
    ret_gp = EuclideanGP(self.X, self.Y, kernel, mean_func, noise_var, *args, **kwargs)
    return ret_gp, gp_cts_hps, gp_dscr_hps
  # EuclideanGPFitter ends here -------------------------------------------------------


# Part II: EuclideanMFGP and EuclideanMFGPFitter
# ======================================================================================
# MFGP and Fitter in Euclidean spaces: An instantiation of MFGP when both fidel_space and
# domain are euclidean.
class EuclideanMFGP(mf_gp.MFGP):
  """ An MFGP for Euclidean spaces. """

  def __init__(self, ZZ, XX, YY, mf_kernel,
               kernel_scale, fidel_kernel, domain_kernel,
               mean_func, noise_var, *args, **kwargs):
    """ Constructor. ZZ, XX, YY are the fidelity points, domain points and labels
        respectively.
    """
    if len(ZZ) != 0:
      self.fidel_dim = len(ZZ[0])
      self.domain_dim = len(XX[0])
    if fidel_kernel is not None and domain_kernel is not None:
      self.fidel_kernel = fidel_kernel
      self.domain_kernel = domain_kernel
      self.fidel_dim = fidel_kernel.dim
      self.domain_dim = domain_kernel.dim
    elif 'fidel_dim' in kwargs and 'domain_dim' in kwargs:
      self.fidel_dim = kwargs['fidel_dim']
      self.domain_dim = kwargs['domain_dim']
    else:
      raise Exception('Specify fidel_dim and domain_dim.')
    self.fidel_coords = list(range(self.fidel_dim))
    self.domain_coords = list(range(self.fidel_dim, self.fidel_dim + self.domain_dim))
    if mf_kernel is None:
      mf_kernel = gp_kernel.CoordinateProductKernel(self.fidel_dim + self.domain_dim, \
                                         kernel_scale, [fidel_kernel, domain_kernel], \
                                         [self.fidel_coords, self.domain_coords],)
      # Otherwise, we assume mf_kernel is already an appropriate kernel
    super(EuclideanMFGP, self).__init__(ZZ, XX, YY, mf_kernel, mean_func, noise_var,
                                        *args, **kwargs)

  def _test_fidel_domain_dims(self, test_fidel_dim, test_domain_dim):
    """ Tests if test_fidel_dim and test_domain_dim are equal to self.fidel_dim and
        self.domain_dim respectively and if not raises an error.
        Mostly for internal use. """
    if test_fidel_dim != self.fidel_dim or test_domain_dim != self.domain_dim:
      raise ValueError('ZZ, XX dimensions should be (%d, %d). Given (%d, %d)'%( \
                       self.fidel_dim, self.domain_dim, test_fidel_dim, test_domain_dim))

  def get_ZX_from_ZZ_XX(self, ZZ, XX):
    """ Gets the coordinates in the joint space from the individual fidelity and
        domain spaces. """
    ordering = np.argsort(self.fidel_coords + self.domain_coords)
    if hasattr(ZZ, '__iter__') and len(ZZ) == 0:
      return []
    elif hasattr(ZZ[0], '__iter__'):
      # A list of points
      self._test_fidel_domain_dims(len(ZZ[0]), len(XX[0]))
      ZX_unordered = np.concatenate((np.array(ZZ), np.array(XX)), axis=1)
      ZX = ZX_unordered[:, ordering]
      return list(ZX)
    else:
      # A single new point
      self._test_fidel_domain_dims(len(ZZ), len(XX))
      zx_unordered = np.concatenate((ZZ, XX))
      return zx_unordered[ordering]

  def get_domain_pts(self, data_idxs=None):
    """ Returns only the domain points. """
    data_idxs = data_idxs if data_idxs is not None else range(self.num_tr_data)
    return [self.XX[i] for i in data_idxs]

  def get_fidel_pts(self, data_idxs=None):
    """ Returns only the fidelity points. """
    data_idxs = data_idxs if data_idxs is not None else range(self.num_tr_data)
    return [self.ZZ[i] for i in data_idxs]


# A GPFitter for EuclideanMFGP objects. For now, this only considers product kernels
# between the fidel_space and domain.
class EuclideanMFGPFitter(mf_gp.MFGPFitter):
  """ A fitter for GPs in multi-fidelity optimisation. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, ZZ, XX, YY, options=None, reporter=None):
    """ Constructor. options should either be a Namespace, a list or None. """
    reporter = get_reporter(reporter)
    options = load_options(euclidean_mf_gp_args, partial_options=options)
    self.fidel_dim = len(ZZ[0])
    self.domain_dim = len(XX[0])
    self.input_dim = self.fidel_dim + self.domain_dim
    super(EuclideanMFGPFitter, self).__init__(ZZ, XX, YY, options, reporter)

  # Child set up Methods
  # ===================================================================================
  def _child_set_up(self):
    """ Sets parameters fro GPFitter. """
    # pylint: disable=too-many-branches
    # Check args - so that we don't have to keep doing this all the time
    if self.options.fidel_kernel_type not in ['se', 'matern', 'poly', 'expdecay']:
      raise ValueError('Unknown fidel_kernel_type. Should be in {se, matern, poly, ' +
                       'expdecay.')
    if self.options.domain_kernel_type not in ['se', 'matern', 'poly']:
      raise ValueError('Unknown domain_kernel_type. Should be either se or poly.')
    if self.options.noise_var_type not in ['tune', 'label', 'value']:
      raise ValueError('Unknown noise_var_type. Should be either tune, label or value.')
    if self.options.mean_func_type not in ['mean', 'median', 'const', 'zero',
                                           'upper_bound', 'tune']:
      raise ValueError(('Unknown mean_func_type. Should be one of ',
                        'mean/median/const/zero.'))
    # Set some parameters we will be using often.
    self.ZZ_std_norm = np.linalg.norm(self.ZZ, 'fro') + 5e-5
    self.XX_std_norm = np.linalg.norm(self.XX, 'fro') + 5e-5
    self.ZX_std_norm = np.sqrt(self.ZZ_std_norm**2 + self.XX_std_norm**2)

    # Bounds for the hyper parameters
    # -------------------------------
    # Kernel scale
    self.scale_log_bounds = [np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)]
    self.cts_hp_bounds.append(self.scale_log_bounds)
    self.param_order.append(["scale", "cts"])
    # Fidelity kernel
    if self.options.fidel_kernel_type == 'se':
      self._fidel_se_kernel_setup()
    elif self.options.fidel_kernel_type == 'matern':
      self._fidel_matern_kernel_setup()
    elif self.options.fidel_kernel_type == 'poly':
      self._fidel_poly_kernel_setup()
    elif self.options.fidel_kernel_type == 'expdecay':
      self._fidel_expdecay_kernel_setup()
    elif self.options.fidel_kernel_type == 'esp':
      self._fidel_esp_kernel_setup()
    # Domain kernel
    if self.options.domain_kernel_type == 'se':
      self._domain_se_kernel_setup()
    elif self.options.domain_kernel_type == 'matern':
      self._domain_matern_kernel_setup()
    elif self.options.domain_kernel_type == 'poly':
      self._domain_poly_kernel_setup()
    elif self.options.domain_kernel_type == 'esp':
      self._domain_esp_kernel_setup()
    # Additive grouping for domain kernel (this has to come after fidelity kernel set up).
    if self.options.domain_use_additive_gp:
      self.domain_add_group_size_idx_in_dscr_hp_vals = len(self.dscr_hp_vals)
      self.domain_add_max_group_size = min(self.options.domain_add_max_group_size,
                                           self.domain_dim)
      self.dscr_hp_vals.append([x+1 for x in range(self.domain_add_max_group_size)])
      self.param_order.append(["additive_grp", "dscr"])

  # Functions to set up each fidelity kernel -------------------------------------------
  def _fidel_se_kernel_setup(self):
    """ Sets up the fidelity kernel as an SE kernel. """
    self._fidel_se_matern_kernel_setup_common()

  def _fidel_matern_kernel_setup(self):
    """ Sets up the fidelity kernel as a Matern kernel. """
    self._fidel_se_matern_kernel_setup_common()
    # Set optimisation values for the nu parameter
    if self.options.fidel_matern_nu < 0:
      self.dscr_hp_vals.append([0.5, 1.5, 2.5])
      self.param_order.append(["nu", "dscr"])

  def _fidel_se_matern_kernel_setup_common(self):
    """ Common operators for setting up as a SE or Matern kernel. """
    if (hasattr(self.options, 'fidel_bandwidth_log_bounds') and
        self.options.fidel_bandwidth_log_bounds is not None):
      self.fidel_bandwidth_log_bounds = self.options.fidel_bandwidth_log_bounds
    else:
      self.fidel_bandwidth_log_bounds = self._get_bandwidth_log_bounds( \
        self.fidel_dim, self.ZX_std_norm, self.options.fidel_use_same_bandwidth)
    self.cts_hp_bounds.extend(self.fidel_bandwidth_log_bounds)
    if self.options.fidel_use_same_bandwidth:
      self.param_order.append(["same_dim_bandwidths", "cts"])
    else:
      for _ in range(self.fidel_dim):
        self.param_order.append(["dim_bandwidths", "cts"])

  def _fidel_poly_kernel_setup(self):
    """ Sets up the fidelity kernel as a Poly kernel. """
    self.fidel_scaling_log_bounds = self._get_poly_kernel_bounds(self.ZZ, self.XX, \
                                        self.options.fidel_use_same_scalings)
    self.cts_hp_bounds.extend(self.fidel_scaling_log_bounds)

  def _fidel_expdecay_kernel_setup(self):
    """ Sets up the fidelity kernel as an exponential decay kernel. """
    # offset
    if (hasattr(self.options, 'fidel_expdecay_offset_log_bounds') and
        self.options.fidel_expdecay_offset_log_bounds is not None):
      self.fidel_expdecay_offset_log_bounds = \
        self.options.fidel_expdecay_offset_log_bounds
    else:
      scale_range = self.Y_var / np.sqrt(self.num_tr_data)
      self.fidel_expdecay_offset_log_bounds = \
        [np.log(0.1 * scale_range), np.log(10 * scale_range)]
    # power log bounds
    if (hasattr(self.options, 'fidel_expdecay_power_log_bounds') and
        self.options.fidel_expdecay_power_log_bounds is not None):
      self.fidel_expdecay_power_log_bounds = \
        self.options.fidel_expdecay_power_log_bounds
    else:
      self.fidel_expdecay_power_log_bounds = \
        [[np.log(1e-1), np.log(50)]] * self.fidel_dim
    self.cts_hp_bounds.append(self.fidel_expdecay_offset_log_bounds)
    self.cts_hp_bounds.extend(self.fidel_expdecay_power_log_bounds)

  def _fidel_esp_kernel_setup(self):
    """ Sets up the fidelity kernel as ESP kernel. """
    if (hasattr(self.options, 'fidel_bandwidth_log_bounds') and
        self.options.fidel_bandwidth_log_bounds is not None):
      self.fidel_bandwidth_log_bounds = self.options.fidel_bandwidth_log_bounds
    else:
      self.fidel_bandwidth_log_bounds = self._get_bandwidth_log_bounds( \
                                        self.fidel_dim, self.ZX_std_norm, False)
    self.cts_hp_bounds.extend(self.fidel_bandwidth_log_bounds)
    for _ in range(self.fidel_dim):
      self.param_order.append(["dim_bandwidths", "cts"])
    if self.options.fidel_esp_kernel_type == 'matern' and \
       self.options.fidel_esp_matern_nu < 0:
      self.dscr_hp_vals.append([0.5, 1.5, 2.5])
      self.param_order.append(["nu", "dscr"])

  # Functions to set up each domain kernel -------------------------------------------
  def _domain_se_kernel_setup(self):
    """ Sets up the domainity kernel as an SE kernel. """
    self._domain_se_matern_kernel_setup_common()

  def _domain_matern_kernel_setup(self):
    """ Sets up the domainity kernel as a Matern kernel. """
    self._domain_se_matern_kernel_setup_common()
    # Set optimisation values for the nu parameter
    if self.options.domain_matern_nu < 0:
      self.dscr_hp_vals.append([0.5, 1.5, 2.5])
      self.param_order.append(["nu", "dscr"])

  def _domain_se_matern_kernel_setup_common(self):
    """ Sets up the domain kernel as a SE kernel. """
    if (hasattr(self.options, 'domain_bandwidth_log_bounds') and
        self.options.domain_bandwidth_log_bounds is not None):
      self.domain_bandwidth_log_bounds = self.options.domain_bandwidth_log_bounds
    else:
      self.domain_bandwidth_log_bounds = self._get_bandwidth_log_bounds( \
                                         self.domain_dim, self.ZX_std_norm, False)
    self.cts_hp_bounds.extend(self.domain_bandwidth_log_bounds)
    if self.options.domain_use_same_bandwidth:
      self.param_order.append(["same_dim_bandwidths", "cts"])
    else:
      for _ in range(self.domain_dim):
        self.param_order.append(["dim_bandwidths", "cts"])

  def _domain_poly_kernel_setup(self):
    """ Sets up the domain kernel as a Poly kernel. """
    self.domain_scaling_log_bounds = self._get_poly_kernel_bounds(self.ZZ, self.XX, \
                                        self.options.domain_use_same_scalings)
    self.cts_hp_bounds.extend(self.domain_scaling_log_bounds)

  def _domain_esp_kernel_setup(self):
    """ Sets up the domain kernel as ESP kernel. """
    if (hasattr(self.options, 'domain_bandwidth_log_bounds') and
        self.options.domain_bandwidth_log_bounds is not None):
      self.domain_bandwidth_log_bounds = self.options.domain_bandwidth_log_bounds
    else:
      self.domain_bandwidth_log_bounds = self._get_bandwidth_log_bounds( \
         self.domain_dim, self.ZX_std_norm, self.options.domain_use_same_bandwidth)
    self.cts_hp_bounds.extend(self.domain_bandwidth_log_bounds)
    for _ in range(self.domain_dim):
      self.param_order.append(["dim_bandwidths", "cts"])
    if self.options.domain_esp_kernel_type == 'matern' and \
       self.options.domain_esp_matern_nu < 0:
      self.dscr_hp_vals.append([0.5, 1.5, 2.5])
      self.param_order.append(["nu", "dscr"])

  @classmethod
  def _get_bandwidth_log_bounds(cls, dim, single_bw_bounds, use_same_bandwidth):
    """ Gets bandwidths for the SE kernel. """
    if isinstance(single_bw_bounds, float) or isinstance(single_bw_bounds, int):
      single_bw_bounds = [0.01*single_bw_bounds, 10*single_bw_bounds]
    single_bandwidth_log_bounds = [np.log(x) for x in single_bw_bounds]
    bandwidth_log_bounds = ([single_bandwidth_log_bounds] if use_same_bandwidth
                            else [single_bandwidth_log_bounds] * dim)
    return bandwidth_log_bounds

  def _get_poly_kernel_bounds(self, ZZ, XX, use_same_scalings):
    """ Gets bandwidths for the Polynomial kerne. """
    raise NotImplementedError('Yet to implement polynomial kernel.')
  # _child_set_up methods end here -----------------------------------------------------

  # fit_gp Methods
  # ====================================================================================
  def _optimise_cts_hps_for_given_dscr_hps(self, given_dscr_hps):
    """ Optimises the continuous hyper-parameters for the given discrete hyper-params.
        Overrides the methods from GPFiter
    """
    if not self.options.domain_use_additive_gp:
      return super(EuclideanMFGPFitter, self)._optimise_cts_hps_for_given_dscr_hps( \
                                                                           given_dscr_hps)
    else:
      return optimise_cts_hps_for_given_dscr_hps_in_add_model(given_dscr_hps, \
        self.options.domain_num_groups_per_group_size, self.domain_dim, \
        self.hp_tune_max_evals, self.cts_hp_optimise, self._tuning_objective)

  def _sample_cts_dscr_hps_for_rand_exp_sampling(self):
    """ Samples continous and discrete hyper-parameters for rand_exp_sampling. """
    if not self.options.domain_use_additive_gp:
      return super(EuclideanMFGPFitter, self)._sample_cts_dscr_hps_for_rand_exp_sampling()
    else:
      return sample_cts_dscr_hps_for_rand_exp_sampling_in_add_model( \
        self.hp_tune_max_evals, self.cts_hp_bounds, self.domain_dim, self.dscr_hp_vals, \
        self.domain_add_group_size_idx_in_dscr_hp_vals, self._tuning_objective)

  # build_gp Methods
  # ====================================================================================
  @classmethod
  def _prep_init_fidel_domain_kernel_hyperparams(cls, kernel_type, dim, matern_nu,
                                                 poly_order, esp_order, esp_matern_nu):
    """ Wrapper to pack the kernel hyper-parameters into a dictionary. """
    hyperparams = {}
    hyperparams['dim'] = dim
    if kernel_type == 'matern' and matern_nu > 0:
      hyperparams['nu'] = matern_nu
    elif kernel_type == 'poly':
      hyperparams['order'] = poly_order
    elif kernel_type == 'esp':
      if esp_order > 0:
        hyperparams['esp_order'] = esp_order
      if esp_matern_nu > 0:
        hyperparams['esp_matern_nu'] = esp_matern_nu
    return hyperparams

  def _prep_init_fidel_kernel_hyperparams(self):
    """ Wrapper to pack the fidelity kernel hyper-parameters into a dictionary. """
    options = self.options
    return self._prep_init_fidel_domain_kernel_hyperparams(options.fidel_kernel_type, \
                   self.fidel_dim, options.fidel_matern_nu, options.fidel_poly_order, \
                                options.fidel_esp_order, options.fidel_esp_matern_nu)

  def _prep_init_domain_kernel_hyperparams(self):
    """ Wrapper to pack the domain kernel hyper-parameters into a dictionary. """
    options = self.options
    return self._prep_init_fidel_domain_kernel_hyperparams(options.domain_kernel_type, \
                 self.domain_dim, options.domain_matern_nu, options.domain_poly_order, \
                               options.domain_esp_order, options.domain_esp_matern_nu)

  def _child_build_gp(self, mean_func, noise_var, gp_cts_hps, gp_dscr_hps,
                      other_gp_params=None, *args, **kwargs):
    """ Builds a Multi-fidelity GP from the hyper-parameters. """
    # IMPORTANT: The order of the code matters in this function. Do not change.
    # Kernel scale ---------------------------------------
    ke_scale = np.exp(gp_cts_hps[0])
    gp_cts_hps = gp_cts_hps[1:]
    # Fidelity kernel ------------------------------------
    fidel_kernel_hyperparams = self._prep_init_fidel_kernel_hyperparams()
    fidel_kernel, gp_cts_hps, gp_dscr_hps = \
      get_euclidean_integral_gp_kernel_with_scale(self.options.fidel_kernel_type, 1.0, \
        fidel_kernel_hyperparams, gp_cts_hps, gp_dscr_hps, \
        self.options.fidel_use_same_bandwidth, None, self.options.fidel_esp_kernel_type)
    # Domain kernel --------------------------------------
    # The code for the domain kernel should come after fidelity. Otherwise, its a bug.
    domain_kernel_hyperparams = self._prep_init_domain_kernel_hyperparams()
    if self.options.domain_use_additive_gp:
      gp_dscr_hps = gp_dscr_hps[:-1] # The first element is the group size
      add_gp_groupings = other_gp_params.add_gp_groupings
    else:
      add_gp_groupings = None
    domain_kernel, gp_cts_hps, gp_dscr_hps = \
      get_euclidean_integral_gp_kernel_with_scale(self.options.domain_kernel_type, 1.0, \
        domain_kernel_hyperparams, gp_cts_hps, gp_dscr_hps, \
        self.options.domain_use_same_bandwidth, add_gp_groupings, \
        self.options.domain_esp_kernel_type)
    # Construct and return MF GP
    ret_gp = EuclideanMFGP(self.ZZ, self.XX, self.YY, None, ke_scale, fidel_kernel,
                           domain_kernel, mean_func, noise_var, reporter=self.reporter)
    return ret_gp, gp_cts_hps, gp_dscr_hps
  # _child_build_gp methods end here ---------------------------------------------------


# Part III: Ancillary Functions
# ===============================================

# A function to optimise continous hyperparams in an additive model-----------------------
# Used by EuclideanGPFitter and EuclideanMFGPFitter.
def optimise_cts_hps_for_given_dscr_hps_in_add_model(given_dscr_hps, \
    num_groups_per_group_size, dim, hp_tune_max_evals, cts_hp_optimise, \
    tuning_objective):
  """ Optimises the continuous hyper-parameters for an additive model. """
  group_size = given_dscr_hps[-1] # The first is the max group size
  if num_groups_per_group_size < 0:
    if group_size == 1:
      num_groups_per_group_size = 1
    else:
      num_groups_per_group_size = max(5, min(2 * dim, 25))
  grp_best_hps = None
  grp_best_val = -np.inf
  grp_best_other_params = None
  # Now try out different groups picking a random grouping each time.
  for _ in range(num_groups_per_group_size):
    rand_perm = list(np.random.permutation(dim))
    groupings = [rand_perm[i:i+group_size]
                 for i in range(0, dim, group_size)]
    other_gp_params = Namespace(add_gp_groupings=groupings)
    # _tuning_objective is usually defined in gp_core.py
    cts_tuning_objective = lambda arg: tuning_objective(arg, given_dscr_hps[:],
                                                        other_gp_params=other_gp_params)
    max_evals = int(max(500, hp_tune_max_evals/num_groups_per_group_size))
    opt_cts_val, opt_cts_hps, _ = cts_hp_optimise(cts_tuning_objective, max_evals)
    if opt_cts_val > grp_best_val:
      grp_best_val = opt_cts_val
      grp_best_hps = opt_cts_hps
      grp_best_other_params = other_gp_params
  return grp_best_val, grp_best_hps, grp_best_other_params


def sample_cts_dscr_hps_for_rand_exp_sampling_in_add_model(num_evals, cts_hp_bounds, \
    dim, dscr_hp_vals, add_group_size_idx_in_dscr_hp_vals, tuning_objective):
  # IMPORTANT: We are assuming that the add_group_size is the first
  """ Samples the hyper-paramers for an additive model. """
  agsidhp = add_group_size_idx_in_dscr_hp_vals
  sample_cts_hps = []
  sample_dscr_hps = []
  sample_other_gp_params = []
  sample_obj_vals = []
  for _ in range(num_evals):
    group_size = np.random.choice(dscr_hp_vals[agsidhp])
    rand_perm = list(np.random.permutation(dim))
    groupings = [rand_perm[i:i+group_size] for i in range(0, dim, group_size)]
    curr_other_gp_params = Namespace(add_gp_groupings=groupings)
    curr_dscr_hps = random_sample_from_discrete_domain(dscr_hp_vals)
    curr_dscr_hps[agsidhp] = group_size
    curr_cts_hps = map_to_bounds(np.random.random((len(cts_hp_bounds),)), cts_hp_bounds)
    curr_obj_val = tuning_objective(curr_cts_hps, curr_dscr_hps, curr_other_gp_params)
    # Now add to the lists
    sample_cts_hps.append(curr_cts_hps)
    sample_dscr_hps.append(curr_dscr_hps)
    sample_other_gp_params.append(curr_other_gp_params)
    sample_obj_vals.append(curr_obj_val)
  sample_probs = np.exp(sample_obj_vals)
  sample_probs = sample_probs / sample_probs.sum()
  return sample_cts_hps, sample_dscr_hps, sample_other_gp_params, sample_probs


# Utilities for building Euclidean/Integral kernels
def prep_euclidean_integral_kernel_hyperparams(kernel_type, gp_fitter_options,
                                               domain_dim):
  """ Wrapper to pack the kernel hyper-parameters into a dictionary. """
  hyperparams = {}
  hyperparams['dim'] = domain_dim
  if kernel_type == 'matern' and gp_fitter_options.matern_nu > 0:
    hyperparams['nu'] = gp_fitter_options.matern_nu
  elif kernel_type == 'poly':
    hyperparams['order'] = gp_fitter_options.poly_order
  elif kernel_type == 'esp':
    if gp_fitter_options.esp_order > 0:
      hyperparams['esp_order'] = gp_fitter_options.esp_order
    if gp_fitter_options.esp_matern_nu > 0:
      hyperparams['esp_matern_nu'] = gp_fitter_options.esp_matern_nu
  return hyperparams


# A function to obtain a Euclidean GP kernel from arguments -----------------------
def get_euclidean_integral_gp_kernel(kernel_type, kernel_hyperparams, gp_cts_hps,
                                     gp_dscr_hps, use_same_bandwidth,
                                     add_gp_groupings=None, esp_kernel_type=None):
  """ A method to parse the GP kernel from hyper-parameters. Assumes the scale is
      part of the hyper-parameters as well. """
  scale = np.exp(gp_cts_hps[0]) # First the scale and then the rest of the hyper-params.
  gp_cts_hps = gp_cts_hps[1:]
  return get_euclidean_integral_gp_kernel_with_scale(kernel_type, scale, \
    kernel_hyperparams, gp_cts_hps, gp_dscr_hps, use_same_bandwidth, add_gp_groupings, \
    esp_kernel_type)

# The following function is used by EuclideanGPFitter and EuclideanMFGPFitter.
def get_euclidean_integral_gp_kernel_with_scale(kernel_type, scale, kernel_hyperparams,
                                                gp_cts_hps, gp_dscr_hps,
                                                use_same_bandwidth, add_gp_groupings=None,
                                                esp_kernel_type=None):
  """ A method to parse the GP kernel from hyper-parameters. Assumes the scale is
      not part of the hyper-parameters but given as an argument. """
  # pylint: disable=too-many-branches
  # pylint: disable=too-many-statements
  dim = kernel_hyperparams['dim']
  # For the additive ones ---------------------
  esp_order = None
  if kernel_type == 'esp':
    if 'esp_order' in kernel_hyperparams:
      esp_order = kernel_hyperparams['esp_order']
    else:
      esp_order = gp_dscr_hps[-1]
      gp_dscr_hps = gp_dscr_hps[:-1]
  is_additive = False
  if add_gp_groupings is None:
    add_gp_groupings = [list(range(dim))]
    grp_scale = scale
  elif esp_order is None:
    is_additive = True
    grp_scale = 1.0
  # Extract features and create a placeholder for the kernel constructors.
  # Some common functionality for the se, matern and poly kernels.
  if kernel_type in ['se', 'matern', 'poly']:
    if use_same_bandwidth:
      ke_dim_bandwidths = [np.exp(gp_cts_hps[0])] * dim
      gp_cts_hps = gp_cts_hps[1:]
    else:
      ke_dim_bandwidths = np.exp(gp_cts_hps[0:dim])
      gp_cts_hps = gp_cts_hps[dim:]
  elif kernel_type in ['esp']:
    if use_same_bandwidth:
      esp_dim_bandwidths = np.exp(gp_cts_hps[0:dim])
      gp_cts_hps = gp_cts_hps[dim:]
    else:
      esp_dim_bandwidths = np.exp(gp_cts_hps[0:dim*dim])
      gp_cts_hps = gp_cts_hps[dim*dim:]
  # Now iterate through the kernels for the rest of the parameters.
  if kernel_type == 'se':
    grp_kernels = [gp_kernel.SEKernel(dim=len(grp), scale=grp_scale, \
                     dim_bandwidths=get_sublist_from_indices(ke_dim_bandwidths, grp))
                   for grp in add_gp_groupings]
  elif kernel_type == 'matern':
    if 'nu' not in kernel_hyperparams or kernel_hyperparams['nu'] < 0:
      matern_nu = gp_dscr_hps[0]
      gp_dscr_hps = gp_dscr_hps[1:]
    else:
      matern_nu = kernel_hyperparams['nu']
    grp_kernels = [gp_kernel.MaternKernel(dim=len(grp), nu=matern_nu, scale=grp_scale, \
                     dim_bandwidths=get_sublist_from_indices(ke_dim_bandwidths, grp))
                   for grp in add_gp_groupings]
  elif kernel_type == 'poly':
    if 'order' not in kernel_hyperparams or kernel_hyperparams['order'] < 0:
      poly_order = gp_dscr_hps[0]
      gp_dscr_hps = gp_dscr_hps[1:]
    else:
      poly_order = kernel_hyperparams['order']
    grp_kernels = [gp_kernel.PolyKernel(dim=len(grp), order=poly_order, scale=grp_scale, \
                     dim_scalings=get_sublist_from_indices(ke_dim_bandwidths, grp))
                   for grp in add_gp_groupings]
  elif kernel_type == 'expdecay':
    exp_decay_offset = np.exp(gp_cts_hps[0])
    exp_decay_powers = np.exp(gp_cts_hps[1:dim+1])
    gp_cts_hps = gp_cts_hps[dim+1:]
    grp_kernels = [gp_kernel.ExpDecayKernel(dim=len(grp), scale=grp_scale, \
                    offset=exp_decay_offset, powers=exp_decay_powers)
                   for grp in add_gp_groupings]
  elif kernel_type == 'esp':
    if not np.isscalar(esp_order):
      esp_order = np.asscalar(esp_order)
    if esp_kernel_type == 'se':
      grp_kernels = [gp_kernel.ESPKernelSE(dim=dim, scale=scale, order=int(esp_order),
                                           dim_bandwidths=esp_dim_bandwidths)]
    elif esp_kernel_type == 'matern':
      if 'esp_matern_nu' not in kernel_hyperparams:
        matern_nu = [gp_dscr_hps[0]] * dim
        gp_dscr_hps = gp_dscr_hps[1:]
      else:
        matern_nu = [kernel_hyperparams['esp_matern_nu']] * dim
      grp_kernels = [gp_kernel.ESPKernelMatern(dim=dim, nu=matern_nu,
                                               scale=scale, order=int(esp_order),
                                               dim_bandwidths=esp_dim_bandwidths)]
  else:
    raise Exception('Unknown kernel type %s!'%(kernel_type))
  if is_additive:
    euc_kernel = gp_kernel.AdditiveKernel(scale=scale, kernel_list=grp_kernels,
                                          groupings=add_gp_groupings)
  else:
    euc_kernel = grp_kernels[0]
  return euc_kernel, gp_cts_hps, gp_dscr_hps


