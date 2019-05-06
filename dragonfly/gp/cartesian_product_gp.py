"""
  Harness for GPs on cartesian product domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-method
# pylint: disable=no-member

from __future__ import print_function
from argparse import Namespace
import numpy as np
# Local imports
from ..exd import domains
from ..exd.cp_domain_utils import load_cp_domain_from_config_file
from . import gp_core, mf_gp
from .euclidean_gp import get_euclidean_integral_gp_kernel_with_scale, \
                            prep_euclidean_integral_kernel_hyperparams
from .kernel import CartesianProductKernel, HammingKernel
from ..utils.general_utils import get_idxs_from_list_of_lists
from ..utils.option_handler import get_option_specs, load_options
from ..utils.reporters import get_reporter

# Part 1: Parameters and Arguments
# ================================

# Domain kernel parameters
_DFLT_DOMAIN_EUC_KERNEL_TYPE = 'matern'
_DFLT_DOMAIN_INT_KERNEL_TYPE = 'matern'
_DFLT_DOMAIN_DISCRETE_NUMERIC_KERNEL_TYPE = 'matern'
_DFLT_DOMAIN_DISCRETE_KERNEL_TYPE = 'hamming'
_DFLT_DOMAIN_NN_KERNEL_TYPE = 'otmann'
_DFLT_DOMAIN_MATERN_NU = 2.5
# Multi-fidelity parameters
_DFLT_FIDEL_EUC_KERNEL_TYPE = 'se'
_DFLT_FIDEL_INT_KERNEL_TYPE = 'se'
_DFLT_FIDEL_DISCRETE_NUMERIC_KERNEL_TYPE = 'se'
_DFLT_FIDEL_DISCRETE_KERNEL_TYPE = 'hamming'
_DFLT_FIDEL_MATERN_NU = 2.5

# Basic parameters
basic_cart_product_gp_args = [
  # For Euclidean domains ------------------------------------------------------------
  # These parameters will be shared by both Euclidean domains and Discrete Euclidean
  # domains.
  get_option_specs('dom_euc_kernel_type', False, 'default',
    'Kernel type for euclidean domains. '),
  get_option_specs('dom_euc_use_same_bandwidth', False, False,
    ('If true, will use same bandwidth on all dimensions. Applies only '
     'when kernel_type is se or matern. Default=False.')),
  get_option_specs('dom_euc_matern_nu', False, 'default', \
    'Specify nu value for matern kernel. If negative, will fit.'),
  get_option_specs('dom_euc_poly_order', False, 1,
    'Order of the polynomial kernle to be used for Euclidean domains. ' + \
    'Default is 1 (linear kernel).'),
  # Additive models for Euclidean spaces
  get_option_specs('dom_euc_use_additive_gp', False, False,
    'Whether or not to use an additive GP. '),
  get_option_specs('dom_euc_add_max_group_size', False, 6,
    'The maximum number of groups in the additive grouping. '),
  get_option_specs('dom_euc_add_grouping_criterion', False, 'randomised_ml',
    'Specify the grouping algorithm, should be one of {randomised_ml}'),
  get_option_specs('dom_euc_num_groups_per_group_size', False, -1,
    'The number of groups to try per group size.'),
  get_option_specs('dom_euc_add_group_size_criterion', False, 'sampled',
    'Specify how to pick the group size, should be one of {max,sampled}.'),
  # ESP for Euclidean spaces
  get_option_specs('dom_euc_esp_order', False, -1,
    'Order of the esp kernel. '),
  get_option_specs('dom_euc_esp_kernel_type', False, 'se',
    'Specify type of kernel. This depends on the application.'),
  get_option_specs('dom_euc_esp_matern_nu', False, 'default', \
    ('Specify the nu value for matern kernel. If negative, will fit.')),
  # For Integral domains --------------------------------------------------------------
  get_option_specs('dom_int_kernel_type', False, 'default',
    'Kernel type for integral domains. '),
  get_option_specs('dom_int_use_same_bandwidth', False, False,
    ('If true, will use same bandwidth on all dimensions. Applies only '
     'when kernel_type is se or matern. Default=False.')),
  get_option_specs('dom_int_matern_nu', False, 'default', \
    'Specify nu value for matern kernel. If negative, will fit.'),
  get_option_specs('dom_int_poly_order', False, 1,
    'Order of the polynomial kernle to be used for Integral domains. ' + \
    'Default is 1 (linear kernel).'),
  # Additive models for Integral spaces
  get_option_specs('dom_int_use_additive_gp', False, False,
    'Whether or not to use an additive GP. '),
  get_option_specs('dom_int_add_max_group_size', False, 6,
    'The maximum number of groups in the additive grouping. '),
  get_option_specs('dom_int_add_grouping_criterion', False, 'randomised_ml',
    'Specify the grouping algorithm, should be one of {randomised_ml}'),
  get_option_specs('dom_int_num_groups_per_group_size', False, -1,
    'The number of groups to try per group size.'),
  get_option_specs('dom_int_add_group_size_criterion', False, 'sampled',
    'Specify how to pick the group size, should be one of {max,sampled}.'),
  # ESP for Integral spaces
  get_option_specs('dom_int_esp_order', False, -1,
    'Order of the esp kernel. '),
  get_option_specs('dom_int_esp_kernel_type', False, 'se',
    'Specify type of kernel. This depends on the application.'),
  get_option_specs('dom_int_esp_matern_nu', False, 'default', \
    ('Specify the nu value for matern kernel. If negative, will fit.')),
  # For Discrete Numeric domains --------------------------------------------------------
  get_option_specs('dom_disc_num_kernel_type', False, 'default',
    'Kernel type for discrete numeric domains. '),
  get_option_specs('dom_disc_num_use_same_bandwidth', False, False,
    ('If true, will use same bandwidth on all dimensions. Applies only '
     'when kernel_type is se or matern. Default=False.')),
  get_option_specs('dom_disc_num_matern_nu', False, 'default', \
    'Specify nu value for matern kernel. If negative, will fit.'),
  get_option_specs('dom_disc_num_poly_order', False, 1,
    'Order of the polynomial kernle to be used for Integral domains. ' + \
    'Default is 1 (linear kernel).'),
  # ESP for Discrete Numeric spaces
  get_option_specs('dom_disc_num_esp_order', False, -1,
    'Order of the esp kernel. '),
  get_option_specs('dom_disc_num_esp_kernel_type', False, 'se',
    'Specify type of kernel. This depends on the application.'),
  get_option_specs('dom_disc_num_esp_matern_nu', False, 'default', \
    ('Specify the nu value for matern kernel. If negative, will fit.')),
  # For Discrete domains -----------------------------------------------------------------
  get_option_specs('dom_disc_kernel_type', False, 'default',
    'Kernel type for discrete domains.'),
  get_option_specs('dom_disc_hamming_use_same_weight', False, False,
    'If true, use same weight for all dimensions of the hamming kernel.'),
  # For NN domains -----------------------------------------------------------------------
  get_option_specs('dom_nn_kernel_type', False, 'default',
    'Kernel type for NN Domains.'),
  # Arguments specific to OTMANN ---------------------------------------------------------
  get_option_specs('otmann_dist_type', False, 'lp-emd',
    'The type of distance. Should be lp, emd or lp-emd.'),
  get_option_specs('otmann_kernel_type', False, 'lpemd_sum',
    'The Otmann kernel type. Should be one of lp, emd, lpemd_sum, or lpemd_prod.'),
  get_option_specs('otmann_choose_mislabel_struct_coeffs', False, 'use_given',
    ('How to choose the mislabel and struct coefficients. Should be one of ' +
     'tune_coeffs or use_given. In the latter case, otmann_mislabel_coeffs and ' +
     'otmann_struct_coeffs should be non-empty.')),
  get_option_specs('otmann_mislabel_coeffs', False, '1.0-1.0-1.0-1.0',
    'The mislabel coefficients specified as a string. If -1, it means we will tune.'),
  get_option_specs('otmann_struct_coeffs', False, '0.1-0.25-0.61-1.5',
    'The struct coefficients specified as a string. If -1, it means we will tune.'),
  get_option_specs('otmann_lp_power', False, 1,
    'The powers to use in the LP distance for the kernel.'),
  get_option_specs('otmann_emd_power', False, 2,
    'The powers to use in the EMD distance for the kernel.'),
  get_option_specs('otmann_non_assignment_penalty', False, 1.0,
    'The non-assignment penalty for the OTMANN distance.'),
  ]

cartesian_product_gp_args = gp_core.mandatory_gp_args + basic_cart_product_gp_args

basic_mf_cart_product_gp_args = [ \
  # Fidelity kernel Euclidean -----------------------------------------------------------
  get_option_specs('fidel_euc_kernel_type', False, 'se', \
    ('Type of kernel for the Euclidean part of the fidelity space. Should be se, ' +
     'matern, poly or expdecay')),
  get_option_specs('fidel_euc_matern_nu', False, 2.5, \
    ('Specify the nu value for the matern kernel. If negative, will fit.')),
  get_option_specs('fidel_euc_use_same_bandwidth', False, False, \
    ('If true, will use same bandwidth on all Euclidean fidelity dimensions. Applies ' +
     'only when fidel_kernel_type is se or matern. Default=False.')),
  # Fidelity kernel Integral -----------------------------------------------------------
  get_option_specs('fidel_int_kernel_type', False, 'se', \
    'Type of kernel for the fidelity space. Should be se, matern, poly or expdecay'),
  get_option_specs('fidel_int_matern_nu', False, 2.5, \
    ('Specify the nu value for the matern kernel. If negative, will fit.')),
  get_option_specs('fidel_int_use_same_bandwidth', False, False, \
    ('If true, will use same bandwidth on all integral fidelity dimensions. Applies ' +
     'only when fidel_kernel_type is se or matern. Default=False.')),
  # Fidelity kernel Discrete Numeric ---------------------------------------------------
  get_option_specs('fidel_disc_num_kernel_type', False, 'se', \
    'Type of kernel for the fidelity space. Should be se, matern, poly or expdecay'),
  get_option_specs('fidel_disc_num_matern_nu', False, 2.5, \
    ('Specify the nu value for the matern kernel. If negative, will fit.')),
  get_option_specs('fidel_disc_num_use_same_bandwidth', False, False, \
    ('If true, will use same bandwidth on all integral fidelity dimensions. Applies ' +
     'only when fidel_kernel_type is se or matern. Default=False.')),
  # Fidelity Kernel Discrete ----------------------------------------------------------
  get_option_specs('fidel_disc_kernel_type', False, 'default',
    'Kernel type for discrete domains.'),
  get_option_specs('fidel_disc_hamming_use_same_weight', False, False,
    'If true, use same weight for all dimensions of the hamming kernel.'),
  ]

cartesian_product_mf_gp_args = cartesian_product_gp_args + basic_mf_cart_product_gp_args


def get_default_kernel_type(domain_type):
  """ Returns default kernel type for the domain. """
  if domain_type == 'euclidean':
    return _DFLT_DOMAIN_EUC_KERNEL_TYPE
  elif domain_type == 'discrete_euclidean':
    return _DFLT_DOMAIN_EUC_KERNEL_TYPE
  elif domain_type == 'integral':
    return _DFLT_DOMAIN_INT_KERNEL_TYPE
  elif domain_type == 'prod_discrete':
    return _DFLT_DOMAIN_DISCRETE_KERNEL_TYPE
  elif domain_type == 'prod_discrete_numeric':
    return _DFLT_DOMAIN_DISCRETE_NUMERIC_KERNEL_TYPE
  elif domain_type == 'neural_network':
    return _DFLT_DOMAIN_NN_KERNEL_TYPE
  else:
    raise ValueError('Unknown domain_type: %s.'%(domain_type))


# Part 2: CPGP and CPGPFitter
# ====================================================================================
class CPGP(gp_core.GP):
  """ GP class for Cartesian Product GPs. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, X, Y, kernel, mean_func, noise_var, domain_lists_of_dists=None,
               build_posterior=True, reporter=None,
               handle_non_psd_kernels='project_first'):
    """
      Constructor
      X, Y: data
      kernel: a kernel object
    """
    if domain_lists_of_dists is None:
      domain_lists_of_dists = [None] * kernel.num_kernels
    self.domain_lists_of_dists = domain_lists_of_dists
    super(CPGP, self).__init__(X, Y, kernel, mean_func, noise_var, build_posterior,
                               reporter, handle_non_psd_kernels)

  def set_domain_lists_of_dists(self, domain_lists_of_dists):
    """ Sets lists of distances. """
    self.domain_lists_of_dists = domain_lists_of_dists

  def _child_str(self):
    """ String representation of the GP. """
    if len(self.X) > 0:
      mean_str = 'mu[#0]=%0.4f, '%(self.mean_func([self.X[0]])[0])
    else:
      mean_str = ''
    return mean_str + str(self.kernel)

  def _get_training_kernel_matrix(self):
    """ Returns the training kernel matrix. """
    n = len(self.X)
    ret = self.kernel.hyperparams['scale'] * np.ones((n, n))
    for idx, kern in enumerate(self.kernel.kernel_list):
      if self.domain_lists_of_dists[idx] is not None:
        ret *= kern.evaluate_from_dists(self.domain_lists_of_dists[idx])
      else:
        curr_X = get_idxs_from_list_of_lists(self.X, idx)
        ret *= kern(curr_X, curr_X)
    return ret


class CPMFGP(mf_gp.MFGP):
  """ Multi-fidelity GP Class for Cartesian Product GPs. """

  def __init__(self, ZZ, XX, YY, mf_kernel, mean_func, noise_var,
               kernel_scale=None, fidel_space_kernel=None, domain_kernel=None,
               fidel_space_lists_of_dists=None, domain_lists_of_dists=None,
               build_posterior=True, reporter=None,
               handle_non_psd_kernels='project_first'):
    """ Constructor. """
    # pylint: disable=too-many-arguments
    if mf_kernel is None:
      mf_kernel = CartesianProductKernel(kernel_scale,
                                         [fidel_space_kernel, domain_kernel])
    self.kernel_scale = kernel_scale
    self.fidel_space_kernel = fidel_space_kernel
    self.domain_kernel = domain_kernel
    # Now check the lists_of_dists
    if fidel_space_lists_of_dists is None:
      fidel_space_lists_of_dists = [None] * mf_kernel[0].num_kernels
    if domain_lists_of_dists is None:
      domain_lists_of_dists = [None] * mf_kernel[0].num_kernels
    self.fidel_space_lists_of_dists = fidel_space_lists_of_dists
    self.domain_lists_of_dists = domain_lists_of_dists
    # Finally call the super constructor.
    super(CPMFGP, self).__init__(ZZ, XX, YY, mf_kernel, mean_func, noise_var,
                                 build_posterior=build_posterior,
                                 reporter=reporter,
                                 handle_non_psd_kernels=handle_non_psd_kernels)

  def _child_str(self):
    """ String representation of the GP. """
    if len(self.X) > 0:
      mean_str = 'mu=%0.4f, '%(self.mean_func([self.X[0]])[0])
    else:
      mean_str = ''
    return mean_str + str(self.kernel)

  def set_fidel_space_lists_of_dists(self, fidel_space_lists_of_dists):
    """ Sets the fidel_space lists of dists. """
    self.fidel_space_lists_of_dists = fidel_space_lists_of_dists

  def set_domain_lists_of_dists(self, domain_lists_of_dists):
    """ Sets the domain lists of dists. """
    self.domain_lists_of_dists = domain_lists_of_dists

  @classmethod
  def _get_train_kernel_matrix_from_data_and_lists_of_dists(cls, data, kern,
                                                            lists_of_dists):
    """ Returns the training kernel matrix given the data, kernel, and lists_of_dists. """
    n = len(data)
    ret = kern.hyperparams['scale'] * np.ones((n, n))
    for idx, kern in enumerate(kern.kernel_list):
      if lists_of_dists[idx] is not None:
        ret *= kern.evaluate_from_dists(lists_of_dists[idx])
      else:
        curr_data = get_idxs_from_list_of_lists(data, idx)
        ret *= kern(curr_data, curr_data)
    return ret

  def _get_training_kernel_matrix(self):
    """ Returns the training kernel matrix. """
    if self.fidel_space_kernel is None or self.domain_kernel is None:
      return self.kernel(self.X, self.X)
    else:
      KF = self._get_train_kernel_matrix_from_data_and_lists_of_dists(self.ZZ,
             self.fidel_space_kernel, self.fidel_space_lists_of_dists)
      KD = self._get_train_kernel_matrix_from_data_and_lists_of_dists(self.XX,
             self.domain_kernel, self.domain_lists_of_dists)
      return self.kernel_scale * KF * KD


# Part 3: GP Fitters
# ================================

class CPGPFitter(gp_core.GPFitter):
  """ Fits a cartesian product GP by tuning the hyper-parameters. """

  def __init__(self, X, Y, domain, domain_kernel_ordering=None,
               domain_lists_of_dists=None, domain_dist_computers=None,
               options=None, reporter=None):
    """ Constructor. """
    # pylint: disable=no-member
    # Set domain
    if isinstance(domain, str):
      domain, config_orderings = load_cp_domain_from_config_file(domain)
      if domain_kernel_ordering is None:
        domain_kernel_ordering = config_orderings.kernel_ordering
    # Check before assignment
    if not isinstance(domain, domains.Domain):
      raise ValueError('domain should be instance of domain.Domain or a file name' +
                       ' to a configuration json/protobuf file describing the domain.')
    if domain_kernel_ordering is None:
      raise ValueError('If domain is not a config file name, domain_kernel_ordering' +
                       ' cannot be None.')
    self.domain_num_domains = domain.num_domains
    self.domain = domain
    self.domain_kernel_ordering = domain_kernel_ordering
    # Now create a struct to store kernel dependent parameters
    if domain_lists_of_dists is None:
      domain_lists_of_dists = [None] * self.domain_num_domains
    if domain_dist_computers is None:
      domain_dist_computers = [None] * self.domain_num_domains
    self.domain_lists_of_dists = domain_lists_of_dists
    self.domain_dist_computers = domain_dist_computers
    self.domain_kernel_params_for_each_domain = [Namespace() for _ in
                                          range(self.domain_num_domains)]
    for idx in range(self.domain_num_domains):
      self.domain_kernel_params_for_each_domain[idx].list_of_dists = \
        domain_lists_of_dists[idx]
      self.domain_kernel_params_for_each_domain[idx].dist_computer = \
        domain_dist_computers[idx]
    # Create reporter and options
    reporter = get_reporter(reporter)
    options = load_options(cartesian_product_gp_args, partial_options=options)
    # Call super constructor
    super(CPGPFitter, self).__init__(X, Y, options, reporter)

  def _child_set_up(self):
    """ Set up parameters for the CPGPFitter. """
    # Set up for the kernel scale
    self.param_order.append(['kernel_scale', 'cts'])
    self.kernel_scale_log_bounds = [np.log(0.03 * self.Y_var), np.log(30 * self.Y_var)]
    self.cts_hp_bounds.append(self.kernel_scale_log_bounds)
    _set_up_hyperparams_for_domain(self, self.X, self.domain, 'dom',
                                   self.domain_kernel_ordering,
                                   self.domain_kernel_params_for_each_domain,
                                   self.domain_dist_computers,
                                   self.domain_lists_of_dists)

  # Methods to build the GP ------------------------------------------------------
  def _child_build_gp(self, mean_func, noise_var, gp_cts_hps, gp_dscr_hps,
                      other_gp_params=None, *args, **kwargs):
    """ Builds a Domain product GP. """
    kernel_scale = np.exp(gp_cts_hps[0])
    gp_cts_hps = gp_cts_hps[1:]
    cp_kernel, gp_cts_hps, gp_dscr_hps = _build_kernel_for_domain(self.domain, 'dom',
      kernel_scale, gp_cts_hps, gp_dscr_hps, other_gp_params, self.options,
      self.domain_kernel_ordering, self.domain_kernel_params_for_each_domain)
    ret_gp = CPGP(self.X, self.Y, cp_kernel, mean_func, noise_var,
                  domain_lists_of_dists=self.domain_lists_of_dists, *args, **kwargs)
    return ret_gp, gp_cts_hps, gp_dscr_hps


class CPMFGPFitter(mf_gp.MFGPFitter):
  """ A Fitter for multi-fidelity GPs in Cartesian Product spaces. """

  def __init__(self, ZZ, XX, YY, config=None,
               fidel_space=None, domain=None,
               fidel_space_kernel_ordering=None, domain_kernel_ordering=None,
               fidel_space_lists_of_dists=None, domain_lists_of_dists=None,
               fidel_space_dist_computers=None, domain_dist_computers=None,
               options=None, reporter=None):
    """ Constructor. """
    # pylint: disable=too-many-arguments
    # Load options
    reporter = get_reporter(reporter)
    options = load_options(cartesian_product_mf_gp_args, partial_options=options)
    # Read from config
    if config is not None:
      if isinstance(config, str):
        from ..exd.exd_utils import load_config_file
        config = load_config_file(config)
      self.config = config
      self.fidel_space = config.fidel_space
      self.domain = config.domain
      self.fidel_space_kernel_ordering = config.fidel_space_orderings.kernel_ordering
      self.domain_kernel_ordering = config.domain_orderings.kernel_ordering
      self.fidel_space_num_domains = self.fidel_space.num_domains
    elif fidel_space is not None and domain is not None and \
      fidel_space_kernel_ordering is not None and domain_kernel_ordering is not None:
      self.config = None
      self.fidel_space = fidel_space
      self.domain = domain
      self.fidel_space_kernel_ordering = fidel_space_kernel_ordering
      self.domain_kernel_ordering = domain_kernel_ordering
    else:
      raise ValueError('Provide either config or parameters fidel_space, domain' +
                       ' fidel_space_kernel_ordering, and domain_kernel_ordering.')
    self.fidel_space_num_domains = self.fidel_space.num_domains
    self.domain_num_domains = self.domain.num_domains
    # Create a structure to store kernel parameters
    self.fidel_space_kernel_params_for_each_domain = \
      [Namespace() for _ in range(self.fidel_space_num_domains)]
    self.domain_kernel_params_for_each_domain = \
      [Namespace() for _ in range(self.domain_num_domains)]
    # Store lists of dists & dist_computers
    if fidel_space_lists_of_dists is None:
      fidel_space_lists_of_dists = [None] * self.fidel_space_num_domains
    if domain_lists_of_dists is None:
      domain_lists_of_dists = [None] * self.domain_num_domains
    if fidel_space_dist_computers is None:
      fidel_space_dist_computers = [None] * self.fidel_space_num_domains
    if domain_dist_computers is None:
      domain_dist_computers = [None] * self.domain_num_domains
    self.fidel_space_lists_of_dists = fidel_space_lists_of_dists
    self.domain_lists_of_dists = domain_lists_of_dists
    self.fidel_space_dist_computers = fidel_space_dist_computers
    self.domain_dist_computers = domain_dist_computers
    for idx in range(self.fidel_space_num_domains):
      self.fidel_space_kernel_params_for_each_domain[idx].list_of_dists = \
        fidel_space_lists_of_dists[idx]
      self.fidel_space_kernel_params_for_each_domain[idx].dist_computer = \
        fidel_space_dist_computers[idx]
    for idx in range(self.domain_num_domains):
      self.domain_kernel_params_for_each_domain[idx].list_of_dists = \
        domain_lists_of_dists[idx]
      self.domain_kernel_params_for_each_domain[idx].dist_computer = \
        domain_dist_computers[idx]
    # Call super constructor
    super(CPMFGPFitter, self).__init__(ZZ, XX, YY, options, reporter)

  # Child set up
  def _child_set_up(self):
    """ Sets parameters for the GPFitter. """
    self.param_order.append(['kernel_scale', 'cts'])
    self.kernel_scale_log_bounds = [np.log(0.03 * self.Y_var), np.log(30 * self.Y_var)]
    self.cts_hp_bounds.append(self.kernel_scale_log_bounds)
    _set_up_hyperparams_for_domain(self, self.ZZ, self.fidel_space, 'fidel',
                                   self.fidel_space_kernel_ordering,
                                   self.fidel_space_kernel_params_for_each_domain,
                                   self.fidel_space_dist_computers,
                                   self.fidel_space_lists_of_dists)
    _set_up_hyperparams_for_domain(self, self.XX, self.domain, 'dom',
                                   self.domain_kernel_ordering,
                                   self.domain_kernel_params_for_each_domain,
                                   self.domain_dist_computers,
                                   self.domain_lists_of_dists)

  def _child_build_gp(self, mean_func, noise_var, gp_cts_hps, gp_dscr_hps,
                      other_gp_params=None, *args, **kwargs):
    """ Builds a Cartesian product Multi-fidelity GP. """
    kernel_scale = np.exp(gp_cts_hps[0])
    gp_cts_hps = gp_cts_hps[1:]
    # Create kernels
    fidel_space_kernel, gp_cts_hps, gp_dscr_hps = _build_kernel_for_domain(
      self.fidel_space, 'fidel', 1.0, gp_cts_hps, gp_dscr_hps, other_gp_params,
      self.options, self.fidel_space_kernel_ordering,
      self.fidel_space_kernel_params_for_each_domain)
    domain_kernel, gp_cts_hps, gp_dscr_hps = _build_kernel_for_domain(self.domain, 'dom',
      1.0, gp_cts_hps, gp_dscr_hps, other_gp_params, self.options,
      self.domain_kernel_ordering, self.domain_kernel_params_for_each_domain)
    # build the GP
    ret_gp = CPMFGP(self.ZZ, self.XX, self.YY, None, mean_func, noise_var,
                    kernel_scale, fidel_space_kernel, domain_kernel,
                    self.fidel_space_lists_of_dists, self.domain_lists_of_dists,
                    *args, **kwargs)
    return ret_gp, gp_cts_hps, gp_dscr_hps


# Part 4: Some utilities we will use above
# ========================================

# 4.1 Utilities to set up GP Hyperparams -------------------------------------------------
def _set_up_hyperparams_for_domain(fitter, X_data, gp_domain, dom_prefix,
  kernel_ordering, kernel_params_for_each_domain, dist_computers, lists_of_dists):
  """ This modifies the fitter object. """
  # pylint: disable=too-many-branches
  for dom_idx, dom, kernel_type in \
    zip(range(gp_domain.num_domains), gp_domain.list_of_domains, kernel_ordering):
    dom_type = dom.get_type()
    dom_identifier = '%s-%d-%s'%(dom_prefix, dom_idx, dom_type)
    # Kernel type
    if kernel_type == '' or kernel_type is None:
      # If none is specified, use the one given in options
      kernel_type = _get_kernel_type_from_options(dom_type, dom_prefix,
                                                  fitter.options)
    if kernel_type == 'default':
      kernel_type = get_default_kernel_type(dom.get_type())
    # Iterate through each individual domain and add it to the hyper parameters
    curr_dom_Xs = get_idxs_from_list_of_lists(X_data, dom_idx)
    # Some conditional options
    if dom_type in ['euclidean', 'integral', 'prod_discrete_numeric',
                    'discrete_euclidean']:
      use_same_bw, matern_nu, esp_kernel_type, esp_matern_nu = \
        _get_euc_int_options(dom_type, dom_prefix, fitter.options)
      # Now set things up depending on the kernel type
      if kernel_type == 'se':
        fitter.cts_hp_bounds, fitter.param_order = _set_up_dim_bandwidths(
          dom_identifier, curr_dom_Xs, use_same_bw, dom.get_dim(),
          fitter.cts_hp_bounds, fitter.param_order)
      elif kernel_type == 'matern':
        if isinstance(matern_nu, float) and matern_nu < 0:
          fitter.dscr_hp_vals.append([0.5, 1.5, 2.5])
          fitter.param_order.append(['%s-%s'%(dom_identifier, 'matern_nu'), 'dscr'])
        fitter.cts_hp_bounds, fitter.param_order = _set_up_dim_bandwidths(
          dom_identifier, curr_dom_Xs, use_same_bw, dom.get_dim(),
          fitter.cts_hp_bounds, fitter.param_order)
      elif kernel_type == 'poly':
        raise NotImplementedError('Not implemented polynomial kernels yet.')
      elif kernel_type == 'expdecay':
        # Offset
        scale_range = fitter.Y_var / np.sqrt(fitter.num_tr_data + 0.0001)
        fitter.cts_hp_bounds.append(
          [np.log(0.1 * scale_range), np.log(10 * scale_range)])
        fitter.param_order.append(['%s-expdecay_log_offset'%(dom_identifier), 'cts'])
        # Power
        fitter.cts_hp_bounds.extend([[np.log(1e-1), np.log(50)]] * dom.get_dim())
        identifiers = [['%s-expdecay_log_power-%d'%(dom_identifier, i), 'cts']
                       for i in range(dom.get_dim())]
        fitter.param_order.extend(identifiers)
      elif kernel_type == 'esp':
        fitter.cts_hp_bounds, fitter.param_order = _set_up_dim_bandwidths(
          dom_identifier, curr_dom_Xs, use_same_bw, dom.get_dim(),
          fitter.cts_hp_bounds, fitter.param_order)
        # Matern nu
        if esp_kernel_type == 'matern' and esp_matern_nu < 0:
          fitter.dscr_hp_vals.append([0.5, 1.5, 2.5])
          fitter.param_order.append(['%s-%s'%(dom_identifier, 'esp_matern_nu'), 'dscr'])
        # ESP order
        esp_order_vals = range(1, dom.get_dim() // 2)
        fitter.param_order.append(['%s-%s'%(dom_identifier, 'esp_order'), 'dscr'])
        fitter.dscr_hp_vals.append(esp_order_vals)
      else:
        raise ValueError('Unknown kernel type "%s" for "%s" spaces.'%(kernel_type,
                         dom.get_type()))
    elif dom.get_type() in ['prod_discrete']:
      if kernel_type == 'hamming':
        disc_hamming_use_same_weight, = _get_disc_options(dom_prefix, fitter.options)
        fitter.cts_hp_bounds, fitter.param_order = _set_up_hamming_weights(dom_identifier,
          disc_hamming_use_same_weight, dom.get_dim(),
          fitter.cts_hp_bounds, fitter.param_order)
      else:
        raise ValueError('Unknown kernel type "%s" for "%s" spaces.'%(kernel_type,
                         dom.get_type()))
    elif dom.get_type() in ['neural_network']:
      if kernel_type == 'otmann':
        fitter.cts_hp_bounds, fitter.param_order = \
          _set_up_nn_domain_otmann(dom_idx, dom, dom_identifier, curr_dom_Xs,
            fitter.options, fitter.cts_hp_bounds, fitter.param_order,
            kernel_params_for_each_domain, dist_computers, lists_of_dists)
      else:
        raise ValueError('Unknown kernel type "%s" for "%s" spaces.'%(kernel_type,
                         dom.get_type()))


def _get_euc_int_options(dom_type, dom_prefix, options):
  """ Returns fieds from options depending on dom_type. """
  dom_type_code_dic = {'euclidean': 'euc',
                       'integral': 'int',
                       'prod_discrete_numeric': 'disc_num',
                       'discrete_euclidean': 'euc',
                      }
  def _extract_from_options(dom_type_str, property_str):
    """ Extracts the property from the dom_type. """
    attr_name_in_options = dom_prefix + '_' + dom_type_code_dic[dom_type_str] + \
                           '_' + property_str
    if hasattr(options, attr_name_in_options):
      return getattr(options, attr_name_in_options)
    else:
      return None
  # Now extract each property that we need
  use_same_bw = _extract_from_options(dom_type, 'use_same_bandwidth')
  matern_nu = _extract_from_options(dom_type, 'matern_nu')
  esp_kernel_type = _extract_from_options(dom_type, 'esp_kernel_type')
  esp_matern_nu = _extract_from_options(dom_type, 'esp_matern_nu')
  return use_same_bw, matern_nu, esp_kernel_type, esp_matern_nu


def _get_disc_options(dom_prefix, options):
  """ Returns the options, depending on dom_type. """
  def _extract_from_options(property_str):
    """ Extracts property from options. """
    attr_name_in_options = dom_prefix + '_' + property_str
    if hasattr(options, attr_name_in_options):
      return getattr(options, attr_name_in_options)
    else:
      return None
  disc_hamming_use_same_weight = _extract_from_options('disc_hamming_use_same_weight')
  return (disc_hamming_use_same_weight, )


def _get_kernel_type_from_options(dom_type, dom_prefix, options):
  """ Returns kernel type from options. """
  dom_type_descr_dict = {'euclidean': 'euc',
                         'discrete_euclidean': 'euc',
                         'integral': 'int',
                         'prod_discrete_numeric': 'disc_num',
                         'prod_discrete': 'disc',
                         'neural_network': 'nn',
                        }
  if dom_type not in dom_type_descr_dict.keys():
    raise ValueError('Unknown domain type %s.'%(dom_type))
  attr_name = '%s_%s_kernel_type'%(dom_prefix, dom_type_descr_dict[dom_type])
  return getattr(options, attr_name)


def _set_up_dim_bandwidths(dom_identifier, curr_dom_Xs, use_same_bandwidth, dim,
                           cts_hp_bounds, param_order):
  """ Set up for dim bandwidths. """
  hp_identifier = '%s-%s'%(dom_identifier, 'dom_bandwidths')
  if len(curr_dom_Xs) > 0:
    # If there is data
    assert len(curr_dom_Xs[0]) == dim # temporary: check this condition
    curr_dom_Xs = np.array(curr_dom_Xs)
    curr_dom_mean_diffs = curr_dom_Xs - curr_dom_Xs.mean(axis=0)
    curr_dom_X_std_norms = [np.linalg.norm(curr_dom_mean_diffs[:, idx]) + 1e-4
                            for idx in range(dim)]
  else:
    curr_dom_X_std_norms = [1.0] * dim
  if use_same_bandwidth:
    bandwidth_log_bounds = [np.log(0.01 * curr_dom_X_std_norms.min()),
                            np.log(100 * curr_dom_X_std_norms.max())]
    cts_hp_bounds.append(bandwidth_log_bounds)
    param_order.append([hp_identifier + '-same_bandwidth', 'cts'])
  else:
    bandwidth_log_bounds = [[np.log(0.01 * elem), np.log(100 * elem)] \
                            for elem in curr_dom_X_std_norms]
    cts_hp_bounds.extend(bandwidth_log_bounds)
    param_order.extend([[hp_identifier + '-%d'%(idx), 'cts']
                         for idx in range(len(bandwidth_log_bounds))])
  return cts_hp_bounds, param_order


def _set_up_hamming_weights(dom_identifier, use_same_weights, dim,
                            cts_hp_bounds, param_order):
  """ Set up for dim_weights in hamming kernel. """
  if use_same_weights or dim == 1:
    return cts_hp_bounds, param_order
  elif dim == 2:
    cts_hp_bounds.append([0, 1])
    param_order.append([dom_identifier + '-hamming_wt-2D', 'cts'])
  elif dim > 2:
    cts_hp_bounds.extend([[0, 1]] * dim)
    param_order.extend([['%s-hamming_wts-%d'%(dom_identifier, idx), 'cts']
                             for idx in range(dim)])
  else:
    raise ValueError('dim=%0.4f should be a positive integer.'%(dim))
  return cts_hp_bounds, param_order


def _set_up_nn_domain_otmann(dom_idx, dom, dom_identifier, curr_dom_Xs, options,
  cts_hp_bounds, param_order, kernel_params_for_each_domain, dist_computers,
  lists_of_dists):
  """ Set up for NN Domain. """
  _process_otmann_kernel_params(dom_idx, options, kernel_params_for_each_domain,
                                dist_computers)
  # Check if we need to pre-compute distances -----------------------------------------
  if options.otmann_choose_mislabel_struct_coeffs == 'use_given' and \
    kernel_params_for_each_domain[dom_idx].list_of_dists is None:
    # Then compute the distances
    if dist_computers[dom_idx] is None:
      from ..nn.otmann import get_otmann_distance_computer_from_args
      curr_dist_computer = get_otmann_distance_computer_from_args(
        dom.nn_type, options.otmann_non_assignment_penalty,
        kernel_params_for_each_domain[dom_idx].otmann_mislabel_coeffs,
        kernel_params_for_each_domain[dom_idx].otmann_struct_coeffs,
        kernel_params_for_each_domain[dom_idx].otmann_dist_type)
    else:
      curr_dist_computer = dist_computers[dom_idx]
    # Add the current distance computer to dist_computers
    kernel_params_for_each_domain[dom_idx].otmann_distance_computer = \
      curr_dist_computer
    dist_computers[dom_idx] = curr_dist_computer
    # Now compute lists of dists
    curr_list_of_dists = curr_dist_computer(curr_dom_Xs, curr_dom_Xs)
    kernel_params_for_each_domain[dom_idx].list_of_dists = \
      curr_list_of_dists
    lists_of_dists[dom_idx] = curr_list_of_dists
  # Create bounds here ----------------------------------------------------------------
  # 1. Relative weights for LP/EMD terms
  if kernel_params_for_each_domain[dom_idx].otmann_kernel_type == 'lpemd_sum':
    cts_hp_bounds.append([0, 1]) # controls relative weighting of each term
    param_order.append([dom_identifier + '-lp_emd_tradeoff', 'cts'])
  # 2. Betas
  lp_beta_log_bounds = [[np.log(1e-9), np.log(1e-3)]] * \
    kernel_params_for_each_domain[dom_idx].otmann_num_mislabel_struct_coeffs
  emd_beta_log_bounds = [[np.log(1e-1), np.log(1e2)]] * \
    kernel_params_for_each_domain[dom_idx].otmann_num_mislabel_struct_coeffs
  if kernel_params_for_each_domain[dom_idx].otmann_dist_type == 'lp':
    all_beta_bounds = lp_beta_log_bounds
  elif kernel_params_for_each_domain[dom_idx].otmann_dist_type == 'emd':
    all_beta_bounds = emd_beta_log_bounds
  elif kernel_params_for_each_domain[dom_idx].otmann_dist_type == 'lp-emd':
    all_beta_bounds = [j for i in zip(lp_beta_log_bounds, emd_beta_log_bounds)
                       for j in i]
  cts_hp_bounds.extend(all_beta_bounds)
  param_order.extend([[dom_identifier + '-beta-%d'%(idx), 'cts']
                       for idx in range(len(all_beta_bounds))])
  # 3 & 4. mislabel/struct coeffs
  if kernel_params_for_each_domain[dom_idx].otmann_to_tune_mislabel_struct_coeffs:
    cts_hp_bounds.append([0.001, 2.0]) # mislabel coefficient (not in log space)
    param_order.append([dom_identifier + '-mislabel_coeff', 'cts'])
    cts_hp_bounds.append([0.001, 2.0]) # structural coefficient (not in log space)
    param_order.append([dom_identifier + '-struct_coeff', 'cts'])
  return cts_hp_bounds, param_order


def _process_otmann_kernel_params(dom_idx, options, kernel_params_for_each_domain,
                                  dist_computers):
  """ Preprocesses the structural and mislabel coefficients for domain in dom_idx. """
  # First check if dist_type and kernel_type are consistent ----------------------------
  otmann_dist_type = options.otmann_dist_type
  otmann_kernel_type = options.otmann_kernel_type
  otmann_choose_mislabel_struct_coeffs = \
    options.otmann_choose_mislabel_struct_coeffs
  otmann_mislabel_coeffs = options.otmann_mislabel_coeffs
  otmann_struct_coeffs = options.otmann_struct_coeffs
  if otmann_dist_type in ['lp', 'emd'] and otmann_kernel_type != otmann_dist_type:
    raise ValueError('If dist_type is %s, then kernel_type should be %s.'%(
                     otmann_dist_type, otmann_dist_type))
  elif otmann_dist_type == 'lp-emd' and \
       otmann_kernel_type not in ['lpemd_prod', 'lpemd_sum']:
    raise ValueError('If otmann_dist_type is lp-emd, then otmann_kernel_type should ' +
                     'be lpemd_sum or lpemd_prod.')
  # Check if the mislabel/struct coeffs are give as appropriate
  if otmann_choose_mislabel_struct_coeffs == 'use_given' and \
    (otmann_mislabel_coeffs == '' or otmann_struct_coeffs == ''):
    raise ValueError(('If choose_mislabel_struct_coeffs is use_given, then ' +
      'mislabel_coeffs and struct_coeffs cannot be empty. Given mislabel_coeffs=%s,' +
      ' struct_coeffs=%s')%(otmann_mislabel_coeffs, otmann_struct_coeffs))
  # Now process the mislabel and struct coeffs -----------------------------------------
  if otmann_choose_mislabel_struct_coeffs == 'tune_coeffs':
    # If they are integers or floats
    otmann_num_mislabel_struct_coeffs = 1
    otmann_to_tune_mislabel_struct_coeffs = True
  elif (isinstance(otmann_mislabel_coeffs, list) and
        isinstance(otmann_struct_coeffs, list)) or \
       (isinstance(otmann_mislabel_coeffs, str) and
        isinstance(otmann_struct_coeffs, str)):
    # If they are lists or strings
    if isinstance(otmann_mislabel_coeffs, str):
      # If they are strings, obtain the lists
      otmann_mislabel_coeffs = [float(x) for x in otmann_mislabel_coeffs.split('-')]
      otmann_struct_coeffs = [float(x) for x in otmann_struct_coeffs.split('-')]
    else:
      otmann_mislabel_coeffs = otmann_mislabel_coeffs
      otmann_struct_coeffs = otmann_struct_coeffs
    # Other parameters
    otmann_num_mislabel_struct_coeffs = len(otmann_mislabel_coeffs)
    otmann_to_tune_mislabel_struct_coeffs = False
    # Check if the lengths are the same
    if len(otmann_mislabel_coeffs) != len(otmann_struct_coeffs):
      raise ValueError('Length of mislabel and structural coefficients must be same.' +
        'Given: mislabel: %s (%d), struct:%s (%d).'%(
        str(otmann_mislabel_coeffs), len(otmann_mislabel_coeffs),
        str(otmann_struct_coeffs), len(otmann_struct_coeffs)))
  else:
    raise ValueError('Bad format for mislabel and structural coefficients.' +
      'Given: mislabel: %s, struct:%s.'%(
      str(otmann_mislabel_coeffs), str(otmann_struct_coeffs)))
  # Now store them in kernel_params_for_each_domain ------------------------------------
  kernel_params_for_each_domain[dom_idx].otmann_mislabel_coeffs = \
    otmann_mislabel_coeffs
  kernel_params_for_each_domain[dom_idx].otmann_struct_coeffs = \
    otmann_struct_coeffs
  kernel_params_for_each_domain[dom_idx].otmann_num_mislabel_struct_coeffs = \
    otmann_num_mislabel_struct_coeffs
  kernel_params_for_each_domain[dom_idx].otmann_to_tune_mislabel_struct_coeffs = \
    otmann_to_tune_mislabel_struct_coeffs
  kernel_params_for_each_domain[dom_idx].otmann_choose_mislabel_struct_coeffs = \
    otmann_choose_mislabel_struct_coeffs
  kernel_params_for_each_domain[dom_idx].otmann_dist_type = otmann_dist_type
  kernel_params_for_each_domain[dom_idx].otmann_kernel_type = otmann_kernel_type
  # store other parameters from options
  kernel_params_for_each_domain[dom_idx].otmann_lp_power = \
    options.otmann_lp_power
  kernel_params_for_each_domain[dom_idx].otmann_emd_power = \
    options.otmann_emd_power
  kernel_params_for_each_domain[dom_idx].otmann_non_assignment_penalty = \
    options.otmann_non_assignment_penalty
  # Add the distance computer
  kernel_params_for_each_domain[dom_idx].otmann_distance_computer = \
    dist_computers[dom_idx]


# 4.2 Utilities to build the GP --------------------------------------------------------
def _build_kernel_for_domain(domain, dom_prefix, kernel_scale, gp_cts_hps, gp_dscr_hps,
  other_gp_params, options, kernel_ordering, kernel_params_for_each_domain):
  """ Builds the kernel for the domain. """
  kernel_list = []
  # Iterate through each domain and build the corresponding kernel
  for dom_idx, dom, kernel_type in \
    zip(range(domain.num_domains), domain.list_of_domains,
        kernel_ordering):
    dom_type = dom.get_type().lower()
    if kernel_type == '' or kernel_type is None:
      # If none is specified, use the one given in options
      kernel_type = _get_kernel_type_from_options(dom_type, 'dom', options)
    if kernel_type == 'default':
      kernel_type = get_default_kernel_type(dom.get_type())
    if dom_type in ['euclidean', 'integral', 'prod_discrete_numeric',
                    'discrete_euclidean']:
      curr_kernel_hyperparams = _prep_kernel_hyperparams_for_euc_int_kernels(
                                  kernel_type, dom, dom_prefix, options)
      use_same_bw, _, esp_kernel_type, _ = \
        _get_euc_int_options(dom_type, 'dom', options)
      if hasattr(other_gp_params, 'add_gp_groupings') and \
        other_gp_params.add_gp_groupings is not None:
        add_gp_groupings = other_gp_params.add_gp_groupings[dom_idx]
      else:
        add_gp_groupings = None
      curr_kernel, gp_cts_hps, gp_dscr_hps = \
        get_euclidean_integral_gp_kernel_with_scale(kernel_type, 1.0, \
          curr_kernel_hyperparams, gp_cts_hps, gp_dscr_hps, use_same_bw,
          add_gp_groupings, esp_kernel_type)
    elif dom_type == 'prod_discrete':
      curr_kernel_hyperparams = _prep_kernel_hyperparams_for_discrete_kernels(
                                     kernel_type, dom, dom_prefix, options)
      curr_kernel, gp_cts_hps, gp_dscr_hps = \
        get_discrete_kernel(kernel_type, curr_kernel_hyperparams, gp_cts_hps,
                            gp_dscr_hps)
    elif dom_type == 'neural_network':
      curr_kernel_hyperparams = _prep_kernel_hyperparams_for_nn_kernels(
                                  kernel_type, dom,
                                  kernel_params_for_each_domain[dom_idx])
      curr_kernel, gp_cts_hps, gp_dscr_hps = \
        get_neural_network_kernel(kernel_type, curr_kernel_hyperparams, gp_cts_hps,
                                  gp_dscr_hps)
    else:
      raise NotImplementedError(('Not implemented _child_build_gp for dom_type ' +
                                 '%s yet.')%(dom_type))
    kernel_list.append(curr_kernel)
  return CartesianProductKernel(kernel_scale, kernel_list), gp_cts_hps, gp_dscr_hps


def _prep_kernel_hyperparams_for_euc_int_kernels(kernel_type, dom, dom_prefix, options):
  """ Prepares the kernel hyperparams. """
  # This function retrieves a particular value from the options
  def _get_option_val_from_args(dom_type_str, attr_name):
    """ Internal function to get the options. """
    attr_name_in_options = dom_prefix + '_' + dom_type_str + '_' + attr_name
    if hasattr(options, attr_name_in_options):
      return getattr(options, attr_name_in_options)
    else:
      return None
  # This function retrieves several relevant values in a namepace.
  def _get_options_for_domain(dom_type_str):
    """ Internal function to return options for the domain. """
    default_matern_nu = _DFLT_DOMAIN_MATERN_NU if dom_prefix == 'dom' else \
                        _DFLT_FIDEL_MATERN_NU
    # set them in namespace
    matern_nu_in_options = _get_option_val_from_args(dom_type_str, 'matern_nu')
    matern_nu = default_matern_nu if matern_nu_in_options == 'default' else \
                  matern_nu_in_options
    esp_matern_nu_in_options = _get_option_val_from_args(dom_type_str, 'esp_matern_nu')
    esp_matern_nu = default_matern_nu if esp_matern_nu_in_options == 'default' else \
                  esp_matern_nu_in_options
    euc_int_options = Namespace(matern_nu=matern_nu,
                        esp_matern_nu=esp_matern_nu,
                        poly_order=_get_option_val_from_args(dom_type_str, 'poly_order'),
                        esp_order=_get_option_val_from_args(dom_type_str, 'esp_order'),
                        )
    return euc_int_options

  # Call above functions with relevant arguments.
  if dom.get_type() in ['euclidean', 'discrete_euclidean']:
    euc_int_options = _get_options_for_domain('euc')
  elif dom.get_type() == 'integral':
    euc_int_options = _get_options_for_domain('int')
  elif dom.get_type() == 'prod_discrete_numeric':
    euc_int_options = _get_options_for_domain('disc_num')
  else:
    raise ValueError('kernel_type should be euclidean, integral or ' +
                     'prod_discrete_numeric in this function.')
  return prep_euclidean_integral_kernel_hyperparams(kernel_type, euc_int_options,
                                                    dom.get_dim())

def _prep_kernel_hyperparams_for_discrete_kernels(kernel_type, dom, dom_prefix, options):
  """ Prepares the kernel hyperparams. """
  hamming_wt_attr_name = dom_prefix + '_' + 'disc_hamming_use_same_weight'
  ret = {'disc_hamming_use_same_weight': getattr(options, hamming_wt_attr_name),
         'dim': dom.get_dim(),
         'kernel_type': kernel_type,
        }
  return ret

def _prep_kernel_hyperparams_for_nn_kernels(kernel_type, dom, kernel_params_for_dom):
  """ Prepares the kernel hyper-parameters. """
  ret = vars(kernel_params_for_dom)
  ret['nn_type'] = dom.nn_type
  ret['kernel_type'] = kernel_type
  return ret

def get_discrete_kernel(kernel_type, kernel_hyperparams, gp_cts_hps, gp_dscr_hps):
  """ Returns the kernel for discrete hyper-parameters. """
  dim = kernel_hyperparams['dim']
  if kernel_type == 'hamming':
    if dim == 1 or kernel_hyperparams['disc_hamming_use_same_weight']:
      dim_wts = np.ones((dim,))/float(dim)
    elif dim == 2:
      dim_1_wt = gp_cts_hps[0]
      dim_wts = np.array([dim_1_wt, 1 - dim_1_wt])
      gp_cts_hps = gp_cts_hps[1:]
    else:
      dim_wts_unnorm = np.array(gp_cts_hps[0:dim])
      dim_wts_sum = dim_wts_unnorm.sum()
      if dim_wts_sum > 0:
        dim_wts = dim_wts_unnorm / dim_wts_sum
      else:
        dim_wts = np.ones((dim,))/float(dim)
      gp_cts_hps = gp_cts_hps[dim:]
    kern = HammingKernel(dim_wts)
  else:
    raise ValueError('Unknown kernel_type "%s" for discrete spaces.'%(kernel_type))
  return kern, gp_cts_hps, gp_dscr_hps


def get_neural_network_kernel(kernel_type, kernel_hyperparams, gp_cts_hps,
                              gp_dscr_hps):
  """ Returns the kernel for the Neural Network. """
  # pylint: disable=too-many-branches
  if kernel_type == 'otmann':
    # 1. Relative weights
    if kernel_hyperparams['otmann_kernel_type'] == 'lpemd_sum':
      lp_emd_tradeoff = gp_cts_hps[0]
      lp_emd_alphas = [lp_emd_tradeoff, 1 - lp_emd_tradeoff]
      gp_cts_hps = gp_cts_hps[1:]
    # 2. Betas
    if kernel_hyperparams['otmann_dist_type'] in ['lp', 'emd']:
      betas = np.exp(gp_cts_hps[:kernel_hyperparams['otmann_num_mislabel_struct_coeffs']])
      gp_cts_hps = gp_cts_hps[kernel_hyperparams['otmann_num_mislabel_struct_coeffs']:]
    elif kernel_hyperparams['otmann_dist_type'] == 'lp-emd':
      betas = np.exp(gp_cts_hps[:
                2 * kernel_hyperparams['otmann_num_mislabel_struct_coeffs']])
      gp_cts_hps = gp_cts_hps[
                     2 * kernel_hyperparams['otmann_num_mislabel_struct_coeffs']:]
    # 3 & 4. Mislabel/struct coeffs
    if kernel_hyperparams['otmann_to_tune_mislabel_struct_coeffs']:
      mislabel_coeffs = [gp_cts_hps[0]]
      struct_coeffs = [gp_cts_hps[1]]
      gp_cts_hps = gp_cts_hps[2:]
    else:
      mislabel_coeffs = kernel_hyperparams['otmann_mislabel_coeffs']
      struct_coeffs = kernel_hyperparams['otmann_struct_coeffs']
    # Determine the powers
    if kernel_hyperparams['otmann_dist_type'] == 'lp':
      powers = [kernel_hyperparams['otmann_lp_power']] * \
                kernel_hyperparams['otmann_num_mislabel_struct_coeffs']
    elif kernel_hyperparams['otmann_dist_type'] == 'emd':
      powers = [kernel_hyperparams['otmann_emd_power']] * \
                kernel_hyperparams['otmann_num_mislabel_struct_coeffs']
    elif kernel_hyperparams['otmann_dist_type'] == 'lp-emd':
      powers = [kernel_hyperparams['otmann_lp_power'],
                kernel_hyperparams['otmann_emd_power']] * \
                kernel_hyperparams['otmann_num_mislabel_struct_coeffs']
    else:
      raise ValueError('Unknown distance type: %s'%(
                        kernel_hyperparams['otmann_dist_type']))
    # Check if there is a distance computer
    if 'otmann_distance_computer' in kernel_hyperparams and \
      kernel_hyperparams['otmann_distance_computer'] is not None:
      tp_comp = kernel_hyperparams['otmann_distance_computer']
    else:
      from ..nn.otmann import get_otmann_distance_computer_from_args
      tp_comp = get_otmann_distance_computer_from_args(kernel_hyperparams['nn_type'],
                                 kernel_hyperparams['otmann_non_assignment_penalty'],
                                 mislabel_coeffs, struct_coeffs,
                                 kernel_hyperparams['otmann_dist_type'])
    # Now create the kernel
    from ..nn.otmann import DistProdNNKernel, DistSumNNKernel
    if kernel_hyperparams['otmann_kernel_type'] in ['lpemd_prod', 'lp', 'sum']:
      kern = DistProdNNKernel(tp_comp, betas, 1.0, powers)
    elif kernel_hyperparams['otmann_kernel_type'] in ['lpemd_sum']:
      kern = DistSumNNKernel(tp_comp, lp_emd_alphas, betas, powers)
    else:
      raise ValueError('Unknown otmann_kernel_type: %s.'%(
                        kernel_hyperparams['otmann_kernel_type']))
  else:
    raise ValueError('Unknown kernel_type %s.'%(kernel_type))
  # return
  return kern, gp_cts_hps, gp_dscr_hps

