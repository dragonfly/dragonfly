"""
  A module for fitting a GP and tuning its kernel.
  -- kandasamy@cs.cmu.edu
  -- kvysyara@andrew.cmu.edu
"""

from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements

from argparse import Namespace
from itertools import product as itertools_product
import sys
import numpy as np
# Local imports
from ..distributions.model import Model
from ..distributions import continuous
from ..distributions import discrete
from ..utils.general_utils import stable_cholesky, draw_gaussian_samples, \
     project_symmetric_to_psd_cone, solve_lower_triangular, solve_upper_triangular
from ..utils.oper_utils import direct_ft_maximise, pdoo_maximise, \
                             random_maximise, random_sample_cts_dscr
from ..utils.option_handler import get_option_specs, load_options
from ..utils.reporters import get_reporter


# These are mandatory requirements. Every GP implementation should probably use them.
mandatory_gp_args = [ \
  get_option_specs('hp_tune_criterion', False, 'ml',
                   'Which criterion to use when tuning hyper-parameters. Other ' +
                   'options are post_sampling and post_mean.'),
  get_option_specs('hp_tune_probs', False, 'uniform', \
    'With what probability should we choose each strategy given in hp_tune_criterion.' + \
    'If "uniform" we we will use uniform probabilities and if "adaptive" we will use ' + \
    'adaptive probabilities which weight acquisitions according to how well they do.'),
  get_option_specs('ml_hp_tune_opt', False, 'default',
                   'Which optimiser to use when maximising the tuning criterion.'),
  get_option_specs('hp_tune_max_evals', False, -1,
                   'How many evaluations to use when maximising the tuning criterion.'),
  get_option_specs('handle_non_psd_kernels', False, 'guaranteed_psd',
                   'How to handle kernels that are non-psd.'),
  # The mean and noise variance of the GP
  get_option_specs('mean_func', False, None,
                   ('The mean function. If not None, will use this instead of the' +
                    'other options below')),
  get_option_specs('mean_func_type', False, 'tune',
                   ('Specify the type of mean function. Should be mean, median, const ',
                    'zero, or tune. If const, specifcy value in mean-func-const.')),
  get_option_specs('mean_func_const', False, 0.0,
                   'The constant value to use if mean_func_type is const.'),
  get_option_specs('noise_var_type', False, 'tune', \
    ('Specify how to obtain the noise variance. Should be tune, label or value. ' \
     'Specify appropriate value in noise_var_label or noise_var_value')),
  get_option_specs('noise_var_label', False, 0.05,
                   'The fraction of label variance to use as noise variance.'),
  get_option_specs('noise_var_value', False, 0.1,
                   'The (absolute) value to use as noise variance.'),
  get_option_specs('post_hp_tune_method', False, 'slice',
                   'Which sampling to use when maximising the tuning criterion. Other ' +
                   'option is nuts.'),
  get_option_specs('post_hp_tune_burn', False, -1,
                   'How many initial samples to ignore during sampling.'),
  get_option_specs('post_hp_tune_offset', False, 25,
                   'How many samples to ignore between samples.'), \
  ]


def _check_feature_label_lengths_and_format(X, Y):
  """ Checks if the length of X and Y are the same. """
  if len(X) != len(Y):
    raise ValueError('Length of X (' + len(X) + ') and Y (' + \
      len(Y) + ') do not match.')

def _get_default_ml_hp_tune_opt(num_hps):
  """ Return default optimisation method for maximum likelihood. """
  if num_hps > 60:
    return 'pdoo'
  else:
    return 'direct'


# Main GP class -------------------------------------------------------
class GP(object):
  """ Base class for Gaussian processes. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, X, Y, kernel, mean_func, noise_var, build_posterior=True,
               reporter=None, handle_non_psd_kernels='guaranteed_psd'):
    """ Constructor. """
    super(GP, self).__init__()
    _check_feature_label_lengths_and_format(X, Y)
    self.set_data(X, Y, build_posterior=False)
    self.kernel = kernel
    self.mean_func = mean_func
    self.noise_var = noise_var
    self.reporter = get_reporter(reporter)
    self.handle_non_psd_kernels = handle_non_psd_kernels
    # Some derived attribues.
    self.num_tr_data = len(self.Y)
    # Initialise other attributes we will need.
    self.L = None
    self.alpha = None
    self.K_trtr_wo_noise = None
    self._set_up()
    # Build posterior if necessary
    if build_posterior:
      self.build_posterior()

  def _set_up(self):
    """ Additional set up. """
    # Test that if the kernel is not guaranteed to be PSD, then have have a way to
    # handle it.
    if not self.kernel.is_guaranteed_psd():
      assert self.handle_non_psd_kernels in ['project_first',
                                             'try_before_project']

  def _write_message(self, msg):
    """ Writes a message via the reporter or the std out. """
    if self.reporter:
      self.reporter.write(msg)
    else:
      sys.stdout.write(msg)

  def set_data(self, X, Y, build_posterior=True):
    """ Sets the data to X and Y. """
    self.X = list(X)
    self.Y = list(Y)
    self.num_tr_data = len(self.Y)
    if build_posterior:
      self.build_posterior()

  def add_data_single(self, x_new, y_new, *args, **kwargs):
    """ Adds a single data point. """
    self.add_data_multiple([x_new], [y_new], *args, **kwargs)

  def add_data_multiple(self, X_new, Y_new, build_posterior=True):
    """ Adds new data to the GP. If rebuild is true it rebuilds the posterior. """
    _check_feature_label_lengths_and_format(X_new, Y_new)
    self.X.extend(X_new)
    self.Y.extend(Y_new)
    self.num_tr_data = len(self.Y)
    if build_posterior:
      self.build_posterior()

  # Methods for computing the posterior ----------------------------------------------
  def _get_training_kernel_matrix(self):
    """ Returns the training kernel matrix. Writing this as a separate method in case,
        the kernel computation can be done efficiently for a child class.
    """
    return self.kernel(self.X, self.X)

  def build_posterior(self):
    """ Builds the posterior GP by computing the mean and covariance. """
    K_trtr_wo_noise = self._get_training_kernel_matrix()
    self.K_trtr_wo_noise = K_trtr_wo_noise
    self.L = _get_cholesky_decomp(K_trtr_wo_noise, self.noise_var,
                                  self.handle_non_psd_kernels)
    Y_centred = self.Y - self.mean_func(self.X)
    self.alpha = solve_upper_triangular(self.L.T,
                                        solve_lower_triangular(self.L, Y_centred))

  def eval(self, X_test, uncert_form='none'):
    """ Evaluates the GP on X_test. If uncert_form is
          covar: returns the entire covariance on X_test (nxn matrix)
          std: returns the standard deviations on the test set (n vector)
          none: returns nothing (default).
    """
    # Compute the posterior mean.
    test_mean = self.mean_func(X_test)
    K_tetr = self.kernel(X_test, self.X)
    pred_mean = test_mean + K_tetr.dot(self.alpha)
    # Compute the posterior variance or standard deviation as required.
    if uncert_form == 'none':
      uncert = None
    else:
      K_tete = self.kernel(X_test, X_test)
      V = solve_lower_triangular(self.L, K_tetr.T)
      post_covar = K_tete - V.T.dot(V)
      post_covar = get_post_covar_from_raw_covar(post_covar, self.noise_var,
                                                 self.kernel.is_guaranteed_psd())
      if uncert_form == 'covar':
        uncert = post_covar
      elif uncert_form == 'std':
        uncert = np.sqrt(np.diag(post_covar))
      else:
        raise ValueError('uncert_form should be none, covar or std.')
    return pred_mean, uncert

  def eval_with_hallucinated_observations(self, X_test, X_halluc, uncert_form='none'):
    """ Evaluates the GP with additional hallucinated observations in the
        kernel matrix. """
    pred_mean, _ = self.eval(X_test, uncert_form='none') # Just compute the means.
    if uncert_form == 'none':
      uncert = None
    else:
      # Computed the augmented kernel matrix and its cholesky decomposition.
      X_aug = self.X + X_halluc
      K_haha = self.kernel(X_halluc, X_halluc) # kernel for hallucinated data
      K_trha = self.kernel(self.X, X_halluc) # kernel with training and hallucinated data
      aug_K_trtr_wo_noise = np.vstack((np.hstack((self.K_trtr_wo_noise, K_trha)),
                                       np.hstack((K_trha.T, K_haha))))
      aug_L = _get_cholesky_decomp(aug_K_trtr_wo_noise, self.noise_var,
                                   self.handle_non_psd_kernels)
      # Augmented kernel matrices for the test data
      aug_K_tete = self.kernel(X_test, X_test)
      aug_K_tetr = self.kernel(X_test, X_aug)
      aug_V = solve_lower_triangular(aug_L, aug_K_tetr.T)
      aug_post_covar = aug_K_tete - aug_V.T.dot(aug_V)
      aug_post_covar = get_post_covar_from_raw_covar(aug_post_covar, self.noise_var,
                                                     self.kernel.is_guaranteed_psd())
      if uncert_form == 'covar':
        uncert = aug_post_covar
      elif uncert_form == 'std':
        uncert = np.sqrt(np.diag(aug_post_covar))
      else:
        raise ValueError('uncert_form should be none, covar or std.')
    return (pred_mean, uncert)

  def compute_log_marginal_likelihood(self):
    """ Computes the log marginal likelihood. """
    Y_centred = self.Y - self.mean_func(self.X)
    ret = -0.5 * Y_centred.T.dot(self.alpha) - (np.log(np.diag(self.L))).sum() \
          - 0.5 * self.num_tr_data * np.log(2*np.pi)
    return ret

  def compute_grad_log_marginal_likelihood(self, param, *args):
    """ Computes the gradient of log marginal likelihood. """
    alpha = np.expand_dims(self.alpha, axis=0)
    if param == 'noise_var':
      grad_m = self.noise_var*np.identity(len(self.X))
    elif param == 'noise_mean':
      return np.asscalar(np.matmul(alpha, np.ones((len(self.Y), 1))))
    else:
      grad_m = self.kernel.gradient(param, self.X, self.X, *args)
    grad_m = np.matmul(alpha.T, np.matmul(alpha, grad_m)) - \
             solve_upper_triangular(self.L.T, solve_lower_triangular(self.L, grad_m))
    return 0.5 * np.trace(grad_m)

  def __str__(self):
    """ Returns a string representation of the GP. """
    return '%s, noise-var=%0.3f (n=%d)'%(self._child_str(), self.noise_var, len(self.Y))

  def _child_str(self):
    """ String representation for child GP. """
    raise NotImplementedError('Implement in child class. !')

  def draw_samples(self, num_samples, X_test=None, mean_vals=None, covar=None):
    """ Draws num_samples samples at returns their values at X_test. """
    if X_test is not None:
      mean_vals, covar = self.eval(X_test, 'covar')
    return draw_gaussian_samples(num_samples, mean_vals, covar)

  def draw_samples_with_hallucinated_observations(self, num_samples, X_test,
                                                  X_halluc):
    """ Draws samples with hallucinated observations. """
    mean_vals, aug_covar = self.eval_with_hallucinated_observations(X_test, \
                                             X_halluc, uncert_form='covar')
    return draw_gaussian_samples(num_samples, mean_vals, aug_covar)

  def visualise(self, file_name=None, boundary=None, true_func=None,
                num_samples=20, conf_width=3):
    """ Visualises the GP. """
    # pylint: disable=unused-variable
    # pylint: disable=too-many-locals
    if hasattr(self.kernel, 'dim') and self.kernel.dim != 1:
      self._write_message('Cannot visualise only in 1 dimension.\n')
    else:
      import matplotlib.pyplot as plt
      fig = plt.figure()
      N = 400
      leg_handles = []
      leg_labels = []
      if not boundary:
        boundary = [min(self.X), max(self.X)]
      grid = np.linspace(boundary[0], boundary[1], N).reshape((N, 1))
      (pred_vals, pred_stds) = self.eval(grid, 'std')
      # Shade a high confidence region
      conf_band_up = pred_vals + conf_width * pred_stds
      conf_band_down = pred_vals - conf_width * pred_stds
      leg_conf = plt.fill_between(grid.ravel(), conf_band_up, conf_band_down,
                                  color=[0.9, 0.9, 0.9])
      # Plot the samples
      gp_samples = self.draw_samples(num_samples, grid)
      plt.plot(grid, gp_samples.T, '--', linewidth=0.5)
      # plot the true function if available.
      if true_func:
        leg_true = plt.plot(grid, true_func(grid), 'b--', linewidth=3,
                            label='true function')
        leg_handles.append(leg_true)
      # Plot the posterior mean
      leg_post_mean = plt.plot(grid, pred_vals, 'k-', linewidth=4,
                               label='post mean')
      # Finally plot the training data.
      leg_data = plt.plot(self.X, self.Y, 'kx', mew=4, markersize=10,
                          label='data')
      # Finally either plot or show the figure
      if file_name is None:
        plt.show()
      else:
        fig.savefig(file_name)


class GPFitter(object):
  """ Class for fitting Gaussian processes. """
  # pylint: disable=attribute-defined-outside-init
  # pylint: disable=no-value-for-parameter

  def __init__(self, X, Y, options, reporter='default'):
    """ Constructor. """
    super(GPFitter, self).__init__()
    assert len(X) == len(Y)
    self.reporter = get_reporter(reporter)
    options = load_options(mandatory_gp_args, partial_options=options)
    self.options = options
    self.X = X
    self.Y = Y
    self.num_data = len(X)
    self._set_up()

  def _set_up(self):
    """ Sets up a bunch of ancillary parameters. """
    # Set place-holders for hp_bounds or hp_vals depending on whether we are using ml or
    # post_sampling/mean
    # -----------------------------------------------------------------------------
    self.cts_hp_bounds = [] # The bounds for each hyper parameter should be a
                            # num_cts_hps x 2 array where the 1st/2nd columns are the
                            # lowe/upper bounds.
    self.dscr_hp_vals = []  # A list of lists where each list contains a list of values
                            # for each discrete hyper-parameter.
    self.param_order = []
    # Set up for common things - including mean and noise variance hyper-parameters
    # -----------------------------------------------------------------------------
    epsilon = 0.0001
    self.Y_var = np.array(self.Y).std() ** 2 + epsilon if len(self.Y) > 0 else \
                 epsilon
    self._set_up_mean_and_noise_variance_bounds()
    # Set up hyper-parameters for the child.
    # -----------------------------------------------------------------------------
    self._child_set_up()
    self._hp_tune_method_set_up()
    self.cts_hp_bounds = np.array(self.cts_hp_bounds)
    # Some post child set up
    if 'ml' in self.methods_to_use:
      # The number of hyper parameters
      self.num_hps = len(self.cts_hp_bounds) + len(self.dscr_hp_vals)
      self._set_up_ml_hp_tune()
    if 'post_sampling' in self.methods_to_use:
      self._set_up_post_sampling_hp_tune()
      self.num_hps = len(self.hp_priors) # The number of hyper parameters
    if 'post_mean' in self.methods_to_use:
      self._set_up_post_mean_hp_tune()
      self.num_hps = len(self.hp_priors) # The number of hyper parameters

  def _hp_tune_method_set_up(self):
    """ Sets up probabilities for hp tune methods. """
    self.methods_to_use = [elem.lower() for elem in
                           self.options.hp_tune_criterion.split('-')]
    for method in self.methods_to_use:
      if method not in ['ml', 'post_sampling', 'post_mean']:
        raise ValueError('hp_tune_criterion should be ml or post_sampling.')
    self.methods_to_use_counter = {key: 0 for key in self.methods_to_use}
    if self.options.hp_tune_probs == 'uniform':
      self.hp_tune_probs = np.ones(len(self.methods_to_use)) /\
                           float(len(self.methods_to_use))
    elif self.options.hp_tune_probs == 'adaptive':
      self.hp_tune_uniform_sampling_prob = 0.05
      self.hp_tune_sampling_weights = {key: 1.0 for key in self.methods_to_use}
      self.hp_tune_probs = self._get_adaptive_hp_tune_probs()
    else:
      self.hp_tune_probs = np.array([float(x) for x in
                                     self.options.hp_tune_probs.split('-')])
      if len(self.hp_tune_probs) != len(self.methods_to_use):
        self.hp_tune_probs = np.ones(len(self.methods_to_use)) /\
                             float(len(self.methods_to_use))
    self.hp_tune_probs = self.hp_tune_probs / self.hp_tune_probs.sum()

  def _get_adaptive_hp_tune_probs(self):
    """ Computes the adaptive hyper parameter tuning method probs. """
    num_methods = len(self.methods_to_use)
    uniform_sampling_probs = self.hp_tune_uniform_sampling_prob * \
                             np.ones((num_methods,)) / num_methods
    hp_tune_succ_counter = np.array([self.hp_tune_sampling_weights[key] for
                                     key in self.methods_to_use])
    hp_tune_use_counter = np.array([self.methods_to_use_counter[key] for
                                    key in self.methods_to_use])
    hp_tune_weights = hp_tune_succ_counter / np.sqrt(1 + hp_tune_use_counter)
    hp_tune_norm_weights = hp_tune_weights / hp_tune_weights.sum()
    adaptive_sampling_probs = (1 - self.hp_tune_uniform_sampling_prob) * \
                              hp_tune_norm_weights
    ret = uniform_sampling_probs + adaptive_sampling_probs
    return ret / ret.sum()

  def _set_up_mean_and_noise_variance_bounds(self):
    """ Sets up bounds for the mean value and the noise. """
    # 1. The prior mean for the GP.
    if not (hasattr(self.options, 'mean_func') and self.options.mean_func is not None) \
      and self.options.mean_func_type == 'tune':
      Y_std = np.sqrt(self.Y_var)
      if len(self.Y) > 0:
        Y_median = np.median(self.Y)
        Y_half_range = 0.5 * (max(self.Y) - min(self.Y)) if len(self.Y) > 0 else 1.0
      else:
        Y_median = 0.0
        Y_half_range = 1.0
      Y_width = 0.5 * (Y_half_range + Y_std)
      self.mean_func_bounds = [Y_median - 3 * Y_width, Y_median + 3 * Y_width]
      self.cts_hp_bounds.append(self.mean_func_bounds)
      self.param_order.append(["noise_mean", "cts"])
    # 2. The noise variance
    if self.options.noise_var_type == 'tune':
      self.noise_var_log_bounds = [np.log(0.005 * self.Y_var), np.log(0.2 * self.Y_var)]
      self.cts_hp_bounds.append(self.noise_var_log_bounds)
      self.param_order.append(["noise_var", "cts"])

  def _child_set_up(self):
    """ Here you should set up parameters for the child, such as the bounds for the
        optimiser etc. """
    raise NotImplementedError('Implement _child_set_up in a child method.')

  def _set_up_ml_hp_tune(self):
    """ Sets up optimiser for ml hp tune. """
    # pylint: disable=too-many-branches
    # define the following internal functions to abstract things out more.
    def _direct_wrap(*args):
      """ A wrapper so as to only return the optimal point. """
      opt_val, opt_pt, _ = direct_ft_maximise(*args)
      return opt_val, opt_pt, None
    def _pdoo_wrap(*args):
      """ A wrapper so as to only return the optimal point. """
      opt_val, opt_pt, _ = pdoo_maximise(*args)
      return opt_val, opt_pt, None
    def _rand_wrap(*args):
      """ A wrapper so as to only return the optimal point. """
      opt_val, opt_pt, _ = random_maximise(*args, vectorised=False)
      return opt_val, opt_pt, None
    def _rand_exp_sampling_wrap(*args):
      """ A wrapper so as to only return the optimal point. """
      sample_cts_hps, sample_dscr_hps, lml_vals = \
        random_sample_cts_dscr(*args, vectorised=False)
      sample_probs = np.exp(lml_vals - max(lml_vals))
      sample_probs = sample_probs / sample_probs.sum()
      return sample_cts_hps, sample_dscr_hps, sample_probs
    # Set some parameters
    if self.options.ml_hp_tune_opt == 'default':
      self.ml_hp_tune_opt_method = _get_default_ml_hp_tune_opt(self.num_hps)
    else:
      self.ml_hp_tune_opt_method = self.options.ml_hp_tune_opt
    if (hasattr(self.options, 'hp_tune_max_evals') and
        self.options.hp_tune_max_evals is not None and
        self.options.hp_tune_max_evals > 0):
      self.hp_tune_max_evals = self.options.hp_tune_max_evals
    else:
      if self.ml_hp_tune_opt_method in ['direct', 'pdoo']:
        self.hp_tune_max_evals = min(1e4, max(500, self.num_hps * 50))
      elif self.ml_hp_tune_opt_method == 'rand':
        self.hp_tune_max_evals = min(1e4, max(500, self.num_hps * 200))
      elif self.ml_hp_tune_opt_method == 'rand_exp_sampling':
        self.hp_tune_max_evals = min(1e5, max(500, self.num_hps * 400))
    # Now set up the function that will be used for optimising
    if self.ml_hp_tune_opt_method == 'direct':
      self.cts_hp_optimise = lambda obj, max_evals: _direct_wrap(obj, self.cts_hp_bounds,
                                                                 max_evals)
    elif self.ml_hp_tune_opt_method == 'pdoo':
      self.cts_hp_optimise = lambda obj, max_evals: _pdoo_wrap(obj, self.cts_hp_bounds,
                                                               max_evals)
    elif self.ml_hp_tune_opt_method == 'rand':
      self.cts_hp_optimise = lambda obj, max_evals: _rand_wrap(obj, self.cts_hp_bounds,
                                                               max_evals)
    elif self.ml_hp_tune_opt_method == 'rand_exp_sampling':
      self.hp_sampler = lambda obj, max_evals: _rand_exp_sampling_wrap(obj, \
                          self.cts_hp_bounds, self.dscr_hp_vals, max_evals)

  def _set_up_post_sampling_hp_tune(self):
    """ Sets up posterior sampling for tuning the parameters of the GP. """
    if self.options.post_hp_tune_method == 'slice':
      self.hp_sampler_cts = lambda model, init_sample, num_samples, burn: \
                               model.draw_samples('slice', num_samples, init_sample, burn)
    elif self.options.post_hp_tune_method == 'nuts':
      self.hp_sampler_cts = lambda model, init_sample, num_samples, burn: \
                                model.draw_samples('nuts', num_samples, init_sample, burn)

    self.hp_sampler_dscr = lambda model, init_sample, num_samples: \
                                model.draw_samples('metropolis', num_samples, init_sample)

    # Priors for all the hyper parameters
    self.hp_priors = []
    for _, bounds in enumerate(self.cts_hp_bounds):
      self.hp_priors.append(continuous.ContinuousUniform(bounds[0], bounds[-1]))

    for _, vals in enumerate(self.dscr_hp_vals):
      self.hp_priors.append(discrete.Categorical(vals, np.repeat(1.0/len(vals),
                                                                 len(vals))))

  def _set_up_post_mean_hp_tune(self):
    """ Sets up using the posterior mean for tuning the parameters of the GP. """
    raise NotImplementedError('Not implemented post_mean yet.')

  def build_gp(self, gp_cts_hps, gp_dscr_hps, other_gp_params=None, *args, **kwargs):
    """ A method which builds a GP from the given gp_hyperparameters. It calls
        _child_build_gp after running some checks. """
    # pylint: disable=too-many-branches
    # Check the length of the hyper-parameters
    if self.num_hps != len(gp_cts_hps) + len(gp_dscr_hps):
      raise ValueError('gp_hyperparams should be of length %d. Given length: %d.'%(
          self.num_hps, len(gp_cts_hps) + len(gp_dscr_hps)))
    # Compute the mean and the noise variance first
    # Mean function -------------------------------------
    if hasattr(self.options, 'mean_func') and self.options.mean_func is not None:
      mean_func = self.options.mean_func
    else:
      if self.options.mean_func_type == 'mean':
        mean_func_const_value = np.mean(self.Y)
      elif self.options.mean_func_type == 'median':
        mean_func_const_value = np.median(self.Y)
      elif self.options.mean_func_type == 'upper_bound':
        mean_func_const_value = np.mean(self.Y) + 3 * np.std(self.Y)
      elif self.options.mean_func_type == 'const':
        mean_func_const_value = self.options.mean_func_const
      elif self.options.mean_func_type == 'tune':
        mean_func_const_value = np.asscalar(gp_cts_hps[0])
        gp_cts_hps = gp_cts_hps[1:]
      else:
        mean_func_const_value = 0
      def _get_mean_func(_mean_func_const_value):
        """ Returns the mean function from the constant value. """
        return lambda x: np.array([_mean_func_const_value] * len(x))
      mean_func = _get_mean_func(mean_func_const_value)
    # Noise variance ------------------------------------
    if self.options.noise_var_type == 'tune':
      noise_var = np.exp(gp_cts_hps[0])
      gp_cts_hps = gp_cts_hps[1:]
    elif self.options.noise_var_type == 'label':
      noise_var = self.options.noise_var_label * (self.Y.std() ** 2)
    else:
      noise_var = self.options.noise_var_value
    ret_gp, ret_cts_hps, ret_dscr_hps = self._child_build_gp(mean_func, noise_var, \
       gp_cts_hps, gp_dscr_hps, other_gp_params=other_gp_params, *args, **kwargs)
    assert len(ret_cts_hps) == 0
    assert len(ret_dscr_hps) == 0
    return ret_gp

  def _child_build_gp(self, mean_func, noise_var, gp_cts_hps, gp_dscr_hps,
                      other_gp_params=None, *args, **kwargs):
    """ A method which builds the child GP from the given gp_hyperparameters. Should be
        implemented in a child method. """
    raise NotImplementedError('Implement _child_build_gp in a child method.')

  def _tuning_objective(self, gp_cts_hps, gp_dscr_hps, other_gp_params=None,
                        *args, **kwargs):
    """ This function computes the tuning objective (such as the marginal likelihood)
        which is to be maximised in fit_gp. """
    built_gp = self.build_gp(gp_cts_hps, gp_dscr_hps, other_gp_params=other_gp_params,
                             *args, **kwargs)
    if 'ml' in self.methods_to_use or 'marginal_likelihood' in self.methods_to_use:
      ret = built_gp.compute_log_marginal_likelihood()
    elif 'cv' in self.methods_to_use or 'cross_validation' in self.methods_to_use:
      raise NotImplementedError('Yet to implement cross validation based hp-tuning.')
    else:
      raise ValueError('hp_tune_criterion should be either ml or cv')
    return ret

  def _tuning_objective_post_sampling(self, gp_cts_hps, gp_dscr_hps, \
       other_gp_params=None, param=None, param_num=None, gradient=False, *args, **kwargs):
    """ This function computes the tuning objective for posterior sampling case. """
    built_gp = self.build_gp(gp_cts_hps, gp_dscr_hps, other_gp_params=other_gp_params,
                             *args, **kwargs)
    if gradient:
      ret = built_gp.compute_grad_log_marginal_likelihood(param, param_num)
    else:
      ret = built_gp.compute_log_marginal_likelihood()
    return ret

  def _optimise_cts_hps_for_given_dscr_hps(self, given_dscr_hps):
    """ Optimises the continuous hyper-parameters for the given discrete hyper-params.
        Can be overridden by a child class if necessary.
    """
    cts_tuning_obj = lambda arg: self._tuning_objective(arg, list(given_dscr_hps))
    opt_cts_val, opt_cts_hps, _ = self.cts_hp_optimise(cts_tuning_obj,
                                                       self.hp_tune_max_evals)
    return opt_cts_val, opt_cts_hps, None

  def _sample_cts_dscr_hps_for_rand_exp_sampling(self):
    """ Samples continous and discrete hyper-parameters for rand_exp_sampling. """
    sample_cts_hps, sample_dscr_hps, sample_probs = \
      self.hp_sampler(self._tuning_objective, self.hp_tune_max_evals)
    sample_other_gp_params = [None] * len(sample_cts_hps)
    return sample_cts_hps, sample_dscr_hps, sample_other_gp_params, sample_probs

  def _sample_cts_dscr_hps_for_post_sampling(self, num_samples):
    """ Samples continous and discrete hyper-parameters for post_sampling. """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    def _logp(x):
      """ Computes log probability of observations at x. """
      # sum of log probability priors
      if self.parameter == 'additive_grp':
        if x < 0:
          return -np.inf
        permut = list(np.random.RandomState(seed=x[0]).permutation(self.add_dim))
        groupings = [permut[i:i+self.group_size]
                     for i in range(0, self.add_dim, self.group_size)]
        self.other_gp_params = Namespace(add_gp_groupings=groupings)
      elif type(self.hp_priors[self.curr_hp]).__name__ == "Categorical":
        self.hps[self.curr_hp] = self.hp_priors[self.curr_hp].get_category(np.asscalar(x))
      else:
        self.hps[self.curr_hp] = x

      lp = 0
      for i, prior in enumerate(self.hp_priors):
        if type(self.hp_priors[i]).__name__ == "Categorical":
          lp += prior.logp(self.hp_priors[i].get_id(self.hps[i]))
        else:
          lp += prior.logp(self.hps[i])
      if not np.isfinite(lp):
        return lp

      lp += self._tuning_objective_post_sampling(self.hps[0:len(self.cts_hp_bounds)], \
                self.hps[len(self.cts_hp_bounds):self.num_hps], self.other_gp_params)
      return lp

    def _pdf(x):
      """ Computes probability of observations at x. """
      val = _logp(x)
      if not np.isfinite(val):
        return val
      return np.exp(val)

    def _grad_logp(x):
      """ Computes gradient of log probability of observations at x. """
      if not np.isfinite(self.hp_priors[self.curr_hp].logp(x)):
        return 0
      self.hps[self.curr_hp] = x
      lp = self.hp_priors[self.curr_hp].grad_logp(self.hps[self.curr_hp])
      lp += self._tuning_objective_post_sampling(self.hps[0:len(self.cts_hp_bounds)], \
                self.hps[len(self.cts_hp_bounds):self.num_hps], self.other_gp_params, \
                self.parameter, self.param_num, True)
      return lp

    offset = self.options.post_hp_tune_offset
    _model = Model(_pdf, _logp, _grad_logp)
    self.hps = np.ones((self.num_hps,))
    best_cts_hps = np.zeros([num_samples, len(self.cts_hp_bounds)])
    best_dscr_hps = np.zeros([num_samples, len(self.dscr_hp_vals)])
    best_other_gp_params = [None] * num_samples
    total_samples = (num_samples - 1)*offset + 1
    cts_hps = np.zeros([total_samples, len(self.cts_hp_bounds)])
    dscr_hps = np.zeros([total_samples, len(self.dscr_hp_vals)])
    _other_gp_params = [None] * total_samples
    self.param_num = -1

    # burn value
    if self.options.post_hp_tune_burn == -1:
      burn = int(np.sqrt(self.num_hps)*100)
      np.clip(burn, 100, 2000)
    else:
      burn = self.options.post_hp_tune_burn

    # Initializing hyper parameters
    for i in range(len(self.cts_hp_bounds)):
      self.hps[i] = self.hp_priors[i].get_mean()
    for i in range(len(self.cts_hp_bounds), self.num_hps):
      self.hps[i] = self.hp_priors[i].draw_samples('random', 1)
      if type(self.hp_priors[i]).__name__ == "Categorical":
        self.hps[i] = self.hp_priors[i].get_category(int(self.hps[i]))

    # Grouping intitialization for additive gp
    if vars(self.options).get('use_additive_gp', False) or \
       vars(self.options).get('domain_use_additive_gp', False):
      if vars(self.options).get('domain_use_additive_gp', False):
        self.add_dim = self.domain_dim
        add_max_group_size = self.domain_add_max_group_size
      else:
        self.add_dim = self.dim
        add_max_group_size = self.add_max_group_size
      permut = list(np.random.permutation(self.add_dim))
      groupings = [permut[i:i+int(self.hps[-1])]
                   for i in range(0, self.add_dim, int(self.hps[-1]))]
      self.other_gp_params = Namespace(add_gp_groupings=groupings)
    else:
      self.other_gp_params = Namespace(add_gp_groupings=None)

    # Randomly selcting a parameter and sampling it using tuning objective
    order = list(range(self.num_hps))
    np.random.shuffle(order)
    for i in order:
      self.curr_hp = i
      self.parameter = self.param_order[i][0]
      if self.parameter == "dim_bandwidths":
        self.param_num = self.param_num + 1
      dscr_i = i - len(self.cts_hp_bounds)
      if self.param_order[i][-1] == "cts":
        cts_hps[:, i] = np.squeeze(self.hp_sampler_cts(_model, self.hps[i],
                                                       total_samples, burn), axis=1)
        self.hps[i] = cts_hps[0, i]
      elif self.parameter != "additive_grp":
        if type(self.hp_priors[i]).__name__ == "Categorical":
          init_sample = self.hp_priors[i].get_id(self.hps[i])
          dscr_hp = np.squeeze(self.hp_sampler_dscr(_model, init_sample, total_samples),
                               axis=1)
          for j, val in enumerate(dscr_hp):
            dscr_hps[j, dscr_i] = self.hp_priors[i].get_category(int(val))
        else:
          dscr_hps[:, dscr_i] = np.squeeze(self.hp_sampler_dscr(_model, \
                                                self.hps[i], total_samples), axis=1)
        self.hps[i] = dscr_hps[0, dscr_i]
      else:
        dscr_hps[:, dscr_i] = np.random.randint(1, add_max_group_size + 1, total_samples)
        seed = int(np.random.randint(self.add_dim))
        for j in range(total_samples):
          self.group_size = int(dscr_hps[j, dscr_i])
          seed = int(np.squeeze(self.hp_sampler_dscr(_model, seed, 1), axis=1))
          permut = list(np.random.RandomState(seed=seed).permutation(self.add_dim))
          groupings = [permut[k:k+self.group_size]
                       for k in range(0, self.add_dim, self.group_size)]
          _other_gp_params[j] = Namespace(add_gp_groupings=groupings)
        self.other_gp_params = _other_gp_params[0]

    for i in range(num_samples):
      best_cts_hps[i, :] = cts_hps[i*offset, :]
      best_dscr_hps[i, :] = dscr_hps[i*offset, :]
      best_other_gp_params[i] = _other_gp_params[i*offset]

    return best_cts_hps, best_dscr_hps, best_other_gp_params

  def get_next_gp(self):
    """ Gets the GP to use in the current iteration. """
    if self.options.hp_tune_probs == 'adaptive':
      self.hp_tune_probs = self._get_adaptive_hp_tune_probs()
    method = np.random.choice(self.methods_to_use, p=self.hp_tune_probs)
    fit_type = self.hp_tune_results[method][0]
    if fit_type in ['fitted_gp', 'post_fitted_gp']:
      gp = self.hp_tune_results[method][1]
    elif fit_type in ['sample_hps_with_probs', 'post_sample_hps_with_probs']:
      next_gp_hps = self.hp_tune_results[method][1].pop(0)
      self.hp_tune_results[method][1].append(next_gp_hps)
      gp = self.build_gp(next_gp_hps[0], next_gp_hps[1], other_gp_params=next_gp_hps[2],
                         build_posterior=False)
    return (fit_type, method, gp)

  def update_hp_tune_method_weight(self, method, weight_to_add=1):
    """ Updates hp_tune method weight in the case of adaptive tuning. """
    if self.options.hp_tune_probs == 'adaptive':
      self.hp_tune_sampling_weights[method] += weight_to_add

  def fit_gp_for_gp_bandit(self, num_samples=1):
    """ Fits a GP according to the tuning criterions. """
    self.hp_tune_results = {}
    for method in self.methods_to_use:
      ret = self.fit_gp(num_samples, method)
      if ret[0] == 'fitted_gp':
        self.hp_tune_results[method] = (ret[0], ret[1])
      elif ret[0] == 'post_fitted_gp':
        self.hp_tune_results[method] = (ret[0], ret[1])
      elif ret[0] == 'sample_hps_with_probs':
        # in this case, ret is a 5-tuple where ret[0] is sample_hps_with_probs, ret[1] is
        # a list of continuous hps, ret[2] is a list of discrete hps, ret[3] is either
        # None or a namespace containing the field add_gp_groupings.
        # See gp.gp_core.fit_gp
        sample_hps = list(zip(ret[1], ret[2], ret[3]))
        sample_probs = ret[-1]
        # Sample the hyper_parameters we will use until the next time we build a model.
        if sum(sample_probs > 0) >= num_samples:
          to_replace = self.options.rand_exp_sampling_replace
        else:
          to_replace = True
        use_hps_idxs = np.random.choice(len(sample_hps),
                                        size=(num_samples,),
                                        replace=to_replace,
                                        p=sample_probs)
        self.hp_tune_results[method] = (ret[0], [sample_hps[idx] for idx in use_hps_idxs])
      elif ret[0] == 'post_sample_hps_with_probs':
        # in this case, ret is a 4-tuple where ret[0] is post_sample_hps_with_probs,
        # ret[1] is a list of continuous hps, ret[2] is a list of discrete hps, ret[3]
        # is either None or a namespace containing the field add_gp_groupings.
        # See gp.gp_core.fit_gp
        self.hp_tune_results[method] = (ret[0], list(zip(ret[1], ret[2], ret[3])))
      else:
        raise ValueError('Unknown option %s for results of fit_gp.'%(ret[0]))

  def fit_gp(self, num_samples=1, hp_tune_criterion=None):
    """ Fits a GP according to the tuning criterion. Returns the best GP along with the
        hyper-parameters. """
    if hp_tune_criterion is None:
      hp_tune_criterion = self.options.hp_tune_criterion
    if hp_tune_criterion == 'ml':
      if self.ml_hp_tune_opt_method in ['direct', 'rand', 'pdoo']:
        best_cts_hps = None
        best_dscr_hps = None
        best_other_params = None
        best_hps_val = -np.inf
        for dscr_hps in itertools_product(*self.dscr_hp_vals):
          opt_cts_val, opt_cts_hps, opt_other_params = \
             self._optimise_cts_hps_for_given_dscr_hps(dscr_hps)
          if opt_cts_val > best_hps_val:
            best_cts_hps = list(opt_cts_hps)
            best_dscr_hps = list(dscr_hps)
            best_other_params = opt_other_params
            best_hps_val = opt_cts_val
        opt_gp = self.build_gp(best_cts_hps, best_dscr_hps,
                               other_gp_params=best_other_params)
        opt_hps = (best_cts_hps, best_dscr_hps)
        return 'fitted_gp', opt_gp, opt_hps
      elif self.ml_hp_tune_opt_method == 'rand_exp_sampling':
        sample_cts_hps, sample_dscr_hps, sample_other_gp_params, sample_probs = \
          self._sample_cts_dscr_hps_for_rand_exp_sampling()
        return ('sample_hps_with_probs', sample_cts_hps, sample_dscr_hps,
                sample_other_gp_params, sample_probs)
    elif hp_tune_criterion == 'post_sampling':
      sample_cts_hps, sample_dscr_hps, sample_other_gp_params = \
        self._sample_cts_dscr_hps_for_post_sampling(num_samples)
      if num_samples == 1:
        opt_gp = self.build_gp(sample_cts_hps[0], sample_dscr_hps[0],
                               other_gp_params=sample_other_gp_params[0])
        opt_hps = (sample_cts_hps, sample_dscr_hps)
        return 'post_fitted_gp', opt_gp, opt_hps
      else:
        return ('post_sample_hps_with_probs', sample_cts_hps, sample_dscr_hps,
                sample_other_gp_params)

      # GPFitter class ends here ---------------------------------------------------------


# Some utilities we will be using above -------------------------------------------------
def _get_cholesky_decomp(K_trtr_wo_noise, noise_var, handle_non_psd_kernels):
  """ Computes cholesky decomposition after checking how to handle non-psd kernels. """
  if handle_non_psd_kernels == 'try_before_project':
    K_trtr_w_noise = K_trtr_wo_noise + noise_var * np.eye(K_trtr_wo_noise.shape[0])
    try:
      # If the cholesky decomposition on the (noise added) matrix succeeds, return!
      L = stable_cholesky(K_trtr_w_noise, add_to_diag_till_psd=False)
      return L
    except np.linalg.linalg.LinAlgError:
      # otherwise, project and return
      return _get_cholesky_decomp(K_trtr_wo_noise, noise_var, 'project_first')
  elif handle_non_psd_kernels == 'project_first':
    # project the Kernel (without noise) to the PSD cone and return
    K_trtr_wo_noise = project_symmetric_to_psd_cone(K_trtr_wo_noise)
    return _get_cholesky_decomp(K_trtr_wo_noise, noise_var, 'guaranteed_psd')
  elif handle_non_psd_kernels == 'guaranteed_psd':
    K_trtr_w_noise = K_trtr_wo_noise + noise_var * np.eye(K_trtr_wo_noise.shape[0])
    return stable_cholesky(K_trtr_w_noise)
  else:
    raise ValueError('Unknown option for handle_non_psd_kernels: %s'%(
        handle_non_psd_kernels))

def get_post_covar_from_raw_covar(raw_post_covar, noise_var, is_guaranteed_psd):
  """ Computes the posterior covariance from the raw_post_covar. This is mostly to
      account for the fact that the kernel may not be psd.
  """
  if is_guaranteed_psd:
    return raw_post_covar
  else:
    epsilon = 0.05 * noise_var
    return project_symmetric_to_psd_cone(raw_post_covar, epsilon=epsilon)

