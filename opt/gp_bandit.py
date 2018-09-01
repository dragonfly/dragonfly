"""
  Harness for GP Bandit Optimisation.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=redefined-builtin

from argparse import Namespace
import numpy as np

# Local imports
# from gp.gp_core import GP
from exd import domains
from exd.exd_core import mf_exd_args
from exd.exd_utils import get_euclidean_initial_qinfos
from gp.euclidean_gp import EuclideanGPFitter, euclidean_gp_args
from gp.euclidean_gp import EuclideanMFGPFitter, euclidean_mf_gp_args
from opt import gpb_acquisitions
from opt.blackbox_optimiser import blackbox_opt_args, BlackboxOptimiser, \
                                   CalledMFOptimiserWithSFCaller
from utils.ancillary_utils import get_list_as_str
from utils.option_handler import get_option_specs, load_options
from utils.reporters import get_reporter

gp_bandit_args = [ \
  get_option_specs('acq', False, 'ei-ucb-ttei', \
    'Which acquisition to use: ts, ucb, ei, ttei, bucb. If using multiple ' + \
    'give them as a hyphen separated list e.g. ucb-ts-ei-ttei'),
  get_option_specs('acq_probs', False, 'adaptive', \
    'With what probability should we choose each strategy given in acq. If "uniform" ' + \
    'we we will use uniform probabilities and if "adaptive" we will use adaptive ' + \
    'probabilities which weight acquisitions according to how well they do.'),
  get_option_specs('acq_opt_method', False, 'direct', \
    'Which optimiser to use when maximising the acquisition function.'),
  get_option_specs('handle_parallel', False, 'halluc', \
    'How to handle parallelisations. Should be halluc or naive.'),
  get_option_specs('acq_opt_max_evals', False, -1, \
    'Number of evaluations when maximising acquisition. If negative uses default value.'),
  # The following are for managing GP hyper-parameters. They override hp_tune_criterion
  # and ml_hp_tune_opt from the GP args.
  get_option_specs('gpb_hp_tune_criterion', False, 'ml-post_sampling',
                   'Which criterion to use when tuning hyper-parameters. Other ' +
                   'options are post_sampling and post_mean.'),
  get_option_specs('gpb_hp_tune_probs', False, '0.3-0.7', \
    'With what probability should we choose each strategy given in hp_tune_criterion.' + \
    'If "uniform" we we will use uniform probabilities and if "adaptive" we will use ' + \
    'adaptive probabilities which weight acquisitions according to how well they do.'),
  get_option_specs('gpb_ml_hp_tune_opt', False, 'direct',
                   'Which optimiser to use when maximising the tuning criterion.'),
  get_option_specs('gpb_post_hp_tune_method', False, 'slice',
                   'Which sampling to use when maximising the tuning criterion. Other ' +
                   'option is nuts'),
  get_option_specs('gpb_post_hp_tune_burn', False, -1,
                   'How many initial samples to ignore during sampling.'),
  get_option_specs('gpb_post_hp_tune_offset', False, 25,
                   'How many samples to ignore between samples.'),
  get_option_specs('rand_exp_sampling_replace', False, False, \
    'Whether to replace already sampled values or not in rand_exp_sampling.'),
  # For multi-fidelity BO
  get_option_specs('mf_strategy', False, 'boca',
                   'Which multi-fidelity strategy to use. Should be one of {boca}.'),
  # The following are perhaps not so important. Some have not been implemented yet.
  get_option_specs('shrink_kernel_with_time', False, 0,
                   'If True, shrinks the kernel with time so that we don\'t get stuck.'),
  get_option_specs('perturb_thresh', False, 1e-4, \
    ('If the next point chosen is too close to an exisiting point by this times the ' \
     'diameter, then we will perturb the point a little bit before querying. This is ' \
     'mainly to avoid numerical stability issues.')),
  get_option_specs('track_every_time_step', False, 0,
                   ('If 1, it tracks every time step.')),
  get_option_specs('next_pt_std_thresh', False, 0.005, \
    ('If the std of the queried point queries below this times the kernel scale ', \
     'frequently we will reduce the bandwidth range')), \
  ]

mf_gp_bandit_args = [ \
  get_option_specs('target_fidel_to_opt_query_frac_max', False, 0.5,
                   ('A target to maintain on the number of queries to fidel_to_opt.')),
  get_option_specs('target_fidel_to_opt_query_frac_min', False, 0.25,
                   ('A target to maintain on the number of queries to fidel_to_opt.')),
  get_option_specs('boca_thresh_window_length', False, 20, \
    ('The window length to keep checking if the target fidel_to_opt is achieved.')),
  get_option_specs('boca_thresh_coeff_init', False, 0.01,
                   ('The coefficient to use in determining the threshold for boca.')),
  get_option_specs('boca_thresh_multiplier', False, 1.1, \
    ('The amount by which to multiply/divide the threshold coeff for boca.')),
  get_option_specs('boca_max_low_fidel_cost_ratio', False, 0.90, \
    ('If the fidel_cost_ratio is larger than this, just query at fidel_to_opt.')), \
  ]

euclidean_specific_gp_bandit_args = [ \
  get_option_specs('euc_init_method', False, 'latin_hc', \
    'Method to obtain initial queries. Is used if get_initial_qinfos is None.'), \
  ]

def get_all_gp_bandit_args_from_gp_args(gp_args):
  """ Returns the GP bandit arguments from the arguments for the GP. """
  return gp_args + blackbox_opt_args + gp_bandit_args

def get_all_mf_gp_bandit_args_from_gp_args(gp_args):
  """ Returns the GP bandit arguments from the arguments for the GP. """
  return gp_args + blackbox_opt_args + gp_bandit_args + mf_exd_args + mf_gp_bandit_args


# The GPBandit Class
# ========================================================================================
class GPBandit(BlackboxOptimiser):
  """ GPBandit Class. """
  # pylint: disable=attribute-defined-outside-init

  # Constructor.
  def __init__(self, func_caller, worker_manager, is_mf=False,
               options=None, reporter=None):
    """ Constructor. """
    self._is_mf = is_mf
    if is_mf and not func_caller.is_mf():
      raise CalledMFOptimiserWithSFCaller(self, func_caller)
    super(GPBandit, self).__init__(func_caller, worker_manager, None,
                                   options=options, reporter=reporter)

  def is_an_mf_method(self):
    """ Returns Truee since this is a MF method. """
    return self._is_mf

  def _get_method_str(self):
    """ Returns a string describing the method. """
    gpb_str = 'mfbo-%s'%(self.options.mf_strategy) if self.is_an_mf_method() else 'bo'
    return '%s(%s)'%(gpb_str, self.options.acq)

  def _opt_method_set_up(self):
    """ Some set up for the GPBandit class. """
    # Set up acquisition optimisation
    self.gp = None
    self._set_up_acq_opt()
    self.acqs_to_use = [elem.lower() for elem in self.options.acq.split('-')]
    self.acqs_to_use_counter = {key: 0 for key in self.acqs_to_use}
    if self.options.acq_probs == 'uniform':
      self.acq_probs = np.ones(len(self.acqs_to_use)) / float(len(self.acqs_to_use))
    elif self.options.acq_probs == 'adaptive':
      self.acq_uniform_sampling_prob = 0.05
      self.acq_sampling_weights = {key: 1.0 for key in self.acqs_to_use}
      self.acq_probs = self._get_adaptive_ensemble_acq_probs()
    else:
      self.acq_probs = np.array([float(x) for x in self.options.acq_probs.split('-')])
    self.acq_probs = self.acq_probs / self.acq_probs.sum()
    assert len(self.acq_probs) == len(self.acqs_to_use)
    # Override options for hp_tune_criterion and ml_hp_tune_opt
    self.options.hp_tune_criterion = self.options.gpb_hp_tune_criterion
    self.options.hp_tune_probs = self.options.gpb_hp_tune_probs
    self.options.ml_hp_tune_opt = self.options.gpb_ml_hp_tune_opt
    self.options.post_hp_tune_method = self.options.gpb_post_hp_tune_method
    self.options.post_hp_tune_burn = self.options.gpb_post_hp_tune_burn
    self.options.post_hp_tune_offset = self.options.gpb_post_hp_tune_offset
    # To store in history
    self.history.query_acqs = []
    self.to_copy_from_qinfo_to_history['curr_acq'] = 'query_acqs'
    # For multi-fidelity
    if self.is_an_mf_method():
      self.mf_params_for_anc_data = {}
      if self.options.mf_strategy == 'boca':
        self.mf_params_for_anc_data['boca_thresh_coeff'] = \
          self.options.boca_thresh_coeff_init
        self.mf_params_for_anc_data['boca_max_low_fidel_cost_ratio'] = \
          self.options.boca_max_low_fidel_cost_ratio
    self._child_opt_method_set_up()

  def _child_opt_method_set_up(self):
    """ Set up for child class. Override this method in child class"""
    pass

  def _get_adaptive_ensemble_acq_probs(self):
    """ Computes the adaptive ensemble acqusitions probs. """
    num_acqs = len(self.acqs_to_use)
    uniform_sampling_probs = self.acq_uniform_sampling_prob * \
                             np.ones((num_acqs,)) / num_acqs
    acq_succ_counter = np.array([self.acq_sampling_weights[key] for
                                 key in self.acqs_to_use])
    acq_use_counter = np.array([self.acqs_to_use_counter[key] for
                                key in self.acqs_to_use])
    acq_weights = acq_succ_counter / np.sqrt(1 + acq_use_counter)
    acq_norm_weights = acq_weights / acq_weights.sum()
    adaptive_sampling_probs = (1 - self.acq_uniform_sampling_prob) * acq_norm_weights
    ret = uniform_sampling_probs + adaptive_sampling_probs
    return ret / ret.sum()

  def _get_acq_opt_method(self):
    """ Returns the method for optimising the acquisition. """
    if self.options.acq_opt_method == 'dflt_domain_opt_method':
      return self.domain.dflt_domain_opt_method
    else:
      return self.options.acq_opt_method

  def _set_up_acq_opt(self):
    """ Sets up optimisation for acquisition. """
    # First set up function to get maximum evaluations.
    if isinstance(self.options.acq_opt_max_evals, int):
      if self.options.acq_opt_max_evals > 0:
        self.get_acq_opt_max_evals = lambda t: self.options.acq_opt_max_evals
      else:
        self.get_acq_opt_max_evals = None
    else: # In this case, the user likely passed a function here.
      self.get_acq_opt_max_evals = self.options.acq_opt_max_evals
    # Additional set up based on the specific optimisation procedure
    acq_opt_method = self._get_acq_opt_method()
    if acq_opt_method in ['direct']:
      self._set_up_acq_opt_direct()
    elif acq_opt_method in ['pdoo']:
      self._set_up_acq_opt_pdoo()
    elif acq_opt_method == 'rand':
      self._set_up_acq_opt_rand()
    elif acq_opt_method == 'ga':
      self._set_up_acq_opt_ga()
    else:
      raise NotImplementedError('Not implemented acquisition optimisation for %s yet.'%( \
                                self.options.acq_opt_method))

  def _opt_method_update_history(self, qinfo):
    """ Update history for GP bandit specific statistics. """
    if hasattr(qinfo, 'curr_acq'):
      self.acqs_to_use_counter[qinfo.curr_acq] += 1
      if self.options.acq_probs == 'adaptive' and \
         (len(self.history.curr_opt_vals) >= 2 and
          self.history.curr_opt_vals[-1] > self.history.curr_opt_vals[-2]):
        self.acq_sampling_weights[qinfo.curr_acq] += 1
    if hasattr(self, 'gp_processor') and (len(self.history.curr_opt_vals) >= 2 and \
       self.history.curr_opt_vals[-1] > self.history.curr_opt_vals[-2]):
      self.gp_processor.gp_fitter.update_hp_tune_method_weight( \
                                                         self.gp_processor.hp_tune_method)
    self._child_opt_method_update_history(qinfo)

  def _child_opt_method_update_history(self, qinfo):
    """ Update history for child GP bandit specific statistics. """
    pass

  # Any of these set up methods can be overridden by a child class -------------------
  def _set_up_acq_opt_direct(self):
    """ Sets up optimisation for acquisition using direct/pdoo. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 1 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     1000, 3e4)

  def _set_up_acq_opt_pdoo(self):
    """ Sets up optimisation for acquisition using direct/pdoo. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 2 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     2000, 5e4)

  def _set_up_acq_opt_rand(self):
    """ Sets up optimisation for acquisition using random search. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 10 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     2000, 3e4)

  def _set_up_acq_opt_ga(self):
    """ Sets up optimisation for the acquisition using genetic algorithms. """
    raise NotImplementedError('Not Implemented GA for usual GP Bandits.')

  # Managing the GP ---------------------------------------------------------
  def _set_next_gp(self):
    """ Returns the next GP. """
    if not hasattr(self, 'gp_processor') or self.gp_processor is None:
      self._build_new_gp()
    ret = self.gp_processor.gp_fitter.get_next_gp()
    self.gp_processor.fit_type = ret[0]
    self.gp_processor.hp_tune_method = ret[1]
    self.gp = ret[2]
    self._child_set_next_gp()
    if self.gp_processor.fit_type in ['sample_hps_with_probs', \
                                      'post_sample_hps_with_probs']:
      reg_data = self._get_gp_reg_data()
      self._child_set_gp_data(reg_data)
    if self.step_idx == self.last_model_build_at or \
       self.step_idx == self.last_model_build_at + self.worker_manager.num_workers:
      # printing multiple times every model build is a way to check that rand_exp_sampling
      # is picking different hyper-parameters.
      self._report_current_gp()

  def _child_set_next_gp(self):
    """ Sets the next GP in child class """
    pass

  def _child_set_gp_data(self, reg_data):
    """ Set data in child. """
    if self.is_an_mf_method():
      self.gp.set_mf_data(reg_data[0], reg_data[1], reg_data[2], build_posterior=True)
    else:
      self.gp.set_data(reg_data[0], reg_data[1], build_posterior=True)

  def _child_build_new_model(self):
    """ Builds a new model. """
    self._build_new_gp()

  def _report_current_gp(self):
    """ Reports the current GP. """
    gp_fit_report_str = '    -- Fitting GP (j=%d): %s'%(self.step_idx, str(self.gp))
    self.reporter.writeln(gp_fit_report_str)

  def _get_opt_method_report_results_str(self):
    """ Any details to include in a child method when reporting results.
    """
    if len(self.acqs_to_use) > 1:
      acq_str = ', acqs=' + get_list_as_str([self.acqs_to_use_counter[key] for key in
                                             self.acqs_to_use])
    else:
      acq_str = ''
    if self.is_an_mf_method() and self.options.mf_strategy == 'boca':
      mf_str = ', boca-thresh=%0.3f'%(self.mf_params_for_anc_data['boca_thresh_coeff'])
    else:
      mf_str = ''
    return acq_str + mf_str

  def _get_gp_reg_data(self):
    """ Returns the current data to be added to the GP. """
    reg_X = self.prev_eval_points + self.history.query_points
    reg_Y = self.prev_eval_vals + self.history.query_vals
    if self.is_an_mf_method():
      reg_Z = self.prev_eval_fidels + self.history.query_fidels
      return reg_Z, reg_X, reg_Y
    else:
      return reg_X, reg_Y

  def _get_gp_fitter(self, reg_data, use_additive=False):
    """ Returns a GP Fitter. """
    if self.is_an_mf_method():
      return self._get_mf_gp_fitter(reg_data, use_additive)
    else:
      return self._get_non_mf_gp_fitter(reg_data, use_additive)

  def _get_mf_gp_fitter(self, reg_data, use_additive=False):
    """ Returns the Multi-fidelity GP Fitter. Can be overridded by a child class. """
    raise NotImplementedError('Implement in a Child class.')

  def _get_non_mf_gp_fitter(self, reg_data, use_additive=False):
    """ Returns the NOn-Multi-fidelity GP Fitter. Can be overridded by a child class. """
    raise NotImplementedError('Implement in a Child class.')

  def _build_new_gp(self):
    """ Builds a GP with the data in history and stores in self.gp. """
    if hasattr(self.func_caller, 'init_gp') and self.func_caller.init_gp is not None:
      # If you know the true GP.
      raise NotImplementedError('Not implemented passing given GP yet.')
    else:
      if self.options.shrink_kernel_with_time:
        raise NotImplementedError('Not implemented kernel shrinking for the GP yet.')
      # Invoke the GP fitter.
      reg_data = self._get_gp_reg_data()
      gp_fitter = self._get_gp_fitter(reg_data)
      # Fits gp and adds it to gp_processor
      gp_fitter.fit_gp_for_gp_bandit(self.options.build_new_model_every)
      self.gp = None # Mostly to avoid bugs
      self.gp_processor = Namespace()
      self.gp_processor.gp_fitter = gp_fitter
      self._child_build_new_gp(reg_data)

  def _child_build_new_gp(self, reg_data):
    """ Builds an extra GP in child class if required. """
    pass

  def _add_data_to_model(self, qinfos):
    """ Add data to self.gp """
    if self.gp is None:
      return
    if len(qinfos) == 0:
      return
    new_points = [qinfo.point for qinfo in qinfos]
    new_vals = [qinfo.val for qinfo in qinfos]
    if self.is_an_mf_method():
      new_fidels = [qinfo.fidel for qinfo in qinfos]
      self._add_data_to_gp((new_fidels, new_points, new_vals))
    else:
      self._add_data_to_gp((new_points, new_vals))

  def _add_data_to_gp(self, new_data):
    """ Adds data to the GP. """
    # Add data to the GP only if we will be repeating with the same GP.
    if self.gp_processor.fit_type == 'fitted_gp':
      if self.is_an_mf_method():
        self.gp.add_mf_data_multiple(new_data[0], new_data[1], new_data[2])
      else:
        self.gp.add_data_multiple(new_data[0], new_data[1])
    self._child_add_data_to_gp(new_data)

  def _child_add_data_to_gp(self, new_data):
    """ Adds data from the child class to the GP. """
    pass

  # Methods needed for initialisation ----------------------------------------
  def _child_optimise_initialise(self):
    """ No additional initialisation for GP bandit. """
    self._build_new_gp()

  # Methods needed for optimisation ------------------------------------------
  def _get_next_acq(self):
    """ Gets the acquisition to use in the current iteration. """
    if self.options.acq_probs == 'adaptive':
      self.acq_probs = self._get_adaptive_ensemble_acq_probs()
    ret = np.random.choice(self.acqs_to_use, p=self.acq_probs)
    return ret

  def _get_ancillary_data_for_acquisition(self, curr_acq):
    """ Returns ancillary data for the acquisitions. """
    max_num_acq_opt_evals = self.get_acq_opt_max_evals(self.step_idx)
    ret = Namespace(curr_acq=curr_acq,
                    max_evals=max_num_acq_opt_evals,
                    t=self.step_idx,
                    domain=self.domain,
                    curr_max_val=self.curr_opt_val,
                    eval_points_in_progress=self.eval_points_in_progress,
                    acq_opt_method=self.options.acq_opt_method,
                    handle_parallel=self.options.handle_parallel,
                    mf_strategy=self.options.mf_strategy,
                    is_mf=self.is_an_mf_method(),
                   )
    if curr_acq == 'add_ucb':
      ret.domain_bounds = self.domain.bounds
    if self.is_an_mf_method():
      for key, value in list(self.mf_params_for_anc_data.items()):
        setattr(ret, key, value)
      ret.eval_fidels_in_progress = self.eval_fidels_in_progress
      ret.eval_fidel_points_in_progress = self.gp.get_ZX_from_ZZ_XX( \
                              self.eval_fidels_in_progress, self.eval_points_in_progress)
    return ret

  def _determine_next_query(self):
    """ Determine the next point for evaluation. """
    raise NotImplementedError('Implement in a Child class.')

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determine the next batch of evaluation points. """
    raise NotImplementedError('Implement in a Child class.')

  def _main_loop_pre_boca(self):
    """ Things to be done before each iteration of the optimisation loop for BOCA. """
    last_window_queries_at_f2o = self.history.query_at_fidel_to_opts[ \
                                                 -self.options.boca_thresh_window_length:]
    last_window_f2o_frac = sum(last_window_queries_at_f2o) / \
                              float(self.options.boca_thresh_window_length)
    if last_window_f2o_frac <= self.options.target_fidel_to_opt_query_frac_min:
      self.mf_params_for_anc_data['boca_thresh_coeff'] *= \
        self.options.boca_thresh_multiplier
    elif last_window_f2o_frac >= self.options.target_fidel_to_opt_query_frac_max:
      self.mf_params_for_anc_data['boca_thresh_coeff'] /= \
        self.options.boca_thresh_multiplier
    if len(self.history.query_vals) > 1:
      self.mf_params_for_anc_data['y_range'] = \
        max(self.history.query_vals) - min(self.history.query_vals)
    else:
      self.mf_params_for_anc_data['y_range'] = 1.0

  def _main_loop_pre(self):
    """ Things to be done before each iteration of the optimisation loop. """
    self._set_next_gp() # set the next GP
    # For BOCA
    if self.is_an_mf_method():
      if self.options.mf_strategy == 'boca':
        self._main_loop_pre_boca()
# GP Bandit class ends here ==========================================================


# A Euclidean GP Bandit
class EuclideanGPBandit(GPBandit):
  """ A GP Bandit for Euclidean Spaces. """

  # Constructor.
  def __init__(self, func_caller, worker_manager, is_mf=False,
               options=None, reporter=None):
    """ Constructor. """
    if options is None:
      reporter = get_reporter(reporter)
      if is_mf:
        all_args = get_all_mf_gp_bandit_args_from_gp_args(euclidean_mf_gp_args)
      else:
        all_args = get_all_gp_bandit_args_from_gp_args(euclidean_gp_args)
      all_args += euclidean_specific_gp_bandit_args
      options = load_options(all_args, reporter=reporter)
    super(EuclideanGPBandit, self).__init__(func_caller, worker_manager, is_mf=is_mf,
                                            options=options, reporter=reporter)

  def _get_mf_gp_fitter(self, reg_data, use_additive=False):
    """ Returns the Multi-fidelity GP Fitter. Can be overridded by a child class. """
    options = Namespace(**vars(self.options))
    if use_additive:
      options.domain_use_additive_gp = use_additive
    if use_additive and options.domain_kernel_type == 'esp':
      options.domain_kernel_type = options.domain_esp_kernel_type
    return EuclideanMFGPFitter(reg_data[0], reg_data[1], reg_data[2],
                               options=options, reporter=self.reporter)

  def _get_non_mf_gp_fitter(self, reg_data, use_additive=False):
    """ Returns the NOn-Multi-fidelity GP Fitter. Can be overridded by a child class. """
    options = Namespace(**vars(self.options))
    if use_additive:
      options.use_additive_gp = use_additive
    if use_additive and options.kernel_type == 'esp':
      options.kernel_type = options.esp_kernel_type
    return EuclideanGPFitter(reg_data[0], reg_data[1],
                             options=options, reporter=self.reporter)

  def _child_opt_method_set_up(self):
    """ Some set up for the EuclideanGPBandit class. """
    self.add_gp = None
    # Flag for creating extra add_gp
    self.req_add_gp = False
    if self.is_an_mf_method():
      if not self.options.domain_use_additive_gp and 'add_ucb' in self.acqs_to_use:
        self.req_add_gp = True
    else:
      if not self.options.use_additive_gp and 'add_ucb' in self.acqs_to_use:
        self.req_add_gp = True

  def _child_build_new_gp(self, reg_data):
    """ Builds an additive GP if required and stores it in self.add_gp. """
    if self.req_add_gp:
      add_gp_fitter = self._get_gp_fitter(reg_data, use_additive=True)
      # Fits gp and adds it to add_gp_processor
      add_gp_fitter.fit_gp_for_gp_bandit(self.options.build_new_model_every)
      self.add_gp = None # Mostly to avoid bugs
      self.add_gp_processor = Namespace()
      self.add_gp_processor.gp_fitter = add_gp_fitter

  def _child_add_data_to_gp(self, new_data):
    """ Adds data from the child class to the EuclideanGP """
    if hasattr(self, 'add_gp_processor') and \
               self.add_gp_processor.fit_type == 'fitted_gp':
      if self.is_an_mf_method():
        if self.add_gp is not None:
          self.add_gp.add_mf_data_multiple(new_data[0], new_data[1], new_data[2])
      else:
        if self.add_gp is not None:
          self.add_gp.add_data_multiple(new_data[0], new_data[1])

  def _child_opt_method_update_history(self, qinfo):
    """ Update history for EuclideanGP bandit specific statistics. """
    if hasattr(self, 'add_gp_processor') and (len(self.history.curr_opt_vals) >= 2 and \
       self.history.curr_opt_vals[-1] > self.history.curr_opt_vals[-2]):
      self.add_gp_processor.gp_fitter.update_hp_tune_method_weight( \
                                                     self.add_gp_processor.hp_tune_method)

  def _child_set_next_gp(self):
    if hasattr(self, 'add_gp_processor'):
      ret = self.add_gp_processor.gp_fitter.get_next_gp()
      self.add_gp_processor.fit_type = ret[0]
      self.add_gp_processor.hp_tune_method = ret[1]
      self.add_gp = ret[2]
      if self.add_gp_processor.fit_type in ['sample_hps_with_probs',
                                            'post_sample_hps_with_probs']:
        reg_data = self._get_gp_reg_data()
        self._child_set_add_gp_data(reg_data)


  def _child_set_add_gp_data(self, reg_data):
    """ Set data in child. Can be overridden by a child class. """
    if self.is_an_mf_method():
      self.add_gp.set_mf_data(reg_data[0], reg_data[1], reg_data[2], build_posterior=True)
    else:
      self.add_gp.set_data(reg_data[0], reg_data[1], build_posterior=True)

  def _determine_next_query(self):
    """ Determine the next point for evaluation. """
    curr_acq = self._get_next_acq()
    anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
    select_pt_func = getattr(gpb_acquisitions.asy, curr_acq)
    qinfo = Namespace(curr_acq=curr_acq)
    if self.is_an_mf_method():
      if self.options.mf_strategy == 'boca':
        next_eval_fidel, next_eval_point = gpb_acquisitions.boca(select_pt_func, \
          self.gp, anc_data, self.func_caller)
        qinfo.fidel = next_eval_fidel
        qinfo.point = next_eval_point
      else:
        raise ValueError('Unknown mf_strategy: %s.'%(self.options.mf_strategy))
    else:
      if self.add_gp is None or curr_acq != 'add_ucb':
        next_eval_point = select_pt_func(self.gp, anc_data)
      else:
        next_eval_point = select_pt_func(self.add_gp, anc_data)
      qinfo.point = next_eval_point
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determine the next batch of evaluation points. """
    # TODO: use get_syn_recos_from_asy_in_gpb_acquisitions
    curr_acq = self._get_next_acq()
    anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
    select_pt_func = getattr(gpb_acquisitions.syn, curr_acq)
    if self.is_an_mf_method():
      raise NotImplementedError('Not Implemented synchronous mf yet!')
    else:
      if self.add_gp is None or curr_acq != 'add_ucb':
        next_batch_of_eval_points = select_pt_func(batch_size, self.gp, anc_data)
      else:
        next_batch_of_eval_points = select_pt_func(batch_size, self.add_gp, anc_data)
      qinfos = [Namespace(point=pt) for pt in next_batch_of_eval_points]
    return qinfos

  def _get_initial_qinfos(self, num_init_evals):
    """ Returns initial qinfos. """
    if self.is_an_mf_method():
      return get_euclidean_initial_qinfos(self.options.euc_init_method, num_init_evals, \
               self.domain.bounds, self.options.fidel_init_method, \
               self.fidel_space.bounds, self.func_caller.fidel_to_opt, \
               self.options.init_set_to_fidel_to_opt_with_prob)
    else:
      return get_euclidean_initial_qinfos(self.options.init_method, num_init_evals,
                                          self.domain.bounds)

  def _set_up_acq_opt_ga(self):
    """ Sets up optimisation for the acquisition using genetic algorithms. """
    raise ValueError('Do not use GA for optimising Euclidean Domains.')

# APIs for Euclidean GP Bandit optimisation. ---------------------------------------------
# 1. Optimisation from a FunctionCaller object.
def gpb_from_func_caller(func_caller, worker_manager, max_capital, is_mf, mode=None,
                         acq=None, mf_strategy=None, domain_add_max_group_size=-1,
                         options=None, reporter='default'):
  """ GP Bandit optimisation from a utils.function_caller.FunctionCaller instance.
    domain_add_max_group_size indicates whether we should use an additive model or not.
    If its negative, then use a non-additive model. If its positive, then use an additive
    model with maximum group size as given. If zero, then use the default in options.
  """
  # pylint: disable=too-many-branches
  reporter = get_reporter(reporter)
  if is_mf:
    if isinstance(func_caller.fidel_space, domains.EuclideanDomain) and \
       isinstance(func_caller.domain, domains.EuclideanDomain):
      optimiser_constructor = EuclideanGPBandit
      dflt_list_of_options = get_all_mf_gp_bandit_args_from_gp_args(euclidean_mf_gp_args)
  else:
    if isinstance(func_caller.domain, domains.EuclideanDomain):
      optimiser_constructor = EuclideanGPBandit
      dflt_list_of_options = get_all_gp_bandit_args_from_gp_args(euclidean_gp_args)
    else:
      raise ValueError('Random optimiser not implemented for domain of type %s.'%( \
                       type(func_caller.domain)))
  # Load options
  if options is None:
    reporter = get_reporter(reporter)
    options = load_options(dflt_list_of_options, reporter=reporter)
  if acq is not None:
    options.acq = acq
  if mode is not None:
    options.mode = mode
  if mf_strategy is not None:
    options.mf_strategy = mf_strategy
  # Additive model
  if domain_add_max_group_size >= 0:
    if is_mf:
      options.domain_use_additive_gp = True
      if domain_add_max_group_size > 0:
        options.domain_add_max_group_size = domain_add_max_group_size
    else:
      options.use_additive_gp = True
      if domain_add_max_group_size > 0:
        options.add_max_group_size = domain_add_max_group_size
  # create optimiser and return
  optimiser = optimiser_constructor(func_caller, worker_manager, is_mf=is_mf,
                                    options=options, reporter=reporter)
  return optimiser.optimise(max_capital)

