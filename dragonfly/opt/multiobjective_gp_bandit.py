"""
  Multi-Objective Gaussian Process Bandit Optimisation.
  - bparia@cs.cmu.edu
  - kandasamy@cs.cmu.edu

  The framework allows implementing several methods for multi-objective Bayesian
  optimisation, although we have only implemented the MOORS algorithm from,
  "A Flexible Framework for Multi-Objective Bayesian Optimization using Random
   Scalarizations", Paria, Kandasamy, Poczos.
  (Available at https://arxiv.org/pdf/1805.12168.pdf.)
"""

from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=redefined-builtin

from argparse import Namespace
import numpy as np

# Local imports
from ..exd.cp_domain_utils import get_processed_func_from_raw_func_for_cp_domain, \
                                load_cp_domain_from_config_file
from ..exd import domains
from ..exd.exd_utils import get_euclidean_initial_qinfos, get_cp_domain_initial_qinfos, EVAL_ERROR_CODE
from ..exd.experiment_caller import CPMultiFunctionCaller
from ..gp.euclidean_gp import EuclideanGPFitter
from ..gp.cartesian_product_gp import CPGPFitter
from . import multiobjective_gpb_acquisitions
from .blackbox_optimiser import CalledMFOptimiserWithSFCaller
from .multiobjective_optimiser import MultiObjectiveOptimiser
from .gp_bandit import GPBandit, get_all_euc_gp_bandit_args, get_all_cp_gp_bandit_args
from ..utils.option_handler import get_option_specs, load_options
from ..utils.reporters import get_reporter


_NO_MF_FOR_MOGPB_ERR_MSG = 'Multi-fidelity support has not been implemented yet' + \
                           ' for multi-objective GP Bandits.'

multiobjective_gp_bandit_args = [
  get_option_specs('moo_strategy', False, 'moors', \
    'Get name of multi-objective strategy. So far, Dragonfly only supports moors.'),
  # Arguments for MOORS
  get_option_specs('moors_scalarisation', False, 'tchebychev', \
    'Scalarisation for MOORS. Should be "tchebychev" or "linear".'),
  get_option_specs('moors_weight_sampler', False, 'flat_uniform', \
    'A weight sampler for moors.'),
  get_option_specs('moors_reference_point', False, None, \
    'Reference point for MOORS.'),
  # Prior mean functions
  get_option_specs('moo_gpb_prior_means', False, None, \
    'Prior GP mean functions for Multi-objective GP bandits.'),
]


def get_all_euc_moo_gp_bandit_args(additional_args=None):
  """ Returns all arguments for Euclidean MOO GP Bandits. """
  if additional_args is None:
    additional_args = []
  return get_all_euc_gp_bandit_args(additional_args + multiobjective_gp_bandit_args)

def get_all_cp_moo_gp_bandit_args(additional_args=None):
  """ Returns all arguments for Euclidean MOO GP Bandits. """
  if additional_args is None:
    additional_args = []
  return get_all_cp_gp_bandit_args(additional_args + multiobjective_gp_bandit_args)


# Some utilities for MOORS

def _get_moors_weight_sampler(multi_func_caller, weight_sampler):
  """ Returns a weight sampler. """
  if hasattr(weight_sampler, '__call__'):
    return weight_sampler
  elif weight_sampler == 'flat_uniform':
    def _get_uniform_weight_sampler(_num_funcs):
      """ Draws flat uniform weights. """
      return lambda: np.abs(np.random.normal(loc=0.0, scale=10, size=(_num_funcs,)))
    return _get_uniform_weight_sampler(multi_func_caller.num_funcs)
  else:
    raise ValueError('Could not process argument for weight_sampler: %s.'%(
                      weight_sampler))

def _get_moors_reference_point(multi_func_caller, reference_point):
  """ Returns the reference point. """
  if hasattr(reference_point, '__len__') and \
    len(reference_point) == multi_func_caller.num_funcs:
    return reference_point
  elif reference_point is None:
    return [-1.0] * multi_func_caller.num_funcs
  else:
    raise ValueError('Could not process argument for reference_point: %s.'%(
                     reference_point))


def get_default_moo_acquisition_for_domain(domain):
  """ Returns the default acquisition for the domain. """
  if domain.get_type() == 'euclidean':
#     return 'ucb-ts-add_ucb'
    return 'ucb-ts'
  else:
    return 'ucb-ts'


# The MultiObjectiveGPBandit Class
# ========================================================================================
class MultiObjectiveGPBandit(MultiObjectiveOptimiser, GPBandit):
  """ MultiObjGPBandit Class. """
  # pylint: disable=attribute-defined-outside-init

  # Constructor.
  def __init__(self, multi_func_caller, worker_manager, is_mf=False,
               options=None, reporter=None):
    """ Constructor. """
    self._is_mf = is_mf
    if is_mf:
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    if is_mf and not multi_func_caller.is_mf():
      raise CalledMFOptimiserWithSFCaller(self, multi_func_caller)
    super(MultiObjectiveGPBandit, self).__init__(multi_func_caller, worker_manager,
                                                 None, options=options, reporter=reporter)

  def _get_method_str(self):
    """ Returns a string describing the method. """
    gpb_str = 'mfmobo-%s'%(self.options.mf_strategy) if self.is_an_mf_method() else 'mobo'
    return '%s(%s)'%(gpb_str, '-'.join(self.acqs_to_use))

  def _multi_opt_method_set_up(self):
    """ Some set up for the GPBandit class. """
    # Set up acquisition optimisation
    self.gps = [None] * self.multi_func_caller.num_funcs
    super(MultiObjectiveGPBandit, self)._opt_method_set_up()
    delattr(self, 'gp')
    # MOO set up
    if self.options.moo_strategy == 'moors':
      self.moors_weight_sampler = _get_moors_weight_sampler(self.multi_func_caller,
                                                  self.options.moors_weight_sampler)
      self.moors_reference_point = _get_moors_reference_point(self.multi_func_caller,
                                                  self.options.moors_reference_point)
    else:
      raise ValueError('Unknown MOO Strategy %s.'%(self.options.moo_strategy))
    # Do any child specific set up
    self._domain_specific_multi_opt_method_set_up()

  @classmethod
  def _get_default_acquisition_for_domain(cls, domain):
    """ Return default acqusition for domain. """
    return get_default_moo_acquisition_for_domain(domain)

  def _domain_specific_multi_opt_method_set_up(self):
    """ Any domain specific set up for multi-objective GP bandit. """
    pass

  @classmethod
  def _compare_two_sets_of_obj_values(cls, obj_vals_1, obj_vals_2):
    """ Compares two sets of objective values and returns a 3-tuple where the first
        element is the number of objectives on which obj_vals_1 is better than obj_vals_2,
        the third vice versa, and the second is the number of equal objective values.
    """
    ret = [0, 0, 0]
    for obj1, obj2 in zip(obj_vals_1, obj_vals_2):
      if obj1 > obj2:
        ret[0] += 1
      elif obj1 == obj2:
        ret[1] += 1
      else:
        ret[2] += 1
    return tuple(ret)

  def _multi_opt_method_update_history(self, qinfo):
    """ Update history for GP bandit specific statistics. """
    if len(self.history.curr_pareto_vals) >= 2:
      num_improvements, _, _ = self._compare_two_sets_of_obj_values(
        self.history.curr_pareto_vals[-1], self.history.curr_pareto_vals[-2])
      if hasattr(qinfo, 'curr_acq'):
        self.acqs_to_use_counter[qinfo.curr_acq] += 1
        if self.options.acq_probs == 'adaptive':
          self.acq_sampling_weights[qinfo.curr_acq] += num_improvements
      if hasattr(self, 'gp_processors') and hasattr(qinfo, 'hp_tune_method'):
        for gp_proc in self.gp_processors:
          gp_proc.gp_fitter.update_hp_tune_method_weight(qinfo.hp_tune_method,
                                                         num_improvements)
      self._domain_specific_multi_opt_method_update_history(qinfo, num_improvements)

  def _domain_specific_multi_opt_method_update_history(self, qinfo, num_improvements):
    """ Updates to the history specific to the domain. """
    pass

  # Managing the GP ---------------------------------------------------------
  def _set_next_gp(self):
    """ Returns the next GP. """
    if not hasattr(self, 'gp_processors') or self.gp_processors is None:
      self._build_new_gps()
    self.gps = []
    for gp_processor in self.gp_processors:
      ret = gp_processor.gp_fitter.get_next_gp()
      gp_processor.fit_type = ret[0]
      gp_processor.hp_tune_method = ret[1]
      self.gps.append(ret[2])
    self._domain_specific_set_next_gp()
    for i, gp_processor in enumerate(self.gp_processors):
      if gp_processor.fit_type in ['sample_hps_with_probs', \
                                        'post_sample_hps_with_probs']:
        reg_data = self._get_moo_gp_reg_data(i)
        self._child_set_moo_gp_data(reg_data, i)
    if self.step_idx == self.last_model_build_at and \
       self.options.report_model_on_each_build:
      self._report_current_gps()

  def _domain_specific_set_next_gp(self):
    """ Sets the next GP in child class """
    pass

  def _child_set_gp_data(self, reg_data):
    """ Set data in child. """
    raise NotImplementedError('Use _get_moo_gp_reg_data instead!')

  def _child_set_moo_gp_data(self, reg_data, obj_ind):
    """ Set data in child. """
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      self.gps[obj_ind].set_data(reg_data[0], reg_data[1], build_posterior=True)

  def _child_build_new_model(self):
    """ Builds a new model. """
    self._build_new_gps()

  def _report_model(self):
    """ Report the current model. """
    # We will do this separately since posterior sampling complicates how we keep
    # track of the GPs. See _set_next_gp()
    pass

  def _report_current_gps(self):
    """ Reports the current GP. """
    for gp_idx, gp in enumerate(self.gps):
      gp_fit_report_str = '    -- GP-%d at iter %d: %s'%(
                            gp_idx, self.step_idx, str(gp))
      self.reporter.writeln(gp_fit_report_str)

  def _get_opt_method_header_str(self):
    """ Header for optimisation Method. """
    return ', acqs=<num_times_each_acquisition_was_used>'

  def _get_opt_method_report_results_str(self):
    """ Any details to include in a child method when reporting results.
    """
    ret_list = ['%s:%d'%(key, self.acqs_to_use_counter[key]) for key in self.acqs_to_use]
    ret = 'acqs=[' + ', '.join(ret_list) + ']'
    return ', ' + ret

  def _get_gp_reg_data(self):
    """ Returns the current data to be added to the GP. """
    raise NotImplementedError('Use _get_moo_gp_reg_data instead!')

  def _get_moo_gp_reg_data(self, obj_ind):
    """ Returns the current data to be added to the GP. """
    # pylint: disable=no-member
    reg_X = self.prev_eval_points + self.history.query_points
    reg_Y = self.prev_eval_vals + self.history.query_vals
    row_idx = [i for i,y in enumerate(reg_Y)
               if y != EVAL_ERROR_CODE]
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      return ([reg_X[i] for i in row_idx],
              [reg_Y[i][obj_ind] for i in row_idx])

  def _get_gp_fitter(self, gp_idx, use_additive=False):
    """ Returns a GP Fitter. """
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      return self._get_non_mf_gp_fitter(gp_idx, use_additive)

  def _get_mf_gp_fitter(self, gp_idx, use_additive=False):
    """ Returns the Multi-fidelity GP Fitter. Can be overridded by a child class. """
    raise NotImplementedError('Implement in a Child class.')

  def _get_non_mf_gp_fitter(self, gp_idx, use_additive=False):
    """ Returns the NOn-Multi-fidelity GP Fitter. Can be overridded by a child class. """
    raise NotImplementedError('Implement in a Child class.')

  def _get_options_for_gp_fitter(self, gp_idx, *args, **kwargs):
    """ Returns options for the GP Fitter. """
    # pylint: disable=unused-argument
    gpf_options = Namespace(**vars(self.options))
    if not hasattr(self.options, 'moo_gpb_prior_means') or \
      self.options.moo_gpb_prior_means is None:
      mean_func = None
    else:
      mean_func = self.options.moo_gpb_prior_means[gp_idx]
    gpf_options.mean_func = mean_func
    return gpf_options

  def _build_new_gps(self):
    """ Builds a GP with the data in history and stores in self.gp. """
    if hasattr(self.multi_func_caller, 'init_gps') and\
       self.multi_func_caller.init_gps is not None:
      # If you know the true GP.
      raise NotImplementedError('Not implemented passing given GP yet.')
    else:
      if self.options.shrink_kernel_with_time:
        raise NotImplementedError('Not implemented kernel shrinking for the GP yet.')
      # Invoke the GP fitter.
      self.gp_processors = []
      for i in range(self.multi_func_caller.num_funcs):
        gp_fitter = self._get_gp_fitter(i)
        # Fits gp and adds it to gp_processor
        gp_fitter.fit_gp_for_gp_bandit(self.options.build_new_model_every)
        gp_processor = Namespace()
        gp_processor.gp_fitter = gp_fitter
        self.gp_processors.append(gp_processor)
      self.gps = None # Mostly to avoid bugs
      self._domain_specific_build_new_gps()

  def _domain_specific_build_new_gps(self):
    """ Builds an extra GP in child class if required. """
    pass

  def _domain_specific_build_new_gp(self, _):
    """ Builds an extra GP in child class if required. """
    raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)

  def _add_data_to_model(self, qinfos):
    """ Add data to self.gp """
    if self.gps is None:
      return
    qinfos = [qinfo for qinfo in qinfos
              if qinfo.val != EVAL_ERROR_CODE]
    if len(qinfos) == 0:
      return
    new_points = [qinfo.point for qinfo in qinfos]
    new_vals = [qinfo.val for qinfo in qinfos]
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      self._add_data_to_gps((new_points, new_vals))

  def _add_data_to_gps(self, new_data):
    """ Adds data to the GP. """
    # Add data to the GP only if we will be repeating with the same GP.
    if hasattr(self, 'gp_processors') and hasattr(self.gp_processors[0], 'fit_type') and \
      self.gp_processors[0].fit_type == 'fitted_gp':
      if self.is_an_mf_method():
        raise NotImplementedError("MF with MO is not supported yet")
      else:
        for i, gp in enumerate(self.gps):
          if self.gp_processors[i].fit_type == 'fitted_gp':
            vals = [y[i] for y in new_data[1]]
            gp.add_data_multiple(new_data[0], vals)
    self._domain_specific_add_data_to_gp(new_data)

  def _domain_specific_add_data_to_gp(self, new_data):
    """ Adds data from the child class to the GP. """
    pass

  # Methods needed for initialisation ----------------------------------------
  def _child_optimise_initialise(self):
    """ No additional initialisation for GP bandit. """
    self._build_new_gps()

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
                    curr_pareto_vals=self.curr_pareto_vals,
                    eval_points_in_progress=self.eval_points_in_progress,
                    acq_opt_method=self.acq_opt_method,
                    handle_parallel=self.options.handle_parallel,
                    mf_strategy=self.options.mf_strategy,
                    is_mf=self.is_an_mf_method(),
                    num_funcs=self.multi_func_caller.num_funcs,
                   )
    if curr_acq == 'add_ucb':
      ret.domain_bounds = self.domain.bounds
    if self.options.moo_strategy == 'moors':
      ret.obj_weights = self.moors_weight_sampler()
      ret.reference_point = self.moors_reference_point
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    return ret

  def _get_name_in_multiobjective_acquisition_module(self, curr_acq):
    """ Returns the name of the acquisition as used in multiobjective_gpb_acquisitions.py.
    """
    if self.options.moo_strategy == 'moors':
      scalarisation_str = 'lin' if self.options.moors_scalarisation == 'linear' else 'tch'
      return scalarisation_str + '_' + curr_acq
    else:
      raise ValueError('Unknown MOO Strategy %s.'%(self.options.moo_strategy))

  def _determine_next_query(self):
    """ Determine the next point for evaluation. """
    curr_acq = self._get_next_acq()
    anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
    select_pt_func = getattr(multiobjective_gpb_acquisitions.asy,
                       self._get_name_in_multiobjective_acquisition_module(curr_acq))
    qinfo_hp_tune_method = self.gp_processors[0].hp_tune_method
    qinfo = Namespace(curr_acq=curr_acq, hp_tune_method=qinfo_hp_tune_method)
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      next_eval_point = select_pt_func(self.gps, anc_data)
      qinfo.point = next_eval_point
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determine the next batch of evaluation points. """
    curr_acq = self._get_next_acq()
    anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
    select_pt_func = getattr(multiobjective_gpb_acquisitions.syn,
                       self._get_name_in_multiobjective_acquisition_module(curr_acq))
    qinfo_hp_tune_method = self.gp_processors[0].hp_tune_method
    if self.is_an_mf_method():
      raise NotImplementedError('Not Implemented synchronous mf yet!')
    else:
      next_batch_of_eval_points = select_pt_func(batch_size, self.gps, anc_data)
      qinfos = [Namespace(point=pt, hp_tune_method=qinfo_hp_tune_method)
                for pt in next_batch_of_eval_points]
    return qinfos

  def _main_loop_pre_boca(self):
    """ Things to be done before each iteration of the optimisation loop for BOCA. """
    raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)

  def _main_loop_pre(self):
    """ Things to be done before each iteration of the optimisation loop. """
    self._set_next_gp() # set the next GP
    # For BOCA
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
      # if self.options.mf_strategy == 'boca':
      #   self._main_loop_pre_boca()

# MultiObjectiveGPBandit class ends here =================================================


# A Multi Objective Euclidean GP Bandit
class EuclideanMultiObjectiveGPBandit(MultiObjectiveGPBandit):
  """ A GP Bandit for Euclidean Spaces. """

  # Constructor.
  def __init__(self, multi_func_caller, worker_manager, is_mf=False,
               options=None, reporter=None):
    """ Constructor. """
    if is_mf:
      raise NotImplementedError("MF support for MO not implemented yet")
      # all_args = get_all_mf_gp_bandit_args_from_gp_args(euclidean_mf_gp_args)
    else:
      all_args = get_all_euc_moo_gp_bandit_args()
    options = load_options(all_args, partial_options=options)
    super(EuclideanMultiObjectiveGPBandit, self).__init__(multi_func_caller,
      worker_manager, is_mf=is_mf, options=options, reporter=reporter)

  def _get_mf_gp_fitter(self, gp_idx, use_additive=False):
    """ Returns the Multi-fidelity GP Fitter. Can be overridded by a child class. """
    raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)

  def _get_non_mf_gp_fitter(self, gp_idx, use_additive=False):
    """ Returns the NOn-Multi-fidelity GP Fitter. Can be overridded by a child class. """
    # pylint: disable=no-member
    options = self._get_options_for_gp_fitter(gp_idx)
    reg_data = self._get_moo_gp_reg_data(gp_idx)
    if use_additive:
      options.use_additive_gp = use_additive
    if use_additive and options.kernel_type == 'esp':
      options.kernel_type = options.domain_esp_kernel_type
    return EuclideanGPFitter(reg_data[0], reg_data[1],
                             options=options, reporter=self.reporter)


  def _domain_specific_multi_opt_method_set_up(self):
    """ Some set up for the EuclideanGPBandit class. """
    self.add_gps = None
    # Flag for creating extra add_gp
    self.req_add_gp = False
    if self.is_an_mf_method():
      raise NotImplementedError("MF support for MO has not been implemented yet")
    else:
      if not self.options.use_additive_gp and 'add_ucb' in self.acqs_to_use:
        self.req_add_gp = True

  def _domain_specific_build_new_gps(self):
    """ Builds an additive GP if required and stores it in self.add_gp. """
    if self.req_add_gp:
      self.add_gp_processors = []
      for i in range(self.multi_func_caller.num_funcs):
        add_gp_fitter = self._get_gp_fitter(i, use_additive=True)
        # Fits gp and adds it to add_gp_processor
        add_gp_fitter.fit_gp_for_gp_bandit(self.options.build_new_model_every)
        add_gp_processor = Namespace()
        add_gp_processor.gp_fitter = add_gp_fitter
        self.add_gp_processors.append(add_gp_processor)
      self.add_gps = None # Mostly to avoid bugs

  def _domain_specific_add_data_to_gp(self, new_data):
    """ Adds data from the child class to the EuclideanGP """
    if hasattr(self, 'add_gp_processors'):
      if self.is_an_mf_method():
        raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
      else:
        for add_gp_processor in self.add_gp_processors:
          if add_gp_processor.fit_type == 'fitted_gp' and self.add_gps is not None:
            split_data = _split_multi_objective_into_single_objective(
                           new_data[0], new_data[1])
            for i, add_gp in enumerate(self.add_gps):
              data_pair = split_data[i]
              add_gp.add_data_multiple(data_pair[0], data_pair[1])

  def _domain_specific_multi_opt_method_update_history(self, qinfo, num_improvements):
    """ Update history for GP bandit specific statistics. """
    if hasattr(self, 'add_gp_processors') and hasattr(qinfo, 'hp_tune_method'):
      for agp in self.add_gp_processors:
        agp.gp_fitter.update_hp_tune_method_weight(qinfo.hp_tune_method, num_improvements)

  def _domain_specific_set_next_gp(self):
    self.add_gps = []
    if hasattr(self, 'add_gp_processors'):
      for i, add_gp_processor in enumerate(self.add_gp_processors):
        ret = add_gp_processor.gp_fitter.get_next_gp()
        add_gp_processor.fit_type = ret[0]
        add_gp_processor.hp_tune_method = ret[1]
        add_gp = ret[2]
        self.add_gps.append(add_gp)
        if add_gp_processor.fit_type in ['sample_hps_with_probs',
                                         'post_sample_hps_with_probs']:
          reg_data = self._get_moo_gp_reg_data(i)
          self._child_set_add_gp_data(reg_data, i)

  def _child_set_add_gp_data(self, reg_data, obj_ind):
    """ Set data in child. Can be overridden by a child class. """
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      self.add_gps[obj_ind].set_data(reg_data[0], reg_data[1], build_posterior=True)

  def _determine_next_query(self):
    """ Determine the next point for evaluation. Override for Euclidean spaces. """
    curr_acq = self._get_next_acq()
    anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
    select_pt_func = getattr(multiobjective_gpb_acquisitions.asy,
                       self._get_name_in_multiobjective_acquisition_module(curr_acq))
    if curr_acq == 'add_ucb':
      qinfo_hp_tune_method = self.add_gp_processors[0].hp_tune_method
    else:
      qinfo_hp_tune_method = self.gp_processors[0].hp_tune_method
    qinfo = Namespace(curr_acq=curr_acq, hp_tune_method=qinfo_hp_tune_method)
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      if self.add_gps is None or curr_acq != 'add_ucb':
        next_eval_point = select_pt_func(self.gps, anc_data)
      else:
        next_eval_point = select_pt_func(self.add_gps, anc_data)
      qinfo.point = next_eval_point
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determine the next batch of evaluation points. Override for Euclidean spaces. """
    curr_acq = self._get_next_acq()
    anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
    select_pt_func = getattr(multiobjective_gpb_acquisitions.syn,
                       self._get_name_in_multiobjective_acquisition_module(curr_acq))
    if curr_acq == 'add_ucb':
      qinfo_hp_tune_method = self.add_gp_processors[0].hp_tune_method
    else:
      qinfo_hp_tune_method = self.gp_processors[0].hp_tune_method
    if self.is_an_mf_method():
      raise NotImplementedError('Not Implemented synchronous mf yet!')
    else:
      if self.add_gps is None or curr_acq != 'add_ucb':
        next_batch_of_eval_points = select_pt_func(batch_size, self.gps, anc_data)
      else:
        next_batch_of_eval_points = select_pt_func(batch_size, self.add_gps, anc_data)
      qinfos = [Namespace(point=pt, hp_tune_method=qinfo_hp_tune_method)
                for pt in next_batch_of_eval_points]
    return qinfos

  def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
    """ Returns initial qinfos. """
    if self.is_an_mf_method():
      return get_euclidean_initial_qinfos(self.options.init_method, num_init_evals, \
               self.domain.bounds, self.options.fidel_init_method, \
               self.fidel_space.bounds, self.func_caller.fidel_to_opt, \
               self.options.init_set_to_fidel_to_opt_with_prob)
    else:
      return get_euclidean_initial_qinfos(self.options.init_method, num_init_evals,
                                          self.domain.bounds)

  def _domain_specific_acq_opt_set_up(self):
    """ Set up acquisition optimisation for the child class. """
    if self.acq_opt_method.lower() in ['direct']:
      self._set_up_euc_acq_opt_direct()
    elif self.acq_opt_method.lower() in ['pdoo']:
      self._set_up_euc_acq_opt_pdoo()
    elif self.acq_opt_method.lower() == 'rand':
      self._set_up_euc_acq_opt_rand()
    else:
      raise NotImplementedError('Not implemented acquisition optimisation for %s yet.'%( \
                                self.acq_opt_method))

  # Any of these set up methods can be overridden by a child class -------------------
  def _set_up_euc_acq_opt_direct(self):
    """ Sets up optimisation for acquisition using direct/pdoo. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 1 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     1000, 3e4)

  def _set_up_euc_acq_opt_pdoo(self):
    """ Sets up optimisation for acquisition using direct/pdoo. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 2 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     2000, 6e4)

  def _set_up_euc_acq_opt_rand(self):
    """ Sets up optimisation for acquisition using random search. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 10 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     2000, 3e4)


class CPMultiObjectiveGPBandit(MultiObjectiveGPBandit):
  """ A GP Bandit class on Cartesian product spaces. """

  def __init__(self, multi_func_caller, worker_manager, is_mf=False,
               domain_dist_computers=None, options=None, reporter=None):
    """ Constructor. """
    if is_mf:
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      all_args = get_all_cp_moo_gp_bandit_args()
    options = load_options(all_args, partial_options=options)
    self.domain_dist_computers = domain_dist_computers
    super(CPMultiObjectiveGPBandit, self).__init__(multi_func_caller, worker_manager,
                                          is_mf=is_mf, options=options, reporter=reporter)

  def _domain_specific_multi_opt_method_set_up(self):
    """ Any domain specific set up for multi-objective GP bandit. """
    self.domain_lists_of_dists = None
    if self.domain_dist_computers is None:
      self.domain_dist_computers = [None] * self.domain.num_domains
    self.kernel_params_for_each_domain = [{} for _ in range(self.domain.num_domains)]
    # Create a Dummy GP Fitter so that we can get the mislabel and struct coeffs for
    # otmann.
    if self.is_an_mf_method():
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      dummy_gp_fitter = CPGPFitter([], [], self.func_caller.domain,
         domain_kernel_ordering=self.func_caller.domain_orderings.kernel_ordering,
         domain_lists_of_dists=None,
         domain_dist_computers=None,
         options=self.options, reporter=self.reporter)
    # Pre-compute distances for all sub-domains in domain - not doing for fidel_space
    # since we don't expect pre-computing distances will be necessary there.
    for idx, dom in enumerate(self.domain.list_of_domains):
      if dom.get_type() == 'neural_network' and self.domain_dist_computers[idx] is None:
        from nn.otmann import get_otmann_distance_computer_from_args
        otm_mislabel_coeffs =  \
          dummy_gp_fitter.domain_kernel_params_for_each_domain[idx].otmann_mislabel_coeffs
        otm_struct_coeffs =  \
          dummy_gp_fitter.domain_kernel_params_for_each_domain[idx].otmann_struct_coeffs
        self.domain_dist_computers[idx] = get_otmann_distance_computer_from_args(
          dom.nn_type, self.options.otmann_non_assignment_penalty,
          otm_mislabel_coeffs, otm_struct_coeffs, self.options.otmann_dist_type)
        self.kernel_params_for_each_domain[idx]['otmann_dist_type'] = \
          self.options.otmann_dist_type

  def _get_mf_gp_fitter(self, gp_idx, use_additive=False):
    """ Returns the Multi-fidelity GP Fitter. Can be overridded by a child class. """
    raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)

  def _get_non_mf_gp_fitter(self, gp_idx, use_additive=False):
    """ Returns the NOn-Multi-fidelity GP Fitter. Can be overridded by a child class. """
    options = self._get_options_for_gp_fitter(gp_idx)
    reg_data = self._get_moo_gp_reg_data(gp_idx)
    return CPGPFitter(reg_data[0], reg_data[1], self.func_caller.domain,
             domain_kernel_ordering=self.func_caller.domain_orderings.kernel_ordering,
             domain_lists_of_dists=self.domain_lists_of_dists,
             domain_dist_computers=self.domain_dist_computers,
             options=options, reporter=self.reporter)

  def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
    """ Returns initial qinfos. """
    if self.is_an_mf_method():
      return get_cp_domain_initial_qinfos(self.domain, num_init_evals,
        fidel_space=self.fidel_space, fidel_to_opt=self.func_caller.fidel_to_opt,
        set_to_fidel_to_opt_with_prob=self.options.init_set_to_fidel_to_opt_with_prob,
        dom_euclidean_sample_type='latin_hc',
        dom_integral_sample_type='latin_hc',
        dom_nn_sample_type='rand',
        fidel_space_euclidean_sample_type='latin_hc',
        fidel_space_integral_sample_type='latin_hc',
        fidel_space_nn_sample_type='rand', *args, **kwargs)
    else:
      return get_cp_domain_initial_qinfos(self.domain, num_init_evals,
                                          dom_euclidean_sample_type='latin_hc',
                                          dom_integral_sample_type='latin_hc',
                                          dom_nn_sample_type='rand', *args, **kwargs)

  # Set up for acquisition optimisation --------------------------------------------
  def _domain_specific_acq_opt_set_up(self):
    """ Set up acquisition optimisation for the child class. """
    if self.acq_opt_method.lower() in ['direct']:
      self._set_up_cp_acq_opt_direct()
    elif self.acq_opt_method.lower() in ['pdoo']:
      self._set_up_cp_acq_opt_pdoo()
    elif self.acq_opt_method.lower() == 'rand':
      self._set_up_cp_acq_opt_rand()
    elif self.acq_opt_method.lower().startswith('ga'):
      self._set_up_cp_acq_opt_ga()
    else:
      raise ValueError('Unrecognised acq_opt_method "%s".'%(self.acq_opt_method))

  # Any of these set up methods can be overridden by a child class -------------------
  def _set_up_cp_acq_opt_with_params(self, lead_const, min_iters, max_iters):
    """ Set up acquisition optimisation with params. """
    if self.get_acq_opt_max_evals is None:
      dim_factor = lead_const * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(dim_factor * np.sqrt(min(t, 1000)),
                                                     min_iters, max_iters)

  def _set_up_cp_acq_opt_direct(self):
    """ Sets up optimisation for acquisition using direct/pdoo. """
    self._set_up_cp_acq_opt_with_params(2, 1000, 3e4)

  def _set_up_cp_acq_opt_pdoo(self):
    """ Sets up optimisation for acquisition using direct/pdoo. """
    self._set_up_cp_acq_opt_with_params(2, 2000, 6e4)

  def _set_up_cp_acq_opt_rand(self):
    """ Set up optimisation for acquisition using rand. """
    self._set_up_cp_acq_opt_with_params(1, 1000, 3e4)

  def _set_up_cp_acq_opt_ga(self):
    """ Set up optimisation for acquisition using rand. """
    domain_types = [dom.get_type() for dom in self.domain.list_of_domains]
    if 'neural_network' in domain_types:
      # Because Neural networks can be quite expensive
      self._set_up_cp_acq_opt_with_params(1, 500, 2e4)
    else:
      self._set_up_cp_acq_opt_with_params(1, 1000, 3e4)





# Utilities
def _split_multi_objective_into_single_objective(X, Y):
  """ Splits k-objective observations into k sets of
      single objective observations.
  """
  num_obj = len(Y[0])
  for y in Y:
    assert len(y) == num_obj
  data_pairs = []
  for i in range(num_obj):
    pair = (X, [y[i] for y in Y])
    data_pairs.append(pair)
  return data_pairs


# APIs for Euclidean GP Bandit optimisation. ---------------------------------------------
# 1. Optimisation from a FunctionCaller object.
def multiobjective_gpb_from_multi_func_caller(multi_func_caller, worker_manager,
  max_capital, is_mf, mode=None, acq=None, moo_strategy='moors',
  moo_strategy_parameters=None, mf_strategy=None, domain_add_max_group_size=-1,
  options=None, reporter='default'):
  # pylint: disable=too-many-arguments
  """ Multi-objective GPBandit from a function caller.
  """
  # pylint: disable=too-many-branches
  reporter = get_reporter(reporter)
  if is_mf:
    raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
  else:
    if isinstance(multi_func_caller.domain, domains.EuclideanDomain):
      optimiser_constructor = EuclideanMultiObjectiveGPBandit
      dflt_list_of_options = get_all_euc_moo_gp_bandit_args()
    elif isinstance(multi_func_caller.domain, domains.CartesianProductDomain):
      optimiser_constructor = CPMultiObjectiveGPBandit
      dflt_list_of_options = get_all_cp_moo_gp_bandit_args()
    else:
      raise ValueError('GPBandit not implemented for domain of type %s.'%( \
                       type(multi_func_caller.domain)))
  # Load options
  options = load_options(dflt_list_of_options, partial_options=options)
  if acq is not None:
    options.acq = acq
  if mode is not None:
    options.mode = mode
  if mf_strategy is not None:
    options.mf_strategy = mf_strategy
  if moo_strategy == 'moors':
    if moo_strategy_parameters is not None:
      if hasattr(moo_strategy_parameters, 'moors_weight_sampler') and \
        moo_strategy_parameters.moors_weight_sampler is not None:
        options.moors_weight_sampler = moo_strategy_parameters.moors_weight_sampler
      if hasattr(moo_strategy_parameters, 'moors_reference_point') and \
        moo_strategy_parameters.moors_reference_point is not None:
        options.moors_reference_point = moo_strategy_parameters.moors_reference_point
  else:
    raise ValueError('Unknown MOO Strategy %s.'%(moo_strategy))
  # Domain Specific Parameters
  if isinstance(multi_func_caller.domain, domains.EuclideanDomain) \
    and domain_add_max_group_size >= 0:
    if is_mf:
      raise NotImplementedError(_NO_MF_FOR_MOGPB_ERR_MSG)
    else:
      options.use_additive_gp = True
      if domain_add_max_group_size > 0:
        options.add_max_group_size = domain_add_max_group_size
  # create optimiser and return
  optimiser = optimiser_constructor(multi_func_caller, worker_manager, is_mf=is_mf,
                                    options=options, reporter=reporter)
  return optimiser.optimise(max_capital)


def cp_multiobjective_gpb_from_raw_args(raw_funcs, domain_config_file, *args, **kwargs):
  """ A random multi-objective optimiser on CP spaces. """
  # pylint: disable=no-member
  cp_dom, orderings = load_cp_domain_from_config_file(domain_config_file)
  proc_funcs = [get_processed_func_from_raw_func_for_cp_domain(
                  rf, cp_dom, orderings.index_ordering, orderings.dim_ordering)
                for rf in raw_funcs]
  multi_func_caller = CPMultiFunctionCaller(proc_funcs, cp_dom, raw_funcs=raw_funcs,
                                            domain_orderings=orderings)
  return multiobjective_gpb_from_multi_func_caller(multi_func_caller, *args, **kwargs)

