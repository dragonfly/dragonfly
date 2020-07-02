"""
  Defines a class for Multi-objective Blackbox Optimisation.
  -- bparia@cs.cmu.edu
  -- kandasamy@cs.cmu.edu
"""

# NB: In this file, the acronym MOO/moo refers to multi-objective optimisation. --KK

# pylint: disable=abstract-class-little-used
# pylint: disable=invalid-name

from __future__ import division
from argparse import Namespace
import numpy as np
# Local imports
from ..exd.exd_core import ExperimentDesigner, exd_core_args
from ..exd.experiment_caller import MultiFunctionCaller, FunctionCaller
from ..exd.exd_utils import EVAL_ERROR_CODE
from ..utils.general_utils import update_pareto_set


multiobjective_opt_args = exd_core_args

_NO_MF_FOR_MOO_ERR_MSG = 'Multi-fidelity support has not been implemented yet' + \
                         ' for multi-objective optimisation.'


class MultiObjectiveOptimiser(ExperimentDesigner):
  """ Blackbox Optimiser Class. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, multi_func_caller, worker_manager, model=None, options=None,
               reporter=None):
    """ Constructor. """
    assert isinstance(multi_func_caller, MultiFunctionCaller) and not \
           isinstance(multi_func_caller, FunctionCaller)
    self.multi_func_caller = multi_func_caller
    # If it is not a list, computing the non-dominated set is equivalent
    # to computing the maximum.
    self.domain = self.multi_func_caller.domain
    super(MultiObjectiveOptimiser, self).__init__(multi_func_caller, worker_manager,
                                                  model, options, reporter)

  def _exd_child_set_up(self):
    """ Set up for the optimisation. """
    if self.multi_func_caller.is_mf():
      # self.num_fidel_to_opt_calls = 0
      raise NotImplementedError(_NO_MF_FOR_MOO_ERR_MSG)
    self._moo_set_up()
    self._multi_opt_method_set_up()
    self.prev_eval_vals = [] # for optimiser, prev_eval_vals

  def _moo_set_up(self):
    """ Set up for black-box optimisation. """
    # Initialise optimal values and points
    # (Optimal point for MF problems is not defined)
    # (Instead a set of pareto optimal points will be maintained)
    self.curr_pareto_vals = []
    self.curr_pareto_points = []
    self.curr_true_pareto_vals = []
    self.curr_true_pareto_points = []
    # Set up history
    self.history.query_vals = []
    self.history.query_true_vals = []
    self.history.curr_pareto_vals = []
    self.history.curr_pareto_points = []
    self.history.curr_true_pareto_vals = []
    self.history.curr_true_pareto_points = []
    if self.multi_func_caller.is_mf():
      # self.history.query_at_fidel_to_opts = []
      raise NotImplementedError(_NO_MF_FOR_MOO_ERR_MSG)
    # Set up attributes to be copied from history
    self.to_copy_from_qinfo_to_history['val'] = 'query_vals'
    self.to_copy_from_qinfo_to_history['true_val'] = 'query_true_vals'
    # Set up previous evaluations
    self.prev_eval_vals = []
    self.prev_eval_true_vals = []
    self.history.prev_eval_vals = self.prev_eval_vals
    self.history.prev_eval_true_vals = self.prev_eval_true_vals

  def _multi_opt_method_set_up(self):
    """ Any set up for the specific optimisation method. """
    raise NotImplementedError('Implement in Optimisation Method class.')

  def _get_problem_str(self):
    """ Description of the problem. """
    return 'Multi-objective Optimisation'

  # Book-keeping ----------------------------------------------------------------
  def _exd_child_update_history(self, qinfo):
    """ Updates to the history specific to optimisation. """
    # Update the best point/val
    # check fidelity
    if self.multi_func_caller.is_mf():
      raise NotImplementedError(_NO_MF_FOR_MOO_ERR_MSG)
    else:
      self._update_opt_point_and_val(qinfo)
    # Now add to history
    self.history.curr_pareto_vals.append(self.curr_pareto_vals)
    self.history.curr_pareto_points.append(self.curr_pareto_points)
    self.history.curr_true_pareto_vals.append(self.curr_true_pareto_vals)
    self.history.curr_true_pareto_points.append(self.curr_true_pareto_points)
    # Any method specific updating
    self._multi_opt_method_update_history(qinfo)

  def _update_opt_point_and_val(self, qinfo, query_is_at_fidel_to_opt=None):
    """ Updates the optimum point and value according the data in qinfo.
        Can be overridden by a child class if you want to do anything differently.
    """
    if query_is_at_fidel_to_opt is not None:
      if not query_is_at_fidel_to_opt:
        # if the fidelity queried at is not fidel_to_opt, then return
        return
    if qinfo.val == EVAL_ERROR_CODE:
      return
    # Optimise curr_opt_val and curr_true_opt_val
    self.curr_pareto_vals, self.curr_pareto_points = update_pareto_set(
      self.curr_pareto_vals, self.curr_pareto_points, qinfo.val, qinfo.point)
    self.curr_true_pareto_vals, self.curr_true_pareto_points = update_pareto_set(
      self.curr_true_pareto_vals, self.curr_true_pareto_points, qinfo.true_val,
      qinfo.point)

  def _multi_opt_method_update_history(self, qinfo):
    """ Any updates to the history specific to the method. """
    pass # Pass by default. Not necessary to override.

  def _get_exd_child_header_str(self):
    """ Header for black box optimisation. """
    ret = '#Pareto=<num_pareto_optimal_points_found>'
    ret += self._get_opt_method_header_str()
    return ret

  @classmethod
  def _get_opt_method_header_str(cls):
    """ Header for optimisation method. """
    return ''

  def _get_exd_child_report_results_str(self):
    """ Returns a string describing the progress in optimisation. """
    best_val_str = '#Pareto: %d'%(len(self.curr_pareto_vals))
    opt_method_str = self._get_opt_method_report_results_str()
    return best_val_str + opt_method_str + ', '

  def _get_opt_method_report_results_str(self):
    """ Any details to include in a child method when reporting results.
        Can be overridden by a child class.
    """
    #pylint: disable=no-self-use
    return ''

  def _exd_child_handle_prev_evals_in_options(self):
    """ Handles pre-evaluations. """
    ret = 0
    for qinfo in self.options.prev_evaluations.qinfos:
      if not hasattr(qinfo, 'true_val'):
        qinfo.true_val = [-np.inf] * len(qinfo.val)
      if self.multi_func_caller.is_mf():
        raise NotImplementedError(_NO_MF_FOR_MOO_ERR_MSG)
      else:
        self._update_opt_point_and_val(qinfo)
      self.prev_eval_points.append(qinfo.point)
      self.prev_eval_vals.append(qinfo.val)
      self.prev_eval_true_vals.append(qinfo.true_val)
      ret += 1
    return ret

  def _child_handle_data_loaded_from_file(self, loaded_data_from_file):
    """ Handles evaluations from file. """
    query_points = loaded_data_from_file['points']
    num_pts_in_file = len(query_points)
    query_vals = loaded_data_from_file['vals']
    assert num_pts_in_file == len(query_vals)
    if 'true_vals' in loaded_data_from_file:
      query_true_vals = loaded_data_from_file['true_vals']
      assert num_pts_in_file == len(query_vals)
    else:
      query_true_vals = [[-np.inf] * self.multi_func_caller.num_funcs] * len(query_vals)
    # Multi-fidelity
    if self.multi_func_caller.is_mf():
      raise NotImplementedError('Not implemented multi-fidelity MOO yet.')
    # Now Iterate through each point
    for pt, val, true_val in zip(query_points, query_vals, query_true_vals):
      qinfo = Namespace(point=pt, val=val, true_val=true_val)
      if self.multi_func_caller.is_mf():
        raise NotImplementedError('Not implemented multi-fidelity MOO yet.')
      else:
        self._update_opt_point_and_val(qinfo)
      self.prev_eval_points.append(qinfo.point)
      self.prev_eval_vals.append(qinfo.val)
      self.prev_eval_true_vals.append(qinfo.true_val)
    return num_pts_in_file

  def _exd_child_get_data_to_save(self):
    """ Return data to save. """
    ret = {'points': self.prev_eval_points + self.history.query_points,
           'vals': self.prev_eval_vals + self.history.query_vals,
           'true_vals': self.prev_eval_true_vals + self.history.query_true_vals}
    if self.multi_func_caller.is_mf():
      raise NotImplementedError('Not implemented multi-fidelity MOO yet.')
    num_data_saved = len(ret['points'])
    return ret, num_data_saved

  def _child_run_experiments_initialise(self):
    """ Handles any initialisation before running experiments. """
    self._opt_method_optimise_initalise()

  def _opt_method_optimise_initalise(self):
    """ Any routine to run for a method just before optimisation routine. """
    pass # Pass by default. Not necessary to override.

  def optimise(self, max_capital):
    """ Calling optimise with optimise the function. A wrapper for run_experiments from
        BlackboxExperimenter. """
    ret = self.run_experiments(max_capital)
    return ret

  def _get_final_return_quantities(self):
    """ Return the curr_opt_val, curr_opt_point and history. """
    return self.curr_pareto_vals, self.curr_pareto_points, self.history

