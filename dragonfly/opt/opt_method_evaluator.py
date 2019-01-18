"""
  Harness for conducting black box optimisation evaluations.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=maybe-no-member

from argparse import Namespace
from datetime import datetime
import os
import numpy as np
# Local imports
from .blackbox_optimiser import OptInitialiser
from ..utils.method_evaluator import BaseMethodEvaluator


class OptMethodEvaluator(BaseMethodEvaluator):
  """ Base class for evaluating methods for optimisation. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, study_name, func_caller, worker_manager, max_capital, methods,
               num_trials, save_dir, evaluation_options, save_file_prefix='',
               method_options=None, reporter=None, **kwargs):
    """ Constructor. Also see BasicExperimenter for more args. """
    # pylint: disable=too-many-arguments
    self.func_caller = func_caller
    save_file_name = self._get_save_file_name(save_dir, study_name, \
                     worker_manager.num_workers, save_file_prefix, \
                     worker_manager.get_time_distro_info(), max_capital)
    super(OptMethodEvaluator, self).__init__(study_name, num_trials,
                                             save_file_name, reporter=reporter, **kwargs)
    self.worker_manager = worker_manager
    self.max_capital = float(max_capital)
    # Methods
    self.methods = methods
    self.num_methods = len(methods)
    self.domain = func_caller.domain
    self.method_options = (method_options if method_options else
                           {key: None for key in methods})
    # Experiment options will have things such as if the evaluations are noisy,
    # the time distributions etc.
    self.evaluation_options = evaluation_options
    self._set_up_saving()

  def _get_save_file_name(self, save_dir, study_name, num_workers, save_file_prefix,
                          time_distro_str, max_capital):
    """ Gets the save file name. """
    save_file_prefix = save_file_prefix if save_file_prefix else study_name
    noisy_str = 'noisy' if self.func_caller.is_noisy() else 'noiseless'
    save_file_name = '%s-%s-M%d-%s-c%d-%s.mat'%(save_file_prefix, noisy_str, num_workers,
      time_distro_str, int(max_capital), datetime.now().strftime('%m%d-%H%M%S'))
    save_file_name = os.path.join(save_dir, save_file_name)
    return save_file_name

  def _set_up_saving(self):
    """ Runs some routines to set up saving. """
    # Store methods and the options in to_be_saved
    self.to_be_saved.max_capital = self.max_capital
    self.to_be_saved.num_workers = self.worker_manager.num_workers
    self.to_be_saved.methods = self.methods
    self.to_be_saved.method_options = self.method_options # Some error created here.
    self.to_be_saved.time_distro_str = self.worker_manager.get_time_distro_info()
    # Data about the problem
    self.to_be_saved.true_maxval = (self.func_caller.maxval
                                    if self.func_caller.maxval is not None else -np.inf)
    self.to_be_saved.true_argmax = (self.func_caller.argmax \
                                  if self.func_caller.argmax is not None else 'not-known')
    self.to_be_saved.domain_type = self.domain.get_type()
    self.to_be_saved.is_noisy = self.func_caller.is_noisy()
    # For the results
    self.data_to_be_saved = ['query_step_idxs',
                             'query_points',
                             'query_vals',
                             'query_true_vals',
                             'query_send_times',
                             'query_receive_times',
                             'query_eval_times',
                             'query_worker_ids',
                             'curr_opt_vals',
                             'curr_opt_points',
                             'curr_true_opt_vals',
                             'curr_true_opt_points',
                             'num_jobs_per_worker']
    self.data_to_be_saved_if_available = ['query_fidels',
                                          'query_cost_at_fidels',
                                          'query_at_fidel_to_opts',
                                          'prev_eval_points',
                                          'prev_eval_vals']
    self.data_not_to_be_mat_saved.extend(['method_options', 'query_points',
                                          'curr_opt_points', 'curr_true_opt_points'])
    self.data_not_to_be_pickled.extend(['method_options'])
    for data_type in self.data_to_be_saved:
      setattr(self.to_be_saved, data_type, self._get_new_empty_results_array())
    for data_type in self.data_to_be_saved_if_available:
      setattr(self.to_be_saved, data_type, self._get_new_empty_results_array())

  def _get_new_empty_results_array(self):
    """ Returns a new empty arrray to be used for saving results. """
#     return np.empty((self.num_methods, 0), dtype=np.object)
    return np.array([[] for _ in range(self.num_methods)], dtype=np.object)

  def _get_new_iter_results_array(self):
    """ Returns an empty array to be used for saving results of current iteration. """
#     return np.empty((self.num_methods, 1), dtype=np.object)
    return np.array([['-'] for _ in range(self.num_methods)], dtype=np.object)

  def _print_method_header(self, full_method_name):
    """ Prints a header for the current method. """
    trial_header = '-- Exp %d/%d on %s:: %s with cap %0.4f. ----------------------'%( \
      self.trial_iter, self.num_trials, self.study_name, full_method_name, \
      self.max_capital)
    self.reporter.writeln(trial_header)

  def get_iteration_header(self):
    """ Header for iteration. """
    noisy_str = ('no-noise' if self.func_caller.noise_type == 'no_noise' else
                 'noisy(%0.2f)'%(self.func_caller.noise_scale))
    maxval_str = ('?' if self.func_caller.maxval is None
                  else '%0.5f'%(self.func_caller.maxval))
    ret = '%s (M=%d), td: %s, max=%s, max-capital %0.2f, %s'%(self.study_name, \
      self.worker_manager.num_workers, self.to_be_saved.time_distro_str, maxval_str, \
      self.max_capital, noisy_str)
    return ret

  def _print_method_result(self, method, comp_opt_val, num_evals):
    """ Prints the result for this method. """
    result_str = 'Method: %s achieved max-val %0.5f in %d evaluations.\n'%(method, \
                  comp_opt_val, num_evals)
    self.reporter.writeln(result_str)

  def _get_prev_eval_qinfos(self):
    """ Gets the initial qinfos for all methods. """
    if self.evaluation_options.prev_eval_points.lower() == 'generate':
      init_pool_qinfos = self._get_initial_pool_qinfos()
    elif self.evaluation_options.prev_eval_points.lower() == 'none':
      return None
    else:
      # Load from the file.
      raise NotImplementedError('Not written reading results from file yet.')
    # Create an initialiser
    initialiser = OptInitialiser(self.func_caller, self.worker_manager)
    initialiser.options.get_initial_qinfos = lambda _: init_pool_qinfos
    initialiser.options.max_num_steps = 0
    _, _, init_hist = initialiser.initialise()
    return init_hist.query_qinfos

  def _get_initial_pool_qinfos(self):
    """ Returns the intial pool to bootstrap methods in one evaluation. """
    raise NotImplementedError('Implement in a child class.')

  def _optimise_with_method_on_func_caller(self, method, func_caller, worker_manager,
                                           max_capital, meth_options, reporter):
    """ Run method on the function caller and return. """
    raise NotImplementedError('Implement in a child class!')

  def run_trial_iteration(self):
    """ Runs each method in self.methods once and stores the results to be saved. """
    curr_iter_results = Namespace()
    for data_type in self.data_to_be_saved:
      setattr(curr_iter_results, data_type, self._get_new_iter_results_array())
    for data_type in self.data_to_be_saved_if_available:
      setattr(curr_iter_results, data_type, self._get_new_iter_results_array())

    # Fetch pre-evaluation points.
    self.worker_manager.reset()
    prev_eval_qinfos = self._get_prev_eval_qinfos()
    if prev_eval_qinfos is not None:
      prev_eval_vals = [qinfo.val for qinfo in prev_eval_qinfos]
      self.reporter.writeln('Using %d pre-eval points with values. eval: %s (%0.4f).'%( \
                            len(prev_eval_qinfos), prev_eval_vals, max(prev_eval_vals)))
    else:
      self.reporter.writeln('Not using any pre-eval points.')


    # Will go through each method in this loop.
    for meth_iter in range(self.num_methods):
      curr_method = self.methods[meth_iter]
      curr_meth_options = self.method_options[curr_method]
      # Set prev_eval points and vals
      if prev_eval_qinfos is not None:
        curr_meth_options.prev_evaluations = Namespace(qinfos=prev_eval_qinfos)
      else:
        curr_meth_options.prev_evaluations = None
      # Reset worker manager
      self.worker_manager.reset()
      self.reporter.writeln( \
        '\nResetting worker manager: worker_manager.experiment_designer:%s'%( \
        str(self.worker_manager.experiment_designer)))

      # Call the method here.
      self._print_method_header(curr_method)
      history = self._optimise_with_method_on_func_caller(curr_method, self.func_caller, \
                  self.worker_manager, self.max_capital, curr_meth_options, self.reporter)

      # Now save results for current method
      for data_type in self.data_to_be_saved:
        data = getattr(history, data_type)
        data_pointer = getattr(curr_iter_results, data_type)
        data_pointer[meth_iter, 0] = data
      for data_type in self.data_to_be_saved_if_available:
        if hasattr(history, data_type):
          data = getattr(history, data_type)
        else:
          data = ['xx'] * len(history.query_points)
        data_pointer = getattr(curr_iter_results, data_type)
        data_pointer[meth_iter, 0] = data
      # Print out results
      comp_opt_val = history.curr_true_opt_vals[-1]
      num_evals = len(history.curr_true_opt_vals)
      self._print_method_result(curr_method, comp_opt_val, num_evals)
      # Save results of current iteration
      self.update_to_be_saved(curr_iter_results)
      self.save_pickle()
      self.save_results()
    # Save here
    self.update_to_be_saved(curr_iter_results)
    self.save_pickle()
    # No need to explicitly save_results() here - it is done by the parent class.

  def update_to_be_saved(self, curr_iter_results):
    """ Updates the results of the data to be saved with curr_iter_results."""
    for data_type in self.data_to_be_saved + self.data_to_be_saved_if_available:
      data = getattr(curr_iter_results, data_type)
      curr_data_to_be_saved = getattr(self.to_be_saved, data_type)
      if curr_data_to_be_saved.shape[1] == self.trial_iter:
        updated_data_to_be_saved = curr_data_to_be_saved
        updated_data_to_be_saved[:, -1] = data.ravel()
      elif curr_data_to_be_saved.shape[1] < self.trial_iter:
        updated_data_to_be_saved = np.concatenate((curr_data_to_be_saved, data), axis=1)
      else:
        raise ValueError('Something wrong with data saving.')
      setattr(self.to_be_saved, data_type, updated_data_to_be_saved)

