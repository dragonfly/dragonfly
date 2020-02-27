"""
  Harness for Blackbox Experiment Design. Implements a parent class that can be inherited
  by all methods for black-box optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=super-on-old-class
# pylint: disable=invalid-name
# pylint: disable=redefined-builtin

from __future__ import division
from argparse import Namespace
import numpy as np
import os
import time
# Local imports
from .exd_utils import EVAL_ERROR_CODE
from .exd_utils import postprocess_data_to_save_for_domain, \
                       preprocess_loaded_data_for_domain
from ..utils.option_handler import get_option_specs
from ..utils.reporters import get_reporter

exd_core_args = [
  get_option_specs('max_num_steps', False, 1e7,
    'If exceeds this many evaluations, stop.'),
  get_option_specs('capital_type', False, 'return_value',
    'Should be one of return_value, cputime, or realtime'),
  get_option_specs('mode', False, 'asy',
    'If \'syn\', uses synchronous parallelisation, else asynchronous.'),
  get_option_specs('build_new_model_every', False, 17,
    'Updates the model via a suitable procedure every this many iterations.'),
  get_option_specs('report_model_on_each_build', False, 0,
    'If True, will report the model each time it is built.'),
  get_option_specs('report_results_every', False, 13,
    'Report results every this many iterations.'),
  # Initialisation
  get_option_specs('init_capital', False, 'default',
    ('The capital to be used for initialisation.')),
  get_option_specs('init_capital_frac', False, None,
    'The fraction of the total capital to be used for initialisation.'),
  get_option_specs('num_init_evals', False, 20,
    'The number of evaluations for initialisation. If <0, will use default.'),
  # The amount of effort we will use for initialisation is prioritised by init_capital,
  # init_capital_frac and num_init_evals.
  get_option_specs('prev_evaluations', False, None,
    'Data for any previous evaluations.'),
  get_option_specs('get_initial_qinfos', False, None,
    'A function to obtain initial qinfos.'),
  get_option_specs('init_method', False, 'rand',
    'Method to obtain initial queries. Is used if get_initial_qinfos is None.'),
  # Save and load results
  get_option_specs('progress_load_from_and_save_to', False, None,
    ('Load progress (from possibly a previous run) from and save results to this file. ' +
     'Overrides the progress_save_to and progress_load_from options.')),
  get_option_specs('progress_load_from', False, None,
    'Load progress (from possibly a previous run) from this file.'),
  get_option_specs('progress_save_to', False, None,
    'Save progress to this file.'),
  get_option_specs('progress_save_every', False, 5,
    'Save progress to progress_save_to every progress_save_every iterations.'),
  get_option_specs('progress_report_on_each_save', False, True,
    'If true, will report each time results are saved.'),
  ]


mf_exd_args = [
  get_option_specs('fidel_init_method', False, 'rand',
    'Method to obtain initial fidels. Is used if get_initial_qinfos is None.'),
  get_option_specs('init_set_to_fidel_to_opt_with_prob', False, 0.25,
    'Method to obtain initial fidels. Is used if get_initial_qinfos is None.'),
  ]


class ExperimentDesigner(object):
  """ BlackboxExperimenter Class. """
  #pylint: disable=attribute-defined-outside-init
  #pylint: disable=too-many-instance-attributes

  # Methods needed for construction -------------------------------------------------
  def __init__(self, experiment_caller, worker_manager=None, model=None,
               options=None, reporter=None, ask_tell_mode=False):
    """ Constructor.
        experiment_caller is an ExperimentCaller instance.
        worker_manager is a WorkerManager instance.
    """
    # Set up attributes.
    self.experiment_caller = experiment_caller
    self.domain = experiment_caller.domain
    self.worker_manager = worker_manager
    self.options = options
    self.reporter = get_reporter(reporter)
    self.model = model
    self.ask_tell_mode = ask_tell_mode
    # Other set up
    self._set_up()

  def _set_up(self):
    """ Some additional set up routines. """
    # Set up some book keeping parameters
    self.available_capital = 0.0
    self.num_completed_evals = 0
    self.step_idx = 0
    self.num_succ_queries = 0
    # Initialise worker manager
    if not self.ask_tell_mode:
      self.worker_manager.set_experiment_designer(self)
      copyable_params_from_worker_manager = ['num_workers']
      for param in copyable_params_from_worker_manager:
        setattr(self, param, getattr(self.worker_manager, param))
    # Other book keeping stuff
    self.last_report_at = 0
    self.last_model_build_at = 0
    self.eval_points_in_progress = []
    self.eval_idxs_in_progress = []
    # Set initial history
    # query infos will maintain a list of namespaces which contain information about
    # the query in send order. Everything else will be saved in receive order.
    self.history = Namespace(query_step_idxs=[],
                             query_points=[],
                             query_send_times=[],
                             query_receive_times=[],
                             query_eval_times=[],
                             query_worker_ids=[],
                             query_qinfos=[],
                            )
    if not self.ask_tell_mode:
      self.history.job_idxs_of_workers = {k: [] for k in self.worker_manager.worker_ids}
    self.to_copy_from_qinfo_to_history = {
        'step_idx': 'query_step_idxs',
        'point': 'query_points',
        'send_time': 'query_send_times',
        'receive_time': 'query_receive_times',
        'eval_time': 'query_eval_times',
        'worker_id': 'query_worker_ids',
      }
    # Set pre_eval_points and results
    self.prev_eval_points = []
    self.history.prev_eval_points = self.prev_eval_points
    # Multi-fidelity Set up
    if self.is_an_mf_method() or self.experiment_caller.is_mf():
      self._mf_set_up()
    # Call the child set up.
    self._exd_child_set_up()
    # Saving and loading set up
    self._save_and_load_set_up()
    # Post child set up.
    method_prefix = 'asy' if self.is_asynchronous() else 'syn'
    self.full_method_name = method_prefix + '-' + self._get_method_str()
    self.history.full_method_name = self.full_method_name

  def _mf_set_up(self):
    """ Multi-fidelity set up. """
    assert self.experiment_caller.is_mf()
    self.fidel_space = self.experiment_caller.fidel_space
    # Set up history
    self.history.query_fidels = []
    self.history.query_cost_at_fidels = []
    self.to_copy_from_qinfo_to_history['fidel'] = 'query_fidels'
    self.to_copy_from_qinfo_to_history['cost_at_fidel'] = 'query_cost_at_fidels'
    self.eval_fidels_in_progress = []
    # Set up previous evaluations
    self.prev_eval_fidels = []

  def _save_and_load_set_up(self):
    """ Saving and loading set up. """
    # First set up file names ------------------------------------------------------------
    if self.options.progress_load_from_and_save_to:
      if isinstance(self.options.progress_load_from_and_save_to, str):
        lfast = [self.options.progress_load_from_and_save_to]
      else:
        lfast = self.options.progress_load_from_and_save_to
      lfast = [elem for elem in lfast if os.path.exists(elem)]
      if len(lfast) > 0:
        load_from = lfast
      else:
        load_from = None
      save_to = self.options.progress_load_from_and_save_to
    else:
      load_from = None
      save_to = None
      if self.options.progress_load_from:
        load_from = self.options.progress_load_from
      if self.options.progress_save_to:
        save_to = self.options.progress_save_to
    # save file names in progress_io_params
    if isinstance(load_from, str):
      load_from = [load_from]
    if isinstance(save_to, (list, tuple)):
      save_to = save_to[0]
    self.progress_io_params = Namespace(load_from=load_from, save_to=save_to)
    self.last_progress_saved_at = 0

  def _exd_child_set_up(self):
    """ Set up for a child class of Blackbox Experiment designer. """
    raise NotImplementedError('Implement in a child class of ExperimentDesigner.')

  def _get_method_str(self):
    """ Return a string describing the method. """
    raise NotImplementedError('Implement in a child class of ExperimentDesigner.')

  def _get_problem_str(self):
    """ Return a string describing the method. """
    raise NotImplementedError('Implement in a child class of ExperimentDesigner.')

  def is_asynchronous(self):
    """ Returns true if asynchronous."""
    return self.options.mode.lower().startswith('asy')

  def is_an_mf_method(self):
    """ Returns True if the method is a multi-fidelity method. """
    raise NotImplementedError('Implement in a method implementation.')

  # Book-keeping -------------------------------------------------------------
  def _update_history(self, qinfo):
    """ qinfo is a namespace information about the query. """
    # Update the number of jobs done by each worker regardless
    if not self.ask_tell_mode:
      self.history.job_idxs_of_workers[qinfo.worker_id].append(qinfo.step_idx)
    self.history.query_qinfos.append(qinfo) # Save the qinfo
    # Now store in history
    for qinfo_name, history_name in self.to_copy_from_qinfo_to_history.items():
      attr_list = getattr(self.history, history_name)
      try:
        attr_list.append(getattr(qinfo, qinfo_name))
      except AttributeError:
        attr_list.append('xxx')
    # Do any child update
    self._exd_child_update_history(qinfo)
    # Check for unsuccessful queries
    if qinfo.val != EVAL_ERROR_CODE:
      self.num_succ_queries += 1

  def _exd_child_update_history(self, qinfo):
    """ Any updates to history from the child class. """
    raise NotImplementedError('Implement in a child class of BlackboxExperimenter.')

  def _get_jobs_for_each_worker(self):
    """ Returns the number of jobs for each worker as a list. """
    jobs_each_worker = [len(elem) for elem in
                        list(self.history.job_idxs_of_workers.values())]
    if self.num_workers <= 5:
      jobs_each_worker_str = str(jobs_each_worker)
    else:
      return '[min:%d, max:%d]'%(min(jobs_each_worker), max(jobs_each_worker))
    return  jobs_each_worker_str

  def _get_curr_job_idxs_in_progress(self):
    """ Returns the current job indices in progress. """
    if self.num_workers <= 4:
      return str(self.eval_idxs_in_progress)
    else:
      total_in_progress = len(self.eval_idxs_in_progress)
      min_idx = (-1 if total_in_progress == 0 else min(self.eval_idxs_in_progress))
      max_idx = (-1 if total_in_progress == 0 else max(self.eval_idxs_in_progress))
      dif = -1 if total_in_progress == 0 else max_idx - min_idx
      return '[min:%d, max:%d, dif:%d, tot:%d]'%(min_idx, max_idx, dif, total_in_progress)

  @classmethod
  def _get_multiple_workers_str(cls):
    """ Get string if there are multiple workers. """
    return ''

  def _print_header(self):
    """ Print header. """
    header_str = 'Legend: <iteration_number> (<num_successful_queries>, ' + \
                 '<fraction_of_capital_spent>):: '
    child_header_str = self._get_exd_child_header_str()
    self.reporter.writeln(header_str + child_header_str)

  @classmethod
  def _get_exd_child_header_str(cls):
    """ Get header for child. """
    return ''

  def _report_curr_results(self):
    """ Writes current result to reporter. """
    cap_frac = (np.nan if self.available_capital <= 0 else
                self.get_curr_spent_capital()/self.available_capital)
    report_str = '#%03d (%03d, %0.3f):: '%(self.step_idx, self.num_succ_queries,
                                           cap_frac)
    report_str += self._get_exd_child_report_results_str()
    report_str += self._get_multiple_workers_str()
    self.reporter.writeln(report_str)
    self.last_report_at = self.step_idx

  def _get_exd_child_report_results_str(self):
    """ Returns a string for the specific child method describing the progress. """
    raise NotImplementedError('Implement in a child class of BlackboxExperimenter.')

  # Methods needed for initialisation ----------------------------------------
  def perform_initial_queries(self):
    """ Perform initial queries. """
    # If we already have some pre_eval points then do this.
    # pylint: disable=too-many-branches
    num_data_loaded_from_file = self._load_prev_evaluations_data_from_file()
    num_data_loaded_from_options = self._handle_prev_evals_in_options()
    if num_data_loaded_from_file + num_data_loaded_from_options > 0:
      # If data is given in a file or in self.options.prev_evaluations then use that as
      # the initial set.
      pass
    else:
      # Set the initial capital
      if self.options.init_capital == 'default':
        self.init_capital = np.clip(5 * self.domain.get_dim(),
                                    max(5.0, 0.025 * self.available_capital),
                                    max(5.0, 0.075 * self.available_capital))
      elif self.options.init_capital is not None:
        self.init_capital = float(self.options.init_capital)
      elif self.options.init_capital_frac is not None:
        self.init_capital = float(self.options.init_capital_frac) * self.available_capital
      else:
        self.init_capital = None
      # Set the function to return the initial qinfos
      if hasattr(self.options, 'get_initial_qinfos') and \
         self.options.get_initial_qinfos is not None:
        get_initial_qinfos = self.options.get_initial_qinfos
      else:
        get_initial_qinfos = self._get_initial_qinfos
      # Send out the initial queries to the worker manager
      if self.init_capital is not None:
        initial_qinfos_w_init_capital = []
        _num_tries_for_init_pool_sampling = 0
        points = 0
        while True:
          if len(initial_qinfos_w_init_capital) == 0:
            if self.options.capital_type == 'return_value':
              num_initial_queries_w_init_capital = int(self.init_capital)
            else:
              num_initial_queries_w_init_capital = int(2 * self.init_capital)
            initial_qinfos_w_init_capital = \
              get_initial_qinfos(num_initial_queries_w_init_capital,
                                 verbose_constraint_satisfaction=False)
          if len(initial_qinfos_w_init_capital) == 0:
            _num_tries_for_init_pool_sampling += 1
            if _num_tries_for_init_pool_sampling % 10 == 0:
              from warnings import warn
              warn(('Sampling an initial pool failed despite %d attempts -- will ' +
                    'continue trying but consider reparametrising your domain if ' +
                    'this problem persists.')%(_num_tries_for_init_pool_sampling))
            continue
          qinfo = initial_qinfos_w_init_capital.pop(0)
          if self.ask_tell_mode:
            self.first_qinfos.append(qinfo)
            points += 1
            if points > self.init_capital:
              break
          else:
            self.step_idx += 1
            self._wait_for_a_free_worker()
            if self._terminate_initialisation():
              cap_frac = (np.nan if self.available_capital <= 0 else
                          self.get_curr_spent_capital()/self.available_capital)
              self.reporter.writeln('Capital spent on initialisation: %0.4f(%0.4f).'%(
                  self.get_curr_spent_capital(), cap_frac))
              break
            self._dispatch_single_experiment_to_worker_manager(qinfo)
      else:
        num_init_evals = int(self.options.num_init_evals)
        if num_init_evals > 0:
          num_init_evals = max(self.num_workers, num_init_evals)
          init_qinfos = get_initial_qinfos(num_init_evals,
                                           verbose_constraint_satisfaction=False)
          for qinfo in init_qinfos:
            if self.ask_tell_mode:
              self.first_qinfos.append(qinfo)
            else:
              self.step_idx += 1
              self._wait_for_a_free_worker()
              self._dispatch_single_experiment_to_worker_manager(qinfo)

  def _load_prev_evaluations_data_from_file(self):
    """ Load data from a prior file. """
    # pylint: disable=broad-except
    # pylint: disable=unexpected-keyword-arg
    def _raise_load_file_exception(_file_name, err_msg):
      """ Raises an error message. """
      raise Exception('Could not pickle load data from %s. Returned error: \n%s'%(
                      _file_name, err_msg))
    if self.progress_io_params.load_from is None:
      return 0
    else:
      ret = 0
      import pickle
      for load_file_name in self.progress_io_params.load_from:
        try:
          load_file_handle = open(load_file_name, 'rb')
        except IOError as e:
          _raise_load_file_exception(load_file_name, e)
        try:
          pickle_loaded_data = pickle.load(load_file_handle)
        except Exception as e1:
          try:
            pickle_loaded_data = pickle.load(load_file_handle, encoding='latin1')
          except Exception as e2:
            _raise_load_file_exception(load_file_name, '%s\n%s'%(e1, e2))
        loaded_data = preprocess_loaded_data_for_domain(pickle_loaded_data,
                                                        self.experiment_caller)
        ret += self._child_handle_data_loaded_from_file(loaded_data)
        self.reporter.writeln('Loaded %d data from files %s.'%(
                              ret, self.progress_io_params.load_from))
      return ret

  def _child_handle_data_loaded_from_file(self, loaded_data_from_file):
    """ A child class handles the data loaded and returns the number of data. """
    raise NotImplementedError('Implement in a child class.')

  def _handle_prev_evals_in_options(self):
    """ Handles previous evaluations. """
    if hasattr(self.options, 'prev_evaluations') and \
       self.options.prev_evaluations is not None:
      ret = self._exd_child_handle_prev_evals_in_options()
    else:
      ret = 0
    if ret > 0:
      self.reporter.writeln('Loaded %d data from self.options.prev_evaluations.')
    return ret

  def _exd_child_handle_prev_evals_in_options(self):
    """ Handles pre-evaluations. """
    raise NotImplementedError('Implement in a child class of BlackboxExperimenter.')

  def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
    """ Returns the initial qinfos. Can be overridden by a child class. """
    # pylint: disable=unused-argument
    # pylint: disable=no-self-use
    return []

  def initialise_capital(self):
    """ Initialises capital. """
    if self.options.capital_type == 'return_value':
      self.spent_capital = 0.0
    elif self.options.capital_type == 'cputime':
      self.init_cpu_time_stamp = time.clock()
    elif self.options.capital_type == 'realtime':
      self.init_real_time_stamp = time.time()

  def get_curr_spent_capital(self):
    """ Computes the current spent time. """
    if self.options.capital_type == 'return_value':
      return self.spent_capital
    elif self.options.capital_type == 'cputime':
      return time.clock() - self.init_cpu_time_stamp
    elif self.options.capital_type == 'realtime':
      return time.time() - self.init_real_time_stamp

  def set_curr_spent_capital(self, spent_capital):
    """ Sets the current spent capital. Useful only in synthetic set ups."""
    if self.options.capital_type == 'return_value':
      self.spent_capital = spent_capital

  def _print_method_description(self):
    """ Prints method description at the beginning of the routine. """
    method_str = self._get_method_str()
    problem_str = self._get_problem_str()
    if self.num_workers > 1:
      parallel_mode = '-asynchronous' if self.is_asynchronous() else '-synchronous'
      method_str += parallel_mode
    self.reporter.writeln('%s with %s using capital %s (%s)'%(problem_str, method_str,
                          str(self.available_capital), self.options.capital_type))

  def run_experiment_initialise(self):
    """ Initialisation before running initialisation. """
    self._print_method_description()
    self.initialise_capital()
    self.perform_initial_queries()
    self._child_run_experiments_initialise()
    self._print_header()

  def _child_run_experiments_initialise(self):
    """ Handles any initialisation before running experiments. """
    raise NotImplementedError('Implement in a child class of BlackboxExperimenter.')

  # Methods needed for querying ----------------------------------------------------
  def _wait_till_free(self, is_free, poll_time):
    """ Waits until is_free returns true. """
    keep_looping = True
    while keep_looping:
      last_receive_time = is_free()
      if last_receive_time is not None:
        # Get the latest set of results and dispatch the next job.
        self.set_curr_spent_capital(last_receive_time)
        latest_results = self.worker_manager.fetch_latest_results()
        for qinfo in latest_results:
          if self.experiment_caller.is_mf() and not hasattr(qinfo, 'cost_at_fidel'):
            qinfo.cost_at_fidel = qinfo.eval_time
          self._update_history(qinfo)
          self._remove_from_in_progress(qinfo)
        self._add_data_to_model(latest_results)
        keep_looping = False
      else:
        time.sleep(poll_time)

  def _wait_for_a_free_worker(self):
    """ Checks if a worker is free and updates with the latest results. """
    self._wait_till_free(self.worker_manager.a_worker_is_free,
                         self.worker_manager.get_poll_time_real())

  def _wait_for_all_free_workers(self):
    """ Checks to see if all workers are free and updates with latest results. """
    self._wait_till_free(self.worker_manager.all_workers_are_free,
                         self.worker_manager.get_poll_time_real())

  def _add_to_in_progress(self, qinfos):
    """ Adds jobs to in progress. """
    for qinfo in qinfos:
      self.eval_idxs_in_progress.append(qinfo.step_idx)
      self.eval_points_in_progress.append(qinfo.point)
      if self.is_an_mf_method():
        self.eval_fidels_in_progress.append(qinfo.fidel)

  def _remove_from_in_progress(self, qinfo):
    """ Removes a job from the in progress status. """
    completed_eval_idx = self.eval_idxs_in_progress.index(qinfo.step_idx)
    self.eval_idxs_in_progress.pop(completed_eval_idx)
    self.eval_points_in_progress.pop(completed_eval_idx)
    if self.is_an_mf_method():
      self.eval_fidels_in_progress.pop(completed_eval_idx)

  def _dispatch_single_experiment_to_worker_manager(self, qinfo):
    """ Dispatches an experiment to the worker manager. """
    # Create a new qinfo namespace and dispatch new job.
    qinfo.send_time = self.get_curr_spent_capital()
    qinfo.step_idx = self.step_idx
    self.worker_manager.dispatch_single_experiment(self.experiment_caller, qinfo)
    self._add_to_in_progress([qinfo])
  
  def _dispatch_single_experiment_ask_tell_mode(self, qinfo):
    """ Dispatches an experiment in ask-tell mode. """
    qinfo.send_time = self.get_curr_spent_capital()
    qinfo.step_idx = self.step_idx
    qinfo.eval_time = 1.0
    qinfo.receive_time = qinfo.send_time + qinfo.eval_time
    self._add_to_in_progress([qinfo])

  def _dispatch_batch_of_experiments_to_worker_manager(self, qinfos):
    """ Dispatches a batch of experiments to the worker manager. """
    for idx, qinfo in enumerate(qinfos):
      qinfo.send_time = self.get_curr_spent_capital()
      qinfo.step_idx = self.step_idx + idx
    self.worker_manager.dispatch_batch_of_experiments(self.func_caller, qinfos)
    self._add_to_in_progress(qinfos)

  # Methods needed for running experiments ---------------------------------------
  def _terminate_now(self):
    """ Returns true if we should terminate now. """
    if self.step_idx >= self.options.max_num_steps:
      self.reporter.writeln('Exceeded %d evaluations. Terminating Now!' %
                            (self.options.max_num_steps))
      return True
    return self.get_curr_spent_capital() >= self.available_capital

  def _terminate_initialisation(self):
    """ Returns true if we should terminate initialization now. """
    return self.get_curr_spent_capital() >= self.init_capital

  def add_capital(self, capital):
    """ Adds capital. """
    self.available_capital += float(capital)

  def _determine_next_query(self):
    """ Determine the next point for evaluation. """
    raise NotImplementedError('Implement in a child class.!')

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determine the next batch of eavluation points. """
    raise NotImplementedError('Implement in a child class.!')

  def _post_process_next_eval_point(self, point):
    """ Post-process the next point for evaluation. By default, returns same point. """
    #pylint: disable=no-self-use
    return point

  def _add_data_to_model(self, qinfos):
    """ Adds data to model. """
    raise NotImplementedError('Implement in a method child class.')

  def _build_new_model(self):
    """ Builds a new model. """
    self.last_model_build_at = self.step_idx
    self._child_build_new_model()
    if self.options.report_model_on_each_build:
      self._report_model()

  def _child_build_new_model(self):
    """ Builds a new model. """
    raise NotImplementedError('Implement in a child class.!')

  def _report_model(self):
    """ Reports the model. Can be overridden by a child class. """
    pass

  def _update_capital(self, qinfos):
    """ Updates the capital according to the cost of the current query. """
    if not hasattr(qinfos, '__iter__'):
      qinfos = [qinfos]
    query_receive_times = []
    for idx in len(qinfos):
      if self.options.capital_type == 'return_value':
        query_receive_times[idx] = qinfos[idx].send_time + qinfos[idx].eval_time
      elif self.options.capital_type == 'cputime':
        query_receive_times[idx] = time.clock() - self.init_cpu_time_stamp
      elif self.options.capital_type == 'realtime':
        query_receive_times[idx] = time.time() - self.init_real_time_stamp
      # Finally add the receive time of the job to qinfo.
      qinfos[idx].receive_time = query_receive_times[idx]
      qinfos[idx].eval_time = qinfos[idx].receive_time - qinfos[idx].send_time
      if qinfos[idx].eval_time < 0:
        raise ValueError(('Something wrong with the timing. send: %0.4f, receive: %0.4f,'
                          + ' eval: %0.4f.') % (qinfos[idx].send_time, \
                                        qinfos[idx].receive_time, qinfos[idx].eval_time))
    # Compute the maximum of all receive times
    max_query_receive_times = max(query_receive_times)
    return max_query_receive_times
  
  # Methods for exposing ask-tell interface
  def initialise(self):
    """ Initialisation for ask-tell interface. """
    self.initialise_capital()
    self.first_qinfos = []
    self.perform_initial_queries()
    self._child_run_experiments_initialise()

  def ask(self, n_points=None):
    """Get recommended point as part of the ask interface.
    Wrapper for _determine_next_query.

    Args:
      n_points (int or None): Represents the number of points to ask for.
      None will prompt ask() to return a single point, whereas an int will 
      prompt ask() to return a list of `n_points` points.
    """
    raise NotImplementedError('Implement in a child class.')

  def tell(self, points):
    """Add data points to be evaluated to return recommendations.
    
    Args:
      points (:obj:`list` of :obj:`tuples`): Contains tuples that hold
        the explanatory variable value(s) of the points as its first value,
        the response variable value(s) as its second value, and a third 
        z-value for multifidelity optimisation.
    """    
    raise NotImplementedError('Implement in a child class.')

  # Methods for saving progress --------------------------------------------------
  def _save_progress_to_file(self):
    """ Saves progress to file. """
    self.last_progress_saved_at = self.step_idx
    if self.progress_io_params.save_to is None:
      return
    data_to_save, num_data = self._exd_child_get_data_to_save()
    data_to_save = postprocess_data_to_save_for_domain(data_to_save,
                                                       self.experiment_caller)
    save_file_handle = open(self.progress_io_params.save_to, 'wb')
    import pickle
    pickle.dump(data_to_save, save_file_handle)
    if self.options.progress_report_on_each_save:
      self.reporter.writeln('Saved %d data to %s.'%(
        num_data, os.path.abspath(self.progress_io_params.save_to)))

  def _exd_child_get_data_to_save(self):
    """ Returns data to be saved as a dictionary. """
    raise NotImplementedError('Implement in a child class.')

  # Methods for running experiments ----------------------------------------------
  def _asynchronous_run_experiment_routine(self):
    """ Optimisation routine for asynchronous part. """
    self._wait_for_a_free_worker()
    qinfo = self._determine_next_query()
    if self.experiment_caller.is_mf() and not hasattr(qinfo, 'fidel'):
      qinfo.fidel = self.experiment_caller.fidel_to_opt
    self._dispatch_single_experiment_to_worker_manager(qinfo)
    self.step_idx += 1

  def _synchronous_run_experiment_routine(self):
    """ Optimisation routine for the synchronous part. """
    self._wait_for_all_free_workers()
    qinfos = self._determine_next_batch_of_queries(self.num_workers)
    self._dispatch_batch_of_experiments_to_worker_manager(qinfos)
    self.step_idx += self.num_workers

  def _run_experiment_wrap_up(self):
    """ Wrap up before concluding running experiments. """
    self.worker_manager.close_all_queries()
    self._wait_for_all_free_workers()
    self._report_curr_results()
    # Store additional data
    self.history.num_jobs_per_worker = np.array(self._get_jobs_for_each_worker())
    # Save results
    self._save_progress_to_file()

  def _main_loop_pre(self):
    """ Anything to be done before each iteration of the main loop. Mostly in case
        this is needed by a child class. """
    pass

  def _main_loop_post(self):
    """ Anything to be done after each iteration of the main loop. Mostly in case
        this is needed by a child class. """
    pass

  # run_experiments method --------------
  def run_experiments(self, max_capital):
    """ This is the main loop which executes the experiments in a loop. """
    self.add_capital(max_capital)
    self.run_experiment_initialise()

    # Main loop --------------------
    while not self._terminate_now():
      # Main loop pre
      self._main_loop_pre()
      if self.step_idx - self.last_report_at >= self.options.report_results_every:
        self._report_curr_results()
      # Experimentation step
      if self.is_asynchronous():
        self._asynchronous_run_experiment_routine()
      else:
        self._synchronous_run_experiment_routine()
      # Book keeeping
      if self.step_idx - self.last_model_build_at >= self.options.build_new_model_every:
        self._build_new_model()
      if self.step_idx - self.last_progress_saved_at >= self.options.progress_save_every:
        self._save_progress_to_file()
      # Main loop post
      self._main_loop_post()

    # Wrap up and return
    self._run_experiment_wrap_up()
    return self._get_final_return_quantities()

  def _get_final_return_quantities(self):
    """ Gets the final quantities to be returned. Can be overriden by a child. """
    return self.history

