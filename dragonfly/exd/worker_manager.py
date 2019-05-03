"""
  A manager for multiple workers.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import print_function
from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used
# pylint: disable=abstract-class-little-used

from argparse import Namespace
from multiprocessing import Process
import numpy as np
import os
import pickle
import shutil
import time
try:
  from sets import Set
except ImportError:
  Set = set

# Local
from .exd_utils import EVAL_ERROR_CODE

_TIME_TOL = 1e-5


class AbstractWorkerManager(object):
  """ A Base class for a worker manager. """

  def __init__(self, worker_ids):
    """ Constructor. """
    if hasattr(worker_ids, '__iter__'):
      self.worker_ids = worker_ids
    else:
      self.worker_ids = list(range(worker_ids))
    self.num_workers = len(self.worker_ids)
    # These will be set in reset
    self.experiment_designer = None
    self.latest_results = None
    # Reset
    self.reset()

  def reset(self):
    """ Resets everything. """
    self.experiment_designer = None
    self.latest_results = [] # A list of namespaces
    self._child_reset()

  def _child_reset(self):
    """ Child reset. """
    raise NotImplementedError('Implement in a child class.')

  def fetch_latest_results(self):
    """ Returns the latest results. """
    ret_idxs = []
    for i in range(len(self.latest_results)):
      if (self.latest_results[i].receive_time <=
            self.experiment_designer.get_curr_spent_capital() + _TIME_TOL):
        ret_idxs.append(i)
    keep_idxs = [i for i in range(len(self.latest_results)) if i not in ret_idxs]
    ret = [self.latest_results[i] for i in ret_idxs]
    self.latest_results = [self.latest_results[i] for i in keep_idxs]
    return ret

  def close_all_queries(self):
    """ Closes all queries. """
    raise NotImplementedError('Implement in a child class.')

  def set_experiment_designer(self, experiment_designer):
    """ Set the experiment designer. """
    self.experiment_designer = experiment_designer

  def a_worker_is_free(self):
    """ Returns true if a worker is free. """
    raise NotImplementedError('Implement in a child class.')

  def all_workers_are_free(self):
    """ Returns true if all workers are free. """
    raise NotImplementedError('Implement in a child class.')

  def _dispatch_experiment(self, func_caller, qinfo, **kwargs):
    """ Dispatches job. """
    raise NotImplementedError('Implement in a child class.')

  def dispatch_single_experiment(self, func_caller, qinfo, **kwargs):
    """ Dispatches job. """
    raise NotImplementedError('Implement in a child class.')

  def dispatch_batch_of_experiments(self, func_caller, qinfos, **kwargs):
    """ Dispatches an entire batch of experiments. """
    raise NotImplementedError('Implement in a child class.')

  def get_time_distro_info(self):
    """ Returns information on the time distribution. """
    #pylint: disable=no-self-use
    return ''

  def get_poll_time_real(self):
    """ Returns the poll time. """
    raise NotImplementedError('Implement in a child class.')


# A synthetic worker manager - for simulating multiple workers ---------------------------
class SyntheticWorkerManager(AbstractWorkerManager):
  """ A Worker manager for synthetic functions. Mostly to be used in simulations. """

  def __init__(self, num_workers, time_distro='caller_eval_cost',
               time_distro_params=None):
    """ Constructor. """
    self.worker_pipe = None
    super(SyntheticWorkerManager, self).__init__(num_workers)
    # Set up the time sampler
    self.time_distro = time_distro
    self.time_distro_params = time_distro_params
    self.time_sampler = None
    self._set_up_time_sampler()

  def _set_up_time_sampler(self):
    """ Set up the sampler for the time random variable. """
    self.time_distro_params = Namespace() if self.time_distro_params is None else \
                              self.time_distro_params
    if self.time_distro == 'caller_eval_cost':
      pass
    elif self.time_distro == 'const':
      if not hasattr(self.time_distro_params, 'const_val'):
        self.time_distro_params.const_val = 1
      self.time_sampler = lambda num_samples: (np.ones((num_samples,)) *
                                               self.time_distro_params.const_val)
    elif self.time_distro == 'uniform':
      if not hasattr(self.time_distro_params, 'ub'):
        self.time_distro_params.ub = 2.0
        self.time_distro_params.lb = 0.0
      ub = self.time_distro_params.ub
      lb = self.time_distro_params.lb
      self.time_sampler = lambda num_samples: (np.random.random((num_samples,)) *
                                               (ub - lb) + lb)
    elif self.time_distro == 'halfnormal':
      if not hasattr(self.time_distro_params, 'ub'):
        self.time_distro_params.sigma = np.sqrt(np.pi/2)
      self.time_sampler = lambda num_samples: np.abs(np.random.normal(
        scale=self.time_distro_params.sigma, size=(num_samples,)))
    else:
      raise NotImplementedError('Not implemented time_distro = %s yet.'%(
                                self.time_distro))

  def _child_reset(self):
    """ Child reset. """
    self.worker_pipe = [[wid, 0.0] for wid in self.worker_ids]

  def sort_worker_pipe(self):
    """ Sorts worker pipe by finish time. """
    self.worker_pipe.sort(key=lambda x: x[-1])

  def a_worker_is_free(self):
    """ Returns true if a worker is free. """
    return self.worker_pipe[0][-1] # Always return true as this is synthetic.

  def all_workers_are_free(self):
    """ Returns true if all workers are free. """
    return self.worker_pipe[-1][-1]

  def close_all_queries(self):
    """ Close all queries. """
    pass

  def _dispatch_experiment(self, func_caller, qinfo, worker_id, **kwargs):
    """ Dispatch experiment. """
    # Set worker id and whether or not eval_time should be returned
    qinfo.worker_id = worker_id # indicate which worker
    qinfo = func_caller.eval_from_qinfo(qinfo, **kwargs)
    if self.time_distro == 'caller_eval_cost':
      if hasattr(qinfo, 'caller_eval_cost') and qinfo.caller_eval_cost is not None:
        qinfo.eval_time = qinfo.caller_eval_cost
      else:
        qinfo.eval_time = 1.0
    else:
      qinfo.eval_time = float(self.time_sampler(1))
    qinfo.receive_time = qinfo.send_time + qinfo.eval_time
    # Store the result in latest_results
    self.latest_results.append(qinfo)
    return qinfo

  def dispatch_single_experiment(self, func_caller, qinfo, **kwargs):
    """ Dispatch a single experiment. """
    worker_id = self.worker_pipe[0][0]
    qinfo = self._dispatch_experiment(func_caller, qinfo, worker_id, **kwargs)
    # Sort the pipe
    self.worker_pipe[0][-1] = qinfo.receive_time
    self.sort_worker_pipe()

  def dispatch_batch_of_experiments(self, func_caller, qinfos, **kwargs):
    """ Dispatches an entire batch of experiments. """
    assert len(qinfos) == self.num_workers
    for idx in range(self.num_workers):
      qinfo = self._dispatch_experiment(func_caller, qinfos[idx],
                                             self.worker_pipe[idx][0], **kwargs)
      self.worker_pipe[idx][-1] = qinfo.receive_time
    self.sort_worker_pipe()

  def get_time_distro_info(self):
    """ Returns information on the time distribution. """
    return self.time_distro

  def get_poll_time_real(self):
    """ Return 0.0 as the poll time. """
    return 0.0


# A worker manager which spawns a new thread for each process ---------------------------
class MultiProcessingWorkerManager(AbstractWorkerManager):
  """ A worker manager which spawns a new thread for each worker. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, worker_ids, tmp_dir,
               poll_time=0.5, sleep_time_after_new_process=0.5):
    """ Constructor. """
    super(MultiProcessingWorkerManager, self).__init__(worker_ids)
    self.poll_time = poll_time
    self.sleep_time_after_new_process = sleep_time_after_new_process
    self.tmp_dir = tmp_dir
    self._rwm_set_up()
    self._child_reset()

  def _rwm_set_up(self):
    """ Sets things up for the child. """
    # Create the result directories. """
    self.result_dir_names = {wid:'%s/result_%s'%(self.tmp_dir, str(wid)) for wid in
                                                 self.worker_ids}
    # Create the working directories
    self.working_dir_names = {wid:'%s/working_%s/tmp'%(self.tmp_dir,
                               str(wid)) for wid in self.worker_ids}
    # Create the last receive times
    self.last_receive_times = {wid:0.0 for wid in self.worker_ids}
    # Create file names
    self._result_file_name = 'result.p'
    self._num_file_read_attempts = 10

  @classmethod
  def _delete_dirs(cls, list_of_dir_names):
    """ Deletes a list of directories. """
    for dir_name in list_of_dir_names:
      if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

  @classmethod
  def _delete_and_create_dirs(cls, list_of_dir_names):
    """ Deletes a list of directories and creates new ones. """
    for dir_name in list_of_dir_names:
      if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
      os.makedirs(dir_name)

  def _child_reset(self):
    """ Resets child. """
    # Delete/create the result and working directories.
    if not hasattr(self, 'result_dir_names'): # Just for the super constructor.
      return
    self._delete_and_create_dirs(list(self.result_dir_names.values()))
    self._delete_dirs(list(self.working_dir_names.values()))
    self.free_workers = Set(self.worker_ids)
    self.func_callers_for_each_worker = {wid:None for wid in self.worker_ids}
    self.qinfos_in_progress = {wid:None for wid in self.worker_ids}
    self.worker_processes = {wid:None for wid in self.worker_ids}

  def _get_result_file_name_for_worker(self, worker_id):
    """ Computes the result file name for the worker. """
    return os.path.join(self.result_dir_names[worker_id], self._result_file_name)

  def _read_result_from_file(self, result_file_name):
    """ Reads the result from the file name. """
    #pylint: disable=bare-except
    num_attempts = 0
    while num_attempts < self._num_file_read_attempts:
      try:
        file_reader = open(result_file_name, 'rb')
        result = pickle.load(file_reader)
        break
      except:
        print('Encountered error when reading %s. Trying again.'%(result_file_name))
        time.sleep(self.poll_time)
        file_reader.close()
        result = EVAL_ERROR_CODE
    return result

  def _read_result_from_worker_and_update(self, worker_id):
    """ Reads the result from the worker. """
    # pylint: disable=maybe-no-member
    # Read the file
    result_file_name = self._get_result_file_name_for_worker(worker_id)
    result_qinfo = self._read_result_from_file(result_file_name)
    saved_qinfo = self.qinfos_in_progress[worker_id]
    # Now update the relevant qinfo and put it to latest_results
    if isinstance(result_qinfo, Namespace):
      assert self.func_callers_for_each_worker[worker_id].domain.members_are_equal(
               result_qinfo.point, saved_qinfo.point)
      qinfo = result_qinfo
    elif result_qinfo == EVAL_ERROR_CODE:
      qinfo = saved_qinfo
      qinfo.val = EVAL_ERROR_CODE
    else:
      raise ValueError('Could not read qinfo object: %s.'%(str(qinfo)))
    qinfo.receive_time = self.experiment_designer.get_curr_spent_capital()
    qinfo.eval_time = qinfo.receive_time - qinfo.send_time
    if not hasattr(qinfo, 'true_val'):
      qinfo.true_val = qinfo.val
    self.latest_results.append(qinfo)
    # Update receive time
    self.last_receive_times[worker_id] = qinfo.receive_time
    # Delete the file.
    os.remove(result_file_name)
    # Delete content in a working directory.
    shutil.rmtree(self.working_dir_names[worker_id])
    # Add the worker to the list of free workers and clear qinfos in progress.
    self.worker_processes[worker_id].terminate()
    self.worker_processes[worker_id] = None
    self.qinfos_in_progress[worker_id] = None
    self.func_callers_for_each_worker[worker_id] = None
    self.free_workers.add(worker_id)

  def _worker_is_free(self, worker_id):
    """ Checks if worker with worker_id is free. """
    if worker_id in self.free_workers:
      return True
    worker_result_file_name = self._get_result_file_name_for_worker(worker_id)
    if os.path.exists(worker_result_file_name):
      self._read_result_from_worker_and_update(worker_id)
    else:
      return False

  def _get_last_receive_time(self):
    """ Returns the last time we received a job. """
    all_receive_times = list(self.last_receive_times.values())
    return max(all_receive_times)

  def a_worker_is_free(self):
    """ Returns true if a worker is free. """
    for wid in self.worker_ids:
      if self._worker_is_free(wid):
        return self._get_last_receive_time()
    return None

  def all_workers_are_free(self):
    """ Returns true if all workers are free. """
    all_are_free = True
    for wid in self.worker_ids:
      all_are_free = self._worker_is_free(wid) and all_are_free
    if all_are_free:
      return self._get_last_receive_time()
    else:
      return None

  def _dispatch_experiment(self, func_caller, qinfo, worker_id, **kwargs):
    """ Dispatches experiment to worker_id. """
    #pylint: disable=star-args
    if self.qinfos_in_progress[worker_id] is not None:
      err_msg = 'qinfos_in_progress: %s,\nfree_workers: %s.'%(
                   str(self.qinfos_in_progress), str(self.free_workers))
      print(err_msg)
      raise ValueError('Check if worker is free before sending experiment.')
    # First add all the data to qinfo
    qinfo.worker_id = worker_id
    qinfo.working_dir = self.working_dir_names[worker_id]
    qinfo.result_file = self._get_result_file_name_for_worker(worker_id)
    # Create the working directory
    os.makedirs(qinfo.working_dir)
    # Dispatch the experiment in a new process
    target_func = lambda: func_caller.eval_from_qinfo(qinfo, **kwargs)
    self.worker_processes[worker_id] = Process(target=target_func)
    self.worker_processes[worker_id].start()
    time.sleep(self.sleep_time_after_new_process)
    # Add the qinfo to the in progress bar and remove from free_workers
    self.qinfos_in_progress[worker_id] = qinfo
    self.func_callers_for_each_worker[worker_id] = func_caller
    self.free_workers.discard(worker_id)

  def dispatch_single_experiment(self, func_caller, qinfo, **kwargs):
    """ Dispatches a single experiment to a free worker. """
    worker_id = self.free_workers.pop()
    self._dispatch_experiment(func_caller, qinfo, worker_id, **kwargs)

  def dispatch_batch_of_experiments(self, func_caller, qinfos, **kwargs):
    """ Dispatches a batch of experiments. """
    assert len(qinfos) == self.num_workers
    for idx in range(self.num_workers):
      self._dispatch_experiment(func_caller, qinfos[idx], self.worker_ids[idx], **kwargs)

  def close_all_queries(self):
    """ Closes all queries. """
    pass

  def get_time_distro_info(self):
    """ Returns information on the time distribution. """
    return 'realtime'

  def get_poll_time_real(self):
    """ Return 0.0 as the poll time. """
    return self.poll_time


# For legacy purposes ----------------------------------------------------------------
RealWorkerManager = MultiProcessingWorkerManager

