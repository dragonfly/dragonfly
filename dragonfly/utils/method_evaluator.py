"""
  Harness to evaluate methods on a given task and save results.
  -- kandasamy@cs.cmu.edu
"""

from argparse import Namespace
import random
from time import time
import pickle
from scipy.io import savemat as sio_savemat
import numpy as np
# Local imports
from .reporters import get_reporter

# Changes to variable names from previous version
# BasicExperimenter -> BaseMethodEvaluator
# experiment_name -> study_name
# num_experiments -> num_trials
# experiment_iter -> trial_iter
# run_experiments -> run_trials
# run_experiment_iteration -> run_trial_iteration

class BaseMethodEvaluator(object):
  """ Base class for evaluating multiple methods for an ED/Optimisation task. """

  def __init__(self, study_name, num_trials, save_file_name,
               save_file_extension='', reporter='default', random_seed_val='time'):
    """ Constructor.
        random_seed_val: If None we will not change the random seed. If it is
          'time' we will set it to a time based value. If it is an int, we will set
          the seed to that value.
    """
    self.study_name = study_name
    self.num_trials = num_trials
    if save_file_extension == '':
      save_file_parts = save_file_name.split('.')
      save_file_name = save_file_parts[0]
      save_file_extension = save_file_parts[1]
    self.save_file_extension = save_file_extension
    self.save_file_full_name = save_file_name + '.' + self.save_file_extension
    self.pickle_file_name = save_file_name + '.p'
    self.reporter = get_reporter(reporter)
    self.to_be_saved = Namespace(study_name=self.study_name)
    self.data_not_to_be_mat_saved = []
    self.data_not_to_be_pickled = []
    # We will need these going forward.
    self.trial_iter = 0
    # Set the random seed
    if random_seed_val is not None:
      if random_seed_val == 'time':
        random_seed_val = int(time() * 100) % 100000
      self.reporter.writeln('Setting random seed to %d.'%(random_seed_val))
      np.random.seed(random_seed_val)
      random.seed(random_seed_val)

  def save_results(self):
    """ Saves results in save_file_full_name. """
    self.reporter.write('Saving results (trial-iter:%d) to %s ...  '%(self.trial_iter, \
                         self.save_file_full_name))
    try:
      if self.save_file_extension == 'mat':
        dict_to_be_saved = vars(self.to_be_saved)
        dict_to_be_mat_saved = {key:val for key, val in dict_to_be_saved.items()
                                if key not in self.data_not_to_be_mat_saved}
        # Fix for crash in single fidelity case -- replacing None with 'xx'.
        for i in range(dict_to_be_mat_saved['query_eval_times'].shape[0]):
          if isinstance(dict_to_be_mat_saved['query_eval_times'][i, -1], str):
            continue
          query_eval_times = [val if val is not None else 'xx' for val in \
                              dict_to_be_mat_saved['query_eval_times'][i, -1]]
          dict_to_be_mat_saved['query_eval_times'][i, -1] = query_eval_times
        sio_savemat(self.save_file_full_name, mdict=dict_to_be_mat_saved)
      else:
        raise NotImplementedError('Only implemented saving mat files so far.')
      save_successful = True
    except IOError:
      save_successful = False
    # Report saving status
    if save_successful:
      self.reporter.writeln('successful.')
    else:
      self.reporter.writeln('unsuccessful!!')

  def save_pickle(self):
    """ Dumps to everything. """
    save_in = open(self.pickle_file_name, 'wb')
    dict_to_be_saved = vars(self.to_be_saved)
    dict_to_be_pickled = {key:val for key, val in dict_to_be_saved.items()
                          if key not in self.data_not_to_be_pickled}
    pickle.dump(dict_to_be_pickled, save_in)
    save_in.close()

  def terminate_now(self):
    """ Returns true if we should terminate now. Can be overridden in a child class. """
    return self.trial_iter >= self.num_trials

  def run_trials(self):
    """ This runs the trial. Each trial executes each method on a problem. """
    self.reporter.writeln(self.get_trial_header())
    while not self.terminate_now():
      # Prelims
      self.trial_iter += 1
      iter_header = ('\nEXP %d/%d:: '%(self.trial_iter, self.num_trials)
                     + self.get_iteration_header())
      iter_header += '\n' + '=' * len(iter_header) + '\n'
      self.reporter.writeln(iter_header)
      # R trial iteration.
      self.run_trial_iteration()
      # Save results
      self.save_results()
    # Wrap up the trials
    self.wrapup_trials()

  def get_trial_header(self):
    """ Something to pring before running all the trials. Can be overridden in a
        child class."""
    # pylint: disable=no-self-use
    return ''

  def get_iteration_header(self):
    """ A header for the particular iteration. """
    # pylint: disable=no-self-use
    return ''

  def run_trial_iteration(self):
    """ Implements the current iteration of the exeperiment. """
    raise NotImplementedError('Implement this in a child class.')

  def wrapup_trials(self):
    """ Any code to wrap up the trials goes here. """
    # pylint: disable=no-self-use
    pass

