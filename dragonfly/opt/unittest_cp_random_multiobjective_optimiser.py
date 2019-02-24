"""
  Unit tests for Random CP optimiser on Cartesian product domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used


import os
# Local imports
from ..test_data.multiobjective_hartmann.multiobjective_hartmann \
         import objectives as moo_hartmann
from ..test_data.multiobjective_park.multiobjective_park \
         import objectives as moo_park
from ..exd.cp_domain_utils import get_raw_point_from_processed_point, \
                                load_config_file
from ..exd.experiment_caller import get_multifunction_caller_from_config
from ..exd.worker_manager import SyntheticWorkerManager
from . import random_multiobjective_optimiser
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.reporters import get_reporter




class CPMultiObjectiveOptimiserBaseTestCase(object):
  """ Base test class for optimisers on Cartesian product spaces. """
  # pylint: disable=no-member

  def setUp(self):
    """ Set up. """
    self.max_capital = 20
    self._child_set_up()
    self.worker_manager_1 = SyntheticWorkerManager(1, time_distro='const')
    self.worker_manager_3 = SyntheticWorkerManager(3, time_distro='halfnormal')
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_pardir = os.path.dirname(file_dir)
    self.opt_problems = [
      (test_data_pardir + '/test_data/multiobjective_hartmann/config.json',
                                                           (moo_hartmann,)),
      (test_data_pardir + '/test_data/multiobjective_park/config.json', (moo_park,)),
      ]

  def _child_set_up(self):
    """ Child set up. """
    pass

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _run_optimiser(cls, raw_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Run the optimiser from given args. """
    raise NotImplementedError('Implement in a child class.')

  def test_instantiation(self):
    """ Tests instantiation of the optimiser. """
    self.report('Test instantiation of multi-objective optimiser.')
    for idx, (dcf, (raw_prob_funcs, )) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing instantiation of optimiser for %s.'%(
                  idx + 1, len(self.opt_problems), dcf), 'test_result')
      config = load_config_file(dcf)
      multi_func_caller = get_multifunction_caller_from_config(raw_prob_funcs, config)
      optimiser = self._child_instantiate_optimiser(
        multi_func_caller, self.worker_manager_3, options=None,
        reporter=get_reporter('silent'))
      self.report('Instantiated %s object.'%(type(optimiser)))
      for attr in dir(optimiser):
        if not attr.startswith('_'):
          self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))),
                      'test_result')

  def _test_optimiser_results(self, raw_prob_funcs, pareto_vals, pareto_points,
                              history, dcf):
    """ Tests optimiser results. """
    config = load_config_file(dcf)
    multi_func_caller = get_multifunction_caller_from_config(raw_prob_funcs, config)
    raw_pareto_points = [get_raw_point_from_processed_point(pop, config.domain,
                                         config.domain_orderings.index_ordering,
                                         config.domain_orderings.dim_ordering)
                         for pop in pareto_points]
    self.report('Pareto opt point [-1]: proc=%s, raw=%s.'%(pareto_points[-1],
                                                           raw_pareto_points[-1]))
    saved_in_history = [key for key, _ in list(history.__dict__.items()) if not
                        key.startswith('__')]
    self.report('Stored in history: %s.'%(saved_in_history), 'test_result')
    assert len(history.curr_pareto_vals) == len(history.curr_pareto_points)
    for val in pareto_vals:
      assert len(val) == multi_func_caller.num_funcs
    for pt in pareto_points:
      assert len(pt) == config.domain.num_domains
    self.report('Pareto optimal points: %s.'%(pareto_points))
    self.report('Pareto optimal values: %s.'%(pareto_vals))

  def test_optimisation_single(self):
    """ Test optimisation with a single worker. """
    self.report('')
    self.report('Testing %s with one worker.'%(type(self)))
    for idx, (dcf, (raw_prob_funcs, )) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing optimisation  with 1 worker on %s.'%(
                  idx + 1, len(self.opt_problems), dcf), 'test_result')
      self.worker_manager_1.reset()
      pareto_vals, pareto_points, history = self._run_optimiser(raw_prob_funcs, dcf,
        self.worker_manager_1, self.max_capital, 'asy')
      self._test_optimiser_results(raw_prob_funcs, pareto_vals, pareto_points, history,
                                   dcf)
      self.report('')

  def test_optimisation_asynchronous(self):
    """ Testing random optimiser with three asynchronous workers. """
    self.report('')
    self.report('Testing %s with three asynchronous workers.'%(type(self)))
    for idx, (dcf, (raw_prob_funcs, )) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing optimisation  with 3 asynchronous workers on %s.'%(
                  idx + 1, len(self.opt_problems), dcf), 'test_result')
      self.worker_manager_3.reset()
      pareto_vals, pareto_points, history = self._run_optimiser(raw_prob_funcs, dcf,
        self.worker_manager_3, self.max_capital, 'asy')
      self._test_optimiser_results(raw_prob_funcs, pareto_vals, pareto_points, history,
                                   dcf)
      self.report('')


class CPRandomMultiObjectiveOptimiserTestCase(
  CPMultiObjectiveOptimiserBaseTestCase, BaseTestClass):
  """ Unit tests for random multi-objective optimisation. """

  @classmethod
  def _child_instantiate_optimiser(cls, multi_func_caller, worker_manager, options,
                                   reporter):
    """ Instantiate optimiser. """
    return random_multiobjective_optimiser.CPRandomMultiObjectiveOptimiser(
             multi_func_caller, worker_manager, options, reporter)

  @classmethod
  def _run_optimiser(cls, raw_prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Runs multi-objective optimiser. """
    rmoo = random_multiobjective_optimiser
    return rmoo.cp_random_multiobjective_optimisation_from_raw_args(raw_prob_funcs,
             domain_config_file, worker_manager, max_capital, mode, *args, **kwargs)


if __name__ == '__main__':
  execute_tests()

