"""
  Unit tests for random multi-objective optimiser in Euclidean domains.
  -- bparia@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

import numpy as np
# Local imports
from ..exd.experiment_caller import EuclideanMultiFunctionCaller
from ..exd.worker_manager import SyntheticWorkerManager
from . import random_multiobjective_optimiser
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils import reporters


class EuclideanMultiObjectiveOptimiserBaseTestCase(object):
  """ A base test class for all optimisers on Euclidean domains. """
  # pylint: disable=no-member
  # pylint: disable=invalid-name

  def setUp(self):
    """ set up. """
    self.test_max_capital = 20
    self.multi_func_caller = EuclideanMultiFunctionCaller(
        funcs=[lambda x: (x[0] * x[1]), lambda x: (x[0] * np.sqrt(1-x[1]**2))],
        raw_domain=np.asarray([[0.1, 0.9]]*2),
        descr='simple_sphere',
        vectorised=False,
        to_normalise_domain=True,
    )
    self.worker_manager_1 = SyntheticWorkerManager(1, time_distro='const')
    self.worker_manager_3 = SyntheticWorkerManager(3, time_distro='halfnormal')

  @classmethod
  def _child_instantiate_optimiser(cls, multi_func_caller, worker_manager, options,
                                   reporter):
    """ Instantiate a specific optimiser. """
    raise NotImplementedError('Implement in a child test class.')

  @classmethod
  def run_optimiser(cls, multi_func_caller, worker_manager, max_capital, mode,
                    *args, **kwargs):
    """ Runs optimiser. """
    raise NotImplementedError('Implement in a child test class.')

  def test_instantiation(self):
    """ Tests instantiation of the optimiser. """
    optimiser = self._child_instantiate_optimiser(
        self.multi_func_caller, self.worker_manager_3, options=None,
        reporter=reporters.get_reporter('silent'))
    self.report('Instantiated %s object.'%(type(optimiser)))
    for attr in dir(optimiser):
      if not attr.startswith('_'):
        self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))),
                    'test_result')

  def _test_optimiser_results(self, pareto_vals, pareto_points, history):
    """ Tests optimiser results. """
    assert len(history.curr_pareto_vals) == len(history.curr_pareto_points)
    for val in pareto_vals:
      assert len(val) == self.multi_func_caller.num_funcs
    for val in pareto_points:
      assert len(val) == self.multi_func_caller.domain.get_dim()
    self.report('Pareto optimal points: %s.'%(pareto_points))
    self.report('Pareto optimal values: %s.'%(pareto_vals))

  def test_optimisation_single(self):
    """ Test optimisation with a single worker. """
    self.report('Testing %s with one worker.'%(type(self)))
    pareto_vals, pareto_points, history = self.run_optimiser(
        self.multi_func_caller, self.worker_manager_1, self.test_max_capital, 'asy')
    self._test_optimiser_results(pareto_vals, pareto_points, history)
    self.report('')
    return pareto_vals, pareto_points, history

  def test_optimisation_asynchronous(self):
    """ Testing random optimiser with three asynchronous workers. """
    self.report('Testing %s with three asynchronous workers.'%(type(self)))
    pareto_vals, pareto_points, history = self.run_optimiser(
        self.multi_func_caller, self.worker_manager_3, self.test_max_capital, 'asy')
    self._test_optimiser_results(pareto_vals, pareto_points, history)
    self.report('')
    return pareto_vals, pareto_points, history


class EuclideanRandomMultiObjectiveOptimiserTestCase(
  EuclideanMultiObjectiveOptimiserBaseTestCase, BaseTestClass):
  """ Unit-tests for random multi-objective optimisation. """

  @classmethod
  def _child_instantiate_optimiser(cls, multi_func_caller, worker_manager, options,
                                   reporter):
    """ Instantiate optimiser. """
    return random_multiobjective_optimiser.EuclideanRandomMultiObjectiveOptimiser(
             multi_func_caller, worker_manager, options, reporter)

  @classmethod
  def run_optimiser(cls, multi_func_caller, worker_manager, max_capital, mode,
                    *args, **kwargs):
    """ Runs multi-objective optimiser. """
    rmoo = random_multiobjective_optimiser
    return rmoo.random_multiobjective_optimisation_from_multi_func_caller(
             multi_func_caller, worker_manager, max_capital, mode, *args, **kwargs)


if __name__ == '__main__':
  execute_tests()

