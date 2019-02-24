"""
  Unit tests for multi-objective optimisation using GP Bandits.
  -- kandasamy@cs.cmu.edu
"""
import unittest

from . import multiobjective_gp_bandit
from .unittest_euclidean_random_multiobjective_optimiser import \
  EuclideanMultiObjectiveOptimiserBaseTestCase
from ..utils.base_test_class import BaseTestClass, execute_tests


@unittest.skip
class EuclideanRandomMultiObjectiveOptimiserTestCase(
  EuclideanMultiObjectiveOptimiserBaseTestCase, BaseTestClass):
  """ Unit-tests for random multi-objective optimisation. """

  @classmethod
  def _child_instantiate_optimiser(cls, multi_func_caller, worker_manager, options,
                                   reporter):
    """ Instantiate optimiser. """
    return multiobjective_gp_bandit.EuclideanMultiObjectiveGPBandit(multi_func_caller,
             worker_manager, is_mf=False, options=options, reporter=reporter)

  @classmethod
  def run_optimiser(cls, multi_func_caller, worker_manager, max_capital, mode,
                    *args, **kwargs):
    """ Runs multi-objective optimiser. """
    return multiobjective_gp_bandit.multiobjective_gpb_from_multi_func_caller(
             multi_func_caller, worker_manager, max_capital, is_mf=False,
             mode=mode, *args, **kwargs)


if __name__ == '__main__':
  execute_tests()

