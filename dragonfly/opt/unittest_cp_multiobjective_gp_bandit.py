"""
  Unit tests for Multi-objective GP Bandits in Cartesian product domains.
  -- kandasamy@cs.cmu.edu
"""
import unittest

from . import multiobjective_gp_bandit
from .unittest_cp_random_multiobjective_optimiser import \
     CPMultiObjectiveOptimiserBaseTestCase
from ..utils.base_test_class import BaseTestClass, execute_tests


@unittest.skip
class CPMultiObjectiveGPBanditTestCase(
  CPMultiObjectiveOptimiserBaseTestCase, BaseTestClass):
  """ Unit-tests for multi-objective optimisation with GP Bandits. """

  @classmethod
  def _child_instantiate_optimiser(cls, multi_func_caller, worker_manager, options,
                                   reporter):
    """ Instantiate optimiser. """
    return multiobjective_gp_bandit.CPMultiObjectiveGPBandit(multi_func_caller,
             worker_manager, is_mf=False, options=options, reporter=reporter)

  @classmethod
  def _run_optimiser(cls, raw_prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Runs multi-objective optimiser. """
    return multiobjective_gp_bandit.cp_multiobjective_gpb_from_raw_args(
             raw_prob_funcs, domain_config_file, worker_manager, max_capital, is_mf=False,
             mode=mode, *args, **kwargs)


if __name__ == '__main__':
  execute_tests()

