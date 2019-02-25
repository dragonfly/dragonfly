"""
  Unit tests for Multi-fidelity GP Bandits on Cartesian Product domains.
  -- kandasamy@cs.cmu.edu
"""
# Writing it as a separate file (unlike unittest_cp_random_optimiser.py) since these tests
# can be quite expensive.

import unittest

# Local imports
from . import gp_bandit
from .unittest_cp_random_optimiser import MFCPOptimiserBaseTestCase
from ..utils.base_test_class import BaseTestClass, execute_tests


class MFCPGPBanditTestCaseDefinitions(object):
  """ Definitions for unit tests for GP Bandits on cartesian product spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    return gp_bandit.CPGPBandit(func_caller, worker_manager, options=options,
                                reporter=reporter)

  @classmethod
  def _run_optimiser(cls, prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Run the optimiser. """
    return gp_bandit.mf_cp_gpb_from_raw_args(prob_funcs[0], prob_funcs[1],
             domain_config_file, worker_manager=worker_manager, max_capital=max_capital,
             is_mf=True, mode=mode, *args, **kwargs)


@unittest.skip
class MFCPGPBanditTestCase(MFCPGPBanditTestCaseDefinitions,
                           MFCPOptimiserBaseTestCase,
                           BaseTestClass):
  """ Unit tests for GP Bandits on cartesian product spaces. """
  pass


if __name__ == '__main__':
  execute_tests()

