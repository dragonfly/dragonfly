"""
  Unit tests for GP Bandits on Cartesian Product domains.
  -- kandasamy@cs.cmu.edu
"""


# Local imports
from opt import gp_bandit
from opt.unittest_cp_random_optimiser import CPOptimiserBaseTestCase
from utils.base_test_class import BaseTestClass, execute_tests


class CPGPBanditTestCase(CPOptimiserBaseTestCase, BaseTestClass):
  """ Unit tests for GP Bandits on cartesian product spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    return gp_bandit.CPGPBandit(func_caller, worker_manager, options=options,
                                reporter=reporter)

  @classmethod
  def _run_optimiser(cls, prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Run the optimiser. """
    return gp_bandit.cp_gpb_from_raw_args(prob_funcs[0], domain_config_file,
                                          worker_manager=worker_manager,
                                          max_capital=max_capital, is_mf=False,
                                          mode=mode, *args, **kwargs)


if __name__ == '__main__':
  execute_tests()

