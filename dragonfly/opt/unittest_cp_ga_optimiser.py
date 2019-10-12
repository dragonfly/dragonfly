"""
  Unit tests for the GA Optimiser on Cartesian spaces.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

# Local
from . import cp_ga_optimiser
from .unittest_cp_random_optimiser import CPOptimiserBaseTestCase
from ..utils.base_test_class import BaseTestClass, execute_tests


class CPGAOPtimiserTestCaseDefinitions(object):
  """ Definitions of unit tests for GA optimiser on cartesian product spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    return cp_ga_optimiser.CPGAOptimiser(func_caller, worker_manager, options=options,
                                         reporter=reporter)

  @classmethod
  def _run_optimiser(cls, prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Run the optimiser. """
    return cp_ga_optimiser.cp_ga_optimiser_from_raw_args(prob_funcs[0],
             domain_config_file, worker_manager, max_capital, mode=mode, *args, **kwargs)


class CPGAOPtimiserTestCase(CPGAOPtimiserTestCaseDefinitions,
                            CPOptimiserBaseTestCase,
                            BaseTestClass):
  """ Unit tests for GA optimiser on cartesian product spaces. """
  def test_ask_tell(self):
    """ Testing random optimiser with ask tell interface. """
    self.report('Testing %s using the ask-tell interface.'%(type(self)))
    opt = cp_ga_optimiser.CPGAOptimiser(self.func_caller, self.worker_manager_1)

    def evaluate(x):
      return self.func_caller.eval_single(x)[0] # Get only the value

    for _ in range(100):
      x = opt.ask()
      y = evaluate(x)
      opt.tell(x, y)
      self.report('x: %s, y: %s'%(x, y))


if __name__ == '__main__':
  execute_tests()

