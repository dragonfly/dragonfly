"""
  Unit tests for the GA Optimiser on Cartesian spaces.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

# Local
from . import cp_ga_optimiser
from ..exd import domains
from ..exd.domains import CartesianProductDomain
from ..exd.experiment_caller import CPFunctionCaller
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
    list_of_domains = [
      domains.EuclideanDomain([[0, 2.3], [3.4, 8.9], [0.12, 1.0]]),
      domains.IntegralDomain([[0, 10], [-10, 100], [45, 78.4]]),
    ]
    def evaluate(x):
      return sum(x[0]) + sum(x[1])

    func_caller = CPFunctionCaller(None, CartesianProductDomain(list_of_domains), domain_orderings=None)
    opt = cp_ga_optimiser.CPGAOptimiser(func_caller, ask_tell_mode=True)
    opt.initialise()

    best_x, best_y = None, float('-inf')
    for _ in range(20):
      x = opt.ask()
      y = evaluate(x)
      opt.tell([(x, y)])
      self.report('x: %s, y: %s'%(x, y))
      if y > best_y:
        best_x, best_y = x, y
    self.report("-----------------------------------------------------")
    self.report("Optimal Value: %s, Optimal Point: %s"%(best_y, best_x))


if __name__ == '__main__':
  execute_tests()

