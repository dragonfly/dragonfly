"""
  Unit tests for the GA Optimiser on Cartesian spaces.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

# Local
from . import cp_ga_optimiser
from ..apis.opt import maximise_function
from ..exd import domains
from ..exd.cp_domain_utils import load_cp_domain_from_config_file, load_config_file
from ..exd.domains import CartesianProductDomain, EuclideanDomain, IntegralDomain
from ..exd.experiment_caller import CPFunctionCaller
from ..test_data.park1_3.park1_3 import park1_3
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
    """ Testing GA optimiser with ask tell interface. """
    self.report('Testing %s using the ask-tell interface.'%(type(self)))

    domain, orderings = load_cp_domain_from_config_file('dragonfly/test_data/park1_3/config.json')
    def evaluate(x):
      return park1_3(x)

    func_caller = CPFunctionCaller(None, domain, domain_orderings=orderings)
    opt = cp_ga_optimiser.CPGAOptimiser(func_caller, ask_tell_mode=True)
    opt.initialise()

    best_x, best_y = None, float('-inf')
    for _ in range(60):
      x = opt.ask()
      y = evaluate(x)
      opt.tell([(x, y)])
      self.report('x: %s, y: %s'%(x, y))
      if y > best_y:
        best_x, best_y = x, y
    self.report("-----------------------------------------------------")
    self.report("Optimal Value: %s, Optimal Point: %s"%(best_y, best_x))

    self.report("-----------------------------------------------------")
    config = load_config_file('dragonfly/test_data/park1_3/config.json')
    self.report("Regular optimisation using maximise_function")
    val, pt, _ = maximise_function(evaluate, domain, 60, opt_method='ga', config=config)
    self.report("Optimal Value: %s, Optimal Point: %s"%(val, pt))


if __name__ == '__main__':
  execute_tests()

