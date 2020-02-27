"""
  Unit tests for GP Bandits on Cartesian Product domains.
  -- kandasamy@cs.cmu.edu
"""
import unittest

# Local imports
from . import gp_bandit
from ..apis.opt import maximise_function
from ..exd.domains import CartesianProductDomain, EuclideanDomain, IntegralDomain
from ..exd.experiment_caller import CPFunctionCaller
from ..exd.cp_domain_utils import load_config_file, load_cp_domain_from_config_file
from ..test_data.park1_3.park1_3 import park1_3
from .unittest_cp_random_optimiser import CPOptimiserBaseTestCase
from ..utils.base_test_class import BaseTestClass, execute_tests


class CPGPBanditTestCaseDefinitions(object):
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

class CPGPBanditAskTellTestCase(CPOptimiserBaseTestCase, BaseTestClass):
  """ Unit test for the GP Bandit in Euclidean spaces for the ask-tell interface. """
  def test_instantiation(self):
    pass
  
  def test_optimisation_single(self):
    pass

  def test_optimisation_asynchronous(self):
    pass

  def test_optimisation_synchronous(self):
    pass

  def test_ask_tell(self):
    """ Testing CP GP Bandit optimiser with ask tell interface. """
    self.report('Testing %s using the ask-tell interface.'%(type(self)))
    domain, orderings = load_cp_domain_from_config_file('dragonfly/test_data/park1_3/config.json')
    def evaluate(x):
      return park1_3(x)

    func_caller = CPFunctionCaller(None, domain, domain_orderings=orderings)
    opt = gp_bandit.CPGPBandit(func_caller, ask_tell_mode=True)
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

    self.report("-----------------------------------------------------")
    config = load_config_file('dragonfly/test_data/park1_3/config.json')
    self.report("Regular optimisation using maximise_function")
    val, pt, _ = maximise_function(evaluate, domain, 20, opt_method='bo', config=config)
    self.report("Optimal Value: %s, Optimal Point: %s"%(val, pt))


@unittest.skip
class CPGPBanditTestCase(CPGPBanditTestCaseDefinitions,
                         CPOptimiserBaseTestCase,
                         BaseTestClass):
  """ Unit tests for GP Bandits on cartesian product spaces. """
  pass


if __name__ == '__main__':
  execute_tests()

