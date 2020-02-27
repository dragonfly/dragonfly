"""
  Unit tests for Random optimiser on Cartesian Product Domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=unused-argument

import os
import unittest
# Local imports
try:
  from ..test_data.park1_3.park1_3 import park1_3
  from ..test_data.park1_3.park1_3_mf import park1_3_mf
  from ..test_data.park1_3.park1_3_mf import cost as cost_park1_3_mf
  from ..test_data.park2_4.park2_4 import park2_4
  from ..test_data.park2_4.park2_4_mf import park2_4_mf
  from ..test_data.park2_4.park2_4_mf import cost as cost_park2_4_mf
  from ..exd.cp_domain_utils import get_processed_func_from_raw_func_for_cp_domain, \
                                  get_raw_point_from_processed_point, \
                                  load_cp_domain_from_config_file, \
                                  load_config_file, load_config
  from ..exd.domains import CartesianProductDomain, EuclideanDomain, IntegralDomain
  from ..exd.experiment_caller import CPFunctionCaller, \
                                      get_multifunction_caller_from_config
  from ..exd.worker_manager import SyntheticWorkerManager
  from ..apis.opt import maximise_function
  from . import random_optimiser
  from ..utils.ancillary_utils import is_nondecreasing, get_list_of_floats_as_str
  from ..utils.base_test_class import BaseTestClass, execute_tests
  from ..utils.reporters import get_reporter
  RUN_TESTS = True
except ImportError:
  RUN_TESTS = False


@unittest.skipIf(not RUN_TESTS, "Unable to import submodules")
class CPOptimiserBaseTestCase(object):
  """ Base test class for Optimisers on Cartesian Product Spaces. """
  # pylint: disable=no-member

  def setUp(self):
    """ Set up. """
    self.max_capital = 30
    self._child_set_up()
    self.worker_manager_1 = SyntheticWorkerManager(1, time_distro='const')
    self.worker_manager_3 = SyntheticWorkerManager(3, time_distro='halfnormal')
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_pardir = os.path.dirname(file_dir)
    self.opt_problems = [
      (test_data_pardir + '/test_data/park1_3/config.json', (park1_3,)),
      (test_data_pardir + '/test_data/park2_4/config.json', (park2_4,)),
      ]

  def _child_set_up(self):
    """ Child set up. """
    pass

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _run_optimiser(cls, raw_func, domain_config_file, worker_manager, max_capital, mode,
                     *args, **kwargs):
    """ Run the optimiser from given args. """
    raise NotImplementedError('Implement in a child class.')

  def test_instantiation(self):
    """ Tests instantiation of the optimiser. """
    self.report('Testing instantiation of optimiser.')
    for idx, (dcf, raw_prob_funcs) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing instantiation of optimiser for %s.'%(
                  idx + 1, len(self.opt_problems), dcf), 'test_result')
      config = load_config_file(dcf)
      if hasattr(config, 'fidel_space'):
        raw_func, raw_fidel_cost_func = raw_prob_funcs
      else:
        raw_func = raw_prob_funcs[0]
        raw_fidel_cost_func = None
      func_caller = get_multifunction_caller_from_config(raw_func, config,
                                  raw_fidel_cost_func=raw_fidel_cost_func)
      optimiser = self._child_instantiate_optimiser(func_caller, self.worker_manager_1,
                    options=None, reporter=get_reporter('silent'))
      self.report('Instantiated %s object.'%(type(optimiser)))
      for attr in dir(optimiser):
        if not attr.startswith('_'):
          self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))),
                      'test_result')

  def _test_optimiser_results(self, opt_val, opt_point, history, dcf):
    """ Tests optimiser results. """
    cp_domain, orderings = load_cp_domain_from_config_file(dcf)
    raw_opt_point = get_raw_point_from_processed_point(opt_point,
                        cp_domain, orderings.index_ordering, orderings.dim_ordering)
    self.report('Opt point: proc=%s, raw=%s.'%(opt_point, raw_opt_point))
    self.report('True opt val sequence: %s.'%(
      get_list_of_floats_as_str(history.curr_true_opt_vals)))
    saved_in_history = [key for key, _ in list(history.__dict__.items()) if not
                        key.startswith('__')]
    self.report('Stored in history: %s.'%(saved_in_history), 'test_result')
    assert opt_val == history.curr_opt_vals[-1]
    assert len(opt_point) == cp_domain.num_domains
    assert is_nondecreasing(history.curr_opt_vals)
    assert is_nondecreasing(history.curr_true_opt_vals)

  def test_optimisation_single(self):
    """ Tests running of the optimiser. """
    self.report('')
    self.report('Testing optimisation on a single worker.')
    for idx, (dcf, raw_prob_funcs) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing optimisation  with 1 worker on %s.'%(
                  idx + 1, len(self.opt_problems), dcf), 'test_result')
      self.worker_manager_1.reset()
      opt_val, opt_point, history = self._run_optimiser(raw_prob_funcs, dcf,
                                      self.worker_manager_1, self.max_capital, 'asy')
      self._test_optimiser_results(opt_val, opt_point, history, dcf)
      self.report('')

  def test_optimisation_asynchronous(self):
    """ Tests running the optimiser in asynchronous mode. """
    self.report('')
    self.report('Testing optimisation on three asynchronous workers.')
    for idx, (dcf, raw_prob_funcs) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing optimisation with 3 asynchronous workers on %s.'%(
                   idx+1, len(self.opt_problems), dcf), 'test_result')
      self.worker_manager_3.reset()
      opt_val, opt_point, history = self._run_optimiser(raw_prob_funcs, dcf,
                              self.worker_manager_3, self.max_capital, 'asy')
      self._test_optimiser_results(opt_val, opt_point, history, dcf)
      self.report('')

  def test_optimisation_synchronous(self):
    """ Tests running the optimiser in asynchronous mode. """
    self.report('')
    self.report('Testing optimisation on three synchronous workers.')
    for idx, (dcf, raw_prob_funcs) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing optimisation with 3 synchronous workers on %s.'%(
                   idx+1, len(self.opt_problems), dcf), 'test_result')
      self.worker_manager_3.reset()
      opt_val, opt_point, history = self._run_optimiser(raw_prob_funcs, dcf,
                              self.worker_manager_3, self.max_capital, 'syn')
      self._test_optimiser_results(opt_val, opt_point, history, dcf)
      self.report('')

@unittest.skipIf(not RUN_TESTS, "Unable to import submodules")
class CPRandomOptimiserAskTellTestCase(CPOptimiserBaseTestCase, BaseTestClass):
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
    """ Testing CP Random optimiser with ask tell interface. """
    self.report('Testing %s using the ask-tell interface.'%(type(self)))
    
    domain, orderings = load_cp_domain_from_config_file('dragonfly/test_data/park1_3/config.json')
    def evaluate(x):
      return park1_3(x)

    func_caller = CPFunctionCaller(None, domain, domain_orderings=orderings)
    opt = random_optimiser.CPRandomOptimiser(func_caller, ask_tell_mode=True)
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
    val, pt, _ = maximise_function(evaluate, domain, 20, opt_method='rand', config=config)
    self.report("Optimal Value: %s, Optimal Point: %s"%(val, pt))


@unittest.skipIf(not RUN_TESTS, "Unable to import submodules")
class MFCPOptimiserBaseTestCase(CPOptimiserBaseTestCase):
  """ Base test class for Multi-fidelity Optimisers on Cartesian Product Spaces. """
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
      (test_data_pardir + '/test_data/park1_3/config_mf.json',
                                (park1_3_mf, cost_park1_3_mf)),
      (test_data_pardir + '/test_data/park2_4/config_mf.json',
                                (park2_4_mf, cost_park2_4_mf)),
      ]

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _run_optimiser(cls, raw_func, domain_config_file, worker_manager, max_capital, mode,
                     *args, **kwargs):
    """ Run the optimiser from given args. """
    raise NotImplementedError('Implement in a child class.')


@unittest.skipIf(not RUN_TESTS, "Unable to import submodules")
class CPRandomOptimiserTestCaseDefinitions(object):
  """ Define Unit tests for the Random optimiser on cartesian product spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    return random_optimiser.CPRandomOptimiser(func_caller, worker_manager,
                                              options=options, reporter=reporter)

  @classmethod
  def _run_optimiser(cls, raw_prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Run the optimiser from given args. """
    raw_func = raw_prob_funcs[0]
    cp_dom, orderings = load_cp_domain_from_config_file(domain_config_file)
    proc_func = get_processed_func_from_raw_func_for_cp_domain(
                  raw_func, cp_dom, orderings.index_ordering, orderings.dim_ordering)
    func_caller = CPFunctionCaller(proc_func, cp_dom, raw_func=raw_func,
                                   domain_orderings=orderings)
    return random_optimiser.random_optimiser_from_func_caller(func_caller, worker_manager,
                                                       max_capital, mode, *args, **kwargs)


@unittest.skipIf(not RUN_TESTS, "Unable to import submodules")
class CPRandomOptimiserTestCase(CPRandomOptimiserTestCaseDefinitions,
                                CPOptimiserBaseTestCase,
                                BaseTestClass):
  """ Unit tests for the Random optimiser on cartesian product spaces. """
  pass


@unittest.skipIf(not RUN_TESTS, "Unable to import submodules")
class MFCPRandomOptimiserTestCaseDefinitions(object):
  """ Unit tests for Multi-fidelity random optimiser on cartesian product spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    return random_optimiser.MFCPRandomOptimiser(func_caller, worker_manager,
                                                options=options, reporter=reporter)

  @classmethod
  def _run_optimiser(cls, raw_prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Run the optimiser from given args. """
    raw_func, raw_fidel_cost_func = raw_prob_funcs
    config = load_config_file(domain_config_file)
    func_caller = get_multifunction_caller_from_config(raw_func, config,
                                raw_fidel_cost_func=raw_fidel_cost_func)
    return random_optimiser.mf_random_optimiser_from_func_caller(func_caller,
             worker_manager, max_capital, mode)


@unittest.skipIf(not RUN_TESTS, "Unable to import submodules")
class MFCPRandomOptimiserTestCase(MFCPRandomOptimiserTestCaseDefinitions,
                                  MFCPOptimiserBaseTestCase,
                                  BaseTestClass):
  """ Unit tests for Multi-fidelity random optimiser on cartesian product spaces. """
  pass


if __name__ == '__main__':
  execute_tests()

