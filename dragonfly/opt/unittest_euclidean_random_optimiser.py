"""
  Unit tests for random optimiser.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

import numpy as np
# Local imports
from ..exd import domains
from . import random_optimiser as random_optimiser
from ..exd.experiment_caller import EuclideanFunctionCaller
from ..exd.worker_manager import SyntheticWorkerManager
from ..apis.opt import maximise_function, maximise_multifidelity_function
from ..utils.ancillary_utils import is_nondecreasing, get_list_of_floats_as_str, \
                                  get_rounded_list
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.euclidean_synthetic_functions import get_syn_func_caller, get_mf_cost_function
from ..utils import reporters


# A base class for all Optimisers on Euclidean spaces ----------------------------
class EuclideanOptimisersBaseTestCase(object):
  """ A base test class for all optimisers on Euclidean domains. """
  # pylint: disable=no-member
  # pylint: disable=invalid-name

  def setUp(self):
    """ set up. """
    self.test_max_capital = 60 # for hartmann 6
    self.func_caller = get_syn_func_caller('hartmann6',
                                           noise_type='gauss', noise_scale=0.1)
    self.worker_manager_1 = SyntheticWorkerManager(1, time_distro='const')
    self.worker_manager_3 = SyntheticWorkerManager(3, time_distro='halfnormal')

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate a specific optimiser. """
    raise NotImplementedError('Implement in a child test class.')

  @classmethod
  def run_optimiser(cls, func_caller, worker_manager, max_capital, mode, *args, **kwargs):
    """ Runs optimiser. """
    raise NotImplementedError('Implement in a child test class.')

  def test_instantiation(self):
    """ Tests instantiation of the optimiser. """
    optimiser = self._child_instantiate_optimiser(self.func_caller, self.worker_manager_3,
                  options=None, reporter=reporters.get_reporter('silent'))
    self.report('Instantiated %s object.'%(type(optimiser)))
    for attr in dir(optimiser):
      if not attr.startswith('_'):
        self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))),
                    'test_result')

  def _test_optimiser_results(self, opt_val, opt_point, history):
    """ Tests optimiser results. """
    assert opt_val == history.curr_opt_vals[-1]
    assert opt_point.shape[0] == self.func_caller.domain.dim
    assert is_nondecreasing(history.curr_opt_vals)
    assert is_nondecreasing(history.curr_true_opt_vals)
    self.report('True opt val sequence: %s.'%(
      get_list_of_floats_as_str(history.curr_true_opt_vals)))
    saved_in_history = [key for key, _ in list(history.__dict__.items()) if not
                        key.startswith('__')]
    self.report('Stored in history: %s.'%(saved_in_history), 'test_result')

  def test_optimisation_single(self):
    """ Test optimisation with a single worker. """
    self.report('')
    self.report('Testing %s with one worker.'%(type(self)))
    opt_val, opt_point, history = self.run_optimiser(self.func_caller,
      self.worker_manager_1, self.test_max_capital, 'asy')
    self._test_optimiser_results(opt_val, opt_point, history)
    self.report('')
    return opt_val, opt_point, history

  def test_optimisation_asynchronous(self):
    """ Testing random optimiser with three asynchronous workers. """
    self.report('Testing %s with three asynchronous workers.'%(type(self)))
    opt_val, opt_point, history = self.run_optimiser(self.func_caller,
      self.worker_manager_3, self.test_max_capital, 'asy')
    self._test_optimiser_results(opt_val, opt_point, history)
    self.report('')
    return opt_val, opt_point, history

  def test_optimisation_synchronous(self):
    """ Testing random optimiser with three synchronous workers. """
    self.report('Testing %s with three synchronous workers.'%(type(self)))
    opt_val, opt_point, history = self.run_optimiser(self.func_caller,
      self.worker_manager_3, self.test_max_capital, 'syn')
    self._test_optimiser_results(opt_val, opt_point, history)
    self.report('')
    return opt_val, opt_point, history

# A test class for MF random optimisers ---------------------------------------------
def get_multi_fidelity_history_str(history):
  """ Prints the history of the multifidelity optimisation. """
  fidel_dim = len(history.query_fidels[0])
  print_fidels = np.array([get_rounded_list(qf) for qf in history.query_fidels]).T
  # First the function values
  val_title_str = ', '.join(['fidel(dim=%d)'%(fidel_dim), 'val', 'curr_true_opt_val'])
  val_title_str = val_title_str + '\n' + '-' * len(val_title_str)
  print_vals = get_rounded_list(history.query_vals)
  print_curr_opt_vals = get_rounded_list(history.curr_true_opt_vals)
  val_history_mat_str_T = np.vstack((print_fidels, print_vals, print_curr_opt_vals))
  val_history_mat_list = [get_list_of_floats_as_str(row) for row in
                          val_history_mat_str_T.T]
  val_history_mat_str = '\n'.join(val_history_mat_list)
  # NOw the fidels
  fidel_title_str = ', '.join(['fidel(%d)'%(fidel_dim), 'eval_times', 'fidel_costs',
                                'cum(fidel_costs)', 'receive_times'])
  fidel_title_str = fidel_title_str + '\n' + '-' * len(fidel_title_str)
  fidel_history_mat_str_T = np.vstack((print_fidels, history.query_eval_times,
                    history.query_cost_at_fidels,
                    np.cumsum(history.query_cost_at_fidels), history.query_receive_times))
  fidel_history_mat_list = [get_list_of_floats_as_str(row) for row in
                            fidel_history_mat_str_T.T]
  fidel_history_mat_str = '\n'.join(fidel_history_mat_list)
  return val_title_str + '\n' + val_history_mat_str + '\n\n' + \
         fidel_title_str + '\n' + fidel_history_mat_str + '\n'


# Multi-fidelity Optimisers base test case
class MFEuclideanOptimisersBaseTestCase(EuclideanOptimisersBaseTestCase):
  """ A base test class for all optimisers on Euclidean domains. """
  # pylint: disable=abstract-method
  # pylint: disable=no-member

  def setUp(self):
    """ Set up. Override the set up from the parent class. """
    super(MFEuclideanOptimisersBaseTestCase, self).setUp()
    self.func_caller = get_syn_func_caller('hartmann6', noise_type='gauss',
                                           noise_scale=0.1, fidel_dim=1)

  def test_optimisation_single(self):
    """ Test optimisation with a single worker. """
    _, _, history = \
      super(MFEuclideanOptimisersBaseTestCase, self).test_optimisation_single()
    mf_history_str = get_multi_fidelity_history_str(history)
    self.report(mf_history_str)

  def test_optimisation_asynchronous(self):
    """ Testing random optimiser with four asynchronous workers. """
    _, _, history = \
      super(MFEuclideanOptimisersBaseTestCase, self).test_optimisation_asynchronous()
    mf_history_str = get_multi_fidelity_history_str(history)
    self.report(mf_history_str)

  def test_ask_tell(self):
    """ Testing random optimiser with ask tell interface. """
    self.report('Testing %s using the ask-tell interface.'%(type(self)))
    domain = domains.EuclideanDomain([[0, 2.3], [3.4, 8.9], [0.12, 1.0]])
    fidel_bounds = [[0, 1]]
    fidel_space = domains.EuclideanDomain(fidel_bounds)
    fidel_to_opt = [0.5]
    fidel_cost = get_mf_cost_function(fidel_bounds)
    func_caller = EuclideanFunctionCaller(None, domain, raw_fidel_space=fidel_space,
                                          fidel_cost_func=fidel_cost, raw_fidel_to_opt=fidel_to_opt)
    opt = random_optimiser.MFEuclideanRandomOptimiser(func_caller, ask_tell_mode=True)
    opt.initialise()

    def evaluate(z, x):
      return get_syn_func_caller('hartmann3', noise_type='gauss', noise_scale=0.1, fidel_dim=1).func(z, x)

    best_z, best_x, best_y = None, None, float('-inf')
    for _ in range(60):
      point = opt.ask()
      z, x = point[0], point[1]
      y = evaluate(z, x)
      opt.tell([(z, x, y)])
      self.report('z: %s, x: %s, y: %s'%(z, x, y))
      if y > best_y:
        best_z, best_x, best_y = z, x, y
    self.report("-----------------------------------------------------")
    self.report("Optimal Value: %s, Optimal Point: %s (Fidel: %s)"%(best_y, best_x, best_z))

    self.report("-----------------------------------------------------")
    self.report("Regular optimisation using maximise_multifidelity_function")
    val, pt, _ = maximise_multifidelity_function(evaluate, fidel_space, domain, fidel_to_opt, fidel_cost, 60, opt_method='rand')
    self.report("Optimal Value: %s, Optimal Point: %s"%(val, pt))

# Now the Testcases for Random Euclidean Optimisers =================================

class EuclideanRandomOptimiserTestCase(EuclideanOptimisersBaseTestCase, BaseTestClass):
# class EuclideanRandomOptimiserTestCase(EuclideanOptimisersBaseTestCase):
  """ Unittest for the EuclideanRandomOptimiser class. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager,
                                   options=None, reporter=None):
    """ Instantiates a random optimiser in Euclidean space. """
    return random_optimiser.EuclideanRandomOptimiser(func_caller, worker_manager, options,
                                                     reporter)

  @classmethod
  def run_optimiser(cls, func_caller, worker_manager, max_capital, mode, *args, **kwargs):
    """ Runs optimiser. """
    return random_optimiser.random_optimiser_from_func_caller(
             func_caller, worker_manager, max_capital, mode)
  
  def test_ask_tell(self):
    """ Testing random optimiser with ask tell interface. """
    self.report('Testing %s using the ask-tell interface.'%(type(self)))
    domain = domains.EuclideanDomain([[0, 2.3], [3.4, 8.9], [0.12, 1.0]])
    func_caller = EuclideanFunctionCaller(None, domain)
    opt = random_optimiser.EuclideanRandomOptimiser(func_caller, ask_tell_mode=True)
    opt.initialise()

    def evaluate(x):
      return get_syn_func_caller('hartmann3', noise_type='gauss', noise_scale=0.1).func(x)
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
    self.report("Regular optimisation using maximise_function")
    val, pt, _ = maximise_function(evaluate, domain, 60, opt_method='rand')
    self.report("Optimal Value: %s, Optimal Point: %s"%(val, pt))
    

class MFEucRandomOptimiserTestCase(MFEuclideanOptimisersBaseTestCase, BaseTestClass):
# class MFEucRandomOptimiserTestCase(MFEuclideanOptimisersBaseTestCase):
  """ Unit tests for the Multi-fidelity Euclidean random optimiser. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager,
                                   options=None, reporter=None):
    """ Instantiates a random MF optimiser in Euclidean space. """
    return random_optimiser.MFEuclideanRandomOptimiser(func_caller, worker_manager,
                                                       options=options, reporter=reporter)

  @classmethod
  def run_optimiser(cls, func_caller, worker_manager, max_capital, mode, *args, **kwargs):
    """ Runs optimiser. """
    return random_optimiser.mf_random_optimiser_from_func_caller(
                              func_caller, worker_manager, max_capital, mode,
                              call_fidel_to_opt_prob=0.333)


if __name__ == '__main__':
  execute_tests()

