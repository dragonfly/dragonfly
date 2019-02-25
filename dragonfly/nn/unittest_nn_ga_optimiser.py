"""
  Unit tests for the genetic algorithm optimiser.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

from copy import deepcopy
from argparse import Namespace
import unittest
# Local
from . import nn_ga_optimiser
from .nn_domains import NNDomain
from .syn_nn_functions import cnn_syn_func1, mlp_syn_func1
from .nn_domains import CNNConstraintChecker, MLPConstraintChecker
from .nn_modifiers import get_nn_modifier_from_args
from .unittest_nn_modifier_class import networks_are_equal
from .nn_opt_utils import get_initial_cnn_pool, get_initial_mlp_pool
from ..exd.experiment_caller import FunctionCaller
from ..exd.worker_manager import SyntheticWorkerManager
from ..utils.reporters import get_reporter
from ..utils.option_handler import load_options
from ..utils.base_test_class import BaseTestClass, execute_tests


def check_if_pre_eval_networks_have_changed(options_1, options_2):
  """ Checks if the pre_eval networks have changed. Raises an error if they have. """
  assert options_1.pre_eval_vals is not options_2.pre_eval_vals and \
         options_1.pre_eval_vals == options_2.pre_eval_vals
  assert options_1.pre_eval_true_vals is not options_2.pre_eval_true_vals and \
         options_1.pre_eval_true_vals == options_2.pre_eval_true_vals
  for idx in range(len(options_1.pre_eval_points)):
    assert options_1.pre_eval_points[idx] is not options_2.pre_eval_points[idx]
    assert networks_are_equal(options_1.pre_eval_points[idx],
                              options_2.pre_eval_points[idx])

def get_optimiser_args(tester, nn_type, optimiser_args):
  """ Returns arguments for the optimiser. """
  reporter = get_reporter('default')
  options = load_options(optimiser_args, reporter=reporter)
  if nn_type == 'cnn':
    options.pre_eval_points = tester.cnn_init_pool
    options.pre_eval_vals = tester.cnn_init_vals
    options.pre_eval_true_vals = tester.cnn_init_vals
    constraint_checker = tester.cnn_constraint_checker
    mutation_op = tester.cnn_mutation_op
  else:
    options.pre_eval_points = tester.mlp_init_pool
    options.pre_eval_vals = tester.mlp_init_vals
    options.pre_eval_true_vals = tester.mlp_init_vals
    constraint_checker = tester.mlp_constraint_checker
    mutation_op = tester.mlp_mutation_op
  options_clone = deepcopy(options)
  return options, options_clone, reporter, constraint_checker, mutation_op

def get_nn_opt_arguments():
  """ Returns arguments for NN Optimisation. """
  ret = Namespace()
  ret.cnn_constraint_checker = CNNConstraintChecker(50, 5, 1e8, 0, 5, 5, 200, 1024, 8)
  ret.mlp_constraint_checker = MLPConstraintChecker(50, 5, 1e8, 0, 5, 5, 200, 1024, 8)
  ret.cnn_mutation_op = get_nn_modifier_from_args(ret.cnn_constraint_checker,
                                                   [0.5, 0.25, 0.125, 0.075, 0.05])
  ret.mlp_mutation_op = get_nn_modifier_from_args(ret.mlp_constraint_checker,
                                                   [0.5, 0.25, 0.125, 0.075, 0.05])
  # Create the initial pool
  ret.cnn_init_pool = get_initial_cnn_pool()
  ret.cnn_init_vals = [cnn_syn_func1(cnn) for cnn in ret.cnn_init_pool]
  ret.mlp_init_pool = get_initial_mlp_pool('reg')
  ret.mlp_init_vals = [mlp_syn_func1(mlp) for mlp in ret.mlp_init_pool]
  # Create a domain
  ret.nn_domain = NNDomain('cnn', ret.cnn_constraint_checker)
  ret.cnn_domain = NNDomain('cnn', ret.cnn_constraint_checker)
  ret.mlp_domain = NNDomain('mlp-reg', ret.mlp_constraint_checker)
  # Return
  return ret


# Tester class ========================================================================
@unittest.skip
class GAOptimiserTestCase(BaseTestClass):
  """ Unit test for GA Optimisation. """

  def setUp(self):
    """ Set up. """
    ret = get_nn_opt_arguments()
    for key, val in ret.__dict__.items():
      setattr(self, key, val)

  def test_instantiation(self):
    """ Test creation of object. """
    self.report('Testing GA Optimiser instantiation.')
    func_caller = FunctionCaller(cnn_syn_func1, self.nn_domain)
    worker_manager = SyntheticWorkerManager(1, time_distro='const')
    optimiser = nn_ga_optimiser.NNGAOptimiser(func_caller,
      worker_manager, self.cnn_mutation_op, reporter='silent')
    self.report('Instantiated GAOptimiser object.')
    for attr in dir(optimiser):
      if not attr.startswith('_'):
        self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))),
                    'test_result')

  def _get_optimiser_args(self, nn_type):
    """ Returns the options and reporter. """
    return get_optimiser_args(self, nn_type, nn_ga_optimiser.nn_ga_opt_args)

  @classmethod
  def _test_optimiser_results(cls, opt_val, _, history, options, options_clone):
    """ Tests optimiser results. """
    assert opt_val == history.curr_opt_vals[-1]
    check_if_pre_eval_networks_have_changed(options, options_clone)

  def test_ga_optimisation_single(self):
    """ Test optimisation. """
    self.report('Testing GA Optimiser with just one worker.')
    worker_manager = SyntheticWorkerManager(1, time_distro='const')
    func_caller = FunctionCaller(cnn_syn_func1, self.nn_domain)
    options, options_clone, reporter, _, mutation_op = self._get_optimiser_args('cnn')
    opt_val, opt_pt, history = nn_ga_optimiser.nn_ga_optimise_from_args(
      func_caller, worker_manager, 20, 'asy', mutation_op,
      options=options, reporter=reporter)
    self._test_optimiser_results(opt_val, opt_pt, history, options, options_clone)
    self.report('')

  def test_optimisation_asynchronous(self):
    """ Test optimisation. """
    self.report('Testing GA Optimiser with four workers asynchronously.')
    worker_manager = SyntheticWorkerManager(4, time_distro='halfnormal')
    func_caller = FunctionCaller(mlp_syn_func1, self.nn_domain)
    options, options_clone, reporter, _, mutation_op = self._get_optimiser_args('mlp')
    opt_val, opt_pt, history = nn_ga_optimiser.nn_ga_optimise_from_args(
      func_caller, worker_manager, 20, 'asy', mutation_op,
      options=options, reporter=reporter)
    self._test_optimiser_results(opt_val, opt_pt, history, options, options_clone)
    self.report('')


if __name__ == '__main__':
  execute_tests()

