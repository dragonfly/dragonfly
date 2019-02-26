"""
  Unittests for optimisers on CP domains with NN variables.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=abstract-method

import os
import unittest
# Local
from ..exd.worker_manager import SyntheticWorkerManager
from ..opt.unittest_cp_random_optimiser import CPOptimiserBaseTestCase, \
                                               CPRandomOptimiserTestCaseDefinitions, \
                                               MFCPRandomOptimiserTestCaseDefinitions
from ..opt.unittest_cp_ga_optimiser import CPGAOPtimiserTestCaseDefinitions
from ..opt.unittest_cp_gp_bandit import CPGPBanditTestCaseDefinitions
from ..opt.unittest_mf_cp_gp_bandit import MFCPGPBanditTestCaseDefinitions
from ..test_data.syn_cnn_2.syn_cnn_2 import syn_cnn_2
from ..test_data.syn_cnn_2.syn_cnn_2_mf import syn_cnn_2_mf
from ..test_data.syn_cnn_2.syn_cnn_2_mf import cost as cost_syn_cnn_2_mf
from ..utils.base_test_class import BaseTestClass, execute_tests


class NNCPOptimiserBaseTestCase(CPOptimiserBaseTestCase):
  """ Unit-tests Definitions for CP domains with NN variables. """
  # pylint: disable=no-init

  def setUp(self):
    """ Set up. """
    self.max_capital = 30
    self._child_set_up()
    self.worker_manager_1 = SyntheticWorkerManager(1, time_distro='const')
    self.worker_manager_3 = SyntheticWorkerManager(3, time_distro='halfnormal')
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_pardir = os.path.dirname(file_dir)
    self.opt_problems = [
      (test_data_pardir + '/test_data/syn_cnn_2/config.json', (syn_cnn_2,)),
      ]


class MFNNCPOptimiserBaseTestCase(CPOptimiserBaseTestCase):
  """ Unit-tests Definitions for CP domains with NN variables. """
  # pylint: disable=no-init

  def setUp(self):
    """ Set up. """
    self.max_capital = 20
    self._child_set_up()
    self.worker_manager_1 = SyntheticWorkerManager(1, time_distro='const')
    self.worker_manager_3 = SyntheticWorkerManager(3, time_distro='halfnormal')
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_pardir = os.path.dirname(file_dir)
    self.opt_problems = [
      (test_data_pardir + '/test_data/syn_cnn_2/config_mf.json',
                              (syn_cnn_2_mf, cost_syn_cnn_2_mf)),
      ]


# Random Optimiser Unit tests -----------------------------------------------------
@unittest.skip
class NNCPRandomOptimiserTestCase(CPRandomOptimiserTestCaseDefinitions,
                                  NNCPOptimiserBaseTestCase,
                                  BaseTestClass):
  """ Unit tests for the Random optimiser on cartesian product spaces. """
  pass


# @unittest.skip
class MFNNCPRandomOptimiserTestCase(MFCPRandomOptimiserTestCaseDefinitions,
                                    MFNNCPOptimiserBaseTestCase,
                                    BaseTestClass):
  """ Unit tests for Multi-fidelity random optimiser on cartesian product spaces. """
  pass


# GA Optimiser Unit tests -----------------------------------------------------------
@unittest.skip
class NNCPGAOPtimiserTestCase(CPGAOPtimiserTestCaseDefinitions,
                              NNCPOptimiserBaseTestCase,
                              BaseTestClass):
  """ Unit tests for GA optimiser on cartesian product spaces. """
  pass


# GPBandit Unit tests ---------------------------------------------------------------
@unittest.skip
class NNCPGPBanditTestCase(CPGPBanditTestCaseDefinitions,
                           NNCPOptimiserBaseTestCase,
                           BaseTestClass):
  """ Unit tests for GP Bandits on cartesian product spaces. """
  pass

@unittest.skip
class MFCPGPBanditTestCase(MFCPGPBanditTestCaseDefinitions,
                           MFNNCPOptimiserBaseTestCase,
                           BaseTestClass):
  """ Unit tests for GP Bandits on cartesian product spaces. """
  pass


if __name__ == '__main__':
  execute_tests()

