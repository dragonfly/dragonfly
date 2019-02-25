"""
  Unittests for Cartesian product GP with NN domains. Testing these separately since
  we do not need to have nn dependencies in the gp module.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

import os
import numpy as np
import unittest
# Local
from ..gp.unittest_cartesian_product_gp import get_test_dataset, \
     gen_cpmfgp_test_data_from_config_file, \
     CPGPTestCaseDefinitions, CPMFGPTestCaseDefinitions
from ..nn import nn_examples
from ..test_data.syn_cnn_2.syn_cnn_2 import syn_cnn_2
from ..test_data.syn_cnn_2.syn_cnn_2_mf import syn_cnn_2_mf
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.general_utils import map_to_bounds


def get_cnns():
  """ Returns the initial pool for CNNs. """
  vgg_nets = [nn_examples.get_vgg_net(1),
              nn_examples.get_vgg_net(2),
              nn_examples.get_vgg_net(3),
              nn_examples.get_vgg_net(4)]
  blocked_cnns = [nn_examples.get_blocked_cnn(3, 1, 1), # 3
                  nn_examples.get_blocked_cnn(3, 2, 1), # 6
                  nn_examples.get_blocked_cnn(3, 3, 1), # 9
                  nn_examples.get_blocked_cnn(3, 4, 1), # 12
                  nn_examples.get_blocked_cnn(3, 5, 1), # 15
                  nn_examples.get_blocked_cnn(4, 4, 1), # 16
                 ]
  other_cnns = nn_examples.generate_cnn_architectures()
  ret = vgg_nets + blocked_cnns + other_cnns
  np.random.shuffle(ret)
  return ret


def get_initial_mlp_pool(class_or_reg):
  """ Returns the initial pool of MLPs. """
  blocked_mlps = [nn_examples.get_blocked_mlp(class_or_reg, 3, 2), # 6
                  nn_examples.get_blocked_mlp(class_or_reg, 4, 2), # 8
                  nn_examples.get_blocked_mlp(class_or_reg, 5, 2), # 10
                  nn_examples.get_blocked_mlp(class_or_reg, 3, 4), # 12
                  nn_examples.get_blocked_mlp(class_or_reg, 6, 2), # 12
                  nn_examples.get_blocked_mlp(class_or_reg, 8, 2), # 16
                  nn_examples.get_blocked_mlp(class_or_reg, 6, 3), # 18
                  nn_examples.get_blocked_mlp(class_or_reg, 10, 2), #20
                  nn_examples.get_blocked_mlp(class_or_reg, 4, 6), #24
                  nn_examples.get_blocked_mlp(class_or_reg, 8, 3), #24
                 ]
  other_mlps = nn_examples.generate_mlp_architectures()
  ret = blocked_mlps + other_mlps
  np.random.shuffle(ret)
  return ret


def gen_cpnnmfgp_test_data(num_tr_data, num_te_data):
  """ Generates data on all functions. """
  file_dir = os.path.dirname(os.path.realpath(__file__))
  test_data_dir = os.path.dirname(file_dir)
  test_problems = [
    (test_data_dir + '/test_data/syn_cnn_2/config_mf.json', syn_cnn_2_mf),
    ]
  ret = [gen_cpmfgp_test_data_from_config_file(cfn, rf, num_tr_data, num_te_data)
         for cfn, rf in test_problems]
  return ret


def get_cp_nn_gp_test_data():
  """ Create test data. """
  # pylint: disable=too-many-locals
  file_dir = os.path.dirname(os.path.realpath(__file__))
  test_data_dir = os.path.dirname(file_dir)
  ret = []
  n_train = 200
  n_test = 300
  # Dataset 1
  all_cnns = get_cnns()
  num_train = max(len(all_cnns), n_train)
  num_test = max(len(all_cnns), n_test)
  domain_file_name = test_data_dir + '/' + 'test_data/syn_cnn_2/config.json'
  func = syn_cnn_2
  x1_bounds = np.array([[0, 1], [0, 1], [10, 14]])
  x4_elems = [4, 10, 23, 45, 78, 87.1, 91.8, 99, 75.7, 28.1, 3.141593]
    # Create training set
  X_tr_0 = np.random.choice(all_cnns, num_train)
  X_tr_1 = map_to_bounds(np.random.random((num_train, len(x1_bounds))), x1_bounds)
  X_tr_2 = [[elem1, elem2] for (elem1, elem2) in zip(
             np.random.choice(['foo', 'bar'], num_train),
             np.random.choice(['foo', 'bar'], num_train))]
  X_tr_3 = np.random.choice(x4_elems, num_train)
  X_train = [[x0, x1, x2, [x3]] for (x0, x1, x2, x3) in \
             zip(X_tr_0, X_tr_1, X_tr_2, X_tr_3)]
    # Create test set
  X_te_0 = np.random.choice(all_cnns, num_test)
  X_te_1 = map_to_bounds(np.random.random((num_test, len(x1_bounds))), x1_bounds)
  X_te_2 = [[elem1, elem2] for (elem1, elem2) in zip(
             np.random.choice(['foo', 'bar'], num_test),
             np.random.choice(['foo', 'bar'], num_test))]
  X_te_3 = np.random.choice(x4_elems, num_test)
  X_test = [[x0, x1, x2, [x3]] for (x0, x1, x2, x3) in \
             zip(X_te_0, X_te_1, X_te_2, X_te_3)]
  ret.append(get_test_dataset(domain_file_name, func, X_train, X_test))
  return ret


class CPNNGPTestCase(CPGPTestCaseDefinitions, BaseTestClass):
  """ Unit tests for Cartesian product GPs with NN domains. """

  def setUp(self):
    """ Set up. """
    self.cpgp_datasets = get_cp_nn_gp_test_data()
    self.num_datasets = len(self.cpgp_datasets)


@unittest.skip
class CPMFGPTestCase(CPMFGPTestCaseDefinitions, BaseTestClass):
  """ Unit tests for Multi-fidelity Cartesian Product GPs. """

  def setUp(self):
    """ Set up. """
    num_tr_data = 50
    num_te_data = 50
    self.cpmfgp_datasets = gen_cpnnmfgp_test_data(num_tr_data, num_te_data)
    self.num_datasets = len(self.cpmfgp_datasets)



if __name__ == '__main__':
  execute_tests()

