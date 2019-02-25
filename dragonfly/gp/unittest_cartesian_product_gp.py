"""
  Unit tests for cartesian product GPs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

from argparse import Namespace
import numpy as np
import os
from time import time
# Local
from ..exd.cp_domain_utils import load_cp_domain_from_config_file, \
                                get_processed_func_from_raw_func_for_cp_domain, \
                                get_processed_func_from_raw_func_via_config, \
                                load_config_file, sample_from_config_space
from . import cartesian_product_gp as cpgp
from ..test_data.park1_3.park1_3 import park1_3
from ..test_data.park1_3.park1_3_mf import park1_3_mf
from ..test_data.park2_4.park2_4 import park2_4
from ..test_data.park2_4.park2_4_mf import park2_4_mf
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.general_utils import map_to_bounds, get_idxs_from_list_of_lists


def get_test_dataset(domain_file_name, func, X_train, X_test, kernel=None):
  """ An internal function to organise what is being returned. """
  domain, orderings = load_cp_domain_from_config_file(domain_file_name)
  func = get_processed_func_from_raw_func_for_cp_domain(func, domain, \
           orderings.index_ordering, orderings.dim_ordering)
  Y_train = [func(x) for x in X_train]
  Y_test = [func(x) for x in X_test]
  return Namespace(domain_file_name=domain_file_name, X_train=X_train, Y_train=Y_train,
                   X_test=X_test, Y_test=Y_test, kernel=kernel)


def gen_cpgp_test_data():
  """ Generates CPGP test data. """
  file_dir = os.path.dirname(os.path.realpath(__file__))
  test_data_dir = os.path.dirname(file_dir)
  # pylint: disable=too-many-statements
  # pylint: disable=too-many-locals
  # TODO: test against a naive GP in each of these cases
  # Now define the datasets --------------------------------------------------------------
  ret = []
  n_train = 200
  n_test = 300
  # Dataset 1
  domain_file_name = test_data_dir + '/' + 'test_data/example_configs/eg01.json'
  func = lambda x: x[0][0]**2 + 2*x[1][0] * x[0][0] + 3*x[1][0] + 2.1
  bounds = np.array([[-5, 10], [0, 15]])
  X_train = map_to_bounds(np.random.random((n_train, 2)), bounds)
  X_test = map_to_bounds(np.random.random((n_train, 2)), bounds)
  X_train = [[x] for x in X_train]
  X_test = [[x] for x in X_test]
  ret.append(get_test_dataset(domain_file_name, func, X_train, X_test))
  # Dataset 2
  domain_file_name = test_data_dir + '/' + 'test_data/park2_4/config.json'
  func = park2_4
  bounds = np.array([[0, 1], [0, 1], [103, 194], [10, 11]])
  X_tr_num = map_to_bounds(np.random.random((n_train, len(bounds))), bounds)
  X_te_num = map_to_bounds(np.random.random((n_test, len(bounds))), bounds)
  X_tr_disc = np.random.choice(['rabbit', 'dog', 'hamster', 'gerbil', 'ferret'], n_train)
  X_te_disc = np.random.choice(['rabbit', 'dog', 'hamster', 'gerbil', 'ferret'], n_test)
  X_train = [[[x[0], x[1], x[3]], [int(x[2])], [xd]]
             for (x, xd) in zip(X_tr_num, X_tr_disc)]
  X_test = [[[x[0], x[1], x[3]], [int(x[2])], [xd]]
             for (x, xd) in zip(X_te_num, X_te_disc)]
  ret.append(get_test_dataset(domain_file_name, func, X_train, X_test))
  # Dataset 3
  domain_file_name = test_data_dir + '/' + 'test_data/park1_3/config.json'
  func = park1_3
  bounds = np.array([[10, 16], [0, 1], [0, 1]])
  x2_elems = [4, 10, 23, 45, 78, 87.1, 91.8, 99, 75.7, 28.1, 3.141593]
  modify_func = lambda x1, x2: [[x1[0]],
                  [round(x1[1] * 20)/20, round(x1[2] * 20)/20, x2]]
  X_train_1 = map_to_bounds(np.random.random((n_train, len(bounds))), bounds)
  X_train_2 = np.random.choice(x2_elems, n_train)
  X_train = [modify_func(x1, x2) for (x1, x2) in zip(X_train_1, X_train_2)]
  X_test_1 = map_to_bounds(np.random.random((n_test, len(bounds))), bounds)
  X_test_2 = np.random.choice(x2_elems, n_test)
  X_test = [modify_func(x1, x2) for (x1, x2) in zip(X_test_1, X_test_2)]
  ret.append(get_test_dataset(domain_file_name, func, X_train, X_test))
  # Return
  return ret


def gen_cpmfgp_test_data_from_config_file(config_file_name, raw_func,
  num_tr_data, num_te_data):
  """ Generates datasets for CP Multi-fidelity GP fitting. """
  # Generate data
  def _generate_data(_proc_func, _config, _num_data):
    """ Generates data. """
    ZX_proc = sample_from_config_space(_config, _num_data)
    YY_proc = [_proc_func(z, x) for (z, x) in ZX_proc]
    ZZ_proc = get_idxs_from_list_of_lists(ZX_proc, 0)
    XX_proc = get_idxs_from_list_of_lists(ZX_proc, 1)
    return ZZ_proc, XX_proc, YY_proc, ZX_proc
  # Get dataset for testing
  def _get_dataset_for_testing(_proc_func, _config, _num_tr_data, _num_te_data):
    """ Get dataset for testing. """
    ZZ_train, XX_train, YY_train, ZX_train = _generate_data(_proc_func, _config,
                                                            _num_tr_data)
    ZZ_test, XX_test, YY_test, ZX_test = _generate_data(_proc_func, _config, _num_te_data)
    return Namespace(config_file_name=config_file_name, config=config, raw_func=raw_func,
      ZZ_train=ZZ_train, XX_train=XX_train, YY_train=YY_train, ZX_train=ZX_train,
      ZZ_test=ZZ_test, XX_test=XX_test, YY_test=YY_test, ZX_test=ZX_test)
  # Generate the data and return
  config = load_config_file(config_file_name)
  proc_func = get_processed_func_from_raw_func_via_config(raw_func, config)
  return _get_dataset_for_testing(proc_func, config, num_tr_data, num_te_data)


def gen_cpmfgp_test_data(num_tr_data, num_te_data):
  """ Generates data on all functions. """
  file_dir = os.path.dirname(os.path.realpath(__file__))
  test_data_dir = os.path.dirname(file_dir)
  test_problems = [
    (test_data_dir + '/test_data/park1_3/config_mf.json', park1_3_mf),
    (test_data_dir + '/test_data/park2_4/config_mf.json', park2_4_mf),
    ]
  ret = [gen_cpmfgp_test_data_from_config_file(cfn, rf, num_tr_data, num_te_data)
         for cfn, rf in test_problems]
  return ret


def fit_cpgp_with_dataset(dataset, FitterClass):
  """ Fits and returns a CPGP. """
  gp_fitter = FitterClass(dataset.X_train, dataset.Y_train, dataset.domain_file_name)
  ret_gp = gp_fitter.fit_gp()
  return ret_gp[1], gp_fitter.domain


def fit_cpmfgp_with_dataset(dataset, *args, **kwargs):
  """ Fits and returns a CPMFGP. """
  config = dataset.config
  mfgp_fitter = cpgp.CPMFGPFitter(dataset.ZZ_train, dataset.XX_train, dataset.YY_train,
    fidel_space=config.fidel_space, domain=config.domain,
    fidel_space_kernel_ordering=config.fidel_space_orderings.kernel_ordering,
    domain_kernel_ordering=config.domain_orderings.kernel_ordering,
    *args, **kwargs)
  ret = mfgp_fitter.fit_gp()
  return ret[1]


class CPGPTestCaseDefinitions(object):
  """ Defing unit tests for CartesianProduct GPs. """

  @classmethod
  def _compute_rmse(cls, Y_test, preds):
    """ Computes root mean square error. """
    return np.linalg.norm(Y_test - preds)/np.sqrt(len(Y_test))

  def _test_prediction(self, Y_test_pred, Y_test, Y_train):
    """ Tests prediction on the training set. """
    Y_test_pred = np.array(Y_test_pred)
    Y_test = np.array(Y_test)
    Y_train = np.array(Y_train)
    mean_train_err = self._compute_rmse(Y_test, Y_train.mean())
    mean_test_err = self._compute_rmse(Y_test, Y_test.mean())
    mean_gp_pred_error = self._compute_rmse(Y_test, Y_test_pred)
    test_succ = mean_gp_pred_error < mean_train_err
    test_str = 'gp_err=%0.4f, train_mean_err=%0.4f, test_mean_err=%0.4f, succ=%d'%(
      mean_gp_pred_error, mean_train_err, mean_test_err, test_succ)
    return test_succ, test_str

  def test_gp_fitting_and_creation(self):
    """ Tests fitting and creation. """
    self.report('Testing fitting Cartesian Product GP. Probabilistic test, might fail.')
    num_successes = 0
    for idx, dataset in enumerate(self.cpgp_datasets):
      fit_start_time = time()
      fitted_gp, cp_domain = fit_cpgp_with_dataset(dataset, cpgp.CPGPFitter)
      fit_end_time = time()
      fit_time = fit_end_time - fit_start_time
      lml = fitted_gp.compute_log_marginal_likelihood()
      gp_preds, _ = fitted_gp.eval(dataset.X_test, uncert_form='none')
      test_succ, test_str = self._test_prediction(gp_preds, dataset.Y_test,
                                                  dataset.Y_train)
      self.report('[%d/%d] Fitted GP: lml=%0.4f, GP=%s, time_taken=%0.4fs'%(
        idx+1, len(self.cpgp_datasets), lml, fitted_gp, fit_time), 'test_result')
      self.report('   Test result: %s.'%(test_str), 'test_result')
      self.report('   Domain=%s.'%(cp_domain), 'test_result')
      num_successes += test_succ
    success_frac = num_successes/float(self.num_datasets)
    self.report('Num successes = %d/%d (%0.4f).\n'%(num_successes, self.num_datasets,
                success_frac), 'test_result')
    assert success_frac > 0.5


class CPGPTestCase(CPGPTestCaseDefinitions, BaseTestClass):
  """ Test cases for CPGP. """

  def setUp(self):
    """ Set up. """
    self.cpgp_datasets = gen_cpgp_test_data()
    self.num_datasets = len(self.cpgp_datasets)


class CPMFGPTestCaseDefinitions(CPGPTestCaseDefinitions):
  """ Defing unit tests for Multi-fidelity Cartesian Product GPs. """

  def test_gp_fitting_and_creation(self):
    """ Tests fitting and creation. """
    self.report('Testing fitting Multi-fidelity Cartesian Product GPs. ' +
                'Probabilistic test, might fail.')
    num_successes = 0
    for idx, dataset in enumerate(self.cpmfgp_datasets):
      fit_start_time = time()
      fitted_gp = fit_cpmfgp_with_dataset(dataset)
      fit_end_time = time()
      fit_time = fit_end_time - fit_start_time
      lml = fitted_gp.compute_log_marginal_likelihood()
      gp_preds, _ = fitted_gp.eval_at_fidel(dataset.ZZ_test, dataset.XX_test,
                                            uncert_form='none')
      preds_with_eval_func, _ = fitted_gp.eval(dataset.ZX_test, uncert_form='none')
      diff_eval_err = self._compute_rmse(gp_preds, preds_with_eval_func)
      test_succ, test_str = self._test_prediction(gp_preds, dataset.YY_test,
                                                  dataset.YY_train)
      self.report('[%d/%d] Fitted GP: lml=%0.4f, GP=%s, time taken=%0.4fs.'%(
        idx+1, len(self.cpmfgp_datasets), lml, fitted_gp, fit_time), 'test_result')
      self.report('   Test result: %s, diff_eval_err=%0.4f.'%(test_str, diff_eval_err),
                  'test_result')
      self.report('   fidel_space=%s.'%(dataset.config.fidel_space), 'test_result')
      self.report('   domain=%s.'%(dataset.config.domain), 'test_result')
      num_successes += test_succ
      assert diff_eval_err < 1e-5 * np.linalg.norm(gp_preds)
    success_frac = num_successes/float(self.num_datasets)
    self.report('Num successes = %d/%d (%0.4f).\n'%(num_successes, self.num_datasets,
                success_frac), 'test_result')
    assert success_frac > 0.5


class CPMFGPTestCase(CPMFGPTestCaseDefinitions, BaseTestClass):
  """ Unit tests for Multi-fidelity Cartesian Product GPs. """

  def setUp(self):
    """ Set up. """
    num_tr_data = 300
    num_te_data = 300
    self.cpmfgp_datasets = gen_cpmfgp_test_data(num_tr_data, num_te_data)
    self.num_datasets = len(self.cpmfgp_datasets)


if __name__ == '__main__':
  execute_tests()

