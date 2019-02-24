"""
  Unit tests for classes/methods in euclidean_gp.py
  -- kandasamy@cs.cmu.edu
"""
from __future__ import division
import unittest

# pylint: disable=invalid-name

import numpy as np
# Local
from .euclidean_gp import EuclideanGP, EuclideanGPFitter, euclidean_gp_args
from .unittest_gp_core import compute_average_prediction_error
from .unittest_gp_core import gen_gp_test_data
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.option_handler import load_options


# Some wrappers to build a GP ----------------------------------------------------------
def build_euc_gp_with_dataset(dataset, kernel_type):
  """ Function to build a SE GP with the dataset. """
  mean_func = lambda x: np.array([np.median(dataset[1])] * len(x))
  noise_var = dataset[1].std()**2/20
  kernel_hyperparams = dataset[2].hyperparams
  kernel_hyperparams['dim'] = len(dataset[0][0])
  kernel_hyperparams['nu'] = 2.5
  return EuclideanGP(dataset[0], dataset[1], kernel_type, mean_func, noise_var,
                     kernel_hyperparams)

def build_se_gp_with_dataset(dataset):
  """ Builds an SE GP with the dataset. """
  return build_euc_gp_with_dataset(dataset, 'se')

def build_matern_gp_with_dataset(dataset):
  """ builds an se gp with the dataset. """
  return build_euc_gp_with_dataset(dataset, 'matern')

def fit_se_gp_with_dataset(dataset):
  """ A wrapper to fit a gp using the dataset. """
  options = load_options(euclidean_gp_args)
  options.kernel_type = 'se'
  ret_fit_gp = (EuclideanGPFitter(dataset[0], dataset[1],
                                  options=options, reporter=None)).fit_gp()
  assert ret_fit_gp[0] == 'fitted_gp'
  return ret_fit_gp[1]

def fit_matern_gp_with_dataset(dataset, nu=-1.0):
  """ A wrapper to fit a GP with a matern kernel using the dataset. """
  options = load_options(euclidean_gp_args)
  options.kernel_type = 'matern'
  options.matern_nu = nu
  ret_fit_gp = (EuclideanGPFitter(dataset[0], dataset[1],
                                  options=options, reporter=None)).fit_gp()
  assert ret_fit_gp[0] == 'fitted_gp'
  return ret_fit_gp[1]

def fit_poly_gp_with_dataset(dataset, poly_order):
  """ A wrapper to fit a GP with a polynomial kernel using the dataset. """
  #pylint: disable=unused-argument
  raise NotImplementedError('Not implemented yet.')

def fit_add_gp_with_dataset(dataset, options):
  """ Fits an additive gp with dataset. """
  options.use_additive_gp = True
  ret_fit_gp = (EuclideanGPFitter(dataset[0], dataset[1],
                                  options=options, reporter=None)).fit_gp()
  assert ret_fit_gp[0] == 'fitted_gp'
  return ret_fit_gp[1]

def fit_add_se_gp_with_dataset(dataset):
  """ Fits an Additive GP with SE kernels. """
  options = load_options(euclidean_gp_args)
  options.kernel_type = 'se'
  return fit_add_gp_with_dataset(dataset, options)

def fit_add_matern_gp_with_dataset(dataset, nu=1.5):
  """ Fits an Additive GP with Matern kernels. """
  options = load_options(euclidean_gp_args)
  options.kernel_type = 'matern'
  options.matern_nu = nu
  return fit_add_gp_with_dataset(dataset, options)


# A base class that needs to be inherited by all classes ------------------------
class EuclideanGPFitterBaseTestClass(object):
  """ Some common functionality for all Euclidean tests. """
  # pylint: disable=no-member

  def setUp(self):
    """ Set up for the tests. """
    self.datasets = gen_gp_test_data()
    self.datasets = self.datasets[0:2]

  def _marg_likelihood_test(self, get_gp_func1, get_gp_func2, descr1, descr2):
    """ This tests for the marginal likelihood. GP2 is expected to do better. """
    num_successes = 0
    for dataset in self.datasets:
      gp1 = get_gp_func1(dataset)
      gp2 = get_gp_func2(dataset)
      lml1 = gp1.compute_log_marginal_likelihood()
      lml2 = gp2.compute_log_marginal_likelihood()
      success = lml1 <= lml2
      self.report('(N,D)=%s:: %s-lml=%0.4f, %s-lml=%0.4f, succ=%d'%(
          str(dataset[0].shape), descr1, lml1, descr2, lml2, success), 'test_result')
      num_successes += success
    assert num_successes == len(self.datasets)

  def _prediction_test(self, get_gp_func1, get_gp_func2, descr1, descr2):
    """ Tests for prediction on a test set. GP2 is expected to do better. """
    num_successes = 0
    for dataset in self.datasets:
      gp1 = get_gp_func1(dataset)
      preds1, _ = gp1.eval(dataset[3])
      err1 = compute_average_prediction_error(dataset, preds1)
      gp2 = get_gp_func2(dataset)
      preds2, _ = gp2.eval(dataset[3])
      err2 = compute_average_prediction_error(dataset, preds2)
      success = err2 <= err1
      self.report('(N,D)=%s:: %s-err=%0.4f, %s-err=%0.4f, succ=%d'%(
          str(dataset[0].shape), descr1, err1, descr2, err2, success), 'test_result')
      self.report('  -- GP: %s'%(str(gp2)), 'test_result')
      num_successes += success
    assert num_successes > 0.6 *len(self.datasets)


# Test for naive predictions vs fitted predictions --------------------------------
@unittest.skip
class EuclideanGPFitterTestCase(EuclideanGPFitterBaseTestClass, BaseTestClass):
  """ Unit tests for the EuclideanGP and EuclideanGPFitter class. """
  # pylint: disable=too-many-locals

  def test_se_marg_likelihood(self):
    """ This tests for the marginal likelihood for an SE GP. """
    self.report('Marginal likelihood for SEGP. Probabilistic test, might fail.')
    self._marg_likelihood_test(build_se_gp_with_dataset, fit_se_gp_with_dataset,
                               'naive', 'fitted')

  def test_se_prediction(self):
    """ Tests for prediction on a test set with an SEGP. """
    self.report('Prediction for an SE kernel. Probabilistic test, might fail.')
    self._prediction_test(build_se_gp_with_dataset, fit_se_gp_with_dataset,
                          'naive', 'fitted')

  def test_matern_marg_likelihood(self):
    """ This tests for the marginal likelihood for a Matern GP. """
    self.report('Marginal likelihood for MaternGP. Probabilistic test, might fail.')
    self._marg_likelihood_test(build_matern_gp_with_dataset, fit_matern_gp_with_dataset,
                               'naive', 'fitted')

  def test_matern_prediction(self):
    """ Tests for prediction on a test set with a MaternGP. """
    self.report('Prediction for a Matern kernel. Probabilistic test, might fail.')
    self._prediction_test(build_matern_gp_with_dataset, fit_matern_gp_with_dataset,
                          'naive', 'fitted')


# Test for non-additive vs additive predictions --------------------------------
@unittest.skip
class AdditiveGPFitterTestCase(EuclideanGPFitterBaseTestClass, BaseTestClass):
# class AdditiveGPFitterTestCase(EuclideanGPFitterBaseTestClass):
  """ Unit tests for the additive kernel. """

  def setUp(self):
    """ Set up for the tests. """
    self.datasets = gen_gp_test_data()
    self.datasets = self.datasets[1:3]

  def test_se_marg_likelihood(self):
    """ Test for the Marginal likelihood of an additive GP. """
    self.report('Marginal Likelihoods for non-additive vs additive SE GP.' +
                ' Probabilistic test, might fail.')
    self._marg_likelihood_test(fit_se_gp_with_dataset, fit_add_se_gp_with_dataset,
                               'non-additive', 'additive')

  def test_se_prediction(self):
    """ Test for the Marginal likelihood of an additive GP. """
    self.report('Marginal Likelihoods for non-additive vs additive SE GP.' +
                ' Probabilistic test, might fail.')
    self._prediction_test(fit_se_gp_with_dataset, fit_add_se_gp_with_dataset,
                          'non-additive', 'additive')

  def test_matern_marg_likelihood(self):
    """ Test for the Marginal likelihood of an additive GP. """
    self.report('Marginal Likelihoods for non-additive vs additive Matern GP.' +
                ' Probabilistic test, might fail.')
    self._marg_likelihood_test(fit_matern_gp_with_dataset, fit_add_matern_gp_with_dataset,
                               'non-additive', 'additive')

  def test_matern_prediction(self):
    """ Test for the Marginal likelihood of an additive GP. """
    self.report('Marginal Likelihoods for non-additive vs additive Matern GP.' +
                ' Probabilistic test, might fail.')
    self._prediction_test(fit_matern_gp_with_dataset, fit_add_matern_gp_with_dataset,
                          'non-additive', 'additive')



if __name__ == '__main__':
  execute_tests(2349)
#   execute_tests()

