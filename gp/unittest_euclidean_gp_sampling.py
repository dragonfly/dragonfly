"""
  Unit tests for gp fit using sampling - slice or nuts
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import division

# pylint: disable=invalid-name

import numpy as np
# Local
from gp.euclidean_gp import EuclideanGP, EuclideanGPFitter, euclidean_gp_args
from gp.unittest_gp_core import compute_average_prediction_error
from gp.unittest_gp_core import gen_gp_test_data
from utils.base_test_class import BaseTestClass, execute_tests
from utils.option_handler import load_options


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

def fit_se_gp_with_dataset(dataset, method='slice'):
  """ A wrapper to fit a gp using the dataset. """
  options = load_options(euclidean_gp_args)
  options.kernel_type = 'se'
  if method is not None:
    options.hp_tune_criterion = 'post_sampling'
    options.post_hp_tune_method = method
  ret_fit_gp = (EuclideanGPFitter(dataset[0], dataset[1],
                                  options=options, reporter=None)).fit_gp()
  if method is not None:
    assert ret_fit_gp[0] == 'post_fitted_gp'
  else:
    assert ret_fit_gp[0] == 'fitted_gp'
  return ret_fit_gp[1]

def fit_matern_gp_with_dataset(dataset, nu=-1.0, method='slice'):
  """ A wrapper to fit a GP with a matern kernel using the dataset. """
  options = load_options(euclidean_gp_args)
  options.kernel_type = 'matern'
  options.matern_nu = nu
  if method is not None:
    options.hp_tune_criterion = 'post_sampling'
    options.post_hp_tune_method = method
  ret_fit_gp = (EuclideanGPFitter(dataset[0], dataset[1],
                                  options=options, reporter=None)).fit_gp()
  if method is not None:
    assert ret_fit_gp[0] == 'post_fitted_gp'
  else:
    assert ret_fit_gp[0] == 'fitted_gp'
  return ret_fit_gp[1]

# A base class that needs to be inherited by all classes ------------------------
class EuclideanGPFitterBaseTestClass(object):
  """ Some common functionality for all Euclidean tests. """
  # pylint: disable=no-member

  def setUp(self):
    """ Set up for the tests. """
    self.datasets = gen_gp_test_data()
    self.datasets = self.datasets[0:2]
    self.gp3 = []
    self.err3 = []
    self.rand_2 = False
    self.kernel = ''

  def _prediction_test(self, get_gp_func1, get_gp_func2, descr1, descr2, descr3,
                       kernel, method='slice'):
    """ Tests for prediction on a test set. GP2 is expected to do better. """
    if self.kernel != kernel:
      self.rand = False
      self.kernel = kernel
      self.gp3 = []
      self.err3 = []
    num_successes = 0
    for i, dataset in enumerate(self.datasets):
      gp1 = get_gp_func1(dataset)
      preds1, _ = gp1.eval(dataset[3])
      err1 = compute_average_prediction_error(dataset, preds1)
      gp2 = get_gp_func2(dataset, method=method)
      preds2, _ = gp2.eval(dataset[3])
      err2 = compute_average_prediction_error(dataset, preds2)
      if not self.rand:
        self.gp3.append(get_gp_func2(dataset, method=None))
        preds3, _ = self.gp3[i].eval(dataset[3])
        self.err3.append(compute_average_prediction_error(dataset, preds3))
      success = err2 <= err1 and err2 <= self.err3[i]
      self.report('(N,D)=%s:: %s-err=%0.4f, %s-err=%0.4f, %s-err=%0.4f, succ=%d'%(
          str(dataset[0].shape), descr1, err1, descr2, err2, descr3, self.err3[i],
          success), 'test_result')
      self.report('  -- Sampling GP: %s'%(str(gp2)), 'test_result')
      self.report('  -- Direct GP: %s'%(str(self.gp3[i])), 'test_result')
      num_successes += success
    self.rand = True
    assert num_successes > 0.6 *len(self.datasets)


# Test for naive predictions vs sampling fitted predictions vs direct fitted predictions
class EuclideanGPFitterTestCase(EuclideanGPFitterBaseTestClass, BaseTestClass):
  """ Unit tests for the EuclideanGP and EuclideanGPFitter class. """
  # pylint: disable=too-many-locals

  def test_se_prediction_slice(self):
    """ Tests for prediction on a test set with an SEGP using slice sampling. """
    self.report('Prediction for an SE kernel using slice sampling. '
                'Probabilistic test, might fail.')
    self._prediction_test(build_se_gp_with_dataset, fit_se_gp_with_dataset,
                          'naive', 'sampling-fit', 'direct-fit', 'se')

  def test_se_prediction_nuts(self):
    """ Tests for prediction on a test set with an SEGP using nuts sampling. """
    self.report('Prediction for an SE kernel using nuts sampling. '
                'Probabilistic test, might fail.')
    self._prediction_test(build_se_gp_with_dataset, fit_se_gp_with_dataset,
                          'naive', 'sampling-fit', 'direct-fit', 'se', 'nuts')

  def test_matern_prediction_slice(self):
    """ Tests for prediction on a test set with an Matern GP using slice sampling. """
    self.report('Prediction for an Matern kernel using slice sampling. '
                'Probabilistic test, might fail.')
    self._prediction_test(build_matern_gp_with_dataset, fit_matern_gp_with_dataset,
                          'naive', 'sampling-fit', 'direct-fit', 'matern')

  def test_matern_prediction_nuts(self):
    """ Tests for prediction on a test set with an Matern GP using nuts sampling. """
    self.report('Prediction for an Matern kernel using nuts sampling. '
                'Probabilistic test, might fail.')
    self._prediction_test(build_matern_gp_with_dataset, fit_matern_gp_with_dataset,
                          'naive', 'sampling-fit', 'direct-fit', 'matern', 'nuts')


if __name__ == '__main__':
  execute_tests()
