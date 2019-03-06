"""
  Calls Tree based ensemble methods for regression/classification and returns either
  classification or regression error/accuracy.
  -- kandasamy@cs.cmu.edu
"""

from __future__ import print_function
# pylint: disable=invalid-name

import numpy as np
from sklearn import ensemble


def get_tr_dataset_size_from_z0(z0):
  """ Returns the training dataset size. """
  return int(np.round(np.exp(z0)))


def _get_avg_regression_error(Y1, Y2):
  """ Returns the negative prediction error. """
  if len(Y1) != len(Y2):
    raise ValueError('Lenghts of Y1 (%d), and Y2 (%d) should be the same.'%(
                      len(Y1), len(Y2)))
  num_data = float(len(Y1))
  differences = np.array(Y1) - np.array(Y2)
  finite_diffs = differences[np.isfinite(differences)]
  if len(finite_diffs) > 0:
    max_finite_diff = finite_diffs.max()
  else:
    max_finite_diff = 10.0
  differences[np.logical_not(np.isfinite(differences))] = \
    np.clip(10*max_finite_diff, 10.0, 100.0)
  diff_sq = np.linalg.norm(differences)**2
  ret = diff_sq / num_data
  ret = min(ret, 100.0)
  return ret


def _get_avg_classification_error(Y1, Y2):
  """ Return the classification error. """
  if len(Y1) != len(Y2):
    raise ValueError('Lenghts of Y1 (%d), and Y2 (%d) should be the same.'%(
                      len(Y1), len(Y2)))
  num_data = float(len(Y1))
  unequal_posns = [y1 != y2 for (y1, y2) in zip(Y1, Y2)]
  ret = sum(unequal_posns) / num_data
  if not np.isfinite(ret):
    ret = 1.0
  return ret


def skltree_train_and_validate(method, X_tr, Y_tr, X_va, Y_va, hyperparam_dict,
                               num_tr_data_to_use=None, num_va_data_to_use=None,
                               shuffle_data_when_using_a_subset=False,
                               error_or_accuracy='accuracy'):
  """ Inputs:
        method: should be one of gbc, gbr, rfc, rfr which stand for
          gradient boosted classification, gradient boosted regression,
          random forest classification, random forest regression respectively.
        X_tr, Y_tr, X_va, Y_va: Data.
        hyperparam_dict: dictionary of hyper-parameters.
        num_tr_data_to_use, num_va_data_to_use: how much of the training and validation
          sets to use for training.
        shuffle_data_when_using_a_subset: if True, will shuffle dataset.
  """
  # Prelims
  if method in ['gbc', 'rfc']:
    prob_type = 'classification'
  elif method in ['gbr', 'rfr']:
    prob_type = 'regression'
  else:
    raise ValueError('Unknown method %s.'%(method))
  if num_tr_data_to_use is None:
    num_tr_data_to_use = len(X_tr)
  if num_va_data_to_use is None:
    num_va_data_to_use = len(X_va)
  num_tr_data_to_use = int(num_tr_data_to_use)
  num_va_data_to_use = int(num_va_data_to_use)
  if num_tr_data_to_use < len(X_tr) and shuffle_data_when_using_a_subset:
    X_tr = np.copy(X_tr)
    np.random.shuffle(X_tr)
  if num_va_data_to_use < len(X_va) and shuffle_data_when_using_a_subset:
    X_va = np.copy(X_va)
    np.random.shuffle(X_va)
  # Get relevant subsets
  X_tr = X_tr[:num_tr_data_to_use]
  Y_tr = Y_tr[:num_tr_data_to_use]
  X_va = X_va[:num_va_data_to_use]
  Y_va = Y_va[:num_va_data_to_use]
  # Train method
  method_to_construct_dict = {'gbr': ensemble.GradientBoostingRegressor,
                              'gbc': ensemble.GradientBoostingClassifier,
                              'rfr': ensemble.RandomForestRegressor,
                              'rfc': ensemble.RandomForestClassifier,
                             }
  model_constructor = method_to_construct_dict[method]
  model = model_constructor(**hyperparam_dict)
  model.fit(X_tr, Y_tr)
  # Compute validation error
  predictions = model.predict(X_va)
  if prob_type == 'classification':
    ret = _get_avg_classification_error(predictions, Y_va)
    ret = ret if error_or_accuracy == 'error' else (1 - ret)
  elif prob_type == 'regression':
    ret = _get_avg_regression_error(predictions, Y_va)
    ret = ret if error_or_accuracy == 'error' else -ret
  # Return
  return ret


def skltree_train_and_validate_wrapper(method, hyperparam_dict, data,
  num_tr_data_to_use, max_tr_data_size, max_va_data_size, *args, **kwargs):
  """ A wrapper to do some common operations. """
  X_tr = data['train']['x'][:max_tr_data_size]
  Y_tr = data['train']['y'][:max_tr_data_size]
  X_va = data['vali']['x'][:max_va_data_size]
  Y_va = data['vali']['y'][:max_va_data_size]
  return skltree_train_and_validate(method, X_tr, Y_tr, X_va, Y_va, hyperparam_dict,
                                    num_tr_data_to_use=num_tr_data_to_use,
                                    *args, **kwargs)


def gbr_train_and_validate(hyperparam_vect, data, num_tr_data_to_use,
                           max_tr_data_size, max_va_data_size, *args, **kwargs):
  """ Train and validate using gradient boosted regression.
      The order of elements in hyperparam_vect should follow config_gbr_mf.json.
  """
  method = 'gbr'
  hyperparam_dict = {'loss':hyperparam_vect[0],
                     'learning_rate':10**hyperparam_vect[1],
                     'n_estimators':hyperparam_vect[2],
                     'subsample':hyperparam_vect[3],
                     'criterion':hyperparam_vect[4],
                     'min_samples_split':hyperparam_vect[5],
                     'min_samples_leaf':hyperparam_vect[6],
                     'max_depth':hyperparam_vect[7],
                    }
  return skltree_train_and_validate_wrapper(method, hyperparam_dict, data,
           num_tr_data_to_use, max_tr_data_size, max_va_data_size, *args, **kwargs)


def rfr_train_and_validate(hyperparam_vect, data, num_tr_data_to_use,
                           max_tr_data_size, max_va_data_size, *args, **kwargs):
  """ Train and validate using random forest regression.
      The order of elements in hyperparam_vect should follow config_gbr_mf.json.
  """
  method = 'rfr'
  hyperparam_dict = {'n_estimators':hyperparam_vect[0],
                     'criterion':hyperparam_vect[1],
                     'max_depth':hyperparam_vect[2],
                     'min_samples_split':hyperparam_vect[3],
                     'min_samples_leaf':hyperparam_vect[4],
                     'max_features':hyperparam_vect[5],
                    }
  return skltree_train_and_validate_wrapper(method, hyperparam_dict, data,
           num_tr_data_to_use, max_tr_data_size, max_va_data_size, *args, **kwargs)

