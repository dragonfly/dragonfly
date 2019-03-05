"""
  Tuning the hyperparameters of Gradient boosted classification on the Protein structure
  prediction data.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=unexpected-keyword-arg

import pickle
# Local
from skltree import gbr_train_and_validate, get_tr_dataset_size_from_z0

try:
  import os
  import sys
  file_name = 'naval_propulsion.p'
  curr_dir_path = os.path.dirname(os.path.realpath(__file__))
  data_path = os.path.join(curr_dir_path, file_name)
  if sys.version_info[0] < 3:
    DATA = pickle.load(open(data_path, 'rb'))
  else:
    DATA = pickle.load(open(data_path, 'rb'), encoding='latin1')
except IOError:
  print(('Could not load file %s. Make sure the file %s is in the same directory as ' +
         'this file or pass the dataset to the function.')%(file_name, data_path))

MAX_TR_DATA_SIZE = 9000
MAX_VA_DATA_SIZE = 2000


def objective(z, x):
  """ Objective. """
  num_tr_data_to_use = get_tr_dataset_size_from_z0(z[0])
  return gbr_train_and_validate(x, DATA, num_tr_data_to_use,
                                MAX_TR_DATA_SIZE, MAX_VA_DATA_SIZE)

def cost(z):
  """ Compute cost. """
  num_tr_data_to_use = get_tr_dataset_size_from_z0(z[0])
  return num_tr_data_to_use / float(MAX_TR_DATA_SIZE)

