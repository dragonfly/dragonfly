"""
  A demo of neural architecture search using the NASBOT algorithm on an MLP
  (Multi-Layer Perceptron) architecture search problem.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from __future__ import print_function
from argparse import Namespace
import time
import os
import shutil
# Local
from mlp_function_caller import MLPFunctionCaller
# from cnn_function_caller import CNNFunctionCaller
from dragonfly.opt.gp_bandit import bo_from_func_caller
from dragonfly.exd.worker_manager import MultiProcessingWorkerManager
from dragonfly.utils.reporters import get_reporter
# Visualise
try:
  from dragonfly.nn.nn_visualise import visualise_nn
except ImportError as e:
  print(e)
  visualise_nn = None

# Data - MLP
# For the MLP experiments, the data should be in a pickle file stored as a dictionary.
# The 'train' key should point to the training data while 'vali' points to the validation
# data. For example, after data = pic.load(file_name), data['train']['x'] should point
# to the features of the training data.
# The slice and indoor_location datasets are available at
# http://www.cs.cmu.edu/~kkandasa/dragonfly_datasets.html as examples.
# Download them into this directory to run the demo.

# Data - CNN
# We use the Cifar10 dataset which is converted to .tfrecords format for tensorflow.
# You can either download the original dataset from www.cs.toronto.edu/~kriz/cifar.html
# and follow the instructions in
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.p
# Alternatively, they are available in the required format at
# www.cs.cmu.edu/~kkandasa/nasbot_datasets.html as examples.
# Put the xxx.tfrecords in a directory named cifar-10-data in the demos directory to run
# this demo.

# Results
# The progress of optimization will be logged in mlp_experiment_dir_<time>/log where
# <time> is a time stamp.

# DATASET = 'cifar10';
DATASET = 'slice';
# DATASET = 'indoor'

# Which GPU IDs are available
GPU_IDS = [0]
# GPU_IDS = [1, 2]

# Config file which specifies the domain
CIFAR_DATA_DIR = 'cifar-10-data'
MLP_CONFIG_FILE = 'config_mlp_mf.json'
CNN_CONFIG_FILE = 'config_cnn_mf.json'

# Where to store temporary model checkpoints
EXP_DIR = 'experiment_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))
LOG_FILE = os.path.join(EXP_DIR, 'log')
TMP_DIR = './tmp_' + DATASET

# Function to return the name of the file containing dataset
def get_train_file_name(dataset):
  """ Return train params. """
  # get file name
  if dataset == 'slice':
    train_pickle_file = 'slice_localisation.p'
  elif dataset == 'indoor':
    train_pickle_file = 'indoor_location.p'
  return train_pickle_file

# Specify the budget (in seconds) -- this is 8 hours
BUDGET = 8 * 60 * 60

# Obtain a reporter object

def main():
  """ Main function. """
  # Make directories
  if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
  if os.path.exists(EXP_DIR):
    shutil.rmtree(EXP_DIR)
  os.mkdir(TMP_DIR)
  os.mkdir(EXP_DIR)
  # Obtain a reporter
  reporter = get_reporter(open(LOG_FILE, 'w')) # Writes to file log_mlp

  # First, obtain a function caller: A function_caller is used to evaluate a function
  # defined on a given domain (and a fidelity space). The train_params
  # can be used to specify additional training parameters such as the batch size etc.
  if DATASET == 'cifar10':
    # We have defined the CNNFunctionCaller in cnn_function_caller.py.
    train_params = Namespace(data_dir=CIFAR_DATA_DIR)
    func_caller = CNNFunctionCaller(CNN_CONFIG_FILE, train_params, reporter=reporter,
                                    tmp_dir=TMP_DIR)
  else:
    # We have defined the MLPFunctionCaller in mlp_function_caller.py.
    train_params = Namespace(data_train_file=get_train_file_name(DATASET))
    func_caller = MLPFunctionCaller(MLP_CONFIG_FILE, train_params, reporter=reporter,
                                    tmp_dir=TMP_DIR)
  # Obtain a worker manager: A worker manager (defined in opt/worker_manager.py) is used
  # to manage (possibly) multiple workers. For a MultiProcessingWorkerManager,
  # the budget should be given in wall clock seconds.
  worker_manager = MultiProcessingWorkerManager(GPU_IDS, EXP_DIR)

  # Run the optimiser
  opt_val, opt_point, _ = bo_from_func_caller(func_caller, worker_manager, BUDGET,
                                           is_mf=True, reporter=reporter)
  # Convert to "raw" format
  raw_opt_point = func_caller.get_raw_domain_point_from_processed(opt_point)
  opt_nn = raw_opt_point[0] # Because first index in the config file is the neural net.

  # Print the optimal value and visualise the best network.
  reporter.writeln('\nOptimum value found: %0.5f'%(opt_val))
  if visualise_nn is not None:
    visualise_file = os.path.join(EXP_DIR, 'optimal_network')
    reporter.writeln('Optimal network visualised in %s.eps.'%(visualise_file))
    visualise_nn(opt_nn, visualise_file)
  else:
    reporter.writeln('Install graphviz (pip install graphviz) to visualise the network.')


if __name__ == '__main__':
  main()

