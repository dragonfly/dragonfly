"""
  Function caller for the MLP experiments.
  -- willie@cs.cmu.edu
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=arguments-differ
# pylint: disable=invalid-name

import sys

if sys.version_info[0] < 3:
  import cPickle as pic
else:
  import pickle as pic
from copy import deepcopy
import os
import shutil
import tempfile
from time import sleep
# Local
from nn_function_caller import NNFunctionCaller
from cg import run_tensorflow

_MAX_TRIES = 3
_SLEEP_BETWEEN_TRIES_SECS = 3


def get_default_mlp_tf_params():
  """ Default MLP training parameters for tensorflow. """
  return {
    'trainBatchSize':256,
    'valiBatchSize':1000,
    'trainNumStepsPerLoop':100,
    'valiNumStepsPerLoop':5,
    }


class MLPFunctionCaller(NNFunctionCaller):
  """ Function caller to be used in the MLP experiments. """

  def __init__(self, config, train_params, descr='', debug_mode=False,
               reporter='silent', tmp_dir='/tmp_mlp'):
    """ Constructor. """
    super(MLPFunctionCaller, self).__init__(config, train_params, descr, debug_mode,
                                            reporter)
    self.root_tmp_dir = tmp_dir
    # Load data
    with open(self.train_params.data_train_file, 'rb') as input_file:
      if sys.version_info[0] < 3:
        data = pic.load(input_file)
      else:
        data = pic.load(input_file, encoding='latin1')
    self.data_train = data['train']
    self.data_vali = data['vali']
    self.reporter.writeln('Loaded data: ' + self.train_params.data_train_file)
    self.reporter.writeln('Training data shape: ' + 'x: ' +
                          str(self.data_train['x'].shape) +
                          ', ' + 'y: ' + str(self.data_train['y'].shape))
    self.reporter.writeln('Validation data shape: ' + 'x: ' +
                          str(self.data_vali['x'].shape) +
                          ', ' + 'y: ' + str(self.data_vali['y'].shape))
    # Check tf_params
    if not hasattr(self.train_params, 'tf_params'):
      self.train_params.tf_params = get_default_mlp_tf_params()

  def _eval_validation_score(self, raw_fidel, raw_point, qinfo):
    """ Evaluates the validation score. """
    # pylint: disable=bare-except
    # First extract out the arguments - this has to follow the order in the config file.
    nn = raw_point[0]
    learning_rate = 10 ** raw_point[1]
    num_loops = raw_fidel[0]
    curr_tf_params = deepcopy(self.train_params.tf_params)
    curr_tf_params['learningRate'] = learning_rate
    curr_tf_params['numLoops'] = num_loops
#     self.reporter.writeln('Eval: %s, %0.4f, %0d.'%(str(nn), learning_rate, num_loops))
    # Set up
    os.environ['CUDA_VISIBLE_DEVICES'] = str(qinfo.worker_id)
    num_tries = 0
    succ_eval = False
    while num_tries < _MAX_TRIES and not succ_eval:
      try:
        tmp_dir = tempfile.mkdtemp(dir=self.root_tmp_dir)
        vali_score = run_tensorflow.compute_validation_error(nn, self.data_train,
                       self.data_vali, 0, curr_tf_params, tmp_dir)
        succ_eval = True
      except Exception as e:
        sleep(_SLEEP_BETWEEN_TRIES_SECS)
        num_tries += 1
        self.reporter.writeln(' -- Failed on try %d with gpu %d, %s'%(
                              num_tries, qinfo.worker_id, e))
    self._clean_up(tmp_dir)
    return vali_score

  @classmethod
  def _clean_up(cls, tmp_dir):
    """ Clean up files created by tensorflow. """
    # pylint: disable=bare-except
    # pylint: disable=broad-except
    # pylint: disable=unnecessary-pass
    try:
      shutil.rmtree(tmp_dir)
    except:
      pass

