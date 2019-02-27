"""
  Function caller for the CNN experiments.
  -- willie@cs.cmu.edu
"""

# pylint: disable=arguments-differ
# pylint: disable=invalid-name

from copy import deepcopy
import os
import shutil
import tempfile
from time import sleep
# Local
from nn_function_caller import NNFunctionCaller
from cg.cifar import run_tensorflow_cifar


_MAX_TRIES = 3
_SLEEP_BETWEEN_TRIES_SECS = 3

def get_default_cnn_tf_params():
  """ Default MLP training parameters for tensorflow. """
  return {
    'trainBatchSize':32,
    'valiBatchSize':32,
    'trainNumStepsPerLoop':2000,
    'valiNumStepsPerLoop':616,
    }

class CNNFunctionCaller(NNFunctionCaller):
  """ Function caller to be used in the MLP experiments. """

  def __init__(self, config, train_params, descr='', debug_mode=False,
               reporter='silent', tmp_dir='/tmp_cnn'):
    """ Constructor. """
    super(CNNFunctionCaller, self).__init__(config, train_params, descr, debug_mode,
                                            reporter)
    self.root_tmp_dir = tmp_dir
    self.data_file_str = self.train_params.data_dir
    # Check tf_params
    if not hasattr(self.train_params, 'tf_params'):
      self.train_params.tf_params = get_default_cnn_tf_params()

  def _eval_validation_score(self, raw_fidel, raw_point, qinfo):
    """ Evaluates the validation score. """
    # pylint: disable=unused-argument
    # pylint: disable=bare-except
    # Extract out the arguments
    nn = raw_point[0]
    learning_rate = 10 ** raw_point[1]
    num_loops = raw_fidel[0]
    curr_tf_params = deepcopy(self.train_params.tf_params)
    curr_tf_params['learningRate'] = learning_rate
    curr_tf_params['numLoops'] = num_loops
#     self.reporter.writeln('Eval: %s, %0.8f, %0d.'%(str(nn), learning_rate, num_loops))
    # Set up
    os.environ['CUDA_VISIBLE_DEVICES'] = str(qinfo.worker_id)
    num_tries = 0
    succ_eval = False
    while num_tries < _MAX_TRIES and not succ_eval:
      try:
        tmp_dir = tempfile.mkdtemp(dir=self.root_tmp_dir)
        vali_error = run_tensorflow_cifar.compute_validation_error(nn, self.data_file_str,
                      qinfo.worker_id, curr_tf_params, tmp_dir)
        succ_eval = True
      except Exception as e:
        sleep(_SLEEP_BETWEEN_TRIES_SECS)
        num_tries += 1
        self.reporter.writeln(' -- Failed on try %d with gpu %d: %s.'%(
                              num_tries, qinfo.worker_id, e))
    self._clean_up(tmp_dir)
    return vali_error

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

