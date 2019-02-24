"""
  A synthetic function on CNNs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

# Local
from dragonfly.nn.syn_nn_functions import cnn_syn_func1

def syn_cnn_1(x):
  """ Computes the Branin function. """
  return cnn_syn_func1(x[0])


# Write a function like this called 'obj'.
def objective(x):
  """ Objective. """
  return syn_cnn_1(x)


def main(x):
  """ main function. """
  return syn_cnn_1(x)

