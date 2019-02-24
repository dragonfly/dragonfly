"""
  A synthetic function on CNNs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import numpy as np
# Local
from dragonfly.nn.syn_nn_functions import cnn_syn_func1


def syn_cnn_1_mf(z, x):
  """ Computes the Branin function. """
  return cnn_syn_func1(x[0]) * (np.exp(z[0] - 5.0))


def cost(z):
  """ Cost function. """
  return z[0] * 1.3


# Write a function like this called 'obj'.
def objective(z, x):
  """ Objective. """
  return syn_cnn_1_mf(z, x)


def main(z, x):
  """ main function. """
  return syn_cnn_1_mf(z, x), cost(z)

