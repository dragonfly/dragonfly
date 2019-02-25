"""
  A synthetic function on CNNs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import sys
sys.path.append('..')
import numpy as np
# Local
from ..park2_4.park2_4 import sub_park_1, sub_park_2, sub_park_3, sub_park_4
from ...nn.syn_nn_functions import syn_func1_common


def _chooser_park(chooser, y1, y2, y3, y4):
  """ Internal park function. """
  x = [y1, y2, y3, y4]
  if chooser == 'foofoo':
    return sub_park_1(x)
  elif chooser == 'foobar':
    return sub_park_2(x)
  elif chooser == 'barfoo':
    return sub_park_3(x)
  elif chooser == 'barbar':
    return sub_park_4(x)
  else:
    raise ValueError('Unknown chooser %s.'%(chooser))


def _park(z1, z2, x1, x2, x3, x4):
  """ Internal park function. """
  x1 = max(1e-4, x1)
  ret1 = (x1/2) * (np.sqrt(1 + (x2 + x3**2)*x4/(x1**2)) - 1) * z1
  ret2 = (x1 + 3.2*x4) * np.exp(1 + np.sin(x3)) * z2
  return ret1 + ret2


def syn_cnn_2(x):
  """ Computes the Branin function. """
  return syn_cnn_2_z_x([1.0, 1.0], x)


def syn_cnn_2_z_x(z, x):
  """ Computes the Branin function. """
  cnn = x[2]
  chooser = x[1][0] + x[1][1]
  y1 = x[0][0]
  y2 = x[0][1]
  y3 = (x[3] - 10.0) / 4
  y4 = x[4]/100
  # compute individual functions
  a = syn_func1_common(cnn) * z[0]
  b = _chooser_park(chooser, y1, y2, y3, y4)
  c = _park(z[0], z[1], y2, y1, y4, y3)
  return a * b + c


# Write a function like this called 'obj'.
def objective(x):
  """ Objective. """
  return syn_cnn_2(x)


def main(x):
  """ main function. """
  return syn_cnn_2(x)

