"""
  Hartmann function in 3 dimensions.
  This function assumes that the input is a list of 2 elements where the first is a one
  dimensional object and the second is a two dimensional object. The purpose is to
  demonstrate setting up complex spaces in dragonfly.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import numpy as np


def hartmann3_2(x):
  """ Hartmann function in 3D. """
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  return hartmann3_2_alpha(x, alpha)


def hartmann3_2_alpha(x, alpha):
  """ Hartmann function in 3D with alpha. """
  pt = np.array([x[0][1], x[1][0], x[0][0]])
  A = np.array([[3.0, 10, 30],
                [0.1, 10, 35],
                [3.0, 10, 30],
                [0.1, 10, 35]], dtype=np.float64)
  P = 1e-4 * np.array([[3689, 1170, 2673],
                       [4699, 4387, 7470],
                       [1091, 8732, 5547],
                       [381, 5743, 8828]], dtype=np.float64)
  log_sum_terms = (A * (P - pt)**2).sum(axis=1)
  return alpha.dot(np.exp(-log_sum_terms))


# Write a function like this called obj.
def objective(x):
  """ Objective. """
  return hartmann3_2(x)

def main(x):
  """ Main function. """
  return hartmann3_2(x)

