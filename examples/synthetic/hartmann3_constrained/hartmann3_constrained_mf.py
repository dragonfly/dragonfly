"""
  Hartmann function in 3 dimensions with constraints.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import numpy as np

OPT_TO_FIDEL_PT = 9.1


def hartmann3_constrained_mf(z, x):
  """ Exactly the same as the Hartmann3 function. """
  pt = np.array(x)
  A = np.array([[3.0, 10, 30],
                [0.1, 10, 35],
                [3.0, 10, 30],
                [0.1, 10, 35]], dtype=np.float64)
  P = 1e-4 * np.array([[3689, 1170, 2673],
                       [4699, 4387, 7470],
                       [1091, 8732, 5547],
                       [381, 5743, 8828]], dtype=np.float64)
  log_sum_terms = (A * (P - pt)**2).sum(axis=1)
  alpha = np.array([1.0, 1.2, 3.0, 3.2]) - 0.01 * abs(z[0] - OPT_TO_FIDEL_PT)
  return alpha.dot(np.exp(-log_sum_terms))


def cost(z):
  """ Cost function """
  return z[0]/OPT_TO_FIDEL_PT


# Write a function like this called objective.
def objective(z, x):
  """ Objective. """
  return hartmann3_constrained_mf(z, x)


def main(z, x):
  """ Main function. """
  return objective(z, x), cost(z)

