"""
  Hartmann function in 6 dimensions.
  This function assumes that the input is a list of 4 elements in the following order a
  one dimensional int, a 2D float, a 1D float, a 2D int.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=bad-whitespace

import numpy as np

def hartmann6_4(x):
  """ Hartmann function in 3D. """
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  return hartmann6_4_alpha(x, alpha)

def hartmann6_4_alpha(x, alpha):
  """ Hartmann function in 3D. """
  pt = np.array([x[1][1]/10.0,
                 (x[0][0] - 224)/100.0,
                 x[3][0]/92.0,
                 x[2][0],
                 x[3][1]/92.0,
                 x[1][0]/10.0,
                ])
  A = np.array([[  10,   3,   17, 3.5, 1.7,  8],
                [0.05,  10,   17, 0.1,   8, 14],
                [   3, 3.5,  1.7,  10,  17,  8],
                [  17,   8, 0.05,  10, 0.1, 14]], dtype=np.float64)
  P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091,  381]], dtype=np.float64)
  log_sum_terms = (A * (P - pt)**2).sum(axis=1)
  return alpha.dot(np.exp(-log_sum_terms))


# Write a function like this called main
def objective(x):
  """ Main function. """
  return hartmann6_4(x)

def main(x):
  """ Main function. """
  return hartmann6_4(x)

