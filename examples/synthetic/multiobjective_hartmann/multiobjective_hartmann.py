"""
  Multi-objective version of the Hartmann functions.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name


import numpy as np


def _get_01_coords(x):
  """ Returns the coordinates normalised to (0,1). """
  return [x[1][1]/10.0,
          (x[0] - 224)/100.0,
          x[3][0]/92.0,
          x[2][0],
          x[3][1]/92.0,
          x[1][0]/10.0,
         ]


def hartmann3_01(x):
  """ Hartmann function in 3D with alpha. """
  # pylint: disable=bad-whitespace
  x = np.array(x)
  A = np.array([[3.0, 10, 30],
                [0.1, 10, 35],
                [3.0, 10, 30],
                [0.1, 10, 35]], dtype=np.float64)
  P = 1e-4 * np.array([[3689, 1170, 2673],
                       [4699, 4387, 7470],
                       [1091, 8732, 5547],
                       [381, 5743, 8828]], dtype=np.float64)
  log_sum_terms = (A * (P - x)**2).sum(axis=1)
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  return alpha.dot(np.exp(-log_sum_terms))


def hartmann3_by_2_1_01(x):
  """ Summing two Hartmann3 functions to get a 6D function. """
  return hartmann3_01([x[0], x[4], x[2]]) + hartmann3_01([x[3], x[1], x[5]])

def hartmann3_by_2_1(x):
  """ Compute in [0,1]^6 domain. """
  return hartmann3_by_2_1_01(_get_01_coords(x))


def hartmann3_by_2_2_01(x):
  """ Summing two Hartmann3 functions to get a 6D function. """
  return hartmann3_01([x[5], x[0], x[2]]) + hartmann3_01([x[4], x[1], x[3]])

def hartmann3_by_2_2(x):
  """ Compute in [0,1]^6 domain. """
  return hartmann3_by_2_2_01(_get_01_coords(x))


def hartmann6_01(x):
  """ Hartmann function in 3D. """
  # pylint: disable=bad-whitespace
  A = np.array([[  10,   3,   17, 3.5, 1.7,  8],
                [0.05,  10,   17, 0.1,   8, 14],
                [   3, 3.5,  1.7,  10,  17,  8],
                [  17,   8, 0.05,  10, 0.1, 14]], dtype=np.float64)
  P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091,  381]], dtype=np.float64)
  log_sum_terms = (A * (P - x)**2).sum(axis=1)
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  return alpha.dot(np.exp(-log_sum_terms))

def hartmann6(x):
  """ Compute in [0,1]^6 domain. """
  return hartmann6_01(_get_01_coords(x))


# Define the following
objectives = [hartmann3_by_2_1, hartmann6, hartmann3_by_2_2]

def compute_objectives(x):
  """ Computes the objectives. """
  return [obj(x) for obj in objectives]

def main(x):
  """ Main function. """
  return compute_objectives(x)

