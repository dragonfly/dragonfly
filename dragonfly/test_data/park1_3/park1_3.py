"""
  Park1 function with three domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import numpy as np

def park1_3(x):
  """ Computes the park1 function. """
  return park1_3_z_x([1.0, 1.0, 1.0], x)

def park1_3_z_x(z, x):
  """ Computes the park1 function. """
  x1 = max(x[0][0], 0.01) * np.sqrt(z[0])
  x2 = x[0][1] * np.sqrt(z[1])
  x3 = x[1]/100 * np.sqrt(z[2])
  x4 = (x[2] - 10)/6.0 * np.sqrt((z[0] + z[1] + z[2]) / 3.0)
  ret1 = (x1/2) * (np.sqrt(1 + (x2 + x3**2)*x4/(x1**2)) - 1)
  ret2 = (x1 + 3*x4) * np.exp(1 + np.sin(x3))
  return ret1 + ret2

# Write a function like this called obj.
def objective(x):
  """ Objective. """
  return park1_3(x)

def main(x):
  """ main function. """
  return park1_3(x)

