"""
  Hartmann function in 3 dimensions.
  This function assumes that the input is a list of 2 elements where the first is a one
  dimensional object and the second is a two dimensional object. The purpose is to
  demonstrate setting up complex spaces in dragonfly.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=invalid-name

import numpy as np
# Local
try:
  from hartmann3_2 import hartmann3_2_alpha
except ImportError:
  from .hartmann3_2 import hartmann3_2_alpha


def hartmann3_2_mf(z, x):
  """ Hartmann function in 3D. """
  f0 = z[0][0] / 999.0
  f1 = z[0][1] / 999.0
  f2 = (z[1] + 1)/2.0
  f3 = z[2]
  f = np.array([f0, f1, f2, f3])
  delta = np.array([0.1] * 4)
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  alpha_z = alpha - delta * (1 - f)
  return hartmann3_2_alpha(x, alpha_z)


def cost(z):
  """ Cost function. """
  ret = 0.1 + 0.9 * (z[0][0]/999.0 + (z[0][1]/1000.0)**3 + (z[1] + 1.0)/2.0 * z[2]) / 3.0
  return ret


# Write a function like this called obj.
def objective(z, x):
  """ Objective. """
  return hartmann3_2_mf(z, x)

def main(z, x):
  """ Main function. """
  return hartmann3_2_mf(z, x), cost(z)

