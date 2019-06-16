"""
  Hartmann function in 6D (4 inputs) with a 3D (2 input) fidelity space.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import numpy as np
# Local
try:
  from hartmann6_4 import hartmann6_4_alpha
except ImportError:
  from .hartmann6_4 import hartmann6_4_alpha

def hartmann6_4_mf(z, x):
  """ Hartmann function in 3D. """
  f0 = z[0][0] / (9467.18 - 1234.9)
  f1 = z[0][1] / (9467.18 - 1234.9)
  f2 = (len(z[1]) + 1.5 * float('a' in z[1]) + 0.5 * float('i' in z[1])) / 4.5
  f3 = z[2][0] / float(234 - 123)
  f = np.array([f0, f1, f2, f3])
  delta = np.array([0.1] * 4)
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  alpha_z = alpha - delta * (1 - f)
  return hartmann6_4_alpha(x, alpha_z)

def cost(z):
  """ Cost function. """
  return 0.1 + 0.9/4 * (z[0][0]/9467.18 + (z[0][1]/9467.18)**2 + float(z[1] == 'ghij') \
                        + ((z[2][0] - 123.0)/(234.0 - 123.0))**3)


# Write a function like this called obj.
def objective(z, x):
  """ Main function. """
  return hartmann6_4_mf(z, x)

def main(z, x):
  """ Main function. """
  return hartmann6_4_mf(z, x), cost(z)

