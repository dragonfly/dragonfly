"""
  Borehole function with constraints.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=bad-whitespace

import numpy as np

def borehole_constrained(x):
  """ Computes the Bore Hole function. """
  return borehole_constrained_z(x, [1.0, 1.0])

def borehole_constrained_z(x, z):
  """ Computes the Bore Hole function. """
  # pylint: disable=bad-whitespace
  rw = x[0][0]
  L  = x[1][0] * (1680 - 1120.0) + 1120
  Kw = x[1][1] * (12045 - 9855) + 9855
  Tu = x[2]
  Tl = x[3]
  Hu = x[4][0]/2.0 + 990.0
  Hl = x[4][1]/2.0 + 700.0
  r  = x[5]
  # Compute high fidelity function
  frac2 = 2*L*Tu/(np.log(r/rw) * rw**2 * Kw + 0.001) * np.exp(z[1] - 1)
  f2 = 2 * np.pi * Tu * (Hu - Hl)/(np.log(r/rw) * (1 + frac2 + Tu/Tl))
  f1 = 5 * Tu * (Hu - Hl)/(np.log(r/rw) * (1.5 + frac2 + Tu/Tl))
  return f2 * z[0] + (1-z[0]) * f1


# Write a function like this called obj.
def objective(x):
  """ Objective. """
  return borehole_constrained(x)

