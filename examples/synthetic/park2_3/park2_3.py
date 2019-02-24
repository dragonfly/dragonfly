"""
  Park2 function.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=invalid-name

import numpy as np
# Local imports
from synthetic.park2_4.park2_4 import sub_park_1, sub_park_2, sub_park_3, sub_park_4

def park2_3(x):
  """ Computes the Park2_3 function. """
  return park2_3_z_x([1.0, 1.0], x)

def park2_3_z_x(z, x):
  """ Computes the Park2_3 function. """
  chooser = x[0][0] + x[0][1]
  y1 = x[1][0] * np.sqrt(z[0])
  y2 = x[1][1] * np.sqrt(z[0])
  y3 = (x[2][0] - 10.0) / 4
  y4 = (x[2][1] - 10.0) / 4
  x = [y1, y2, y3, y4]
  if chooser == 'foofoo':
    return sub_park_1(x) * np.sqrt(z[1])
  elif chooser == 'foobar':
    return sub_park_2(x) * np.sqrt(z[1])
  elif chooser == 'barfoo':
    return sub_park_3(x) * np.sqrt(z[1])
  elif chooser == 'barbar':
    return sub_park_4(x) * np.sqrt(z[1])
  else:
    raise ValueError('Unknown chooser %s.'%(chooser))

# Write a function like this called 'obj'.
def objective(x):
  """ Objective. """
  return park2_3(x)

def main(x):
  """ main function. """
  return park2_3(x)

