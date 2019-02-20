"""
  Parkd function.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import numpy as np

def sub_park_1(x):
  """ Computes the park function """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  x4 = x[3]
  ret = (2.0/3.0) * np.exp(x1 + x2) - x4*np.sin(x3) + x3
  return ret

def sub_park_2(x):
  """ Computes the park function """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  x4 = x[3]
  ret = (2.0/3.0) * np.exp(x1 + 2*x2) - x4*np.sin(x3) + x3
  return ret

def sub_park_3(x):
  """ Computes park function """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  x4 = x[3]
  ret = (2.5/3.0) * np.exp(2*x1 + x2) - x4*np.sin(x3) + x3
  return ret

def sub_park_4(x):
  """ Computes park function """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  x4 = x[3]
  ret = (1.7/3.0) * np.exp(1.3*x1 + x2) - x4*np.sin(1.1*x3) + x3
  return ret

def park2_4_z(z, x):
  """ Computes the Parkd function. """
  y1 = x[0][0]
  y2 = x[0][1]
  chooser = x[1]
  y3 = (x[2] - 103.0) / 91.0
  y4 = x[3] + 10.0
  x = [y1, y2, y3, y4]
  if chooser == 'rabbit':
    ret = sub_park_1(x)
  elif chooser == 'dog':
    ret = sub_park_2(x)
  elif chooser == 'gerbil':
    ret = sub_park_3(x)
  elif chooser in ['hamster', 'ferret']:
    ret = sub_park_4(x)
  return ret * np.exp(z - 1)

def park2_4(x):
  """ Computes the Parkd function. """
  return park2_4_z(0.93242, x)

# Write a function like this called obj.
def objective(x):
  """ Objective. """
  return park2_4(x)


def main(x):
  """ main function. """
  return park2_4(x)

