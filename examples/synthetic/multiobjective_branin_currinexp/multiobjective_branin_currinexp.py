"""
  Multiobjective Branin and Currin Exponential Functions.
  -- kandasamy@cs.cmu.edu
"""

# You should have one of the following two in this file:
#   1. objectives: A list of functions, each of which corresponds to an objective.
#   2. A function called compute_objectives which returns the value of all objectives when
#      called as a list, and an integer called num_objectives which indicates the number
#      of objectives.

# pylint: disable=invalid-name

import numpy as np


def branin_with_params(x, a, b, c, r, s, t):
  """ Computes the Branin function. """
  x1 = x[0]
  x2 = x[1]
  neg_ret = float(a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)
  return - float(neg_ret)

def branin(x):
  """ Branin function."""
  a = 1
  b = 5.1/(4*np.pi**2)
  c = 5/np.pi
  r = 6
  s = 10
  t = 1/(8*np.pi)
  return branin_with_params(x, a, b, c, r, s, t)


def currin_exp_01(x):
  """ Currin exponential function. """
  x1 = x[0]
  x2 = x[1]
  val_1 = 1 - np.exp(-1/(2 * x2))
  val_2 = (2300*x1**3 + 1900*x1**2 + 2092*x1 + 60) / (100*x1**3 + 500*x1**2 + 4*x1 + 20)
  return float(val_1 * val_2)


def currin_exp(x):
  """ Currint exponential in branin bounds. """
  return currin_exp_01([x[0] * 15 - 5, x[1] * 15])



# # Option 1: Define objectives to be a list of the functions
# objectives = [branin, currin_exp]

# Option 2: Define num_objectives to be the number of objectives and compute_objectives to
# return a list of values of each function.
num_objectives = 2
def compute_objectives(x):
  """ Computes the objectives. """
  return [branin(x), currin_exp_01(x)]

