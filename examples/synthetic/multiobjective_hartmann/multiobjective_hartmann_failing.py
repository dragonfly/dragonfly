"""
  Multi-objective version of the Hartmann functions, with random failures.
"""

# pylint: disable=invalid-name

try:
  from multiobjective_hartmann import hartmann3_by_2_1, hartmann6, hartmann3_by_2_2
except ImportError:
  from .multiobjective_hartmann import hartmann3_by_2_1, hartmann6, hartmann3_by_2_2

import random
from dragonfly.exd.exd_utils import EVAL_ERROR_CODE

random.seed(1)

num_objectives = 3
def compute_objectives(x):
  """ Computes the objectives. """
  if random.random() < 0.25:
      return [hartmann3_by_2_1(x), hartmann6(x), hartmann3_by_2_2(x)]
  else:
      #alternatively: raise Exception()
      return EVAL_ERROR_CODE
