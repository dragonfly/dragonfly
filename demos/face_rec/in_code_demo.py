"""
  Face rec function.
  -- kvysyara@andrew.cmu.edu
"""

from __future__ import print_function
import numpy as np
import math
from maximise_function import maximise_function
# Local imports
from demos.face_rec.face_rec import face_rec

def main():
  domain_bounds = [[1, 500], [0, 1000], [0, 1]]
  max_capital = 25
  opt_val, opt_pt = maximise_function(face_rec, domain_bounds, max_capital,
                                      hp_tune_criterion='post_sampling',
                                      hp_tune_method='slice')
  print('Optimum Value in %d evals: %0.4f'%(max_capital, opt_val))
  print('Optimum Point: %s'%(opt_pt))

if __name__ == '__main__':
  main()

