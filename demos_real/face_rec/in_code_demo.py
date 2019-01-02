"""
  Face rec function.
  -- kvysyara@andrew.cmu.edu
"""

from __future__ import print_function
from dragonfly import maximise_function
# Local imports
from demos_real.face_rec.face_rec import face_rec

def main():
  """ Main function. """
  # pylint: disable=unused-variable
  domain_bounds = [[1, 500], [0, 1000], [0, 1]]
  max_capital = 25
  opt_val, opt_pt, history = maximise_function(face_rec, domain_bounds, max_capital)
  print('Optimum Value in %d evals: %0.4f'%(max_capital, opt_val))
  print('Optimum Point: %s'%(opt_pt))

if __name__ == '__main__':
  main()

