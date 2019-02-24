"""
  A simple test for the fortran direct library.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=relative-import

from __future__ import print_function
import numpy as np
try:
  import direct
  RUN_TESTS = True
except ImportError:
  RUN_TESTS = False


def main():
  """ Main function. """
  dim = 8
  obj = lambda x: (np.dot(x-0.1, x), 0)
  lower_bounds = [-1] * dim
  upper_bounds = [1] * dim
  dim = len(lower_bounds)
  eps = 1e-5
  max_func_evals = 5000
  max_iterations = max_func_evals
  algmethod = 0
  _log_file = 'log_dir_file'
  _kky_results_file = 'results_direct_file'
  fglobal = -1e100
  fglper = 0.01
  volper = -1.0
  sigmaper = -1.0

  def _objective_wrap(x, iidata, ddata, cdata, n, iisize, idsize, icsize):
    """ A wrapper to comply with the fortran requirements. """
    return obj(x)

  iidata = np.ones(0, dtype=np.int32)
  ddata = np.ones(0, dtype=np.float64)
  cdata = np.ones([0, 40], dtype=np.uint8)


  soln = direct.direct(_objective_wrap,
                       eps,
                       max_func_evals,
                       max_iterations,
                       np.array(lower_bounds, dtype=np.float64),
                       np.array(upper_bounds, dtype=np.float64),
                       algmethod,
                       _log_file,
                       _kky_results_file,
                       fglobal,
                       fglper,
                       volper,
                       sigmaper,
                       iidata,
                       ddata,
                       cdata
                      )
  print('Solution: %s'%(str(soln)))


if __name__ == '__main__':
  if RUN_TESTS:
    main()
  else:
    pass
