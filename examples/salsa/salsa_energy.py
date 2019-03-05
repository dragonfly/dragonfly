"""
  Demo on the Shrunk Additive Least Squares Approximations method (SALSA) on the energy
  appliances dataset.
  -- kandasamy@cs.cmu.edu

  SALSA is Kernel Ridge Regression with special kind of kernel structure.
  We tune for the following parameters in the method.
    - Kernel type: {se, matern0.5, matern1.5, matern2.5}
    - Additive Order: An integer in (1, d) where d is the dimension.
    - Kernel scale: float
    - Bandwidths for each dimension: float
    - L2 Regularisation penalty: float

  If you use this experiment, please cite the following paper.
    - Kandasamy K, Yu Y, "Additive Approximations in High Dimensional Nonparametric
      Regression via the SALSA", International Conference on Machine Learning, 2016.
"""

# pylint: disable=invalid-name

from salsa_energy_mf import salsa_compute_negative_validation_error, MAX_TR_DATA_SIZE

def objective(x):
  """ Objective. """
  return salsa_compute_negative_validation_error([MAX_TR_DATA_SIZE], x)

