"""
  Adapted from pgmpy library: https://github.com/pgmpy/pgmpy
  -- kvysyara@andrew.cmu.edu
"""

# Local imports
from .check_functions import _check_1d_array_object, _check_length_equal

class BaseSimulateHamiltonianDynamics(object):
  """
  Base class for proposing new values of position and momentum
  by simulating Hamiltonian Dynamics.

  Classes inheriting this base class can be passed as an argument for
  simulate_dynamics in inference algorithms.

  Parameters
  ----------
  model : Model for which DiscretizeTime object is initialized

  position : A 1d array like object (numpy.ndarray or list)
      Vector representing values of parameter position( or X)

  momentum: A 1d array like object (numpy.ndarray or list)
      Vector representing the proposed value for momentum (velocity)

  stepsize: Float
      stepsize for the simulating dynamics
  """

  def __init__(self, model, position, momentum, stepsize):

    position = _check_1d_array_object(position, 'position')

    momentum = _check_1d_array_object(momentum, 'momentum')

    _check_length_equal(position, momentum, 'position', 'momentum')

    self.position = position
    self.momentum = momentum
    self.stepsize = stepsize
    self.model = model

    # new_position is the new proposed position, new_momentum is the new
    # proposed momentum, new_grad_lop is the value of grad log at new_position
    self.new_position = self.new_momentum = self.new_grad_logp = None

  def get_proposed_values(self):
    """
    Returns new proposed values of position and momentum
    """
    return self.new_position, self.new_momentum, self.new_grad_logp


class LeapFrog(BaseSimulateHamiltonianDynamics):
  """
  Class for simulating hamiltonian dynamics using leapfrog method

  Parameters
  ----------
  model : An instance of pgmpy.models
      Model for which DiscretizeTime object is initialized

  position : A 1d array like object (numpy.ndarray or list)
      Vector representing values of parameter position( or X)

  momentum: A 1d array like object (numpy.ndarray or list)
      Vector representing the proposed value for momentum (velocity)

  stepsize: Float
      stepsize for the simulating dynamics
  """

  def __init__(self, model, position, momentum, stepsize):

    BaseSimulateHamiltonianDynamics.__init__(self, model, position, momentum, stepsize)

    self.new_position, self.new_momentum, self.new_grad_logp = self._get_proposed_values()

  def _get_proposed_values(self):
    """
    Method to perform time splitting using leapfrog
    """
    # Take half step in time for updating momentum
    momentum_bar = self.momentum + 0.5*self.stepsize*self.model.grad_logp(self.position)

    # Take full step in time for updating position position
    position_bar = self.position + self.stepsize * momentum_bar

    grad_log = self.model.grad_logp(position_bar)

    # Take remaining half step in time for updating momentum
    momentum_bar = momentum_bar + 0.5 * self.stepsize * grad_log

    return position_bar, momentum_bar, grad_log
