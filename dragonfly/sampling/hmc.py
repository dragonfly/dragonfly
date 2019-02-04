"""
  Adapted from pgmpy library: https://github.com/pgmpy/pgmpy
  A collection of methods for sampling from continuous models
  -- kvysyara@andrew.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=arguments-differ

from __future__ import division
from math import sqrt

import numpy as np

from .check_functions import _check_1d_array_object
from .base import LeapFrog, BaseSimulateHamiltonianDynamics

class HamiltonianMC(object):
  """
  Class for performing sampling using simple
  Hamiltonian Monte Carlo

  Parameters:
  -----------
  model: Model from which sampling has to be done

  simulate_dynamics: A class to propose future values of
    momentum and position in time by simulating
    Hamiltonian Dynamics

  References
  ----------
  R.Neal. Handbook of Markov Chain Monte Carlo,
  chapter 5: MCMC Using Hamiltonian Dynamics.
  CRC Press, 2011.
  """

  def __init__(self, model, simulate_dynamics=LeapFrog):

    if not issubclass(simulate_dynamics, BaseSimulateHamiltonianDynamics):
      raise TypeError("split_time must be an instance of " +
                      "BaseSimulateHamiltonianDynamics")

    self.model = model
    self.simulate_dynamics = simulate_dynamics
    self.accepted_proposals = 0.0
    self.acceptance_rate = 0

  def _acceptance_prob(self, position, position_bar, momentum, momentum_bar):
    """
    Returns the acceptance probability for given new position(position) and momentum
    """

    # Parameters to help in evaluating Joint distribution P(position, momentum)
    _logp = self.model.logp(position)
    _logp_bar = self.model.logp(position_bar)

    # acceptance_prob = P(position_bar, momentum_bar)/ P(position, momentum)
    if not np.isfinite(_logp_bar):
      potential_change = _logp_bar
    else:
      potential_change = _logp_bar - _logp  # Negative change
    kinetic_change = 0.5 * np.float(np.dot(momentum_bar.T, momentum_bar) -\
                                    np.dot(momentum.T, momentum))

    # acceptance probability
    #return np.exp(potential_change - kinetic_change)
    return potential_change - kinetic_change

  def _get_condition(self, acceptance_prob, a):
    """
    Temporary method to fix issue in numpy 0.12 #852
    """
    #return (1/(acceptance_prob ** a)) > np.power(2.0, -a)
    #return (acceptance_prob ** a) > np.power(2.0, -a)
    return (acceptance_prob * a) > np.log(np.power(2.0, -a))

  def _find_reasonable_stepsize(self, position, stepsize_app=1):
    """
    Method for choosing initial value of stepsize

    References
    -----------
    Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
    Machine Learning Research 15 (2014) 1351-1381
    """
    # momentum = N(0, I)
    momentum = np.reshape(np.random.normal(0, 1, len(position)), position.shape)

    # Take a single step in time
    position_bar, momentum_bar, _ = self.simulate_dynamics(self.model, position,\
                                    momentum, stepsize_app).get_proposed_values()

    acceptance_prob = self._acceptance_prob(position, position_bar, momentum,\
                                            momentum_bar)

    # a = 2I[acceptance_prob] -1
    #a = 2*(acceptance_prob > 0.5) - 1
    a = 2*(acceptance_prob > np.log(0.5)) - 1

    condition = self._get_condition(acceptance_prob, a)

    while condition:
      stepsize_app = np.power(2.0, a)*stepsize_app

      position_bar, momentum_bar, _ = self.simulate_dynamics(self.model, position,\
                                      momentum, stepsize_app).get_proposed_values()

      acceptance_prob = self._acceptance_prob(position, position_bar, momentum,\
                                              momentum_bar)

      condition = self._get_condition(acceptance_prob, a)

    return stepsize_app

  def _sample(self, position, trajectory_length, stepsize, lsteps=None):
    """
    Runs a single sampling iteration to return a sample
    """
    # Resampling momentum
    momentum = np.reshape(np.random.normal(0, 1, len(position)), position.shape)

    # position_m here will be the previous sampled value of position
    position_bar, momentum_bar = position.copy(), momentum

    # Number of steps L to simulate dynamics
    if lsteps is None:
      lsteps = int(max(1, round(trajectory_length / stepsize, 0)))

    grad_bar = self.model.grad_logp(position_bar)

    for _ in range(lsteps):
      position_bar, momentum_bar, grad_bar =\
        self.simulate_dynamics(self.model, position_bar, momentum_bar,\
                               stepsize, grad_bar).get_proposed_values()

    acceptance_prob = self._acceptance_prob(position, position_bar, momentum,
                                            momentum_bar)

    # Metropolis acceptance probability
    alpha = 1
    if acceptance_prob < 0:
      alpha = np.exp(acceptance_prob)

    # Accept or reject the new proposed value of position, i.e position_bar
    if np.random.rand() < alpha:
      position = position_bar.copy()
      self.accepted_proposals += 1.0

    return position, alpha

  def sample(self, initial_pos, num_samples, trajectory_length, stepsize=None, burn=1000):
    """
    Method to return samples using Hamiltonian Monte Carlo

    Parameters
    ----------
    initial_pos: A 1d array like object
      Vector representing values of parameter position, the starting
      state in markov chain.

    num_samples: int
      Number of samples to be generated

    trajectory_length: int or float
      Target trajectory length, stepsize * number of steps(L),
      where L is the number of steps taken per HMC iteration,
      and stepsize is step size for splitting time method.

    stepsize: float, defaults to None
      The stepsize for proposing new values of position and momentum in simulate_dynamics
      If None, then will be choosen suitably

    burn: int, defaults to 1000
      Number of samples to ignore in the beginning
    """


    self.accepted_proposals = 1.0
    initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')

    if stepsize is None:
      stepsize = self._find_reasonable_stepsize(initial_pos)

    #samples = np.zeros(num_samples, dtype=types).view(np.recarray)
    samples = np.zeros(num_samples, dtype=float)

    # Assigning after converting into tuple because value was being
    # changed after assignment. Reason for this is unknown
    position_m = initial_pos

    lsteps = int(max(1, round(trajectory_length / stepsize, 0)))
    for i in range(1, num_samples+burn+1):

      # Genrating sample
      position_m, _ = self._sample(position_m, trajectory_length, stepsize, lsteps)
      # ignore first few samples
      if i >= burn:
        samples[i-burn-1] = position_m

    self.acceptance_rate = self.accepted_proposals / num_samples

    return samples

class HamiltonianMCDA(HamiltonianMC):
  """
  Class for performing sampling in Continuous model
  using Hamiltonian Monte Carlo with dual averaging for
  adaptaion of parameter stepsize.

  Parameters:
  -----------
  model: Model from which sampling has to be done

  simulate_dynamics: Class to propose future states of position
    and momentum in time by simulating HamiltonianDynamics

  delta: float (in between 0 and 1), defaults to 0.65
    The target HMC acceptance probability

  References
  -----------
  Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
  Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
  Machine Learning Research 15 (2014) 1351-1381
  """

  def __init__(self, model, simulate_dynamics=LeapFrog, delta=0.65):

    if not isinstance(delta, float) or delta > 1.0 or delta < 0.0:
      raise ValueError("delta should be a floating value in between 0 and 1")

    self.delta = delta

    super(HamiltonianMCDA, self).__init__(model=model,\
                                          simulate_dynamics=simulate_dynamics)

  def _adapt_params(self, stepsize, stepsize_bar, h_bar, mu, index_i, alpha, n_alpha=1):
    """
    Run tha adaptation for stepsize for better proposals of position
    """
    gamma = 0.05  # free parameter that controls the amount of shrinkage towards mu
    t0 = 10.0  # free parameter that stabilizes the initial iterations of the algorithm
    kappa = 0.75
    # See equation (6) section 3.2.1 for details

    estimate = 1.0 / (index_i + t0)
    h_bar = (1 - estimate) * h_bar + estimate * (self.delta - (alpha/n_alpha))

    stepsize = np.exp(mu - ((sqrt(index_i)/gamma)*h_bar))
    i_kappa = index_i ** (-kappa)
    stepsize_bar = np.exp(i_kappa*np.log(stepsize) + (1 - i_kappa)*np.log(stepsize_bar))

    return stepsize, stepsize_bar, h_bar

  def sample(self, initial_pos, num_adapt, num_samples, trajectory_length, stepsize=None,
             burn=1000):
    """
    Method to return samples using Hamiltonian Monte Carlo

    Parameters
    ----------
    initial_pos: A 1d array like object
      Vector representing values of parameter position, the starting
      state in markov chain.

    num_adapt: int
      The number of interations to run the adaptation of stepsize

    num_samples: int
      Number of samples to be generated

    trajectory_length: int or float
      Target trajectory length, stepsize * number of steps(L),
      where L is the number of steps taken per HMC iteration,
      and stepsize is step size for splitting time method.

    stepsize: float , defaults to None
      The stepsize for proposing new values of position and momentum in simulate_dynamics
      If None, then will be choosen suitably

    burn: int, defaults to 1000
      Number of samples to ignore in the beginning
    """

    self.accepted_proposals = 1.0

    initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')

    if stepsize is None:
      stepsize = self._find_reasonable_stepsize(initial_pos)

    if num_adapt <= 1:  # Return samples genrated using Simple HMC algorithm
      return HamiltonianMC.sample(self, initial_pos, num_samples, trajectory_length,
                                  stepsize)

    # stepsize is epsilon
    # freely chosen point, after each iteration xt(/position) is shrunk towards it
    mu = np.log(10.0 * stepsize)
    # log(10 * stepsize) large values to save computation
    # stepsize_bar is epsilon_bar
    stepsize_bar = 1.0
    h_bar = 0.0
    # See equation (6) section 3.2.1 for details

    #samples = np.zeros(num_samples, dtype=types).view(np.recarray)
    samples = np.zeros(num_samples, dtype=float)
    position_m = initial_pos

    for i in range(1, num_samples+burn+1):

      # Genrating sample
      position_m, alpha = self._sample(position_m, trajectory_length, stepsize)
      # ignore first few samples
      if i >= burn:
        samples[i-burn-1] = position_m

      # Adaptation of stepsize till num_adapt iterations
      if i <= num_adapt:
        stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar,
                                                           mu, i, alpha)
      else:
        stepsize = stepsize_bar

    self.acceptance_rate = self.accepted_proposals / num_samples

    return samples
