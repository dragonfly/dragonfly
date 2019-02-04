"""
  NUTS Sampler -- Adapted from pgmpy library: https://github.com/pgmpy/pgmpy
  -- kvysyara@andrew.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=arguments-differ
# pylint: disable=too-many-locals
# pylint: disable=unbalanced-tuple-unpacking

from __future__ import division

import numpy as np

# Local imports
from .base import LeapFrog
from .hmc import HamiltonianMCDA
from .check_functions import _check_1d_array_object

class NoUTurnSampler(HamiltonianMCDA):
  """
  Class for performing sampling in Continuous model
  using No U Turn Sampler (a variant of Hamiltonian Monte Carlo)

  Parameters:
  -----------
  model: Model from which sampling has to be done

  simulate_dynamics: Class to propose future states of position
      and momentum in time by simulating HamiltonianDynamics

  References
  ----------
  Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
  Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
  Machine Learning Research 15 (2014) 1351-1381
  """

  def __init__(self, model, simulate_dynamics=LeapFrog):

    super(NoUTurnSampler, self).__init__(model=model, simulate_dynamics=simulate_dynamics)

  def _initalize_tree(self, position, momentum, slice_var, stepsize):
    """
    Initalizes root node of the tree, i.e depth = 0
    """

    position_bar, momentum_bar, _ = self.simulate_dynamics(self.model, position,\
                                    momentum, stepsize).get_proposed_values()
    logp_bar = self.model.logp(position_bar)

    hamiltonian = logp_bar - 0.5 * np.dot(momentum_bar, momentum_bar)

    candidate_set_size = slice_var <= np.exp(hamiltonian)
    if slice_var > 0:
      accept_set_bool = hamiltonian > (np.log(slice_var) - 1000) # delta_max = 1000
    else:
      accept_set_bool = slice_var < np.exp(hamiltonian + 1000) # delta_max = 1000

    return position_bar, momentum_bar, candidate_set_size, accept_set_bool

  def _update_acceptance_criteria(self, position_forward, position_backward,\
                                  momentum_forward, momentum_backward, accept_set_bool,\
                                  candidate_set_size, candidate_set_size2):
    """
    Checks the acceptance criteria and updates it
    """

    criteria1 = np.dot((position_forward - position_backward), momentum_backward) >= 0

    criteria2 = np.dot((position_forward - position_backward), momentum_forward) >= 0

    accept_set_bool = accept_set_bool and criteria1 and criteria2
    candidate_set_size += candidate_set_size2

    return accept_set_bool, candidate_set_size

  def _build_tree(self, position, momentum, slice_var, direction, depth, stepsize):
    """
    Recursively builds a tree for proposing new position and momentum
    """

    # Parameter names in algorithm (here -> representation in algorithm)
    # position->theta, momentum->r, slice_var->u, direction->v, depth->j,
    # stepsize->epsilon, candidate_set_size->n, accept_set_bool->s
    if depth == 0:
      # Take single leapfrog step in the given direction (direction * stepsize)
      position_bar, momentum_bar, candidate_set_size, accept_set_bool =\
        self._initalize_tree(position, momentum, slice_var, direction * stepsize)

      return (position_bar, momentum_bar, position_bar, momentum_bar, position_bar,\
              candidate_set_size, accept_set_bool)

    else:
      # Build left and right subtrees
      (position_backward, momentum_backward, position_forward, momentum_forward,\
       position_bar, candidate_set_size, accept_set_bool) = self._build_tree(position,\
                                 momentum, slice_var, direction, depth - 1, stepsize)
      if accept_set_bool == 1:
        if direction == -1:
          # Build tree in backward direction
          (position_backward, momentum_backward, _, _, position_bar2,\
           candidate_set_size2, accept_set_bool2) = self._build_tree(position_backward,\
                          momentum_backward, slice_var, direction, depth - 1, stepsize)
        else:
          # Build tree in forward direction
          (_, _, position_forward, momentum_forward, position_bar2, candidate_set_size2,\
           accept_set_bool2) = self._build_tree(position_forward, momentum_forward,\
                              slice_var, direction, depth - 1, stepsize)

        if candidate_set_size2 + candidate_set_size:
          if np.random.rand() < \
              min(1, candidate_set_size2/(candidate_set_size2 + candidate_set_size)):
            position_bar = position_bar2
        else:
          position_bar = position_bar2

        accept_set_bool, candidate_set_size =\
          self._update_acceptance_criteria(position_forward, position_backward,\
                          momentum_forward, momentum_backward, accept_set_bool2,\
                          candidate_set_size, candidate_set_size2)

      return (position_backward, momentum_backward, position_forward, momentum_forward,
              position_bar, candidate_set_size, accept_set_bool)

  def _sample(self, position, stepsize):
    """
    Returns a sample using a single iteration of NUTS
    """

    # Re-sampling momentum
    momentum = np.random.normal(0, 1, len(position))

    # Initializations
    depth = 0
    position_backward, position_forward = position, position
    momentum_backward, momentum_forward = momentum, momentum
    candidate_set_size = accept_set_bool = 1
    log_pdf = self.model.logp(position)

    # Resample slice variable `u`
    slice_var = np.random.uniform(0, np.exp(log_pdf - 0.5 * np.dot(momentum, momentum)))

    while accept_set_bool == 1:
      direction = np.random.choice([-1, 1], p=[0.5, 0.5])
      if direction == -1:
        # Build a tree in backward direction
        (position_backward, momentum_backward, _, _, position_bar, candidate_set_size2,\
         accept_set_bool2) = self._build_tree(position_backward, momentum_backward,\
                                              slice_var, direction, depth, stepsize)
      else:
        # Build tree in forward direction
        (_, _, position_forward, momentum_forward, position_bar, candidate_set_size2,\
         accept_set_bool2) = self._build_tree(position_forward, momentum_forward,\
                                              slice_var, direction, depth, stepsize)
      if accept_set_bool2 == 1:
        if np.random.rand() < min(1, candidate_set_size2/candidate_set_size):
          position = position_bar.copy()

      accept_set_bool, candidate_set_size =\
              self._update_acceptance_criteria(position_forward, position_backward,\
              momentum_forward, momentum_backward, accept_set_bool2, candidate_set_size,\
              candidate_set_size2)
      depth += 1

    return position

  def sample(self, initial_pos, num_samples, burn=150, stepsize=None):
    """
    Method to return samples using No U Turn Sampler

    Parameters
    ----------
    initial_pos: A 1d array like object
        Vector representing values of parameter position, the starting
        state in markov chain.

    num_samples: int
        Number of samples to be generated

    burn: int, defaults to 150
        Number of samples to be ignored in the beginning

    stepsize: float , defaults to None
        The stepsize for proposing new values of position and momentum in
        simulate_dynamics. If None, then will be choosen suitably
    """

    if not hasattr(initial_pos, '__len__'):
      initial_pos = np.array([initial_pos])
    initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')

    if stepsize is None:
      stepsize = self._find_reasonable_stepsize(initial_pos)

    samples = np.zeros([num_samples, len(initial_pos)], dtype=float)
    position_m = initial_pos

    for i in range(1, num_samples+burn+1):
      # Genrating sample
      position_m = self._sample(position_m, stepsize)
      # ignore first few samples
      if i >= burn:
        samples[i-burn-1] = position_m

    return samples


class NoUTurnSamplerDA(NoUTurnSampler):
  """
  Class for performing sampling in Continuous model
  using No U Turn sampler with dual averaging for
  adaptation of parameter stepsize.

  Parameters:
  -----------
  model: Model from which sampling has to be done

  simulate_dynamics: Class to propose future states of position
      and momentum in time by simulating HamiltonianDynamics

  delta: float (in between 0 and 1), defaults to 0.65
      The target HMC acceptance probability

  References
  ----------
  Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
  Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
  Machine Learning Research 15 (2014) 1351-1381
  """

  def __init__(self, model, simulate_dynamics=LeapFrog, delta=0.65):

    if not isinstance(delta, float) or delta > 1.0 or delta < 0.0:
      raise ValueError("delta should be a floating value in between 0 and 1")

    self.delta = delta
    super(NoUTurnSamplerDA, self).__init__(model=model,\
                                  simulate_dynamics=simulate_dynamics)

  def _build_tree(self, position, momentum, slice_var, direction, depth, stepsize,\
                  position0, momentum0):
    """
    Recursively builds a tree for proposing new position and momentum
    """
    if depth == 0:

      position_bar, momentum_bar, candidate_set_size, accept_set_bool =\
        self._initalize_tree(position, momentum, slice_var, direction * stepsize)

      #alpha =\
      #      min(1, self._acceptance_prob(position, position_bar, momentum, momentum_bar))
      acceptance_prob = self._acceptance_prob(position, position_bar, momentum,
                                              momentum_bar)
      alpha = 1
      if acceptance_prob < 0:
        alpha = np.exp(acceptance_prob)

      return (position_bar, momentum_bar, position_bar, momentum_bar, position_bar,\
              candidate_set_size, accept_set_bool, alpha, 1)

    else:
      (position_backward, momentum_backward, position_forward, momentum_forward,\
       position_bar, candidate_set_size, accept_set_bool, alpha, n_alpha) =\
       self._build_tree(position, momentum, slice_var, direction, depth - 1, stepsize,\
                                                                  position0, momentum0)

      if accept_set_bool == 1:
        if direction == -1:
          # Build tree in backward direction
          (position_backward, momentum_backward, _, _, position_bar2,
           candidate_set_size2, accept_set_bool2, alpha2, n_alpha2) =\
           self._build_tree(position_backward, momentum_backward, slice_var, direction,\
                                              depth - 1, stepsize, position0, momentum0)
        else:
          # Build tree in forward direction
          (_, _, position_forward, momentum_forward, position_bar2,
           candidate_set_size2, accept_set_bool2, alpha2, n_alpha2) =\
           self._build_tree(position_forward, momentum_forward, slice_var, direction,\
                                            depth - 1, stepsize, position0, momentum0)

        if (candidate_set_size2 + candidate_set_size) != 0:
          if np.random.rand() <\
                min(1, candidate_set_size2/(candidate_set_size2 + candidate_set_size)):
            position_bar = position_bar2
        else:
          position_bar = position_bar2

        alpha += alpha2
        n_alpha += n_alpha2
        accept_set_bool, candidate_set_size =\
        self._update_acceptance_criteria(position_forward, position_backward,\
        momentum_forward, momentum_backward, accept_set_bool2, candidate_set_size,\
                           candidate_set_size2)

      return (position_backward, momentum_backward, position_forward, momentum_forward,\
              position_bar, candidate_set_size, accept_set_bool, alpha, n_alpha)

  def _sample(self, position, stepsize):
    """
    Returns a sample using a single iteration of NUTS with dual averaging
    """

    # Re-sampling momentum
    momentum = np.random.normal(0, 1, len(position))

    # Initializations
    depth = 0
    position_backward, position_forward = position, position
    momentum_backward, momentum_forward = momentum, momentum
    candidate_set_size = accept_set_bool = 1
    position_m_1 = position
    log_pdf = self.model.logp(position)

    # Resample slice variable `u`
    slice_var = np.random.uniform(0, np.exp(log_pdf - 0.5 * np.dot(momentum, momentum)))

    while accept_set_bool == 1:
      direction = np.random.choice([-1, 1], p=[0.5, 0.5])
      if direction == -1:
        # Build a tree in backward direction
        (position_backward, momentum_backward, _, _, position_bar, candidate_set_size2,\
         accept_set_bool2, alpha, n_alpha) =\
         self._build_tree(position_backward, momentum_backward, slice_var, direction,\
                          depth, stepsize, position_m_1, momentum)
      else:
        # Build tree in forward direction
        (_, _, position_forward, momentum_forward, position_bar, candidate_set_size2,\
         accept_set_bool2, alpha, n_alpha) =\
         self._build_tree(position_forward, momentum_forward, slice_var, direction,\
                          depth, stepsize, position_m_1, momentum)

      # Sample position at random from C
      if accept_set_bool2 == 1:
        if np.random.rand() < min(1, candidate_set_size2/candidate_set_size):
          position = position_bar

      accept_set_bool, candidate_set_size =\
      self._update_acceptance_criteria(position_forward, position_backward,\
      momentum_forward, momentum_backward, accept_set_bool2, candidate_set_size,\
                                           candidate_set_size2)

      depth += 1

    return position, alpha, n_alpha

  def sample(self, initial_pos, num_samples, burn=150, num_adapt=10, stepsize=None):
    """
    Returns samples using No U Turn Sampler with dual averaging

    Parameters
    ----------
    initial_pos: A 1d array like object
        Vector representing values of parameter position, the starting
        state in markov chain.

    num_adapt: int
        The number of interations to run the adaptation of stepsize

    num_samples: int
        Number of samples to be generated

    burn: int, defaults to 150
        Number of samples to be ignored in the beginning

    stepsize: float, defaults to None
        The stepsize for proposing new values of position and momentum in
        simulate_dynamics. If None, then will be choosen suitably
    """

    if not hasattr(initial_pos, '__len__'):
      initial_pos = np.array([initial_pos])
    initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')

    if stepsize is None:
      stepsize = self._find_reasonable_stepsize(initial_pos)

    if num_adapt <= 1:
      return NoUTurnSampler(self.model,\
                  self.simulate_dynamics).sample(initial_pos, num_samples, stepsize)

    mu = np.log(10.0 * stepsize)
    stepsize_bar = 1.0
    h_bar = 0.0

    samples = np.zeros([num_samples, len(initial_pos)], dtype=float)
    position_m = initial_pos

    for i in range(1, num_samples+burn+1):

      position_m, alpha, n_alpha = self._sample(position_m, stepsize)
      #ignore first few samples
      if i >= burn:
        samples[i-burn-1] = position_m

      if i <= num_adapt:
        stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar,\
                                                           mu, i, alpha, n_alpha)
      else:
        stepsize = stepsize_bar

    return samples
