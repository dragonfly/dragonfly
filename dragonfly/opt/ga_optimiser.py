"""
  Implements Genetic algorithms for black-box optimisation.
  --kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

from argparse import Namespace
from numpy.random import choice
# Local imports
from .blackbox_optimiser import BlackboxOptimiser, blackbox_opt_args
from ..utils.general_utils import sample_according_to_exp_probs
from ..utils.option_handler import get_option_specs, load_options

ga_specific_opt_args = [
  get_option_specs('num_mutations_per_epoch', False, 5,
    'Number of mutations per epoch.'),
  get_option_specs('num_candidates_to_mutate_from', False, -1,
    'The number of candidates to choose the mutations from.'),
  get_option_specs('fitness_sampler_scaling_const', False, 2,
    'The scaling constant for sampling according to exp_probs.'),
  ]

ga_opt_args = ga_specific_opt_args + blackbox_opt_args


class GAOptimiser(BlackboxOptimiser):
  """ Class for optimisation based on Genetic algorithms. """

  def __init__(self, func_caller, worker_manager=None, mutation_op=None, crossover_op=None,
               options=None, reporter=None, ask_tell_mode=False):
    """ Constructor.
      mutation_op: A function which takes in a list of objects and modifies them.
      crossover_op: A function which takes in two objects and performs a cross-over
                    operation.
      So far we have not implemented cross-over but included here in case we want to
      include it in the future.
      For other arguments, see BlackboxOptimiser
    """
    # TODO: implement cross-over operation
    options = load_options(ga_opt_args, partial_options=options)
    super(GAOptimiser, self).__init__(func_caller, worker_manager, model=None,
                                      options=options, reporter=reporter,
                                      ask_tell_mode=ask_tell_mode)
    self.mutation_op = mutation_op
    self.crossover_op = crossover_op
    self.to_eval_points = []

  def _opt_method_set_up(self):
    """ Additional set up. """
    # pylint: disable=attribute-defined-outside-init
    # Set up parameters for the mutations
    self.method_name = 'GA'
    self.num_mutations_per_epoch = self.options.num_mutations_per_epoch
    self.num_candidates_to_mutate_from = self.options.num_candidates_to_mutate_from

  def _opt_method_optimise_initialise(self):
    """ No initialisation for GA. """
    self.generate_new_eval_points()

  def _add_data_to_model(self, qinfos):
    """ Update the optimisation model. """
    pass

  def _child_build_new_model(self):
    """ Build new optimisation model. """
    pass

  def _get_candidates_to_mutate_from(self, num_mutations, num_candidates_to_mutate_from):
    """ Returns the candidates to mutate from. """
    all_prev_eval_points = self.prev_eval_points + self.history.query_points
    all_prev_eval_vals = self.prev_eval_vals + self.history.query_vals
    if num_candidates_to_mutate_from <= 0:
      idxs_to_mutate_from = sample_according_to_exp_probs(all_prev_eval_vals,
                              num_mutations, replace=True,
                              scaling_const=self.options.fitness_sampler_scaling_const,
                              sample_uniformly_if_fail=True)
      num_mutations_arg_to_mutation_op = [(idxs_to_mutate_from == i).sum() for i
                                          in range(len(all_prev_eval_points))]
      candidates_to_mutate_from = all_prev_eval_points
    else:
      cand_idxs_to_mutate_from = sample_according_to_exp_probs(all_prev_eval_vals,
        num_candidates_to_mutate_from, replace=False,
        scaling_const=self.options.fitness_sampler_scaling_const)
      candidates_to_mutate_from = [all_prev_eval_points[i] for i in
                                   cand_idxs_to_mutate_from]
      num_mutations_arg_to_mutation_op = num_mutations
    return candidates_to_mutate_from, num_mutations_arg_to_mutation_op

  def generate_new_eval_points(self, num_mutations=None,
                               num_candidates_to_mutate_from=None):
    """ Generates the mutations. """
    new_candidates = []
    num_tries = 0
    num_mutations_to_try = self.num_mutations_per_epoch if num_mutations is None \
                           else num_mutations
    while len(new_candidates) == 0:
      num_tries += 1
      generated_from_mutation_op = self.generate_new_eval_points_from_mutation_op(
                                     num_mutations_to_try, num_candidates_to_mutate_from)
      points_in_domain = [elem for elem in generated_from_mutation_op if
                          self.domain.is_a_member(elem)]
      new_candidates.extend(points_in_domain)
      if len(points_in_domain) == 0:
        if num_tries % 10 == 0:
          error_msg = ('Could not generate any points in domain from given mutation ' +
                       'operator despite %d tries with up to %d candidates.')%(num_tries,
                       num_mutations_to_try)
          self.reporter.writeln(error_msg)
        if num_tries >= 51:
          error_msg = ('Could not generate any points in domain from given mutation ' +
            'operator despite %d tries with up to %d candidates. Quitting now.')%(
            num_tries, num_mutations_to_try)
          raise ValueError(error_msg)
        # Try a larger number of mutations the next time
        num_mutations_to_try = int(num_mutations_to_try * 1.2 + 1)
    new_candidates = new_candidates[:num_mutations]
    self.to_eval_points.extend(new_candidates)

  def generate_new_eval_points_from_mutation_op(self, num_mutations=None,
                                                num_candidates_to_mutate_from=None):
    """ Generates the mutations. """
    num_mutations = self.num_mutations_per_epoch if num_mutations is None else \
                      num_mutations
    num_candidates_to_mutate_from = self.num_candidates_to_mutate_from if \
      num_candidates_to_mutate_from is None else num_candidates_to_mutate_from
    candidates_to_mutate_from, num_mutations_arg_to_mutation_op = \
      self._get_candidates_to_mutate_from(num_mutations, num_candidates_to_mutate_from)
    new_eval_points = self.mutation_op(candidates_to_mutate_from,
                                       num_mutations_arg_to_mutation_op)
    return new_eval_points

  def _determine_next_query(self):
    """ Determine the next point for evaluation. """
    if len(self.to_eval_points) == 0:
      self.generate_new_eval_points()
    ret = self.to_eval_points.pop(0)
    return Namespace(point=ret)

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of eval points. Not implementing for now. """
    qinfos = [self._determine_next_query() for _ in range(batch_size)]
    return qinfos

  def _get_method_str(self):
    """ Returns a string describing the method. """
    return 'ga'

  def is_an_mf_method(self):
    """ Returns False. """
    return False


# A GA optimiser with random fitness values ----------------------------------------------
class GARandOptimiser(GAOptimiser):
  """ Same as the GA optimiser, but the candidates to mutate from are picked randomly.
      This is used in the RAND baseline.
  """
  # pylint: disable=abstract-method

  def _child_set_up(self):
    """ Additional set up. """
    super(GARandOptimiser, self)._child_set_up()
    self.method_name = 'randGA'

  def _get_candidates_to_mutate_from(self, num_mutations, num_candidates_to_mutate_from):
    """ Returns a random list of points from the evaluations to mutate from. """
    all_prev_eval_points = self.prev_eval_points + self.history.query_points
    candidates_to_mutate_from = choice(all_prev_eval_points,
                                       self.num_candidates_to_mutate_from,
                                       replace=False)
    return candidates_to_mutate_from, num_mutations


# APIs
# ======================================================================================
def ga_optimise_from_args(func_caller, worker_manager, max_capital, mode, mutation_op,
                          is_rand=False, crossover_op=None, options=None,
                          reporter='default'):
  """ GA optimisation from args. """
  options = load_options(ga_opt_args, partial_options=options)
  options.mode = mode
  optimiser_class = GARandOptimiser if is_rand else GAOptimiser
  return (optimiser_class(func_caller, worker_manager, mutation_op, crossover_op,
                          options, reporter)).optimise(max_capital)

