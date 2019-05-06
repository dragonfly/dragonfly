"""
  A demo on a multi-objective electrolyte optimisation task using various features of
  Dragonfly.
  -- kirthevasank
"""

from __future__ import print_function
from argparse import Namespace
from dragonfly import multiobjective_maximise_functions, load_config_file
# Local imports
import moo_3d
import moo_5d
from prior_means import conductivity_prior_mean_3d, conductivity_prior_mean_5d, \
                        conductivity_prior_mean_3d_mf


# choose problem
PROBLEM = '3d'
# PROBLEM = '3d_euc'
# PROBLEM = '5d'

# chooser dict
_CHOOSER_DICT = {
  '3d': (moo_3d.compute_objectives, moo_3d.num_objectives, 'config_3d.json'),
  '3d_euc': (moo_3d.compute_objectives, moo_3d.num_objectives, 'config_3d_cts.json'),
  '5d': (moo_5d.compute_objectives, moo_5d.num_objectives, 'config_5d.json'),
  }

USE_CONDUCTIVITY_PRIOR_MEAN = True
# USE_CONDUCTIVITY_PRIOR_MEAN = False


def main():
  """ Main function. """
  compute_objectives, num_objectives, config_file = _CHOOSER_DICT[PROBLEM]
  config = load_config_file(config_file)
  moo_objectives = (compute_objectives, num_objectives)

  # Specify options
  options = Namespace(
    build_new_model_every=5, # update the model every 5 iterations
    report_results_every=4, # report progress every 6 iterations
    report_model_on_each_build=True, # report the model when you build it.
    )

  # Dragonfly allows specifying a mean for the GP prior - if there is prior knowledge
  # on the rough behaviour of the function to be optimised, this is one way that
  # information can be incorporated into the model.
  if USE_CONDUCTIVITY_PRIOR_MEAN:
    if PROBLEM in ['3d', '3d_euc']:
      options.gps_prior_means = (conductivity_prior_mean_3d, None)
    elif PROBLEM == '5d':
      options.gps_prior_means = (conductivity_prior_mean_5d, None)
    # The _unproc indicates that the mean function is "unprocessed". Dragonfly converts
    # the domain specified given in the configuration to an internal order which may
    # have reordered the variables. The _unproc tells that the function
    # should be called in the original format.

  # Optimise
  max_num_evals = 60
  pareto_opt_pts, pareto_opt_vals, history = multiobjective_maximise_functions(
    moo_objectives, config.domain, max_num_evals, config=config, options=options)
  print(pareto_opt_pts)
  print(pareto_opt_vals)


if __name__ == '__main__':
  main()

