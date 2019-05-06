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


# choose problem
PROBLEM = '3d'
# PROBLEM = '5d'

# chooser dict
_CHOOSER_DICT = {
  '3d': (moo_3d.compute_objectives, moo_3d.num_objectives, 'config_3d.json'),
  '5d': (moo_5d.compute_objectives, moo_5d.num_objectives, 'config_5d.json'),
  }


def main():
  """ Main function. """
  compute_objectives, num_objectives, config_file = _CHOOSER_DICT[PROBLEM]
  config = load_config_file(config_file)
  moo_objectives = (compute_objectives, num_objectives)

  # Specify options
  options = Namespace(
    build_new_model_every=5, # update the model every 5 iterations
    report_results_every=6, # report progress every 6 iterations
    )

  # Optimise
  max_num_evals = 60
  pareto_opt_pts, pareto_opt_vals, history = multiobjective_maximise_functions(
    moo_objectives, config.domain, max_num_evals, config=config, options=options)
  print(pareto_opt_pts)
  print(pareto_opt_vals)


if __name__ == '__main__':
  main()

