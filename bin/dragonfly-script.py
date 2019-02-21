#!/usr/bin/env python

"""
  Main APIs and command line tool for GP Bandit Optimisation.
  -- kandasamy@cs.cmu.edu
  -- kvysyara@andrew.cmu.edu

  Usage:
  python dragonfly-script.py --config <config file in .json or .pb format> --options <options file>
  See README in main repository for examples.
"""

# pylint: disable=relative-import
# pylint: disable=invalid-name
# pylint: disable=maybe-no-member
# pylint: disable=no-member

from __future__ import print_function
import os
import imp
import sys
# Local

from dragonfly import maximise_function, minimise_function, \
                      maximise_multifidelity_function, \
                      minimise_multifidelity_function, \
                      multiobjective_maximise_functions, \
                      multiobjective_minimise_functions
from dragonfly.exd.cp_domain_utils import load_config_file
from dragonfly.exd.exd_utils import get_unique_list_of_option_args
from dragonfly.opt.gp_bandit import get_all_euc_gp_bandit_args, \
                               get_all_cp_gp_bandit_args, get_all_mf_euc_gp_bandit_args, \
                               get_all_mf_cp_gp_bandit_args
from dragonfly.opt.multiobjective_gp_bandit import get_all_euc_moo_gp_bandit_args, \
                                         get_all_cp_moo_gp_bandit_args
from dragonfly.utils.option_handler import get_option_specs, load_options

dragonfly_args = [ \
  get_option_specs('config', False, None, 'Path to the json or pb config file. '),
  get_option_specs('options', False, None, 'Path to the options file. '),
  get_option_specs('max_or_min', False, 'max', 'Whether to maximise or minimise. '),
  get_option_specs('max_capital', False, -1.0,
      'Maximum capital (available budget) to be used in the experiment. '),
  get_option_specs('is_multi_objective', False, False,
                   'If True, will treat it as a multiobjective optimisation problem. '),
                 ]


def main():
  """ Main function. """
  # First load arguments
  all_args = dragonfly_args + get_all_euc_gp_bandit_args() + get_all_cp_gp_bandit_args() \
             + get_all_mf_euc_gp_bandit_args() + get_all_mf_cp_gp_bandit_args() \
             + get_all_euc_moo_gp_bandit_args() + get_all_cp_moo_gp_bandit_args()
  all_args = get_unique_list_of_option_args(all_args)
  options = load_options(all_args, cmd_line=True)

  # Load domain and objective
  config = load_config_file(options.config)
  if hasattr(config, 'fidel_space'):
    is_mf = True
  else:
    is_mf = False
  expt_dir = os.path.dirname(os.path.abspath(os.path.realpath(options.config)))
  if not os.path.exists(expt_dir):
    raise ValueError("Experiment directory does not exist.")
  objective_file_name = config.name
  obj_module = imp.load_source(objective_file_name,
                               os.path.join(expt_dir, objective_file_name + '.py'))

  # Set capital
  if options.max_capital < 0:
    raise ValueError('Specify max_capital (time or number of evaluations). for ' +
                     'optimisation')

  # Call optimiser
  _print_prefix = 'Maximising' if options.max_or_min == 'max' else 'Minimising'
  call_to_optimise = {
    'single': {'max': maximise_function, 'min': minimise_function},
    'single_mf': {'max': maximise_multifidelity_function,
                  'min': minimise_multifidelity_function},
    'multi': {'max': multiobjective_maximise_functions,
              'min': multiobjective_minimise_functions},
  }
  if not options.is_multi_objective:
    if is_mf:
      print('%s function on fidel_space: %s, domain %s.'%(_print_prefix,
            config.fidel_space, config.domain))
      opt_val, opt_pt, history = call_to_optimise['single_mf'][options.max_or_min](
        obj_module.objective, domain=None, fidel_space=None,
        fidel_to_opt=config.fidel_to_opt, fidel_cost_func=obj_module.cost,
        max_capital=options.max_capital, config=config, options=options)
    else:
      print('%s function on domain %s.'%(_print_prefix, config.domain))
      opt_val, opt_pt, history = call_to_optimise['single'][options.max_or_min](
        obj_module.objective, domain=None, max_capital=options.max_capital, config=config,
        options=options)
    print('Optimum Value in %d evals: %0.4f'%(len(history.curr_opt_points), opt_val))
    print('Optimum Point: %s.'%(opt_pt))
  else:
    if is_mf:
      raise ValueError('Multi-objective multi-fidelity optimisation has not been ' +
                       'implemented yet.')
    else:
      print('%s multiobjective functions on domain %s with %d functions.'%(_print_prefix,
            config.domain, len(obj_module.objectives)))
      pareto_values, pareto_points, history = \
        call_to_optimise['multi'][options.max_or_min](obj_module.objectives,
        domain=None, max_capital=options.max_capital, config=config, options=options)
    num_pareto_points = len(pareto_points)
    print('Found %d Pareto Points: %s.'%(num_pareto_points, pareto_points))
    print('Corresponding Pareto Values: %s.'%(pareto_values))


if __name__ == '__main__':
  sys.path.insert(0, os.getcwd())
  main()

