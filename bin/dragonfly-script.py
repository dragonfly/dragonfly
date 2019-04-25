#!/usr/bin/env python

"""
  Command line tool for Dragonfly.
  -- kandasamy@cs.cmu.edu
  -- kvysyara@andrew.cmu.edu

  Usage:
  python dragonfly-script.py --config <config file in .json or .pb format> --options <options file>
  See README in main repository for examples.
"""

# pylint: disable=relative-import
# pylint: disable=invalid-name
# pylint: disable=import-error

from __future__ import print_function
import os
# import imp
from importlib import import_module
import sys
# Local

from dragonfly import maximise_function, minimise_function, \
                      maximise_multifidelity_function, \
                      minimise_multifidelity_function, \
                      multiobjective_maximise_functions, \
                      multiobjective_minimise_functions
from dragonfly.exd.cp_domain_utils import load_config_file
from dragonfly.exd.exd_utils import get_unique_list_of_option_args
from dragonfly.utils.option_handler import get_option_specs, load_options
# Get options
from dragonfly.opt.ga_optimiser import ga_opt_args
from dragonfly.opt.gp_bandit import get_all_euc_gp_bandit_args, \
                            get_all_cp_gp_bandit_args, get_all_mf_euc_gp_bandit_args, \
                            get_all_mf_cp_gp_bandit_args
from dragonfly.opt.random_optimiser import euclidean_random_optimiser_args, \
                                   mf_euclidean_random_optimiser_args, \
                                   cp_random_optimiser_args, mf_cp_random_optimiser_args
from dragonfly.opt.multiobjective_gp_bandit import get_all_euc_moo_gp_bandit_args, \
                                                   get_all_cp_moo_gp_bandit_args
from dragonfly.opt.random_multiobjective_optimiser import \
              euclidean_random_multiobjective_optimiser_args, \
              cp_random_multiobjective_optimiser_args


dragonfly_args = [ \
  get_option_specs('config', False, None, 'Path to the json or pb config file. '),
  get_option_specs('options', False, None, 'Path to the options file. '),
  get_option_specs('max_or_min', False, 'max', 'Whether to maximise or minimise. '),
  get_option_specs('max_capital', False, -1.0,
    'Maximum capital (available budget) to be used in the experiment. '),
  get_option_specs('capital_type', False, 'return_value',
    'Maximum capital (available budget) to be used in the experiment. '),
  get_option_specs('is_multi_objective', False, 0,
    'If True, will treat it as a multiobjective optimisation problem. '),
  get_option_specs('opt_method', False, 'bo',
    ('Optimisation method. Default is bo. This should be one of bo, ga, ea, direct, ' +
     ' pdoo, or rand, but not all methods apply to all problems.')),
  get_option_specs('report_progress', False, 'default',
    ('How to report progress. Should be one of "default" (prints to stdout), ' +
     '"silent" (no reporting), or a filename (writes to file).')),
                 ]


def get_command_line_args():
  """ Returns all arguments for the command line. """
  ret = dragonfly_args + \
        ga_opt_args + \
        euclidean_random_optimiser_args + cp_random_optimiser_args + \
        mf_euclidean_random_optimiser_args + mf_cp_random_optimiser_args + \
        get_all_euc_gp_bandit_args() + get_all_cp_gp_bandit_args() + \
        get_all_mf_euc_gp_bandit_args() + get_all_mf_cp_gp_bandit_args() + \
        euclidean_random_multiobjective_optimiser_args + \
        cp_random_multiobjective_optimiser_args + \
        get_all_euc_moo_gp_bandit_args() + get_all_cp_moo_gp_bandit_args()
  return get_unique_list_of_option_args(ret)


def main():
  """ Main function. """
  options = load_options(get_command_line_args(), cmd_line=True)
  # Load domain and objective
  config = load_config_file(options.config)
  if hasattr(config, 'fidel_space'):
    is_mf = True
  else:
    is_mf = False

  # Load module
  expt_dir = os.path.dirname(os.path.abspath(os.path.realpath(options.config)))
  if not os.path.exists(expt_dir):
    raise ValueError("Experiment directory does not exist.")
  sys.path.append(expt_dir)
  obj_module = import_module(config.name, expt_dir)
  sys.path.remove(expt_dir)

  # Set capital
  if options.max_capital < 0:
    raise ValueError('max_capital (time or number of evaluations) must be positive.')

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
      print('%s multi-fidelity function on\n Fidelity-Space: %s.\n Domain: %s.\n'%(
            _print_prefix, config.fidel_space, config.domain))
      opt_val, opt_pt, history = call_to_optimise['single_mf'][options.max_or_min](
        obj_module.objective, fidel_space=None, domain=None,
        fidel_to_opt=config.fidel_to_opt, fidel_cost_func=obj_module.cost,
        max_capital=options.max_capital, capital_type=options.capital_type,
        opt_method=options.opt_method, config=config, options=options,
        reporter=options.report_progress)
    else:
      print('%s function on Domain: %s.\n'%(_print_prefix, config.domain))
      opt_val, opt_pt, history = call_to_optimise['single'][options.max_or_min](
        obj_module.objective, domain=None, max_capital=options.max_capital,
        capital_type=options.capital_type, opt_method=options.opt_method,
        config=config, options=options, reporter=options.report_progress)
    print('Optimum Value in %d evals: %0.4f'%(len(history.curr_opt_points), opt_val))
    print('Optimum Point: %s.'%(opt_pt))
  else:
    if is_mf:
      raise ValueError('Multi-objective multi-fidelity optimisation has not been ' +
                       'implemented yet.')
    else:
      # Check format of function caller
      if hasattr(obj_module, 'objectives'):
        objectives_to_pass = obj_module.objectives
        num_objectives = len(objectives_to_pass)
      else:
        num_objectives = obj_module.num_objectives
        objectives_to_pass = (obj_module.compute_objectives, obj_module.num_objectives)
      print('%s %d multiobjective functions on Domain: %s.\n'%(_print_prefix,
            num_objectives, config.domain))
      print(objectives_to_pass)
      pareto_values, pareto_points, history = \
        call_to_optimise['multi'][options.max_or_min](objectives_to_pass,
        domain=None, max_capital=options.max_capital, capital_type=options.capital_type,
        opt_method=options.opt_method, config=config, options=options,
        reporter=options.report_progress)
    num_pareto_points = len(pareto_points)
    print('Found %d Pareto Points: %s.'%(num_pareto_points, pareto_points))
    print('Corresponding Pareto Values: %s.'%(pareto_values))


if __name__ == '__main__':
  sys.path.insert(0, os.getcwd())
  main()

