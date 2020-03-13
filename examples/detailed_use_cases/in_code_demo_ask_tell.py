"""
  A demo on the ask-tell interface of Dragonfly.
  -- anthonyhsyu
"""

from __future__ import print_function
from argparse import Namespace
from dragonfly import load_config_file, maximise_function, maximise_multifidelity_function
from dragonfly.apis.api_utils import preprocess_multifidelity_arguments
from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from dragonfly.opt import random_optimiser, cp_ga_optimiser, gp_bandit
# Local imports
import obj_3d
import obj_3d_mf
import obj_5d

# choose problem
# PROBLEM = '3d'      # Optimisation problem with 3 variables
# PROBLEM = '3d_mf'   # Optimisation problem with 3 variables and 1 fidelity variable
PROBLEM = '3d_euc'  # Optimisation problem with 3 variables all of which are continuous
# PROBLEM = '5d'      # Optimisation problem with 5 variables

# chooser dict
_CHOOSER_DICT = {
  '3d': (obj_3d.objective, 'config_3d.json', None),
  '3d_euc': (obj_3d.objective, 'config_3d_cts.json', None),
  '3d_mf': (obj_3d_mf.objective, 'config_3d_mf.json', obj_3d_mf.cost),
  '5d': (obj_5d.objective, 'config_5d.json', None),
  }

def main():
  """ Main function. """
  # Load configuration file
  objective, config_file, mf_cost = _CHOOSER_DICT[PROBLEM]
  config = load_config_file(config_file)

  # Specify optimisation method -----------------------------------------------------
  opt_method = 'bo'
  # opt_method = 'ga'
  # opt_method = 'rand'

  # Optimise
  max_capital = 60
  domain, domain_orderings = config.domain, config.domain_orderings
  if PROBLEM in ['3d', '5d']:
    # Create function caller.
    # Note there is no function passed in to the Function Caller object.
    func_caller = CPFunctionCaller(None, domain, domain_orderings=domain_orderings)

    if opt_method == 'bo':
      opt = gp_bandit.CPGPBandit(func_caller, ask_tell_mode=True)
    elif opt_method == 'ga':
      opt = cp_ga_optimiser.CPGAOptimiser(func_caller, ask_tell_mode=True)
    elif opt_method == 'rand':
      opt = random_optimiser.CPRandomOptimiser(func_caller, ask_tell_mode=True)
    opt.initialise()

    # Optimize using the ask-tell interface
    # User continually asks for the next point to evaluate, then tells the optimizer the
    # new result to perform Bayesian optimisation.
    best_x, best_y = None, float('-inf')
    for _ in range(max_capital):
      x = opt.ask()
      y = objective(x)
      opt.tell([(x, y)])
      print('x: %s, y: %s'%(x, y))
      if y > best_y:
        best_x, best_y = x, y
    print("Optimal Value: %s, Optimal Point: %s"%(best_y, best_x))

    # Compare results with the maximise_function API
    print("-------------")
    print("Compare with maximise_function API:")
    opt_val, opt_pt, history = maximise_function(objective, config.domain, max_capital,
                                 opt_method=opt_method, config=config)
  
  elif PROBLEM == '3d_euc':
    # Create function caller.
    # Note there is no function passed in to the Function Caller object.
    domain = domain.list_of_domains[0]
    func_caller = EuclideanFunctionCaller(None, domain)

    if opt_method == 'bo':
      opt = gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)
    elif opt_method == 'ga':
      raise ValueError("Invalid opt_method %s"%(opt_method))   
    opt.initialise()

    # Optimize using the ask-tell interface
    # User continually asks for the next point to evaluate, then tells the optimizer the
    # new result to perform Bayesian optimisation.
    best_x, best_y = None, float('-inf')
    for _ in range(max_capital):
      # Optionally, you can add an integer argument `n_points` to ask to have it return
      # `n_points` number of points. These points will be returned as a list.
      # No argument for `n_points` returns a single point from ask.
      x = opt.ask()
      y = objective(x)
      opt.tell([(x, y)])
      print('x: %s, y: %s'%(x, y))
      if y > best_y:
        best_x, best_y = x, y
    print("Optimal Value: %s, Optimal Point: %s"%(best_y, best_x))

    # Compare results with the maximise_function API
    print("-------------")
    print("Compare with maximise_function API:")
    opt_val, opt_pt, history = maximise_function(objective, config.domain, max_capital,
                                 opt_method=opt_method, config=config)

  else:
    # Create function caller.
    # Note there is no function passed in to the Function Caller object.
    (ask_tell_fidel_space, ask_tell_domain, _, ask_tell_mf_cost, ask_tell_fidel_to_opt, ask_tell_config, _) = \
      preprocess_multifidelity_arguments(config.fidel_space, domain, [objective],
                                         mf_cost, config.fidel_to_opt, config)
    func_caller = CPFunctionCaller(None, ask_tell_domain, domain_orderings=domain_orderings,
                                   fidel_space=ask_tell_fidel_space, fidel_cost_func=ask_tell_mf_cost,
                                   fidel_to_opt=ask_tell_fidel_to_opt,
                                   fidel_space_orderings=config.fidel_space_orderings,
                                   config=ask_tell_config)
    if opt_method == 'bo':
      opt = gp_bandit.CPGPBandit(func_caller, is_mf=True, ask_tell_mode=True)
    else:
      raise ValueError("Invalid opt_method %s"%(opt_method))   
    opt.initialise()

    # Optimize using the ask-tell interface
    # User continually asks for the next point to evaluate, then tells the optimizer the
    # new result to perform Bayesian optimisation.
    best_z, best_x, best_y = None, None, float('-inf')
    for _ in range(max_capital):
      point = opt.ask()
      z, x = point[0], point[1]
      y = objective(z, x)
      opt.tell([(z, x, y)])
      print('z: %s, x: %s, y: %s'%(z, x, y))
      if y > best_y:
        best_z, best_x, best_y = z, x, y
    print("Optimal Value: %s, Optimal Point: %s"%(best_y, best_x))

    # Compare results with the maximise_multifidelity_function API
    print("-------------")
    print("Compare with maximise_multifidelity_function API:")
    opt_val, opt_pt, history = maximise_multifidelity_function(objective,
                                 config.fidel_space, config.domain, config.fidel_to_opt,
                                 mf_cost, max_capital, opt_method=opt_method,
                                 config=config)

  print('opt_pt: %s'%(str(opt_pt)))
  print('opt_val: %s'%(str(opt_val)))


if __name__ == '__main__':
  main()
