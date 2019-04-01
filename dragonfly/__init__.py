"""
  Dragonfly: An open source library for scalable and robust Bayesian optimisation.

  This file contains the main APIs. See apis/opt.py and apis/moo.py for function
  definitions and descriptions of the arguments.

  Rerefernce:
    "Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian
    optimisation with Dragonfly", Kandasamy K, Vysyaraju K V, Neiswanger W, Paria B,
    Collins C R, Schneider J, Poczos B, Xing E P.

  -- kandasamy@cs.cmu.edu
"""

# Main APIs
from .apis.opt import maximise_function, minimise_function, \
                      maximise_multifidelity_function, minimise_multifidelity_function, \
                      maximize_function, minimize_function, \
                      maximize_multifidelity_function, minimize_multifidelity_function
from .apis.moo import multiobjective_maximise_functions, \
                      multiobjective_minimise_functions, \
                      multiobjective_maximize_functions, \
                      multiobjective_minimize_functions

from .exd.cp_domain_utils import load_config_file, load_config

