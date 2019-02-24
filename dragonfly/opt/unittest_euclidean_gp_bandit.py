"""
  Unit tests for EuclideanGPBandit.
  -- kandasamy@cs.cmu.edu
"""
import unittest

# Local imports
from ..gp.euclidean_gp import euclidean_gp_args
from . import gp_bandit
from .unittest_euclidean_random_optimiser import EuclideanOptimisersBaseTestCase, \
                                                    MFEuclideanOptimisersBaseTestCase
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.option_handler import load_options


@unittest.skip
class EuclideanGPBanditTestCase(EuclideanOptimisersBaseTestCase, BaseTestClass):
# class EuclideanGPBanditTestCase(EuclideanOptimisersBaseTestCase):
  """ Unit tests for the GP Bandit in Euclidean spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager,
                                   options=None, reporter=None):
    """ Instantiates a GP bandit in Euclidean space. """
    return gp_bandit.EuclideanGPBandit(func_caller, worker_manager, is_mf=False,
             options=options, reporter=reporter)

  @classmethod
  def run_optimiser(cls, func_caller, worker_manager, max_capital, mode, *args, **kwargs):
    """ Runs optimiser. """
    return gp_bandit.gpb_from_func_caller(func_caller, worker_manager, max_capital,
             mode=mode, is_mf=False, *args, **kwargs)


# class EuclideanAddGPBanditTestCase(EuclideanOptimisersBaseTestCase, BaseTestClass):
class EuclideanAddGPBanditTestCase(EuclideanOptimisersBaseTestCase):
  """ Unit tests for Additive GP Bandit in Euclidean spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager,
                                   options=None, reporter=None):
    """ Instantiates an additive GP bandit in Euclidean space. """
    if options is None:
      opt_args = gp_bandit.get_all_gp_bandit_args(euclidean_gp_args)
      options = load_options(opt_args)
    options.use_additive_gp = True
    return gp_bandit.EuclideanGPBandit(func_caller, worker_manager, is_mf=False,
             options=options, reporter=reporter)

  @classmethod
  def run_optimiser(cls, func_caller, worker_manager, max_capital, mode, *args, **kwargs):
    """ Runs optimiser. """
    return gp_bandit.gpb_from_func_caller(func_caller, worker_manager, max_capital,
             mode=mode, acq='add_ucb', is_mf=False, domain_add_max_group_size=0,
             *args, **kwargs)


@unittest.skip
class MFEuclideanGPBanditTestCase(MFEuclideanOptimisersBaseTestCase, BaseTestClass):
# class MFEuclideanGPBanditTestCase(MFEuclideanOptimisersBaseTestCase):
  """ Unit tests for multi-fidelity GP Bandit Optimiser in Euclidean spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager,
                                   options=None, reporter=None):
    """ Instantiates a MFGP bandit in Euclidean space. """
    return gp_bandit.EuclideanGPBandit(func_caller, worker_manager, is_mf=True,
             options=options, reporter=reporter)

  @classmethod
  def run_optimiser(cls, func_caller, worker_manager, max_capital, mode, *args, **kwargs):
    """ Runs optimiser. """
    return gp_bandit.gpb_from_func_caller(func_caller, worker_manager, max_capital,
             mode=mode, is_mf=True, *args, **kwargs)


# class MFEuclideanAddGPBanditTestCase(MFEuclideanOptimisersBaseTestCase, BaseTestClass):
class MFEuclideanAddGPBanditTestCase(MFEuclideanOptimisersBaseTestCase):
  """ Unit tests for Additive GP Bandit in MF Euclidean spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager,
                                   options=None, reporter=None):
    """ Instantiates an additive MF GP bandit in Euclidean space. """
    if options is None:
      opt_args = gp_bandit.get_all_gp_bandit_args(euclidean_gp_args)
      options = load_options(opt_args)
    options.use_additive_gp = True
    return gp_bandit.EuclideanGPBandit(func_caller, worker_manager, is_mf=False,
             options=options, reporter=reporter)

  @classmethod
  def run_optimiser(cls, func_caller, worker_manager, max_capital, mode, *args, **kwargs):
    """ Runs optimiser. """
    return gp_bandit.gpb_from_func_caller(func_caller, worker_manager, max_capital,
             mode=mode, acq='add_ucb', is_mf=True, domain_add_max_group_size=0,
             *args, **kwargs)


# class MFEuclideanESPGPBanditTestCase(MFEuclideanOptimisersBaseTestCase, BaseTestClass):
class MFEuclideanESPGPBanditTestCase(MFEuclideanOptimisersBaseTestCase):
  """ Unit tests for GP Bandit with ESP kernel in MF Euclidean spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager,
                                   options=None, reporter=None):
    """ Instantiates an additive MF GP bandit in Euclidean space. """
    if options is None:
      opt_args = gp_bandit.get_all_gp_bandit_args(euclidean_gp_args)
      options = load_options(opt_args)
    options.domain_kernel_type = 'esp'
    return gp_bandit.EuclideanGPBandit(func_caller, worker_manager, is_mf=False,
             options=options, reporter=reporter)

  @classmethod
  def run_optimiser(cls, func_caller, worker_manager, max_capital, mode, *args, **kwargs):
    """ Runs optimiser. """
    return gp_bandit.gpb_from_func_caller(func_caller, worker_manager, max_capital,
             mode=mode, acq='add_ucb', is_mf=True, domain_add_max_group_size=0,
             *args, **kwargs)


if __name__ == '__main__':
  execute_tests()

