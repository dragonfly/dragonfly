"""
  Unit tests for Random optimiser on Cartesian Product Domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name

# Local imports
from demos_synthetic.borehole_6.borehole_6 import borehole_6
from demos_synthetic.borehole_6.borehole_6_mf import borehole_6_mf
from demos_synthetic.borehole_6.borehole_6_mf import cost as cost_borehole_6_mf
from demos_synthetic.branin.branin import branin
from demos_synthetic.branin.branin_mf import branin_mf
from demos_synthetic.branin.branin_mf import cost as cost_branin_mf
from demos_synthetic.hartmann3_2.hartmann3_2 import hartmann3_2
from demos_synthetic.hartmann3_2.hartmann3_2_mf import hartmann3_2_mf
from demos_synthetic.hartmann3_2.hartmann3_2_mf import cost as cost_hartmann3_2_mf
from demos_synthetic.hartmann6_4.hartmann6_4 import hartmann6_4
from demos_synthetic.hartmann6_4.hartmann6_4_mf import hartmann6_4_mf
from demos_synthetic.hartmann6_4.hartmann6_4_mf import cost as cost_hartmann6_4_mf
from demos_synthetic.park2_4.park2_4 import park2_4
from demos_synthetic.park2_4.park2_4_mf import park2_4_mf
from demos_synthetic.park2_4.park2_4_mf import cost as cost_park2_4_mf
from demos_synthetic.park2_3.park2_3 import park2_3
from demos_synthetic.park2_3.park2_3_mf import park2_3_mf
from demos_synthetic.park2_3.park2_3_mf import cost as cost_park2_3_mf
from demos_synthetic.park1_3.park1_3 import park1_3
from demos_synthetic.park1_3.park1_3_mf import park1_3_mf
from demos_synthetic.park1_3.park1_3_mf import cost as cost_park1_3_mf
from demos_synthetic.syn_cnn_1.syn_cnn_1 import syn_cnn_1
from demos_synthetic.syn_cnn_1.syn_cnn_1_mf import syn_cnn_1_mf
from demos_synthetic.syn_cnn_1.syn_cnn_1_mf import cost as cost_syn_cnn_1_mf
from demos_synthetic.syn_cnn_2.syn_cnn_2 import syn_cnn_2
from demos_synthetic.syn_cnn_2.syn_cnn_2_mf import syn_cnn_2_mf
from demos_synthetic.syn_cnn_2.syn_cnn_2_mf import cost as cost_syn_cnn_2_mf
from ..exd.cp_domain_utils import get_processed_func_from_raw_func_for_cp_domain, \
                                get_raw_point_from_processed_point, \
                                load_cp_domain_from_config_file, \
                                load_config_file
from ..exd.experiment_caller import CPFunctionCaller, get_multifunction_caller_from_config
from ..exd.worker_manager import SyntheticWorkerManager
from . import random_optimiser
from ..utils.ancillary_utils import is_nondecreasing, get_list_of_floats_as_str
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.reporters import get_reporter


class CPOptimiserBaseTestCase(object):
  """ Base test class for Optimisers on Cartesian Product Spaces. """
  # pylint: disable=no-member

  def setUp(self):
    """ Set up. """
    self.max_capital = 30
    self._child_set_up()
    self.worker_manager_1 = SyntheticWorkerManager(1, time_distro='const')
    self.worker_manager_3 = SyntheticWorkerManager(3, time_distro='halfnormal')
    self.opt_problems = [
      ('../demos_synthetic/branin/config.json', (branin,)),
      ('../demos_synthetic/hartmann3_2/config.json', (hartmann3_2,)),
      ('../demos_synthetic/hartmann6_4/config.json', (hartmann6_4,)),
      ('../demos_synthetic/borehole_6/config.json', (borehole_6,)),
      ('../demos_synthetic/park2_4/config.json', (park2_4,)),
      ('../demos_synthetic/park2_3/config.json', (park2_3,)),
      ('../demos_synthetic/park1_3/config.json', (park1_3,)),
      ('../demos_synthetic/syn_cnn_1/config.json', (syn_cnn_1,)),
      ('../demos_synthetic/syn_cnn_2/config.json', (syn_cnn_2,)),
      ]

  def _child_set_up(self):
    """ Child set up. """
    pass

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _run_optimiser(cls, raw_func, domain_config_file, worker_manager, max_capital, mode,
                     *args, **kwargs):
    """ Run the optimiser from given args. """
    raise NotImplementedError('Implement in a child class.')

  def test_instantiation(self):
    """ Tests instantiation of the optimiser. """
    self.report('Testing instantiation of optimiser.')
    for idx, (dcf, raw_prob_funcs) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing instantiation of optimiser for %s.'%(
                  idx + 1, len(self.opt_problems), dcf), 'test_result')
      config = load_config_file(dcf)
      if hasattr(config, 'fidel_space'):
        raw_func, raw_fidel_cost_func = raw_prob_funcs
      else:
        raw_func = raw_prob_funcs[0]
        raw_fidel_cost_func = None
      func_caller = get_multifunction_caller_from_config(raw_func, config,
                                  raw_fidel_cost_func=raw_fidel_cost_func)
      optimiser = self._child_instantiate_optimiser(func_caller, self.worker_manager_1,
                    options=None, reporter=get_reporter('silent'))
      self.report('Instantiated %s object.'%(type(optimiser)))
      for attr in dir(optimiser):
        if not attr.startswith('_'):
          self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))),
                      'test_result')

  def _test_optimiser_results(self, opt_val, opt_point, history, dcf):
    """ Tests optimiser results. """
    cp_domain, orderings = load_cp_domain_from_config_file(dcf)
    raw_opt_point = get_raw_point_from_processed_point(opt_point,
                        cp_domain, orderings.index_ordering, orderings.dim_ordering)
    self.report('Opt point: proc=%s, raw=%s.'%(opt_point, raw_opt_point))
    self.report('True opt val sequence: %s.'%(
      get_list_of_floats_as_str(history.curr_true_opt_vals)))
    saved_in_history = [key for key, _ in list(history.__dict__.items()) if not
                        key.startswith('__')]
    self.report('Stored in history: %s.'%(saved_in_history), 'test_result')
    assert opt_val == history.curr_opt_vals[-1]
    assert len(opt_point) == cp_domain.num_domains
    assert is_nondecreasing(history.curr_opt_vals)
    assert is_nondecreasing(history.curr_true_opt_vals)

  def test_optimisation_single(self):
    """ Tests running of the optimiser. """
    self.report('')
    self.report('Testing optimisation on a single worker.')
    for idx, (dcf, raw_prob_funcs) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing optimisation  with 1 worker on %s.'%(
                  idx + 1, len(self.opt_problems), dcf), 'test_result')
      self.worker_manager_1.reset()
      opt_val, opt_point, history = self._run_optimiser(raw_prob_funcs, dcf,
                                      self.worker_manager_1, self.max_capital, 'asy')
      self._test_optimiser_results(opt_val, opt_point, history, dcf)
      self.report('')

  def test_optimisation_asynchronous(self):
    """ Tests running the optimiser in asynchronous mode. """
    self.report('')
    self.report('Testing optimisation on three asynchronous workers.')
    for idx, (dcf, raw_prob_funcs) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing optimisation with 3 asynchronous workers on %s.'%(
                   idx+1, len(self.opt_problems), dcf), 'test_result')
      self.worker_manager_3.reset()
      opt_val, opt_point, history = self._run_optimiser(raw_prob_funcs, dcf,
                              self.worker_manager_3, self.max_capital, 'asy')
      self._test_optimiser_results(opt_val, opt_point, history, dcf)
      self.report('')

  def test_optimisation_synchronous(self):
    """ Tests running the optimiser in asynchronous mode. """
    self.report('')
    self.report('Testing optimisation on three synchronous workers.')
    for idx, (dcf, raw_prob_funcs) in enumerate(self.opt_problems):
      self.report('[%d/%d] Testing optimisation with 3 synchronous workers on %s.'%(
                   idx+1, len(self.opt_problems), dcf), 'test_result')
      self.worker_manager_3.reset()
      opt_val, opt_point, history = self._run_optimiser(raw_prob_funcs, dcf,
                              self.worker_manager_3, self.max_capital, 'syn')
      self._test_optimiser_results(opt_val, opt_point, history, dcf)
      self.report('')


class MFCPOptimiserBaseTestCase(CPOptimiserBaseTestCase):
  """ Base test class for Multi-fidelity Optimisers on Cartesian Product Spaces. """
  # pylint: disable=no-member

  def setUp(self):
    """ Set up. """
    self.max_capital = 20
    self._child_set_up()
    self.worker_manager_1 = SyntheticWorkerManager(1, time_distro='const')
    self.worker_manager_3 = SyntheticWorkerManager(3, time_distro='halfnormal')
    self.opt_problems = [
      ('../demos_synthetic/branin/config_mf.json',
        (branin_mf, cost_branin_mf)),
      ('../demos_synthetic/hartmann3_2/config_mf.json',
        (hartmann3_2_mf, cost_hartmann3_2_mf)),
      ('../demos_synthetic/hartmann6_4/config_mf.json',
        (hartmann6_4_mf, cost_hartmann6_4_mf)),
      ('../demos_synthetic/borehole_6/config_mf.json',
        (borehole_6_mf, cost_borehole_6_mf)),
      ('../demos_synthetic/park1_3/config_mf.json',
        (park1_3_mf, cost_park1_3_mf)),
      ('../demos_synthetic/park2_3/config_mf.json',
        (park2_3_mf, cost_park2_3_mf)),
      ('../demos_synthetic/park2_4/config_mf.json',
        (park2_4_mf, cost_park2_4_mf)),
      ('../demos_synthetic/syn_cnn_1/config_mf.json',
        (syn_cnn_1_mf, cost_syn_cnn_1_mf)),
      ('../demos_synthetic/syn_cnn_2/config_mf.json',
        (syn_cnn_2_mf, cost_syn_cnn_2_mf)),
      ]

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _run_optimiser(cls, raw_func, domain_config_file, worker_manager, max_capital, mode,
                     *args, **kwargs):
    """ Run the optimiser from given args. """
    raise NotImplementedError('Implement in a child class.')


class CPRandomOptimiserTestCase(CPOptimiserBaseTestCase, BaseTestClass):
  """ Unit tests for the Random optimiser on cartesian product spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    return random_optimiser.CPRandomOptimiser(func_caller, worker_manager,
                                              options=options, reporter=reporter)

  @classmethod
  def _run_optimiser(cls, raw_prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Run the optimiser from given args. """
    raw_func = raw_prob_funcs[0]
    cp_dom, orderings = load_cp_domain_from_config_file(domain_config_file)
    proc_func = get_processed_func_from_raw_func_for_cp_domain(
                  raw_func, cp_dom, orderings.index_ordering, orderings.dim_ordering)
    func_caller = CPFunctionCaller(proc_func, cp_dom, raw_func=raw_func,
                                   domain_orderings=orderings)
    return random_optimiser.random_optimiser_from_func_caller(func_caller, worker_manager,
                                                       max_capital, mode, *args, **kwargs)


class MFCPRandomOptimiserTestCase(MFCPOptimiserBaseTestCase, BaseTestClass):
  """ Unit tests for Multi-fidelity random optimiser on cartesian product spaces. """

  @classmethod
  def _child_instantiate_optimiser(cls, func_caller, worker_manager, options, reporter):
    """ Instantiate the optimiser. """
    return random_optimiser.MFCPRandomOptimiser(func_caller, worker_manager,
                                                options=options, reporter=reporter)

  @classmethod
  def _run_optimiser(cls, raw_prob_funcs, domain_config_file, worker_manager, max_capital,
                     mode, *args, **kwargs):
    """ Run the optimiser from given args. """
    raw_func, raw_fidel_cost_func = raw_prob_funcs
    config = load_config_file(domain_config_file)
    func_caller = get_multifunction_caller_from_config(raw_func, config,
                                raw_fidel_cost_func=raw_fidel_cost_func)
    return random_optimiser.mf_random_optimiser_from_func_caller(func_caller,
             worker_manager, max_capital, mode)


if __name__ == '__main__':
  execute_tests()

