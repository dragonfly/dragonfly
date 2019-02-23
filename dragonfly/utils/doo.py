"""
  Deterministic Optimistic Optimization (DOO) and its parallel version (PDOO)
  Adapted from MFTREE_DET library -- https://github.com/rajatsen91/MFTREE_DET
  References:
    - Sen et al, 2018. Multi-fidelity Black-box Optimization with Hierarchical Partitions
      (http://proceedings.mlr.press/v80/sen18a.html)
    - Grill et al, 2015: Black-box optimization of noisy functions with Unknown
      Smoothness (https://dl.acm.org/citation.cfm?id=2969314)
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-self-use

from __future__ import division

from functools import total_ordering
from argparse import Namespace
try:
  import Queue as Qu
except ImportError:
  import queue as Qu
import numpy as np

from .general_utils import map_to_bounds


class DOOFunction(object):
  """ This just creates a wrapper to call the function by appropriately creating bounds
      and querying appropriately. """

  def __init__(self, func, domain_bounds, vectorised=False):
    """ Constructor.
        func: takes an argument x which is a point in the domain.
        domain_bounds: are the bounds of the domain space
        vectorised: If True it means func can take multiple inputs and produce multiple
                    outputs. If False, the functions can take only single inputs in
                    'column' form.
    """
    self.func = func
    self.domain_bounds = np.array(domain_bounds)
    self.domain_dim = len(domain_bounds)
    self.vectorised = vectorised

  def get_unnormalised_coords(self, X):
    """ Maps points in the cube to the original space. """
    ret_X = None if X is None else map_to_bounds(X, self.domain_bounds)
    return ret_X

  def eval_single_point(self, X):
    """ Evaluates X at the given Z at a single point. """
    if not self.vectorised:
      return float(self.func(X))
    else:
      X = np.array(X).reshape((1, self.domain_dim))
      return float(self.func(X))

  def eval_single_point_normalised(self, X):
    """ Evaluates X at the given Z at a single point using normalised coordinates. """
    X = self.get_unnormalised_coords(X)
    return self.eval_single_point(X)

  def eval_cost_single_point_normalised(self, Z):
    """ Evaluates the cost function at a single point using normalised coordinates. """
    # pylint: disable=unused-argument
    return 1.0


@total_ordering
class Node(object):
  """ Base object for the node of a tree. """

  def __init__(self, cell, value, fidel, upp_bound, height, dimension):
    """ cell: tuple """
    self.cell = cell
    self.value = value
    self.fidelity = fidel
    self.upp_bound = upp_bound
    self.height = height
    self.dimension = dimension

  def __cmp__(self, other):
    return cmp(other.upp_bound, self.upp_bound)

  def __lt__(self, other):
    return other.upp_bound < self.upp_bound

  def __eq__(self, other):
    return other.upp_bound == self.upp_bound


class OptTree(object):
  """ Class for Tree search based optimisation. """

  def __init__(self, doo_obj, nu_max, rho_max, total_budget, K, C_init, tol=1e-3,
               Randomize=False):
    self.doo_obj = doo_obj
    self.nu_max = nu_max
    self.rho_max = rho_max
    self.total_budget = total_budget
    self.K = K
    self.C = C_init
    self.tol = tol
    self.value_dict = {}
    self.qcount = 0
    self.last_z_hist = []
    self.Randomize = Randomize
    self.query_vals = []
    self.query_pts = []

  def get_value(self, cell, fidel):
    """ cell: tuple """
    # pylint: disable=unused-argument
    self.qcount = self.qcount + 1
    x = np.array([(s[0]+s[1])/2.0 for s in list(cell)])
    y = self.doo_obj.eval_single_point_normalised(x)
    if len(self.query_vals) <= self.total_budget:
      self.query_pts.append(x)
      self.query_vals.append(y)
    return y

  def get_queried_pts(self):
    """ Returns queried points and corresponding values. """
    return self.query_pts, self.query_vals

  def querie(self, cell, height, rho, nu, dimension, option=1):
    """ Query. """
    diam = nu*(rho**height)
    if option == 1:
      z = min(max(1 - diam/self.C, self.tol), 1.0)
    else:
      z = 1.0
    if cell in self.value_dict:
      current = self.value_dict[cell]
      if abs(current.fidelity - z) <= self.tol:
        value = current.value
        cost = 0
      else:
        value = self.get_value(cell, z)
        self.last_z_hist = self.last_z_hist + [z]
        if abs(value - current.value) > self.C*abs(current.fidelity - z):
          self.C = 2.0*self.C
        current.value = value
        current.fidelity = z
        self.value_dict[cell] = current
        cost = self.doo_obj.eval_cost_single_point_normalised([z])
    else:
      value = self.get_value(cell, z)
      self.last_z_hist = self.last_z_hist + [z]
      bhi = diam + self.C*(1.0 - z) + value
      self.value_dict[cell] = Node(cell, value, z, bhi, height, dimension)
      cost = self.doo_obj.eval_cost_single_point_normalised([z])

    bhi = diam + self.C*(1.0 - z) + value
    current_object = Node(cell, value, z, bhi, height, dimension)
    return current_object, cost

  def split_children(self, current, rho, nu, option=1):
    """ Split children. """
    pcell = list(current.cell)
    span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]
    if self.Randomize:
      dimension = np.random.choice(range(len(pcell)))
    else:
      dimension = np.argmax(span)
    dd = len(pcell)
    if dimension == current.dimension:
      dimension = (current.dimension - 1)%dd
    cost = 0
    h = current.height + 1
    l = np.linspace(pcell[dimension][0], pcell[dimension][1], self.K+1)
    children = []
    for i in range(len(l)-1):
      cell = []
      for j, _ in enumerate(pcell):
        if j != dimension:
          cell = cell + [pcell[j]]
        else:
          cell = cell + [(l[i], l[i+1])]
      child, c = self.querie(tuple(cell), h, rho, nu, dimension, option)
      children = children + [child]
      cost = cost + c

    return children, cost

  def run_DOO(self, budget, nu, rho):
    """ Runs DOO optimisation. """
    leaf_Q = Qu.PriorityQueue()
    d = self.doo_obj.domain_dim
    cell = tuple([tuple([0, 1]) for _ in range(d)])
    height = 0
    cost = 0
    current, c = self.querie(cell, height, rho, nu, 0)
    cost = cost + c
    leaf_Q.put(current)
    dict_of_points = {}
    while cost <= budget:
      current = leaf_Q.get()
      dict_of_points[current.cell] = {'val':current.value, 'fidel': current.fidelity,
                                      'height':current.height}
      children, curr_cost = self.split_children(current, rho, nu)
      if current.cell == children[0].cell:
        break
      cost = cost + curr_cost
      for child in children:
        leaf_Q.put(child)
    while not leaf_Q.empty():
      c = leaf_Q.get()
      dict_of_points[c.cell] = {'val':c.value, 'fidel': c.fidelity, 'height':c.height}

    #maxi = float(-sys.maxint - 1)
    maxi = float('-inf')
    point = 0
    maxh = 0
    val = 0
    fidel = 0
    for key in dict_of_points:
      c = dict_of_points[key]
      if c['val'] - self.C*(1.0 - c['fidel']) > maxi:    #- nu*(rho**c.height) > maxi:
        maxi = c['val'] - self.C*(1.0 - c['fidel'])      #- nu*(rho**c.height)
        val = c['val']
        fidel = c['fidel']
        point = np.array([(s[0]+s[1])/2 for s in key])
        maxh = c['height']

    return val, fidel, point, cost, maxh

  def run_PDOO(self, mult=0.5):
    """ Runs DOO optimisation iteratively based on the budget. """
    Dm = int(np.log(self.K)/np.log(1/self.rho_max))
    n = self.total_budget / \
        self.doo_obj.eval_cost_single_point_normalised(np.array([1.0]))
    N = int(mult * Dm * np.log(n/np.log(n)))
    budget = self.total_budget/float(N)
    nu = self.nu_max
    total_cost = 0.0
    results = []
    for i in range(N):
      rho = (self.rho_max)**(float(N)/(N-i))
      est, fidel, point, cost, h = self.run_DOO(budget, nu, rho)
      results = results + [(est, fidel, point, cost, h)]
      total_cost = total_cost + cost
    temp = [s[0] - self.C*(1 - s[1]) for s in results]
    index = np.argmax(temp)
    self.last_results = (temp[index], results[index][2], total_cost)

    return results, index

def pdoo_wrap(doo_obj, total_budget, nu_max=1.0, rho_max=0.9, K=2, C_init=0.8, tol=1e-3,
              POO_mult=0.5, Randomize=False, return_history=False):
  """ Wrapper for running PDOO optimisation. """
  # pylint: disable=too-many-locals
  total_budget = total_budget * doo_obj.eval_cost_single_point_normalised([1.0])
  opt_tree = OptTree(doo_obj, nu_max, rho_max, total_budget, K, C_init, tol, Randomize)
  results, index = opt_tree.run_PDOO(POO_mult)
  max_pt = doo_obj.get_unnormalised_coords(results[index][2])
  # IF not return history
  if not return_history:
    return results[index][0], max_pt, None
  history = Namespace()
  max_iter = int(total_budget)
  query_pts, query_vals = opt_tree.get_queried_pts()
  max_val = max(query_vals)
  history.query_step_idxs = [i for i in range(max_iter)]
  history.query_send_times = list(range(0, max_iter))
  history.query_receive_times = list(range(1, max_iter+1))
  history.query_points = [doo_obj.get_unnormalised_coords(x) for x in query_pts]
  history.query_vals = query_vals
  history.query_true_vals = query_vals
  history.curr_opt_vals = []
  history.curr_opt_points = []
  curr_max = -np.inf
  for idx, qv in enumerate(history.query_vals):
    if qv >= curr_max:
      curr_max = qv
      curr_opt_point = history.query_points[idx]
    history.curr_opt_vals.append(curr_max)
    history.curr_opt_points.append(curr_opt_point)
  history.query_eval_times = [1 for _ in range(max_iter)]
  history.curr_true_opt_vals = history.curr_opt_vals
  return max_val, max_pt, history


def pdoo_maximise_from_args(func, bounds, max_capital, *args, **kwargs):
  """ PDOO Maximise from Arguments. """
  doo_obj = DOOFunction(func, bounds)
  return pdoo_wrap(doo_obj, max_capital, *args, **kwargs)

