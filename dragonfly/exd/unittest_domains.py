"""
  Unit tests for domains.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from __future__ import absolute_import
import numpy as np
# Local imports
from . import domains
from ..utils.general_utils import map_to_bounds
from ..utils.base_test_class import BaseTestClass, execute_tests

class DomainBaseTestCase(object):
  """ Unit tests for Base class. """
  # pylint: disable=no-member

  def setUp(self):
    """ Set up. """
    self.domain_obj = None
    self.points = None
    self.non_points = None
    self._child_set_up()

  def _child_set_up(self):
    """ Child set up. """
    raise NotImplementedError('Implement in a child class.')

  def test_instantiation(self):
    """ Tests instantiation of an object. """
    self.report('Testing instantiation of %s class.'%(type(self.domain_obj)))
    self.report('Created object: %s'%(self.domain_obj), 'test_result')

  def test_membership(self):
    """ Testing membership. """
    self.report('Testing membership of %s class.'%(type(self.domain_obj)))
    points_are_in = [self.domain_obj.is_a_member(pt) for pt in self.points]
    non_points_are_not_in = [not self.domain_obj.is_a_member(pt) for pt in
                             self.non_points]
    assert all(points_are_in)
    assert all(non_points_are_not_in)


class EuclideanDomainTestCase(DomainBaseTestCase, BaseTestClass):
  """ Test class for Euclidean Objects. """

  def _child_set_up(self):
    """ Child set up. """
    self.domain_obj = domains.EuclideanDomain([[0, 2.3], [3.4, 8.9], [0.12, 1.0]])
    self.points = [map_to_bounds(np.random.random((self.domain_obj.dim,)),
                                 self.domain_obj.bounds)
                   for _ in range(5)]
    self.non_points = [map_to_bounds(np.random.random((self.domain_obj.dim,)),
                                     np.array([[3.5, 9.8], [-1.0, 1.1], [2.3, 4.5]]))
                       for _ in range(5)]


class IntegralDomainTestCase(DomainBaseTestCase, BaseTestClass):
  """ Test class for IntegralDomain Objects. """

  def _child_set_up(self):
    """ Child set up. """
    self.domain_obj = domains.IntegralDomain([[0, 10], [-10, 100], [45, 78.4]])
    self.points = [[9, 56, 78], [5, 0, 68], [0, -1, 70]]
    self.non_points = [[11, 0, 67], [5.6, 11, 58], [4, 3.0, 70], [9, 56, 67, 9]]


class DiscreteDomainTestCase(DomainBaseTestCase, BaseTestClass):
  """ Discrete Domain. """

  def _child_set_up(self):
    """ Child set up. """
    self.domain_obj = domains.DiscreteDomain(['abc', 5, 6.5, int, 'k'])
    self.points = ['k', type(4), 6.5]
    self.non_points = ['ab', 75.8, 'qwerty', None]


class DiscreteNumericDomainTestCase(DomainBaseTestCase, BaseTestClass):
  """ Discrete Numeric Domain. """

  def _child_set_up(self):
    """ Child set up. """
    self.domain_obj = domains.DiscreteNumericDomain([4, 5, 207.2, -2.3])
    self.points = [207.2, 5, -2.3]
    self.non_points = [5.4, -1.1, 'kky', None]

  def test_non_numeric_discrete_domain(self):
    """ Constructor. """
    self.report('Testing if exception is raised for non numeric elements in a ' +
                'discrete domain.')
    exception_raised = False
    try:
      domains.DiscreteNumericDomain(['abc', 5, 6.5, int, 'k'])
    except ValueError:
      exception_raised = True
    assert exception_raised


class ProdDiscreteDomainTestCase(DomainBaseTestCase, BaseTestClass):
  """ ProdDiscreteDomain Domain. """

  def _child_set_up(self):
    """ Child set up. """
    self.domain_obj = domains.ProdDiscreteDomain([['abc', 5, 6.5], [None, float]])
    self.points = [('abc', float), [6.5, None], [5, None]]
    self.non_points = [['abc', float, float], [5, 7], 'qwerty', [99], 6]


class ProdDiscreteNumericDomain(DomainBaseTestCase, BaseTestClass):
  """ ProdDiscreteDomain Domain. """

  def _child_set_up(self):
    """ Child set up. """
    self.domain_obj = domains.ProdDiscreteNumericDomain(
                        [[4.3, 2.1, 9.8, 10],
                         [11.2, -23.1, 19.8],
                         [1123, 213, 1980, 1.1]])
    self.points = [[2.1, -23.1, 1980], (10, 11.2, 1.1), [9.8, 19.8, 1123]]
    self.non_points = [[2.1 -13.1, 1980], ('kky', 11.2, 1.1), [9.8, 19.8, 1123, 21]]

  def test_non_numeric_prod_discrete_domain(self):
    """ Constructor. """
    self.report('Testing if exception is raised non numeric product discrete domain.')
    exception_raised = False
    try:
      domains.ProdDiscreteNumericDomain(
        [[2.1, 9.8, 10],
         [11.2, -23.1, 19.8],
         [1123, 213, 'squirrel', 1.1]])
    except ValueError:
      exception_raised = True
    assert exception_raised


class CartesianProductDomainTestCase(DomainBaseTestCase, BaseTestClass):
  """ Unit tests for CartesianProductDomain. """

  def _child_set_up(self):
    """ Child set up. """
    list_of_domains = [
      domains.EuclideanDomain([[0, 2.3], [3.4, 8.9], [0.12, 1.0]]),
      domains.IntegralDomain([[0, 10], [-10, 100], [45, 78.4]]),
      domains.ProdDiscreteDomain([['abc', 5, 6.5], [None, float]]),
      domains.DiscreteDomain(['abc', 5, 6.5, int, 'k']),
      ]
    self.domain_obj = domains.CartesianProductDomain(list_of_domains)
    self.points = [
      [[2.2, 3.6, 0.99], [5, 0, 68], ('abc', float), int],
      [[1.2, 6.4, 0.2], [0, -1, 70], [6.5, None], 'k'],
      [[0.2, 7.9, 0.5], [0, -1, 69], ['abc', None], 5],
      ]
    self.non_points = [
      [[2.2, 3.6, 0.99], [5, 0, 68], ('abc', float), float],
      [[1.2, 6.4, 2.2], [0, -1, 70], [6.5, None], 'k'],
      [[0.2, 7.9, 0.5], [0, -1, 69], ['abc', None, float], 5],
      [[3.2, 7.9, 0.5], [-20, -1, 69], ['abc', None, float], 5],
      [[2.2, 3.6, 0.99], [5, 0, 68], ('abc', float), int, (3.2, 8.5)],
      ]


if __name__ == '__main__':
  execute_tests()

