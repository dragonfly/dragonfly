"""
  Parser for json and protocol buffer files
  -- kvysyara@andrew.cmu.edu
"""

# pylint: disable=invalid-name

from __future__ import absolute_import

import sys
import json
from collections import OrderedDict
from numbers import Number
import numpy as np

# Python 3 issue with unicode classes
try:
  UNICODE_EXISTS = bool(type(unicode))
except NameError:
  unicode = str


def unicode_to_str(data):
  """ Unicode to string conversion. """
  if not isinstance(data, str):
    return data.encode('utf-8')
  return data


def _load_fidel_to_opt_parameters(param):
  """ Loads fidel_to_opt parameters. """
  if isinstance(param, (list, tuple)):
    ret = []
    for elem in param:
      ret.append(_load_fidel_to_opt_parameters(elem))
  else:
    ret = param # just return the param
    if isinstance(param, unicode):
      ret = unicode_to_str(ret)
  return ret


def _load_domain_constraints(domain_constraints):
  """ Loads the domain constraints. """
  processed_constraints = []
  # The constraints will be represented as a list of 3-tuples. The first of each tuple
  # will be the name of the constraint, the second will be the constraint, and the third
  # to store any ancillary information. The third item will be a dictionary containing
  # any other information specified in the configuration file.
  for _, constraint_data in domain_constraints.items():
    curr_constraint_dict = {}
    for key, val in constraint_data.items():
      key = unicode_to_str(key)
      val = unicode_to_str(val)
      curr_constraint_dict[key] = val
    processed_constraints.append(curr_constraint_dict)
  return processed_constraints



def load_parameters(config):
  """ Parses all the parameters from json config file. """
  # pylint: disable=too-many-branches
  exp_info = {}
  exp_info['name'] = unicode_to_str(config.get('name'))
  if exp_info['name'] is None:
    raise ValueError('Experiment name is required')
  # Domain -------------------------------------------------------
  parameters = []
  _parameters = config['domain']
  if isinstance(_parameters, dict):
    for key in list(_parameters.keys()):
      param = load_parameter(_parameters[key], key)
      parameters.append(param)
  elif isinstance(_parameters, list):
    for _parameter in _parameters:
      param = load_parameter(_parameter)
      parameters.append(param)
  else:
    raise ValueError('Wrong parameter type.')
  # domain_constraints -------------------------------------------
  domain_constraints = config.get('domain_constraints', None)
  if domain_constraints is not None:
    domain_constraints = _load_domain_constraints(domain_constraints)
  # Fidelity space -----------------------------------------------
  fidel_parameters = []
  _parameters = config.get('fidel_space', {})
  if isinstance(_parameters, dict):
    for key in list(_parameters.keys()):
      param = load_parameter(_parameters[key], key)
      fidel_parameters.append(param)
  elif isinstance(_parameters, list):
    for _parameter in _parameters:
      param = load_parameter(_parameter)
      fidel_parameters.append(param)
  else:
    raise ValueError('Wrong parameter type.')
  # fidel_space_constraints ---------------------------------------
  fidel_space_constraints = config.get('fidel_space_constraints', None)
  if fidel_space_constraints is not None:
    fidel_space_constraints = _load_domain_constraints(fidel_space_constraints)
  # fidel_to_opt --------------------------------------------------
  fidel_to_opt = config.get('fidel_to_opt', None)
  if fidel_to_opt is not None:
    fidel_to_opt = _load_fidel_to_opt_parameters(fidel_to_opt)
  # Return
  return {'exp_info':exp_info, 'domain':parameters, 'fidel_space':fidel_parameters,
          'fidel_to_opt':fidel_to_opt, 'domain_constraints':domain_constraints,
          'fidel_space_constraints':fidel_space_constraints}


def load_parameter(parameter, key=None):
  """ Parses each parameter and return a dict """
  # pylint: disable=too-many-branches
  # pylint: disable=too-many-statements
  # Common parameters
  _name = parameter.get('name', key)
  if _name is None:
    raise ValueError('Parameter name is required')
  _type = parameter.get('type', 'float')
  _kernel = parameter.get('kernel', '')
  # Common for Euclidean, Integral, Discrete, Discrete Numeric domains
  _dim = parameter.get('dim', "")
  # For Euclidean/Integral
  _min = parameter.get('min', -np.inf)
  _max = parameter.get('max', np.inf)
  # For Discrete
  _items = parameter.get('items', '')
  # For neural networks -- see below

  # Now process them
  param = {}
  param['name'] = unicode_to_str(_name)
  param['kernel'] = unicode_to_str(_kernel)
  param['type'] = unicode_to_str(_type).lower()
  # First for regular domains
  if param['type'] in ['float', 'int', 'discrete', 'discrete_numeric', 'boolean']:
    if not isinstance(_dim, Number):
      _dim = unicode_to_str(_dim)
    if _dim != "":
      _dim = int(_dim)
    param['dim'] = _dim
    if param['type'] in ['float', 'int']:
      param['min'] = _min
      param['max'] = _max
    elif param['type'] == 'discrete':
      if _items == '':
        raise ValueError('List of items required')
      if isinstance(_items, list):
        param['items'] = [unicode_to_str(i) for i in _items]
      else:
        param['items'] = unicode_to_str(_items).split('-')
    elif param['type'] == 'discrete_numeric':
      if _items == '':
        raise ValueError('List or range of items required')
      elif ':' not in _items:
        param['items'] = [float(x) for x in unicode_to_str(_items).split('-')]
      else:
        _range = [float(x) for x in unicode_to_str(_items).split(':')]
        param['items'] = list(np.arange(_range[0], _range[2], _range[1]))
  elif param['type'].startswith(('cnn', 'mlp')):
    nn_params = {}
    nn_params['max_num_layers'] = parameter.get('max_num_layers', 'inf')
    nn_params['min_num_layers'] = parameter.get('min_num_layers', 0)
    nn_params['max_mass'] = parameter.get('max_mass', 'inf')
    nn_params['min_mass'] = parameter.get('min_mass', 0)
    nn_params['max_in_degree'] = parameter.get('max_in_degree', 'inf')
    nn_params['max_out_degree'] = parameter.get('max_out_degree', 'inf')
    nn_params['max_num_edges'] = parameter.get('max_num_edges', 'inf')
    nn_params['max_num_units_per_layer'] = parameter.get('max_num_units_per_layer', 'inf')
    nn_params['min_num_units_per_layer'] = parameter.get('min_num_units_per_layer', 0)
    # For CNNs add strides
    if param['type'].startswith('cnn'):
      nn_params['max_num_2strides'] = parameter.get('max_num_2strides', 'inf')
    for nnp_key, nnp_val in nn_params.items():
      if isinstance(nnp_val, str):
        nnp_val = unicode_to_str(nnp_val)
      nnp_val = np.inf if nnp_val == 'inf' else nnp_val
      param[nnp_key] = nnp_val
    # Finally add the following
    param['dim'] = ''
  else:
    raise ValueError('Unknown type %s.'%(param['type']))
  return param


def read_json(config_file):
  """ Read from json file. """
  try:
    with open(config_file, 'r') as _file:
      config = json.load(_file, object_pairs_hook=OrderedDict)
      _file.close()
  except Exception as e:
    raise Exception('Error in loading config file: ' + config_file + '.\n -- ' + str(e))
  return load_parameters(config)


def read_pb(config_file):
  """ Read from protocol buffer file. """
  # pylint: disable=import-error
  try:
    from google.protobuf import text_format
    from google.protobuf.json_format import MessageToDict
  except ImportError:
    raise ImportError('Protocol Buffer library is not installed')
  # Read PB file
  from . import config_pb2
  config_pb = config_pb2.Experiment()
  _file = open(config_file, "rb")
  text_format.Merge(_file.read(), config_pb)
  _file.close()
  # Load parameters and return
  config = MessageToDict(config_pb)
  config['fidel_space'] = config.pop('fidelSpace', {})
  return load_parameters(config)


def config_parser(config_file):
  """Reads config files and creates domain objects. """
  if config_file.endswith('.json'):
    params = read_json(config_file)
  elif config_file.endswith('.pb'):
    params = read_pb(config_file)
  else:
    raise ValueError('Wrong Config file: %s' % (config_file))
  return params


if __name__ == '__main__':
  if len(sys.argv) < 2:
    raise ValueError('Need Config File.')
  config_parser(sys.argv[1])

