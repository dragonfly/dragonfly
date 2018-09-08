"""
  Parser for json and protocol buffer files
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import

import sys
import json
from collections import OrderedDict
import numpy as np

def unicode_to_str(data):
  """ Unicode to string conversion. """
  if not isinstance(data, str):
    return data.encode('utf-8')
  return data

def load_parameters(config):
  """ Parses all the parameters from json config file. """
  exp_info = {}
  exp_info['name'] = unicode_to_str(config.get('name'))
  if exp_info['name'] is None:
    raise ValueError('Experiment name is required')

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

  return {"exp_info" : exp_info, "domain" : parameters, "fidel_space" : fidel_parameters}

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
  if param['type'] in ['float', 'int', 'discrete', 'discrete_numeric']:
    if not isinstance(_dim, (int, float, long)):
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
    nn_params['min_mass'] = parameter.get('min_mass', 'inf')
    nn_params['max_in_degree'] = parameter.get('max_in_degree', 'inf')
    nn_params['max_out_degree'] = parameter.get('max_out_degree', 'inf')
    nn_params['max_num_edges'] = parameter.get('max_num_edges', 'inf')
    nn_params['max_num_units_per_layer'] = parameter.get('max_num_units_per_layer', 'inf')
    nn_params['min_num_units_per_layer'] = parameter.get('min_num_units_per_layer', 0)
    # For CNNs add strides
    if param['type'].startswith('cnn'):
      nn_params['max_num_2strides'] = parameter.get('max_num_2strides', 'inf')
    for nnp_key, nnp_val in nn_params.iteritems():
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
  except:
    raise Exception('Error in loading config file: ' + config_file)

  return load_parameters(config)

def read_pb(config_file):
  """ Read from protocol buffer file. """
  # pylint: disable=import-error
  try:
    from google.protobuf import text_format
    from google.protobuf.json_format import MessageToDict
  except ImportError:
    raise ImportError('Protocol Buffer library is not installed')

  from parse import config_pb2
  config_pb = config_pb2.Experiment()

  _file = open(config_file, "rb")
  text_format.Merge(_file.read(), config_pb)
  _file.close()

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
