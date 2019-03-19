<span style="font-size:3em">**Main APIs**</span>

&nbsp;

## maximise_function
```python
opt_val, opt_pt, history = maximise_function(func, domain, max_capital,
                                             capital_type='num_evals',
                                             opt_method='bo',
                                             config=None,
                                             options=None,
                                             reporter='default')
```
Maximises a function `func` over the domain `domain`.  
**Arguments:**  
- `func`: The function to be maximised.  
- `domain`: The domain over which the function should be maximised, should be an instance  of the Domain class in [`exd/domains.py`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/domains.py).  If domain is a list of the form `[[l1, u1], [l2, u2], ...]` where `li < ui`, then we create a Euclidean domain with lower bounds `li` and upper bounds `ui` along each dimension.  
- `max_capital`: The maximum capital (time budget or number of evaluations) available for optimisation.  
- `capital_type`: The type of capital. Should be one of `'num_evals'`, `'return_value'` or
  `'realtime'`.  Default is `'num_evals'` which indicates the number of evaluations. If
  `'realtime'`, we will use wall clock time.  
- `opt_method`: The method used for optimisation. Could be one of `'bo'`, `'rand'`, `'ga'`,
  `'ea'`, `'direct'`, or `'pdoo'`. Default is `'bo'`.  `'bo'`: Bayesian optimisation,
 `'ea'`/`'ga'`: Evolutionary algorithm, `'rand'`: Random search, `'direct'`: Dividing Rectangles, `'pdoo'`: PDOO. 
<li> config: Either a configuration file or or parameters returned by
             [`exd.cp_domain_utils.load_config_file`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/cp_domain_utils.py). `config` can be `None` only if `domain`
             is a [`EuclideanDomain`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/domains.py) object. </li>
- `options`: Additional hyper-parameters for optimisation, as a namespace.
- `reporter`: A stream to print progress made during optimisation, or one of the following
  strings `'default'`, `'silent'`. If `'silent'`, then it suppresses all outputs. If `'default'`, writes to stdout.  
**Returns**:  
- `opt_val`: The maximum value found during the optimisation procedure.  
- `opt_pt`: The corresponding optimum point.  
- `history`: A record of the optimisation procedure which include the point evaluated and the values at each time step.  


&nbsp;

## minimise_function
```python
opt_val, opt_pt, history = minimise_function(func, domain, max_capital,
                                             capital_type='num_evals',
                                             opt_method='bo',
                                             config=None,
                                             options=None,
                                             reporter='default')
```
**Arguments:**  
Same as `maximise_function` (see above), but now `func` is to be minimised.  
**Returns**:  
Same as `maximise_function` (see above), but now `opt_val` is the minimum value found during optimisation.

&nbsp;

## maximise_multifidelity_function
```
maximise_multifidelity_function(func, fidel_space, domain, fidel_to_opt,
                                fidel_cost_func, max_capital,
                                capital_type='return_value',
                                opt_method='bo',
                                config=None,
                                options=None,
                                reporter='default'):
```
Maximises a multi-fidelity function 'func' over the domain 'domain' and fidelity
space 'fidel_space'. See the [BOCA paper](https://arxiv.org/pdf/1703.06240.pdf) for more
information on multi-fidelity optimisation.  

**Arguments:**
<ul>
<li> `'func'`: The function to be maximised. Takes two arguments `func(z, x)` where `z` is a
member of the fidelity space and `x` is a member of the domain. </li>
<li> `'fidel_space'`: The fidelity space from which the approximations are obtained.
                  Should be an instance of the Domain class in [`exd/domains.py`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/domains.py).
                  If of the form `[[l1, u1], [l2, u2], ...]` where `li < ui`, then we will
                  create a Euclidean domain with lower bounds `li` and upper bounds
                  `ui` along each dimension. </li>
<li> `'domain'`: The domain over which the function should be maximised, should be an
             instance of the Domain class in [`exd/domains.py`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/domains.py).
             If domain is a list of the form `[[l1, u1], [l2, u2], ...]` where `li < ui`,
             then we will create a Euclidean domain with lower bounds `li` and upper bounds
             `ui` along each dimension. </li>
<li> `'fidel_to_opt'`: The point at the fidelity space at which we wish to maximise func. </li>
<li> `'max_capital'`: The maximum capital (time budget or number of evaluations) available
                  for optimisation. </li>
<li> `'capital_type'`: The type of capital. Should be one of 'return_value' or 'realtime'.
                   Default is `'return_value'` which indicates we will use the value returned
                   by `fidel_cost_func`. If `'realtime'`, we will use wall clock time. </li>
<li> `'opt_method'`: The method used for optimisation. Could be one of `'bo'` or `'rand'`.
                 Default is `'bo'`. `'bo'`: Bayesian optimisation, `'rand'`: Random search. </li>
<li> `'config'`: Either a configuration file or or parameters returned by
             [`exd.cp_domain_utils.load_config_file`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/cp_domain_utils.py). `config` can be `None` only if `domain`
             is a [`EuclideanDomain`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/domains.py) object. </li>
<li> `'options'`: Additional hyper-parameters for optimisation, as a namespace.
<li> `'reporter'`: A stream to print progress made during optimisation, or one of the
               following strings 'default', 'silent'. If 'silent', then it suppresses
               all outputs. If 'default', writes to stdout. </li>
</ul>
Alternatively, `domain` and `fidelity` space could be `None` if `config` is either a
path_name to a configuration file or has configuration parameters.  

**Returns:**
<ul>
<li> `'opt_val'`: The maximum value found during the optimisation procdure. </li>
<li> `'opt_pt'`: The corresponding optimum point. </li>
<li> `'history'`: A record of the optimisation procedure which include the point evaluated
           and the values at each time step. </li>
</ul>


&nbsp;

## minimise_multifidelity_function
```
maximise_multifidelity_function(func, fidel_space, domain, fidel_to_opt,
                                fidel_cost_func, max_capital,
                                capital_type='return_value',
                                opt_method='bo',
                                config=None,
                                options=None,
                                reporter='default'):
```
**Arguments:**  
Same as `maximise_multifidelity_function` (see above), but now `func` is to be minimised.  
**Returns**:  
Same as `maximise_multifidelity_function` (see above), but now `opt_val` is the minimum value found during optimisation.

&nbsp;

## multiobjective_maximise_function
```python
multiobjective_maximise_functions(funcs, domain, max_capital,
                                  capital_type='num_evals',
                                  opt_method='bo',
                                  config=None,
                                  options=None,
                                  reporter='default')
```
Jointly optimises the functions 'funcs' over the domain 'domain'.  

**Arguments:**
<li> funcs: The functions to be co-optimised (maximised). </li>
<li> `domain`: The domain over which the function should be maximised, should be an
instance  of the Domain class in
[`exd/domains.py`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/domains.py).
If domain is a list of the form `[[l1, u1], [l2, u2], ...]` where `li < ui`, then we
create a Euclidean domain with lower bounds `li` and upper bounds `ui` along each
dimension. </li>
<li> `max_capital`: The maximum capital (time budget or number of evaluations) available for optimisation.   </li>
<li> `capital_type`: The type of capital. Should be one of `'num_evals'`, `'return_value'` or
  `'realtime'`.  Default is `'num_evals'` which indicates the number of evaluations. If
  `'realtime'`, we will use wall clock time.   </li>

<li> `'opt_method'`: The method used for optimisation. Could be one of `'bo'` or `'rand'`.
                 Default is `'bo'`. `'bo'`: Bayesian optimisation, `'rand'`: Random search. </li>
<li> `'config'`: Either a configuration file or or parameters returned by
             [`exd.cp_domain_utils.load_config_file`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/cp_domain_utils.py). `config` can be `None` only if `domain`
             is a [`EuclideanDomain`](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/domains.py) object. </li>
<li> `'options'`: Additional hyper-parameters for optimisation, as a namespace. </li>
<li> `'reporter'`: A stream to print progress made during optimisation, or one of the
               following strings 'default', 'silent'. If 'silent', then it suppresses
               all outputs. If 'default', writes to stdout. </li>
Alternatively, domain could be None if config is either a path_name to a
configuration file or has configuration parameters.

**Returns:**
<ul>
<li> `'pareto_values'`: The pareto optimal values found during the optimisation procdure. </li>
<li> `'pareto_points'`: The corresponding pareto optimum points in the domain. </li>
<li> `'history'`: A record of the optimisation procedure which include the point evaluated
         and the values at each time step. </li>
</ul>



&nbsp;

## multiobjective_minimise_function
```python
multiobjective_minimise_functions(funcs, domain, max_capital,
                                  capital_type='num_evals',
                                  opt_method='bo',
                                  config=None,
                                  options=None,
                                  reporter='default')
```
**Arguments:**  
Same as `multiobjective_maximise_functions` (see above), but now `funcs` are to be minimised.   
**Returns**:  
Same as `multiobjective_maximise_functions` (see above), but now `pareto_values` are the Pareto optimal minimum values found during optimisation. 


&nbsp;

## maximize_function
Alternative spelling for `maximise_function`

## minimize_function
Alternative spelling for `minimise_function`

## maximize_multifidelity_function
Alternative spelling for `maximise_function`

## minimize_multifidelity_function
Alternative spelling for `minimise_multifidelity_function`

## multiobjective_maximize_functions
Alternative spelling for `multiobjective_maximise_functions`

## multiobjective_minimize_functions
Alternative spelling for `multiobjective_minimise_functions`


