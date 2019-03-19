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
**Arguments:**  
- `func`: The function to be maximised.  
- `domain`: The domain over which the function should be maximised, should be an instance of the Domain class in exd/domains.py.  If domain is a list of the form `[[l1, u1], [l2, u2], ...]` where `li < ui`, then we create a Euclidean domain with lower bounds `li` and upper bounds `ui` along each dimension.  
- `max_capital`: The maximum capital (time budget or number of evaluations) available for optimisation.  
- `capital_type`: The type of capital. Should be one of `'return_value'` or `'realtime'`.  Default is return_value which indicates we will use the value returned by fidel_cost_func. If realtime, we will use wall clock time.  
- `opt_method`: The method used for optimisation. Could be one of bo, rand, ga, ea, direct, or pdoo. Default is bo.  bo: Bayesian optimisation, ea/ga: Evolutionary algorithm, rand: Random search, direct: Dividing Rectangles, pdoo: PDOO. 
- `config`: Contains configuration parameters that are typically returned by exd.cp_domain_utils.load_config_file. config can be None only if domain is a EuclideanDomain object.  
- `options`: Additional hyper-parameters for optimisation, as a namespace.
- `reporter`: A stream to print progress made during optimisation, or one of the following strings 'default', `'silent'`. If `'silent'`, then it suppresses all outputs. If `'default'`, writes to stdout.  
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
maximise_multifidelity_function(func, fidel_space, domain, fidel_to_opt, fidel_cost_func, max_capital,
                                capital_type='return_value',
                                opt_method='bo',
                                config=None,
                                options=None,
                                reporter='default'):
```

&nbsp;

## minimise_multifidelity_function
```
maximise_multifidelity_function(func, fidel_space, domain, fidel_to_opt, fidel_cost_func, max_capital,
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

Test list
- item 1
- item 2
* item 3
* item 4
+ item 5
+ item 6


Test list 2
1 item 1
2 item 2
3 item 3



Test list 3
<ul>
<li> item 1 </li> 
<li> item 2 </li>
<li> item 3 </li>
</ul>

