Main APIs
=========

# maximise_function
```python
opt_val, opt_pt, history = maximise_function(func, domain, max_capital, capital_type='num_evals', opt_method='bo', config=None, options=None, reporter='default')
```
Arguments:
*  func: The function to be maximised.
*  domain: The domain over which the function should be maximised, should be an
           instance of the Domain class in exd/domains.py.
           If domain is a list of the form [[l1, u1], [l2, u2], ...] where li < ui,
           then we will create a Euclidean domain with lower bounds li and upper bounds
           ui along each dimension.
*  max_capital: The maximum capital (time budget or number of evaluations) available
                for optimisation.
*  capital_type: The type of capital. Should be one of 'return_value' or 'realtime'.
                 Default is return_value which indicates we will use the value returned
                 by fidel_cost_func. If realtime, we will use wall clock time.
*  opt_method: The method used for optimisation. Could be one of bo, rand, ga, ea,
               direct, or pdoo. Default is bo.
               bo - Bayesian optimisation, ea/ga: Evolutionary algorithm,
               rand - Random search, direct: Dividing Rectangles, pdoo: PDOO
*  config: Contains configuration parameters that are typically returned by
           exd.cp_domain_utils.load_config_file. config can be None only if domain
           is a EuclideanDomain object.
*  options: Additional hyper-parameters for optimisation, as a namespace.
*  reporter: A stream to print progress made during optimisation, or one of the
            following strings 'default', 'silent'. If 'silent', then it suppresses
            all outputs. If 'default', writes to stdout.
Returns:
*   opt_val: The maximum value found during the optimisatio procdure.
*   opt_pt: The corresponding optimum point.
*   history: A record of the optimisation procedure which include the point evaluated
             and the values at each time step.


