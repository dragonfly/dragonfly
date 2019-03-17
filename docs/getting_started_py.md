# Getting Started in Python Code

Dragonfly can be
imported in python code via the `maximise_function` or `minimise_function` functions in
the main library.

You can import the main API in python code via,
```python
from dragonfly import minimise_function, maximise_function
func = lambda x: lambda x: x ** 4 - x**2 + 0.1 * x
domain = [[-10, 10]]
max_capital = 100
min_val, min_pt, history = minimise_function(func, domain, max_capital)
...
print(min_val, min_pt)
min_val, min_pt, history = maximise_function(lambda x: -func(x), domain, max_capital)
```
Here, `func` is the function to be maximised,
`domain` is the domain over which `func` is to be optimised,
and `max_capital` is the capital available for optimisation.
In simple sequential settings, `max_capital` is simply the maximum number of evaluations
to `func`, but it can also be used to specify an available time budget and other forms
of resource constraints. The `maximise_function` returns history along with the optimum value
and the corresponding optimum point. `history.query_points` contains the points queried by the
algorithm and `history.query_vals` contains the function values.


[`examples/synthetic/branin/in_code_demo.py`](examples/synthetic/branin/in_code_demo.py)
and
[`examples/supernova/in_code_demo.py`](examples/supernova/in_code_demo.py)
demonstrate the use case for the branin and supernova problems respectively.
To execute these files, simply run
```bash
$ python examples/synthetic/branin/in_code_demo.py
$ python examples/supernova/in_code_demo.py
```


&nbsp;

**Multiobjective optimisation**


Similarly, you can import the multi-objective optimisation APIs in python code via,
```python
from dragonfly import multiobjective_maximise_functions
...
pareto_max_values, pareto_points, history = multiobjective_maximise_functions(funcs, domain, max_capital)
pareto_min_values, pareto_points, history = multiobjective_minimise_functions(funcs, domain, max_capital)
```
Here, `funcs` is a list of functions to be maximised,
`domain` is the domain  and `max_capital` is the capital available for optimisation.
`pareto_values` are the
[Pareto optimal](https://en.wikipedia.org/wiki/Multi-objective_optimization#Introduction)
function values and `pareto_points` are the corresponding points in `domain`.

