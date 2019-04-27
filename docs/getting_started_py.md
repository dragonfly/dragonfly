<span style="font-size:3em">**Getting started in Python Code**</span>

&nbsp;

Dragonfly can be
imported in python code via the `maximise_function` or `minimise_function` functions in
the main library.


The main APIs for Dragonfly are declared in
[`dragonfly/__init__.py`](https://github.com/dragonfly/dragonfly/tree/master/dragonfly/__init__.py) and defined in the
[`dragonfly/apis`](https://github.com/dragonfly/dragonfly/tree/master/dragonfly/apis) directory.
For their definitions and arguments, see
[`dragonfly/apis/opt.py`](https://github.com/dragonfly/dragonfly/tree/master/dragonfly/apis/opt.py) and
[`dragonfly/apis/moo.py`](https://github.com/dragonfly/dragonfly/tree/master/dragonfly/apis/moo.py).


You can import the main API in python code via,
```python
from dragonfly import minimise_function, maximise_function
func = lambda x: x ** 4 - x**2 + 0.1 * x
domain = [[-10, 10]]
max_capital = 100
min_val, min_pt, history = minimise_function(func, domain, max_capital)
print(min_val, min_pt)
max_val, max_pt, history = maximise_function(lambda x: -func(x), domain, max_capital)
print(max_val, max_pt)
```
Here, `func` is the function to be maximised,
`domain` is the domain over which `func` is to be optimised,
and `max_capital` is the capital available for optimisation.
In simple sequential settings, `max_capital` is simply the maximum number of evaluations
to `func`, but it can also be used to specify an available time budget and other forms
of resource constraints. The `maximise_function` returns history along with the optimum
value
and the corresponding optimum point. `history.query_points` contains the points queried by
the
algorithm and `history.query_vals` contains the function values.


The domain can be specified via a JSON file or in code.
See 
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/branin/in_code_demo.py),
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/hartmann6_4/in_code_demo.py),
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/discrete_euc/in_code_demo_1.py),
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/discrete_euc/in_code_demo_2.py),
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/hartmann3_constrained/in_code_demo.py),
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/park1_constrained/in_code_demo.py),
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/borehole_constrained/in_code_demo.py),
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/tree_reg/in_code_demo.py),
and
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/nas/demo_nas.py)
for more detailed examples.
You can run them via, for example,
```bash
$ python examples/synthetic/branin/in_code_demo.py
$ python examples/tree_reg/in_code_demo.py
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
See
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/multiobjective_branin_currinexp/in_code_demo.py)
and
[here](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/multiobjective_hartmann/in_code_demo.py)
for examples.


&nbsp;


**Neural Architecture Search**

Dragonfly has APIs for defining neural network architectures, defining distances
between them, and also implements the
[NASBOT](https://arxiv.org/pdf/1802.07191.pdf) algorithm,
which uses Bayesian optimisation and optimal transport for neural architecture search.
You will first need to install the following dependencies.
Cython will also require
a C++ library already installed in the system.
```bash
$ pip install cython POT
```

Below is an architecture search demo on a synthetic function.
See the [`examples/nas`](https://github.com/dragonfly/dragonfly/tree/master/examples/nas) directory for demos of NASBOT in using some large
datasets.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/syn_cnn_1/config.json --options options_files/options_example.txt
```

