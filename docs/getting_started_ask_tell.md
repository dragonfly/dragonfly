<span style="font-size:3em">**Getting started in Ask-Tell Mode (Python)**</span>

&nbsp;

The ask-tell interface in Dragonfly enables step-by-step optimization by directly exposing
the next point to be evaluated in an iteration of Bayesian optimization. 

Two main components are required in this interface: a function caller and an optimizer.

There are two different types of function callers: `EuclideanFunctionCaller` and 
`CPFunctionCaller`. If the domain is limited to Euclidean spaces, use a 
`EuclideanFunctionCaller`, otherwise use a `CPFunctionCaller`. For
Cartesian product spaces, you may also need to define orderings for the domain, which
can be passed in with the `domain_orderings` argument in `CPFunctionCaller`. It is important
to note that no objective function is passed in to the function caller. See the example
below for more details.

There are three different optimizers, similar to how the main API allows specification of
the optimization method using the `opt_method` argument. In this interface, the optimizer
is explicitly created via `<domain>GPBandit`, `<domain>GAOptimiser`, or 
`<domain>RandomOptimiser`, where `<domain>` is replaced by `Euclidean` or `CP` depending on
the domain used. This domain should be consistent with the function caller created. Here,
`ask_tell_mode` should be set to `True` to activate the ask-tell interface.

Finally, call `initialise()` on the created optimizer to begin using the interface.


You can import the necessary components in python code via,
```python
from dragonfly.exd import domains
from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from dragonfly.opt import random_optimiser, cp_ga_optimiser, gp_bandit

max_capital = 100
objective = lambda x: x[0] ** 4 - x[0]**2 + 0.1 * x[0]
domain = domains.EuclideanDomain([[-10, 10]])
func_caller = EuclideanFunctionCaller(None, domain)
opt = gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)
opt.initialise()

for i in range(max_capital):
    x = opt.ask()
    y = objective(x)
    print("x:", x, ", y:", y)
    opt.tell([(x, y)])

```
Here, `objective` is the function to be maximised,
`domain` is the domain over which `objective` is to be optimised,
and `max_capital` is the capital available for optimisation.
In this interface, `max_capital` is simply the maximum number of evaluations
to `objective`.

To minimise the function, simply take the negative of the objective and
perform the same procedure above.

`ask` returns the next point to be evaluated in a numpy array. Once the point
is evaluated by calling the `objective` function, `tell` the point back to the
optimiser. Note that `tell` takes one argument, where `x` and `y` are organized
into a tuple wrapped by a list. 

As with the main API, the domain can be specified via a JSON file or in code.
See the [following example](examples/detailed_use_cases/in_code_demo_ask_tell.py) for more details.
You can run it via, for example,
```bash
$ python examples/detailed_use_cases/in_code_demo_ask_tell.py
```


&nbsp;

**Multifidelity optimisation**

The flow for multifidelity optimisation is very similar. In addition to the required components
mentioned above, a fidelity space and a fidelity value to optimise on is required as well. When 
the objective is evaluated on both the fidelity coordinate `z` and the domain coordinate `x`, 
the fidelity argument must also be specified by calling `opt.tell([z, x, y])`. 

