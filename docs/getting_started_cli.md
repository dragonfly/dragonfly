<span style="font-size:3em">**Getting started in Command Line**</span>

&nbsp;


Dragonfly can be
used directly in the command line by calling
[`dragonfly-script.py`](bin/dragonfly-script.py)
or be imported in python code via the `maximise_function` function in the main library.
To help get started, we have provided some examples in the
[`examples`](examples) directory.
*If you have pip installed, you will need to clone the repository to access the demos.*

To use Dragonfly via the command line, we need to specify the optimisation problem (i.e.
the function to be maximised and the domain) and the optimisation parameters.
The optimisation problem can be specified in
[JSON](https://en.wikipedia.org/wiki/JSON) (recommended) or
[protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) format.
See
[`examples/synthetic/branin/config.json`](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/branin/config.json) and
[`examples/synthetic/branin/config.pb`](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/branin/config.pb) for examples.
Then, specify the optimisation parameters in an options file, in the format shown in
[`examples/options_example.txt`](https://github.com/dragonfly/dragonfly/tree/master/examples/options_example.txt).
We have demonstrated these via some examples below.

We recommend using JSON since we have exhaustively tested JSON format.
If using protocol buffers, you might need to install this package via
`pip install protobuf`.

&nbsp;

**Basic use case**:
The first example is on the
[Branin](https://www.sfu.ca/~ssurjano/branin.html) benchmark for global optimisation,
which is defined in
[`examples/synthetic/branin/branin.py`](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/branin/branin.py).
The branin demo can be run via following commands.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config.json --options options_files/options_example.txt
$ dragonfly-script.py --config synthetic/branin/config.pb --options options_files/options_example.txt
```

&nbsp;

**Minimisation**:
By default, Dragonfly *maximises* functions. To minimise a function, set the
`max_or_min` flag to `min` in the options file as shown in
[`examples/options_example_for_minimisation.txt`](https://github.com/dragonfly/dragonfly/tree/master/examples/options_files/options_example_for_minimisation.txt)
For example,
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config.json --options
options_files/options_example_for_minimisation.txt
```

&nbsp;

**Multi-fidelity Optimisation**:
The multi-fidelity version of the branin demo can be run via following command.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config_mf.json --options options_files/options_example.txt
```

&nbsp;

**Specifying the Domain**:
Dragonfly can be run on Euclidean, integral, discrete, and discrete numeric domains, or a
domain which includes a combination of these variables.
See other demos on synthetic functions in the
[`examples/synthetic`](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic) directory.
For example, to run the multi-fidelity
[`park2_3`](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/park2_3/park2_3_mf.py)
demo, simply run
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/park2_3/config_mf.json --options options_files/options_example.txt
```

&nbsp;

**Optimisation on a time budget**:
Dragonfly also allows specifying a time budget for optimisation.
The next demo is a maximum likelihood problem in computational astrophysics,
where we wish to estimate cosmological parameters from Type Ia supernova data.
The demos for Bayesian optimisation and multi-fidelity Bayesian optimisation
can be run via the following commands.
This uses a time budget of 2 hours.

```bash
$ cd examples
$ dragonfly-script.py --config supernova/config.json --options options_files/options_example_realtime.txt
$ dragonfly-script.py --config supernova/config_mf.json --options options_files/options_example_realtime.txt    # For multi-fidelity version
```


&nbsp;


*Other methods*:
BO is ideally suited for expensive function evaluations - it aims to find the optimum
in as few evaluations and invests significant computation to do so.
This pays dividends if the evaluations are expensive.
However,
if your function evaluations are cheap, we recommended using DiRect or PDOO for
Euclidean domains, and evolutionary algorithms for non-Euclidean domains.
See example below.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config.json --options options_files/options_example_pdoo.txt
$ dragonfly-script.py --config synthetic/park2_3/config_mf.json --options options_files/options_example_ea.txt
```
You will notice that they run significantly faster than BO.
However, these methods will perform worse than BO on the supernova problem as evaluations
are expensive.



&nbsp;

**Multiobjective optimisation**

Dragonfly also provides functionality for multi-objective optimisation.
Some synthetic demos are available in the `multiobjective_xxx` directories in
[`demos_synthetic`](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic).
For example, to run the
[multi-objective hartmann](https://github.com/dragonfly/dragonfly/tree/master/examples/synthetic/multiobjective_hartmann/multiobjective_hartmann.py)
demo, simpy run
```bash
$ dragonfly-script.py --config demos_synthetic/multiobjective_hartmann/config.json --options demos_synthetic/multiobjective_options_example.txt
```

&nbsp;

