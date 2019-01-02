
<img src="https://dragonfly.github.io/images/dragonfly_bigwords.png"/>

---

Dragonfly is an open source python library for scalable Bayesian optimisation.
The library is in alpha version.

Bayesian optimisation is used for optimising black-box functions whose evaluations are
usually expensive. Beyond vanilla optimisation techniques, Dragonfly provides an array of tools to
scale up Bayesian optimisation to expensive large scale problems.
These include features/functionality that are especially suited for
high dimensional optimisation (optimising for a large number of variables),
parallel evaluations in synchronous or asynchronous settings (conducting multiple
evaluations in parallel), multi-fidelity optimisation (using cheap approximations
to speed up the optimisation process), and multi-objective optimisation (optimising
multiple functions simultaneously).

Dragonfly is compatible with python2.7 and python3 and has been tested on Linux,
Mac OS, and Windows platforms.

For questions and bug reports please email kandasamy@cs.cmu.edu


## Installation

* Download the package.
```bash
$ git clone https://github.com/dragonfly/dragonfly.git
```

* In Linux and MacOS, source the set up file in the parent directory, i.e dragonfly.
```bash
$ source set_up
```
In Windows systems, add the parent directory to the `PYTHONPATH` system variable.
[This link](https://superuser.com/questions/949560/how-do-i-set-system-environment-variables-in-windows-10)
describes a few ways to do this.

* Build the direct fortran library. For this `cd` into `utils/direct_fortran` and run
  `bash make_direct.sh`. You will need a fortran compiler such as gnu95. Once this is
  done, you can run `simple_direct_test.py` to make sure that it was installed correctly.
  If unable to build, Dragonfly can still be run but might be slightly slower.

**Requirements:**
Dragonfly requires standard python packages such as `numpy`, `scipy`, and `future` (for
python2/3 compatibility). They can be pip installed via
`$ pip install <package-name>`.

**Testing the Installation**:
To test the installation, run ```bash run_all_tests.sh```. Some of the tests are
probabilistic and could fail at times. If this happens, run the same test several times
and make sure it is not consistently failing. Running all tests will take a while.
You can run each unit test individually simpy via `python unittest_xxx.py`.

## Getting started

Dragonfly can be
used directly in the command line by calling
[`dragonfly.py`](dragonfly.py)
or be imported in python code via the `maximise_function` function in
[`dragonfly.py`](dragonfly.py).
To help get started, we have provided some demos in the `demos_synthetic` directory.

**Via command line**:

To use Dragonfly via the command line, we need to specify the optimisation problem (i.e.
the function to be maximised and the domain) and the optimisation parameters.
We have demonstrated these on the
[`branin`](https://www.sfu.ca/~ssurjano/branin.html) function and a
[`face recognition`](http://scikit-learn.org/0.15/auto_examples/applications/face_recognition.html)
example from [`scikit`](http://scikit-learn.org/0.15/index.html) in the [`demos_synthetic`](demos_synthetic) directory.
The former is a common benchmark for global optimisation while the latter is a
model selection problem in machine learning.
The functions to be maximised in these problems are defined in
[`demos_synthetic/branin/branin.py`](demos_synthetic/branin/branin.py) and
[`demos_real/face_rec/face_rec.py`](demos_real/face_rec/face_rec.py) respectively.
The name of this file and the domain should be specified in
[JSON](https://en.wikipedia.org/wiki/JSON) or
[protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) format.
See
[`demos_synthetic/branin/config.json`](demos_synthetic/branin/config.json) and
[`demos_synthetic/branin/config.pb`](demos_synthetic/branin/config.pb) for examples.
Then, specify the optimisation parameters in an options file, in the format shown in
[`demos_synthetic/options_example.txt`](demos_synthetic/options_example.txt).

We recommend using JSON since we have exhaustively tested JSON format.
If using protocol buffers, you might need to install this package via
`pip install protobuf`.

The branin demo can be run via following commands.
```bash
$ python dragonfly.py --config demos_synthetic/branin/config.json --options demos_synthetic/options_example.txt
$ python dragonfly.py --config demos_synthetic/branin/config.pb --options demos_synthetic/options_example.txt
```

By default, Dragonfly *maximises* functions. To minimise a function, set the
`max_or_min` flag to `min' in the options file. This can be run via,
```bash
$ python dragonfly.py --config demos_synthetic/branin/config.json --options demos_synthetic/options_example_for_minimisation.txt
```

The multi-fidelity version of the branin demo can be run via following command.
```bash
$ python dragonfly.py --config demos_synthetic/branin/config_mf.json --options demos_synthetic/options_example.txt
```

Dragonfly can be run on Euclidean, integral, discrete, and discrete numeric domains, or a
domain which includes a combination of these variables.
See other demos on synthetic functions in the
[`demos_synthetic`](demos_synthetic) directory.
For example, to run the multi-fidelity [`park2_3`](demos_synthetic/park2_3/park2_3_mf.py)
demo, simpy run
```bash
$ python dragonfly.py --config demos_synthetic/park2_3/config_mf.json --options demos_synthetic/options_example.txt
```


The face recognition demo tunes hyper-parameters of a face regognition model.
It can be run via the following commands.
You will need to install
[`scikit-learn`](http://scikit-learn.org), which can be done via
`pip install scikit-learn`.
Running this demo the first time will be slow since the dataset needs to be downloaded.

```bash
$ python dragonfly.py --config demos_real/face_rec/config.json --options demos_real/face_rec/options.txt
$ python dragonfly.py --config demos_real/face_rec/config.pb --options demos_real/face_rec/options.txt
```

**In python code**:

You can import the main API in python code via,
```python
from dragonfly.dragonfly import maximise_function
...
max_val, max_pt, history = maximise_function(func, domain, max_capital)
min_val, min_pt, history = minimise_function(func, domain, max_capital)
```
Here, `func` is the function to be maximised,
`domain` is the domain over which `func` is to be maximised,
and `max_capital` is the capital available for optimisation.
In simple sequential settings, `max_capital` is simply the maximum number of evaluations
to `func`, but it can also be used to specify an available time budget and other forms
of resource constraints. The `maximise_function` returns history along with the optimum value
and the corresponding optimum point. `history.query_points` contains the points queried by the
algorithm and `history.query_vals` contains the function values.

[`demos_synthetic/branin/in_code_demo.py`](demos_synthetic/branin/in_code_demo.py) and
[`demos_real/face_rec/in_code_demo.py`](demos_real/face_rec/in_code_demo.py)
demonstrate the use case for the branin function and face recognition demos respectively.
To execute this file, simply run
```bash
$ python demos_synthetic/branin/in_code_demo.py
$ python demos_real/face_rec/in_code_demo.py
```

**Multiobjective optimisation**

Dragonfly also provides functionality for multi-objective optimisation.
Some synthetic demos are available in the `multiobjective_xxx` directories in
[`demos_synthetic`](demos_synthetic).
For example, to run the
[`hartmann`](demos_synthetic/multiobjective_hartmann/multiobjective_hartmann.py)
demo, simpy run
```bash
$ python dragonfly.py --config demos_synthetic/multiobjective_hartmann/config.json --options demos_synthetic/multiobjective_options_example.txt
```

Similarly, you can import and run this in python code via,
```python
from dragonfly.dragonfly import multiobjective_maximise_functions
...
pareto_max_values, pareto_points, history = multiobjective_maximise_functions(funcs, domain, max_capital)
pareto_min_values, pareto_points, history = multiobjective_minimise_functions(funcs, domain, max_capital)
```
Here, `funcs` is a list of functions to be maximised,
`domain` is the domain  and `max_capital` is the capital available for optimisation.
`pareto_values` are the
[Pareto optimal](https://en.wikipedia.org/wiki/Multi-objective_optimization#Introduction)
function values and `pareto_points` are the corresponding points in `domain`.

### Contributors

Kirthevasan Kandasamy: [github](https://github.com/kirthevasank),
[webpage](http://www.cs.cmu.edu/~kkandasa/)  
Karun Raju Vysyaraju: [github](https://github.com/karunraju),
[linkedin](https://www.linkedin.com/in/karunrajuvysyaraju)  
Willie Neiswanger: [github](https://github.com/willieneis),
[webpage](http://www.cs.cmu.edu/~wdn/)  
Biswajit Paria: [github](https://github.com/biswajitsc),
[webpage](https://biswajitsc.github.io/)

### Citation
If you use any part of this code in your work, please cite
[this manuscript](http://www.cs.cmu.edu/~kkandasa/docs/proposal.pdf).

### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/dragonfly/dragonfly/blob/master/LICENSE.txt).

"Copyright 2018 Kirthevasan Kandasamy"

- For questions and bug reports please email kandasamy@cs.cmu.edu
