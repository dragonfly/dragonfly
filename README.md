
<img src="https://dragonfly.github.io/images/dragonfly_bigwords.png"/>
-----------------
Dragonfly is an open source python library for scalable Bayesian optimisation.
The library is in alpha version.

Bayesian optimisation is used for optimising black-box functions whose evaluations are
usually expensive. Beyond vanilla optimisation techniques, Dragonfly provides an array of tools to
scale up Bayesian optimisation to expensive large scale problems.
These include features/functionality that are especially suited for,
high dimensional optimisation (optimising for a large number of variables),
parallel evaluations in synchronous or asynchronous settings (conducting multiple
evaluations in parallel), and multi-fidelity optimisation (using cheap approximations
to speed up the optimisation process).

By default, Dragonfly *maximises* functions.
To minimise a function, simply pass the negative of the function.
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
[This
link](https://superuser.com/questions/949560/how-do-i-set-system-environment-variables-in-windows-10)
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
[`maximise_function.py`](maximise_function.py).
To help get started, we have provided some demos in the `demos` directory.

**Via command line**:

To use Dragonfly via the command line, we need to specify the optimisation problem (i.e.
the function to be maximised and the domain) and the optimisation parameters.
We have demonstrated these on the
[`branin`](https://www.sfu.ca/~ssurjano/branin.html) function and a
[`face recognition`](http://scikit-learn.org/0.15/auto_examples/applications/face_recognition.html)
example from [`scikit`](http://scikit-learn.org/0.15/index.html) in the [`demos`](demos) directory.
The former is a common benchmark for global optimisation while the latter is a
model selection problem in machine learning.
The functions to be maximised in these problems are defined in
[`demos/branin/branin.py`](demos/branin/branin.py) and
[`demos/face_rec/face_rec.py`](demos/face_rec/face_rec.py) respectively.
The name of this file and the domain should be specified in
[JSON](https://en.wikipedia.org/wiki/JSON) or
[protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) format.
See
[`demos/branin/config.json`](demos/branin/config.json) and
[`demos/branin/config.pb`](demos/branin/config.pb) for examples.
Then, specify the optimisation parameters in an options file, in the format shown in
[`demos/branin/options.txt`](demos/branin/options.txt).

If using protocol buffers, you might need to install this package via
`pip install protobuf`.

The branin demo can be run via following commands.
```bash
$ python dragonfly.py --config demos/branin/config.json --options demos/branin/options.txt
$ python dragonfly.py --config demos/branin/config.pb --options demos/branin/options.txt
```

The face recognition demo can be run via following commands.
You will need to install 
[`scikit-learn`](http://scikit-learn.org), which can be done via
`pip install scikit-learn`.
Running this demo the first time will be slow since the dataset needs to be downloaded.

```bash
$ python dragonfly.py --config demos/face_rec/config.json --options demos/face_rec/options.txt
$ python dragonfly.py --config demos/face_rec/config.pb --options demos/face_rec/options.txt
```

**In python code**:

You can import the main API in python code via,
```python
from dragonfly.maximise_function import maximise_function
...
max_val, max_pt = maximise_function(func, domain, max_capital)
```
Here, `func` is the function to be maximised,
`domain` is the domain over which `func` is to be maximised,
and `max_capital` is the capital available for optimisation.
In simple sequential settings, `max_capital` is simply the maximum number of evaluations
to `func`, but it can also be used to specify an available time budget and other forms
of resource constraints.

[`demos/branin/in_code_demo.py`](demos/branin/in_code_demo.py) and
[`demos/face_rec/in_code_demo.py`](demos/face_rec/in_code_demo.py)
demonstrate the use case for the branin function and face recognition demos respectively.
To execute this file, simply run
```bash
$ python demos/branin/in_code_demo.py
$ python demos/face_rec/in_code_demo.py
```

### Contributors

Kirthevasan Kandasamy: [github](https://github.com/kirthevasank),
[webpage](http://www.cs.cmu.edu/~kkandasa/)  
Karun Raju Vysyaraju: [github](https://github.com/karunraju),
[linkedin](https://www.linkedin.com/in/karunrajuvysyaraju)  
Willie Neiswanger: [github](https://github.com/willieneis),
[webpage](http://www.cs.cmu.edu/~wdn/)

### Citation
If you use any part of this code in your work, please cite
[this manuscript](http://www.cs.cmu.edu/~kkandasa/docs/proposal.pdf).

### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/dragonfly/dragonfly/blob/master/LICENSE.txt).

"Copyright 2018 Kirthevasan Kandasamy"

- For questions and bug reports please email kandasamy@cs.cmu.edu
