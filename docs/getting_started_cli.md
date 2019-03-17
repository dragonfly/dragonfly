# Getting started in Command Line

Dragonfly can be
used directly in the command line by calling
[`dragonfly-script.py`](../bin/dragonfly-script.py).
To help get started, we have provided some demos in the `demos_synthetic` directory.
If you have pip installed, you will need to clone the repository to access the demos.

To use Dragonfly, we need to specify the optimisation problem (i.e.
the function to be maximised and the domain) and the optimisation parameters.
We have demonstrated these on the
[`branin`](https://www.sfu.ca/~ssurjano/branin.html) function and a
[`face recognition`](http://scikit-learn.org/0.15/auto_examples/applications/face_recognition.html)
example from [`scikit`](http://scikit-learn.org/0.15/index.html) in the [`demos_synthetic`](../demos_synthetic) directory.
The former is a common benchmark for global optimisation while the latter is a
model selection problem in machine learning.
The functions to be maximised in these problems are defined in
[`demos_synthetic/branin/branin.py`](../demos_synthetic/branin/branin.py) and
[`demos_real/face_rec/face_rec.py`](../demos_real/face_rec/face_rec.py) respectively.
The name of this file and the domain should be specified in
[JSON](https://en.wikipedia.org/wiki/JSON) (recommended) or
[protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) format.
See
[`demos_synthetic/branin/config.json`](../demos_synthetic/branin/config.json) and
[`demos_synthetic/branin/config.pb`](../demos_synthetic/branin/config.pb) for examples.
Then, specify the optimisation parameters in an options file, in the format shown in
[`demos_synthetic/options_example.txt`](../demos_synthetic/options_example.txt).

We recommend using JSON since we have exhaustively tested JSON format.
If using protocol buffers, you might need to install this package via
`pip install protobuf`.

The branin demo can be run via following commands.
```bash
$ dragonfly-script.py --config demos_synthetic/branin/config.json --options demos_synthetic/options_example.txt
$ dragonfly-script.py --config demos_synthetic/branin/config.pb --options demos_synthetic/options_example.txt
```
By default, Dragonfly *maximises* functions. To minimise a function, set the
`max_or_min` flag to `min` in the options file as shown in
[`demos_synthetic/options_example_for_minimisation.txt`](../demos_synthetic/options_example_for_minimisation.txt)
For example,
```bash
$ dragonfly-script.py --config demos_synthetic/branin/config.json --options demos_synthetic/options_example_for_minimisation.txt
```


The multi-fidelity version of the branin demo can be run via following command.
```bash
$ dragonfly-script.py --config demos_synthetic/branin/config_mf.json --options demos_synthetic/options_example.txt
```

&nbsp;

Dragonfly can be run on Euclidean, integral, discrete, and discrete numeric domains, or a
domain which includes a combination of these variables.
See other demos on synthetic functions in the
[`demos_synthetic`](../demos_synthetic) directory.
For example, to run the multi-fidelity [`park2_3`](../demos_synthetic/park2_3/park2_3_mf.py)
demo, simpy run
```bash
$ dragonfly-script.py --config demos_synthetic/park2_3/config_mf.json --options demos_synthetic/options_example.txt
```


The face recognition demo tunes hyper-parameters of a face regognition model.
It can be run via the following commands.
You will need to install
[`scikit-learn`](http://scikit-learn.org), which can be done via
`pip install scikit-learn`.
Running this demo the first time will be slow since the dataset needs to be downloaded.

```bash
$ dragonfly-script.py --config demos_real/face_rec/config.json --options demos_real/face_rec/options.txt
$ dragonfly-script.py --config demos_real/face_rec/config.pb --options demos_real/face_rec/options.txt
```

&nbsp;

**Multiobjective optimisation**

Dragonfly also provides functionality for multi-objective optimisation.
Some synthetic demos are available in the `multiobjective_xxx` directories in
[`demos_synthetic`](../demos_synthetic).
For example, to run the
[`hartmann`](../demos_synthetic/multiobjective_hartmann/multiobjective_hartmann.py)
demo, simpy run
```bash
$ dragonfly-script.py --config demos_synthetic/multiobjective_hartmann/config.json --options demos_synthetic/multiobjective_options_example.txt
```

