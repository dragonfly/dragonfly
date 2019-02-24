
A demo for fitting hyper-parameters in Tree based ensemble regression methods such as
gradient boosted regression and random forest classification.


To run this demo, you will need to download the `news_popularity.p` and
`naval_propulsion.p` datasets
into this directory. The datasets are available
[here](http://www.cs.cmu.edu/~kkandasa/dragonfly_datasets.html).
Then, run the following commands from this directory.
```bash
$ dragonfly-script.py --config config_salsa_energy.json --options ../options_files/options_example_realtime.txt
$ dragonfly-script.py --config config_salsa_energy_mf.json --options ../options_files/options_example_realtime.txt # For multi-fidelity version
```

&nbsp;


References
  - `news_popularity.p` Kelwin Fernandes, Pedro Vinagre, and Paulo Cortez. "A Proactive
    Intelligent Decision Support System for Predicting the Popularity of Online News"
    Portuguese Conference on Artificial Intelligence, 2015.
  - `naval_propulsion.p` Andrea Coraddu, Luca Oneto, Aessandro Ghio, Stefano Savio,
    Davide Anguita, and Massimo
    Figari. "Machine Learning Approaches for Improving Condition-based Maintenance of
    Naval Propulsion Plants", Journal of Engineering for the Maritime Environment, 2016.

