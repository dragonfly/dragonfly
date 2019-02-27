## Neural Architecture Search Demo with NASBOT

This Demo has been tested with Tensorflow 1.5.


# Datasets

For the MLP experiments, the data should be in a pickle file stored as a dictionary.
The 'train' key should point to the training data while 'vali' points to the validation
data. For example, after data = pic.load(file_name), data['train']['x'] should point
to the features of the training data.
The slice and indoor_location datasets are available at
http://www.cs.cmu.edu/~kkandasa/dragonfly_datasets.html as examples.
Download them into this directory to run the demo.

For the CNN experiment,
we use the Cifar10 dataset which is converted to .tfrecords format for tensorflow.
You can either download the original dataset from www.cs.toronto.edu/~kriz/cifar.html
and follow the instructions in

https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.p
Alternatively, they are available in the required format at
www.cs.cmu.edu/~kkandasa/nasbot_datasets.html as examples.
Put the xxx.tfrecords in a directory named cifar-10-data in the demos directory to run
this demo.

