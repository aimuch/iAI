# About This Sample
This sample demonstrates how to create an int8 calibrator, build and calibrate an engine for int8 mode,
and finally run inference in int8 mode.

During calibration, the calibrator retrieves a total of 60000 images, which corresponds to the entirety of
the MNIST training set. We have simplified the process of reading and writing a calibration cache in Python,
so that it is now possible to easily cache calibration data to speed up engine builds (see `calibrator.py`
for implementation details).

During inference, the sample loads a random batch from the calibrator, then performs inference on the
whole batch of 100 images.

# Installing Prerequisites
1. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the top-level of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the top-level of this sample.
2. Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
    - This sample requires the [training set](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz), [test set](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) and [test labels](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz).
    - Unzip the files obtained above using the `gunzip` utility. For example, `gunzip t10k-labels-idx1-ubyte.gz`.

# Running the Sample
1. Create a TensorRT inference engine, perform int8 calibration and run inference:
    ```
    python sample.py [-d DATA_DIR] [-d MNIST_DATA_DIR]
    ```
    `DATA_DIR` needs to be specified only if TensorRT is not installed in the default location.
    `MNIST_DATA_DIR` should point to the location where you extracted the MNIST data, i.e. the training set, test set and test labels.
