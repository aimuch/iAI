# About This Sample
This sample demonstrates how to train a model in PyTorch, recreate the network in TensorRT and import weights from the trained model, and finally run inference with a TensorRT engine. `sample.py` imports functions from `model.py` for training the PyTorch model, as well as retrieving test cases from the PyTorch Data Loader.

# Installing Prerequisites
1. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the top-level of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the top-level of this sample.

# Running the Sample
1. Create a TensorRT inference engine and run inference:
    ```
    python sample.py [-d DATA_DIR]
    ```
    The data directory needs to be specified only if TensorRT is not installed in the default location.
