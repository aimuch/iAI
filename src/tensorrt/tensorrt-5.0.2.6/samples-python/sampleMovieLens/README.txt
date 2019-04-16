This is a multilayer perceptron(MLP) based Neural Collaborative Filter Recommender example showing how to generate weights for MovieLens dataset for TensorRT that TensorRT can accelerate.
This sample requires Tensorflow <= 1.7.0 to be installed.
This MLP base NCF was trained via the following method:

Building the sample:
To build the sample:
cd <TensorRT Install>/samples
make -j12

To run the sample:

1. Running Inference:

cd <TensorRT Install>/bin
./sample_movielens  (default batch=32 i.e. num of users)
./sample_movielens -b <N> (batch=N i.e. num of users)
./sample_movielens --verbose (prints inputs, groundtruth values, expected vs predicted probabilities)

2. Help/Usage
    ./sample_movielens -h
    Usage:
        ./sample_movielens[-h]
        -h        Display help information. All single dash optoins enable perf mode.
        -b        Number of Users i.e. BatchSize (default BatchSize=32).
        --useDLA  Specify a DLA engine for layers that support DLA. Value can range from 1 to N, where N is the number of DLA engines on the platform.
        --verbose Enable verbose perf mode.

Training model from scratch:
Step 1:
    git clone https://github.com/hexiangnan/neural_collaborative_filtering.git
    cd neural_collaborative_filtering
    git checkout 0cd2681598507f1cc26d110083327069963f4433

Step 2:
    Apply the patch file, `sampleMovieLensTraining.patch` to save dump the frozen protobuf file with command `patch -l -p1 < <TensorRT Install>/samples/sampleMovieLens/sampleMovieLensTraining.patch`
    Train the MLP based NCF with the command `python MLP.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0.01,0.01,0.01,0.01] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1'
    WARNING: Using 0s for reg_layers will cause undefined behavior when training the network.
    This step will dump two files:
        1. movielens_ratings.txt
        2. sampleMovieLens.pb

Step 3: Convert the Frozen .pb file to .uff format using
    Command: `python3 convert_to_uff.py sampleMovieLens.pb -p preprocess.py`
    preprocess.py is a preprocessing step that needs to be applied to the TensorFlow graph before it can be used by TensorRT.
    The reason for this is that TensorFlow's concatenation operation accounts for the batch dimension while TensorRT's concatenation operation does not.

    Note: convert_to_uff.py utility will get installed here: /usr/local/bin/convert-to-uff.
        This utility gets installed with UFF .whl file installation shipped with TensorRT.
        For installation instructions, see:
        https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/#python and click on the 'TensoRT Python API' link

Step 4:
    Copy sampleMovieLens.uff file to <TensorRT Install>/data/movielens
    Copy movielens_ratings.txt file to <TensorRT Install>/data/movielens

Step 5:
    Follow instruction above to build and run the sample
