This is a simple Multi-Layer Perceptron (MLP) example showing how to generate a MLP that TensorRT can accelerate.
This sample requires Tensorflow > 1.4 to be installed.
This MLP was trained via the following method:
git clone https://github.com/aymericdamien/TensorFlow-Examples.git
cd TensorFlow-Examples

Apply the patch file, `update_mlp.patch` to save the final result with the command `patch -p1 < <TensorRT Install>/samples/sampleMLP/update_mlp.patch`
Train the MINST MLP with the command `python examples/3_NeuralNetworks/multilayer_perceptron.py`
Convert the trained model weights to a format sampleMLP understands via the command `python <TensorRT Install>/samples/sampleMLP/convert_weights.py -m /tmp/sampleMLP.ckpt -o sampleMLP`

mkdir -p <TensorRT Install>/data/mlp
cp sampleMLP.wts2 <TensorRT Install>/data/mlp/

To build the sample:
cd <TensorRT Install>/samples
make

To run the sample:
cd <TensorRT Install>/bin
./sample_mlp
