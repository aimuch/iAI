# About this sample
Object detection is one of the classic computer vision problems. The task, for
given image, is to detect, classify and localize all objects of interest. For
example, imagine that you are developing a self-driving car and you need to
do pedestrian detection - the object detection algorithm would then, for given
image, return bounding box coordinates for each pedestrian in an image.

There have been many advances in recent years in designing models
for object detection. In this sample, we use one of the most popular model
architectures for object detection - SSD.

SSD architecture can be split into two parts: convolutional feature extractor
and detection part. Detection part is usually fixed, but you are free to pick
feature extractor of your choice: like VGG, ResNet, Inception. In this sample
we use Inception_v2 as feature extractor.

When picking an object detection model for our application the usual tradeoff
is between model accuracy and inference time. In this sample we show how
inference time of pretrained network can be greatly improved, without any
decrease in accuracy, using TensorRT. In order to do that, we take a pre-trained
Tensorflow model, and use TensorRT’s UffParser to build a TensorRT inference engine.


# Preparation
There are few things that have to be done before running the object detection script:
  * Install cmake >= 3.8
  * You need to install all Python dependencies - all the libraries our script
  depends on are listed in requirements.txt. If you have pip installed, you
  should be able to install all dependencies with a single command:
  `pip install -r requirements.txt`
  * You need to compile FlattenConcat custom plugin. To do that, enter sample
  directory, and run the following commands:
  ```sh
  mkdir -p build
  cd build
  cmake ..
  make
  cd ..
  ```
  This should use cmake to build FlattenConcat plugin and put it in the
  appropriate directory. This is needed because the frozen model that is
  used in this sample uses some Tensorflow operations that are not
  natively supported in TensorRT.

##### Optional:
This additional step needs to be done if you also want to run VOC evaluation:
  * You need to download VOC dataset. To download dataset, run the following
  commands from sample root directory:
  `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar`
  `tar xvf VOCtest_06-Nov-2007.tar`
  The first command downloads VOC dataset from Oxford servers, and the second
  one unpacks this dataset.
  If you won't save VOC in sample root directory, you'll need to adjust
  `--voc_dir` argument to `voc_evaluation.py` script before running it.
  (The default value of this argument is `<SAMPLE_ROOT>/VOCdevkit/VOC2007)`).


# Running inference script
Before running inference script, make sure you have followed appropriate steps
from Preparation section.

To launch the inference script, run:
`python detect_objects.py <IMAGE_PATH>`
where <IMAGE_PATH> should contain the image you want to run inference on using
the SSD network. The script should work for all the most popular image formats,
like PNG, JPEG, and BMP. Since the model is trained for images of size 300x300,
the input image will be resized to this size (using bilinear interpolation),
if needed.

When the inference script is launched for the first time, it will run the
following things to prepare its workspace:
  * Download pretrained ssd_inception_v2_coco_2017_11_17 model from Tensorflow
  object detection API. The script converts this model to TensorRT format,
  and the conversion is tailored to this specific version of the model.
  * Build a TensorRT inference engine and save it to file. During this step,
  all TensorRT optimizations will be applied to frozen graph. This is
  a time consuming operation and it can take a few minutes.

After the workspace is ready, the script should launch inference on input image,
and save the results to a location that will be printed on standard output.


# Running VOC evaluation script
Before running VOC evaluation script, make sure you have completed
last step from "Preparation".

To run VOC evaluation using TensorRT run:
`python voc_evaluation.py`

To run VOC evaluation using Tensorflow run:
`python voc_evaluation.py tensorflow`
Be aware that this is much slower than TensorRT evaluation.

At the end of the script execution, AP and mAP metrics will be
displayed.


# Advanced features
Both scripts support separate advanced features (like lower precision
inference, changing workspace directory, changing batch size). To
view all features available, do `python <SCRIPT_NAME>.py -h`.
