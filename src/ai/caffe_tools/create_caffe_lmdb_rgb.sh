#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/home/andy/caffe/examples/mydata/slot_classifier/data
DATA=/home/andy/caffe/examples/mydata/slot_classifier/data
TOOLS=/home/andy/caffe/build/tools

TRAIN_DATA_ROOT=/home/andy/caffe/examples/mydata/slot_classifier/data/train/
VAL_DATA_ROOT=/home/andy/caffe/examples/mydata/slot_classifier/data/val/
TEST_DATA_ROOT=/home/andy/caffe/examples/mydata/slot_classifier/data/test/

RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo ">>>>>> Creating train lmdb... >>>>>>"
rm -rf $EXAMPLE/mydata_train_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/mydata_train_lmdb

echo "\n"

echo ">>>>>> Creating val lmdb... >>>>>>"
rm -rf $EXAMPLE/mydata_val_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/mydata_val_lmdb

echo "\n"

echo ">>>>>> Creating test lmdb... >>>>>>"
rm -rf $EXAMPLE/mydata_test_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TEST_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/mydata_test_lmdb

echo "\n"

echo "All done!"
