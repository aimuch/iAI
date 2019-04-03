#!/usr/bin/env sh
DATA=/home/andy/caffe/examples/mydata/slot_classifier/data/

EXAMPLE=/home/andy/caffe/examples/mydata/slot_classifier/data
DATA=/home/andy/caffe/examples/mydata/slot_classifier/data
TOOLS=/home/andy/caffe/build/tools

rm -rf $DATA/mydata_mean.binaryproto

$TOOLS/compute_image_mean $EXAMPLE/mydata_train_lmdb \
  $DATA/mydata_mean.binaryproto

echo "Done."
