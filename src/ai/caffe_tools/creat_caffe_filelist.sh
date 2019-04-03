 #!/usr/bin/env sh
DATA=/home/andy/caffe/examples/mydata/slot_classifier/data

echo "Create train.txt..."
rm -rf $DATA/train.txt

find $DATA/train/error -name *.png | cut -d '/' -f 10- | sed "s/$/ 0/" >> $DATA/train.txt
find $DATA/train/half  -name *.png | cut -d '/' -f 10- | sed "s/$/ 1/" >> $DATA/train.txt
find $DATA/train/invlb -name *.png | cut -d '/' -f 10- | sed "s/$/ 2/" >> $DATA/train.txt
find $DATA/train/invls -name *.png | cut -d '/' -f 10- | sed "s/$/ 3/" >> $DATA/train.txt
find $DATA/train/valid -name *.png | cut -d '/' -f 10- | sed "s/$/ 4/" >> $DATA/train.txt
echo "\n"

echo "Create val.txt..."
rm -rf $DATA/val.txt

find $DATA/val/error -name *.png | cut -d '/' -f 10- | sed "s/$/ 0/" >> $DATA/val.txt
find $DATA/val/half  -name *.png | cut -d '/' -f 10- | sed "s/$/ 1/" >> $DATA/val.txt
find $DATA/val/invlb -name *.png | cut -d '/' -f 10- | sed "s/$/ 2/" >> $DATA/val.txt
find $DATA/val/invls -name *.png | cut -d '/' -f 10- | sed "s/$/ 3/" >> $DATA/val.txt
find $DATA/val/valid -name *.png | cut -d '/' -f 10- | sed "s/$/ 4/" >> $DATA/val.txt
echo "\n"

echo "Create test.txt..."
rm -rf $DATA/test.txt

find $DATA/test/error -name *.png | cut -d '/' -f 10- | sed "s/$/ 0/" >> $DATA/test.txt
find $DATA/test/half  -name *.png | cut -d '/' -f 10- | sed "s/$/ 1/" >> $DATA/test.txt
find $DATA/test/invlb -name *.png | cut -d '/' -f 10- | sed "s/$/ 2/" >> $DATA/test.txt
find $DATA/test/invls -name *.png | cut -d '/' -f 10- | sed "s/$/ 3/" >> $DATA/test.txt
find $DATA/test/valid -name *.png | cut -d '/' -f 10- | sed "s/$/ 4/" >> $DATA/test.txt
echo "\n"

echo "All done!"
