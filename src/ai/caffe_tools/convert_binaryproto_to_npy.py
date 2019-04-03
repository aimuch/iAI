#coding:utf-8
import caffe
import numpy as np
MEAN_PROTO_PATH="/home/andy/caffe/examples/mydata/apa_slot/data/mydata_mean.binaryproto"
MEAN_NPY_PATH = "/home/andy/caffe/examples/mydata/apa_slot/data/mydata_mean.npy"
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( MEAN_PROTO_PATH , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( MEAN_NPY_PATH , out )
