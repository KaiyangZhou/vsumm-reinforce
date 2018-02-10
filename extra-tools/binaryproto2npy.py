import caffe
import numpy as np
import sys

"""This code is used to convert caffe mean file ended with binaryproto to npy file
How to use:
$ python binaryproto2npy.py path_to_caffe_mean_file saved_file_name
"""

if len(sys.argv) != 3:
    print "Usage: python binaryproto2npy.py proto.mean out.npy"
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out )
