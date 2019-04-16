#
# Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

import json
from json import encoder
import numpy as np
import argparse

try:
    from flask import Flask, request, jsonify
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Flask installed.
For installation instructions, see:
http://flask.pocoo.org/""".format(err))

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

from tensorrt.lite import Engine
from tensorrt.infer import LogSeverity

PARSER = argparse.ArgumentParser(description="Example of how to create a Caffe based TensorRT Engine and run inference")
PARSER.add_argument('datadir', help='Path to Python TensorRT data directory (realpath)')

ARGS = PARSER.parse_args()
DATA = ARGS.datadir
LABELS = open(DATA + '/resnet50/class_labels.txt', 'r').read().split('\n') #Get label information

ALLOWED_EXTENTIONS = set(['jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENTIONS

#Covert image to CHW Numpy array (TensorRT expects CHW data)
def image_to_np_CHW(image): return np.asarray(image.resize((engine.input_dim[0].H(), engine.input_dim[0].W()),
                                                           Image.ANTIALIAS)).transpose([2,0,1]).astype(engine.data_type.input_type())

#Post Processing Callback, Should take a 5D Tensor, run post processing and return a single object
def analyze(output_data):
    #Results from the engine are returned as a list of 5D numpy arrays:
    #        (Number of Batches x Batch Size x C x H x W)
    output = output_data.reshape(len(LABELS))

    # Get result
    top = np.argmax(output)
    top = LABELS[top]

    # Get top5
    top5 = np.argpartition(output, -5, axis=-1)[-5:]
    top5 = top5[np.argsort(output[top5])][::-1]
    top5_classes = []
    for i in top5:
        top5_classes.append((LABELS[i], output[i]))

    return [top, top5_classes]

#Arguments to create lite engine
network = {"framework":"tf",                                     #Source framework
           "path":DATA+"/resnet50/resnet50-infer-5.pb",          #Path to frozen model
           "input_nodes":{"input":(3,224,224)},                  #Dictionary of input nodes and their associated dimensions
           "output_nodes":["GPU_0/tower_0/Softmax"],             #List of output nodes
           "logger_severity":LogSeverity.INFO,                   #Debugging info
           "postprocessors":{"GPU_0/tower_0/Softmax":analyze}}   #Postprocessor function table

engine = Engine(**network)

#Web service
app = Flask(__name__)
@app.route("/classify", methods=["POST"])
def json_classify():
    if request.method == 'POST':
        img = Image.open(request.files['file'])
        #Format image to Numpy CHW and run inference, get the results of the single output node
        results = engine.infer(image_to_np_CHW(img))[0]
        #Retrive the results created by the post processor callback
        top_class_label, top5 = results[0], results[1]

        #Format data for JSON
        top5_str = []
        for t in top5:
            top5_str.append((t[0], str(t[1])))
        classification_data = {"top_class": top_class_label, "top5": top5_str}

        return jsonify (
            data = classification_data
        )

    else:
        return jsonify (
            error = "Invalid Request Type"
        )

@app.route("/", methods=['GET', 'POST'])
def html_classify():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = Image.open(request.files['file'])
            #Format image to Numpy CHW and run inference, get the results of the single output node
            results = engine.infer(image_to_np_CHW(img))[0]
            #Retrive the results created by the post processor callback
            top_class_label, top5 = results[0], results[1]

            #Format data for JSON
            top5_str = ""
            for t in top5:
                top5_str += ("<li>" + t[0] + ": " + str(t[1]) + "</li>")

            return ("<!doctype html>"
                "<title> Resnet as a Service </title>"
                "<h1> Classifed </h1>"
                "<p> Looks like a " + top_class_label + "</p>"
                "<h2> Top 5 </h2>"
                "<ul>"
                "" + top5_str + ""
                "</ul>")
        else:
            return '''Invalid Upload'''

    return '''
    <!doctype html>
    <title>Resnet as a Service</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run()
