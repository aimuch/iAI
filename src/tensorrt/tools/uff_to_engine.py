# -*- coding: utf-8 -*-
# Author : Andy Liu
# Last modified: 2019-03-15

# This script is used to convert .uff file to .engine for TX2/PX2 or other NVIDIA Platform
# Using: 
#        python uff_to_engine.py


import os
# import tensorflow as tf
import tensorrt as trt
from tensorrt.parsers import uffparser
import uff

print("TensorRT version = ", trt.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



frozen_input_name = "input"
net_input_shape = (3, 32, 32)
frozen_output_name = "fc_3/frozen"
uff_path = 'model.uff'
engine_path = "model.engine"

def uff2engine(frozen_input_name, net_input_shape,frozen_output_name,uff_path,engine_path):
    with open(uff_path, 'rb') as f:
        uff_model = f.read()
        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        parser = uffparser.create_uff_parser()
        parser.register_input(frozen_input_name, net_input_shape, 0)
        # parser.register_input("input", (3, 128, 128), 0)
        parser.register_output(frozen_output_name)
        engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1<<20 )
        parser.destroy()
        trt.utils.write_engine_to_file(engine_path, engine.serialize())

if __name__ == '__main__':
    
    engine_dir = os.path.dirname(engine_path)
    if not os.path.exists(engine_dir) and not engine_dir == '.' and not engine_dir =='':
        print("Warning !!! %s is not exists, now has create "%engine_dir)
        os.makedirs(engine_dir)

    uff2engine(frozen_input_name, net_input_shape,frozen_output_name,uff_path,engine_path)
    print("Engine file has saved in ", os.path.abspath(engine_path))