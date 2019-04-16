import sys

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

try:
    from tensorrt.parsers import onnxparser
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import argparse
    import numpy as np
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]""".format(err))

# Logger
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

def convert_to_datatype(v):
    if v==8:
        return trt.infer.DataType.INT8
    elif v==16:
        return trt.infer.DataType.HALF
    elif v==32:
        return trt.infer.DataType.FLOAT
    else:
        print("ERROR: Invalid model data type bit depth: " + str(v))
        return trt.infer.DataType.INT8

def get_input_output_names(trt_engine):
    nbindings = trt_engine.get_nb_bindings();
    assert(nbindings == 2)
    maps = {}
    for b in range(0, nbindings):
        dims = trt_engine.get_binding_dimensions(b).to_DimsCHW()
        if (trt_engine.binding_is_input(b)):
            print("Found input: ")
            print(trt_engine.get_binding_name(b))
            print("shape=" + str(dims.C()) + " , " + str(dims.H()) + " , " + str(dims.W()))
            print("dtype=" + str(trt_engine.get_binding_data_type(b)))
            maps["input"] = trt_engine.get_binding_name(b)
        else:
            print("Found output: ")
            print(trt_engine.get_binding_name(b))
            print("shape=" + str(dims.C()) + " , " + str(dims.H()) + " , " + str(dims.W()))
            print("dtype=" + str(trt_engine.get_binding_data_type(b)))
            maps["output"] = trt_engine.get_binding_name(b)
    return maps

def normalize_data(data, inp_dims):
    in_size = inp_dims.C() * inp_dims.H() * inp_dims.W()
    for s in range(0, in_size):
        data[s] = (data[s] / 255 - 0.45) / 0.225
    return data

def read_ascii_file(input_file, size):
    ret = []
    for line in open(input_file, 'r'):
        ret += line.split()

    ret = np.array(ret, np.float32)
    assert(ret.size == size)
    return ret

def prepare_input(input_file, trt_engine, file_format):
    in_out = get_input_output_names(trt_engine)
    input_indx = trt_engine.get_binding_index(in_out["input"])
    inp_dims = trt_engine.get_binding_dimensions(input_indx).to_DimsCHW()

    if (file_format=="ascii"):
        img = read_ascii_file(input_file, inp_dims.C() * inp_dims.H() * inp_dims.W())
    elif (file_format == "ppm"):
        img = preprocess_image(input_file, inp_dims)
    else:
        print("Not supported format")
        sys.exit()
    return img

def process_output(output, file_format, ref_file, topK):
    if file_format == "ascii":
        gold_ref = read_ascii_file(ref_file, output.size)
        res_vec = np.argsort(-output)[:topK]
        ref_vec = np.argsort(-gold_ref)[:topK]
        for k in range(0, topK):
            print(str(res_vec[k]) + "  " + str(ref_vec[k]))
    elif file_format == "ppm":
        ref_vec = read_reference_file(ref_file)
        desc_output = np.argsort(-output)
        output_sorted = np.sort(output)[::-1]
        for k in range(0, topK):
            print("Class : "+str(desc_output[k] + 1)+"  ==  "+ ref_vec[desc_output[k]])
    else:
        print("Format Not Supported")
        sys.exit()

def preprocess_image(image_path, inp_dims):
    ppm_image = Image.open(image_path)
    # resize image
    new_h = 224
    new_w = 224
    size = (new_w, new_h)
    # resize image
    img = ppm_image.resize(size, Image.NEAREST)
    # convert to numpy array
    img = np.array(img)
    # hwc2chw
    img = img.transpose(2, 0, 1)
    # convert image to 1D array
    img = img.ravel()
    # convert image to float
    img = img.astype(np.float32)
    # normalize image data
    img = normalize_data(img, inp_dims)
    return img

def read_reference_file(ref_file):
    with open(ref_file) as f:
        vec = f.readlines()

    vec = [x.strip() for x in vec]
    return vec

def inference_image(context, input_img, batch_size):
    # load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    inp_dims = engine.get_binding_dimensions(0).to_DimsCHW()
    out_dims = engine.get_binding_dimensions(1).to_DimsCHW()
    # output vector size
    output_size = 1000
    # create output array
    output = np.empty(output_size, dtype=np.float32)
    # allocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)
    # create input/output bindings
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # transfer predictions
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # synchronize threads
    stream.synchronize()
    return output

def sample_onnx_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_format", default="ascii", choices=["ascii", "ppm"], type=str, help="input file format. ASCII if not specified.")
    parser.add_argument("-i", "--image_file", type=str, required=True, help="Image or ASCII file")
    parser.add_argument("-r", "--reference_file", type=str, required=True, help="Reference files with correct labels")
    parser.add_argument("-k", "--topK", type=str, required=True, help="Top K values predictions to print")
    parser.add_argument("-m", "--model_file", type=str, required=True, help="ONNX Model file")
    parser.add_argument("-d", "--data_type", default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
    parser.add_argument("-b", "--max_batch_size", default=32, type=int, help="Maximum batch size")
    parser.add_argument("-w", "--max_workspace_size", default=1024*1024, type=int, help="Maximum workspace size")
    parser.add_argument("-v", "--add_verbosity", action="store_true")
    parser.add_argument("-q", "--reduce_verbosity", action="store_true")
    parser.add_argument("-l", "--print_layer_info", action="store_true")
    args = parser.parse_args()

    file_format = str.strip(args.file_format)
    image_file = str.strip(args.image_file)
    reference_file = str.strip(args.reference_file)
    model_file = str.strip(args.model_file)
    topK = int(args.topK)
    max_batch_size = args.max_batch_size
    max_workspace_size = args.max_workspace_size
    data_type = args.data_type
    add_verbosity = args.add_verbosity
    reduce_verbosity = args.reduce_verbosity
    print_layer_info = args.print_layer_info

    print("Input Arguments: ")
    print("file_format", file_format)
    print("image_file", image_file)
    print("reference_file",reference_file)
    print("topK", str(topK))
    print("model_file",model_file)
    print("data_type",data_type)
    print("max_workspace_size",max_workspace_size)
    print("max_batch_size",max_batch_size)
    print("add_verbosity", add_verbosity)
    print("reduce_verbosity", reduce_verbosity)
    print("print_layer_info",print_layer_info)

    # Create onnx_config
    apex = onnxparser.create_onnxconfig()
    apex.set_model_file_name(model_file)
    apex.set_model_dtype(convert_to_datatype(data_type))
    if print_layer_info:
        apex.set_print_layer_info(True)
    if add_verbosity:
        apex.add_verbosity()
    if reduce_verbosity:
        apex.reduce_verbosity()

    # set batch size
    batch_size = 1

    # create parser
    trt_parser = onnxparser.create_onnxparser(apex)
    assert(trt_parser)
    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)
    trt_parser.report_parsing_info()
    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()
    assert(trt_network)

    # create infer builder
    trt_builder = trt.infer.create_infer_builder(G_LOGGER)
    trt_builder.set_max_batch_size(max_batch_size)
    trt_builder.set_max_workspace_size(max_workspace_size)

    if (apex.get_model_dtype() == trt.infer.DataType_kHALF):
        print("-------------------  Running FP16 -----------------------------")
        trt_builder.set_fp16_mode(True)
    elif (apex.get_model_dtype() == trt.infer.DataType_kINT8):
        print("Int8 Model not supported")
        sys.exit()
    else:
        print("-------------------  Running FP32 -----------------------------")

    print("----- Builder is Done -----")
    print("----- Creating Engine -----")
    trt_engine = trt_builder.build_cuda_engine(trt_network)
    print("----- Engine is built -----")

    # create input vector
    input_img = prepare_input(image_file, trt_engine, file_format)

    if input_img.size == 0:
        msg = "sampleONNX the input tensor is of zero size - please check your path to the input or the file type"
        G_LOGGER.log(trt.infer.Logger.Severity_kERROR, msg)

    trt_context = trt_engine.create_execution_context()
    output = inference_image(trt_context, input_img, batch_size)

    # post processing stage
    process_output(output, file_format, reference_file, topK)

    # clean up
    trt_parser.destroy()
    trt_network.destroy()
    trt_context.destroy()
    trt_engine.destroy()
    trt_builder.destroy()
    print("&&&& PASSED Onnx Parser Tested Successfully")

if __name__=="__main__":
    sample_onnx_parser()
