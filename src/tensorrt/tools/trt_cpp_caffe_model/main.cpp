#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;


#include <cstring>
#include <chrono>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#include <opencv2/opencv.hpp>



// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 5;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const std::vector<std::string> directories{ "./", "./" };

std::string locateFile(const std::string& input)
{
    return locateFile(input, directories);
}


void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)    			// output buffer for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(	locateFile(deployFile, directories).c_str(),
																locateFile(modelFile, directories).c_str(),
																*network,
																DataType::kFLOAT);
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// std::cout << "inputIndex = " << inputIndex << ", outputIndex = " << outputIndex << std::endl;

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv)
{
	// read test image file
	float total = 0, ms;
	#define MAX_LINE 1024
	FILE *fpOpen=fopen("/media/andy/Data/DevWorkSpace/Projects/imageClassifier/data/test_abs.txt","r");
	if(fpOpen==NULL)
	{
		std::cout << ">>>> Open Image List Txt Fail >>>> " << std::endl;
		return -1;
	}
	printf(">>>>>>> Open Image List Txt OK >>>>>>> \n");
	fflush(stdout);
	char strLine[MAX_LINE];
	int numberRun =0;
    int iCorrectNum = 0;

	// create a GIE model from the caffe model and serialize it to a stream
	IHostMemory *gieModelStream{nullptr};
	caffeToGIEModel("deploy.prototxt", "final.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);

	while(!feof(fpOpen))
	{
		fgets(strLine, MAX_LINE, fpOpen);
		char *token = strtok(strLine, " ");
		printf(">>>>>>> id: %d,openimg %s\n", numberRun, token);
		fflush(stdout);
		IplImage* testImg = cvLoadImage(token);

        IplImage *cvtimg = cvCreateImage(cvSize(testImg->width, testImg->height), IPL_DEPTH_8U, 3);
        cvCvtColor(testImg, cvtimg, CV_BGR2RGB);

        IplImage *in_img = cvCreateImage(cvSize(INPUT_W, INPUT_H), IPL_DEPTH_8U, INPUT_C);
		cvResize(cvtimg, in_img, CV_INTER_LINEAR);


		int _size = in_img->width * in_img->height * in_img->nChannels * sizeof(float);
		float* hostInput = (float*)malloc(_size);


        // >>>>>>>> OpenCV 3
        // img_read(token, hostInput);
        // <<<<<<<< OpenCV 3

        // >>>>>>>>  OpenCV 2
		int count = 0;
        unsigned char *data = (unsigned char *)in_img->imageData;
		// scale pixel and change HWC->CHW
		for(int c = 0; c < in_img->nChannels; c++){
			for(int j = 0; j < in_img->height; j++){
				for(int i = 0; i < in_img->width; i++){
                    // RGBRGBRGB -> RRRGGGBBB
					hostInput[count++] = (1.0 * data[j * (in_img->widthStep) + i * (in_img->nChannels) + c] - 128.0)/128.0;
				}
			}
		}
        // <<<<<<<< OpenCV 2


		// // deserialize the engine
		IRuntime* runtime = createInferRuntime(gLogger);
		ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
		IExecutionContext *context = engine->createExecutionContext();

		// run inference
		// float *outputs = (float*)malloc(OUTPUT_SIZE * sizeof(float));
		float outputs[OUTPUT_SIZE] = {0.0};

		auto t_start = std::chrono::high_resolution_clock::now();
		doInference(*context, hostInput, outputs, 1);
		auto t_end = std::chrono::high_resolution_clock::now();
		ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        total += ms;

		// print a histogram of the output distribution
		std::cout << "------ Predict ------" << std::endl;
		int maxIdx = 0;
		for (int i = 0; i < OUTPUT_SIZE; ++i)
			if (outputs[i] > outputs[maxIdx])
				maxIdx = i;

		for (int i = 0; i < OUTPUT_SIZE; ++i)
		{
			std::cout << i << " => " << outputs[i] << "\t : ";
			if (i == maxIdx)
				std::cout << "***";
			std::cout << "\n";
		}
		int pre = maxIdx;

		token = strtok(NULL, " ");
		char* label = token;
		if(token == NULL){
			break;
		}
		token = strtok(NULL, " ");

        if(pre==((int)(*label)-'0'))
        {
            iCorrectNum++;
        }

		// free(outputs);
		free(hostInput);
        cvReleaseImage(&cvtimg);
		cvReleaseImage(&testImg);
		cvReleaseImage(&in_img);

		// destroy the engine
		context->destroy();
		engine->destroy();
		runtime->destroy();

		numberRun++;
	}
	if (gieModelStream) {
		gieModelStream->destroy();
	}
	fclose(fpOpen);
	printf(">>>>>>> Prob = %f\n",iCorrectNum*1.0/numberRun);


    total /= numberRun;
    std::cout << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
}
