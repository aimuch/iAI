#include "common.h"
//#include "LocalPrint.hpp"
#include "gie.hpp"
//#include <Checks.hpp>
//# define CHECK (X)
//using namespace cv;
static Logger gLogger;

// Caffe Model
#define SLOT_TF_IN_W        256
#define SLOT_TF_IN_H        256
#define SLOT_TF_IN_C        3
char slot_model_path[] = "./model/gie/slot_model.engine";

// // TensorFLow Molde
// #define SLOT_TF_IN_W        128
// #define SLOT_TF_IN_H        128
// #define SLOT_TF_IN_C        3
// char slot_model_path[] = "./model/apa/gie/slot_classifier_tf.engine";


GieModel *pModel;
GieModel *slot_pModel;
extern "C" void apaCaffeInit(){
    struct ModelPara model_parameter;
    model_parameter.engineNr = "./model/apa/gie/ps_parking3.engine";
    printf("init gie engine:%s;\n", model_parameter.engineNr);
    pModel = new GieModel(&model_parameter);
}

extern "C" void apaCaffeDeinit(void)
{
    delete pModel;
}

extern "C" float* ps_parking_infer(float* hostInput){
#ifdef GPU
    cudaMemcpy(pModel->m_deviceInputData, hostInput, 48 * 192 * sizeof(float), cudaMemcpyHostToDevice);
#endif
    pModel->inferSingleFrame();
    return pModel->m_hostOutputData[0];
}

// >>>>>>> Slot TensorFlow Engine >>>>>>>
extern "C" void slotTfInit(){
    struct ModelPara model_parameter;
    model_parameter.engineNr = slot_model_path;
    printf("Slot Classifier init gie engine: %s;\n", model_parameter.engineNr);
    slot_pModel = new GieModel(&model_parameter);
}

extern "C" void slotTfDelete(void)
{
    delete slot_pModel;
}

extern "C" float* slotTfInfer(float* hostInput){
#ifdef GPU
    cudaMemcpy(slot_pModel->m_deviceInputData, hostInput, SLOT_TF_IN_C * SLOT_TF_IN_W * SLOT_TF_IN_H * sizeof(float), cudaMemcpyHostToDevice);
#endif
    slot_pModel->inferSingleFrame();
    return slot_pModel->m_hostOutputData[0];
}
// <<<<<<< Slot TensorFlow Engine <<<<<<<


GieModel::GieModel(struct ModelPara *p)
{
	m_success = false;
	if (!p) {
		std::cerr << "GieModel must have input to initialize;" << std::endl;
		return;
	}
	para.engineNr = p->engineNr;
	//UserPrintInfo("creatEngine %s;\n", para.engineNr);
	engine = createEngine(std::string(para.engineNr));
	if (!engine) {
		std::cerr << "Engine could not be created!" << std::endl;
		return;
	}
	if (ParseParaFromEngine() < 0) {
		std::cerr << "ParseParaFromEngine fail!" << std::endl;
		return;
	}
	//UserPrintInfo("creatEngine %s -;\n", para.engineNr);

	//pDataCondiPara = (dwDataConditionerParams*)p->priv;
	//dwContext = _context;

	memcpy((void*)p, (void*)&para, sizeof(*p));
	context = engine->createExecutionContext();
	assert(engine->getNbBindings() == 2);

	// create GPU buffers and a stream
	m_networkInputSize = (uint32_t)(para.batchSize * para.inC * para.inH * para.inW * sizeof(float));
	m_networkOutputSize[0] = (uint32_t)(para.batchSize * para.outC * para.outH * para.outW * sizeof(float));
	CHECK(cudaMalloc(&buffers[para.inputIndex], (size_t)m_networkInputSize));
	CHECK(cudaMalloc(&buffers[para.outputIndex], (size_t)m_networkOutputSize[0]));

	m_deviceInputData = (float*)buffers[para.inputIndex];
	m_deviceOutputData[0] = (float*)buffers[para.outputIndex];
	m_hostOutputData[0] = (float*)malloc((size_t)m_networkOutputSize[0]);
	if (!m_hostOutputData[0]) {
		std::cerr << "m_hostOutputData malloc fail" << std::endl;
		return;
	}

	CHECK(cudaStreamCreate(&stream));

	m_success = true;
	//UserPrintInfo("creat GieModel success\n");

	printEngineInfo();

	dispModelPara(&para);
}

GieModel::~GieModel(void)
{
	if (m_hostOutputData[0])
		free(m_hostOutputData[0]);

	if (m_deviceOutputData[0])
		cudaFree(m_deviceOutputData[0]);

	if (m_deviceInputData)
		cudaFree(m_deviceInputData);

	if (context)
		context->destroy();

	if (engine)
		engine->destroy();
}

bool GieModel::inferSingleFrame(void)
{
	if (false == isLoaded()) {
		std::cerr << "Gie Model not load!" << std::endl;
		return false;
	}

	context->enqueue(para.batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(m_hostOutputData[0], m_deviceOutputData[0], m_networkOutputSize[0], cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	return true;
}

void GieModel::printEngineInfo()
{
	const char *dataTypeStr[] =
	{
		"kFLOAT",		//!< FP32 format
		"kHALF",		//!< FP16 format
		"kINT8" 	//!< INT8 format
	};

	const char *dimTypeStr[] =
	{
		"kSPATIAL", 		//!< elements correspond to different spatial data
		"kCHANNEL", 		//!< elements correspond to different channels
		"kINDEX",				//!< elements correspond to different batch index
		"kSEQUENCE" 		//!< elements correspond to different sequence values
	};

	int getNbBindings = engine->getNbBindings();
	printf("getNbBindings: %d;\n", getNbBindings);
	for (int i = 0; i < getNbBindings; i++) {
		printf("---> index %d: %s, input:%s\n", i, engine->getBindingName(i), engine->bindingIsInput(i) == true ? "y" : "n");
		Dims dim = engine->getBindingDimensions(i);
		printf("dim: nbDims:%d;\n", dim.nbDims);
		printf("d{%d,%d,%d}\n", dim.d[0], dim.d[1], dim.d[2]);
		printf("dim type{%s,%s,%s};\n", dimTypeStr[(int)dim.type[0]], dimTypeStr[(int)dim.type[1]], dimTypeStr[(int)dim.type[2]]);
		printf("data type: %s ;\n", dataTypeStr[(int)(engine->getBindingDataType(i))]);
	}
	printf("--------------------\n");
	printf("max batch size: %d;\n", engine->getMaxBatchSize());
	printf("NbLayers: %d;\n", engine->getNbLayers());
	printf("WorkspaceSize: %d;\n", (int)(engine->getWorkspaceSize()));
}

int GieModel::ParseParaFromEngine()
{
	int getNbBindings = engine->getNbBindings();
	if (getNbBindings < 2)
		return -1;

	for (int i = 0; i < getNbBindings; i++) {
		Dims dim = engine->getBindingDimensions(i);
		if (engine->bindingIsInput(i) == true) {
			para.inC = dim.d[0];
			para.inH = dim.d[1];
			para.inW = dim.d[2];
			para.inBlobNr = engine->getBindingName(i);
			para.inputIndex = i;
		} else {
			para.outC = dim.d[0];
			para.outH = dim.d[1];
			para.outW = dim.d[2];
			para.outBlobNr[0] = engine->getBindingName(i);
			para.outputIndex = i;
		}
	}

	para.batchSize = engine->getMaxBatchSize();
	return 0;
}

ICudaEngine* GieModel::createEngine(std::string engineFile)
{
	ICudaEngine *engine;
	if (!engineFile.empty()) {// GIE engine exist?
		char *gieModelStream{ nullptr };
		size_t size{ 0 };
		std::ifstream file(engineFile.c_str(), std::ios::binary);
		if (file.good()) { /* get engine file */
			file.seekg(0, file.end);
			size = file.tellg(); /* get file size first */
			file.seekg(0, file.beg);
			gieModelStream = new char[size];
			assert(gieModelStream);
 			file.read(gieModelStream, size); /* read Model file into buffer */
 			file.close();
 		}
		IRuntime* infer = createInferRuntime(gLogger);
		engine = infer->deserializeCudaEngine(gieModelStream, size, nullptr); //
		if (gieModelStream) delete[] gieModelStream;
		return engine;
	}
	else {
		std::cout << "Anable to load engine, please check your engine file path and name!" << std::endl;
		exit(1);
	}
}

