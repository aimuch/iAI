#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

// set USE_MPS=1 to compile with MPS
#ifndef USE_MPS
#define USE_MPS 0
#endif

// Required to enable MPS Support
#include <atomic>
#include <chrono>
#include <thread>

#if USE_MPS
#ifndef _MSC_VER
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sched.h>
#endif
#endif

#include "NvInfer.h"
#include "NvUffParser.h"
#include "common.h"

using namespace nvinfer1;
using namespace nvuffparser;

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_movielens: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

// constants that are known about the MovieLens (NCF) MLP network.
static const int32_t NUM_USERS{32}; // Total number of users.
static const int32_t TOPK_MOVIES{1}; // The output of the topK layer for MovieLens sample.
static const int32_t NUM_INDICES{100};  // Total numbers of Movies to predict per user.
static const int32_t EMBEDDING_VEC_SIZE{32}; // Embedding vector size of each user and item.
static const int32_t THREADS{5};
static const char* USER_BLOB_NAME{"user_input"}; // user input blob name.
static const char* ITEM_BLOB_NAME{"item_input"}; // item input blob name.
static const char* TOPK_ITEM_PROB{"topk_values"}; // predicted item probability blob name.
static const char* TOPK_ITEM_NAME{"topk_items"}; // predicted item probability blob name.
static const char* RATING_INPUT_FILE{"movielens_ratings.txt"};   // The default input file with 50 users and groundtruth data.
static const char* DEFAULT_WEIGHT_FILE{"sampleMovieLens.wts2"}; // The weight file produced from README.txt
static const char* UFF_MODEL_FILE{"sampleMovieLens.uff"};
static const char* UFF_OUTPUT_NODE{"prediction/Sigmoid"};
static const char* ENGINE_FILE{"sampleMovieLens.engine"};
static const int32_t DEVICE{0};
static const std::vector<std::string> directories{ "data/samples/movielens/", "data/movielens/" };
static Logger gLogger;

class TimerBase
{
public:
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void accumulate() = 0;
    virtual float getElapsedTime() const final { return mMS; }
    virtual bool isCPU() const { return false; }
    virtual void addTimer(TimerBase* rhs) final { mMS += rhs->getElapsedTime(); }
protected:
    TimerBase() {}
    float mMS{0.0f};
}; // class TimerBase

class GpuTimer : public TimerBase
{
public:
    GpuTimer(cudaStream_t stream)
        : mStream(stream)
    {
        CHECK(cudaEventCreate(&mStart));
        CHECK(cudaEventCreate(&mStop));
    }
    virtual ~GpuTimer()
    {
        CHECK(cudaEventDestroy(mStart));
        CHECK(cudaEventDestroy(mStop));
    }
    void start() override final { CHECK(cudaEventRecord(mStart, mStream)); }
    void stop() override final { CHECK(cudaEventRecord(mStop, mStream)); }
    void accumulate() override final
    {
        float ms{0.0f};
        CHECK(cudaEventSynchronize(mStop));
        CHECK(cudaEventElapsedTime(&ms, mStart, mStop));
        mMS += ms;
    }

private:
    cudaEvent_t mStart, mStop;
    cudaStream_t mStream;
}; // class GpuTimer

struct OutputArgs; // forward declaration

struct Args
{
    int32_t embeddingVecSize{EMBEDDING_VEC_SIZE};
    int32_t numUsers{NUM_USERS};   // Total number of users. Should be equal to ratings file users count.
    int32_t topKMovies{TOPK_MOVIES};    // TopK movies per user.
    int32_t numMoviesPerUser{NUM_INDICES};  // The number of movies per user.
    int32_t threads{THREADS};   // MPI Threads
    std::string weightFile{DEFAULT_WEIGHT_FILE};    // Weight file (.wts2) format Movielens sample.
    std::string ratingInputFile{RATING_INPUT_FILE}; // The input rating file.
    std::string uffFile{UFF_MODEL_FILE};
    std::string engineFile{ENGINE_FILE};
    bool enableFP16{false}; // Enable ability to run in FP16 mode.
    bool enableVerbose{false}; // Enable verbose perf analysis.
    bool enablePerf{true}; // Enable verbose perf analysis.
    bool success{true};       
    // The below structures are used to compare the predicted values to inference (ground truth)
    std::map<int32_t, vector<int32_t>> userToItemsMap;  // Lookup for inferred items for each user.
    std::map<int32_t, vector<pair<int32_t, float>>> userToExpectedItemProbMap;    // Lookup for topK items and probs for each user.
    int32_t device{DEVICE};
    std::vector<OutputArgs> pargsVec;
}; // struct args

// The OutptutArgs struct holds intermediate/final outputs generated by the MovieLens structure per user.
struct OutputArgs
{
    int32_t userId; // The user Id per batch.
    int32_t expectedPredictedMaxRatingItem; // The Expected Max Rating Item per user (inference ground truth).
    float expectedPredictedMaxRatingItemProb; // The Expected Max Rating Probability. (inference ground truth).
    vector<int32_t> allItems;   // All inferred items per user.
    vector<std::pair<int32_t, float>> itemProbPairVec;   // Expected topK items and prob per user.
}; // struct pargs

void printHelp(char *appName)
{
    std::cout << "Usage:\n"
        "\t " << appName << "[-h]\n"
        "\t-h      Display help information. All single dash optoins enable perf mode.\n"
        "\t-b      Number of Users i.e. BatchSize (default BatchSize=32).\n"
#if USE_MPS
        "\t-t      Number of MPI Threads (default threads=5).\n"
#endif
        "\t--verbose Enable verbose mode.\n"
        << std::endl;
}

// Parse the arguments and return failure if arguments are incorrect
// or help menu is requested.
void parseArgs(Args &args, int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string argStr(argv[i]);
        
        if (argStr == "-h")
        {
            printHelp(argv[0]);
            exit(EXIT_SUCCESS);
        }
        if (argStr == "-b")
        {
            i++;
            args.numUsers = std::atoi(argv[i]);
        }
#if USE_MPS
        else if (argStr == "-t")
        {
            i++;
            args.threads = std::atoi(argv[i]);
        }
#endif
        else if (argStr == "--verbose")
        {
            args.enableVerbose = true;
        }        
        else
        {
            std::cerr << "Invalid argument: " << argStr << std::endl;
            printHelp(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

void printOutputArgs(OutputArgs &pargs)
{
    cout << "User Id                            :   " << pargs.userId << endl;
    cout << "Expected Predicted Max Rating Item :   " << pargs.expectedPredictedMaxRatingItem << endl;
    cout << "Expected Predicted Max Rating Prob :   " << pargs.expectedPredictedMaxRatingItemProb << endl;
    cout << "Total TopK Items : " << pargs.itemProbPairVec.size() << endl;
    for (unsigned i=0; i<pargs.itemProbPairVec.size(); ++i)
        cout << pargs.itemProbPairVec.at(i).first << " : " << pargs.itemProbPairVec.at(i).second << endl;
    cout << endl << "------------------------------------------------------------" << endl;
}

std::string readNextLine(ifstream &file, char &delim)
{
    std::string line;
    std::getline(file, line);
    auto pos = line.find(delim);
    line = line.substr(pos+1);
    return line;
}

void readInputSample(ifstream &file, OutputArgs &pargs, std::string line, Args args)
{
    // read user name
    char delim = ':';
    auto pos = line.find(delim); 
    line = line.substr(pos+1);
    pargs.userId = std::stoi(line);
    // read items
    std::string items = readNextLine(file, delim);
    items = items.substr(2, items.size()-2);
    std::stringstream ss(items);
    std::string i;
    while(ss >> i) 
    {
        if (ss.peek() == ',' || ss.peek() == ' ')
            ss.ignore();
        i = i.substr(0, i.size()-1);
        pargs.allItems.push_back(stoi(i)); 
    }

    // read expected predicted max rating item
    pargs.expectedPredictedMaxRatingItem = std::stoi(readNextLine(file, delim));
    
    // read expected predicted max rating prob
    std::string prob = readNextLine(file, delim);
    prob = prob.substr(2,  prob.size()-3);
    pargs.expectedPredictedMaxRatingItemProb = std::stof(prob);
    
    // skip line
    std::getline(file, line);
    std::getline(file, line);
    
    // read all the top 10 prediction ratings
    for (int i=0; i<10; ++i) 
    {
        auto pos = line.find(delim);
        int32_t item = std::stoi(line.substr(0, pos-1));
        float prob = std::stof(line.substr(pos+2));
        pargs.itemProbPairVec.emplace_back((make_pair(item, prob)));
        std::getline(file, line);
    } 
}

void parseMovieLensData(Args &args)
{
    std::ifstream file;
    file.open(args.ratingInputFile);
    std::string line;  
    int userIdx = 0;
    while(std::getline(file, line)) 
    {
        OutputArgs pargs;
        readInputSample(file, pargs, line, args);
        args.userToItemsMap.insert(make_pair(userIdx, pargs.allItems));
        args.userToExpectedItemProbMap.insert(make_pair(userIdx, pargs.itemProbPairVec));
        
        // store the pargs in the global data structure. Hack.
        args.pargsVec.push_back(pargs);
        
        userIdx++;
        if (args.enableVerbose) printOutputArgs(pargs);
        pargs.allItems.resize(0);
        pargs.itemProbPairVec.resize(0);
        if (args.numUsers == userIdx)
            break;
    }

    // number of users should be equal to number of users in rating file
    assert(args.numUsers == userIdx);
}

template<typename T1, typename T2>
void printInferenceOutput(void* userInputPtr, void* itemInputPtr, void* topKItemNumberPtr, void* topKItemProbPtr, Args args)
{
    T1* userInput{static_cast<T1*>(userInputPtr)};
    T1* topKItemNumber{static_cast<T1*>(topKItemNumberPtr)};
    T2* topKItemProb{static_cast<T2*>(topKItemProbPtr)};
    
    std::cout << "Num of users : " << args.numUsers << std::endl;
    std::cout << "Num of Movies : " << args.numMoviesPerUser << std::endl;
 
    if (args.enableVerbose)
    {
        cout << "|-----------|------------|-----------------|-----------------|" << endl;
        cout << "|   User    |   Item     |  Expected Prob  |  Predicted Prob |" << endl;
        cout << "|-----------|------------|-----------------|-----------------|" << endl;
    } else
            std::cout << "-----------------------------------------------------------------" << endl;
    
    for (int i=0; i<args.numUsers; ++i) 
    {
        int userIdx = userInput[i * args.numMoviesPerUser];
        int maxPredictedIdx = topKItemNumber[i * args.topKMovies];
        int maxExpectedItem = args.userToExpectedItemProbMap[userIdx].at(0).first;
        int maxPredictedItem = args.userToItemsMap[userIdx].at(maxPredictedIdx);
        //assert(maxExpectedItem == maxPredictedItem); 
        
        if (!args.enableVerbose)
        {
#if USE_MPS
            cout << "| PID : " << setw(4) << getpid() << " | User :" << setw(4) << userIdx << "  |  Expected Item :" << setw(5) << maxExpectedItem << "  |  Predicted Item :" << setw(5) << maxPredictedItem << " | " << endl;
#else
            cout << "| User :" << setw(4) << userIdx << "  |  Expected Item :" << setw(5) << maxExpectedItem << "  |  Predicted Item :" << setw(5) << maxPredictedItem << " | " << endl;
#endif
        }

        if (args.enableVerbose) 
        {
            for (int k=0; k<args.topKMovies; ++k) 
            {
                int predictedIdx = topKItemNumber[i * args.topKMovies + k];
                float predictedProb = topKItemProb[i * args.topKMovies + k];
                float expectedProb = args.userToExpectedItemProbMap[userIdx].at(k).second;
                //int expectedItem = args.userToExpectedItemProbMap[userIdx].at(k).first;
                int predictedItem = args.userToItemsMap[userIdx].at(predictedIdx);
                //assert(expectedItem == predictedItem);
                cout << "|" << setw(10) << userIdx << " | "<< setw(10) << predictedItem << " | " << setw(15) << expectedProb << " | " << setw(15) << predictedProb << " | " << endl;
            }
        }
    }
}

#if USE_MPS
template<class T, int key>
class Shmem
{
public:
    Shmem(size_t count)
    {
        mId = shmget(key, count * sizeof(T), 0666 | IPC_CREAT);
        if (mId == -1)
        {
            std::cout << "SHGET ERROR " << std::endl;
            exit(1);
        }
        mData = shmat(mId, nullptr, 0);
        if (mData == (void*)-1)
        {
            std::cout << "SHMAT ERROR " << std::endl;
            exit(1);
        }
    }
    T* get()
    {
        return static_cast<T*>(mData);
    }
    ~Shmem()
    {
        shmdt(mData);
        shmctl(mId, IPC_RMID, NULL);
    }
private:
    int mId;
    void* mData;
};
#endif

struct Batch
{
	ICudaEngine* engine;
	IExecutionContext *context;
	cudaStream_t stream;
	cudaEvent_t completion, infStart, infEnd;
    void *hostMemory[5];
    void *deviceMemory[5];
	std::vector<size_t> memSizes;
};

Batch createBatch(ICudaEngine* engine, void* userInputPtr, void* itemInputPtr, Args args)
{
    Batch b;
	b.engine = engine;
	b.context = b.engine->createExecutionContext();
	CHECK(cudaStreamCreate(&b.stream));
	CHECK(cudaEventCreateWithFlags(&b.completion, cudaEventDisableTiming));
	CHECK(cudaEventCreate(&b.infStart));
	CHECK(cudaEventCreate(&b.infEnd));

    // In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int userInputIndex = b.engine->getBindingIndex(USER_BLOB_NAME); 
	int itemInputIndex = b.engine->getBindingIndex(ITEM_BLOB_NAME);
    int outputPredictionIndex = b.engine->getBindingIndex(UFF_OUTPUT_NODE);
    int outputItemProbIndex = b.engine->getBindingIndex(TOPK_ITEM_PROB);
    int outputItemNameIndex = b.engine->getBindingIndex(TOPK_ITEM_NAME);

    b.memSizes.push_back(args.numUsers * args.numMoviesPerUser * sizeof(float));
    b.memSizes.push_back(args.numUsers * args.numMoviesPerUser * sizeof(float));
    b.memSizes.push_back(args.numUsers * args.numMoviesPerUser * sizeof(float));
    b.memSizes.push_back(args.numUsers * args.topKMovies * sizeof(float));
    b.memSizes.push_back(args.numUsers * args.topKMovies * sizeof(float));
	
    CHECK(cudaMallocHost(&b.hostMemory[userInputIndex], b.memSizes[userInputIndex]));
	CHECK(cudaMallocHost(&b.hostMemory[itemInputIndex], b.memSizes[itemInputIndex]));
	CHECK(cudaMallocHost(&b.hostMemory[outputPredictionIndex], b.memSizes[outputPredictionIndex]));
	CHECK(cudaMallocHost(&b.hostMemory[outputItemProbIndex], b.memSizes[outputItemProbIndex]));
	CHECK(cudaMallocHost(&b.hostMemory[outputItemNameIndex], b.memSizes[outputItemNameIndex]));

    // copy the data to host memory
    for (unsigned int i=0; i < (b.memSizes[userInputIndex])/sizeof(float); ++i)
    {
        *(static_cast<uint32_t*>(b.hostMemory[userInputIndex]) + i) =  *((uint32_t*)userInputPtr + i);
    }
    for (unsigned int i=0; i < (b.memSizes[itemInputIndex])/sizeof(float); ++i)
    {
        *(static_cast<uint32_t*>(b.hostMemory[itemInputIndex]) + i) =  *((uint32_t*)itemInputPtr + i);
    }
    
    // allocate GPU memory 
    CHECK(cudaMalloc(&b.deviceMemory[userInputIndex], b.memSizes[userInputIndex]));
	CHECK(cudaMalloc(&b.deviceMemory[itemInputIndex], b.memSizes[itemInputIndex]));
	CHECK(cudaMalloc(&b.deviceMemory[outputPredictionIndex], b.memSizes[outputPredictionIndex]));
	CHECK(cudaMalloc(&b.deviceMemory[outputItemProbIndex], b.memSizes[outputItemProbIndex]));
	CHECK(cudaMalloc(&b.deviceMemory[outputItemNameIndex], b.memSizes[outputItemNameIndex]));

    return b;
}

void destroyBatch(Batch& b)
{
	for(auto p: b.hostMemory)
		CHECK(cudaFreeHost(p));
	for(auto p: b.deviceMemory)
		CHECK(cudaFree(p));
	CHECK(cudaStreamDestroy(b.stream));
	CHECK(cudaEventDestroy(b.completion));
	CHECK(cudaEventDestroy(b.infStart));
	CHECK(cudaEventDestroy(b.infEnd));
	b.context->destroy();
}

void submitWork(Batch& b, Args args)
{
	int userInputIndex = b.engine->getBindingIndex(USER_BLOB_NAME); 
	int itemInputIndex = b.engine->getBindingIndex(ITEM_BLOB_NAME);
    int outputPredictionIndex = b.engine->getBindingIndex(UFF_OUTPUT_NODE);
    int outputItemProbIndex = b.engine->getBindingIndex(TOPK_ITEM_PROB);
    int outputItemNameIndex = b.engine->getBindingIndex(TOPK_ITEM_NAME);
   
    // Copy input from host to device
    CHECK(cudaMemcpyAsync(b.deviceMemory[userInputIndex], b.hostMemory[userInputIndex], b.memSizes[userInputIndex], cudaMemcpyHostToDevice, b.stream));
    CHECK(cudaMemcpyAsync(b.deviceMemory[itemInputIndex], b.hostMemory[itemInputIndex], b.memSizes[itemInputIndex], cudaMemcpyHostToDevice, b.stream));
    
    CHECK(cudaEventRecord(b.infStart, b.stream));
    b.context->enqueue(args.numUsers, b.deviceMemory, b.stream, nullptr);
    CHECK(cudaEventRecord(b.infEnd, b.stream));
    
    // copy output from device to host
    CHECK(cudaMemcpyAsync(b.hostMemory[outputPredictionIndex], b.deviceMemory[outputPredictionIndex], b.memSizes[outputPredictionIndex], cudaMemcpyDeviceToHost, b.stream));
    CHECK(cudaMemcpyAsync(b.hostMemory[outputItemProbIndex], b.deviceMemory[outputItemProbIndex], b.memSizes[outputItemProbIndex], cudaMemcpyDeviceToHost, b.stream));
    CHECK(cudaMemcpyAsync(b.hostMemory[outputItemNameIndex], b.deviceMemory[outputItemNameIndex], b.memSizes[outputItemNameIndex], cudaMemcpyDeviceToHost, b.stream));
}

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, IUffParser* parser, Args args)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    
    std::cout << "Begin parsing model..." << std::endl;
   
    auto dType = args.enableFP16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    
    // parse the uff model to populate the network 
    if (!parser->parse(uffFile, *network, dType))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    
    std::cout << "End parsing model..." << std::endl;
   
    // add preprocessing i.e. topk layer to the UFF Network
    // retrieve last layer of UFF Network 
    auto uffLastLayer = network->getLayer(network->getNbLayers()-1);
   
    // Reshape output of fully connected layer numOfMovies x 1 x 1 x 1 to numOfMovies x 1 x 1. 
    auto reshapeLayer = network->addShuffle(*uffLastLayer->getOutput(0));
    reshapeLayer->setReshapeDimensions(Dims3{1, args.numMoviesPerUser, 1});
    assert(reshapeLayer != nullptr);

    // Apply TopK layer to retrieve item probabilities and corresponding index number. 
    auto topK = network->addTopK(*reshapeLayer->getOutput(0), TopKOperation::kMAX, args.topKMovies, 0x2);
    assert(topK != nullptr);

    // Mark outputs for index and probs. Also need to set the item layer type == kINT32.
    topK->getOutput(0)->setName(TOPK_ITEM_PROB);
	topK->getOutput(1)->setName(TOPK_ITEM_NAME);
	
    // specify topK tensors as outputs
    network->markOutput(*topK->getOutput(0));
    network->markOutput(*topK->getOutput(1));

    // set the topK indices tensor as INT32 type 
    topK->getOutput(1)->setType(DataType::kINT32);

	// Build the engine
    builder->setMaxBatchSize(args.numUsers);
    builder->setMaxWorkspaceSize(1 << 30);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    std::cout << "End building engine..." << std::endl;

    // we can clean the network and the parser */
    network->destroy();
    builder->destroy();
    return engine;
}

template <typename T>
void doInference(ICudaEngine* engine, std::atomic_int* sync, void* &userInputPtr, void* &itemInputPtr, Args &args)
{
    Batch b = createBatch(engine, userInputPtr, itemInputPtr, args);

#if USE_MPS 
    // wait for all threads to construct engines
    for ((*sync)--; sync->load(); sched_yield());		
#endif 
    
    auto start = std::chrono::high_resolution_clock::now();
    // run inference for all the threads
    submitWork(b, args);
    cudaStreamSynchronize(b.stream);   
    auto end = std::chrono::high_resolution_clock::now();
	float duration = std::chrono::duration<float, std::milli>(end - start).count();

#if USE_MPS 
    cout << "Done execution : " << getpid() << " . Duration : " << duration << endl;
#else
    cout << "Done execution. Duration : " << duration << endl;
#endif
    
    int outputItemProbIndex = b.engine->getBindingIndex(TOPK_ITEM_PROB);
    int outputItemNameIndex = b.engine->getBindingIndex(TOPK_ITEM_NAME);

    float* topKItemProb = static_cast<float*>(b.hostMemory[outputItemProbIndex]);
    uint32_t* topKItemNumber = static_cast<uint32_t*>(b.hostMemory[outputItemNameIndex]);
   
    printInferenceOutput<uint32_t, float>(userInputPtr, itemInputPtr, topKItemNumber, topKItemProb, args);
                
    // clean the batch after run
    destroyBatch(b);
}

int main(int argc, char *argv[])
{
    Args args;              // Global struct to store arguments
    OutputArgs pargs;       // Ratings file struct
    
    // parse arguments
    parseArgs(args, argc, argv);

#if USE_MPS
    // Create shared memory bindings
	Shmem<std::atomic_int, 161803> semMem(1);       // semaphore for thread synchronization
	semMem.get()->store(args.threads+1);
#endif

    // parse the ratings file and populate ground truth data 
    args.ratingInputFile = locateFile(args.ratingInputFile, directories);
    cout << args.ratingInputFile << endl;
    
    // parse ground truth data and inputs, common to all processes (if using MPS) 
    parseMovieLensData(args);

#if USE_MPS
    // create child process in loop
    for (int i=0; i < args.threads; ++i)
    {
        if (!fork())
        {
#endif
            // allocate input and output buffers on host 
            void* userInput = operator new(args.numUsers * args.numMoviesPerUser * sizeof(float));
            void* itemInput = operator new(args.numUsers * args.numMoviesPerUser * sizeof(float));
            
            // TODO: TensorRT does not share weights across different processes. Its a Limitation of the engine.
            // For each MPS process, we need to create separate engine object.
            
            // create uff parser
            args.uffFile = locateFile(args.uffFile, directories);
            auto parser = createUffParser();

            // register input and output nodes for UFF Parser
            Dims inputIndices;
            inputIndices.nbDims = 1;
            inputIndices.d[0] = args.numMoviesPerUser;
            
            parser->registerInput(USER_BLOB_NAME, inputIndices, UffInputOrder::kNCHW);      // numOfMovies
            parser->registerInput(ITEM_BLOB_NAME, inputIndices, UffInputOrder::kNCHW);      // numOfMovies
             
            parser->registerOutput(UFF_OUTPUT_NODE);                  // UFF File output node

            // load uff model to TensorRT engine
            ICudaEngine* engine = loadModelAndCreateEngine(args.uffFile.c_str(), parser, args);
	        assert(engine!=nullptr);
            
            for (int i=0; i<args.numUsers; ++i)
            { 
                for (int k=0; k<args.numMoviesPerUser; ++k)
                {
                    int idx = i * args.numMoviesPerUser + k;
                    static_cast<uint32_t*>(userInput)[idx] = args.pargsVec[i].userId;
                    static_cast<uint32_t*>(itemInput)[idx] = args.pargsVec[i].allItems.at(k);
                }
            }
            
#if USE_MPS
            // run inference
            doInference<float>(engine, semMem.get(), userInput, itemInput, args);
#else
            doInference<float>(engine, nullptr, userInput, itemInput, args);
#endif
            
            // free memory 
            operator delete(userInput);
            operator delete(itemInput);

            if (engine) engine->destroy();
            if (parser) parser->destroy();
#if USE_MPS
            exit(0); 
        }
    }

    for (;semMem.get()->load()!=1; sched_yield());      // wait for all threads to construct engines           
	
    auto start = std::chrono::high_resolution_clock::now();
    (*semMem.get())--;									// release all threads

    for (int i = 0; i < args.threads; i++)
	{
		int status;
		wait(&status);
        assert(status == 0);
	}
	
    auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "Number of threads executed : " << args.threads << ". Total MPS Run Duration : " << duration << std::endl;
#endif
    return EXIT_SUCCESS;
}
