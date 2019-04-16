#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include "data/benchmarkWriter.h"
#include "data/bleuScoreWriter.h"
#include "data/dataReader.h"
#include "data/dataWriter.h"
#include "data/limitedSamplesDataReader.h"
#include "data/sequenceProperties.h"
#include "data/textReader.h"
#include "data/textWriter.h"
#include "data/vocabulary.h"
#include "deviceBuffer.h"
#include "model/alignment.h"
#include "model/attention.h"
#include "model/beamSearchPolicy.h"
#include "model/componentWeights.h"
#include "model/context.h"
#include "model/debugUtil.h"
#include "model/decoder.h"
#include "model/embedder.h"
#include "model/encoder.h"
#include "model/likelihood.h"
#include "model/lstmDecoder.h"
#include "model/lstmEncoder.h"
#include "model/multiplicativeAlignment.h"
#include "model/projection.h"
#include "model/slpAttention.h"
#include "model/slpEmbedder.h"
#include "model/slpProjection.h"
#include "model/softmaxLikelihood.h"
#include "pinnedHostBuffer.h"
#include "trtUtil.h"
#include "common.h"

#include "NvInfer.h"

bool gPrintComponentInfo = true;
bool gFeedAttentionToInput = true;
bool gInitializeDecoderFromEncoderHiddenStates = true;

int gMaxBatchSize = 128;
int gBeamWidth = 5;
int gMaxInputSequenceLength = 150;
int gMaxOutputSequenceLength = -1;
int gMaxInferenceSamples = -1;
std::string gDataWriterStr = "bleu";
std::string gOutputTextFileName("translation_output.txt");
int gMaxWorkspaceSize = (1 << 28);
std::string gDataDirectory("data/samples/nmt/deen");
bool gEnableProfiling = false;
bool gAggregateProfiling = false;
bool gFp16 = false;
bool gVerbose = false;

std::string gInputTextFileName("newstest2015.tok.bpe.32000.de");
std::string gReferenceOutputTextFileName("newstest2015.tok.bpe.32000.en");
std::string gInputVocabularyFileName("vocab.bpe.32000.de");
std::string gOutputVocabularyFileName("vocab.bpe.32000.en");
std::string gEncEmbedFileName("weights/encembed.bin");
std::string gEncRnnFileName("weights/encrnn.bin");
std::string gDecEmbedFileName("weights/decembed.bin");
std::string gDecRnnFileName("weights/decrnn.bin");
std::string gDecAttFileName("weights/decatt.bin");
std::string gDecMemFileName("weights/decmem.bin");
std::string gDecProjFileName("weights/decproj.bin");
nmtSample::Vocabulary::ptr gOutputVocabulary = std::make_shared<nmtSample::Vocabulary>();

std::string locateNMTFile(const std::string &fpathSuffix)
{
    std::vector<std::string> dirs{std::string(gDataDirectory) + "/", "data/nmt/deen/"};
    return locateFile(fpathSuffix, dirs);
}

struct SimpleProfiler : public nvinfer1::IProfiler
{
    struct Record
    {
        float time{0};
        int count{0};
    };

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        mProfile[layerName].count++;
        mProfile[layerName].time += ms;
    }

    SimpleProfiler(
        const char* name,
        const std::vector<SimpleProfiler>& srcProfilers = std::vector<SimpleProfiler>())
        : mName(name)
    {
        for (const auto& srcProfiler : srcProfilers)
        {
            for (const auto& rec : srcProfiler.mProfile)
            {
                auto it = mProfile.find(rec.first);
                if (it == mProfile.end())
                {
                    mProfile.insert(rec);
                }
                else
                {
                    it->second.time += rec.second.time;
                    it->second.count += rec.second.count;
                }
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const SimpleProfiler& value)
    {
        out << "========== " << value.mName << " profile ==========" << std::endl;
        float totalTime = 0;
        std::string layerNameStr = "TensorRT layer name";
        int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
        for (const auto& elem : value.mProfile)
        {
            totalTime += elem.second.time;
            maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
        }

        auto old_settings = out.flags();
        auto old_precision = out.precision();
        // Output header
        {
            out << std::setw(maxLayerNameLength) << layerNameStr << " ";
            out << std::setw(12) << "Runtime, "
                << "%"
                << " ";
            out << std::setw(12) << "Invocations"
                << " ";
            out << std::setw(12) << "Runtime, ms" << std::endl;
        }
        for (const auto& elem : value.mProfile)
        {
            out << std::setw(maxLayerNameLength) << elem.first << " ";
            out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.second.time * 100.0F / totalTime) << "%"
                << " ";
            out << std::setw(12) << elem.second.count << " ";
            out << std::setw(12) << std::fixed << std::setprecision(2) << elem.second.time << std::endl;
        }
        out.flags(old_settings);
        out.precision(old_precision);
        out << "========== " << value.mName << " total runtime = " << totalTime << " ms ==========" << std::endl;

        return out;
    }

private:
    std::string mName;
    std::map<std::string, Record> mProfile;
};

Logger gLogger;

nmtSample::SequenceProperties::ptr getOutputSequenceProperties()
{
    return gOutputVocabulary;
}

nmtSample::DataReader::ptr getDataReader()
{
    std::shared_ptr<std::istream> textInput(new std::ifstream(locateNMTFile(gInputTextFileName)));
    std::shared_ptr<std::istream> vocabInput(new std::ifstream(locateNMTFile(gInputVocabularyFileName)));
    assert(textInput->good());
    assert(vocabInput->good());

    auto vocabulary = std::make_shared<nmtSample::Vocabulary>();
    *vocabInput >> *vocabulary;

    auto reader = std::make_shared<nmtSample::TextReader>(textInput, vocabulary);

    if (gMaxInferenceSamples >= 0)
        return std::make_shared<nmtSample::LimitedSamplesDataReader>(gMaxInferenceSamples, reader);
    else
        return reader;
}

nmtSample::Embedder::ptr getInputEmbedder()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(locateNMTFile(gEncEmbedFileName));
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::SLPEmbedder>(weights);
}

nmtSample::Embedder::ptr getOutputEmbedder()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(locateNMTFile(gDecEmbedFileName));
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::SLPEmbedder>(weights);
}

nmtSample::Encoder::ptr getEncoder()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(locateNMTFile(gEncRnnFileName));
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::LSTMEncoder>(weights);
}

nmtSample::Alignment::ptr getAlignment()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(locateNMTFile(gDecMemFileName));
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::MultiplicativeAlignment>(weights);
}

nmtSample::Context::ptr getContext()
{
    return std::make_shared<nmtSample::Context>();
}

nmtSample::Decoder::ptr getDecoder()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(locateNMTFile(gDecRnnFileName));
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::LSTMDecoder>(weights);
}

nmtSample::Attention::ptr getAttention()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(locateNMTFile(gDecAttFileName));
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::SLPAttention>(weights);
}

nmtSample::Projection::ptr getProjection()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(locateNMTFile(gDecProjFileName));
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::SLPProjection>(weights);
}

nmtSample::Likelihood::ptr getLikelihood()
{
    return std::make_shared<nmtSample::SoftmaxLikelihood>();
}

nmtSample::BeamSearchPolicy::ptr getSearchPolicy(int endSequenceId, nmtSample::LikelihoodCombinationOperator::ptr likelihoodCombinationOperator)
{
    return std::make_shared<nmtSample::BeamSearchPolicy>(endSequenceId, likelihoodCombinationOperator, gBeamWidth);
}

nmtSample::DataWriter::ptr getDataWriter()
{
    if (gDataWriterStr == "bleu")
    {
        std::shared_ptr<std::istream> textInput(new std::ifstream(locateNMTFile(gReferenceOutputTextFileName)));
        assert(textInput->good());
        return std::make_shared<nmtSample::BLEUScoreWriter>(textInput, gOutputVocabulary);
    }
    else if (gDataWriterStr == "text")
    {
        std::shared_ptr<std::ostream> textOutput(new std::ofstream(gOutputTextFileName));
        assert(textOutput->good());
        return std::make_shared<nmtSample::TextWriter>(textOutput, gOutputVocabulary);
    }
    else if (gDataWriterStr == "benchmark")
    {
        return std::make_shared<nmtSample::BenchmarkWriter>();
    }
    else
    {
        std::cerr << "Invalid data writer specified: " << gDataWriterStr << std::endl;
        assert(0);
        return nmtSample::DataWriter::ptr();
    }
}

bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseBool(const char* arg, const char* longName, bool& value, char shortName = 0)
{
    bool match = false;

    if (shortName)
    {
        match = (arg[0] == '-') && (arg[1] == shortName);
    }
    if (!match && longName)
    {
        const size_t n = strlen(longName);
        match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, longName, n);
    }
    if (match)
    {
        std::cout << longName << ": true" << std::endl;
        value = true;
    }
    return match;
}

void printUsage()
{
    printf("\nOptional params:\n");
    printf("  --help, -h                               Output help message and exit\n");
    printf("  --data_writer=bleu/text/benchmark        Type of the output the app generates (default = %s)\n", gDataWriterStr.c_str());
    printf("  --output_file=<path_to_file>             Path to the output file when data_writer=text (default = %s)\n", gOutputTextFileName.c_str());
    printf("  --batch=<N>                              Batch size (default = %d)\n", gMaxBatchSize);
    printf("  --beam=<N>                               Beam width (default = %d)\n", gBeamWidth);
    printf("  --max_input_sequence_length=<N>          Maximum length for input sequences (default = %d)\n", gMaxInputSequenceLength);
    printf("  --max_output_sequence_length=<N>         Maximum length for output sequences (default = %d), negative value indicates no limit\n", gMaxOutputSequenceLength);
    printf("  --max_inference_samples=<N>              Maximum sample count to run inference for, negative values indicates no limit is set (default = %d)\n", gMaxInferenceSamples);
    printf("  --verbose                                Output info-level messages by TensorRT\n");
    printf("  --max_workspace_size=<N>                 Maximum workspace size (default = %d)\n", gMaxWorkspaceSize);
    printf("  --data_dir=<path_to_data_directory>      Path to the directory where data and weights are located (default = %s)\n", gDataDirectory.c_str());
    printf("  --profile                                Profile TensorRT execution layer by layer. Use benchmark data_writer when profiling on, disregard benchmark results\n");
    printf("  --aggregate_profile                      Merge profiles from multuple Tensor RT engines\n");
    printf("  --fp16                                   Switch on fp16 math\n");
}

bool parseArgs(int argc, char* argv[])
{
    if (argc < 1)
    {
        printUsage();
        return false;
    }

    bool showHelp = false;
    for (int j = 1; j < argc; j++)
    {
        if (parseBool(argv[j], "help", showHelp, 'h'))
            continue;
        if (parseString(argv[j], "data_writer", gDataWriterStr))
            continue;
        if (parseString(argv[j], "output_file", gOutputTextFileName))
            continue;
        if (parseInt(argv[j], "batch", gMaxBatchSize))
            continue;
        if (parseInt(argv[j], "beam", gBeamWidth))
            continue;
        if (parseInt(argv[j], "max_input_sequence_length", gMaxInputSequenceLength))
            continue;
        if (parseInt(argv[j], "max_output_sequence_length", gMaxOutputSequenceLength))
            continue;
        if (parseInt(argv[j], "max_inference_samples", gMaxInferenceSamples))
            continue;
        if (parseBool(argv[j], "verbose", gVerbose))
            continue;
        if (parseInt(argv[j], "max_workspace_size", gMaxWorkspaceSize))
            continue;
        if (parseString(argv[j], "data_dir", gDataDirectory))
            continue;
        if (parseBool(argv[j], "profile", gEnableProfiling))
            continue;
        if (parseBool(argv[j], "aggregate_profile", gAggregateProfiling))
            continue;
        if (parseBool(argv[j], "fp16", gFp16))
            continue;
    }

    if (showHelp)
    {
        printUsage();
        return false;
    }

    return true;
}

nvinfer1::ICudaEngine* getEncoderEngine(
    nmtSample::Embedder::ptr inputEmbedder,
    nmtSample::Encoder::ptr encoder,
    nmtSample::Alignment::ptr alignment)
{
    nvinfer1::IBuilder* encoderBuilder = nvinfer1::createInferBuilder(gLogger);
    encoderBuilder->setMaxBatchSize(gMaxBatchSize);
    encoderBuilder->setMaxWorkspaceSize(gMaxWorkspaceSize);
    encoderBuilder->setFp16Mode(gFp16);

    nvinfer1::INetworkDefinition* encoderNetwork = encoderBuilder->createNetwork();

    // Define inputs for the encoder
    nvinfer1::Dims inputDims{1, {gMaxInputSequenceLength}, {nvinfer1::DimensionType::kSEQUENCE}};
    auto inputEncoderDataTensor = encoderNetwork->addInput("input_encoder_data", nvinfer1::DataType::kINT32, inputDims);
    assert(inputEncoderDataTensor != nullptr);
    nvinfer1::Dims inputSequenceLengthsDims{0, {}, {}};
    auto actualInputSequenceLengthsTensor = encoderNetwork->addInput("actual_input_sequence_lengths", nvinfer1::DataType::kINT32, inputSequenceLengthsDims);
    assert(actualInputSequenceLengthsTensor != nullptr);
    nvinfer1::Dims inputSequenceLengthsWithUnitIndexDims{1, {1}, {nvinfer1::DimensionType::kINDEX}};
    auto actualInputSequenceLengthsWithUnitIndexTensor = encoderNetwork->addInput("actual_input_sequence_lengths_with_index_dim", nvinfer1::DataType::kINT32, inputSequenceLengthsWithUnitIndexDims);
    assert(actualInputSequenceLengthsWithUnitIndexTensor != nullptr);

    auto stateSizes = encoder->getStateSizes();
    std::vector<nvinfer1::ITensor*> encoderInputStatesTensors(stateSizes.size());
    for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_encoder_states_" << i;
        encoderInputStatesTensors[i] = encoderNetwork->addInput(ss.str().c_str(), gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, stateSizes[i]);
        assert(encoderInputStatesTensors[i] != nullptr);
    }

    nvinfer1::ITensor* initializeDecoderIndicesTensor = nullptr;
    if (gInitializeDecoderFromEncoderHiddenStates)
    {
        nvinfer1::Dims inputDims{1, {gBeamWidth}, {nvinfer1::DimensionType::kINDEX}};
        initializeDecoderIndicesTensor = encoderNetwork->addInput("initialize_decoder_indices", nvinfer1::DataType::kINT32, inputDims);
        assert(initializeDecoderIndicesTensor != nullptr);
    }

    nvinfer1::ITensor* inputEncoderEmbeddedTensor;
    inputEmbedder->addToModel(encoderNetwork, inputEncoderDataTensor, &inputEncoderEmbeddedTensor);
    inputEncoderEmbeddedTensor->setName("input_data_embedded");

    nvinfer1::ITensor* memoryStatesTensor;
    std::vector<nvinfer1::ITensor*> encoderOutputStatesTensors(stateSizes.size());
    encoder->addToModel(
        encoderNetwork,
        gMaxInputSequenceLength,
        inputEncoderEmbeddedTensor,
        actualInputSequenceLengthsTensor,
        &encoderInputStatesTensors[0],
        &memoryStatesTensor,
        gInitializeDecoderFromEncoderHiddenStates ? &encoderOutputStatesTensors[0] : nullptr);
    memoryStatesTensor->setName("memory_states");
    encoderNetwork->markOutput(*memoryStatesTensor);
    memoryStatesTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);

    if (alignment->getAttentionKeySize() > 0)
    {
        nvinfer1::ITensor* attentionKeysTensor;
        alignment->addAttentionKeys(
            encoderNetwork,
            memoryStatesTensor,
            &attentionKeysTensor);
        attentionKeysTensor->setName("attention_keys");
        encoderNetwork->markOutput(*attentionKeysTensor);
        attentionKeysTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    // Replicate sequence lengths for the decoder
    {
        auto gatherLayer = encoderNetwork->addGather(*actualInputSequenceLengthsWithUnitIndexTensor, *initializeDecoderIndicesTensor, 0);
        assert(gatherLayer != nullptr);
        gatherLayer->setName("Replicate input sequence lengths for decoder");
        auto actualInputSequenceLengthsReplicatedTensor = gatherLayer->getOutput(0);
        assert(actualInputSequenceLengthsReplicatedTensor != nullptr);
        actualInputSequenceLengthsReplicatedTensor->setName("actual_input_sequence_lengths_replicated");
        encoderNetwork->markOutput(*actualInputSequenceLengthsReplicatedTensor);
        actualInputSequenceLengthsReplicatedTensor->setType(nvinfer1::DataType::kINT32);
    }

    if (gInitializeDecoderFromEncoderHiddenStates)
    {
        for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
        {
            assert(encoderOutputStatesTensors[i] != nullptr);

            // Insert index (Z=1) dimension into tensor
            nvinfer1::ITensor* encoderOutputStatesTensorWithUnitIndex;
            {
                auto shuffleLayer = encoderNetwork->addShuffle(*encoderOutputStatesTensors[i]);
                assert(shuffleLayer != nullptr);
                {
                    std::stringstream ss;
                    ss << "Reshape encoder states for decoder initialization " << i;
                    shuffleLayer->setName(ss.str().c_str());
                }
                nvinfer1::Dims shuffleDims;
                {
                    shuffleDims.nbDims = stateSizes[i].nbDims + 1;
                    shuffleDims.d[0] = 1;
                    shuffleDims.type[0] = nvinfer1::DimensionType::kINDEX;
                    for (int j = 0; j < stateSizes[i].nbDims; ++j)
                    {
                        shuffleDims.d[j + 1] = stateSizes[i].d[j];
                        shuffleDims.type[j + 1] = stateSizes[i].type[j];
                    }
                }
                shuffleLayer->setReshapeDimensions(shuffleDims);
                encoderOutputStatesTensorWithUnitIndex = shuffleLayer->getOutput(0);
                assert(encoderOutputStatesTensorWithUnitIndex != nullptr);
            }
            auto gatherLayer = encoderNetwork->addGather(*encoderOutputStatesTensorWithUnitIndex, *initializeDecoderIndicesTensor, 0);
            assert(gatherLayer != nullptr);
            {
                std::stringstream ss;
                ss << "Replicate encoder states for decoder initialization " << i;
                gatherLayer->setName(ss.str().c_str());
            }
            auto inputDecoderHiddenStatesTensor = gatherLayer->getOutput(0);
            assert(inputDecoderHiddenStatesTensor != nullptr);
            std::stringstream ss;
            ss << "input_decoder_states_" << i;
            inputDecoderHiddenStatesTensor->setName(ss.str().c_str());
            encoderNetwork->markOutput(*inputDecoderHiddenStatesTensor);
            inputDecoderHiddenStatesTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
        }
    }

    auto res = encoderBuilder->buildCudaEngine(*encoderNetwork);
    encoderNetwork->destroy();
    encoderBuilder->destroy();
    return res;
}

nvinfer1::ICudaEngine* getGeneratorEngine(
    nmtSample::Embedder::ptr outputEmbedder,
    nmtSample::Decoder::ptr decoder,
    nmtSample::Alignment::ptr alignment,
    nmtSample::Context::ptr context,
    nmtSample::Attention::ptr attention,
    nmtSample::Projection::ptr projection,
    nmtSample::Likelihood::ptr likelihood)
{
    nvinfer1::IBuilder* generatorBuilder = nvinfer1::createInferBuilder(gLogger);
    generatorBuilder->setMaxBatchSize(gMaxBatchSize);
    generatorBuilder->setMaxWorkspaceSize(gMaxWorkspaceSize);
    generatorBuilder->setFp16Mode(gFp16);

    nvinfer1::INetworkDefinition* generatorNetwork = generatorBuilder->createNetwork();

    // Define inputs for the generator
    auto stateSizes = decoder->getStateSizes();
    std::vector<nvinfer1::ITensor*> decoderInputStatesTensors(stateSizes.size());
    for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        nvinfer1::Dims statesDims;
        {
            statesDims.nbDims = stateSizes[i].nbDims + 1;
            statesDims.d[0] = gBeamWidth;
            statesDims.type[0] = nvinfer1::DimensionType::kINDEX;
            for (int j = 0; j < stateSizes[i].nbDims; ++j)
            {
                statesDims.d[j + 1] = stateSizes[i].d[j];
                statesDims.type[j + 1] = stateSizes[i].type[j];
            }
        }
        decoderInputStatesTensors[i] = generatorNetwork->addInput(ss.str().c_str(), gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, statesDims);
        assert(decoderInputStatesTensors[i] != nullptr);
    }
    nvinfer1::Dims inputDecoderDataDims{1, {gBeamWidth}, {nvinfer1::DimensionType::kINDEX}};
    auto inputDecoderDataTensor = generatorNetwork->addInput("input_decoder_data", nvinfer1::DataType::kINT32, inputDecoderDataDims);
    assert(inputDecoderDataTensor != nullptr);
    nvinfer1::Dims inputSequenceLengthsTeplicatedDims{2, {gBeamWidth, 1}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
    auto actualInputSequenceLengthsReplicatedTensor = generatorNetwork->addInput("actual_input_sequence_lengths_replicated", nvinfer1::DataType::kINT32, inputSequenceLengthsTeplicatedDims);
    assert(actualInputSequenceLengthsReplicatedTensor != nullptr);
    nvinfer1::Dims memoryStatesDims{2, {gMaxInputSequenceLength, alignment->getSourceStatesSize()}, {nvinfer1::DimensionType::kSEQUENCE, nvinfer1::DimensionType::kCHANNEL}};
    auto memoryStatesTensor = generatorNetwork->addInput("memory_states", gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, memoryStatesDims);
    assert(memoryStatesTensor != nullptr);
    nvinfer1::ITensor* attentionKeysTensor = nullptr;
    if (alignment->getAttentionKeySize() > 0)
    {
        nvinfer1::Dims attentionKeysDims{2, {gMaxInputSequenceLength, alignment->getAttentionKeySize()}, {nvinfer1::DimensionType::kSEQUENCE, nvinfer1::DimensionType::kCHANNEL}};
        attentionKeysTensor = generatorNetwork->addInput("attention_keys", gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, attentionKeysDims);
        assert(attentionKeysTensor != nullptr);
    }
    nvinfer1::ITensor* inputAttentionTensor = nullptr;
    if (gFeedAttentionToInput)
    {
        nvinfer1::Dims inputAttentionDims{2, {gBeamWidth, attention->getAttentionSize()}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
        inputAttentionTensor = generatorNetwork->addInput("input_attention", gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, inputAttentionDims);
        assert(inputAttentionTensor != nullptr);
    }
    nvinfer1::Dims inputLikelihoodsDims{2, {gBeamWidth, 1}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
    auto inputLikelihoodsTensor = generatorNetwork->addInput("input_likelihoods", nvinfer1::DataType::kFLOAT, inputLikelihoodsDims);
    assert(inputLikelihoodsTensor != nullptr);
    nvinfer1::Dims inputLikelihoodsReplicateIndicesDims{1, {gBeamWidth}, {nvinfer1::DimensionType::kCHANNEL}};
    auto inputLikelihoodsReplicateIndicesTensor = generatorNetwork->addInput("replicate_likelihoods_indices", nvinfer1::DataType::kINT32, inputLikelihoodsReplicateIndicesDims);
    assert(inputLikelihoodsReplicateIndicesTensor != nullptr);

    // Add output embedder
    nvinfer1::ITensor* inputDecoderEmbeddedTensor;
    outputEmbedder->addToModel(generatorNetwork, inputDecoderDataTensor, &inputDecoderEmbeddedTensor);
    assert(inputDecoderEmbeddedTensor != nullptr);

    // Add concatination of previous attention vector and embedded input for the decoder
    nvinfer1::ITensor* inputDecoderEmbeddedConcatinatedWithAttentionTensor;
    if (gFeedAttentionToInput)
    {
        nvinfer1::ITensor* inputTensors[] = {inputDecoderEmbeddedTensor, inputAttentionTensor};
        auto concatLayer = generatorNetwork->addConcatenation(inputTensors, 2);
        assert(concatLayer != nullptr);
        concatLayer->setName("Concatenate embedded input and attention");
        concatLayer->setAxis(1);
        inputDecoderEmbeddedConcatinatedWithAttentionTensor = concatLayer->getOutput(0);
        assert(inputDecoderEmbeddedConcatinatedWithAttentionTensor != nullptr);
    }

    // Add decoder (single timestep)
    nvinfer1::ITensor* outputDecoderDataTensor;
    std::vector<nvinfer1::ITensor*> decoderOutputStatesTensors(stateSizes.size());
    decoder->addToModel(generatorNetwork, gFeedAttentionToInput ? inputDecoderEmbeddedConcatinatedWithAttentionTensor : inputDecoderEmbeddedTensor,
                        &decoderInputStatesTensors[0], &outputDecoderDataTensor, &decoderOutputStatesTensors[0]);
    for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "output_decoder_states_" << i;
        decoderOutputStatesTensors[i]->setName(ss.str().c_str());
        generatorNetwork->markOutput(*decoderOutputStatesTensors[i]);
        decoderOutputStatesTensors[i]->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    // Add alignment scores
    nvinfer1::ITensor* alignmentScoresTensor;
    alignment->addToModel(
        generatorNetwork,
        (alignment->getAttentionKeySize() > 0) ? attentionKeysTensor : memoryStatesTensor,
        outputDecoderDataTensor,
        &alignmentScoresTensor);

    // Add context
    nvinfer1::ITensor* contextTensor;
    context->addToModel(generatorNetwork, actualInputSequenceLengthsReplicatedTensor, memoryStatesTensor, alignmentScoresTensor, &contextTensor);

    // Add attention
    nvinfer1::ITensor* attentionTensor;
    attention->addToModel(generatorNetwork, outputDecoderDataTensor, contextTensor, &attentionTensor);
    if (gFeedAttentionToInput)
    {
        attentionTensor->setName("output_attention");
        generatorNetwork->markOutput(*attentionTensor);
        attentionTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    // Add projection
    nvinfer1::ITensor* logitsTensor;
    projection->addToModel(generatorNetwork, attentionTensor, &logitsTensor);

    // Replicate input likelihoods across all TopK options
    auto gatherLayer = generatorNetwork->addGather(*inputLikelihoodsTensor, *inputLikelihoodsReplicateIndicesTensor, 1);
    assert(gatherLayer != nullptr);
    gatherLayer->setName("Replicate beam likelihoods");
    auto inputLikelihoodsReplicatedTensor = gatherLayer->getOutput(0);
    assert(inputLikelihoodsReplicatedTensor != nullptr);

    // Add per-ray top-k options generation
    nvinfer1::ITensor* outputCombinedLikelihoodsTensor;
    nvinfer1::ITensor* outputRayOptionIndicesTensor;
    nvinfer1::ITensor* outputVocabularyIndicesTensor;
    likelihood->addToModel(generatorNetwork, gBeamWidth, logitsTensor, inputLikelihoodsReplicatedTensor, &outputCombinedLikelihoodsTensor, &outputRayOptionIndicesTensor, &outputVocabularyIndicesTensor);
    outputCombinedLikelihoodsTensor->setName("output_combined_likelihoods");
    generatorNetwork->markOutput(*outputCombinedLikelihoodsTensor);
    outputRayOptionIndicesTensor->setName("output_ray_option_indices");
    generatorNetwork->markOutput(*outputRayOptionIndicesTensor);
    outputRayOptionIndicesTensor->setType(nvinfer1::DataType::kINT32);
    outputVocabularyIndicesTensor->setName("output_vocabulary_indices");
    generatorNetwork->markOutput(*outputVocabularyIndicesTensor);
    outputVocabularyIndicesTensor->setType(nvinfer1::DataType::kINT32);

    auto res = generatorBuilder->buildCudaEngine(*generatorNetwork);
    generatorNetwork->destroy();
    generatorBuilder->destroy();
    return res;
}

nvinfer1::ICudaEngine* getGeneratorShuffleEngine(const std::vector<nvinfer1::Dims>& decoderStateSizes, int attentionSize)
{
    nvinfer1::IBuilder* shuffleBuilder = nvinfer1::createInferBuilder(gLogger);
    shuffleBuilder->setMaxBatchSize(gMaxBatchSize);
    shuffleBuilder->setMaxWorkspaceSize(gMaxWorkspaceSize);
    shuffleBuilder->setFp16Mode(gFp16);

    nvinfer1::INetworkDefinition* shuffleNetwork = shuffleBuilder->createNetwork();

    nvinfer1::Dims sourceRayIndicesDims{1, {gBeamWidth}, {nvinfer1::DimensionType::kINDEX}};
    auto sourceRayIndicesTensor = shuffleNetwork->addInput("source_ray_indices", nvinfer1::DataType::kINT32, sourceRayIndicesDims);
    assert(sourceRayIndicesTensor != nullptr);

    std::vector<nvinfer1::ITensor*> previousOutputDecoderStatesTensors(decoderStateSizes.size());
    for (int i = 0; i < static_cast<int>(decoderStateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "previous_output_decoder_states_" << i;
        nvinfer1::Dims statesDims;
        {
            statesDims.nbDims = decoderStateSizes[i].nbDims + 1;
            statesDims.d[0] = gBeamWidth;
            statesDims.type[0] = nvinfer1::DimensionType::kINDEX;
            for (int j = 0; j < decoderStateSizes[i].nbDims; ++j)
            {
                statesDims.d[j + 1] = decoderStateSizes[i].d[j];
                statesDims.type[j + 1] = decoderStateSizes[i].type[j];
            }
        }
        previousOutputDecoderStatesTensors[i] = shuffleNetwork->addInput(ss.str().c_str(), gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, statesDims);
        assert(previousOutputDecoderStatesTensors[i] != nullptr);
    }

    nvinfer1::ITensor* previousOutputAttentionTensor = nullptr;
    if (gFeedAttentionToInput)
    {
        nvinfer1::Dims previousOutputAttentionDims{2, {gBeamWidth, attentionSize}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
        previousOutputAttentionTensor = shuffleNetwork->addInput("previous_output_attention", gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, previousOutputAttentionDims);
        assert(previousOutputAttentionTensor != nullptr);
    }

    for (int i = 0; i < static_cast<int>(decoderStateSizes.size()); ++i)
    {
        auto gatherLayer = shuffleNetwork->addGather(*previousOutputDecoderStatesTensors[i], *sourceRayIndicesTensor, 0);
        assert(gatherLayer != nullptr);
        {
            std::stringstream ss;
            ss << "Shuffle decoder states " << i;
            gatherLayer->setName(ss.str().c_str());
        }
        auto inputDecoderHiddenStatesTensor = gatherLayer->getOutput(0);
        assert(inputDecoderHiddenStatesTensor != nullptr);
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        inputDecoderHiddenStatesTensor->setName(ss.str().c_str());
        shuffleNetwork->markOutput(*inputDecoderHiddenStatesTensor);
        inputDecoderHiddenStatesTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    if (gFeedAttentionToInput)
    {
        auto gatherLayer = shuffleNetwork->addGather(*previousOutputAttentionTensor, *sourceRayIndicesTensor, 0);
        assert(gatherLayer != nullptr);
        gatherLayer->setName("Shuffle attention");
        auto inputAttentionTensor = gatherLayer->getOutput(0);
        assert(inputAttentionTensor != nullptr);
        inputAttentionTensor->setName("input_attention");
        shuffleNetwork->markOutput(*inputAttentionTensor);
        inputAttentionTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    auto res = shuffleBuilder->buildCudaEngine(*shuffleNetwork);
    shuffleNetwork->destroy();
    shuffleBuilder->destroy();
    return res;
}

int main(int argc, char** argv)
{
    if (!parseArgs(argc, argv))
        return 1;

    gLogger = Logger(gVerbose ? nvinfer1::ILogger::Severity::kINFO : nvinfer1::ILogger::Severity::kWARNING);

    // Set up output vocabulary
    {
        std::string vocabularyFilePath = gOutputVocabularyFileName;
        std::ifstream vocabStream(locateNMTFile(vocabularyFilePath));
        if (!vocabStream.good())
        {
            printf("Cannot open file %s\n", vocabularyFilePath.c_str());
            return 1;
        }
        vocabStream >> *gOutputVocabulary;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto outputSequenceProperties = getOutputSequenceProperties();
    auto dataReader = getDataReader();
    auto inputEmbedder = getInputEmbedder();
    auto outputEmbedder = getOutputEmbedder();
    auto encoder = getEncoder();
    auto decoder = getDecoder();
    auto alignment = getAlignment();
    auto context = getContext();
    auto attention = getAttention();
    auto projection = getProjection();
    auto likelihood = getLikelihood();
    auto searchPolicy = getSearchPolicy(outputSequenceProperties->getEndSequenceId(), likelihood->getLikelihoodCombinationOperator());
    auto dataWriter = getDataWriter();

    if (gPrintComponentInfo)
    {
        std::cout << "Component Info:" << std::endl;
        std::cout << "- Data Reader: " << dataReader->getInfo() << std::endl;
        std::cout << "- Input Embedder: " << inputEmbedder->getInfo() << std::endl;
        std::cout << "- Output Embedder: " << outputEmbedder->getInfo() << std::endl;
        std::cout << "- Encoder: " << encoder->getInfo() << std::endl;
        std::cout << "- Decoder: " << decoder->getInfo() << std::endl;
        std::cout << "- Alignment: " << alignment->getInfo() << std::endl;
        std::cout << "- Context: " << context->getInfo() << std::endl;
        std::cout << "- Attention: " << attention->getInfo() << std::endl;
        std::cout << "- Projection: " << projection->getInfo() << std::endl;
        std::cout << "- Likelihood: " << likelihood->getInfo() << std::endl;
        std::cout << "- Search Policy: " << searchPolicy->getInfo() << std::endl;
        std::cout << "- Data Writer: " << dataWriter->getInfo() << std::endl;
        std::cout << "End of Component Info" << std::endl;
    }

    std::vector<nvinfer1::Dims> stateSizes = decoder->getStateSizes();

    // A number of consistency checks between components
    assert(alignment->getSourceStatesSize() == encoder->getMemoryStatesSize());
    if (gInitializeDecoderFromEncoderHiddenStates)
    {
        std::vector<nvinfer1::Dims> encoderStateSizes = encoder->getStateSizes();
        assert(stateSizes.size() == encoderStateSizes.size());
        for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
            assert(nmtSample::getVolume(stateSizes[i]) == nmtSample::getVolume(encoderStateSizes[i]));
    }
    assert(projection->getOutputSize() == outputEmbedder->getInputDimensionSize());

    auto inputOriginalHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gMaxInputSequenceLength);
    auto inputHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gMaxInputSequenceLength);
    auto inputOriginalSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize);
    auto inputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize);
    auto maxOutputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize);
    auto outputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize);
    auto outputCombinedLikelihoodHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<float>>(gMaxBatchSize * gBeamWidth);
    auto outputVocabularyIndicesHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto outputRayOptionIndicesHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto sourceRayIndicesHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto sourceLikelihoodsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<float>>(gMaxBatchSize * gBeamWidth);

    // Allocated buffers on GPU to be used as inputs and outputs for TenorRT
    auto inputEncoderDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gMaxInputSequenceLength);
    auto inputSequenceLengthsDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize);
    auto inputSequenceLengthsReplicatedDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto memoryStatesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gMaxInputSequenceLength * encoder->getMemoryStatesSize());
    auto attentionKeysDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>((alignment->getAttentionKeySize() > 0) ? gMaxBatchSize * gMaxInputSequenceLength * alignment->getAttentionKeySize() : 0);
    std::vector<nmtSample::DeviceBuffer<float>::ptr> encoderStatesLastTimestepDeviceBuffers;
    for (auto stateSize : stateSizes)
        encoderStatesLastTimestepDeviceBuffers.push_back(std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * nmtSample::getVolume(stateSize)));
    std::vector<nmtSample::DeviceBuffer<float>::ptr> inputDecoderStatesDeviceBuffers;
    for (auto stateSize : stateSizes)
        inputDecoderStatesDeviceBuffers.push_back(std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth * nmtSample::getVolume(stateSize)));
    std::vector<nmtSample::DeviceBuffer<float>::ptr> outputDecoderStatesDeviceBuffers;
    for (auto stateSize : stateSizes)
        outputDecoderStatesDeviceBuffers.push_back(std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth * nmtSample::getVolume(stateSize)));
    auto inputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    auto outputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    auto outputCombinedLikelihoodDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth);
    auto outputRayOptionIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto sourceRayIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto inputDecoderDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto inputLikelihoodsDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth);

    std::vector<nmtSample::DeviceBuffer<float>::ptr> zeroInputEncoderStatesDeviceBuffers;
    for (auto stateSize : stateSizes)
    {
        auto buf = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * nmtSample::getVolume(stateSize));
        CUDA_CHECK(cudaMemsetAsync(*buf, 0, gMaxBatchSize * nmtSample::getVolume(stateSize) * sizeof(float), stream));
        zeroInputEncoderStatesDeviceBuffers.push_back(buf);
    }

    std::vector<nmtSample::DeviceBuffer<float>::ptr> zeroInputDecoderStatesDeviceBuffers;
    for (auto stateSize : stateSizes)
    {
        auto buf = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth * nmtSample::getVolume(stateSize));
        CUDA_CHECK(cudaMemsetAsync(*buf, 0, gMaxBatchSize * gBeamWidth * nmtSample::getVolume(stateSize) * sizeof(float), stream));
        zeroInputDecoderStatesDeviceBuffers.push_back(buf);
    }

    auto zeroInputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    if (gFeedAttentionToInput)
    {
        CUDA_CHECK(cudaMemsetAsync(*zeroInputAttentionDeviceBuffer, 0, gMaxBatchSize * gBeamWidth * attention->getAttentionSize() * sizeof(float), stream));
    }
    auto startSeqInputDecoderDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    {
        auto startSeqInputDecoderHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gBeamWidth);
        std::fill_n((int*) *startSeqInputDecoderHostBuffer, gMaxBatchSize * gBeamWidth, outputSequenceProperties->getStartSequenceId());
        CUDA_CHECK(cudaMemcpyAsync(*startSeqInputDecoderDeviceBuffer, *startSeqInputDecoderHostBuffer, gMaxBatchSize * gBeamWidth * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto zeroInitializeDecoderIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    CUDA_CHECK(cudaMemsetAsync(*zeroInitializeDecoderIndicesDeviceBuffer, 0, gMaxBatchSize * gBeamWidth * sizeof(int), stream));
    auto initialInputLikelihoodsDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth);
    {
        auto likelihoodCombinationOperator = likelihood->getLikelihoodCombinationOperator();
        auto initialInputLikelihoodsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<float>>(gMaxBatchSize * gBeamWidth);
        for (int sampleId = 0; sampleId < gMaxBatchSize; ++sampleId)
        {
            (*initialInputLikelihoodsHostBuffer)[sampleId * gBeamWidth] = likelihoodCombinationOperator->init();
            for (int rayId = 1; rayId < gBeamWidth; ++rayId)
                (*initialInputLikelihoodsHostBuffer)[sampleId * gBeamWidth + rayId] = likelihoodCombinationOperator->smallerThanMinimalLikelihood();
        }
        CUDA_CHECK(cudaMemcpyAsync(*initialInputLikelihoodsDeviceBuffer, *initialInputLikelihoodsHostBuffer, gMaxBatchSize * gBeamWidth * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto zeroReplicateLikelihoodsIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    CUDA_CHECK(cudaMemsetAsync(*zeroReplicateLikelihoodsIndicesDeviceBuffer, 0, gMaxBatchSize * gBeamWidth * sizeof(int), stream));

    // Create TensorRT engines
    nvinfer1::ICudaEngine* encoderEngine = getEncoderEngine(inputEmbedder, encoder, alignment);
    nvinfer1::ICudaEngine* generatorEngine = getGeneratorEngine(outputEmbedder, decoder, alignment, context, attention, projection, likelihood);
    nvinfer1::ICudaEngine* generatorShuffleEngine = getGeneratorShuffleEngine(decoder->getStateSizes(), attention->getAttentionSize());

    // Setup TensorRT bindings
    std::vector<void*> encoderBindings(encoderEngine->getNbBindings());
    encoderBindings[encoderEngine->getBindingIndex("input_encoder_data")] = (int*) (*inputEncoderDeviceBuffer);
    encoderBindings[encoderEngine->getBindingIndex("actual_input_sequence_lengths")] = (int*) (*inputSequenceLengthsDeviceBuffer);
    encoderBindings[encoderEngine->getBindingIndex("actual_input_sequence_lengths_with_index_dim")] = (int*) (*inputSequenceLengthsDeviceBuffer);
    encoderBindings[encoderEngine->getBindingIndex("actual_input_sequence_lengths_replicated")] = (int*) (*inputSequenceLengthsReplicatedDeviceBuffer);
    encoderBindings[encoderEngine->getBindingIndex("initialize_decoder_indices")] = (int*) (*zeroInitializeDecoderIndicesDeviceBuffer);
    for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_encoder_states_" << i;
        encoderBindings[encoderEngine->getBindingIndex(ss.str().c_str())] = (float*) (*zeroInputEncoderStatesDeviceBuffers[i]);
    }
    encoderBindings[encoderEngine->getBindingIndex("memory_states")] = (float*) (*memoryStatesDeviceBuffer);
    if (alignment->getAttentionKeySize() > 0)
    {
        encoderBindings[encoderEngine->getBindingIndex("attention_keys")] = (float*) (*attentionKeysDeviceBuffer);
    }
    if (gInitializeDecoderFromEncoderHiddenStates)
    {
        for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
        {
            std::stringstream ss;
            ss << "input_decoder_states_" << i;
            encoderBindings[encoderEngine->getBindingIndex(ss.str().c_str())] = (float*) (*inputDecoderStatesDeviceBuffers[i]);
        }
    }

    std::vector<void*> generatorBindings(generatorEngine->getNbBindings());
    generatorBindings[generatorEngine->getBindingIndex("input_decoder_data")] = (int*) (*inputDecoderDeviceBuffer);
    for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        generatorBindings[generatorEngine->getBindingIndex(ss.str().c_str())] = (float*) (*inputDecoderStatesDeviceBuffers[i]);
    }
    generatorBindings[generatorEngine->getBindingIndex("actual_input_sequence_lengths_replicated")] = (int*) (*inputSequenceLengthsReplicatedDeviceBuffer);
    generatorBindings[generatorEngine->getBindingIndex("memory_states")] = (float*) (*memoryStatesDeviceBuffer);
    if (alignment->getAttentionKeySize() > 0)
    {
        generatorBindings[generatorEngine->getBindingIndex("attention_keys")] = (float*) (*attentionKeysDeviceBuffer);
    }
    for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "output_decoder_states_" << i;
        generatorBindings[generatorEngine->getBindingIndex(ss.str().c_str())] = (float*) (*outputDecoderStatesDeviceBuffers[i]);
    }
    generatorBindings[generatorEngine->getBindingIndex("output_combined_likelihoods")] = (float*) (*outputCombinedLikelihoodDeviceBuffer);
    generatorBindings[generatorEngine->getBindingIndex("output_vocabulary_indices")] = (int*) (*inputDecoderDeviceBuffer);
    generatorBindings[generatorEngine->getBindingIndex("output_ray_option_indices")] = (int*) (*outputRayOptionIndicesDeviceBuffer);
    if (gFeedAttentionToInput)
    {
        generatorBindings[generatorEngine->getBindingIndex("input_attention")] = (float*) (*inputAttentionDeviceBuffer);
        generatorBindings[generatorEngine->getBindingIndex("output_attention")] = (float*) (*outputAttentionDeviceBuffer);
    }
    generatorBindings[generatorEngine->getBindingIndex("input_likelihoods")] = (float*) (*inputLikelihoodsDeviceBuffer);
    generatorBindings[generatorEngine->getBindingIndex("replicate_likelihoods_indices")] = (int*) (*zeroReplicateLikelihoodsIndicesDeviceBuffer);

    std::vector<void*> generatorBindingsFirstStep = generatorBindings;
    generatorBindingsFirstStep[generatorEngine->getBindingIndex("input_decoder_data")] = (int*) (*startSeqInputDecoderDeviceBuffer);
    if (!gInitializeDecoderFromEncoderHiddenStates)
    {
        for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
        {
            std::stringstream ss;
            ss << "input_decoder_states_" << i;
            generatorBindingsFirstStep[generatorEngine->getBindingIndex(ss.str().c_str())] = (float*) (*zeroInputDecoderStatesDeviceBuffers[i]);
        }
    }
    if (gFeedAttentionToInput)
    {
        generatorBindingsFirstStep[generatorEngine->getBindingIndex("input_attention")] = (float*) (*zeroInputAttentionDeviceBuffer);
    }
    generatorBindingsFirstStep[generatorEngine->getBindingIndex("input_likelihoods")] = (float*) (*initialInputLikelihoodsDeviceBuffer);

    std::vector<void*> generatorShuffleBindings(generatorShuffleEngine->getNbBindings());
    generatorShuffleBindings[generatorShuffleEngine->getBindingIndex("source_ray_indices")] = (int*) (*sourceRayIndicesDeviceBuffer);
    for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "previous_output_decoder_states_" << i;
        generatorShuffleBindings[generatorShuffleEngine->getBindingIndex(ss.str().c_str())] = (float*) (*outputDecoderStatesDeviceBuffers[i]);
    }
    for (int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        generatorShuffleBindings[generatorShuffleEngine->getBindingIndex(ss.str().c_str())] = (float*) (*inputDecoderStatesDeviceBuffers[i]);
    }
    if (gFeedAttentionToInput)
    {
        generatorShuffleBindings[generatorShuffleEngine->getBindingIndex("previous_output_attention")] = (float*) (*outputAttentionDeviceBuffer);
        generatorShuffleBindings[generatorShuffleEngine->getBindingIndex("input_attention")] = (float*) (*inputAttentionDeviceBuffer);
    }

    // Create Tensor RT contexts
    nvinfer1::IExecutionContext* encoderContext = encoderEngine->createExecutionContext();
    nvinfer1::IExecutionContext* generatorContext = generatorEngine->createExecutionContext();
    nvinfer1::IExecutionContext* generatorShuffleContext = generatorShuffleEngine->createExecutionContext();

    std::vector<SimpleProfiler> profilers;
    if (gEnableProfiling)
    {
        profilers.push_back(SimpleProfiler("Host"));
        profilers.push_back(SimpleProfiler("Encoder"));
        profilers.push_back(SimpleProfiler("Decoder"));
        profilers.push_back(SimpleProfiler("Beam shuffle"));
        encoderContext->setProfiler(&profilers[1]);
        generatorContext->setProfiler(&profilers[2]);
        generatorShuffleContext->setProfiler(&profilers[3]);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    dataWriter->initialize();

    std::vector<int> outputHostBuffer;
    auto startDataRead = std::chrono::high_resolution_clock::now();
    int inputSamplesRead = dataReader->read(gMaxBatchSize, gMaxInputSequenceLength, *inputOriginalHostBuffer, *inputOriginalSequenceLengthsHostBuffer);
    if (gEnableProfiling)
        profilers[0].reportLayerTime("Data Read", std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startDataRead).count());
    // Outer loop over batches of samples
    auto startLatency = std::chrono::high_resolution_clock::now();
    int batchCount = 0;
    while (inputSamplesRead > 0)
    {
        ++batchCount;

        // Sort input sequences in the batch in the order of decreasing length
        // The idea is that shorter input sequences gets translated faster so we can reduce batch size quickly for the generator
        auto startBatchSort = std::chrono::high_resolution_clock::now();
        std::vector<int> samplePositions(inputSamplesRead);
        {
            std::vector<std::pair<int, int>> sequenceSampleIdAndLength(inputSamplesRead);
            for (int sampleId = 0; sampleId < inputSamplesRead; ++sampleId)
                sequenceSampleIdAndLength[sampleId] = std::make_pair(sampleId, ((const int*) *inputOriginalSequenceLengthsHostBuffer)[sampleId]);
            std::sort(
                sequenceSampleIdAndLength.begin(),
                sequenceSampleIdAndLength.end(),
                [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool { return a.second > b.second; });
            for (int position = 0; position < inputSamplesRead; ++position)
            {
                int sampleId = sequenceSampleIdAndLength[position].first;
                ((int*) *inputSequenceLengthsHostBuffer)[position] = ((const int*) *inputOriginalSequenceLengthsHostBuffer)[sampleId];
                std::copy_n(
                    ((const int*) *inputOriginalHostBuffer) + sampleId * gMaxInputSequenceLength,
                    gMaxInputSequenceLength,
                    ((int*) *inputHostBuffer) + position * gMaxInputSequenceLength);
                samplePositions[sampleId] = position;
            }
        }
        if (gEnableProfiling)
            profilers[0].reportLayerTime("Intra-batch Sort", std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startBatchSort).count());

        CUDA_CHECK(cudaMemcpyAsync(*inputEncoderDeviceBuffer, *inputHostBuffer, inputSamplesRead * gMaxInputSequenceLength * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(*inputSequenceLengthsDeviceBuffer, *inputSequenceLengthsHostBuffer, inputSamplesRead * sizeof(int), cudaMemcpyHostToDevice, stream));

        // Overlap host and device: Read data for the next batch while encode for this one is running
        std::future<int> nextInputSamplesReadFuture = std::async(
            std::launch::async,
            [&]() { return dataReader->read(gMaxBatchSize, gMaxInputSequenceLength, *inputOriginalHostBuffer, *inputOriginalSequenceLengthsHostBuffer); });

        encoderContext->enqueue(inputSamplesRead, &encoderBindings[0], stream, nullptr);

        // Limit output sequences length to input_sequence_length * 2
        std::transform((const int*) *inputSequenceLengthsHostBuffer, (const int*) *inputSequenceLengthsHostBuffer + inputSamplesRead, (int*) *maxOutputSequenceLengthsHostBuffer, [](int i) {int r = i * 2; if (gMaxOutputSequenceLength >= 0) r = std::min(r, gMaxOutputSequenceLength); return r; });
        searchPolicy->initialize(inputSamplesRead, *maxOutputSequenceLengthsHostBuffer);
        int batchMaxOutputSequenceLength = *std::max_element((int*) *maxOutputSequenceLengthsHostBuffer, (int*) *maxOutputSequenceLengthsHostBuffer + inputSamplesRead);
        outputHostBuffer.resize(gMaxBatchSize * batchMaxOutputSequenceLength);

        // Inner loop over generator timesteps
        int validSampleCount = searchPolicy->getTailWithNoWorkRemaining();
        for (int outputTimestep = 0; (outputTimestep < batchMaxOutputSequenceLength) && (validSampleCount > 0); ++outputTimestep)
        {
            // Generator initialization and beam shuffling
            if (outputTimestep == 0)
            {
                generatorContext->enqueue(validSampleCount, &generatorBindingsFirstStep[0], stream, nullptr);
            }
            else
            {
                generatorShuffleContext->enqueue(validSampleCount, &generatorShuffleBindings[0], stream, nullptr);
                generatorContext->enqueue(validSampleCount, &generatorBindings[0], stream, nullptr);
            }

            CUDA_CHECK(cudaMemcpyAsync(*outputCombinedLikelihoodHostBuffer, *outputCombinedLikelihoodDeviceBuffer, validSampleCount * gBeamWidth * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(*outputVocabularyIndicesHostBuffer, *inputDecoderDeviceBuffer, validSampleCount * gBeamWidth * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(*outputRayOptionIndicesHostBuffer, *outputRayOptionIndicesDeviceBuffer, validSampleCount * gBeamWidth * sizeof(int), cudaMemcpyDeviceToHost, stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto startBeamSearch = std::chrono::high_resolution_clock::now();
            searchPolicy->processTimestep(
                validSampleCount,
                *outputCombinedLikelihoodHostBuffer,
                *outputVocabularyIndicesHostBuffer,
                *outputRayOptionIndicesHostBuffer,
                *sourceRayIndicesHostBuffer,
                *sourceLikelihoodsHostBuffer);
            if (gEnableProfiling)
                profilers[0].reportLayerTime("Beam Search", std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startBeamSearch).count());

            CUDA_CHECK(cudaMemcpyAsync(*sourceRayIndicesDeviceBuffer, *sourceRayIndicesHostBuffer, validSampleCount * gBeamWidth * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(*inputLikelihoodsDeviceBuffer, *sourceLikelihoodsHostBuffer, validSampleCount * gBeamWidth * sizeof(float), cudaMemcpyHostToDevice, stream));

            validSampleCount = searchPolicy->getTailWithNoWorkRemaining();
        } // for(int outputTimestep

        auto startBacktrack = std::chrono::high_resolution_clock::now();
        searchPolicy->readGeneratedResult(
            inputSamplesRead,
            batchMaxOutputSequenceLength,
            &outputHostBuffer[0],
            *outputSequenceLengthsHostBuffer);
        if (gEnableProfiling)
            profilers[0].reportLayerTime("Read Result", std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startBacktrack).count());

        auto startDataWrite = std::chrono::high_resolution_clock::now();
        for (int sampleId = 0; sampleId < inputSamplesRead; ++sampleId)
        {
            int position = samplePositions[sampleId];
            dataWriter->write(
                &outputHostBuffer[0] + position * batchMaxOutputSequenceLength,
                ((const int*) *outputSequenceLengthsHostBuffer)[position],
                ((const int*) *inputSequenceLengthsHostBuffer)[position]);
        }
        if (gEnableProfiling)
            profilers[0].reportLayerTime("Data Write", std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startDataWrite).count());

        auto startDataRead = std::chrono::high_resolution_clock::now();
        inputSamplesRead = nextInputSamplesReadFuture.get();
        if (gEnableProfiling)
            profilers[0].reportLayerTime("Data Read", std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startDataRead).count());
    }
    float totalLatency = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startLatency).count();

    dataWriter->finalize();

    if (gDataWriterStr == "benchmark")
    {
        std::cout << "Average latency (without data read) = " << totalLatency / static_cast<float>(batchCount) << " ms" << std::endl;
    }

    if (gEnableProfiling)
    {
        if (gAggregateProfiling)
        {
            SimpleProfiler aggregateProfiler("Aggregate", profilers);
            std::cout << aggregateProfiler;
        }
        else
        {
            for (const auto& profiler : profilers)
                std::cout << profiler;
        }
    }

    encoderContext->destroy();
    generatorContext->destroy();
    generatorShuffleContext->destroy();

    encoderEngine->destroy();
    generatorEngine->destroy();
    generatorShuffleEngine->destroy();

    cudaStreamDestroy(stream);

    return 0;
}
