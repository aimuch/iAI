#ifndef SAMPLE_NMT_BLEU_SCORE_WRITER_
#define SAMPLE_NMT_BLEU_SCORE_WRITER_

#include <istream>
#include <memory>
#include <vector>

#include "dataWriter.h"
#include "vocabulary.h"

namespace nmtSample
{
/** \class BLEUScoreWriter
    *
    * \brief all it does is to evaluate BLEU score
    *
    */
class BLEUScoreWriter : public DataWriter
{
public:
    BLEUScoreWriter(std::shared_ptr<std::istream> referenceTextInput,
                    Vocabulary::ptr vocabulary,
                    int maxOrder = 4);

    void write(
        const int* hOutputData,
        int actualOutputSequenceLength,
        int actualInputSequenceLength) override;

    void initialize() override;

    void finalize() override;

    std::string getInfo() override;

    float getScore() const;

    ~BLEUScoreWriter() override = default;

private:
    std::shared_ptr<std::istream> mReferenceInput;
    Vocabulary::ptr mVocabulary;
    size_t mReferenceLength;
    size_t mTranslationLength;
    int mMaxOrder;
    bool mSmooth;
    std::vector<size_t> mMatchesByOrder;
    std::vector<size_t> mPossibleMatchesByOrder;
};
}

#endif // SAMPLE_NMT_BLEU_SCORE_WRITER_
