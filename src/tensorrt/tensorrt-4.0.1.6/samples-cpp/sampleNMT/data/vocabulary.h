#ifndef SAMPLE_NMT_VOCABULARY_
#define SAMPLE_NMT_VOCABULARY_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "sequenceProperties.h"

namespace nmtSample
{
/** \class Vocabulary
    *
    * \brief String<->Id bijection storage
    *
    */
class Vocabulary : public SequenceProperties
{
public:
    typedef std::shared_ptr<Vocabulary> ptr;

    Vocabulary();

    friend std::istream& operator>>(std::istream& input, Vocabulary& value);

    /**
        * \brief add new token to vocabulary, ID is auto-generated
        */
    void add(const std::string& token);

    /**
        * \brief get the ID of the token
        */
    int getId(const std::string& token) const;

    /**
        * \brief get token by ID
        */
    std::string getToken(int id) const;

    /**
        * \brief get the number of elements in the vocabulary
        */
    int getSize() const;

    int getStartSequenceId() override;

    int getEndSequenceId() override;

private:
    static const std::string mSosStr;
    static const std::string mUnkStr;
    static const std::string mEosStr;

    std::map<std::string, int> mTokenToId;
    std::vector<std::string> mIdToToken;
    int mNumTokens;

    int mSosId;
    int mEosId;
    int mUnkId;
};
}

#endif // SAMPLE_NMT_VOCABULARY_
