#ifndef SAMPLE_NMT_COMPONENT_
#define SAMPLE_NMT_COMPONENT_

#include <memory>
#include <string>

namespace nmtSample
{
/** \class Component
    *
    * \brief a functional part of the sample 
    *
    */
class Component
{
public:
    typedef std::shared_ptr<Component> ptr;

    /**
        * \brief get the textual description of the component
        */
    virtual std::string getInfo() = 0;

protected:
    Component() = default;

    virtual ~Component() = default;
};
}

#endif // SAMPLE_NMT_COMPONENT_
