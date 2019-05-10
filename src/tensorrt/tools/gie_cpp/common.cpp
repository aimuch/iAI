#include "common.h"
std::string locateFile(const std::string& input, const std::vector<std::string> & directories)
{
    std::string file;
	const int MAX_DEPTH{10};
    bool found{false};
    for (auto &dir : directories)
    {
        file = dir + input;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found) break;
            file = "../" + file;
        }
        if (found) break;
        file.clear();
    }

    assert(!file.empty() && "Could not find a file due to it not existing in the data directory.");
    return file;
}
