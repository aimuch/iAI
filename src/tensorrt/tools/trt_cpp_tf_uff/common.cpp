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

void readPGMFile(const std::string& fileName,  uint8_t *buffer, int inH, int inW, int inC)
{
	std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
	std::string magic, h, w, c, max;
	infile >> magic >> h >> w >> c >> max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(buffer), inH*inW*inC);
}
