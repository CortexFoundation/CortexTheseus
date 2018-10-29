#ifndef TOOL_H
#define TOOL_H
/*#include <platform.h>
#include <device.h>
#include <context.h>
#include <commandQueue.h>
#include <program.h>
*/
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

/** convert the kernel file into a string */
inline int convertToString(const char *filename, std::string& s, size_t &fileSize)
{
    size_t size;
    char*  str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if(!str)
        {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
	fileSize = size;
        s = str;
        delete[] str;
        return 0;
    }
    std::cerr<<"Error: failed to open file\n:"<<filename<<std::endl;
    return -1;
}
#endif
