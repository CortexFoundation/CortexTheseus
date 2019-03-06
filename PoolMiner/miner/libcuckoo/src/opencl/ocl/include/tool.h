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

inline const char* openclGetErrorString(cl_int clResult){
	switch(clResult){
		case CL_SUCCESS					:return "CL_SUCCESS";
		case CL_DEVICE_NOT_FOUND			:return "CL_DEVICE_NOT_FOUND"; 
		case CL_DEVICE_NOT_AVAILABLE			:return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE 			:return "CL_COMPILER_NOT_AVAILABLE";  		 
		case CL_MEM_OBJECT_ALLOCATION_FAILURE           :return "CL_MEM_OBJECT_ALLOCATION_FAILURE";  
		case CL_OUT_OF_RESOURCES                        :return "CL_OUT_OF_RESOURCES";  
		case CL_OUT_OF_HOST_MEMORY                      :return  "CL_OUT_OF_HOST_MEMORY";  
		case CL_PROFILING_INFO_NOT_AVAILABLE            :return "CL_PROFILING_INFO_NOT_AVAILABLE";  
		case CL_MEM_COPY_OVERLAP                        :return "CL_MEM_COPY_OVERLAP";  
		case CL_IMAGE_FORMAT_MISMATCH                   :return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED              :return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE                    :return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE                              :return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET             :return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_COMPILE_PROGRAM_FAILURE                  :return "CL_COMPILE_PROGRAM_FAILURE";  
		case CL_LINKER_NOT_AVAILABLE                     :return "CL_LINKER_NOT_AVAILABLE";  
		case CL_LINK_PROGRAM_FAILURE                     :return "CL_LINK_PROGRAM_FAILURE";  
		case CL_DEVICE_PARTITION_FAILED                  :return "CL_DEVICE_PARTITION_FAILED";  
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE            :return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";  
		case CL_INVALID_VALUE                            : return "CL_INVALID_VALUE";  
		case CL_INVALID_DEVICE_TYPE                      :return "CL_INVALID_DEVICE_TYPE";  
		case CL_INVALID_PLATFORM                        :return "CL_INVALID_PLATFORM";  
		case CL_INVALID_DEVICE                           :return "CL_INVALID_DEVICE";  
		case CL_INVALID_CONTEXT                          :return "CL_INVALID_CONTEXT";  
		case CL_INVALID_QUEUE_PROPERTIES                 :return "CL_INVALID_QUEUE_PROPERTIES";  
		case CL_INVALID_COMMAND_QUEUE                    :return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR                         :return "CL_INVALID_HOST_PTR";  
		case CL_INVALID_MEM_OBJECT                       :return "CL_INVALID_MEM_OBJECT";  
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          :return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";  
		case CL_INVALID_IMAGE_SIZE                       :return "CL_INVALID_IMAGE_SIZE";  
		case CL_INVALID_SAMPLER                          :return "CL_INVALID_SAMPLER";  
		case CL_INVALID_BINARY                           :return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS                    :return "CL_INVALID_BUILD_OPTIONS";  
		case CL_INVALID_PROGRAM                          :return "CL_INVALID_PROGRAM";  
		case CL_INVALID_PROGRAM_EXECUTABLE               :return "CL_INVALID_PROGRAM_EXECUTABLE";  
		case CL_INVALID_KERNEL_NAME                      :return "CL_INVALID_KERNEL_NAME";  
		case CL_INVALID_KERNEL_DEFINITION                :return "CL_INVALID_KERNEL_DEFINITION";  
		case CL_INVALID_KERNEL                           :return "CL_INVALID_KERNEL";  
		case CL_INVALID_ARG_INDEX                        :return "CL_INVALID_ARG_INDEX";  
		case CL_INVALID_ARG_VALUE                        :return "CL_INVALID_ARG_VALUE";  
		case CL_INVALID_ARG_SIZE                         :return "CL_INVALID_ARG_SIZE";  
		case CL_INVALID_KERNEL_ARGS                      :return "CL_INVALID_KERNEL_ARGS";  
		case CL_INVALID_WORK_DIMENSION                   :return "CL_INVALID_WORK_DIMENSION";  
		case CL_INVALID_WORK_GROUP_SIZE                  :return "CL_INVALID_WORK_GROUP_SIZE";  
		case CL_INVALID_WORK_ITEM_SIZE                   :return "CL_INVALID_WORK_ITEM_SIZE";  
		case CL_INVALID_GLOBAL_OFFSET                    :return "CL_INVALID_GLOBAL_OFFSET";  
		case CL_INVALID_EVENT_WAIT_LIST                  :return "CL_INVALID_EVENT_WAIT_LIST";  
		case CL_INVALID_EVENT                            :return "CL_INVALID_EVENT";  
		case CL_INVALID_OPERATION                        :return "CL_INVALID_OPERATION";  
		case CL_INVALID_GL_OBJECT                        :return "CL_INVALID_GL_OBJECT";  
		case CL_INVALID_BUFFER_SIZE                      :return "CL_INVALID_BUFFER_SIZE";  
		case CL_INVALID_MIP_LEVEL                        :return "CL_INVALID_MIP_LEVEL";  
		case CL_INVALID_GLOBAL_WORK_SIZE                 :return "CL_INVALID_GLOBAL_WORK_SIZE";  
		case CL_INVALID_PROPERTY                         :return "CL_INVALID_PROPERTY";  
		case CL_INVALID_IMAGE_DESCRIPTOR                 :return "CL_INVALID_IMAGE_DESCRIPTOR";  
		case CL_INVALID_COMPILER_OPTIONS                 :return "CL_INVALID_COMPILER_OPTIONS";
		case CL_INVALID_LINKER_OPTIONS                   :return "CL_INVALID_LINKER_OPTIONS";
		case CL_INVALID_DEVICE_PARTITION_COUNT           :return "CL_INVALID_DEVICE_PARTITION_COUNT";
		case CL_INVALID_PIPE_SIZE                        :return "CL_INVALID_PIPE_SIZE";
		case CL_INVALID_DEVICE_QUEUE                     :return "CL_INVALID_DEVICE_QUEUE";
	}
	return"";
}
#endif
