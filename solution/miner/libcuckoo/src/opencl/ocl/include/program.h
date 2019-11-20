#ifndef PROGRAM_H
#define PROGRAM_H
inline cl_program
createProgram(cl_context context, const char **source, size_t sourceSize)
{
    cl_int err;
    size_t size[] = { sourceSize };
    cl_program program = clCreateProgramWithSource(context, 1, source, size, &err);
    if (err != CL_SUCCESS)
    {
	perror("failed to create program.\n");
	return NULL;
    }
    return program;
}

inline void
releaseProgram(cl_program program)
{
    if (program)
    {
	clReleaseProgram(program);
    }
}

/**获取编译program出错时，编译器的出错信息*/
inline int
getProgramBuildInfo(cl_program program, cl_device_id device)
{
    size_t log_size;
    char *program_log;
    /* Find size of log and print to std output */
    printf("get program build info : \n");
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    program_log = (char *) malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    return 0;
}

inline int
buildProgram(cl_program program, cl_device_id * deviceId, const char *options)
{
    if (program == NULL || deviceId == NULL)
    {
	printf("error : the program is null\n");
	return -1;
    }
    cl_int err = clBuildProgram(program, 1, deviceId, options, NULL, NULL);
    if (err != CL_SUCCESS)
    {
	perror("failed to build program.\n");
	getProgramBuildInfo(program, *deviceId);
	return -1;
    }
    return 0;
}

inline int
saveBinaryFile(cl_program program, cl_device_id device)
{
    cl_uint numDevices = 0;

    clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof (cl_uint), &numDevices, NULL);

    cl_device_id *devices = new cl_device_id[numDevices];
    clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof (cl_device_id) * numDevices, devices, NULL);

    size_t *programBinarySizes = new size_t[numDevices];
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof (size_t) * numDevices, programBinarySizes, NULL);

    unsigned char **programBinaries = new unsigned char *[numDevices];
    for (cl_uint i = 0; i < numDevices; ++i)
	programBinaries[i] = new unsigned char[programBinarySizes[i]];

    clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof (unsigned char *) * numDevices, programBinaries, NULL);

    for (cl_uint i = 0; i < numDevices; ++i)
    {
	if (devices[i] == device)
	{
	    FILE *fp = fopen("timmer.bin", "wb");
	    fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
	    fclose(fp);
	    break;
	}
    }

    delete[]devices;
    delete[]programBinarySizes;
    for (cl_uint i = 0; i < numDevices; ++i)
	delete[]programBinaries[i];
    delete[]programBinaries;
    return 0;
}

inline cl_program
createByBinaryFile(const char *filename, cl_context context, cl_device_id device)
{
    FILE *fp = fopen(filename, "rb");
	if(fp == NULL){
		printf("open binary file %s failed.\n", filename);
		return NULL;
	}
    size_t binarySize;
    fseek(fp, 0, SEEK_END);
    binarySize = ftell(fp);
    rewind(fp);

    unsigned char *programBinary = new unsigned char[binarySize];
    fread(programBinary, 1, binarySize, fp);
    fclose(fp);

    cl_program program;
    program = clCreateProgramWithBinary(context, 1, &device, &binarySize, (const unsigned char **) &programBinary, NULL, NULL);
	delete[]programBinary; 
	return program;
}
#endif
