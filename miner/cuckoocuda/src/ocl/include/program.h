#ifndef PROGRAM_H
#define PROGRAM_H
inline cl_program createProgram(cl_context context, const char **source, size_t sourceSize){
	cl_int err;
	size_t size[] = {sourceSize};
	cl_program program = clCreateProgramWithSource(context, 1, source, size, &err);
	if(err != CL_SUCCESS){
		perror("failed to create program.\n");
		return NULL;
	}
	return program;
}
inline void releaseProgram(cl_program program){
	if(program){
		clReleaseProgram(program);
	}
}
/**获取编译program出错时，编译器的出错信息*/
inline int getProgramBuildInfo(cl_program program,cl_device_id device)
{
    size_t log_size;
    char *program_log;
    /* Find size of log and print to std output */
    printf("get program build info : \n");
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
    program_log = (char*) malloc(log_size+1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            log_size+1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    return 0;
}

inline int buildProgram(cl_program program, cl_device_id *deviceId, const char*options){
    if(program == NULL || deviceId == NULL){
        printf("error : the program is null\n");
        return -1;
    }
	cl_int err = clBuildProgram(program, 1, deviceId, NULL, NULL, NULL);
	if(err != CL_SUCCESS){
		perror("failed to build program.\n");
		getProgramBuildInfo(program, *deviceId);
		return -1;
	}
	return 0;
}
#endif

