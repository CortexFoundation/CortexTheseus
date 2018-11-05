#include <stdio.h>
#include <stdlib.h>
#include <ocl.h>
const int n = 4;

int main(){
	cl_platform_id platformId = getOnePlatform();
    if(platformId == NULL) return 0;
    getPlatformInfo(platformId);
     printf("get one device.\n");
     cl_device_id deviceId = getOneDevice(platformId, 1);
     if(deviceId == NULL) return 0;
     printf("create context.\n");
     cl_context context = createContext(platformId, deviceId);
     if(context == NULL) return 0;
     printf("create command queue.\n");
     cl_command_queue commandQueue = createCommandQueue(context, deviceId);
     if(commandQueue == NULL) return 0;

     const char *filename = "test.cl";
     string sourceStr;
     size_t size = 0;
     int status  = convertToString(filename, sourceStr, size);
     const char *source = sourceStr.c_str();
     printf("create program.\n");
     cl_program program = createProgram(context, &source, size);
     if(program == NULL) return 0;
     const char options[] = "-I./";
     printf("build program.\n");
     int hosta[8];// = (int*) malloc (sizeof(int) * 256);
     for(int i = 0; i < 8; i++)
	     hosta[i] = i;

     buildProgram(program, &(deviceId), options);

	cl_int clResult;
	printf("create buffer\n");
	cl_mem buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint2)*n, NULL, &clResult);
	if(clResult != CL_SUCCESS){
		printf("create buffer error : %d\n", clResult);
	}
	clEnqueueWriteBuffer(commandQueue, buffer1, CL_TRUE, 0, sizeof(int) * 8, hosta, 0, NULL, NULL);

/*	int initV = 10;
	printf("fill buffer\n");
	clResult |= clEnqueueFillBuffer(commandQueue, buffer1, &initV, sizeof(int), 4*sizeof(int), 4*sizeof(int), 0, NULL, NULL);
*/
	printf("create kernel\n");
	cl_kernel kernel = clCreateKernel(program, "test", &clResult);
	if(clResult != CL_SUCCESS){
		printf("create kernel error : %d\n", clResult);
	}
	printf("set arg\n");
	clResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer1);
	if(clResult != CL_SUCCESS){
		printf("set arg error %d\n",clResult);
	}
//    clEnqueueNDRangeKernel(trimmer->commandQueue, recovery_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	size_t global_work_size[1] = {1};
	printf("run kernel\n");
	clResult = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	if(clResult != CL_SUCCESS){
		printf("run kernel error : %d\n", clResult);
	}
	clFinish(commandQueue);
	
	int *result = (int*)malloc(sizeof(cl_uint)*n);
	printf("read buffer");
	clResult = clEnqueueReadBuffer(commandQueue, buffer1, CL_TRUE, 0, sizeof(cl_uint2)*4, result, 0, NULL, NULL);
	if(clResult != CL_SUCCESS){
		printf("read buffer error %d\n", clResult);
	}
	for(int i = 0; i < 8; i++){
		printf("%d ", result[i]);
	}
	printf("\n");
	return 0;
}
