#ifndef COMMAND_QUEUE_H
#define COMMAND_QUEUE_H
inline cl_command_queue createCommandQueue(cl_context context, cl_device_id deviceId){
	cl_int err;
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceId, 0, &err);
	if(err != CL_SUCCESS){
		perror("failed to create command queue.\n");
		return NULL;
	}
	return commandQueue;
}

inline void releaseCommandQueue(cl_command_queue queue){
	if(queue){
		clReleaseCommandQueue(queue);
	}
}
#endif
