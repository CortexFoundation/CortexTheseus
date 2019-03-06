#ifndef CONTEXT_H
#define CONTEXT_H
inline cl_context createContext(cl_platform_id platformId, cl_device_id deviceId){
	cl_context_properties contextProperties[] = { 
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformId, 
		0 
	};
	cl_int errNum; 
	cl_context context = clCreateContext(contextProperties, 1, &deviceId, NULL, NULL, &errNum);
	if(errNum != CL_SUCCESS || context == NULL){
		perror("failed to create context.\n");
		return NULL;
	}
	return context;
}
inline void releaseContext(cl_context context){
	if(context){
		clReleaseContext(context);
	}
}
#endif
