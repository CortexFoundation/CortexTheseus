#ifndef PLATFORM_H
#define PLATFORM_H
inline cl_platform_id getOnePlatform(){
	cl_platform_id platformId = NULL;
	cl_uint numPlatforms = 0;
	cl_int err = clGetPlatformIDs(1, &platformId, &numPlatforms);
	if(err != CL_SUCCESS || numPlatforms == 0){
		perror("failed to find any opencl platforms.\n");
		return NULL;
	}
	return platformId;
}

inline void getPlatformInfo(cl_platform_id platformId){
	size_t size = 0;
        cl_int status = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 0, NULL, &size);
	if(status != CL_SUCCESS){
		perror( "get platform info eror. \n");
		return;
	}
        char *name = (char*)malloc(size);
        status = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, size, name, NULL);
        printf("platform name: %s\n", name);
	free(name);
}
#endif
