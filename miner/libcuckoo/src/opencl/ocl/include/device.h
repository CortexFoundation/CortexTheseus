#ifndef DEVICE_H
#define DEVICE_H
inline cl_device_id getFirstDevice(cl_platform_id platformId){
	cl_device_id device = NULL;
	cl_int err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if(err != CL_SUCCESS){
		perror("get device failed\n");
		return NULL;
	}
	return device;
}
inline cl_device_id getOneDevice(cl_platform_id platformId, unsigned int deviceId){
	cl_uint numDevices = 0;
	cl_int err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if(err != CL_SUCCESS || numDevices <= deviceId){
		perror("query device failed.\n");
		return NULL;
	}
	cl_device_id *deviceIds = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
	err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numDevices, deviceIds, NULL);
	cl_device_id ret = deviceIds[deviceId];
	free(deviceIds);
	return ret;
}

inline void getAllDeviceInfo(){
	unsigned int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);

	    clGetDeviceInfo(devices[0], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[0], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf("\033[0;32;40m OpenCL driver version: %s\n", value);
            free(value);
        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {


            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
		
	    cl_ulong global_mem_size = 0;
	    clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
	    clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
	
	    printf("\033[0;32;40m GPU #%d: %s, %.0fMB, %u compute units, capability: %s\033[0m \n", i*platformCount + j, "", global_mem_size/1048576.0f, maxComputeUnits, value);
            free(value);
        }

        free(devices);

    }

    free(platforms);
}
#endif
