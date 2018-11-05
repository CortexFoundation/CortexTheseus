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
	return deviceIds[deviceId];
}
#endif
