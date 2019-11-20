#ifndef MONITOR_H
#define MONITOR_H

#include <nvml.h>
#include <stdio.h>

int monitor_init(unsigned int deviceCount){
	nvmlReturn_t result = nvmlInit();
	
	if (NVML_SUCCESS != result)
	{
		printf ("Failed to initialize NVML: %s\n", nvmlErrorString (result));
		return -1;
	}
	
	unsigned int device_count = 0;
	result = nvmlDeviceGetCount(&device_count);
	if (NVML_SUCCESS != result)
		printf ("nvml get device count failed: %s\n", nvmlErrorString (result));
	if(device_count < deviceCount){
		printf("there is no much device\n");
		return -1;
	}
	return device_count;
}

int query_fan_tem(unsigned int device_count, unsigned int *fanSpeeds, unsigned int *temperatures){
	nvmlReturn_t result;	
	for (int i = 0; i < device_count; i++)
	{
		nvmlDevice_t device;
//		char name[64];
//		nvmlPciInfo_t pci;
		result = nvmlDeviceGetHandleByIndex (i, &device);
		if (NVML_SUCCESS != result)
		{
			printf ("Failed to get handle for device %i: %s\n", i, nvmlErrorString (result));
			return -1;
		}
/*
		result = nvmlDeviceGetName (device, name, sizeof (name) / sizeof (name[0]));
		if (NVML_SUCCESS != result)
		{
			printf ("Failed to get name of device %i: %s\n", i, nvmlErrorString (result));
			return 0;
		}

		result = nvmlDeviceGetPciInfo (device, &pci);
		if (NVML_SUCCESS != result)
		{
			printf ("Failed to get pci info for device %i: %s\n", i, nvmlErrorString (result));
			return 0;
		}

		printf ("\033[1;35;40m GPU%d. %s pci busid:[%s] \033[0m \n", i, name, pci.busId);
*/
		// This is a simple example on how you can modify GPU's state
		unsigned int speed = 0;
		result = nvmlDeviceGetFanSpeed(device, &speed);
		if(result != NVML_SUCCESS){
			printf("query gpu fan speed failed!\n");
			return -1;
		}
		//printf("fan speed=%u\n", speed);
		fanSpeeds[i] = speed;
		unsigned int temperature = 0;
		result = nvmlDeviceGetTemperature(device,  NVML_TEMPERATURE_GPU, &temperature);
		//printf("temperature=%u\n", temperature);
		temperatures[i] = temperature;
	}
	return 0;
}

void monitor_shutdown(){
	nvmlReturn_t result = nvmlShutdown ();
	if (NVML_SUCCESS != result)
		printf ("Failed to shutdown NVML: %s\n", nvmlErrorString (result));
}

#endif
