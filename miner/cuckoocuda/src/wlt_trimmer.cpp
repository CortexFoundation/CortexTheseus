#include <time.h>

#include "trimmer_cl.h" 
namespace cuckoogpu {

#define TROMP_SEEDA
#define TROMP_SEEDB
#define TROMP_ROUND
#define TROMP_TAIL

#define TIMER

#define DUCK_A_EDGES (EDGES_A)
#define DUCK_A_EDGES_NX (DUCK_A_EDGES * NX)
#define DUCK_B_EDGES (EDGES_B)
#define DUCK_B_EDGES_NX (DUCK_B_EDGES * NX)

	void print(cl_command_queue queue, cl_mem gpu_data, int num, size_t offset){
		int *tmp = (int*)malloc(sizeof(int) * num);
		cl_int err = clEnqueueReadBuffer(queue, gpu_data, CL_TRUE, offset, num*sizeof(int), tmp, 0, NULL, NULL);
		if(err != CL_SUCCESS){
			printf("read buffer error : %d\n", err);
			return;
		}
		for(int i = 0; i < num; i++){
			printf("%d ", tmp[i]);
		}
		printf("\n");
	}

    edgetrimmer::edgetrimmer(const trimparams _tp, cl_context context, cl_command_queue commandQueue, cl_program program) {
		this->context = context;
		this->commandQueue = commandQueue;
		this->program = program;
        indexesSize = NX * NY * sizeof(u32);
        tp = _tp;

/*        checkCudaErrors(cudaMalloc((void**)&dipkeys, sizeof(siphash_keys)));
        checkCudaErrors(cudaMalloc((void**)&indexesE, indexesSize));
        checkCudaErrors(cudaMalloc((void**)&indexesE2, indexesSize));
*/
		cl_int clResult;
        dipkeys = clCreateBuffer(this->context, CL_MEM_WRITE_ONLY, sizeof(siphash_keys), NULL, &clResult);
		if(clResult != CL_SUCCESS){
			printf("create buffer dipkeys error : %d\n", clResult);
			return;
		}
        indexesE = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize, NULL, &clResult);
		if(clResult != CL_SUCCESS){
			printf("create buffer indexesE error : %d\n", clResult);
			return;
		}
        indexesE2 = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize, NULL, &clResult);
		if(clResult != CL_SUCCESS){
			printf("create buffer indexesE2 error : %d\n", clResult);
			return;
		}

        sizeA = ROW_EDGES_A * NX * sizeof(cl_uint2);
        sizeB = ROW_EDGES_B * NX * sizeof(cl_uint2);

        const size_t bufferSize = sizeA + sizeB;
        fprintf(stderr, "bufferSize: %lu\n", bufferSize);
//        checkCudaErrors(cudaMalloc((void**)&bufferA, bufferSize));
        bufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeA, NULL, NULL);
		if(clResult != CL_SUCCESS){
			printf("create buffer bufferA error : %d\n", clResult);
			return;
		}
		int initV = 0;
//		clEnqueueFillBuffer(commandQueue, bufferA, &initV, sizeof(int), 0, sizeA, 0, NULL, NULL);
//        bufferB  = bufferA + sizeA / sizeof(cl_ulong4);
//        bufferAB = bufferA + sizeB / sizeof(cl_ulong4);
        bufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeB, NULL, &clResult);
        bufferAB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeA, NULL, &clResult);
//		 bufferB = bufferA;
//		 bufferAB = bufferA;
    }

    u64 edgetrimmer::globalbytes() const {
        return (sizeA+sizeB) + 2 * indexesSize + sizeof(siphash_keys);
    }

    edgetrimmer::~edgetrimmer() {
/*        cudaFree(bufferA);
        cudaFree(indexesE2);
        cudaFree(indexesE);
        cudaFree(dipkeys);
        cudaDeviceReset();
*/
        clReleaseMemObject(bufferA);
		clReleaseMemObject(bufferB);
		clReleaseMemObject(bufferAB);
        clReleaseMemObject(indexesE2);
        clReleaseMemObject(indexesE);
        clReleaseMemObject (dipkeys);
        releaseCommandQueue(commandQueue);
        releaseProgram(program);
        releaseContext(context);
    }

    u32 edgetrimmer::trim(uint32_t device) {
        printf("call trim\n");
//        cudaSetDevice(device);

/*#ifdef TIMER
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start)); 
		checkCudaErrors(cudaEventCreate(&stop));
#endif

        cudaMemset(indexesE, 0, indexesSize);
        cudaMemset(indexesE2, 0, indexesSize);
        cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);

        checkCudaErrors(cudaDeviceSynchronize());
*/
        size_t tmpSize = indexesSize;
//        int *initInt = (int*)malloc(tmpSize);
		int initV = 0;
		int bufferA_offset = 0;
		int bufferB_offset = 0;//sizeA / sizeof(cl_ulong4);
		int bufferAB_offset = 0;//sizeB / sizeof(cl_ulong4);
		cl_int clResult; 
        //memset(initInt, 0, tmpSize);
//        clResult = clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clResult = clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
		if(clResult != CL_SUCCESS){
			printf("fill buffer indexesE error: %d\n", clResult);
			return 0;
		}
//        clResult = clEnqueueWriteBuffer(commandQueue, indexesE2, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clResult = clEnqueueFillBuffer(commandQueue, indexesE2, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
		if(clResult != CL_SUCCESS){
			printf("fill buffer indexesE2 error: %d\n", clResult);
			return 0;
		}
        clResult = clEnqueueWriteBuffer(commandQueue, dipkeys, CL_TRUE, 0, sizeof(siphash_keys), &sipkeys, 0, NULL, NULL);
		if(clResult != CL_SUCCESS){
			printf("fill buffer dipkeys error : %d\n", clResult);
			return 0;
		}
/*
#ifdef TIMER
        float durationA, durationB;
        cudaEventRecord(start, NULL);
#endif
*/

        size_t global_work_size[1];
        size_t local_work_size[1];
        global_work_size[0] = tp.genA.blocks * tp.genA.tpb;
        local_work_size[0] = tp.genA.tpb;
        cl_event event;
		int edges_a = EDGES_A;
#ifdef TROMP_SEEDA
        cl_kernel seedA_kernel = clCreateKernel(program, "SeedA", &clResult);
		if(clResult != CL_SUCCESS){
			printf("create kernel error : %d\n", clResult);
			return -1;
		}
        clResult = clSetKernelArg(seedA_kernel, 0, sizeof(cl_mem), (void*)&dipkeys);
		if(clResult != CL_SUCCESS){
			printf("set arg dipkeys error : %d\n", clResult);
			return -1;
		}
        clResult = clSetKernelArg(seedA_kernel, 1, sizeof(cl_mem), (void*)&bufferAB);
		if(clResult != CL_SUCCESS){
			printf("set arg bufferAB error %d\n", clResult);
			return -1;
		}
        clResult = clSetKernelArg(seedA_kernel, 2, sizeof(cl_mem), (void*)&indexesE);
		if(clResult != CL_SUCCESS){
			printf("set arg indexesE error %d \n", clResult);
			return -1;
		}
		clResult = clSetKernelArg(seedA_kernel, 3, sizeof(int), &edges_a);
		if(clResult != CL_SUCCESS){
			printf("set arg edges_a error %d\n", clResult);
			return -1;
		}
		clResult = clSetKernelArg(seedA_kernel, 4, sizeof(int), &bufferAB_offset);
		if(clResult != CL_SUCCESS){
			printf("set arg bufferAB_offset error %d\n", clResult);
			return -1;
		}

//		SeedA<EDGES_A, cl_uint2><<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, bufferAB, (int *)indexesE);
        clResult = clEnqueueNDRangeKernel(commandQueue, seedA_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
			
#else
        cl_kenrel seed2A_kernel = clCreateKernel(program, "Seed2A", NULL);
        clResult = clSetKernelArg(seed2A_kernel, 0, sizeof(cl_mem), (void*)&dipkeys);
		if(clResult != CL_SUCCESS){
			printf("write buffer error\n");
			return -1;
		}
        clResult = clSetKernelArg(seed2A_kernel, 1, sizeof(cl_mem), (void*)&bufferAB);
		if(clResult != CL_SUCCESS){
			printf("write buffer error\n");
			return -1;
		}
        clResult = clSetKernelArg(seed2A_kernel, 2, sizeof(cl_mem), (void*)&indexesE);
		if(clResult != CL_SUCCESS){
			printf("write buffer error\n");
			return -1;
		}
		clResult = clSetKernelArg(seed2A_kernel, 3, sizeof(int), &bufferAB_offset);
		if(clResult != CL_SUCCESS){
			printf("write buffer error\n");
			return -1;
		}
//		Seed2A<<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, bufferAB, (int *)indexesE);
        clResult = clEnqueueNDRangeKernel(commandQueue, seed2A_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
		if(clResult != CL_SUCCESS){
			printf("call Seed2A error : %d\n", clResult);
			return -1;
		}

#endif
		
        clWaitForEvents(1, &event);
/*
#ifdef TIMER
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&durationA, start, stop); 

		cudaEventRecord(start, NULL);
#endif
*/
        u32 halfA0 = 0;
        u32 halfE0 = 0;
        u32 halfA = sizeA/2;///sizeof(cl_ulong4);
        u32 halfE = NX/2;
        
        global_work_size[0] = tp.genB.blocks/2 * tp.genB.tpb;
        local_work_size[0] = tp.genB.tpb;
        cl_kernel seedB_kernel = clCreateKernel(program, "SeedB", &clResult);
        clResult |= clSetKernelArg(seedB_kernel, 0, sizeof(cl_mem), (void*)&dipkeys);
        clResult |= clSetKernelArg(seedB_kernel, 1, sizeof(cl_mem), (void*)&bufferAB);
        clResult |= clSetKernelArg(seedB_kernel, 2, sizeof(cl_mem), (void*)&bufferA);
        clResult |= clSetKernelArg(seedB_kernel, 3, sizeof(cl_mem), (void*)&indexesE);
        clResult |= clSetKernelArg(seedB_kernel, 4, sizeof(cl_mem), (void*)&indexesE2);
        clResult |= clSetKernelArg(seedB_kernel, 5, sizeof(int), &EDGES_A);
        clResult |= clSetKernelArg(seedB_kernel, 6, sizeof(int), &halfA0);
        clResult |= clSetKernelArg(seedB_kernel, 7, sizeof(int), &halfE0);
		clResult |= clSetKernelArg(seedB_kernel, 8, sizeof(int), &bufferAB_offset);
		if(clResult != CL_SUCCESS){
			printf("call SeedB , cl error : %d\n", clResult);
			return -1;
		}
#ifdef TROMP_SEEDB
        clResult = clEnqueueNDRangeKernel(commandQueue, seedB_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);
//		clEnqueueCopyBuffer(commandQueue, bufferB, bufferA, 0, 0, sizeA/2, NULL, NULL, NULL);
//		SeedB<EDGES_A, cl_uint2><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const cl_uint2 *)bufferAB, bufferA, (const int *)indexesE, indexesE2);
        clResult |= clSetKernelArg(seedB_kernel, 0, sizeof(cl_mem), (void*)&dipkeys);
        clResult |= clSetKernelArg(seedB_kernel, 1, sizeof(cl_mem), (void*)(&bufferAB));
        clResult |= clSetKernelArg(seedB_kernel, 2, sizeof(cl_mem), (void*)(&bufferA));
        clResult |= clSetKernelArg(seedB_kernel, 3, sizeof(cl_mem), (void*)(&indexesE));
        clResult |= clSetKernelArg(seedB_kernel, 4, sizeof(cl_mem), (void*)(&indexesE2));
        clResult |= clSetKernelArg(seedB_kernel, 6, sizeof(int), &halfA);
        clResult |= clSetKernelArg(seedB_kernel, 7, sizeof(int), &halfE);
		clResult |= clSetKernelArg(seedB_kernel, 8, sizeof(int), &bufferAB_offset);
        clResult |= clEnqueueNDRangeKernel(commandQueue, seedB_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);
		if(clResult != CL_SUCCESS){
			printf("call seedb error : %d\n",clResult);
			return -1;
		}
//		SeedB<EDGES_A, cl_uint2><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const cl_uint2 *)(bufferAB+halfA), bufferA+halfA, (const int *)(indexesE+halfE), indexesE2+halfE);
#else
        cl_kernel seed2b_kernel = clCreateKernel(program, "Seed2B", &clResult);
        global_work_size[0] = tp.genB.blocks/2 * NX;
        local_work_size[0] = NX;
		clResult |= clSetKernelArg(seed2b_kernel, 0, sizeof(cl_mem), (void*)&bufferAB);
        clResult |= clSetKernelArg(seed2b_kernel, 1, sizeof(cl_mem), (void*)&bufferA);
        clResult |= clSetKernelArg(seed2b_kernel, 2, sizeof(cl_mem), (void*)&indexesE);
        clResult |= clSetKernelArg(seed2b_kernel, 3, sizeof(cl_mem), (void*)&indexesE2);
        clResult |= clSetKernelArg(seed2b_kernel, 4, sizeof(int), &halfA0);
        clResult |= clSetKernelArg(seed2b_kernel, 5, sizeof(int), &halfE0);
		clResult |= clSetKernelArg(seed2b_kernel, 6, sizeof(int), &bufferAB_offset);
        clResult |= clEnqueueNDRangeKernel(commandQueue, seed2b_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(event);
//		Seed2B<<<tp.genB.blocks/2, NX>>>((const cl_uint2 *)bufferAB, bufferA, (const int *)indexesE, indexesE2);

        clResult |= clSetKernelArg(seed2b_kernel, 0, sizeof(cl_mem), (void*)(&bufferAB));
        clResult |= clSetKernelArg(seed2b_kernel, 1, sizeof(cl_mem), (void*)(&bufferA));
        clResult |= clSetKernelArg(seed2b_kernel, 2, sizeof(cl_mem), (void*)(&indexesE));
        clResult |= clSetKernelArg(seed2b_kernel, 3, sizeof(cl_mem), (void*)(&indexesE2));
        clResult |= clSetKernelArg(seed2b_kernel, 4, sizeof(int), &halfA);
        clResult |= clSetKernelArg(seed2b_kernel, 5, sizeof(int), &halfE);
		clResult |= clSetKernelArg(seed2b_kernel, 6, sizeof(int), &bufferAB_offset);
        clResult |= clEnqueueNDRangeKernel(commandQueue, seed2b_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
		if(clResult != CL_SUCCESS){
			printf("call seed2b error : %d\n", clResult);
			return -1;
		}
//		Seed2B<<<tp.genB.blocks/2, NX>>>((const cl_uint2 *)(bufferAB+halfA), bufferA+halfA, (const int *)(indexesE+halfE), indexesE2+halfE);
#endif



//#ifdef INDEX_DEBUG
//		cudaMemcpy(hostA, indexesE2, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
//#endif
/*
#ifdef TIMER
		cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&durationB, start, stop);
		fprintf(stderr, "Seeding completed in %.2f + %.2f ms\n", durationA, durationB);

		cudaEventRecord(start, NULL);
#endif
*/
#ifdef TROMP_ROUND
//		cudaMemset(indexesE, 0, indexesSize);
//        clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
        cl_kernel round_kernel = clCreateKernel(program, "Round", NULL);
        global_work_size[0] = tp.trim.blocks * tp.trim.tpb;
        local_work_size[0] = tp.trim.tpb;
        int constRound = 0;
        edges_a = EDGES_A;
        int edges_b = EDGES_B;
        clSetKernelArg(round_kernel, 0, sizeof(int), &constRound);
        clSetKernelArg(round_kernel, 1, sizeof(cl_mem), (void*)&dipkeys);
        clSetKernelArg(round_kernel, 2, sizeof(cl_mem), (void*)&bufferA);
        clSetKernelArg(round_kernel, 3, sizeof(cl_mem), (void*)&bufferB);
        clSetKernelArg(round_kernel, 4, sizeof(cl_mem), (void*)&indexesE2);
        clSetKernelArg(round_kernel, 5, sizeof(cl_mem), (void*)&indexesE);
        clSetKernelArg(round_kernel, 6, sizeof(int), &edges_a);
        clSetKernelArg(round_kernel, 7, sizeof(int), &edges_b);
        clSetKernelArg(round_kernel, 8, sizeof(int), &bufferA_offset);	
        clSetKernelArg(round_kernel, 9, sizeof(int), &bufferB_offset);
        clEnqueueNDRangeKernel(commandQueue, round_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);

//		Round<EDGES_A, cl_uint2, EDGES_B, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632

//		cudaMemset(indexesE2, 0, indexesSize);
//        clEnqueueWriteBuffer(commandQueue, indexesE2, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE2, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
              
        constRound = 1;
        edges_a = EDGES_B;
        edges_b = EDGES_B/2;
        clSetKernelArg(round_kernel, 0, sizeof(int), &constRound);
        clSetKernelArg(round_kernel, 1, sizeof(cl_mem), (void*)&dipkeys);
        clSetKernelArg(round_kernel, 2, sizeof(cl_mem), (void*)&bufferB);
        clSetKernelArg(round_kernel, 3, sizeof(cl_mem), (void*)&bufferA);
        clSetKernelArg(round_kernel, 4, sizeof(cl_mem), (void*)&indexesE);        
        clSetKernelArg(round_kernel, 5, sizeof(cl_mem), (void*)&indexesE2);
        clSetKernelArg(round_kernel, 6, sizeof(int), &edges_a);
        clSetKernelArg(round_kernel, 7, sizeof(int), &edges_b);
        clSetKernelArg(round_kernel, 8, sizeof(int), &bufferB_offset);
        clSetKernelArg(round_kernel, 9, sizeof(int), &bufferA_offset);
        clEnqueueNDRangeKernel(commandQueue, round_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);


//		Round<EDGES_A, cl_uint2, EDGES_B, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
//		Round<EDGES_B, cl_uint2, EDGES_B/2, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(1, *dipkeys, (const cl_uint2 *)bufferB, (cl_uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .296

//		cudaMemset(indexesE, 0, indexesSize);
//        clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
         
        constRound = 2;
        edges_a = EDGES_B/2;
        edges_b = EDGES_A/4;
        clSetKernelArg(round_kernel, 0, sizeof(int), &constRound);
        clSetKernelArg(round_kernel, 2, sizeof(cl_mem), (void*)&bufferA);
        clSetKernelArg(round_kernel, 3, sizeof(cl_mem), (void*)&bufferB);
        clSetKernelArg(round_kernel, 4, sizeof(cl_mem), (void*)&indexesE2);
        clSetKernelArg(round_kernel, 5, sizeof(cl_mem), (void*)&indexesE);
        clSetKernelArg(round_kernel, 6, sizeof(int), &edges_a);
        clSetKernelArg(round_kernel, 7, sizeof(int), &edges_b);
        clSetKernelArg(round_kernel, 8, sizeof(int), &bufferA_offset);
        clSetKernelArg(round_kernel, 9, sizeof(int), &bufferB_offset);
        clEnqueueNDRangeKernel(commandQueue, round_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);


//		Round<EDGES_A, cl_uint2, EDGES_B, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
//		Round<EDGES_B/2, cl_uint2, EDGES_A/4, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(2, *dipkeys, (const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .176

//		cudaMemset(indexesE2, 0, indexesSize);
//        clEnqueueWriteBuffer(commandQueue, indexesE2, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE2, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
          
        constRound = 3;
        edges_a = EDGES_A/4;
        edges_b = EDGES_B/4;
        clSetKernelArg(round_kernel, 0, sizeof(int), &constRound);
        clSetKernelArg(round_kernel, 2, sizeof(cl_mem), (void*)&bufferB);
        clSetKernelArg(round_kernel, 3, sizeof(cl_mem), (void*)&bufferA);
        clSetKernelArg(round_kernel, 4, sizeof(cl_mem), (void*)&indexesE);
        clSetKernelArg(round_kernel, 5, sizeof(cl_mem), (void*)&indexesE2);
        clSetKernelArg(round_kernel, 6, sizeof(int), &edges_a);
        clSetKernelArg(round_kernel, 7, sizeof(int), &edges_b);
        clSetKernelArg(round_kernel, 8, sizeof(int), &bufferB_offset);
        clSetKernelArg(round_kernel, 9, sizeof(int), &bufferA_offset);
        clEnqueueNDRangeKernel(commandQueue, round_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);

//		Round<EDGES_A, cl_uint2, EDGES_B, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
//		Round<EDGES_A/4, cl_uint2, EDGES_B/4, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(3, *dipkeys, (const cl_uint2 *)bufferB, (cl_uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .117 

#else
//        cudaMemset(indexesE, 0, indexesSize);
//        clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
        cl_kernel round2_kernel = clCreateKernel(program, "Round2", NULL);
        int duck_a_edges = DUCK_A_EDGES;
        int duck_b_edges = DUCK_B_EDGES;
        clSetKernelArg(round2_kernel, 0, sizeof(cl_mem), (void*)&bufferA);
        clSetKernelArg(round2_kernel, 1, sizeof(cl_mem), (void*)&bufferB);
        clSetKernelArg(round2_kernel, 2, sizeof(cl_mem), (void*)&indexesE2);
        clSetKernelArg(round2_kernel, 3, sizeof(cl_mem), (void*)&indexesE);
        clSetKernelArg(round2_kernel, 4, sizeof(int), &duck_a_edges);
        clSetKernelArg(round2_kernel, 5, sizeof(int), &duck_b_edges);
        clSetKernelArg(round2_kernel, 6, sizeof(int), &bufferA_offset);
        clSetKernelArg(round2_kernel, 7, sizeof(int), &bufferB_offset);
        clEnqueueNDRangeKernel(commandQueue, round2_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL,  &event);
        clWaitForEvents(1, &event);
//		Round2<DUCK_A_EDGES, DUCK_B_EDGES><<<tp.trim.blocks, tp.trim.tpb>>>((const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632

//        cudaMemset(indexesE2, 0, indexesSize);
        
//		Round2<DUCK_B_EDGES, DUCK_B_EDGES/2><<<tp.trim.blocks, tp.trim.tpb>>>((const cl_uint2 *)bufferB, (cl_uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .296
//        clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
        int duck_a_edges = DUCK_B_EDGES;
        int duck_b_edges = DUCK_B_EDGES/2;
        clSetKernelArg(round2_kernel, 0, sizeof(cl_mem), (void*)&bufferB);
        clSetKernelArg(round2_kernel, 1, sizeof(cl_mem), (void*)&bufferA);
        clSetKernelArg(round2_kernel, 2, sizeof(cl_mem), (void*)&indexesE);
        clSetKernelArg(round2_kernel, 3, sizeof(cl_mem), (void*)&indexesE2);
        clSetKernelArg(round2_kernel, 4, sizeof(int), &duck_a_edges);
        clSetKernelArg(round2_kernel, 5, sizeof(int), &duck_b_edges);
        clSetKernelArg(round2_kernel, 6, sizeof(int), &bufferB_offset);
        clSetKernelArg(round2_kernel, 7, sizeof(int), &bufferA_offset);
        clEnqueueNDRangeKernel(commandQueue, round2_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);

//        cudaMemset(indexesE, 0, indexesSize);
//        Round2<DUCK_B_EDGES/2, DUCK_A_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>((const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .176
//        clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);

        int duck_a_edges = DUCK_B_EDGES/2;
        int duck_b_edges = DUCK_A_EDGES/4;
        clSetKernelArg(round2_kernel, 0, sizeof(cl_mem), (void*)&bufferA);
        clSetKernelArg(round2_kernel, 1, sizeof(cl_mem), (void*)&bufferB);
        clSetKernelArg(round2_kernel, 2, sizeof(cl_mem), (void*)&indexesE2);
        clSetKernelArg(round2_kernel, 3, sizeof(cl_mem), (void*)&indexesE);
        clSetKernelArg(round2_kernel, 5, sizeof(int), &duck_a_edges);
        clSetKernelArg(round2_kernel, 6, sizeof(int), &duck_b_edges);
        clSetKernelArg(round2_kernel, 7, sizeof(int), &bufferA_offset);
        clSetKernelArg(round2_kernel, 8, sizeof(int), &bufferB_offset);
        clEnqueueNDRangeKernel(commandQueue, round2_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);

//        cudaMemset(indexesE2, 0, indexesSize);
//        Round2<DUCK_A_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>((const cl_uint2 *)bufferB, (cl_uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .117
//        clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
        int duck_a_edges = DUCK_A_EDGES/4;
        int duck_b_edges = DUCK_B_EDGES/4;
        clSetKernelArg(round2_kernel, 0, sizeof(cl_mem), (void*)&bufferB);
        clSetKernelArg(round2_kernel, 1, sizeof(cl_mem), (void*)&bufferA);
        clSetKernelArg(round2_kernel, 2, sizeof(cl_mem), (void*)&indexesE);
        clSetKernelArg(round2_kernel, 3, sizeof(cl_mem), (void*)&indexesE2);
        clSetKernelArg(round2_kernel, 4, sizeof(int), &duck_a_edges);
        clSetKernelArg(round2_kernel, 5, sizeof(int), &duck_b_edges);
        clSetKernelArg(round2_kernel, 6, sizeof(int), &bufferB_offset);
        clSetKernelArg(round2_kernel, 7, sizeof(int), &bufferA_offset);
        clEnqueueNDRangeKernel(commandQueue, round2_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clWaitForEvents(1, &event);

#endif

#ifdef INDEX_DEBUG
//		cudaMemcpy(hostA, indexesE2, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
        clEnqueueReadBuffer(commandQue, indexesE2, CL_TRUE, 0, NX*NY*sizeof(u32), hostA, 0, NULL, NULL);
		fprintf(stderr, "Index Number: %zu\n", hostA[0]);
#endif

//        cudaDeviceSynchronize();

        for (int round = 4; round < tp.ntrims; round += 2) {
#ifdef TROMP_ROUND
//			cudaMemset(indexesE, 0, indexesSize);
//			Round<EDGES_B/4, cl_uint2, EDGES_B/4, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(round, *dipkeys,  (const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
//            clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
			clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
            constRound = round;
            edges_a = EDGES_B/4;
            edges_b = EDGES_B/4;
            clSetKernelArg(round_kernel, 0, sizeof(int), &constRound);
            clSetKernelArg(round_kernel, 2, sizeof(cl_mem), (void*)&bufferA);
            clSetKernelArg(round_kernel, 3, sizeof(cl_mem), (void*)&bufferB);
            clSetKernelArg(round_kernel, 4, sizeof(cl_mem), (void*)&indexesE2);        
            clSetKernelArg(round_kernel, 5, sizeof(cl_mem), (void*)&indexesE);
            clSetKernelArg(round_kernel, 6, sizeof(int), &edges_a);
            clSetKernelArg(round_kernel, 7, sizeof(int), &edges_b);
			clSetKernelArg(round_kernel, 8, sizeof(int), &bufferA_offset);
			clSetKernelArg(round_kernel, 9, sizeof(int), &bufferB_offset);
            clEnqueueNDRangeKernel(commandQueue, round_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
            clWaitForEvents(1, &event);

//			cudaMemset(indexesE2, 0, indexesSize);
//			Round<EDGES_B/4, cl_uint2, EDGES_B/4, cl_uint2><<<tp.trim.blocks, tp.trim.tpb>>>(round+1, *dipkeys,  (const cl_uint2 *)bufferB, (cl_uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2);
//            clEnqueueWriteBuffer(commandQueue, indexesE2, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
			clEnqueueFillBuffer(commandQueue, indexesE2, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
            constRound = round+1;
            edges_a = EDGES_B/4;
            edges_b = EDGES_B/4;
            clSetKernelArg(round_kernel, 0, sizeof(int), &constRound);
            clSetKernelArg(round_kernel, 2, sizeof(cl_mem), (void*)&bufferB);
            clSetKernelArg(round_kernel, 3, sizeof(cl_mem), (void*)&bufferA);
            clSetKernelArg(round_kernel, 4, sizeof(cl_mem), (void*)&indexesE);        
            clSetKernelArg(round_kernel, 5, sizeof(cl_mem), (void*)&indexesE2);
            clSetKernelArg(round_kernel, 6, sizeof(int), &edges_a);
            clSetKernelArg(round_kernel, 7, sizeof(int), &edges_b);
			clSetKernelArg(round_kernel, 8, sizeof(int), &bufferB_offset);
			clSetKernelArg(round_kernel, 9, sizeof(int), &bufferA_offset);
            clEnqueueNDRangeKernel(commandQueue, round_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
            clWaitForEvents(1, &event);

#else
//            cudaMemset(indexesE, 0, indexesSize);
//            Round2<DUCK_B_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>((const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
            
//            clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
			clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
            duck_a_edges = DUCK_B_EDGES/4;
            duck_b_edges = DUCK_B_EDGES/4;
            clSetKernelArg(round2_kernel, 0, sizeof(cl_mem), (void*)&bufferA);
            clSetKernelArg(round2_kernel, 1, sizeof(cl_mem), (void*)&bufferB);
            clSetKernelArg(round2_kernel, 2, sizeof(cl_mem), (void*)&indexesE2);        
            clSetKernelArg(round2_kernel, 3, sizeof(cl_mem), (void*)&indexesE);
            clSetKernelArg(round2_kernel, 4, sizeof(int), &duck_a_edges);
            clSetKernelArg(round2_kernel, 5, sizeof(int), &duck_b_edges);
			clSetKernelArg(round2_kernel, 6, sizeof(int), &bufferA_offset);
			clSetKernelArg(round2_kernel, 7, sizeof(int), &bufferB_offset);
            clEnqueueNDRangeKernel(commandQueue, round_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
            clWaitForEvents(1, &event);

//            cudaMemset(indexesE2, 0, indexesSize);
//            Round2<DUCK_B_EDGES/4, DUCK_B_EDGES/4><<<tp.trim.blocks, tp.trim.tpb>>>((const cl_uint2 *)bufferB, (cl_uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2);

//            clEnqueueWriteBuffer(commandQueue, indexesE2, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
			clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
            duck_a_edges = DUCK_B_EDGES/4;
            duck_b_edges = DUCK_B_EDGES/4;
            clSetKernelArg(round2_kernel, 0, sizeof(cl_mem), (void*)&bufferB);
            clSetKernelArg(round2_kernel, 1, sizeof(cl_mem), (void*)&bufferA);
            clSetKernelArg(round2_kernel, 2, sizeof(cl_mem), (void*)&indexesE);        
            clSetKernelArg(round2_kernel, 3, sizeof(cl_mem), (void*)&indexesE2);
            clSetKernelArg(round2_kernel, 4, sizeof(int), &duck_a_edges);
            clSetKernelArg(round2_kernel, 5, sizeof(int), &duck_b_edges);
			clSetKernelArg(round2_kernel, 6, sizeof(int), &bufferB_offset);
			clSetKernelArg(round2_kernel, 7, sizeof(int), &bufferA_offset);
            clEnqueueNDRangeKernel(commandQueue, round_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
            clWaitForEvents(1, &event);

#endif

#ifdef INDEX_DEBUG
//			cudaMemcpy(hostA, indexesE2, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
            clEnqueueReadBuffer(commandQue, indexesE2, CL_TRUE, 0, NX*NY*sizeof(u32), hostA, 0, NULL, NULL);
			fprintf(stderr, "Index Number: %zu\n", hostA[0]);
#endif
        }

//        checkCudaErrors(cudaDeviceSynchronize()); 

#ifdef TIMER
/*		cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&durationA, start, stop);

		fprintf(stderr, "Round completed in %.2f ms\n", durationA);

		cudaEventRecord(start, NULL);
*/
#endif

#ifdef TROMP_TAIL
//        cudaMemset(indexesE, 0, indexesSize);
//          clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
			clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
//        checkCudaErrors(cudaDeviceSynchronize()); 

//        Tail<DUCK_B_EDGES/4><<<tp.tail.blocks, tp.tail.tpb>>>((const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
//        cudaMemcpy(hostA, indexesE, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
          cl_kernel tail_kernel = clCreateKernel(program, "Tail", NULL);
          global_work_size[0] = tp.tail.blocks * tp.tail.tpb;
          local_work_size[0] = tp.tail.tpb;
          int tail_edges = DUCK_B_EDGES/4;
          clSetKernelArg(tail_kernel, 0, sizeof(cl_mem), (void*)&bufferA);
          clSetKernelArg(tail_kernel, 1, sizeof(cl_mem), (void*)&bufferB);
          clSetKernelArg(tail_kernel, 2, sizeof(cl_mem), (void*)&indexesE2);
          clSetKernelArg(tail_kernel, 3, sizeof(cl_mem), (void*)&indexesE);
          clSetKernelArg(tail_kernel, 4, sizeof(int), &tail_edges);
		  clSetKernelArg(tail_kernel, 5, sizeof(int), &bufferB_offset);
          clEnqueueNDRangeKernel(commandQueue, tail_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL,  &event);
          clWaitForEvents(1, &event);
          clEnqueueReadBuffer(commandQueue, indexesE, CL_TRUE, 0, NX*NY*sizeof(u32), hostA, 0, NULL, NULL);
#else
//		cudaMemset(indexesE, 0, indexesSize);
//		cudaDeviceSynchronize();

//    Tail2<DUCK_B_EDGES/4><<<tp.tail.blocks, NX>>>((const cl_uint2 *)bufferA, (cl_uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
//	  cudaMemcpy(hostA, indexesE, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);

		  cl_kernel tail2_kernel = clCreateKernel(program, "Tail2", NULL);
//		  clEnqueueWriteBuffer(commandQueue, indexesE, CL_TRUE, 0, tmpSize, (void*)initInt, 0, NULL, NULL);
		clEnqueueFillBuffer(commandQueue, indexesE, &initV, sizeof(int), 0, tmpSize, 0, NULL, NULL);
         global_work_size[0] = tp.tail.blocks * NX;
        local_work_size[0] = NX; 
          int tail_edges = DUCK_B_EDGES/4;
          clSetKernelArg(tail2_kernel, 0, sizeof(cl_mem), (void*)&bufferA);
          clSetKernelArg(tail2_kernel, 1, sizeof(cl_mem), (void*)&bufferB);
          clSetKernelArg(tail2_kernel, 2, sizeof(cl_mem), (void*)&indexesE2);
          clSetKernelArg(tail2_kernel, 3, sizeof(cl_mem), (void*)&indexesE);
          clSetKernelArg(tail2_kernel, 4, sizeof(int), &tail_edges);
		  clSetKernelArg(tail2_kernel, 5, sizeof(int), &bufferB_offset);
          clEnqueueNDRangeKernel(commandQueue, tail2_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
          clWaitForEvents(1, &event);
          clEnqueueReadBuffer(commandQueue, indexesE, CL_TRUE, 0, NX*NY*sizeof(u32), hostA, 0, NULL, NULL);

//#else
//		cudaMemset(indexesE, 0, indexesSize);
          
#endif


#ifdef TIMER
/*		cudaEventRecord(stop, NULL);
        checkCudaErrors(cudaEventSynchronize(stop)); 
		cudaEventElapsedTime(&durationA, start, stop);
		fprintf(stderr, "Tail completed in %.2f ms\n", durationA);

		checkCudaErrors(cudaEventDestroy(start));
		checkCudaErrors(cudaEventDestroy(stop));
*/
#endif

  //      checkCudaErrors(cudaDeviceSynchronize());
		fprintf(stderr, "Host A [0]: %zu\n", hostA[0]);
        return hostA[0];
    }


};
