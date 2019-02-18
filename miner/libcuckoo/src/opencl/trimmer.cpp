#include <time.h>

#include "trimmer.h"
namespace cuckoogpu {

#define DUCK_A_EDGES_NX (DUCK_A_EDGES * NX)
#define DUCK_B_EDGES (EDGES_B)
#define DUCK_B_EDGES_NX (DUCK_B_EDGES * NX)
#define DUCK_SIZE_A  129
#define DUCK_SIZE_B  83
#define SUB_BUCKET_SIZE (1<<(ZBITS - 7))

int com(const void *a, const void *b){
	cl_uint2 va = *(cl_uint2*)a;
	cl_uint2 vb = *(cl_uint2*)b;
	if(va.x == vb.y) return va.y - vb.y;
	else return va.x - vb.x;
}

void saveFile(cl_uint2*v, int n, char *filename){
	qsort(v, n, sizeof(cl_uint2), com);
	FILE *fp = fopen(filename, "w");
	for(int i = 0; i < n; i++){
		fprintf(fp, "%d,%d\n", v[i].x, v[i].y);
	}
	fclose(fp);
}

edgetrimmer::edgetrimmer(const trimparams _tp, cl_context context,
			     cl_command_queue commandQueue,
			     cl_program program, int _selected) {
	this->context = context;
	this->commandQueue = commandQueue;
	this->program = program;
	this->selected = _selected;
	indexesSize = 256 * 256 * 4;
	tp = _tp;

	cl_int clResult;
	bufferA1_size =  DUCK_SIZE_A * SUB_BUCKET_SIZE  * (4096-128) * 2;
	bufferA2_size = DUCK_SIZE_A * SUB_BUCKET_SIZE  * 256 * 2;
	bufferB_size = DUCK_SIZE_B * SUB_BUCKET_SIZE * 4096 * 2;
	buffer_size = (DUCK_SIZE_A + DUCK_SIZE_B) * SUB_BUCKET_SIZE * 4096 * 2;

	bufferA1 = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferA1_size * sizeof(uint), NULL, &clResult);
	checkOpenclErrors(clResult);
	bufferA2 = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferA2_size*sizeof(uint), NULL, &clResult);
	checkOpenclErrors(clResult);
	bufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferB_size * sizeof(uint), NULL, &clResult);
	checkOpenclErrors(clResult);
	bufferI1 = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize * sizeof(uint), NULL, &clResult);
	checkOpenclErrors(clResult);
	bufferI2 = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize * sizeof(uint), NULL, &clResult);
	checkOpenclErrors(clResult);
	bufferI3 = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize * sizeof(uint), NULL, &clResult);
	checkOpenclErrors(clResult);
	bufferR = clCreateBuffer(context, CL_MEM_READ_WRITE, PROOFSIZE*2*sizeof(uint), NULL, &clResult);
	checkOpenclErrors(clResult);

	if(this->selected == 0){
		kernel_seedA = clCreateKernel(program, "Cuckoo_FluffySeed2A", &clResult);
		kernel_recovery = clCreateKernel(program, "Cuckoo_FluffyRecovery", &clResult);
	}else{
		kernel_seedA = clCreateKernel(program, "FluffySeed2A", &clResult);
		checkOpenclErrors(clResult);
		kernel_recovery = clCreateKernel(program, "FluffyRecovery", &clResult);
		checkOpenclErrors(clResult);
	}
	kernel_seedB1 = clCreateKernel(program, "FluffySeed2B", &clResult);
	checkOpenclErrors(clResult);
	kernel_seedB2 = clCreateKernel(program, "FluffySeed2B", &clResult);
	checkOpenclErrors(clResult);
	kernel_round1 = clCreateKernel(program, "FluffyRound1", &clResult);
	checkOpenclErrors(clResult);
	kernel_round0 = clCreateKernel(program, "FluffyRoundNO1", &clResult);
	checkOpenclErrors(clResult);
	kernel_roundNA = clCreateKernel(program, "FluffyRoundNON", &clResult);
	checkOpenclErrors(clResult);
	kernel_roundNB = clCreateKernel(program, "FluffyRoundNON", &clResult);
	checkOpenclErrors(clResult);
	kernel_tail = clCreateKernel(program, "FluffyTailO", &clResult);
	checkOpenclErrors(clResult);

}

    u64 edgetrimmer::globalbytes() const {
		return (bufferA1_size + bufferA2_size + bufferB_size + indexesSize * 2) * sizeof(uint);
    }
    edgetrimmer::~edgetrimmer() {
		clReleaseMemObject(bufferA1);
		clReleaseMemObject(bufferA2);
		clReleaseMemObject(bufferB);
		clReleaseMemObject(bufferI1);
		clReleaseMemObject(bufferI2);
		clReleaseMemObject(recoveredges);
		clReleaseMemObject(bufferR);
		releaseCommandQueue(commandQueue);
		releaseProgram(program);
		releaseContext(context);
    }

    u32 edgetrimmer::trim(uint32_t device) {
	const u32 ZERO = 0;

	size_t tmpSize = 64*64*4;
	cl_event event;

	cl_int clResult;
	clResult = clEnqueueFillBuffer(commandQueue, bufferI2, &ZERO, sizeof (int), 0, tmpSize, 0, NULL, NULL);
	clResult |= clEnqueueFillBuffer(commandQueue, bufferI1, &ZERO, sizeof (int), 0, tmpSize, 0, NULL, NULL);
	checkOpenclErrors(clResult);

//seedA
	size_t global_work_size[1];
	size_t local_work_size[1];
	global_work_size[0] = tp.genA.blocks * tp.genA.tpb;
	local_work_size[0] = tp.genA.tpb;
	clResult |= clSetKernelArg(kernel_seedA, 0, sizeof (u64), &sipkeys.k0);
	clResult |= clSetKernelArg(kernel_seedA, 1, sizeof (u64), &sipkeys.k1);
	clResult |= clSetKernelArg(kernel_seedA, 2, sizeof (u64),  &sipkeys.k2);
	clResult |= clSetKernelArg(kernel_seedA, 3, sizeof (u64), &sipkeys.k3);
	clResult |= clSetKernelArg(kernel_seedA, 4, sizeof (cl_mem), (void*)&bufferB);
	clResult |= clSetKernelArg(kernel_seedA, 5, sizeof (cl_mem), (void*)&bufferA1);
	clResult |= clSetKernelArg(kernel_seedA, 6, sizeof (cl_mem), (void*)&bufferI1);
	clResult |= clEnqueueNDRangeKernel(commandQueue, kernel_seedA, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
	
	checkOpenclErrors(clResult);

//seedB1
	global_work_size[0] = tp.genB.blocks * tp.genB.tpb;
	local_work_size[0] = tp.genB.tpb;
	u32 generic_param = 32;
	clResult |= clSetKernelArg(kernel_seedB1, 0, sizeof (cl_mem), (void *) &bufferA1);
	clResult |= clSetKernelArg(kernel_seedB1, 1, sizeof (cl_mem), (void *) &bufferA1);
	clResult |= clSetKernelArg(kernel_seedB1, 2, sizeof (cl_mem), (void *) &bufferA2);
	clResult |= clSetKernelArg(kernel_seedB1, 3, sizeof (cl_mem), (void *) &bufferI1);
	clResult |= clSetKernelArg(kernel_seedB1, 4, sizeof (cl_mem), (void *) &bufferI2);
	clResult |= clSetKernelArg(kernel_seedB1, 5, sizeof (u32), &generic_param);
	checkOpenclErrors(clResult);
	clResult = clEnqueueNDRangeKernel(commandQueue, kernel_seedB1, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
	checkOpenclErrors(clResult);

//seedB2
	global_work_size[0] = tp.genB.blocks * tp.genB.tpb;
	local_work_size[0] = tp.genB.tpb;
	generic_param = 0;
	clResult |= clSetKernelArg(kernel_seedB2, 0, sizeof (cl_mem), (void *) &bufferB);
	clResult |= clSetKernelArg(kernel_seedB2, 1, sizeof (cl_mem), (void *) &bufferA1);
	clResult |= clSetKernelArg(kernel_seedB2, 2, sizeof (cl_mem), (void *) &bufferA2);
	clResult |= clSetKernelArg(kernel_seedB2, 3, sizeof (cl_mem), (void *) &bufferI1);
	clResult |= clSetKernelArg(kernel_seedB2, 4, sizeof (cl_mem), (void *) &bufferI2);
	clResult |= clSetKernelArg(kernel_seedB2, 5, sizeof (u32), &generic_param);
	checkOpenclErrors(clResult);
	clResult = clEnqueueNDRangeKernel(commandQueue, kernel_seedB2, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
	checkOpenclErrors(clResult);

	clResult = clEnqueueFillBuffer(commandQueue, bufferI1, &ZERO, sizeof (int), 0, tmpSize, 0, NULL, NULL);
	checkOpenclErrors(clResult);

//round1
	global_work_size[0] = tp.trim.blocks * tp.trim.tpb;
	local_work_size[0] = tp.trim.tpb;
	u32 edges_a = DUCK_SIZE_A * SUB_BUCKET_SIZE;
	u32 edges_b = DUCK_SIZE_B * SUB_BUCKET_SIZE;
	clResult |= clSetKernelArg(kernel_round1, 0, sizeof (cl_mem), (void *) &bufferA1);
	clResult |= clSetKernelArg(kernel_round1, 1, sizeof (cl_mem), (void *) &bufferA2);
	clResult |= clSetKernelArg(kernel_round1, 2, sizeof (cl_mem), (void *) &bufferB);
	clResult |= clSetKernelArg(kernel_round1, 3, sizeof (cl_mem), (void *) &bufferI2);
	clResult |= clSetKernelArg(kernel_round1, 4, sizeof (cl_mem), (void *) &bufferI1);
	clResult |= clSetKernelArg(kernel_round1, 5, sizeof (u32), &edges_a);
	clResult |= clSetKernelArg(kernel_round1, 6, sizeof (u32), &edges_b);
	clResult |= clEnqueueNDRangeKernel(commandQueue, kernel_round1, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
	checkOpenclErrors(clResult);


	clResult = clEnqueueFillBuffer(commandQueue, bufferI2, &ZERO, sizeof (int), 0, tmpSize, 0, NULL, NULL);
	checkOpenclErrors(clResult);

//round0
	clResult |= clSetKernelArg(kernel_round0, 0, sizeof (cl_mem), (void *) &bufferB);
	clResult |= clSetKernelArg(kernel_round0, 1, sizeof (cl_mem), (void *) &bufferA1);
	clResult |= clSetKernelArg(kernel_round0, 2, sizeof (cl_mem), (void *) &bufferI1);
	clResult |= clSetKernelArg(kernel_round0, 3, sizeof (cl_mem), (void *) &bufferI2);
	clResult |= clEnqueueNDRangeKernel(commandQueue, kernel_round0, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
	checkOpenclErrors(clResult);

	clResult = clEnqueueFillBuffer(commandQueue, bufferI1, &ZERO, sizeof (int), 0, tmpSize, 0, NULL, NULL);
	checkOpenclErrors(clResult);

//roundNB
	clResult |= clSetKernelArg(kernel_roundNB, 0, sizeof (cl_mem), (void *) &bufferA1);
	clResult |= clSetKernelArg(kernel_roundNB, 1, sizeof (cl_mem), (void *) &bufferB);
	clResult |= clSetKernelArg(kernel_roundNB, 2, sizeof (cl_mem), (void *) &bufferI2);
	clResult |= clSetKernelArg(kernel_roundNB, 3, sizeof (cl_mem), (void *) &bufferI1);
	clResult |= clEnqueueNDRangeKernel(commandQueue, kernel_roundNB, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
	checkOpenclErrors(clResult);

	for (int round = 0; round < 80; round += 1)
	{
	    clResult = clEnqueueFillBuffer(commandQueue, bufferI2, &ZERO, sizeof (int), 0, tmpSize, 0, NULL, NULL);
		checkOpenclErrors(clResult);

	//roundNA
		clResult |= clSetKernelArg(kernel_roundNA, 0, sizeof (cl_mem), (void *) &bufferB);
		clResult |= clSetKernelArg(kernel_roundNA, 1, sizeof (cl_mem), (void *) &bufferA1);
		clResult |= clSetKernelArg(kernel_roundNA, 2, sizeof (cl_mem), (void *) &bufferI1);
		clResult |= clSetKernelArg(kernel_roundNA, 3, sizeof (cl_mem), (void *) &bufferI2);
		clResult |= clEnqueueNDRangeKernel(commandQueue, kernel_roundNA, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
		checkOpenclErrors(clResult);

		clResult = clEnqueueFillBuffer(commandQueue, bufferI1, &ZERO, sizeof (int), 0, tmpSize, 0, NULL, NULL);
		checkOpenclErrors(clResult);

	//roundNB
		clResult |= clSetKernelArg(kernel_roundNB, 0, sizeof (cl_mem), (void *) &bufferA1);
		clResult |= clSetKernelArg(kernel_roundNB, 1, sizeof (cl_mem), (void *) &bufferB);
		clResult |= clSetKernelArg(kernel_roundNB, 2, sizeof (cl_mem), (void *) &bufferI2);
		clResult |= clSetKernelArg(kernel_roundNB, 3, sizeof (cl_mem), (void *) &bufferI1);
		clResult |= clEnqueueNDRangeKernel(commandQueue, kernel_roundNB, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
		checkOpenclErrors(clResult);
	}

	clResult = clEnqueueFillBuffer(commandQueue, bufferI2, &ZERO, sizeof (int), 0, tmpSize, 0, NULL, NULL);
	checkOpenclErrors(clResult);
//tail
	global_work_size[0] = tp.tail.blocks * tp.tail.tpb;
	local_work_size[0] = tp.tail.tpb;
	clResult |= clSetKernelArg(kernel_tail, 0, sizeof (cl_mem), (void *) &bufferB);
	clResult |= clSetKernelArg(kernel_tail, 1, sizeof (cl_mem), (void *) &bufferA1);
	clResult |= clSetKernelArg(kernel_tail, 2, sizeof (cl_mem), (void *) &bufferI1);
	clResult |= clSetKernelArg(kernel_tail, 3, sizeof (cl_mem), (void *) &bufferI2);
	clResult |= clEnqueueNDRangeKernel(commandQueue, kernel_tail, 1, NULL,
				   global_work_size, local_work_size, 0, NULL,
				   &event);
	checkOpenclErrors(clResult);
	clFinish(commandQueue);

	clResult = clEnqueueReadBuffer(commandQueue, bufferI2, CL_TRUE, 0, sizeof (u32), &nedges, 0, NULL, NULL);
	checkOpenclErrors(clResult);
	clFinish(commandQueue);

	return nedges;
    }

};
