// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license
#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <vector>
//#include <algorithm>
#include <stdint.h>
#include <sys/time.h>			// gettimeofday
#include <unistd.h>
#include <sys/types.h>
#include "trimmer.h"
#include "solver.h"

namespace cuckoogpu
{

	class cuckoo_hash
	{
	  public:
		u64 * cuckoo;

		cuckoo_hash ()
		{
			cuckoo = new u64[CUCKOO_SIZE];
			memset (cuckoo, 0, CUCKOO_SIZE * sizeof (u64));
		}
		 ~cuckoo_hash ()
		{
			delete[]cuckoo;
		}
		void set (node_t u, node_t v)
		{
			u64 niew = (u64) u << NODEBITS | v;
			for (node_t ui = (u >> IDXSHIFT) & CUCKOO_MASK;; ui = (ui + 1) & CUCKOO_MASK)
			{
				u64 old = cuckoo[ui];
				if (old == 0 || (old >> NODEBITS) == (u & KEYMASK))
				{
					cuckoo[ui] = niew;
					return;
				}
			}
		}
		node_t operator[] (node_t u) const
		{
			for (node_t ui = (u >> IDXSHIFT) & CUCKOO_MASK;; ui = (ui + 1) & CUCKOO_MASK)
			{
				u64 cu = cuckoo[ui];
				if (!cu)
					  return 0;
				if ((cu >> NODEBITS) == (u & KEYMASK))
				{
					assert (((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
					return (node_t) (cu & NODEMASK);
				}
			}
		}
	};

	const static u32 MAXPATHLEN = 8 << ((NODEBITS + 2) / 3);

	int nonce_cmp (const void *a, const void *b)
	{
		return *(u32 *) a - *(u32 *) b;
	}

	struct cuckoo_solver_ctx : public solver_ctx
	{
		cl_uint2 *edges;
		cuckoo_hash *cuckoo;
		cl_uint2 soledges[PROOFSIZE];
		u32 us[MAXPATHLEN];
		u32 vs[MAXPATHLEN];

  		cuckoo_solver_ctx(){}
		cuckoo_solver_ctx (const trimparams tp, uint32_t _device = 0, cl_context context = NULL, cl_command_queue commandQueue = NULL, cl_program program = NULL)
		{
			trimmer = new edgetrimmer (tp, context, commandQueue, program, 0);
			edges = new cl_uint2[MAXEDGES];
			cuckoo = new cuckoo_hash ();
			device = _device;
		}
		void init (const trimparams tp, uint32_t _device = 0, cl_context context = NULL, cl_command_queue commandQueue = NULL, cl_program program = NULL)
		{
			trimmer = new edgetrimmer (tp, context, commandQueue, program, 0);
			edges = new cl_uint2[MAXEDGES];
			cuckoo = new cuckoo_hash ();
			device = _device;
		}
	       	void setheadernonce (char *const header, const uint64_t nonce)
		{
			uint64_t littleEndianNonce = htole64 (nonce);
			char headerBuf[40];
			memcpy (headerBuf, header, 32);
			memcpy (headerBuf + 32, static_cast < uint64_t * >(&littleEndianNonce), sizeof (nonce));
			setheader (headerBuf, 40, &trimmer->sipkeys);
		}

		~cuckoo_solver_ctx ()
		{
			delete cuckoo;
			delete[]edges;
			delete trimmer;
		}

		void recordedge (const u32 i, const u32 u2, const u32 v2)
		{
						if(i < PROOFSIZE){
										soledges[i].x = u2 / 2;
										soledges[i].y = v2 / 2;
						}
		}

		void solution (const u32 * us, u32 nu, const u32 * vs, u32 nv)
		{
			u32 ni = 0;
			recordedge (ni++, *us, *vs);
			while (nu--)
				recordedge (ni++, us[(nu + 1) & ~1], us[nu | 1]);	// u's in even position; v's in odd
			while (nv--)
				recordedge (ni++, vs[nv | 1], vs[(nv + 1) & ~1]);	// u's in odd position; v's in even
			assert (ni == PROOFSIZE);
			sols.resize (sols.size () + PROOFSIZE);

			cl_int clResult;
			clResult = clEnqueueWriteBuffer (trimmer->commandQueue, trimmer->bufferR, CL_TRUE, 0, sizeof (cl_uint2) * PROOFSIZE, soledges, 0, NULL, NULL);
			checkOpenclErrors(clResult);

			int initV = 0;
			clResult = clEnqueueFillBuffer (trimmer->commandQueue, trimmer->bufferI3, &initV, sizeof (int), 0, trimmer->indexesSize, 0, NULL, NULL);
			checkOpenclErrors(clResult);

			clFinish (trimmer->commandQueue);
			clResult |= clSetKernelArg (trimmer->kernel_recovery, 0, sizeof (u64), &trimmer->sipkeys2.k0);
			clResult |= clSetKernelArg (trimmer->kernel_recovery, 1, sizeof (u64), &trimmer->sipkeys2.k1);
			clResult |= clSetKernelArg (trimmer->kernel_recovery, 2, sizeof (u64), &trimmer->sipkeys2.k2);
			clResult |= clSetKernelArg (trimmer->kernel_recovery, 3, sizeof (u64), &trimmer->sipkeys2.k3);
			clResult |= clSetKernelArg (trimmer->kernel_recovery, 4, sizeof (cl_mem), (void*)&trimmer->bufferR);
			clResult |= clSetKernelArg (trimmer->kernel_recovery, 5, sizeof (cl_mem), (void*)&trimmer->bufferI3);
			checkOpenclErrors(clResult);

			cl_event event;
			size_t global_work_size[1], local_work_size[1];
			global_work_size[0] = trimmer->tp.recover.blocks * trimmer->tp.recover.tpb;
			local_work_size[0] = trimmer->tp.recover.tpb;
			clEnqueueNDRangeKernel (trimmer->commandQueue, trimmer->kernel_recovery, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
			clFinish (trimmer->commandQueue);
			clResult = clEnqueueReadBuffer (trimmer->commandQueue, trimmer->bufferI3, CL_TRUE, 0, PROOFSIZE * sizeof (u32), &sols[sols.size () - PROOFSIZE], 0, NULL, NULL);
			checkOpenclErrors(clResult);
			qsort (&sols[sols.size () - PROOFSIZE], PROOFSIZE, sizeof (u32), nonce_cmp);
		}

		u32 path (u32 u, u32 * us)
		{
			u32 nu, u0 = u;
			/* fprintf(stderr, "start %zu\n", u0); */
			for (nu = 0; u; u = (*cuckoo)[u])
			{
				if (nu >= MAXPATHLEN)
				{
					/* fprintf(stderr, "nu: %zu, u: %zu, Maxpathlen: %zu\n", nu, u, MAXPATHLEN); */
					while (nu-- && us[nu] != u) ;
					if (~nu)
					{
						printf ("illegal %4d-cycle from node %d\n", MAXPATHLEN - nu, u0);
						exit (0);
					}
					printf ("maximum path length exceeded\n");
					return 0;	// happens once in a million runs or so; signal trouble
				}
				us[nu++] = u;
			}
			/* fprintf(stderr, "path nu: %zu\n", nu); */
			return nu;
		}

		void addedge (cl_uint2 edge)
		{
			const u32 u0 = edge.x << 1, v0 = (edge.y << 1) | 1;

			if (u0)
			{
				u32 nu = path (u0, us), nv = path (v0, vs);
				if (!nu-- || !nv--)
					return;		// drop edge causing trouble

				if (us[nu] == vs[nv])
				{
					const u32 min = nu < nv ? nu : nv;
					for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
					const u32 len = nu + nv + 1;
					if (len == PROOFSIZE)
					{
						solution (us, nu, vs, nv);
					}
				}
				else if (nu < nv)
				{
					while (nu--)
						cuckoo->set (us[nu + 1], us[nu]);
					cuckoo->set (u0, v0);
				}
				else
				{
					while (nv--)
						cuckoo->set (vs[nv + 1], vs[nv]);
					cuckoo->set (v0, u0);
				}

			}
		}

		int findcycles (u32 nedges)
		{
			sols.clear ();
			memset (cuckoo->cuckoo, 0, CUCKOO_SIZE * sizeof (u64));
			for (u32 i = 0; i < nedges; i++)
			{
				addedge (edges[i]);
			}
			return sols.size() / PROOFSIZE;
		}

		int solve ()
		{
			u32 nedges = trimmer->trim (this->device);
			if (nedges > MAXEDGES)
			{
				fprintf (stderr, "OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges - MAXEDGES, MAXEDGES);
				nedges = MAXEDGES;
			}
			nedges = nedges & CUCKOO_MASK;
			cl_int clResult = clEnqueueReadBuffer (trimmer->commandQueue, trimmer->bufferA1,
				CL_TRUE, 0, nedges * 8, edges, 0, NULL,
				NULL);
			checkOpenclErrors(clResult);

//			findcycles (edges, nedges);
//			return sols.size () / PROOFSIZE;
			trimmer->sipkeys2 = trimmer->sipkeys;
			return nedges;
		}
	};

};								// end of namespace cuckoogpu
/*
cuckoogpu::solver_ctx * ctx = NULL;
int32_t FindSolutionsByGPU (uint8_t * header, uint64_t nonce, uint32_t threadId, result_t * result, uint32_t resultBuffSize, uint32_t * solLength, uint32_t * numSol)
{
	using namespace cuckoogpu;
	using std::vector;
	ctx[threadId].setheadernonce ((char *) header, nonce);	//TODO(tian)
	char headerInHex[65];
	for (uint32_t i = 0; i < 32; i++)
	{
		sprintf (headerInHex + 2 * i, "%02x", *((unsigned int8_t *) (header + i)));
	}
	headerInHex[64] = '\0';

	u32 nsols = ctx[threadId].solve ();
	vector < vector < u32 > >sols;
	vector < vector < u32 > >*solutions = &sols;
	for (unsigned s = 0; s < nsols; s++)
	{
		u32 *prf = &(ctx[threadId].sols[s * PROOFSIZE]);
		solutions->push_back (vector < u32 > ());
		vector < u32 > &sol = solutions->back ();
		for (uint32_t idx = 0; idx < PROOFSIZE; idx++)
		{
			sol.push_back (prf[idx]);
		}
	}
	*solLength = 0;
	*numSol = sols.size ();
	if (sols.size () == 0)
		return 0;
	*solLength = uint32_t (sols[0].size ());
	for (size_t n = 0; n < std::min (sols.size (), (size_t) resultBuffSize / (*solLength)); n++)
	{
		vector < u32 > &sol = sols[n];
		for (size_t i = 0; i < sol.size (); i++)
		{
			result[i + n * (*solLength)] = sol[i];
		}
	}
	return nsols > 0;

}

void initOne (uint32_t index, uint32_t device)
{
	printf ("thread: %d\n", getpid ());
	using namespace cuckoogpu;
	using std::vector;

	trimparams tp;
	//TODO(tian) make use of multiple gpu
	cl_platform_id platformId = getOnePlatform ();
	if (platformId == NULL)
		return;
//	getPlatformInfo (platformId);
	cl_device_id deviceId = getOneDevice (platformId, device);
	if (deviceId == NULL)
		return;
	cl_context context = createContext (platformId, deviceId);
	if (context == NULL)
		return;
	cl_command_queue commandQueue = createCommandQueue (context, deviceId);
	if (commandQueue == NULL)
		return;

	string sourceStr = get_kernel_source();
	size_t size = sourceStr.size();
	const char *source = sourceStr.c_str ();
	cl_program program = createProgram (context, &source, size);

//	cl_program program = createByBinaryFile("trimmer.bin", context, deviceId);	
	if (program == NULL){
		printf("create program error\n");
		return;
	}
	char options[1024] = "-I./";
	sprintf (options, "-I./ -DEDGEBITS=%d -DPROOFSIZE=%d", EDGEBITS, PROOFSIZE);
	
	buildProgram (program, &(deviceId), options);
//	saveBinaryFile(program, deviceId);
	cl_ulong maxThreadsPerBlock = 0;
	clGetDeviceInfo (deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (maxThreadsPerBlock), &maxThreadsPerBlock, NULL);
	assert (tp.genA.tpb <= maxThreadsPerBlock);
	assert (tp.genB.tpb <= maxThreadsPerBlock);
	assert (tp.trim.tpb <= maxThreadsPerBlock);
	assert (tp.tail.tpb <= maxThreadsPerBlock);
	assert (tp.recover.tpb <= maxThreadsPerBlock);
	ctx[index].init(tp, device, context, commandQueue, program);
	printf ("50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, tp.ntrims, NX);

	u64 bytes = ctx[index].trimmer->globalbytes ();
	int unit;
	for (unit = 0; bytes >= 10240; bytes >>= 10, unit++) ;
	printf ("Using %lld%cB of global memory.\n", bytes, " KMGT"[unit]);
}


void CuckooInitialize(uint32_t* devices, uint32_t deviceNum) {
    printf("thread: %d\n", getpid());
    using namespace cuckoogpu;
    using std::vector;

    ctx = new solver_ctx[deviceNum];

    for(uint32_t i = 0; i < deviceNum; i++){
    	initOne(i, devices[i]);
    }
}
*/
