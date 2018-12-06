// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license
#include "solver.h"
#include "graph.hpp"

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

namespace cuckoogpu
{

	struct cuckaroo_solver_ctx:public solver_ctx
	{
		cl_uint2 *edges;
		  graph < edge_t > *cg;
		cl_uint2 soledges[PROOFSIZE];

		  cuckaroo_solver_ctx ()
		{
		}
		cuckaroo_solver_ctx (trimparams tp, uint32_t _device = 0, cl_context context = NULL, cl_command_queue commandQueue = NULL, cl_program program = NULL)
		{
			tp.genA.tpb = 128;
			trimmer = new edgetrimmer (tp, context, commandQueue, program, 1);
			edges = new cl_uint2[MAXEDGES];
			device = _device;
			cg = new graph < edge_t > (MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT);
		}
		void init (trimparams tp, uint32_t _device = 0, cl_context context = NULL, cl_command_queue commandQueue = NULL, cl_program program = NULL)
		{
			tp.genA.tpb = 128;
			trimmer = new edgetrimmer (tp, context, commandQueue, program, 1);
			edges = new cl_uint2[MAXEDGES];
			device = _device;
			cg = new graph < edge_t > (MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT);
		}

		void setheadernonce (char *const header, const uint64_t nonce)
		{
			uint64_t littleEndianNonce = htole64 (nonce);
			char headerBuf[40];
			memcpy (headerBuf, header, 32);
			memcpy (headerBuf + 32, static_cast < uint64_t * >(&littleEndianNonce), sizeof (nonce));
			setheader (headerBuf, 40, &trimmer->sipkeys);
			sols.clear ();
		}

		~cuckaroo_solver_ctx ()
		{
			delete[]edges;
			delete trimmer;
			delete cg;
		}

		int findcycles (cl_uint2 * edges, u32 nedges)
		{
			cg->reset ();
			for (u32 i = 0; i < nedges; i++)
			{
				cg->add_compress_edge (edges[i].x, edges[i].y);
			}
			for (u32 s = 0; s < cg->nsols; s++)
			{
				// print_log("Solution");
				for (u32 j = 0; j < PROOFSIZE; j++)
				{
					soledges[j] = edges[cg->sols[s][j]];
					// print_log(" (%x, %x)", soledges[j].x, soledges[j].y);
				}
				// print_log("\n");
				sols.resize (sols.size () + PROOFSIZE);
				cl_int clResult;
				clResult = clEnqueueWriteBuffer (trimmer->commandQueue, trimmer->recoveredges, CL_TRUE, 0, sizeof (cl_uint2) * PROOFSIZE, soledges, 0, NULL, NULL);
				checkOpenclErrors (clResult);

				int initV = 0;
				clResult = clEnqueueFillBuffer (trimmer->commandQueue, trimmer->indexesE2, &initV, sizeof (int), 0, trimmer->indexesSize, 0, NULL, NULL);
				checkOpenclErrors (clResult);

				clFinish (trimmer->commandQueue);
				cl_kernel recovery_kernel = clCreateKernel (trimmer->program, "Cuckaroo_Recovery", &clResult);
				clResult |= clSetKernelArg (recovery_kernel, 0, sizeof (cl_mem), (void *) &trimmer->dipkeys);
				clResult |= clSetKernelArg (recovery_kernel, 1, sizeof (cl_mem), (void *) &trimmer->indexesE2);
				clResult |= clSetKernelArg (recovery_kernel, 2, sizeof (cl_mem), (void *) &trimmer->recoveredges);
				checkOpenclErrors (clResult);

				cl_event event;
				size_t global_work_size[1], local_work_size[1];
				global_work_size[0] = trimmer->tp.recover.blocks * trimmer->tp.recover.tpb;
				local_work_size[0] = trimmer->tp.recover.tpb;
				clEnqueueNDRangeKernel (trimmer->commandQueue, recovery_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
				clFinish (trimmer->commandQueue);
				clResult = clEnqueueReadBuffer (trimmer->commandQueue, trimmer->indexesE2, CL_TRUE, 0, PROOFSIZE * sizeof (u32), &sols[sols.size () - PROOFSIZE], 0, NULL, NULL);
				checkOpenclErrors (clResult);
				qsort (&sols[sols.size () - PROOFSIZE], PROOFSIZE, sizeof (u32), cg->nonce_cmp);
			}
			return 0;
		}
		int solve ()
		{
			u32 nedges = trimmer->trim (this->device);
			if (nedges > MAXEDGES)
			{
				fprintf (stderr, "OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges - MAXEDGES, MAXEDGES);
				nedges = MAXEDGES;
			}
			// nedges must less then CUCKOO_SIZE, or find-cycle procedure will never stop.
			nedges = nedges & CUCKOO_MASK;
			cl_int clResult = clEnqueueReadBuffer (trimmer->commandQueue, trimmer->bufferB,
				CL_TRUE, trimmer->sizeA, nedges * 8, edges, 0,
				NULL,
				NULL);
			checkOpenclErrors (clResult);
			findcycles (edges, nedges);
			return sols.size () / PROOFSIZE;
		}
	};

};								// end of namespace cuckoogpu
