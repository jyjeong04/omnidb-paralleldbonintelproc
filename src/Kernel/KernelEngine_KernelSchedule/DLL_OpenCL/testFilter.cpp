#include "common.h"
#include "testFilter.h"
#include "KernelScheduler.h"
#include "testScan.h"
#include "Helper.h"
#include "PrimitiveCommon.h"
#include "KernelScheduler.h"
#include "OpenCL_DLL.h"
#include "scheduler.h"
#include <sys/time.h>

// --- PHASE_TIMING: sub-phase wall-clock timing, gated by env PHASE_TIMING ---
// Independent of the engine's DLL_genTimer/DLL_getTimer state. Returns seconds.
static int    ph_on = -1;          // -1=uninit, 0=off, 1=on
static inline int ph_enabled() {
	if (ph_on < 0) ph_on = getenv("PHASE_TIMING") ? 1 : 0;
	return ph_on;
}
static inline double ph_now() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
// OpenCL Vars---------0 for CPU, 1 for GPU
extern cl_context Context;        // OpenCL context
extern cl_program Program;           // OpenCL program
extern cl_command_queue CommandQueue[2];// OpenCL command que
extern cl_platform_id Platform[2];      // OpenCL platform
extern cl_device_id Device[2];          // OpenCL device
extern cl_ulong totalLocalMemory[2];      /**< Max local memory allowed */
extern cl_device_id allDevices[10];

// PE (CPU-assisted prefetch) filter enqueue — used in place of kernel_enqueue for the
// filter map kernel (kid 20) when PE_MODE is on. Mirrors HJprobe_enqueue_PE (testHJ.cpp):
// makes the per-kernel burden-scheduling decision itself (one Kernelscheduler call), and
// ONLY when the filter lands on the CPU device with WASSIZE>0 does it engage WAS — a 1-CU
// WAS_kernel helper (mode 3) on PrefetchCommandQueue prefetches the input-record addresses
// (&d_Rin[pos]) the 7-CU filter posts. GPU-scheduled filters run noWAS (wassize=-1).
static void filter_enqueue_PE(cl_kernel *Kernel, int kid, int wasArgBase,
	int filterWorkSize, int wassize,
	size_t globalSize, size_t groupSize, cl_event *eventList, int *index,
	int *Flag_CPU_GPU, double *burden, int _CPU_GPU)
{
	int preFlag = *Flag_CPU_GPU;
	double preBurden = *burden;
	int CPU_GPU = Kernelscheduler(filterWorkSize, kid, Flag_CPU_GPU, burden, _CPU_GPU);
	static int forceCpu = -1;
	if (forceCpu < 0) forceCpu = getenv("PE_FORCE_CPU") ? atoi(getenv("PE_FORCE_CPU")) : 0;
	if (forceCpu) CPU_GPU = 0;
	*Flag_CPU_GPU = CPU_GPU;

	cl_int err = CL_SUCCESS;
	bool useWAS = (CPU_GPU == 0 && wassize > 0 && g_prefetchEnabled && PrefetchCommandQueue);
	if (useWAS) {
		static int announced = 0;
		if (!announced) { fprintf(stderr, "[PE] WAS filter engaged on CPU sub-device (wassize=%d)\n", wassize); announced = 1; }
	}
	cl_mem was_buffer = NULL, dummy_buffer = NULL, last_tag_buffer = NULL;
	cl_kernel hk = NULL;

	if (useWAS) {
		cl_command_queue Q = CommandQueue[0]; // 7-CU MainCPUSubDevice (repartitioned by cl_init_prefetch)
		size_t wasEntrySize = 4 * sizeof(cl_ulong);
		was_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)wassize * wasEntrySize, NULL, &err);
		dummy_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 64, NULL, &err);
		last_tag_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)wassize * sizeof(cl_ulong), NULL, &err);
		if (!was_buffer || !dummy_buffer || !last_tag_buffer) {
			printf("Error: PE WAS buffer alloc failed, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		void *dm = clEnqueueMapBuffer(Q, dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
		memset(dm, 0, 64);
		clEnqueueUnmapMemObject(Q, dummy_buffer, dm, 0, NULL, NULL);
		clFinish(Q);
		cl_ulong *wm = (cl_ulong *)clEnqueueMapBuffer(Q, was_buffer, CL_TRUE, CL_MAP_WRITE, 0, (size_t)wassize * wasEntrySize, 0, NULL, NULL, &err);
		void *dm2 = clEnqueueMapBuffer(Q, dummy_buffer, CL_TRUE, CL_MAP_READ, 0, 8, 0, NULL, NULL, &err);
		cl_ulong dummyVal = (cl_ulong)(uintptr_t)dm2;
		clEnqueueUnmapMemObject(Q, dummy_buffer, dm2, 0, NULL, NULL);
		for (int j = 0; j < wassize; j++) { wm[j*4] = dummyVal; wm[j*4+1] = dummyVal; wm[j*4+2] = (cl_ulong)-1; wm[j*4+3] = (cl_ulong)-1; }
		clEnqueueUnmapMemObject(Q, was_buffer, wm, 0, NULL, NULL);
		cl_ulong *lm = (cl_ulong *)clEnqueueMapBuffer(Q, last_tag_buffer, CL_TRUE, CL_MAP_WRITE, 0, (size_t)wassize * sizeof(cl_ulong), 0, NULL, NULL, &err);
		memset(lm, 0, (size_t)wassize * sizeof(cl_ulong));
		clEnqueueUnmapMemObject(Q, last_tag_buffer, lm, 0, NULL, NULL);
		clFinish(Q);
		// kernel runs WITH WAS (args wasArgBase..wasArgBase+2)
		clSetKernelArg(*Kernel, wasArgBase + 0, sizeof(cl_mem), &was_buffer);
		clSetKernelArg(*Kernel, wasArgBase + 1, sizeof(cl_int), &wassize);
		clSetKernelArg(*Kernel, wasArgBase + 2, sizeof(cl_mem), &dummy_buffer);
		// launch 1-CU helper (mode 3) on the prefetch sub-device
		hk = clCreateKernel(Program, "WAS_kernel", &err);
		cl_int wmode = 3;
		clSetKernelArg(hk, 0, sizeof(cl_mem), &was_buffer);
		clSetKernelArg(hk, 1, sizeof(cl_int), &wassize);
		clSetKernelArg(hk, 2, sizeof(cl_mem), &dummy_buffer);
		clSetKernelArg(hk, 3, sizeof(cl_mem), &last_tag_buffer);
		clSetKernelArg(hk, 4, sizeof(cl_int), &wmode);
		size_t wg = 1, wl = 1;
		clEnqueueNDRangeKernel(PrefetchCommandQueue, hk, 1, NULL, &wg, &wl, 0, NULL, NULL);
		clFlush(PrefetchCommandQueue);
	} else {
		// GPU (or fission unavailable): noWAS
		cl_mem nb = NULL; cl_int off = -1;
		clSetKernelArg(*Kernel, wasArgBase + 0, sizeof(cl_mem), &nb);
		clSetKernelArg(*Kernel, wasArgBase + 1, sizeof(cl_int), &off);
		clSetKernelArg(*Kernel, wasArgBase + 2, sizeof(cl_mem), &nb);
	}

	// Enqueue + event-chain/burden bookkeeping mirrors kernel_enqueue (KernelScheduler.cpp:108-134).
	size_t groups = globalSize, threads = groupSize;
	if (CPU_GPU && threads > 256) threads = 256;
	cl_int ciErr1;
	if (*index != 0) {
		ciErr1 = clEnqueueNDRangeKernel(CommandQueue[CPU_GPU], *Kernel, 1, NULL, &groups, &threads,
			1, &eventList[(*index - 1) % 2], &eventList[*index % 2]);
		deschedule(preFlag, preBurden);
	} else {
		ciErr1 = clEnqueueNDRangeKernel(CommandQueue[CPU_GPU], *Kernel, 1, NULL, &groups, &threads,
			0, NULL, &eventList[*index]);
	}
	(*index)++;
	if (ciErr1 != CL_SUCCESS) {
		printf("Error %d in clEnqueueNDRangeKernel (PE filter), Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	clFlush(CommandQueue[CPU_GPU]);

	if (useWAS) {
		clFinish(CommandQueue[0]);          // filter complete
		cl_uint *cs = (cl_uint *)clEnqueueMapBuffer(CommandQueue[0], dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
		cs[1] = 1;                          // signal helper to stop
		clEnqueueUnmapMemObject(CommandQueue[0], dummy_buffer, cs, 0, NULL, NULL);
		clFinish(CommandQueue[0]);
		clFinish(PrefetchCommandQueue);
		clReleaseKernel(hk);
		clReleaseMemObject(was_buffer);
		clReleaseMemObject(dummy_buffer);
		clReleaseMemObject(last_tag_buffer);
	}
}

void filterImpl_map_int(cl_mem d_Rin, int beginPos, int rLen,
					cl_mem d_mark, int smallKey, int largeKey, cl_mem d_temp,
					int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x=numThreadPB;
	size_t globalWorkingSetSize=numThreadPB*numBlock;

	cl_getKernel("filterImpl_map_kernel",Kernel);

    // Set the Argument values
    cl_int ciErr1 = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)&d_Rin);	
	ciErr1 |= clSetKernelArg((*Kernel), 1, sizeof(cl_int), (void*)&beginPos);
	ciErr1 |= clSetKernelArg((*Kernel), 2, sizeof(cl_int), (void*)&rLen);
	ciErr1 = clSetKernelArg((*Kernel), 3, sizeof(cl_mem), (void*)&d_mark);	
	ciErr1 |= clSetKernelArg((*Kernel), 4, sizeof(cl_int), (void*)&smallKey);
	ciErr1 |= clSetKernelArg((*Kernel), 5, sizeof(cl_int), (void*)&largeKey);
    ciErr1 |= clSetKernelArg((*Kernel), 6, sizeof(cl_mem), (void*)&d_temp);
	// filterImpl_map_kernel gained 3 WAS args (7,8,9). Legacy path is noWAS:
	// wassize=-1 selects the noWAS branch, so was_buffer/dummy_buffer are unused (NULL).
	cl_mem wasNull = NULL;
	int    wasOff  = -1;
	ciErr1 |= clSetKernelArg((*Kernel), 7, sizeof(cl_mem), (void*)&wasNull);
	ciErr1 |= clSetKernelArg((*Kernel), 8, sizeof(cl_int), (void*)&wasOff);
	ciErr1 |= clSetKernelArg((*Kernel), 9, sizeof(cl_mem), (void*)&wasNull);
       if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        cl_clean(EXIT_FAILURE);
    }
	// PE config: when PE_MODE is on, route the filter through the PE-aware enqueue so a
	// CPU-scheduled filter is assisted by the WAS helper (else plain kernel_enqueue).
	static int peMode = -1, peWas = 0;
	if (peMode < 0) {
		peMode = getenv("PE_MODE") ? atoi(getenv("PE_MODE")) : 0;
		peWas  = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
	}
	if (peMode) {
		filter_enqueue_PE(Kernel, 20, 7, rLen, peWas, globalWorkingSetSize, numThreadsPerBlock_x,
			eventList, index, Flag_CPU_GPU, burden, _CPU_GPU);
	} else {
		kernel_enqueue(rLen,20,
			1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	}

}
void filterImpl_outSize_int(cl_mem d_outSize,cl_mem d_mark,cl_mem d_markOutput,int rLen,int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU )
{
	size_t numThreadsPerBlock_x=numThreadPB;
	size_t globalWorkingSetSize=numThreadPB*numBlock;
	cl_getKernel("filterImpl_outSize_kernel",Kernel);
    // Set the Argument values
    cl_int ciErr1 = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)&d_outSize);	
	ciErr1 = clSetKernelArg((*Kernel), 1, sizeof(cl_mem), (void*)&d_mark);	
	ciErr1 = clSetKernelArg((*Kernel), 2, sizeof(cl_mem), (void*)&d_markOutput);
	ciErr1 = clSetKernelArg((*Kernel), 3, sizeof(cl_int), (void*)&rLen);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        cl_clean(EXIT_FAILURE);
    }
	kernel_enqueue(rLen, 21, 
		1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);


}
void filterImpl_write_int( cl_mem d_Rout, cl_mem d_Rin, cl_mem d_mark, 
						cl_mem d_markOutput, int beginPos, int rLen,
						int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x=numThreadPB;
	size_t globalWorkingSetSize=numThreadPB*numBlock;
	cl_getKernel("filterImpl_write_kernel",Kernel);
    // Set the Argument values
    cl_int ciErr1 = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)&d_Rout);	
	ciErr1 = clSetKernelArg((*Kernel), 1, sizeof(cl_mem), (void*)&d_Rin);	
	ciErr1 = clSetKernelArg((*Kernel), 2, sizeof(cl_mem), (void*)&d_mark);
	ciErr1 = clSetKernelArg((*Kernel), 3, sizeof(cl_mem), (void*)&d_markOutput);
	ciErr1 |= clSetKernelArg((*Kernel), 4, sizeof(cl_int), (void*)&beginPos);
	ciErr1 |= clSetKernelArg((*Kernel), 5, sizeof(cl_int), (void*)&rLen);
	// filterImpl_write_kernel gained 3 WAS args (6,7,8). Legacy path is noWAS (wassize=-1).
	cl_mem wWasNull = NULL;
	int    wWasOff  = -1;
	ciErr1 |= clSetKernelArg((*Kernel), 6, sizeof(cl_mem), (void*)&wWasNull);
	ciErr1 |= clSetKernelArg((*Kernel), 7, sizeof(cl_int), (void*)&wWasOff);
	ciErr1 |= clSetKernelArg((*Kernel), 8, sizeof(cl_mem), (void*)&wWasNull);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        cl_clean(EXIT_FAILURE);
    }
	// PE: when PE_MODE is on, route the scatter/write (kid 23) through the PE-aware enqueue
	// too. WAS args are at base index 6 here (write kernel has 6 original args, 0..5).
	static int peModeW = -1, peWasW = 0;
	if (peModeW < 0) {
		peModeW = getenv("PE_MODE") ? atoi(getenv("PE_MODE")) : 0;
		peWasW  = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
	}
	if (peModeW) {
		filter_enqueue_PE(Kernel, 23, 6, rLen, peWasW, globalWorkingSetSize, numThreadsPerBlock_x,
			eventList, index, Flag_CPU_GPU, burden, _CPU_GPU);
	} else {
		kernel_enqueue(rLen, 23,
			1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	}
}

// --- Tiled-CPU WAS lifecycle (shared by filterImpl + filterImpl_ns tiled paths) ---
// Mirrors the WAS setup/teardown of CL_RangeSelectionOnly_cPDE: allocates the WAS ring +
// dummy ctrl + last-tag buffers, launches the 1-CU WAS_kernel helper (mode 3) on
// PrefetchCommandQueue, and returns the handles. The tiled block loop then sets each map
// kernel's WAS args (base..base+2) to (was_buffer / wassize / dummy_buffer) so the 7-CU
// CPU map posts the input-record addresses it is about to dereference; the helper prefetches
// them. tile_was_end() signals the helper to stop (ctrl flag) and frees everything.
// ENGAGED only for CPU-tiled (!MAP_GPU) when PE_MODE is set AND WASSIZE>0 (caller decides).
struct TileWAS {
	int      active;
	cl_mem   was_buffer;
	cl_mem   dummy_buffer;
	cl_mem   last_tag_buffer;
	cl_kernel hk;
	int      wassize;
};
static void tile_was_begin(struct TileWAS *w, int wassize, cl_command_queue Qw)
{
	w->active = 0; w->was_buffer = NULL; w->dummy_buffer = NULL;
	w->last_tag_buffer = NULL; w->hk = NULL; w->wassize = wassize;
	if (!(wassize > 0 && g_prefetchEnabled && PrefetchCommandQueue)) return;
	cl_int err = CL_SUCCESS;
	size_t wasEntrySize = 4 * sizeof(cl_ulong);
	w->was_buffer      = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)wassize * wasEntrySize, NULL, &err);
	w->dummy_buffer    = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 64, NULL, &err);
	w->last_tag_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)wassize * sizeof(cl_ulong), NULL, &err);
	if (!w->was_buffer || !w->dummy_buffer || !w->last_tag_buffer) {
		printf("Error: tiled-WAS buffer alloc failed, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	void *dm = clEnqueueMapBuffer(Qw, w->dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
	memset(dm, 0, 64);
	clEnqueueUnmapMemObject(Qw, w->dummy_buffer, dm, 0, NULL, NULL);
	clFinish(Qw);
	cl_ulong *wm  = (cl_ulong *)clEnqueueMapBuffer(Qw, w->was_buffer, CL_TRUE, CL_MAP_WRITE, 0, (size_t)wassize * wasEntrySize, 0, NULL, NULL, &err);
	void *dm2 = clEnqueueMapBuffer(Qw, w->dummy_buffer, CL_TRUE, CL_MAP_READ, 0, 8, 0, NULL, NULL, &err);
	cl_ulong dummyVal = (cl_ulong)(uintptr_t)dm2;
	clEnqueueUnmapMemObject(Qw, w->dummy_buffer, dm2, 0, NULL, NULL);
	for (int j = 0; j < wassize; j++) { wm[j*4] = dummyVal; wm[j*4+1] = dummyVal; wm[j*4+2] = (cl_ulong)-1; wm[j*4+3] = (cl_ulong)-1; }
	clEnqueueUnmapMemObject(Qw, w->was_buffer, wm, 0, NULL, NULL);
	cl_ulong *lm = (cl_ulong *)clEnqueueMapBuffer(Qw, w->last_tag_buffer, CL_TRUE, CL_MAP_WRITE, 0, (size_t)wassize * sizeof(cl_ulong), 0, NULL, NULL, &err);
	memset(lm, 0, (size_t)wassize * sizeof(cl_ulong));
	clEnqueueUnmapMemObject(Qw, w->last_tag_buffer, lm, 0, NULL, NULL);
	clFinish(Qw);
	w->hk = clCreateKernel(Program, "WAS_kernel", &err);
	cl_int wmode = 3;   // backward + spin + no-pause (the fixed CPU-optimal variant)
	clSetKernelArg(w->hk, 0, sizeof(cl_mem), &w->was_buffer);
	clSetKernelArg(w->hk, 1, sizeof(cl_int), &wassize);
	clSetKernelArg(w->hk, 2, sizeof(cl_mem), &w->dummy_buffer);
	clSetKernelArg(w->hk, 3, sizeof(cl_mem), &w->last_tag_buffer);
	clSetKernelArg(w->hk, 4, sizeof(cl_int), &wmode);
	size_t wg = 1, wl = 1;
	err = clEnqueueNDRangeKernel(PrefetchCommandQueue, w->hk, 1, NULL, &wg, &wl, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error %d launching tiled-WAS helper, Line %u in file %s !!!\n\n", err, __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	clFlush(PrefetchCommandQueue);
	w->active = 1;
	static int announced = 0;
	if (!announced) { fprintf(stderr, "[PE] WAS tiled-CPU map engaged on prefetch sub-device (wassize=%d)\n", wassize); announced = 1; }
}
static void tile_was_end(struct TileWAS *w, cl_command_queue Qe)
{
	if (!w->active) return;
	cl_int err = CL_SUCCESS;
	cl_uint *cs = (cl_uint *)clEnqueueMapBuffer(Qe, w->dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
	cs[1] = 1;                          // signal helper to stop
	clEnqueueUnmapMemObject(Qe, w->dummy_buffer, cs, 0, NULL, NULL);
	clFinish(Qe);
	clFinish(PrefetchCommandQueue);
	clReleaseKernel(w->hk);
	clReleaseMemObject(w->was_buffer);
	clReleaseMemObject(w->dummy_buffer);
	clReleaseMemObject(w->last_tag_buffer);
	w->active = 0;
}

void filterImpl( cl_mem d_Rin, int beginPos, int rLen, cl_mem* d_Rout, int* outSize,
				int numThread, int numBlock, int smallKey, int largeKey,int *index,cl_event *eventList,cl_kernel *Kernel, int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_mem d_mark;
	CL_MALLOC( &d_mark, sizeof(int)*rLen ) ;

	cl_mem d_markOutput;
	CL_MALLOC( &d_markOutput, sizeof(int)*rLen ) ;

	cl_mem d_temp;
	CL_MALLOC( &d_temp, sizeof(int)*rLen ) ;

	// PHASE_TIMING (E-CPU): accumulators summed across the 15 query calls.
	static double ph_map_s = 0, ph_scan_s = 0, ph_write_s = 0;
	static int    ph_calls = 0;
	double t_map = 0, t_scan = 0, t_write = 0, t0 = 0;
	const int phT = ph_enabled();
	if (phT) t0 = ph_now();

	// ECPU_TILE_MAP toggle: when set, the map phase runs in 32K-tuple BLOCKS instead of one
	// monolithic launch over the full rLen, mirroring cDE-CPU's per-block map (CL_RangeSelectionOnly_cPDE)
	// but operating DIRECTLY on the 8-byte input d_Rin — NO decompression, NO compression. Only the
	// map-launch granularity changes; scan/outSize/write over the full rLen are untouched. Isolates
	// whether E-CPU's slow map is purely the monolithic-over-16M launch granularity.
	//
	// MAP_GPU toggle: selects which device queue runs the map phase.
	//   mapQ = MAP_GPU ? CommandQueue[1] (GPU) : CommandQueue[0] (CPU).
	// 2x2 matrix (both gated; both-unset == byte-identical original scheduler call):
	//   (mapGpu=0, tile=0) -> original filterImpl_map_int via burden scheduler (UNCHANGED default)
	//   (mapGpu=1, tile=0) -> single monolithic clEnqueueNDRangeKernel directly on the GPU queue
	//   (mapGpu=0, tile=1) -> 512x32K sub-buffer block loop on the CPU queue   (current tiled-CPU)
	//   (mapGpu=1, tile=1) -> 512x32K sub-buffer block loop on the GPU queue    (tiled-GPU)
	// Only the map launch's queue/granularity changes; scan/outSize/write are untouched.
	static int ecpuTileMap = -1;
	if (ecpuTileMap < 0) ecpuTileMap = getenv("ECPU_TILE_MAP") ? 1 : 0;
	static int mapGpu = -1;
	if (mapGpu < 0) mapGpu = getenv("MAP_GPU") ? 1 : 0;
	cl_command_queue mapQ = mapGpu ? CommandQueue[1] : CommandQueue[0];
	if (!ecpuTileMap && !mapGpu) {
		// DEFAULT PATH — byte-identical to the original: burden scheduler picks CPU/GPU.
		filterImpl_map_int( d_Rin, beginPos, rLen, d_mark, smallKey, largeKey, d_temp, numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
		clWaitForEvents(1,&eventList[(*index-1)%2]);
	} else if (!ecpuTileMap && mapGpu) {
		// MONOLITHIC-ON-mapQ (GPU): one direct launch over the full rLen, replicating the
		// monolithic filterImpl_map_int kernel args (same predicate bounds, same noWAS -1/NULL),
		// but bypassing the burden scheduler and going straight to mapQ (= CommandQueue[1]).
		cl_int errM = CL_SUCCESS;
		cl_kernel kMapM = clCreateKernel(Program, "filterImpl_map_kernel", &errM);
		if (errM != CL_SUCCESS || !kMapM) {
			printf("Error %d creating MAP_GPU monolithic kernel, Line %u in file %s !!!\n\n", errM, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		cl_mem wasNullM = NULL; cl_int wasOffM = -1;
		int beginPosM = 0;
		clSetKernelArg(kMapM, 0, sizeof(cl_mem), &d_Rin);
		clSetKernelArg(kMapM, 1, sizeof(cl_int), &beginPosM);
		clSetKernelArg(kMapM, 2, sizeof(cl_int), &rLen);
		clSetKernelArg(kMapM, 3, sizeof(cl_mem), &d_mark);
		clSetKernelArg(kMapM, 4, sizeof(cl_int), &smallKey);
		clSetKernelArg(kMapM, 5, sizeof(cl_int), &largeKey);
		clSetKernelArg(kMapM, 6, sizeof(cl_mem), &d_temp);
		clSetKernelArg(kMapM, 7, sizeof(cl_mem), &wasNullM);   // noWAS (matches monolithic filterImpl_map_int)
		clSetKernelArg(kMapM, 8, sizeof(cl_int), &wasOffM);
		clSetKernelArg(kMapM, 9, sizeof(cl_mem), &wasNullM);
		size_t gMono = (size_t)numThread * (size_t)numBlock;   // 131072 (= numThreadPB*numBlock, same as scheduler)
		size_t lMono = (size_t)numThread;                       // 256
		errM = clEnqueueNDRangeKernel(mapQ, kMapM, 1, NULL, &gMono, &lMono, 0, NULL, NULL);
		if (errM != CL_SUCCESS) {
			printf("Error %d enqueue MAP_GPU monolithic map, Line %u in file %s !!!\n\n", errM, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		clFinish(mapQ);
		clReleaseKernel(kMapM);
	} else {
		// TILED-ON-mapQ: 512x32K sub-buffer block loop. mapGpu picks CPU (CommandQueue[0]) or
		// GPU (CommandQueue[1]) for every block enqueue.
		cl_int errT = CL_SUCCESS;
		cl_kernel kMapT = clCreateKernel(Program, "filterImpl_map_kernel", &errT);
		if (errT != CL_SUCCESS || !kMapT) {
			printf("Error %d creating ECPU_TILE_MAP kernel, Line %u in file %s !!!\n\n", errT, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		cl_mem wasNullT = NULL; cl_int wasOffT = -1;
		int B = numBlock > 0 ? numBlock : 8;     // 512 (same block count cDE-CPU uses)
		if (B > rLen) B = 1;
		int blkLen = (rLen + B - 1) / B;          // 32768 at rLen=16M
		size_t lE = (size_t)numThread;            // 256, == cDE-CPU's lThreads
		int beginPosT = 0;
		// PE-CPU WAS on the tiled path: engage only when PE_MODE set AND tiling is on the CPU
		// (!mapGpu). GPU-tiled stays noWAS (WAS is CPU-side; the GPU map won't benefit). The
		// helper wraps the WHOLE block loop (one launch on PrefetchCommandQueue); each block's
		// map kernel posts &d_Rin[pos] into the shared WAS ring.
		static int tilePeMode = -1, tilePeWas = 0;
		if (tilePeMode < 0) {
			tilePeMode = getenv("PE_MODE") ? atoi(getenv("PE_MODE")) : 0;
			tilePeWas  = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
		}
		struct TileWAS tw;
		int wasOn = (tilePeMode && !mapGpu && tilePeWas > 0);
		if (wasOn) tile_was_begin(&tw, tilePeWas, mapQ); else tw.active = 0;
		for (int b = 0; b < B; b++) {
			int off = b * blkLen;
			int len = blkLen;
			if (off + len > rLen) len = rLen - off;
			if (len <= 0) break;
			// Sub-buffers of the EXISTING 8-byte input and output buffers for [off, off+len).
			cl_buffer_region rIn   = { (size_t)off * sizeof(Record), (size_t)len * sizeof(Record) };
			cl_buffer_region rMark = { (size_t)off * sizeof(int),    (size_t)len * sizeof(int)    };
			cl_mem subIn   = clCreateSubBuffer(d_Rin,  CL_MEM_READ_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &rIn,   &errT);
			cl_mem subMark = clCreateSubBuffer(d_mark, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rMark, &errT);
			cl_mem subTemp = clCreateSubBuffer(d_temp, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rMark, &errT);
			if (errT != CL_SUCCESS) {
				printf("Error %d creating ECPU_TILE_MAP sub-buffers (b=%d), Line %u in file %s !!!\n\n", errT, b, __LINE__, __FILE__);
				cl_clean(EXIT_FAILURE);
			}
			clSetKernelArg(kMapT, 0, sizeof(cl_mem), &subIn);
			clSetKernelArg(kMapT, 1, sizeof(cl_int), &beginPosT);
			clSetKernelArg(kMapT, 2, sizeof(cl_int), &len);
			clSetKernelArg(kMapT, 3, sizeof(cl_mem), &subMark);
			clSetKernelArg(kMapT, 4, sizeof(cl_int), &smallKey);
			clSetKernelArg(kMapT, 5, sizeof(cl_int), &largeKey);
			clSetKernelArg(kMapT, 6, sizeof(cl_mem), &subTemp);
			if (tw.active) {
				clSetKernelArg(kMapT, 7, sizeof(cl_mem), &tw.was_buffer);
				clSetKernelArg(kMapT, 8, sizeof(cl_int), &tw.wassize);
				clSetKernelArg(kMapT, 9, sizeof(cl_mem), &tw.dummy_buffer);
			} else {
				clSetKernelArg(kMapT, 7, sizeof(cl_mem), &wasNullT);  // noWAS (matches monolithic filterImpl_map_int)
				clSetKernelArg(kMapT, 8, sizeof(cl_int), &wasOffT);
				clSetKernelArg(kMapT, 9, sizeof(cl_mem), &wasNullT);
			}
			size_t gMap = (size_t)numThread * 8;   // 2048, == cDE-CPU's gMap (kernel strides over the block)
			errT = clEnqueueNDRangeKernel(mapQ, kMapT, 1, NULL, &gMap, &lE, 0, NULL, NULL);
			if (errT != CL_SUCCESS) {
				printf("Error %d enqueue ECPU_TILE_MAP map (b=%d), Line %u in file %s !!!\n\n", errT, b, __LINE__, __FILE__);
				cl_clean(EXIT_FAILURE);
			}
			clFlush(mapQ);
			clReleaseMemObject(subIn);
			clReleaseMemObject(subMark);
			clReleaseMemObject(subTemp);
		}
		clFinish(mapQ);
		if (tw.active) tile_was_end(&tw, mapQ);
		clReleaseKernel(kMapT);
	}
	if (phT) { clFinish(mapQ); t_map = ph_now() - t0; t0 = ph_now(); }
	//prefex sum
	ScanPara *SP;
	SP=(ScanPara*)malloc(sizeof(ScanPara));
	initScan(rLen,SP);
	scanImpl( d_mark, rLen, d_markOutput,index,eventList,Kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	closeScan(SP);

	//get the outSize
	cl_mem d_outSize;
	CL_MALLOC(&d_outSize, sizeof(int)) ;
	filterImpl_outSize_int(d_outSize, d_mark, d_markOutput, rLen,1,1,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	//bufferchecking(d_outSize,sizeof(int));
	cl_readbuffer(outSize,d_outSize, sizeof(int),index,eventList,Flag_CPU_GPU,burden,_CPU_GPU);
	if (phT) { clFinish(CommandQueue[0]); t_scan = ph_now() - t0; t0 = ph_now(); }

	//write the reduced result
	CL_MALLOC( d_Rout, sizeof(Record)*(*outSize) );
	filterImpl_write_int( *d_Rout, d_Rin, d_mark, d_markOutput, beginPos, rLen,numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU );
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	if (phT) {
		clFinish(CommandQueue[0]); t_write = ph_now() - t0;
		ph_map_s += t_map; ph_scan_s += t_scan; ph_write_s += t_write;
		ph_calls++;
		fprintf(stderr, "[PHASE-ECPU] q=%d map=%.4f scan=%.4f write=%.4f total=%.4f\n",
			ph_calls, t_map, t_scan, t_write, t_map + t_scan + t_write);
		if (ph_calls == 15) {
			fprintf(stderr, "[PHASE-ECPU-TOTAL] map=%.4f scan=%.4f write=%.4f total=%.4f\n",
				ph_map_s, ph_scan_s, ph_write_s, ph_map_s + ph_scan_s + ph_write_s);
		}
	}
	CL_FREE(d_mark);
	CL_FREE(d_markOutput);
	CL_FREE( d_temp );
}

void filterImpl_map_ns(cl_mem d_Rin16, int beginPos, int rLen,
					cl_mem d_mark, int smallKey, int largeKey, cl_mem d_temp,
					int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x=numThreadPB;
	size_t globalWorkingSetSize=numThreadPB*numBlock;

	cl_getKernel("filterImpl_map_kernel_ns",Kernel);

	cl_int ciErr1 = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)&d_Rin16);
	ciErr1 |= clSetKernelArg((*Kernel), 1, sizeof(cl_int), (void*)&beginPos);
	ciErr1 |= clSetKernelArg((*Kernel), 2, sizeof(cl_int), (void*)&rLen);
	ciErr1 |= clSetKernelArg((*Kernel), 3, sizeof(cl_mem), (void*)&d_mark);
	ciErr1 |= clSetKernelArg((*Kernel), 4, sizeof(cl_int), (void*)&smallKey);
	ciErr1 |= clSetKernelArg((*Kernel), 5, sizeof(cl_int), (void*)&largeKey);
	ciErr1 |= clSetKernelArg((*Kernel), 6, sizeof(cl_mem), (void*)&d_temp);
	cl_mem nsWasNull = NULL;
	int    nsWasOff  = -1;
	ciErr1 |= clSetKernelArg((*Kernel), 7, sizeof(cl_mem), (void*)&nsWasNull);
	ciErr1 |= clSetKernelArg((*Kernel), 8, sizeof(cl_int), (void*)&nsWasOff);
	ciErr1 |= clSetKernelArg((*Kernel), 9, sizeof(cl_mem), (void*)&nsWasNull);
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clSetKernelArg (ns-map), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	static int nsPeMode = -1, nsPeWas = 0;
	if (nsPeMode < 0) {
		nsPeMode = getenv("PE_MODE") ? atoi(getenv("PE_MODE")) : 0;
		nsPeWas  = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
	}
	if (nsPeMode) {
		filter_enqueue_PE(Kernel, 20, 7, rLen, nsPeWas, globalWorkingSetSize, numThreadsPerBlock_x,
			eventList, index, Flag_CPU_GPU, burden, _CPU_GPU);
	} else {
		kernel_enqueue(rLen,20,
			1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	}
}

void filterImpl_write_ns( cl_mem d_Rout16, cl_mem d_Rin16, cl_mem d_mark,
						cl_mem d_markOutput, int beginPos, int rLen, cl_mem d_RidOut,
						int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x=numThreadPB;
	size_t globalWorkingSetSize=numThreadPB*numBlock;
	cl_getKernel("filterImpl_write_kernel_ns",Kernel);
	cl_int ciErr1 = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)&d_Rout16);
	ciErr1 |= clSetKernelArg((*Kernel), 1, sizeof(cl_mem), (void*)&d_Rin16);
	ciErr1 |= clSetKernelArg((*Kernel), 2, sizeof(cl_mem), (void*)&d_mark);
	ciErr1 |= clSetKernelArg((*Kernel), 3, sizeof(cl_mem), (void*)&d_markOutput);
	ciErr1 |= clSetKernelArg((*Kernel), 4, sizeof(cl_int), (void*)&beginPos);
	ciErr1 |= clSetKernelArg((*Kernel), 5, sizeof(cl_int), (void*)&rLen);
	ciErr1 |= clSetKernelArg((*Kernel), 6, sizeof(cl_mem), (void*)&d_RidOut);
	cl_mem nsWWasNull = NULL;
	int    nsWWasOff  = -1;
	ciErr1 |= clSetKernelArg((*Kernel), 7, sizeof(cl_mem), (void*)&nsWWasNull);
	ciErr1 |= clSetKernelArg((*Kernel), 8, sizeof(cl_int), (void*)&nsWWasOff);
	ciErr1 |= clSetKernelArg((*Kernel), 9, sizeof(cl_mem), (void*)&nsWWasNull);
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clSetKernelArg (ns-write), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	static int nsPeModeW = -1, nsPeWasW = 0;
	if (nsPeModeW < 0) {
		nsPeModeW = getenv("PE_MODE") ? atoi(getenv("PE_MODE")) : 0;
		nsPeWasW  = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
	}
	if (nsPeModeW) {
		filter_enqueue_PE(Kernel, 23, 7, rLen, nsPeWasW, globalWorkingSetSize, numThreadsPerBlock_x,
			eventList, index, Flag_CPU_GPU, burden, _CPU_GPU);
	} else {
		kernel_enqueue(rLen, 23,
			1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	}
}

void filterImpl_ns( cl_mem d_Rin16, int beginPos, int rLen, cl_mem* d_Rout16, int* outSize, cl_mem* d_RidOut,
				int numThread, int numBlock, int smallKey, int largeKey,int *index,cl_event *eventList,cl_kernel *Kernel, int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_mem d_mark;
	CL_MALLOC( &d_mark, sizeof(int)*rLen ) ;
	cl_mem d_markOutput;
	CL_MALLOC( &d_markOutput, sizeof(int)*rLen ) ;
	cl_mem d_temp;
	CL_MALLOC( &d_temp, sizeof(int)*rLen ) ;

	// PHASE_TIMING (cE-CPU): narrow 2-byte non-cPDE path.
	static double phn_map_s = 0, phn_scan_s = 0, phn_write_s = 0;
	static int    phn_calls = 0;
	double t_map = 0, t_scan = 0, t_write = 0, t0 = 0;
	const int phT = ph_enabled();
	if (phT) t0 = ph_now();

	// ECPU_TILE_MAP + MAP_GPU + PE-WAS for the NARROW (compressed) map — mirrors filterImpl's
	// 8-byte path but uses filterImpl_map_kernel_ns and the narrow element sizes (ushort input,
	// int marks/temp). Gated identically: both-unset == byte-identical to the original
	// filterImpl_map_ns monolithic call below. Only the map launch's queue/granularity changes;
	// scan/outSize/write over the full rLen are untouched.
	static int nsEcpuTileMap = -1;
	if (nsEcpuTileMap < 0) nsEcpuTileMap = getenv("ECPU_TILE_MAP") ? 1 : 0;
	static int nsMapGpu = -1;
	if (nsMapGpu < 0) nsMapGpu = getenv("MAP_GPU") ? 1 : 0;
	cl_command_queue nsMapQ = nsMapGpu ? CommandQueue[1] : CommandQueue[0];
	if (!nsEcpuTileMap && !nsMapGpu) {
		// DEFAULT PATH — byte-identical to the original: burden scheduler picks CPU/GPU.
		filterImpl_map_ns( d_Rin16, beginPos, rLen, d_mark, smallKey, largeKey, d_temp, numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
		clWaitForEvents(1,&eventList[(*index-1)%2]);
	} else if (!nsEcpuTileMap && nsMapGpu) {
		// MONOLITHIC-ON-GPU: one direct launch over the full rLen on the GPU queue, noWAS,
		// replicating the monolithic filterImpl_map_ns kernel args.
		cl_int errM = CL_SUCCESS;
		cl_kernel kMapM = clCreateKernel(Program, "filterImpl_map_kernel_ns", &errM);
		if (errM != CL_SUCCESS || !kMapM) {
			printf("Error %d creating ns MAP_GPU monolithic kernel, Line %u in file %s !!!\n\n", errM, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		cl_mem wasNullM = NULL; cl_int wasOffM = -1;
		int beginPosM = 0;
		clSetKernelArg(kMapM, 0, sizeof(cl_mem), &d_Rin16);
		clSetKernelArg(kMapM, 1, sizeof(cl_int), &beginPosM);
		clSetKernelArg(kMapM, 2, sizeof(cl_int), &rLen);
		clSetKernelArg(kMapM, 3, sizeof(cl_mem), &d_mark);
		clSetKernelArg(kMapM, 4, sizeof(cl_int), &smallKey);
		clSetKernelArg(kMapM, 5, sizeof(cl_int), &largeKey);
		clSetKernelArg(kMapM, 6, sizeof(cl_mem), &d_temp);
		clSetKernelArg(kMapM, 7, sizeof(cl_mem), &wasNullM);   // noWAS
		clSetKernelArg(kMapM, 8, sizeof(cl_int), &wasOffM);
		clSetKernelArg(kMapM, 9, sizeof(cl_mem), &wasNullM);
		size_t gMono = (size_t)numThread * (size_t)numBlock;   // == scheduler's globalWorkingSetSize
		size_t lMono = (size_t)numThread;
		errM = clEnqueueNDRangeKernel(nsMapQ, kMapM, 1, NULL, &gMono, &lMono, 0, NULL, NULL);
		if (errM != CL_SUCCESS) {
			printf("Error %d enqueue ns MAP_GPU monolithic map, Line %u in file %s !!!\n\n", errM, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		clFinish(nsMapQ);
		clReleaseKernel(kMapM);
	} else {
		// TILED-ON-mapQ: 512x32K sub-buffer block loop of the NARROW map kernel. nsMapGpu picks
		// CPU (CommandQueue[0]) or GPU (CommandQueue[1]). WAS engaged on CPU-tiled when PE_MODE+WASSIZE>0.
		cl_int errT = CL_SUCCESS;
		cl_kernel kMapT = clCreateKernel(Program, "filterImpl_map_kernel_ns", &errT);
		if (errT != CL_SUCCESS || !kMapT) {
			printf("Error %d creating ns ECPU_TILE_MAP kernel, Line %u in file %s !!!\n\n", errT, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		cl_mem wasNullT = NULL; cl_int wasOffT = -1;
		int B = numBlock > 0 ? numBlock : 8;     // 512
		if (B > rLen) B = 1;
		int blkLen = (rLen + B - 1) / B;          // 32768 at rLen=16M
		size_t lE = (size_t)numThread;            // 256
		int beginPosT = 0;
		static int nsTilePeMode = -1, nsTilePeWas = 0;
		if (nsTilePeMode < 0) {
			nsTilePeMode = getenv("PE_MODE") ? atoi(getenv("PE_MODE")) : 0;
			nsTilePeWas  = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
		}
		struct TileWAS tw;
		int wasOn = (nsTilePeMode && !nsMapGpu && nsTilePeWas > 0);
		if (wasOn) tile_was_begin(&tw, nsTilePeWas, nsMapQ); else tw.active = 0;
		for (int b = 0; b < B; b++) {
			int off = b * blkLen;
			int len = blkLen;
			if (off + len > rLen) len = rLen - off;
			if (len <= 0) break;
			// Sub-buffers of the EXISTING narrow input (ushort) and int mark/temp buffers.
			cl_buffer_region rIn   = { (size_t)off * sizeof(unsigned short), (size_t)len * sizeof(unsigned short) };
			cl_buffer_region rMark = { (size_t)off * sizeof(int),            (size_t)len * sizeof(int)            };
			cl_mem subIn   = clCreateSubBuffer(d_Rin16, CL_MEM_READ_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &rIn,   &errT);
			cl_mem subMark = clCreateSubBuffer(d_mark,  CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rMark, &errT);
			cl_mem subTemp = clCreateSubBuffer(d_temp,  CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rMark, &errT);
			if (errT != CL_SUCCESS) {
				printf("Error %d creating ns ECPU_TILE_MAP sub-buffers (b=%d), Line %u in file %s !!!\n\n", errT, b, __LINE__, __FILE__);
				cl_clean(EXIT_FAILURE);
			}
			clSetKernelArg(kMapT, 0, sizeof(cl_mem), &subIn);
			clSetKernelArg(kMapT, 1, sizeof(cl_int), &beginPosT);
			clSetKernelArg(kMapT, 2, sizeof(cl_int), &len);
			clSetKernelArg(kMapT, 3, sizeof(cl_mem), &subMark);
			clSetKernelArg(kMapT, 4, sizeof(cl_int), &smallKey);
			clSetKernelArg(kMapT, 5, sizeof(cl_int), &largeKey);
			clSetKernelArg(kMapT, 6, sizeof(cl_mem), &subTemp);
			if (tw.active) {
				clSetKernelArg(kMapT, 7, sizeof(cl_mem), &tw.was_buffer);
				clSetKernelArg(kMapT, 8, sizeof(cl_int), &tw.wassize);
				clSetKernelArg(kMapT, 9, sizeof(cl_mem), &tw.dummy_buffer);
			} else {
				clSetKernelArg(kMapT, 7, sizeof(cl_mem), &wasNullT);  // noWAS
				clSetKernelArg(kMapT, 8, sizeof(cl_int), &wasOffT);
				clSetKernelArg(kMapT, 9, sizeof(cl_mem), &wasNullT);
			}
			size_t gMap = (size_t)numThread * 8;   // 2048, kernel strides over the block
			errT = clEnqueueNDRangeKernel(nsMapQ, kMapT, 1, NULL, &gMap, &lE, 0, NULL, NULL);
			if (errT != CL_SUCCESS) {
				printf("Error %d enqueue ns ECPU_TILE_MAP map (b=%d), Line %u in file %s !!!\n\n", errT, b, __LINE__, __FILE__);
				cl_clean(EXIT_FAILURE);
			}
			clFlush(nsMapQ);
			clReleaseMemObject(subIn);
			clReleaseMemObject(subMark);
			clReleaseMemObject(subTemp);
		}
		clFinish(nsMapQ);
		if (tw.active) tile_was_end(&tw, nsMapQ);
		clReleaseKernel(kMapT);
	}
	if (phT) { clFinish(nsMapQ); t_map = ph_now() - t0; t0 = ph_now(); }
	//prefix sum (unchanged, on int marks)
	ScanPara *SP;
	SP=(ScanPara*)malloc(sizeof(ScanPara));
	initScan(rLen,SP);
	scanImpl( d_mark, rLen, d_markOutput,index,eventList,Kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	closeScan(SP);

	//get the outSize
	cl_mem d_outSize;
	CL_MALLOC(&d_outSize, sizeof(int)) ;
	filterImpl_outSize_int(d_outSize, d_mark, d_markOutput, rLen,1,1,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	cl_readbuffer(outSize,d_outSize, sizeof(int),index,eventList,Flag_CPU_GPU,burden,_CPU_GPU);
	if (phT) { clFinish(CommandQueue[0]); t_scan = ph_now() - t0; t0 = ph_now(); }

	int outCount = (*outSize) > 0 ? (*outSize) : 1;
	CL_MALLOC( d_Rout16, sizeof(unsigned short)*outCount );
	CL_MALLOC( d_RidOut, sizeof(int)*outCount );
	filterImpl_write_ns( *d_Rout16, d_Rin16, d_mark, d_markOutput, beginPos, rLen, *d_RidOut,numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU );
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	if (phT) {
		clFinish(CommandQueue[0]); t_write = ph_now() - t0;
		phn_map_s += t_map; phn_scan_s += t_scan; phn_write_s += t_write;
		phn_calls++;
		fprintf(stderr, "[PHASE-CECPU] q=%d map=%.4f scan=%.4f write=%.4f total=%.4f\n",
			phn_calls, t_map, t_scan, t_write, t_map + t_scan + t_write);
		if (phn_calls == 15) {
			fprintf(stderr, "[PHASE-CECPU-TOTAL] map=%.4f scan=%.4f write=%.4f total=%.4f\n",
				phn_map_s, phn_scan_s, phn_write_s, phn_map_s + phn_scan_s + phn_write_s);
		}
	}
	CL_FREE(d_mark);
	CL_FREE(d_markOutput);
	CL_FREE( d_temp );
	CL_FREE( d_outSize );
}

extern "C" int CL_RangeSelectionOnly_cPDE(cl_mem d_Rin16, int rLen,
		int rangeSmallKey, int rangeLargeKey, cl_mem* d_Rout, int* d_RidOut_unused,
		int numThreadPB, int numBlock, int _CPU_GPU)
{
	(void)d_RidOut_unused; (void)_CPU_GPU;
	cl_int err = CL_SUCCESS;

	
	cl_command_queue Qd = (g_decomEnabled && DecomCommandQueue) ? DecomCommandQueue : CommandQueue[0];
	cl_command_queue Qe = (g_decomEnabled && NonDecomCommandQueue) ? NonDecomCommandQueue : CommandQueue[0];
	cl_command_queue Qg = CommandQueue[1];   // GPU queue for GPU-routed E blocks

	int B = numBlock > 0 ? numBlock : 8;
	if (B > rLen) B = 1;
	int blkLen = (rLen + B - 1) / B;

	int execGpuPct = getenv("EXEC_GPU_PCT") ? atoi(getenv("EXEC_GPU_PCT")) : 0;
	if (execGpuPct < 0) execGpuPct = 0;
	if (execGpuPct > 100) execGpuPct = 100;
	int gpuBlocks = (int)((double)B * execGpuPct / 100.0 + 0.5);  // round
	if (gpuBlocks > B) gpuBlocks = B;
	int cpuBlocks = B - gpuBlocks;

	static int announced = 0;
	if (!announced) {
		fprintf(stderr, "[cPDE] decom CUs=%d, exec CUs=%d (fission=%s)\n",
				g_decomCUs, g_nonDecomCUs, (g_decomEnabled ? "on" : "FALLBACK-shared"));
		announced = 1;
	}

	cl_mem d_Rdec = NULL;
	CL_MALLOC(&d_Rdec, sizeof(Record) * rLen);
	cl_mem d_mark = NULL;
	CL_MALLOC(&d_mark, sizeof(int) * rLen);
	cl_mem d_temp = NULL;
	CL_MALLOC(&d_temp, sizeof(int) * rLen);
	if (!d_Rdec || !d_mark || !d_temp) {
		printf("Error: cPDE buffer alloc failed, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}

	cl_kernel kDec = clCreateKernel(Program, "decompress_kernel", &err);
	cl_kernel kMap = clCreateKernel(Program, "filterImpl_map_kernel", &err);
	cl_kernel kMapGpu = clCreateKernel(Program, "filterImpl_map_kernel", &err);
	if (err != CL_SUCCESS || !kDec || !kMap || !kMapGpu) {
		printf("Error %d creating cPDE kernels, Line %u in file %s !!!\n\n", err, __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}

	// noWAS args for the 8-byte map kernel (args 7,8,9): wassize=-1 -> noWAS branch.
	cl_mem wasNull = NULL; cl_int wasOff = -1;

	int peWassize = -1;
	{
		bool peOn = (getenv("PE_MODE") && atoi(getenv("PE_MODE")) != 0);
		int w = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
		if (peOn && w > 0 && g_prefetchEnabled && PrefetchCommandQueue && cpuBlocks > 0) peWassize = w;
	}
	bool useWAS = (peWassize > 0);
	cl_mem was_buffer = NULL, dummy_buffer = NULL, last_tag_buffer = NULL;
	cl_kernel hk = NULL;
	if (useWAS) {
		cl_command_queue Qw = Qe; // init/map on the E queue (exec sub-device) — P serves E
		size_t wasEntrySize = 4 * sizeof(cl_ulong);
		was_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)peWassize * wasEntrySize, NULL, &err);
		dummy_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 64, NULL, &err);
		last_tag_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)peWassize * sizeof(cl_ulong), NULL, &err);
		if (!was_buffer || !dummy_buffer || !last_tag_buffer) {
			printf("Error: cPDE PE WAS buffer alloc failed, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		void *dm = clEnqueueMapBuffer(Qw, dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
		memset(dm, 0, 64);
		clEnqueueUnmapMemObject(Qw, dummy_buffer, dm, 0, NULL, NULL);
		clFinish(Qw);
		cl_ulong *wm = (cl_ulong *)clEnqueueMapBuffer(Qw, was_buffer, CL_TRUE, CL_MAP_WRITE, 0, (size_t)peWassize * wasEntrySize, 0, NULL, NULL, &err);
		void *dm2 = clEnqueueMapBuffer(Qw, dummy_buffer, CL_TRUE, CL_MAP_READ, 0, 8, 0, NULL, NULL, &err);
		cl_ulong dummyVal = (cl_ulong)(uintptr_t)dm2;
		clEnqueueUnmapMemObject(Qw, dummy_buffer, dm2, 0, NULL, NULL);
		for (int j = 0; j < peWassize; j++) { wm[j*4] = dummyVal; wm[j*4+1] = dummyVal; wm[j*4+2] = (cl_ulong)-1; wm[j*4+3] = (cl_ulong)-1; }
		clEnqueueUnmapMemObject(Qw, was_buffer, wm, 0, NULL, NULL);
		cl_ulong *lm = (cl_ulong *)clEnqueueMapBuffer(Qw, last_tag_buffer, CL_TRUE, CL_MAP_WRITE, 0, (size_t)peWassize * sizeof(cl_ulong), 0, NULL, NULL, &err);
		memset(lm, 0, (size_t)peWassize * sizeof(cl_ulong));
		clEnqueueUnmapMemObject(Qw, last_tag_buffer, lm, 0, NULL, NULL);
		clFinish(Qw);
		// launch the 1-CU helper (mode 3) on the prefetch sub-device
		hk = clCreateKernel(Program, "WAS_kernel", &err);
		cl_int wmode = 3;
		clSetKernelArg(hk, 0, sizeof(cl_mem), &was_buffer);
		clSetKernelArg(hk, 1, sizeof(cl_int), &peWassize);
		clSetKernelArg(hk, 2, sizeof(cl_mem), &dummy_buffer);
		clSetKernelArg(hk, 3, sizeof(cl_mem), &last_tag_buffer);
		clSetKernelArg(hk, 4, sizeof(cl_int), &wmode);
		size_t wg = 1, wl = 1;
		err = clEnqueueNDRangeKernel(PrefetchCommandQueue, hk, 1, NULL, &wg, &wl, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error %d launching cPDE WAS helper, Line %u in file %s !!!\n\n", err, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		clFlush(PrefetchCommandQueue);
		fprintf(stderr, "[PE] WAS cPDE-exec(E) engaged on prefetch sub-device (prefetching E's decompressed input, wassize=%d)\n", peWassize);
	}

	// Layout banner: P (prefetch CUs), D (decom CUs), E_cpu (NonDecom CUs that run
	// CPU-E blocks), E_gpu (ON when any block routes to the GPU). gpu_pct shows the
	// requested split. This reflects the actual fission produced by cl_init_decom.
	{
		int pCUs = (g_prefetchEnabled && useWAS) ? 1 : 0;
		fprintf(stderr,
			"[cPDE] P=%d D=%d E_cpu=%d E_gpu=%s (gpu_pct=%d, blocks=%d: %d gpu / %d cpu)\n",
			pCUs, g_decomCUs, (cpuBlocks > 0 ? g_nonDecomCUs : 0),
			(gpuBlocks > 0 ? "ON" : "OFF"), execGpuPct, B, gpuBlocks, cpuBlocks);
	}

	std::vector<cl_event> decEvt(B, NULL);
	std::vector<cl_event> mapEvt(B, NULL);

	// CONTROL TOGGLE (CPDE_MONO_MAP): when set, the per-block map kernel is SKIPPED in
	// the block loop (decompress still runs per-block so d_Rdec is fully populated), and
	// a SINGLE monolithic map over the whole d_Rdec[0..rLen] is launched after the
	// decompress loop drains — replicating E-CPU's filterImpl map launch exactly. Isolates
	// whether E-CPU's slow map is purely the monolithic-over-16M launch granularity.
	static int monoMap = -1;
	if (monoMap < 0) monoMap = getenv("CPDE_MONO_MAP") ? 1 : 0;

	// PHASE_TIMING (cDE-CPU): decmap (block-loop decompress+map) / scan / outsize / write.
	// In CPDE_MONO_MAP mode, decmap is split into dec (decompress-only block loop) and
	// mapmono (the single monolithic map).
	static double phc_decmap_s = 0, phc_scan_s = 0, phc_outsize_s = 0, phc_write_s = 0;
	static double phc_dec_s = 0, phc_mapmono_s = 0;
	static int    phc_calls = 0;
	double t_decmap = 0, t_scan = 0, t_outsize = 0, t_write = 0, tc0 = 0;
	double t_dec = 0, t_mapmono = 0;
	const int phTc = ph_enabled();
	if (phTc) tc0 = ph_now();

	size_t lThreads = (size_t)numThreadPB;
	for (int b = 0; b < B; b++) {
		int off = b * blkLen;
		int len = blkLen;
		if (off + len > rLen) len = rLen - off;
		if (len <= 0) { B = b; break; }

		cl_buffer_region rIn  = { (size_t)off * sizeof(unsigned short), (size_t)len * sizeof(unsigned short) };
		cl_buffer_region rOut = { (size_t)off * sizeof(Record),         (size_t)len * sizeof(Record) };
		cl_mem subIn  = clCreateSubBuffer(d_Rin16, CL_MEM_READ_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &rIn,  &err);
		cl_mem subDec = clCreateSubBuffer(d_Rdec,  CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rOut, &err);
		if (err != CL_SUCCESS) {
			printf("Error %d creating cPDE sub-buffers (b=%d), Line %u in file %s !!!\n\n", err, b, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		clSetKernelArg(kDec, 0, sizeof(cl_mem), &subIn);
		clSetKernelArg(kDec, 1, sizeof(cl_mem), &subDec);
		clSetKernelArg(kDec, 2, sizeof(cl_int), &len);
		size_t gDec = (size_t)numThreadPB * 8; // enough work-items; kernel strides
		if (gDec > (size_t)len) gDec = ((len + lThreads - 1) / lThreads) * lThreads;
		if (gDec == 0) gDec = lThreads;
		err = clEnqueueNDRangeKernel(Qd, kDec, 1, NULL, &gDec, &lThreads, 0, NULL, &decEvt[b]);
		if (err != CL_SUCCESS) {
			printf("Error %d enqueue decompress (b=%d), Line %u in file %s !!!\n\n", err, b, __LINE__, __FILE__);
			cl_clean(EXIT_FAILURE);
		}
		clFlush(Qd);

		if (!monoMap) {
			// per-block map (default cDE-CPU path): map this decompressed block immediately.
			bool gpuE = (b < gpuBlocks);
			cl_buffer_region rMark = { (size_t)off * sizeof(int), (size_t)len * sizeof(int) };
			cl_mem subMark = clCreateSubBuffer(d_mark, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rMark, &err);
			cl_mem subTemp = clCreateSubBuffer(d_temp, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rMark, &err);
			int beginPos = 0;
			cl_kernel kE = gpuE ? kMapGpu : kMap;          // separate kernel obj per device
			cl_command_queue Qexec = gpuE ? Qg : Qe;
			clSetKernelArg(kE, 0, sizeof(cl_mem), &subDec);   // decompressed Record block
			clSetKernelArg(kE, 1, sizeof(cl_int), &beginPos);
			clSetKernelArg(kE, 2, sizeof(cl_int), &len);
			clSetKernelArg(kE, 3, sizeof(cl_mem), &subMark);
			clSetKernelArg(kE, 4, sizeof(cl_int), &rangeSmallKey);
			clSetKernelArg(kE, 5, sizeof(cl_int), &rangeLargeKey);
			clSetKernelArg(kE, 6, sizeof(cl_mem), &subTemp);
			if (!gpuE && useWAS) {
				clSetKernelArg(kE, 7, sizeof(cl_mem), &was_buffer);
				clSetKernelArg(kE, 8, sizeof(cl_int), &peWassize);
				clSetKernelArg(kE, 9, sizeof(cl_mem), &dummy_buffer);
			} else {
				clSetKernelArg(kE, 7, sizeof(cl_mem), &wasNull);  // noWAS
				clSetKernelArg(kE, 8, sizeof(cl_int), &wasOff);
				clSetKernelArg(kE, 9, sizeof(cl_mem), &wasNull);
			}
			size_t gMap = (size_t)numThreadPB * 8;
			size_t lE = lThreads;
			if (gpuE && lE > 256) lE = 256;
			if (gpuE) {
				// keep global a multiple of local
				gMap = (gMap / lE) * lE;
				if (gMap == 0) gMap = lE;
			}
			err = clEnqueueNDRangeKernel(Qexec, kE, 1, NULL, &gMap, &lE, 1, &decEvt[b], &mapEvt[b]);
			if (err != CL_SUCCESS) {
				printf("Error %d enqueue map (b=%d, %s), Line %u in file %s !!!\n\n",
					err, b, (gpuE ? "GPU" : "CPU"), __LINE__, __FILE__);
				cl_clean(EXIT_FAILURE);
			}
			clFlush(Qexec);
			clReleaseMemObject(subMark);
			clReleaseMemObject(subTemp);
		}

		clReleaseMemObject(subIn);
		clReleaseMemObject(subDec);
	}

	clFinish(Qd);
	if (!monoMap) {
		if (cpuBlocks > 0 && Qe) clFinish(Qe);
		if (gpuBlocks > 0) clFinish(Qg);
		// PHASE_TIMING: end of block-loop (decompress on Qd + map on Qe/Qg, both drained above).
		if (phTc) { t_decmap = ph_now() - tc0; tc0 = ph_now(); }
	} else {
		// CONTROL (CPDE_MONO_MAP): the loop above only ran decompress — d_Rdec is now fully
		// populated. clFinish(Qd) ended the dec phase. Now do monolithic map(s) PER DEVICE
		// PORTION, replicating E-CPU's filterImpl map launch granularity (same kernel
		// filterImpl_map_kernel, local=numThreadPB, global=numThreadPB*<portion-blocks>, grid-
		// stride over the portion). This removes the per-32K-block TILING but KEEPS the genuine
		// CPU/GPU device split that EXEC_GPU_PCT selects. Three cases (gpuBlocks is the # of the
		// first blocks routed to GPU; the rest go to CPU):
		//   gpuBlocks == 0  (CPU-only: cDE-CPU/cPDE-CPU): ONE monolithic map over d_Rdec[0..rLen)
		//                    on CommandQueue[0] (Qe sub-device when fission on). WAS on CPU when
		//                    useWAS (cPDE-CPU). == byte-for-byte the original mono behavior.
		//   gpuBlocks == B  (all-GPU: cDE-b): ONE monolithic map over d_Rdec[0..rLen) on Qg (GPU).
		//   0<gpuBlocks<B   (split: cPDE-c): TWO monolithic maps enqueued concurrently then both
		//                    clFinish'd — GPU map over d_Rdec[0 .. gpuBlocks*blkLen) on Qg, CPU map
		//                    over d_Rdec[gpuBlocks*blkLen .. rLen) on Qe. Sub-buffers carve each
		//                    portion; offsets are gpuBlocks*blkLen*sizeof(...) (a multiple of
		//                    blkLen*elt — far above any device base-addr alignment). WAS on the
		//                    CPU portion only (mirrors the per-block path's !gpuE && useWAS).
		// Writes d_mark/d_temp identically to the per-block path; scan/outSize/write unchanged.
		if (phTc) { t_dec = ph_now() - tc0; tc0 = ph_now(); }
		int beginPosM = 0;
		int gpuPortionTuples = gpuBlocks * blkLen;
		if (gpuPortionTuples > rLen) gpuPortionTuples = rLen;
		int cpuPortionTuples = rLen - gpuPortionTuples;   // = cpuBlocks*blkLen (last partial absorbed)

		// --- GPU monolithic map over d_Rdec[0 .. gpuPortionTuples) on Qg ---
		if (gpuBlocks > 0) {
			cl_mem gIn = d_Rdec, gMark = d_mark, gTmp = d_temp;
			cl_mem gSubIn = NULL, gSubMark = NULL, gSubTmp = NULL;
			int gLen = (gpuBlocks < B) ? gpuPortionTuples : rLen;   // full buffer when all-GPU (no sub-buffer)
			if (gpuBlocks < B) {
				cl_buffer_region rgIn   = { 0, (size_t)gLen * sizeof(Record) };
				cl_buffer_region rgMark = { 0, (size_t)gLen * sizeof(int)    };
				gSubIn   = clCreateSubBuffer(d_Rdec, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgIn,   &err);
				gSubMark = clCreateSubBuffer(d_mark, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgMark, &err);
				gSubTmp  = clCreateSubBuffer(d_temp, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgMark, &err);
				if (err != CL_SUCCESS) {
					printf("Error %d creating mono-map GPU sub-buffers, Line %u in file %s !!!\n\n", err, __LINE__, __FILE__);
					cl_clean(EXIT_FAILURE);
				}
				gIn = gSubIn; gMark = gSubMark; gTmp = gSubTmp;
			}
			clSetKernelArg(kMapGpu, 0, sizeof(cl_mem), &gIn);
			clSetKernelArg(kMapGpu, 1, sizeof(cl_int), &beginPosM);
			clSetKernelArg(kMapGpu, 2, sizeof(cl_int), &gLen);
			clSetKernelArg(kMapGpu, 3, sizeof(cl_mem), &gMark);
			clSetKernelArg(kMapGpu, 4, sizeof(cl_int), &rangeSmallKey);
			clSetKernelArg(kMapGpu, 5, sizeof(cl_int), &rangeLargeKey);
			clSetKernelArg(kMapGpu, 6, sizeof(cl_mem), &gTmp);
			clSetKernelArg(kMapGpu, 7, sizeof(cl_mem), &wasNull);   // GPU portion always noWAS
			clSetKernelArg(kMapGpu, 8, sizeof(cl_int), &wasOff);
			clSetKernelArg(kMapGpu, 9, sizeof(cl_mem), &wasNull);
			size_t gG = (size_t)numThreadPB * (size_t)gpuBlocks;   // scale grid to the GPU portion
			size_t lG = (size_t)numThreadPB;
			if (lG > 256) lG = 256;
			gG = (gG / lG) * lG;
			if (gG == 0) gG = lG;
			err = clEnqueueNDRangeKernel(Qg, kMapGpu, 1, NULL, &gG, &lG, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				printf("Error %d enqueue mono-map GPU portion, Line %u in file %s !!!\n\n", err, __LINE__, __FILE__);
				cl_clean(EXIT_FAILURE);
			}
			clFlush(Qg);
			if (gSubIn)   clReleaseMemObject(gSubIn);
			if (gSubMark) clReleaseMemObject(gSubMark);
			if (gSubTmp)  clReleaseMemObject(gSubTmp);
		}

		// --- CPU monolithic map over d_Rdec[gpuPortionTuples .. rLen) on Qe ---
		if (cpuBlocks > 0) {
			cl_mem cIn = d_Rdec, cMark = d_mark, cTmp = d_temp;
			cl_mem cSubIn = NULL, cSubMark = NULL, cSubTmp = NULL;
			int cLen = cpuPortionTuples;
			if (gpuBlocks > 0) {
				// split: carve the trailing CPU portion [gpuPortionTuples .. rLen)
				cl_buffer_region rcIn   = { (size_t)gpuPortionTuples * sizeof(Record), (size_t)cLen * sizeof(Record) };
				cl_buffer_region rcMark = { (size_t)gpuPortionTuples * sizeof(int),    (size_t)cLen * sizeof(int)    };
				cSubIn   = clCreateSubBuffer(d_Rdec, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rcIn,   &err);
				cSubMark = clCreateSubBuffer(d_mark, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rcMark, &err);
				cSubTmp  = clCreateSubBuffer(d_temp, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rcMark, &err);
				if (err != CL_SUCCESS) {
					printf("Error %d creating mono-map CPU sub-buffers, Line %u in file %s !!!\n\n", err, __LINE__, __FILE__);
					cl_clean(EXIT_FAILURE);
				}
				cIn = cSubIn; cMark = cSubMark; cTmp = cSubTmp;
			}
			clSetKernelArg(kMap, 0, sizeof(cl_mem), &cIn);
			clSetKernelArg(kMap, 1, sizeof(cl_int), &beginPosM);
			clSetKernelArg(kMap, 2, sizeof(cl_int), &cLen);
			clSetKernelArg(kMap, 3, sizeof(cl_mem), &cMark);
			clSetKernelArg(kMap, 4, sizeof(cl_int), &rangeSmallKey);
			clSetKernelArg(kMap, 5, sizeof(cl_int), &rangeLargeKey);
			clSetKernelArg(kMap, 6, sizeof(cl_mem), &cTmp);
			if (useWAS) {
				clSetKernelArg(kMap, 7, sizeof(cl_mem), &was_buffer);
				clSetKernelArg(kMap, 8, sizeof(cl_int), &peWassize);
				clSetKernelArg(kMap, 9, sizeof(cl_mem), &dummy_buffer);
			} else {
				clSetKernelArg(kMap, 7, sizeof(cl_mem), &wasNull);   // noWAS (matches E-CPU filterImpl)
				clSetKernelArg(kMap, 8, sizeof(cl_int), &wasOff);
				clSetKernelArg(kMap, 9, sizeof(cl_mem), &wasNull);
			}
			size_t gC = (size_t)numThreadPB * (size_t)cpuBlocks;   // scale grid to the CPU portion
			size_t lC = (size_t)numThreadPB;
			cl_command_queue Qcpu = (gpuBlocks == 0) ? CommandQueue[0] : Qe;  // CPU-only mono uses CommandQueue[0]
			err = clEnqueueNDRangeKernel(Qcpu, kMap, 1, NULL, &gC, &lC, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				printf("Error %d enqueue mono-map CPU portion, Line %u in file %s !!!\n\n", err, __LINE__, __FILE__);
				cl_clean(EXIT_FAILURE);
			}
			clFlush(Qcpu);
			if (cSubIn)   clReleaseMemObject(cSubIn);
			if (cSubMark) clReleaseMemObject(cSubMark);
			if (cSubTmp)  clReleaseMemObject(cSubTmp);
		}

		// drain both portions
		if (gpuBlocks > 0) clFinish(Qg);
		if (cpuBlocks > 0) clFinish((gpuBlocks == 0) ? CommandQueue[0] : Qe);
		if (phTc) { t_mapmono = ph_now() - tc0; t_decmap = t_dec + t_mapmono; tc0 = ph_now(); }
	}
	for (int b = 0; b < B; b++) {
		if (decEvt[b]) clReleaseEvent(decEvt[b]);
		if (mapEvt[b]) clReleaseEvent(mapEvt[b]);
	}

	if (useWAS) {
		cl_uint *cs = (cl_uint *)clEnqueueMapBuffer(Qe, dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
		cs[1] = 1;                          // signal helper to stop
		clEnqueueUnmapMemObject(Qe, dummy_buffer, cs, 0, NULL, NULL);
		clFinish(Qe);
		clFinish(PrefetchCommandQueue);
		clReleaseKernel(hk);
		clReleaseMemObject(was_buffer);
		clReleaseMemObject(dummy_buffer);
		clReleaseMemObject(last_tag_buffer);
	}

	cl_event eventList[2];
	int index = 0;
	cl_kernel Kernel;
	int CPU_GPU; double burden;
	cl_mem d_markOutput = NULL;
	CL_MALLOC(&d_markOutput, sizeof(int) * rLen);

	if (phTc) tc0 = ph_now();   // exclude any (PE-only) helper teardown from scan timing
	ScanPara *SP = (ScanPara*)malloc(sizeof(ScanPara));
	initScan(rLen, SP);
	scanImpl(d_mark, rLen, d_markOutput, &index, eventList, &Kernel, &CPU_GPU, &burden, SP, _CPU_GPU);
	clWaitForEvents(1, &eventList[(index-1)%2]);
	closeScan(SP);
	if (phTc) { clFinish(CommandQueue[0]); t_scan = ph_now() - tc0; tc0 = ph_now(); }

	cl_mem d_outSize = NULL;
	CL_MALLOC(&d_outSize, sizeof(int));
	int outSize = 0;
	filterImpl_outSize_int(d_outSize, d_mark, d_markOutput, rLen, 1, 1, &index, eventList, &Kernel, &CPU_GPU, &burden, _CPU_GPU);
	clWaitForEvents(1, &eventList[(index-1)%2]);
	cl_readbuffer(&outSize, d_outSize, sizeof(int), &index, eventList, &CPU_GPU, &burden, _CPU_GPU);
	clWaitForEvents(1, &eventList[(index-1)%2]);
	deschedule(CPU_GPU, burden);
	if (phTc) { clFinish(CommandQueue[0]); t_outsize = ph_now() - tc0; tc0 = ph_now(); }

	// WRITE phase: materialize the matched tuples into *d_Rout, mirroring the
	// non-cPDE filterImpl/filterImpl_write_int path so cPDE selection does the
	// SAME output work (second full input scan) and is fairly comparable. cPDE
	// operates on decompressed 8-byte Records (d_Rdec), so the 8-byte
	// filterImpl_write_kernel (kid 23) scatters matched Records using
	// d_mark + d_markOutput into *d_Rout (sized sizeof(Record)*outSize).
	if (d_Rout) {
		int beginPosW = 0;
		CL_MALLOC(d_Rout, sizeof(Record) * (outSize > 0 ? outSize : 1));
		filterImpl_write_int(*d_Rout, d_Rdec, d_mark, d_markOutput, beginPosW, rLen,
			numThreadPB, numBlock, &index, eventList, &Kernel, &CPU_GPU, &burden, _CPU_GPU);
		clWaitForEvents(1, &eventList[(index-1)%2]);
		deschedule(CPU_GPU, burden);
	}
	if (phTc) {
		clFinish(CommandQueue[0]); t_write = ph_now() - tc0;
		phc_decmap_s += t_decmap; phc_scan_s += t_scan; phc_outsize_s += t_outsize; phc_write_s += t_write;
		phc_dec_s += t_dec; phc_mapmono_s += t_mapmono;
		phc_calls++;
		if (monoMap) {
			fprintf(stderr, "[PHASE-CPDE] q=%d dec=%.4f mapmono=%.4f (decmap=%.4f) scan=%.4f outsize=%.4f write=%.4f total=%.4f\n",
				phc_calls, t_dec, t_mapmono, t_decmap, t_scan, t_outsize, t_write,
				t_decmap + t_scan + t_outsize + t_write);
		} else {
			fprintf(stderr, "[PHASE-CPDE] q=%d decmap=%.4f scan=%.4f outsize=%.4f write=%.4f total=%.4f\n",
				phc_calls, t_decmap, t_scan, t_outsize, t_write,
				t_decmap + t_scan + t_outsize + t_write);
		}
		if (phc_calls == 15) {
			if (monoMap) {
				fprintf(stderr, "[PHASE-CPDE-TOTAL] dec=%.4f mapmono=%.4f (decmap=%.4f) scan=%.4f outsize=%.4f write=%.4f total=%.4f\n",
					phc_dec_s, phc_mapmono_s, phc_decmap_s, phc_scan_s, phc_outsize_s, phc_write_s,
					phc_decmap_s + phc_scan_s + phc_outsize_s + phc_write_s);
			} else {
				fprintf(stderr, "[PHASE-CPDE-TOTAL] decmap=%.4f scan=%.4f outsize=%.4f write=%.4f total=%.4f\n",
					phc_decmap_s, phc_scan_s, phc_outsize_s, phc_write_s,
					phc_decmap_s + phc_scan_s + phc_outsize_s + phc_write_s);
			}
		}
	}

	clReleaseKernel(Kernel);
	clReleaseEvent(eventList[0]);
	clReleaseEvent(eventList[1]);
	clReleaseKernel(kDec);
	clReleaseKernel(kMap);
	clReleaseKernel(kMapGpu);
	CL_FREE(d_Rdec);
	CL_FREE(d_mark);
	CL_FREE(d_temp);
	CL_FREE(d_markOutput);
	CL_FREE(d_outSize);

	fprintf(stderr, "[cPDE] decom CUs=%d, exec CUs=%d, blocks=%d: %d matched (keys [%d,%d])\n",
			g_decomCUs, g_nonDecomCUs, B, outSize, rangeSmallKey, rangeLargeKey);
	return outSize;
}

void testFilterImpl( int rLen, int numThreadPB, int numBlock)//->corresponding to selection
{
	 int _CPU_GPU=0;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;

	int beginPos = 0;
	int memSize = sizeof(Record)*rLen;
	
	void *Rin;
	HOST_MALLOC(Rin, memSize);
	generateRand( (Record *)Rin, 100, rLen, 0 );
	Record* Rout;

	int smallKey = rand()%100;
	int largeKey = smallKey;

	int* outSize = (int*)malloc( sizeof(int) );
	CL_RangeSelection((Record*) Rin,  rLen, smallKey, largeKey, &Rout, 
		numThreadPB,numBlock , _CPU_GPU);
	printf("CL_RangeSelectionFinish\n");
	CL_PointSelection((Record*) Rin,  rLen, smallKey, &Rout, 
		numThreadPB,numBlock, _CPU_GPU);
	printf("CL_PointSelectionFinish\n");

}