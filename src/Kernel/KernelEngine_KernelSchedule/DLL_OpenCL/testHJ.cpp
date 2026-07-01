#include "common.h"
#include "testScan.h"
#include "Helper.h"
#include "PrimitiveCommon.h"
#include "KernelScheduler.h"
#include "scheduler.h"
#include "OpenCL_DLL.h"
// OpenCL Vars---------0 for CPU, 1 for GPU
extern cl_context Context;        // OpenCL context
extern cl_program Program;           // OpenCL program
extern cl_command_queue CommandQueue[2];// OpenCL command que
extern cl_platform_id Platform[2];      // OpenCL platform
extern cl_device_id Device[2];          // OpenCL device
extern cl_ulong totalLocalMemory[2];      /**< Max local memory allowed */
extern cl_device_id allDevices[10];

void HJbuild_int(cl_mem d_R, cl_mem rHashTable, int rLen, int sLen, int rHashTableBucketNum,
	size_t globalSize, size_t groupSize,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x = groupSize;
	size_t globalWorkingSetSize = globalSize;
	cl_getKernel("build_kernel",Kernel);

	//configure build kernel
	cl_int ciErr1 =  clSetKernelArg((*Kernel),0,sizeof(cl_mem),&d_R);
	ciErr1 |= clSetKernelArg((*Kernel),1,sizeof(cl_mem),&rHashTable);
	ciErr1 |= clSetKernelArg((*Kernel),2,sizeof(cl_uint),(void*)&rLen);
	ciErr1 |= clSetKernelArg((*Kernel),3,sizeof(cl_uint),(void*)&sLen);
	ciErr1 |= clSetKernelArg((*Kernel),4,sizeof(cl_uint),(void*)&rHashTableBucketNum);

	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	kernel_enqueue(rLen, 49
		,1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
}

// PE (CPU-assisted prefetch) probe enqueue. Used in place of kernel_enqueue for the
// probe kernel (kid 50) when PE_MODE is on with WASSIZE>0. It makes the per-kernel
// burden-scheduling decision itself (one Kernelscheduler call), and ONLY when the
// scheduler routes the probe to the CPU device (CPU_GPU==0) does it engage WAS: a
// 1-CU WAS_kernel helper (mode 3 = backward+spin+nopause) on PrefetchCommandQueue
// prefetches the addresses the 7-CU probe posts. When routed to the GPU it runs the
// noWAS probe (wassize=-1). Result is identical either way (PARITY-OK verified).
// Mirrors the verified WAS lifecycle in CL_hj_PE (BinaryJoin.cpp).
static void HJprobe_enqueue_PE(cl_kernel *Kernel, int probeWorkSize, int wassize,
	size_t globalSize, size_t groupSize, cl_event *eventList, int *index,
	int *Flag_CPU_GPU, double *burden, int _CPU_GPU)
{
	int preFlag = *Flag_CPU_GPU;
	double preBurden = *burden;
	int CPU_GPU = Kernelscheduler(probeWorkSize, 50, Flag_CPU_GPU, burden, _CPU_GPU);
	// PE_FORCE_CPU pins the probe to the CPU sub-device so the PE-CPU (WAS) path is
	// exercised deterministically (the burden scheduler's baseline times favor the
	// GPU probe, so it would otherwise rarely land on the CPU).
	static int forceCpu = -1;
	if (forceCpu < 0) forceCpu = getenv("PE_FORCE_CPU") ? atoi(getenv("PE_FORCE_CPU")) : 0;
	// Note: under PE_FORCE_CPU the burden Kernelscheduler already added stays on the
	// scheduler-chosen device (possibly GPU); it self-corrects on the next deschedule.
	// Intentional for this measurement-only hook — do not "fix" by re-accounting here.
	if (forceCpu) CPU_GPU = 0;
	*Flag_CPU_GPU = CPU_GPU;

	cl_int err = CL_SUCCESS;
	bool useWAS = (CPU_GPU == 0 && wassize > 0 && g_prefetchEnabled && PrefetchCommandQueue);
	if (useWAS) {
		static int announced = 0;
		if (!announced) { fprintf(stderr, "[PE] WAS probe engaged on CPU sub-device (wassize=%d)\n", wassize); announced = 1; }
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
			// Fail loudly rather than NULL-deref the maps below (matches the engine's
			// cl_clean-on-OpenCL-error convention used for the enqueue path).
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
		// probe runs WITH WAS
		clSetKernelArg(*Kernel, 7, sizeof(cl_mem), &was_buffer);
		clSetKernelArg(*Kernel, 8, sizeof(cl_int), &wassize);
		clSetKernelArg(*Kernel, 9, sizeof(cl_mem), &dummy_buffer);
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
		// GPU (or fission unavailable): noWAS probe
		cl_mem nb = NULL; cl_int off = -1;
		clSetKernelArg(*Kernel, 7, sizeof(cl_mem), &nb);
		clSetKernelArg(*Kernel, 8, sizeof(cl_int), &off);
		clSetKernelArg(*Kernel, 9, sizeof(cl_mem), &nb);
	}

	// Geometry (num_wg) is controlled by NUM_WG in HJImpl; the probe runs on the fission
	// device CommandQueue[CPU_GPU] at that geometry (no per-path rounding/queue swap).
	size_t groups = globalSize, threads = groupSize;
	cl_command_queue probeQ = CommandQueue[CPU_GPU];
	if (CPU_GPU && threads > 256) threads = 256;
	cl_int ciErr1;
	if (*index != 0) {
		ciErr1 = clEnqueueNDRangeKernel(probeQ, *Kernel, 1, NULL, &groups, &threads,
			1, &eventList[(*index - 1) % 2], &eventList[*index % 2]);
		deschedule(preFlag, preBurden);
	} else {
		ciErr1 = clEnqueueNDRangeKernel(probeQ, *Kernel, 1, NULL, &groups, &threads,
			0, NULL, &eventList[*index]);
	}
	(*index)++;
	if (ciErr1 != CL_SUCCESS) {
		printf("Error %d in clEnqueueNDRangeKernel (PE probe), Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	clFlush(probeQ);

	if (useWAS) {
		clFinish(CommandQueue[0]);          // probe complete
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

void HJprobe_int(cl_mem rHashTable, cl_mem d_S, cl_mem* d_Rout, int rLen, int sLen,
	int rHashTableBucketNum, int resultsNum, size_t globalSize, size_t groupSize,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x = groupSize;
	size_t globalWorkingSetSize = globalSize;
	cl_getKernel("probe_kernel",Kernel);

	//configure probe kernel
	cl_int ciErr1 =  clSetKernelArg((*Kernel),0,sizeof(cl_mem),&rHashTable);
	ciErr1 |= clSetKernelArg((*Kernel),1,sizeof(cl_mem),&d_S);
	ciErr1 |= clSetKernelArg((*Kernel),2,sizeof(cl_mem),d_Rout);
	ciErr1 |= clSetKernelArg((*Kernel),3,sizeof(cl_uint),(void*)&rLen);
	ciErr1 |= clSetKernelArg((*Kernel),4,sizeof(cl_uint),(void*)&sLen);
	ciErr1 |= clSetKernelArg((*Kernel),5,sizeof(cl_uint),(void*)&rHashTableBucketNum);
	ciErr1 |= clSetKernelArg((*Kernel),6,sizeof(cl_uint),(void*)&resultsNum);
	// probe_kernel gained 3 WAS args (7,8,9). This legacy path is noWAS:
	// wassize=-1 selects the noWAS branch, so was_buffer/dummy_buffer are unused (NULL).
	cl_mem wasNull = NULL;
	int    wasOff  = -1;
	ciErr1 |= clSetKernelArg((*Kernel),7,sizeof(cl_mem),&wasNull);
	ciErr1 |= clSetKernelArg((*Kernel),8,sizeof(cl_int),(void*)&wasOff);
	ciErr1 |= clSetKernelArg((*Kernel),9,sizeof(cl_mem),&wasNull);

	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	// PE config: when PE_MODE is on with WASSIZE>0, route the probe through the
	// PE-aware enqueue so that a CPU-scheduled probe is assisted by the WAS helper.
	// Read once; getenv/atoi are idempotent so the lazy init is race-free.
	static int peMode = -1, peWas = 0;
	if (peMode < 0) {
		peMode = getenv("PE_MODE") ? atoi(getenv("PE_MODE")) : 0;
		peWas  = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
	}
	if (peMode) {
		// PE active: route the probe through the PE-aware enqueue. WAS engages only
		// when the probe lands on the CPU AND peWas>0 (else it runs noWAS, matching
		// the kernel_enqueue baseline — which also validates the bookkeeping copy).
		HJprobe_enqueue_PE(Kernel, rLen/rHashTableBucketNum, peWas,
			globalWorkingSetSize, numThreadsPerBlock_x, eventList, index,
			Flag_CPU_GPU, burden, _CPU_GPU);
	} else {
		kernel_enqueue(rLen/rHashTableBucketNum, 50
			,1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	}
}

int HJImpl(cl_mem d_R, cl_uint rLen, cl_mem d_S, cl_uint sLen, cl_mem rHashTable, cl_mem* d_Rout, int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_uint  resultsNum = rLen;
	cl_uint  rHashTableBucketNum = 2 * 1024 * 1024;
	// Geometry control: NUM_WG overrides the work-group count (default 32 = 8192/256).
	// Used to study the power-of-2-globalSize cache-conflict effect and to match geometry
	// across configs. Applies to BOTH build (kid 49) and probe (kid 50).
	size_t groupSize = 256;
	static int hjNumWG = -1;
	if (hjNumWG < 0) hjNumWG = getenv("NUM_WG") ? atoi(getenv("NUM_WG")) : 32;
	size_t globalSize = (size_t)hjNumWG * groupSize;
	CL_MALLOC(d_Rout,sizeof(Record) * resultsNum * 2);

HJbuild_int(d_R, rHashTable, rLen , sLen, rHashTableBucketNum,
	globalSize,groupSize,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);

HJprobe_int(rHashTable,d_S,d_Rout,rLen,sLen,rHashTableBucketNum,
	resultsNum,globalSize,groupSize,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]); 
	return resultsNum;
}

extern "C" int CL_hjOnly_ns(cl_mem d_R, int rLen, cl_mem d_S, int sLen,
                            cl_mem* h_Rout, int narrow, int _CPU_GPU)
{
    cl_event eventList[2];
    int index = 0;
    int CPU_GPU;
    double burden;
    cl_kernel Kernel;
    cl_int err;

    cl_uint bucketNum = 1;
    while ((int)bucketNum < rLen && bucketNum < (1u << 17)) bucketNum <<= 1;
    if (bucketNum < 1024) bucketNum = 1024;
    cl_uint hashBucketCap = 32; // headroom; PK-FK R has 1 tuple/key on average

    cl_uint resultsNum = (cl_uint)sLen;        // bounded output = |S|
    size_t htElems = (size_t)bucketNum * (1 + hashBucketCap * 2);
    size_t htBytes = htElems * sizeof(cl_uint);

    cl_mem rHashTable;
    CL_MALLOC(&rHashTable, htBytes);
    cl_uint zero = 0;
    clEnqueueFillBuffer(CommandQueue[0], rHashTable, &zero, sizeof(cl_uint), 0, htBytes, 0, NULL, NULL);
    // output: [0]=count, then 4 uints/match. Size to the bound |S|.
    CL_MALLOC(h_Rout, sizeof(cl_uint) * (4 + (size_t)resultsNum * 4));
    clEnqueueFillBuffer(CommandQueue[0], *h_Rout, &zero, sizeof(cl_uint), 0, sizeof(cl_uint), 0, NULL, NULL);
    clFinish(CommandQueue[0]);

    size_t groupSize = 256;
    static int hjNumWG = -1;
    if (hjNumWG < 0) hjNumWG = getenv("NUM_WG") ? atoi(getenv("NUM_WG")) : 32;
    size_t globalSize = (size_t)hjNumWG * groupSize;

    // ---- build ----
    cl_getKernel(narrow ? (char*)"build_kernel_ns" : (char*)"build_kernel_w16", &Kernel);
    cl_uint rTupleNum = (cl_uint)rLen;
    err  = clSetKernelArg(Kernel, 0, sizeof(cl_mem), &d_R);
    err |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), &rHashTable);
    err |= clSetKernelArg(Kernel, 2, sizeof(cl_uint), &rTupleNum);
    err |= clSetKernelArg(Kernel, 3, sizeof(cl_uint), &bucketNum);
    err |= clSetKernelArg(Kernel, 4, sizeof(cl_uint), &hashBucketCap);
    if (err != CL_SUCCESS) { printf("Error NS build clSetKernelArg %d, %s:%u\n", err, __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }
    kernel_enqueue(rLen, 49, 1, &globalSize, &groupSize, eventList, &index, &Kernel, &CPU_GPU, &burden, _CPU_GPU);
    clWaitForEvents(1, &eventList[(index - 1) % 2]);
    deschedule(CPU_GPU, burden);
    clReleaseKernel(Kernel);

    // ---- probe (memory-bound scan over S) ----
    cl_getKernel(narrow ? (char*)"probe_kernel_ns" : (char*)"probe_kernel_w16", &Kernel);
    cl_uint sTupleNum = (cl_uint)sLen;
    err  = clSetKernelArg(Kernel, 0, sizeof(cl_mem), &rHashTable);
    err |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), &d_S);
    err |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), h_Rout);
    err |= clSetKernelArg(Kernel, 3, sizeof(cl_uint), &sTupleNum);
    err |= clSetKernelArg(Kernel, 4, sizeof(cl_uint), &bucketNum);
    err |= clSetKernelArg(Kernel, 5, sizeof(cl_uint), &hashBucketCap);
    err |= clSetKernelArg(Kernel, 6, sizeof(cl_uint), &resultsNum);
    {
        cl_mem wasNull = NULL; cl_int wasOff = -1;
        err |= clSetKernelArg(Kernel, 7, sizeof(cl_mem), &wasNull);
        err |= clSetKernelArg(Kernel, 8, sizeof(cl_int), &wasOff);
        err |= clSetKernelArg(Kernel, 9, sizeof(cl_mem), &wasNull);
    }
    if (err != CL_SUCCESS) { printf("Error NS probe clSetKernelArg %d, %s:%u\n", err, __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }
    index = 0;
    static int nsPeMode = -1, nsPeWas = 0;
    if (nsPeMode < 0) {
        nsPeMode = getenv("PE_MODE") ? atoi(getenv("PE_MODE")) : 0;
        nsPeWas  = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
    }
    if (nsPeMode) {
        HJprobe_enqueue_PE(&Kernel, sLen / (int)bucketNum, nsPeWas,
            globalSize, groupSize, eventList, &index, &CPU_GPU, &burden, _CPU_GPU);
    } else {
        kernel_enqueue(sLen, 50, 1, &globalSize, &groupSize, eventList, &index, &Kernel, &CPU_GPU, &burden, _CPU_GPU);
    }
    clWaitForEvents(1, &eventList[(index - 1) % 2]);
    deschedule(CPU_GPU, burden);

    // read the actual match count for the [NS] marker
    cl_uint matched = 0;
    clEnqueueReadBuffer(CommandQueue[0], *h_Rout, CL_TRUE, 0, sizeof(cl_uint), &matched, 0, NULL, NULL);
    fprintf(stderr, "[NS] %s HJ: |R|=%d |S|=%d matched=%u (bucketNum=%u cap=%u)\n",
            narrow ? "compressed" : "masked(wide)", rLen, sLen, matched, bucketNum, hashBucketCap);

    clReleaseKernel(Kernel);
    CL_FREE(rHashTable);
    return (int)resultsNum; // keep result-len semantics identical to CL_hjOnly (=|S|)
}

extern "C" int CL_hjOnly_cPDE(cl_mem d_R, int rLen, cl_mem d_S, int sLen,
                              cl_mem* h_Rout, int _CPU_GPU)
{
    cl_int err = CL_SUCCESS;

    cl_command_queue Qd = (g_decomEnabled && DecomCommandQueue) ? DecomCommandQueue : CommandQueue[0];
    cl_command_queue Qe = (g_decomEnabled && NonDecomCommandQueue) ? NonDecomCommandQueue : CommandQueue[0];
    cl_command_queue Qg = CommandQueue[1];   // GPU queue for GPU-routed E probe blocks

    // Bucket geometry — identical to CL_hjOnly_ns so the match logic (and count) match.
    cl_uint bucketNum = 1;
    while ((int)bucketNum < rLen && bucketNum < (1u << 17)) bucketNum <<= 1;
    if (bucketNum < 1024) bucketNum = 1024;
    cl_uint hashBucketCap = 32;
    cl_uint resultsNum = (cl_uint)sLen;           // bounded output = |S|
    size_t htElems = (size_t)bucketNum * (1 + hashBucketCap * 2);
    size_t htBytes = htElems * sizeof(cl_uint);

    // ---- probe block split (CPU/GPU), like the selection cPDE ----
    int execGpuPct = getenv("EXEC_GPU_PCT") ? atoi(getenv("EXEC_GPU_PCT")) : 0;
    if (execGpuPct < 0) execGpuPct = 0;
    if (execGpuPct > 100) execGpuPct = 100;
    int B = 8;                                    // probe blocks over S
    if (B > sLen) B = 1;
    int blkLen = (sLen + B - 1) / B;
    int gpuBlocks = (int)((double)B * execGpuPct / 100.0 + 0.5);
    if (gpuBlocks > B) gpuBlocks = B;
    int cpuBlocks = B - gpuBlocks;

    // ---- P (WAS) config: engages only on CPU-E probe blocks ----
    int peWassize = -1;
    {
        bool peOn = (getenv("PE_MODE") && atoi(getenv("PE_MODE")) != 0);
        int w = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
        if (peOn && w > 0 && g_prefetchEnabled && PrefetchCommandQueue && cpuBlocks > 0) peWassize = w;
    }
    bool useWAS = (peWassize > 0);

    {
        int pCUs = (g_prefetchEnabled && useWAS) ? 1 : 0;
        fprintf(stderr,
            "[cPDE-HJ] P=%d D=%d E_cpu=%d E_gpu=%s (gpu_pct=%d, blocks=%d: %d gpu / %d cpu)\n",
            pCUs, g_decomCUs, (cpuBlocks > 0 ? (g_decomEnabled ? g_nonDecomCUs : 1) : 0),
            (gpuBlocks > 0 ? "ON" : "OFF"), execGpuPct, B, gpuBlocks, cpuBlocks);
    }

    // ---- hash table + output buffers ----
    cl_mem rHashTable = NULL;
    CL_MALLOC(&rHashTable, htBytes);
    cl_uint zero = 0;
    clEnqueueFillBuffer(CommandQueue[0], rHashTable, &zero, sizeof(cl_uint), 0, htBytes, 0, NULL, NULL);
    // output: [0]=count, then 4 uints/match. Size to the bound |S|.
    CL_MALLOC(h_Rout, sizeof(cl_uint) * (4 + (size_t)resultsNum * 4));
    clEnqueueFillBuffer(CommandQueue[0], *h_Rout, &zero, sizeof(cl_uint), 0, sizeof(cl_uint), 0, NULL, NULL);
    clFinish(CommandQueue[0]);

    cl_kernel kDec = clCreateKernel(Program, "decompress_hj_kernel", &err);
    cl_kernel kBuild = clCreateKernel(Program, "build_kernel_w16", &err);
    cl_kernel kProbe = clCreateKernel(Program, "probe_kernel_w16", &err);     // CPU-E probe
    cl_kernel kProbeGpu = clCreateKernel(Program, "probe_kernel_w16", &err);  // GPU-E probe (separate obj)
    if (err != CL_SUCCESS || !kDec || !kBuild || !kProbe || !kProbeGpu) {
        printf("Error %d creating cPDE-HJ kernels, Line %u in file %s !!!\n\n", err, __LINE__, __FILE__);
        cl_clean(EXIT_FAILURE);
    }

    size_t lThreads = 256;
    size_t buildThreads = 256;
    static int hjNumWG = -1;
    if (hjNumWG < 0) hjNumWG = getenv("NUM_WG") ? atoi(getenv("NUM_WG")) : 32;
    size_t gWide = (size_t)hjNumWG * lThreads;    // global size for build/probe (E)

    cl_mem d_Rdec = NULL;
    CL_MALLOC(&d_Rdec, sizeof(Record) * rLen);
    if (!d_Rdec) { printf("Error: cPDE-HJ d_Rdec alloc failed, %s:%u\n", __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }
    {
        int baseR = 0;
        clSetKernelArg(kDec, 0, sizeof(cl_mem), &d_R);
        clSetKernelArg(kDec, 1, sizeof(cl_mem), &d_Rdec);
        clSetKernelArg(kDec, 2, sizeof(cl_int), &rLen);
        clSetKernelArg(kDec, 3, sizeof(cl_int), &baseR);
        size_t gDec = (size_t)rLen;
        gDec = ((gDec + lThreads - 1) / lThreads) * lThreads;
        if (gDec == 0) gDec = lThreads;
        err = clEnqueueNDRangeKernel(Qd, kDec, 1, NULL, &gDec, &lThreads, 0, NULL, NULL);
        if (err != CL_SUCCESS) { printf("Error %d enqueue R decompress, %s:%u\n", err, __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }
        clFinish(Qd);
    }
    {
        cl_uint rTupleNum = (cl_uint)rLen;
        clSetKernelArg(kBuild, 0, sizeof(cl_mem), &d_Rdec);
        clSetKernelArg(kBuild, 1, sizeof(cl_mem), &rHashTable);
        clSetKernelArg(kBuild, 2, sizeof(cl_uint), &rTupleNum);
        clSetKernelArg(kBuild, 3, sizeof(cl_uint), &bucketNum);
        clSetKernelArg(kBuild, 4, sizeof(cl_uint), &hashBucketCap);
        // build runs on the CPU-E queue (Qe); if E is all-GPU (cpuBlocks==0, fission
        // {n,0}) Qe falls back to CommandQueue[0] which still sees the same context.
        cl_command_queue Qbuild = (g_decomEnabled && NonDecomCommandQueue) ? NonDecomCommandQueue : CommandQueue[0];
        err = clEnqueueNDRangeKernel(Qbuild, kBuild, 1, NULL, &gWide, &buildThreads, 0, NULL, NULL);
        if (err != CL_SUCCESS) { printf("Error %d enqueue build, %s:%u\n", err, __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }
        clFinish(Qbuild);                          // BARRIER: hash table complete before probe
    }

    cl_mem was_buffer = NULL, dummy_buffer = NULL, last_tag_buffer = NULL;
    cl_kernel hk = NULL;
    if (useWAS) {
        cl_command_queue Qw = Qe;                  // init on the E (exec) sub-device
        size_t wasEntrySize = 4 * sizeof(cl_ulong);
        was_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)peWassize * wasEntrySize, NULL, &err);
        dummy_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 64, NULL, &err);
        last_tag_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)peWassize * sizeof(cl_ulong), NULL, &err);
        if (!was_buffer || !dummy_buffer || !last_tag_buffer) {
            printf("Error: cPDE-HJ PE WAS buffer alloc failed, %s:%u\n", __FILE__, __LINE__); cl_clean(EXIT_FAILURE);
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
        hk = clCreateKernel(Program, "WAS_kernel", &err);
        cl_int wmode = 3;                          // backward + spin + nopause (fixed)
        clSetKernelArg(hk, 0, sizeof(cl_mem), &was_buffer);
        clSetKernelArg(hk, 1, sizeof(cl_int), &peWassize);
        clSetKernelArg(hk, 2, sizeof(cl_mem), &dummy_buffer);
        clSetKernelArg(hk, 3, sizeof(cl_mem), &last_tag_buffer);
        clSetKernelArg(hk, 4, sizeof(cl_int), &wmode);
        size_t wg = 1, wl = 1;
        err = clEnqueueNDRangeKernel(PrefetchCommandQueue, hk, 1, NULL, &wg, &wl, 0, NULL, NULL);
        if (err != CL_SUCCESS) { printf("Error %d launching cPDE-HJ WAS helper, %s:%u\n", err, __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }
        clFlush(PrefetchCommandQueue);
        fprintf(stderr, "[PE] WAS cPDE-HJ probe engaged on prefetch sub-device (prefetching probe hash-table reads, wassize=%d)\n", peWassize);
    }

    // noWAS args for GPU-E probe blocks (and CPU-E when PE off).
    cl_mem wasNull = NULL; cl_int wasOff = -1;

    cl_mem d_Sdec = NULL;
    CL_MALLOC(&d_Sdec, sizeof(Record) * sLen);
    if (!d_Sdec) { printf("Error: cPDE-HJ d_Sdec alloc failed, %s:%u\n", __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }

    std::vector<cl_event> decEvt(B, (cl_event)NULL);
    std::vector<cl_event> prbEvt(B, (cl_event)NULL);

    for (int b = 0; b < B; b++) {
        int off = b * blkLen;
        int len = blkLen;
        if (off + len > sLen) len = sLen - off;
        if (len <= 0) { B = b; break; }

        cl_buffer_region rIn  = { (size_t)off * sizeof(unsigned short), (size_t)len * sizeof(unsigned short) };
        cl_buffer_region rOut = { (size_t)off * sizeof(Record),         (size_t)len * sizeof(Record) };
        cl_mem subIn  = clCreateSubBuffer(d_S,    CL_MEM_READ_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &rIn,  &err);
        cl_mem subDec = clCreateSubBuffer(d_Sdec, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rOut, &err);
        if (err != CL_SUCCESS) { printf("Error %d cPDE-HJ S sub-buffers (b=%d), %s:%u\n", err, b, __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }

        // D: decompress this S block on the decom sub-device.
        int baseS = off;                            // keep global rid surrogate
        clSetKernelArg(kDec, 0, sizeof(cl_mem), &subIn);
        clSetKernelArg(kDec, 1, sizeof(cl_mem), &subDec);
        clSetKernelArg(kDec, 2, sizeof(cl_int), &len);
        clSetKernelArg(kDec, 3, sizeof(cl_int), &baseS);
        size_t gDec = (size_t)len;
        gDec = ((gDec + lThreads - 1) / lThreads) * lThreads;
        if (gDec == 0) gDec = lThreads;
        err = clEnqueueNDRangeKernel(Qd, kDec, 1, NULL, &gDec, &lThreads, 0, NULL, &decEvt[b]);
        if (err != CL_SUCCESS) { printf("Error %d enqueue S decompress (b=%d), %s:%u\n", err, b, __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }
        clFlush(Qd);

        // E: probe this decompressed S block (waits on its decompress).
        bool gpuE = (b < gpuBlocks);
        cl_kernel kP = gpuE ? kProbeGpu : kProbe;
        cl_command_queue Qexec = gpuE ? Qg : Qe;
        cl_uint sTupleNum = (cl_uint)len;
        clSetKernelArg(kP, 0, sizeof(cl_mem), &rHashTable);
        clSetKernelArg(kP, 1, sizeof(cl_mem), &subDec);     // decompressed S block (key in .x)
        clSetKernelArg(kP, 2, sizeof(cl_mem), h_Rout);      // SHARED match counter+output
        clSetKernelArg(kP, 3, sizeof(cl_uint), &sTupleNum);
        clSetKernelArg(kP, 4, sizeof(cl_uint), &bucketNum);
        clSetKernelArg(kP, 5, sizeof(cl_uint), &hashBucketCap);
        clSetKernelArg(kP, 6, sizeof(cl_uint), &resultsNum);
        if (!gpuE && useWAS) {
            clSetKernelArg(kP, 7, sizeof(cl_mem), &was_buffer);
            clSetKernelArg(kP, 8, sizeof(cl_int), &peWassize);
            clSetKernelArg(kP, 9, sizeof(cl_mem), &dummy_buffer);
        } else {
            clSetKernelArg(kP, 7, sizeof(cl_mem), &wasNull); // noWAS
            clSetKernelArg(kP, 8, sizeof(cl_int), &wasOff);
            clSetKernelArg(kP, 9, sizeof(cl_mem), &wasNull);
        }
        size_t gP = gWide;
        size_t lP = lThreads;
        if (gpuE && lP > 256) lP = 256;             // GPU local-size cap
        if (gpuE) { gP = (gP / lP) * lP; if (gP == 0) gP = lP; }
        err = clEnqueueNDRangeKernel(Qexec, kP, 1, NULL, &gP, &lP, 1, &decEvt[b], &prbEvt[b]);
        if (err != CL_SUCCESS) { printf("Error %d enqueue probe (b=%d, %s), %s:%u\n", err, b, (gpuE ? "GPU" : "CPU"), __FILE__, __LINE__); cl_clean(EXIT_FAILURE); }
        clFlush(Qexec);

        clReleaseMemObject(subIn);
        clReleaseMemObject(subDec);
    }

    clFinish(Qd);
    if (cpuBlocks > 0 && Qe) clFinish(Qe);
    if (gpuBlocks > 0) clFinish(Qg);
    for (int b = 0; b < B; b++) {
        if (decEvt[b]) clReleaseEvent(decEvt[b]);
        if (prbEvt[b]) clReleaseEvent(prbEvt[b]);
    }

    // ---- P teardown: signal the WAS helper to stop, then join it. ----
    if (useWAS) {
        cl_uint *cs = (cl_uint *)clEnqueueMapBuffer(Qe, dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
        cs[1] = 1;
        clEnqueueUnmapMemObject(Qe, dummy_buffer, cs, 0, NULL, NULL);
        clFinish(Qe);
        clFinish(PrefetchCommandQueue);
        clReleaseKernel(hk);
        clReleaseMemObject(was_buffer);
        clReleaseMemObject(dummy_buffer);
        clReleaseMemObject(last_tag_buffer);
    }

    // read the actual match count for the banner.
    cl_uint matched = 0;
    clEnqueueReadBuffer(CommandQueue[0], *h_Rout, CL_TRUE, 0, sizeof(cl_uint), &matched, 0, NULL, NULL);
    fprintf(stderr, "[cPDE-HJ] D=%d E_cpu=%d E_gpu=%s blocks=%d: |R|=%d |S|=%d matched=%u (bucketNum=%u cap=%u)\n",
            g_decomCUs, (cpuBlocks > 0 ? (g_decomEnabled ? g_nonDecomCUs : 1) : 0),
            (gpuBlocks > 0 ? "ON" : "OFF"), B, rLen, sLen, matched, bucketNum, hashBucketCap);

    clReleaseKernel(kDec);
    clReleaseKernel(kBuild);
    clReleaseKernel(kProbe);
    clReleaseKernel(kProbeGpu);
    CL_FREE(d_Rdec);
    CL_FREE(d_Sdec);
    CL_FREE(rHashTable);
    return (int)resultsNum;
}

void testHJ(int rLen, int sLen)
{
	int _CPU_GPU=0;
	int result = 0;
	cl_uint  rHashTableBucketNum = 2 * 1024 * 1024;

	//size of R and S tables
	int memSizeR = sizeof(Record) * rLen;
	int memSizeS = sizeof(Record) * sLen;
	void * h_R;
	HOST_MALLOC(h_R,memSizeR);
	generateRand((Record*)h_R,TEST_MAX,rLen,0);
	void * h_S;
	HOST_MALLOC(h_S,memSizeS);
	generateRand((Record*)h_S,TEST_MAX,sLen,1);
	Record * h_Rout;
	CL_hj((Record *)h_R,rLen,(Record*)h_S,sLen,&h_Rout,_CPU_GPU);
	printf("HJFinish\n");
}

void pe_selftest() {
  int rLen = 4 * 1024 * 1024, sLen = rLen;
  int W = getenv("WASSIZE") ? atoi(getenv("WASSIZE")) : 1792;
  const cl_uint bucketNum = 2 * 1024 * 1024;
  size_t memSizeR = sizeof(Record) * rLen, memSizeS = sizeof(Record) * sLen;
  size_t htBytes = (size_t)rLen * sizeof(Record) + bucketNum * sizeof(cl_uint);
  cl_uint resultsNum = rLen, zero = 0;
  cl_int err;
  Record* hR = (Record*)malloc(memSizeR);
  Record* hS = (Record*)malloc(memSizeS);
  generateRand(hR, TEST_MAX, rLen, 0);
  generateRand(hS, TEST_MAX, sLen, 1);
  cl_command_queue Q = CommandQueue[0];

  cl_mem d_R = clCreateBuffer(Context, CL_MEM_READ_WRITE, memSizeR, NULL, &err);
  cl_mem d_S = clCreateBuffer(Context, CL_MEM_READ_WRITE, memSizeS, NULL, &err);
  cl_mem d_HT = clCreateBuffer(Context, CL_MEM_READ_WRITE, htBytes, NULL, &err);
  cl_mem d_Rout = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(Record)*resultsNum*2, NULL, &err);
  clEnqueueWriteBuffer(Q, d_R, CL_TRUE, 0, memSizeR, hR, 0, NULL, NULL);
  clEnqueueWriteBuffer(Q, d_S, CL_TRUE, 0, memSizeS, hS, 0, NULL, NULL);
  clEnqueueFillBuffer(Q, d_HT, &zero, sizeof(cl_uint), 0, htBytes, 0, NULL, NULL);
  clFinish(Q);
  { size_t bg=8192, bl=256; cl_kernel bk; cl_getKernel((char*)"build_kernel", &bk);
    clSetKernelArg(bk,0,sizeof(cl_mem),&d_R); clSetKernelArg(bk,1,sizeof(cl_mem),&d_HT);
    clSetKernelArg(bk,2,sizeof(cl_uint),&rLen); clSetKernelArg(bk,3,sizeof(cl_uint),&sLen);
    clSetKernelArg(bk,4,sizeof(cl_uint),&bucketNum);
    clEnqueueNDRangeKernel(Q,bk,1,NULL,&bg,&bl,0,NULL,NULL); clFinish(Q); clReleaseKernel(bk); }

  cl_mem nullbuf = NULL;
  // --- noWAS probe ---
  clEnqueueWriteBuffer(Q, d_Rout, CL_TRUE, 0, sizeof(cl_uint), &zero, 0, NULL, NULL); clFinish(Q);
  { int wassize=-1; size_t pg=8192, pl=256; cl_kernel pk; cl_getKernel((char*)"probe_kernel",&pk);
    clSetKernelArg(pk,0,sizeof(cl_mem),&d_HT); clSetKernelArg(pk,1,sizeof(cl_mem),&d_S);
    clSetKernelArg(pk,2,sizeof(cl_mem),&d_Rout); clSetKernelArg(pk,3,sizeof(cl_uint),&rLen);
    clSetKernelArg(pk,4,sizeof(cl_uint),&sLen); clSetKernelArg(pk,5,sizeof(cl_uint),&bucketNum);
    clSetKernelArg(pk,6,sizeof(cl_uint),&resultsNum); clSetKernelArg(pk,7,sizeof(cl_mem),&nullbuf);
    clSetKernelArg(pk,8,sizeof(cl_int),&wassize); clSetKernelArg(pk,9,sizeof(cl_mem),&nullbuf);
    clEnqueueNDRangeKernel(Q,pk,1,NULL,&pg,&pl,0,NULL,NULL); clFinish(Q); clReleaseKernel(pk); }
  cl_uint n1=0; clEnqueueReadBuffer(Q,d_Rout,CL_TRUE,0,sizeof(cl_uint),&n1,0,NULL,NULL);

  // --- WAS probe (helper mode 3) ---
  int wassize = W;
  size_t wasEntrySize = 4*sizeof(cl_ulong);
  cl_mem was_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)wassize*wasEntrySize, NULL, &err);
  cl_mem dummy_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 64, NULL, &err);
  cl_mem last_tag_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, (size_t)wassize*sizeof(cl_ulong), NULL, &err);
  void* dm = clEnqueueMapBuffer(Q,dummy_buffer,CL_TRUE,CL_MAP_WRITE,0,64,0,NULL,NULL,&err); memset(dm,0,64); clEnqueueUnmapMemObject(Q,dummy_buffer,dm,0,NULL,NULL); clFinish(Q);
  cl_ulong* wm = (cl_ulong*)clEnqueueMapBuffer(Q,was_buffer,CL_TRUE,CL_MAP_WRITE,0,(size_t)wassize*wasEntrySize,0,NULL,NULL,&err);
  void* dm2 = clEnqueueMapBuffer(Q,dummy_buffer,CL_TRUE,CL_MAP_READ,0,8,0,NULL,NULL,&err);
  cl_ulong dummyVal=(cl_ulong)(uintptr_t)dm2; clEnqueueUnmapMemObject(Q,dummy_buffer,dm2,0,NULL,NULL);
  for(int j=0;j<wassize;j++){wm[j*4]=dummyVal;wm[j*4+1]=dummyVal;wm[j*4+2]=(cl_ulong)-1;wm[j*4+3]=(cl_ulong)-1;}
  clEnqueueUnmapMemObject(Q,was_buffer,wm,0,NULL,NULL);
  cl_ulong* lm=(cl_ulong*)clEnqueueMapBuffer(Q,last_tag_buffer,CL_TRUE,CL_MAP_WRITE,0,(size_t)wassize*sizeof(cl_ulong),0,NULL,NULL,&err); memset(lm,0,(size_t)wassize*sizeof(cl_ulong)); clEnqueueUnmapMemObject(Q,last_tag_buffer,lm,0,NULL,NULL); clFinish(Q);
  cl_kernel hk=clCreateKernel(Program,"WAS_kernel",&err); cl_int wmode=3;
  clSetKernelArg(hk,0,sizeof(cl_mem),&was_buffer); clSetKernelArg(hk,1,sizeof(cl_int),&wassize);
  clSetKernelArg(hk,2,sizeof(cl_mem),&dummy_buffer); clSetKernelArg(hk,3,sizeof(cl_mem),&last_tag_buffer);
  clSetKernelArg(hk,4,sizeof(cl_int),&wmode);
  size_t wg=1,wl=1; clEnqueueNDRangeKernel(PrefetchCommandQueue,hk,1,NULL,&wg,&wl,0,NULL,NULL); clFlush(PrefetchCommandQueue);
  clEnqueueWriteBuffer(Q,d_Rout,CL_TRUE,0,sizeof(cl_uint),&zero,0,NULL,NULL); clFinish(Q);
  { size_t pg=8192,pl=256; cl_kernel pk; cl_getKernel((char*)"probe_kernel",&pk);
    clSetKernelArg(pk,0,sizeof(cl_mem),&d_HT); clSetKernelArg(pk,1,sizeof(cl_mem),&d_S);
    clSetKernelArg(pk,2,sizeof(cl_mem),&d_Rout); clSetKernelArg(pk,3,sizeof(cl_uint),&rLen);
    clSetKernelArg(pk,4,sizeof(cl_uint),&sLen); clSetKernelArg(pk,5,sizeof(cl_uint),&bucketNum);
    clSetKernelArg(pk,6,sizeof(cl_uint),&resultsNum); clSetKernelArg(pk,7,sizeof(cl_mem),&was_buffer);
    clSetKernelArg(pk,8,sizeof(cl_int),&wassize); clSetKernelArg(pk,9,sizeof(cl_mem),&dummy_buffer);
    clEnqueueNDRangeKernel(Q,pk,1,NULL,&pg,&pl,0,NULL,NULL); clFinish(Q); clReleaseKernel(pk); }
  cl_uint* cs=(cl_uint*)clEnqueueMapBuffer(Q,dummy_buffer,CL_TRUE,CL_MAP_WRITE,0,64,0,NULL,NULL,&err); cs[1]=1; clEnqueueUnmapMemObject(Q,dummy_buffer,cs,0,NULL,NULL); clFinish(Q); clFinish(PrefetchCommandQueue);
  cl_uint n2=0; clEnqueueReadBuffer(Q,d_Rout,CL_TRUE,0,sizeof(cl_uint),&n2,0,NULL,NULL);

  const char* status = (n1==n2) ? (n1!=0 ? "PARITY-OK" : "EMPTY!!") : "MISMATCH!!";
  printf("[PE-SELFTEST] noWAS matches=%u  WAS(wassize=%d) matches=%u  %s\n", n1, W, n2, status);
  fflush(stdout);
  clReleaseKernel(hk);
  clReleaseMemObject(was_buffer); clReleaseMemObject(dummy_buffer); clReleaseMemObject(last_tag_buffer);
  clReleaseMemObject(d_R); clReleaseMemObject(d_S); clReleaseMemObject(d_HT); clReleaseMemObject(d_Rout);
  free(hR); free(hS);
}
