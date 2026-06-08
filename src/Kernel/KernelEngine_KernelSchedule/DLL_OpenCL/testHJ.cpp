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

	// Enqueue + event-chain/burden bookkeeping mirrors kernel_enqueue
	// (KernelScheduler.cpp:108-134) — the source of truth. Keep in sync if that changes.
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
		printf("Error %d in clEnqueueNDRangeKernel (PE probe), Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
		cl_clean(EXIT_FAILURE);
	}
	clFlush(CommandQueue[CPU_GPU]);

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
	size_t groupSize = 256, globalSize = 8192;
	CL_MALLOC(d_Rout,sizeof(Record) * resultsNum * 2);

HJbuild_int(d_R, rHashTable, rLen , sLen, rHashTableBucketNum,
	globalSize,groupSize,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);

HJprobe_int(rHashTable,d_S,d_Rout,rLen,sLen,rHashTableBucketNum,
	resultsNum,globalSize,groupSize,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]); 
	return resultsNum;
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
