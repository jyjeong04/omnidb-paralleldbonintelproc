#include "common.h"
#include "testFilter.h"
#include "KernelScheduler.h"
#include "testScan.h"
#include "Helper.h"
#include "PrimitiveCommon.h"
#include "KernelScheduler.h"
#include "OpenCL_DLL.h"
#include "scheduler.h"
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

void filterImpl( cl_mem d_Rin, int beginPos, int rLen, cl_mem* d_Rout, int* outSize, 
				int numThread, int numBlock, int smallKey, int largeKey,int *index,cl_event *eventList,cl_kernel *Kernel, int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_mem d_mark;
	CL_MALLOC( &d_mark, sizeof(int)*rLen ) ;

	cl_mem d_markOutput;
	CL_MALLOC( &d_markOutput, sizeof(int)*rLen ) ;

	cl_mem d_temp;
	CL_MALLOC( &d_temp, sizeof(int)*rLen ) ;

	filterImpl_map_int( d_Rin, beginPos, rLen, d_mark, smallKey, largeKey, d_temp, numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]);
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

	//write the reduced result
	CL_MALLOC( d_Rout, sizeof(Record)*(*outSize) );
	filterImpl_write_int( *d_Rout, d_Rin, d_mark, d_markOutput, beginPos, rLen,numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU );
	clWaitForEvents(1,&eventList[(*index-1)%2]); 
	CL_FREE(d_mark);
	CL_FREE(d_markOutput);
	CL_FREE( d_temp );
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