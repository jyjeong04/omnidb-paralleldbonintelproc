#include "common.h"
#include "testJoin.h"
#include "OpenCL_DLL.h"
#include "KernelScheduler.h"
#include "scheduler.h"
extern cl_context Context;               // OpenCL context
extern cl_program Program;               // OpenCL program
extern cl_command_queue CommandQueue[2]; // OpenCL command queues
//extern "C" int CL_ninljOnly(cl_mem d_R,int rLen, cl_mem d_S, int sLen, cl_mem* d_Rout, int _CPU_GPU)
extern "C" int CL_ninlj(Record* h_R, int rLen, Record* h_S, int sLen,Record** h_Rout,int _CPU_GPU )
{
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];//now it seems this is a memory wasted method, change later!!!
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;
	int result=0;
	cl_mem d_Rout;
	cl_mem d_R;
	CL_MALLOC(& d_R, memSizeR);
	cl_writebuffer( d_R, h_R, memSizeR,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	cl_mem d_S;
	CL_MALLOC(& d_S, memSizeS );
	cl_writebuffer( d_S, h_S, memSizeR,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	result=NINJImpl(d_R,rLen,d_S,sLen,&d_Rout, &index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(index-1)%2]); 
	deschedule(CPU_GPU,burden);
	CL_FREE(d_R);
	CL_FREE(d_S);
	CL_FREE(d_Rout);
	clReleaseKernel(Kernel);
	clReleaseEvent(eventList[0]);
	clReleaseEvent(eventList[1]);
	//printf("NINLJFinish\n");
	return result;
}
extern "C" int CL_smjOnly(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem*  h_Joinout,int _CPU_GPU)
{
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];//now it seems this is a memory wasted method, change later!!!
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;
	int numResult = SMJImpl(d_R, rLen, d_S, sLen, h_Joinout,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(index-1)%2]); 
	deschedule(CPU_GPU,burden);
	clReleaseKernel(Kernel); 
	clReleaseEvent(eventList[0]);
	clReleaseEvent(eventList[1]);
	//printf("NINLJFinish\n");
	return numResult;
}
extern "C" int	CL_smj( Record* h_R, int rLen, Record* h_S, int sLen, Record** h_Joinout,int _CPU_GPU )
{
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];//now it seems this is a memory wasted method, change later!!!
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;

	cl_mem d_R;
	cl_mem d_S;
	cl_mem d_Joinout;

	CL_MALLOC( &d_R, sizeof(Record)*rLen );
	CL_MALLOC( &d_S, sizeof(Record)*sLen );
	cl_writebuffer( d_R, h_R, sizeof(Record)*rLen,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );
	cl_writebuffer( d_S, h_S, sizeof(Record)*sLen ,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);

	int numResult = SMJImpl(d_R, rLen, d_S, sLen, &d_Joinout,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);

	*h_Joinout = (Record*)malloc( sizeof(Record)*numResult );
	cl_readbuffer( *h_Joinout, d_Joinout, sizeof(Record)*numResult ,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(index-1)%2]); 
	deschedule(CPU_GPU,burden);
	CL_FREE(d_R);
	CL_FREE(d_S);
	CL_FREE(d_Joinout);
	clReleaseKernel(Kernel); 
	clReleaseEvent(eventList[0]);
	clReleaseEvent(eventList[1]);
	//printf("NINLJFinish\n");
	return numResult;
}

extern "C" int CL_mj( void * h_Rin, int rLen, Record* h_Sin, int sLen, Record** h_Joinout,int _CPU_GPU  )
{
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;
	cl_mem d_Rin;
	cl_mem d_Sin;
	cl_mem d_Joinout;

	CL_MALLOC( &d_Rin, sizeof(Record)*rLen );
	CL_MALLOC( &d_Sin, sizeof(Record)*sLen );

	cl_writebuffer( d_Rin, h_Rin, sizeof(Record)*rLen,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );
	cl_writebuffer( d_Sin, h_Sin, sizeof(Record)*sLen,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );

	int outSize = MJImpl( d_Rin, rLen, d_Sin, sLen, &d_Joinout ,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);

	*h_Joinout = (Record*)malloc( sizeof(Record)*outSize );
	//HOST_MALLOC( (void**)h_Joinout, sizeof(Record)*outSize );

	cl_readbuffer( *h_Joinout, d_Joinout, sizeof(Record)*outSize,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );
	clWaitForEvents(1,&eventList[(index-1)%2]); 
	deschedule(CPU_GPU,burden);
	CL_FREE(d_Rin);
	CL_FREE(d_Sin);
	CL_FREE(d_Joinout);
	clReleaseKernel(Kernel); 
	clReleaseEvent(eventList[0]);
	clReleaseEvent(eventList[1]);
	//printf("CL_mj\n");
	return outSize;
}

extern "C" int CL_hjOnly(cl_mem d_R, int rLen, cl_mem d_S, int sLen,cl_mem* h_Rout ,int _CPU_GPU)
{
	int result = 0;
	cl_uint  rHashTableBucketNum = 2 * 1024 * 1024;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;
	cl_mem rHashTable;
	CL_MALLOC(&rHashTable,rLen * sizeof(Record) + rHashTableBucketNum * sizeof(cl_uint));
	result = HJImpl(d_R,rLen,d_S,sLen,rHashTable,h_Rout,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);	
	clWaitForEvents(1,&eventList[(index-1)%2]); 
	deschedule(CPU_GPU,burden);
	CL_FREE(rHashTable);
	clReleaseKernel(Kernel); 
	clReleaseEvent(eventList[0]);
	clReleaseEvent(eventList[1]);
	return result;
}
extern "C" int CL_hj( Record* h_R, int rLen, Record* h_S, int sLen, Record** h_Rout ,int _CPU_GPU)
{
	int result = 0;
	cl_uint  rHashTableBucketNum = 2 * 1024 * 1024;

	//size of R and S tables
	int memSizeR = sizeof(Record) * rLen;
	int memSizeS = sizeof(Record) * sLen;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;

	cl_mem d_Rout;
	cl_mem d_R;
	CL_MALLOC(&d_R,memSizeR);
	cl_writebuffer(d_R,h_R,memSizeR,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	cl_mem d_S;
	CL_MALLOC(&d_S,memSizeS);
	cl_writebuffer(d_S,h_S,memSizeS,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	cl_mem rHashTable;
	CL_MALLOC(&rHashTable,rLen * sizeof(Record) + rHashTableBucketNum * sizeof(cl_uint));

	result = HJImpl(d_R,rLen,d_S,sLen,rHashTable,&d_Rout,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);
	*h_Rout = (Record*)malloc( sizeof(Record)*result );
	cl_readbuffer((*h_Rout),d_Rout,sizeof(Record)*result,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(index-1)%2]); 
	deschedule(CPU_GPU,burden);
	CL_FREE(d_R);
	CL_FREE(d_S);
	CL_FREE(d_Rout);
	CL_FREE(rHashTable);
	clReleaseKernel(Kernel); 
	clReleaseEvent(eventList[0]);
	clReleaseEvent(eventList[1]);
	//printf("HJFinish\n");
	return result;

}

extern "C" int CL_hj_PE(Record* h_R, int rLen, Record* h_S, int sLen,
                        Record** h_Rout, int _CPU_GPU, int wassize)
{
  // noWAS (any device) OR fission disabled OR GPU: use the existing path.
  if (wassize <= 0 || _CPU_GPU != 0 || !g_prefetchEnabled || !PrefetchCommandQueue)
    return CL_hj(h_R, rLen, h_S, sLen, h_Rout, _CPU_GPU);

  // ----- WAS path: build + probe-with-WAS on the 7-CU MainCPUSubDevice
  //       (CommandQueue[0], repartitioned by cl_init_prefetch),
  //       helper (WAS_kernel mode 3) on the 1-CU PrefetchCommandQueue. -----
  cl_int err;
  cl_command_queue Q = CommandQueue[0];
  const cl_uint bucketNum = 2 * 1024 * 1024;
  size_t memSizeR = sizeof(Record) * rLen;
  size_t memSizeS = sizeof(Record) * sLen;
  size_t htBytes  = (size_t)rLen * sizeof(Record) + bucketNum * sizeof(cl_uint);
  cl_uint resultsNum = rLen;
  cl_uint zero = 0;

  cl_mem d_R  = clCreateBuffer(Context, CL_MEM_READ_WRITE, memSizeR, NULL, &err);
  cl_mem d_S  = clCreateBuffer(Context, CL_MEM_READ_WRITE, memSizeS, NULL, &err);
  cl_mem d_HT = clCreateBuffer(Context, CL_MEM_READ_WRITE, htBytes, NULL, &err);
  cl_mem d_Rout = clCreateBuffer(Context, CL_MEM_READ_WRITE,
                                 sizeof(Record) * resultsNum * 2, NULL, &err);
  clEnqueueWriteBuffer(Q, d_R, CL_TRUE, 0, memSizeR, h_R, 0, NULL, NULL);
  clEnqueueWriteBuffer(Q, d_S, CL_TRUE, 0, memSizeS, h_S, 0, NULL, NULL);
  clEnqueueFillBuffer(Q, d_HT, &zero, sizeof(cl_uint), 0, htBytes, 0, NULL, NULL);
  clEnqueueWriteBuffer(Q, d_Rout, CL_TRUE, 0, sizeof(cl_uint), &zero, 0, NULL, NULL);
  clFinish(Q);

  size_t bg = 8192, bl = 256;
  cl_kernel bk; cl_getKernel((char*)"build_kernel", &bk);
  clSetKernelArg(bk, 0, sizeof(cl_mem), &d_R);
  clSetKernelArg(bk, 1, sizeof(cl_mem), &d_HT);
  clSetKernelArg(bk, 2, sizeof(cl_uint), (void*)&rLen);
  clSetKernelArg(bk, 3, sizeof(cl_uint), (void*)&sLen);
  clSetKernelArg(bk, 4, sizeof(cl_uint), &bucketNum);
  clEnqueueNDRangeKernel(Q, bk, 1, NULL, &bg, &bl, 0, NULL, NULL);
  clFinish(Q);

  size_t wasEntrySize = 4 * sizeof(cl_ulong);
  cl_mem was_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE,
                                     (size_t)wassize * wasEntrySize, NULL, &err);
  cl_mem dummy_buffer = clCreateBuffer(
      Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 64, NULL, &err);
  cl_mem last_tag_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE,
                                          (size_t)wassize * sizeof(cl_ulong), NULL, &err);
  void* dmap = clEnqueueMapBuffer(Q, dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
  memset(dmap, 0, 64);
  clEnqueueUnmapMemObject(Q, dummy_buffer, dmap, 0, NULL, NULL);
  clFinish(Q);
  cl_ulong* wmap = (cl_ulong*)clEnqueueMapBuffer(Q, was_buffer, CL_TRUE, CL_MAP_WRITE, 0,
                                                 (size_t)wassize * wasEntrySize, 0, NULL, NULL, &err);
  void* dmap2 = clEnqueueMapBuffer(Q, dummy_buffer, CL_TRUE, CL_MAP_READ, 0, 8, 0, NULL, NULL, &err);
  cl_ulong dummyVal = (cl_ulong)(uintptr_t)dmap2;
  clEnqueueUnmapMemObject(Q, dummy_buffer, dmap2, 0, NULL, NULL);
  for (int j = 0; j < wassize; j++) {
    wmap[j*4+0] = dummyVal; wmap[j*4+1] = dummyVal;
    wmap[j*4+2] = (cl_ulong)-1; wmap[j*4+3] = (cl_ulong)-1;
  }
  clEnqueueUnmapMemObject(Q, was_buffer, wmap, 0, NULL, NULL);
  cl_ulong* lmap = (cl_ulong*)clEnqueueMapBuffer(Q, last_tag_buffer, CL_TRUE, CL_MAP_WRITE, 0,
                                                 (size_t)wassize * sizeof(cl_ulong), 0, NULL, NULL, &err);
  memset(lmap, 0, (size_t)wassize * sizeof(cl_ulong));
  clEnqueueUnmapMemObject(Q, last_tag_buffer, lmap, 0, NULL, NULL);
  clFinish(Q);

  cl_kernel hk = clCreateKernel(Program, "WAS_kernel", &err);
  cl_int wmode = 3;
  clSetKernelArg(hk, 0, sizeof(cl_mem), &was_buffer);
  clSetKernelArg(hk, 1, sizeof(cl_int), &wassize);
  clSetKernelArg(hk, 2, sizeof(cl_mem), &dummy_buffer);
  clSetKernelArg(hk, 3, sizeof(cl_mem), &last_tag_buffer);
  clSetKernelArg(hk, 4, sizeof(cl_int), &wmode);
  size_t wg = 1, wl = 1;
  clEnqueueNDRangeKernel(PrefetchCommandQueue, hk, 1, NULL, &wg, &wl, 0, NULL, NULL);
  clFlush(PrefetchCommandQueue);

  size_t pg = 8192, pl = 256;
  cl_kernel pk; cl_getKernel((char*)"probe_kernel", &pk);
  clSetKernelArg(pk, 0, sizeof(cl_mem), &d_HT);
  clSetKernelArg(pk, 1, sizeof(cl_mem), &d_S);
  clSetKernelArg(pk, 2, sizeof(cl_mem), &d_Rout);
  clSetKernelArg(pk, 3, sizeof(cl_uint), (void*)&rLen);
  clSetKernelArg(pk, 4, sizeof(cl_uint), (void*)&sLen);
  clSetKernelArg(pk, 5, sizeof(cl_uint), &bucketNum);
  clSetKernelArg(pk, 6, sizeof(cl_uint), &resultsNum);
  clSetKernelArg(pk, 7, sizeof(cl_mem), &was_buffer);
  clSetKernelArg(pk, 8, sizeof(cl_int), &wassize);
  clSetKernelArg(pk, 9, sizeof(cl_mem), &dummy_buffer);
  clEnqueueNDRangeKernel(Q, pk, 1, NULL, &pg, &pl, 0, NULL, NULL);
  clFinish(Q);

  cl_uint* cstop = (cl_uint*)clEnqueueMapBuffer(Q, dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
  cstop[1] = 1;
  clEnqueueUnmapMemObject(Q, dummy_buffer, cstop, 0, NULL, NULL);
  clFinish(Q);
  clFinish(PrefetchCommandQueue);

  // Mirror CL_hj: return resultsNum and read resultsNum Records from offset 0.
  *h_Rout = (Record*)malloc(sizeof(Record) * resultsNum);
  clEnqueueReadBuffer(Q, d_Rout, CL_TRUE, 0, sizeof(Record) * resultsNum, *h_Rout, 0, NULL, NULL);

  clReleaseKernel(hk); clReleaseKernel(pk); clReleaseKernel(bk);
  clReleaseMemObject(was_buffer); clReleaseMemObject(dummy_buffer);
  clReleaseMemObject(last_tag_buffer);
  clReleaseMemObject(d_R); clReleaseMemObject(d_S);
  clReleaseMemObject(d_HT); clReleaseMemObject(d_Rout);
  return (int)resultsNum;
}
