#include "CSSTree.h"
#include "Helper.h"
#include "common.h"
#include "testJoin.h"
#include "testScan.h"
#include "testSplit.h"
#include <cassert> // Include before common.h to ensure assert is defined

// ---- Prefetching globals ----
extern int g_prefetchEnabled;
extern int g_prefetchWASSize;

extern cl_context Context;                    // OpenCL context
extern cl_command_queue CommandQueue[2];      // OpenCL command queues
extern cl_command_queue PrefetchCommandQueue; // WAS prefetch queue
extern cl_command_queue FullCPUCommandQueue;  // Full 8-CU queue (noWAS path)
extern cl_program Program;                    // OpenCL program
extern cl_device_id Device[2];                // OpenCL device (for L3 query)
extern double
    AddGPUBurden_Copy; //->initial in handshaking. fix rLen to 1024*1024
extern double AddCPUBurden_Copy;
extern double
    AddGPUBurden_Read; //->initial in handshaking. fix rLen to 1024*1024
extern double AddCPUBurden_Read;
extern double
    AddGPUBurden_Write; //->initial in handshaking. fix rLen to 1024*1024
extern double AddCPUBurden_Write;
extern double
    AddGPUBurden[60]; //->initial in handshaking. fix rLen to 1024*1024
extern double
    AddCPUBurden[60]; //->initial in handshaking. fix rLen to 1024*1024
extern double speedupGPUoverCPU[60 + 3];
extern double LothresholdForGPUApp;
extern double LothresholdForCPUApp;
extern double LoGPUBurden;
extern double LoCPUBurden;
extern double UpGPUBurden;
extern double UpCPUBurden;

extern cl_mem D1;
extern cl_mem D2;
extern cl_mem D3;
extern cl_mem D4;
extern cl_mem D5; // empty
extern cl_mem D6; // empty
extern cl_mem D7; // int*
extern void *H1;
extern void *H2;
extern void *H3;
extern void *H4;
extern void *H5; // empty
extern void *H6; // empty
extern int from;
extern int to;
extern int rLen;
extern int pLen;
// #define HandshakeDebug

double Count = 1000;
extern FILE *ofp;
void AddGPUBurden_Copy_handshake() {
  double i;
  double sum = 0;
  double scaler = 1000;
  int timer = DLL_genTimer(0);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_copyBuffer(D1, D2, rLen, 1);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("AddGPUBurden_Copy invocatio overhead, %f\n", t);
#endif
  }
  AddGPUBurden_Copy = sum / Count * scaler;
  printf("sum is %lf\n, AddGPUBurden_Copy invocatio overhead in average in "
         "GPU, %lf\n",
         sum, AddGPUBurden_Copy);
  fprintf(ofp, "gc %lf\n", AddGPUBurden_Copy);
}
void AddCPUBurden_Copy_handshake() {
  double i;
  double sum = 0;
  double scaler = 1000;
  int timer = DLL_genTimer(1);
  for (i = 0; i < Count; i++) {

    DLL_getTimer(timer);
    cl_copyBuffer(D1, D2, rLen, 0);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("AddCPUBurden_Copy invocatio overhead, %f\n", t);
#endif
  }
  AddCPUBurden_Copy = sum / Count * scaler;
  printf("sum is %lf\n, AddCPUBurden_Copy invocatio overhead in average in "
         "CPU, %lf\n",
         sum, AddCPUBurden_Copy);
  fprintf(ofp, "cc %lf\n", AddCPUBurden_Copy);
}
void AddGPUBurden_Read_handshake() {
  double i;
  double sum = 0;
  double scaler = 1000;
  int timer = DLL_genTimer(2);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_readbuffer(H1, D1, rLen, 1);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("AddGPUBurden_Read invocatio overhead, %f\n", t);
#endif
  }
  AddGPUBurden_Read = sum / Count * scaler;
  printf("sum is %lf\n, AddGPUBurden_Read invocatio overhead in average in "
         "GPU, %lf\n",
         sum, AddGPUBurden_Read);
  fprintf(ofp, "gr %lf\n", AddGPUBurden_Read);
}
void AddCPUBurden_Read_handshake() {
  double i;
  double sum = 0;
  double scaler = 1000;
  int timer = DLL_genTimer(3);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_readbuffer(H1, D1, rLen, 0);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("AddCPUBurden_Read invocatio overhead, %f\n", t);
#endif
  }
  AddCPUBurden_Read = sum / Count * scaler;
  printf("sum is %lf\n, AddCPUBurden_Read invocatio overhead in average in "
         "CPU, %lf\n",
         sum, AddCPUBurden_Read);
  fprintf(ofp, "cr %lf\n", AddCPUBurden_Read);
}

void AddGPUBurden_Write_handshake() {
  double i;
  double sum = 0;
  double scaler = 1000;
  int timer = DLL_genTimer(4);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_writebuffer(D1, H1, rLen, 1);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("AddGPUBurden_Write invocatio overhead, %f\n", t);
#endif
  }
  AddGPUBurden_Write = sum / Count * scaler;
  printf("sum is %lf\n, AddGPUBurden_Write invocatio overhead in average in "
         "GPU, %lf\n",
         sum, AddGPUBurden_Write);
  fprintf(ofp, "gw %lf\n", AddGPUBurden_Write);
}
void AddCPUBurden_Write_handshake() {
  double i;
  double sum = 0;
  double scaler = 1000;
  int timer = DLL_genTimer(5);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_writebuffer(D1, H1, rLen, 0);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("AddCPUBurden_Write invocatio overhead, %f\n", t);
#endif
  }
  AddCPUBurden_Write = sum / Count * scaler;
  printf("sum is %lf\n, AddCPUBurden_Write invocatio overhead in average in "
         "CPU, %lf\n",
         sum, AddCPUBurden_Write);
  fprintf(ofp, "cw %lf\n", AddCPUBurden_Write);
}
void Project_map_kernel_handshake(int _HandShakeCPU_GPU,
                                  cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 0;
  printf("Kid%d", kid);
  int timer = DLL_genTimer(kid);
  for (i = 0; i < Count; i++) {

    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("projection_map_kernel", _HandShakeKernel);
    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&pLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D3);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("projection_map_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, projection_map_kernel invocatio overhead in average "
           "in GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, projection_map_kernel invocatio overhead in average "
           "in CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void getResult_kernel_handshake(int _HandShakeCPU_GPU,
                                cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 1;
  printf("Kid%d", kid);
  int OPERATOR = 0;
  size_t numThreadsPerBlock_x = 256;
  size_t globalWorkingSetSize = 32 * 64;
  int timer = DLL_genTimer(kid);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_getKernel("getResult_kernel", _HandShakeKernel);
    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int), (void *)&rLen);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int),
                             (void *)&OPERATOR);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("getResult_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, getResult_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, getResult_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void copyLastElement_kernel_handshake(int _HandShakeCPU_GPU,
                                      cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 5;
  printf("Kid%d", kid);
  int base = 0;
  int offset = 0;
  int timer = DLL_genTimer(kid);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("copyLastElement_kernel", _HandShakeKernel);
    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int), (void *)&base);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int), (void *)&offset);

    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("copyLastElement_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, copyLastElement_kernel invocatio overhead in average "
           "in GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, copyLastElement_kernel invocatio overhead in average "
           "in CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void perscanFirstPass_kernel_handshake(int _HandShakeCPU_GPU,
                                       cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 6;
  printf("Kid%d", kid);
  // Set the Argument values
  int sharedMemSize = 3000;
  int bit_isFull = 1;
  int numElementsPerBlock = 512;
  int base = 0;
  int OPERATOR = 0;
  int d_odataOffset = 0;
  // Set the Argument values
  CL_MALLOC(&D4, sizeof(int) * rLen);
  size_t numThreadsPerBlock_x = 256;
  size_t globalWorkingSetSize = 32 * 64;
  int timer = DLL_genTimer(kid);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_getKernel("perscanFirstPass_kernel", _HandShakeKernel);
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D4);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D3);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int),
                             (void *)&numElementsPerBlock);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_int),
                             (void *)&bit_isFull);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_int), (void *)&base);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 7, sizeof(cl_int),
                             (void *)&d_odataOffset);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 8, sizeof(cl_int),
                             (void *)&OPERATOR);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 9, sizeof(cl_int),
                             (void *)&sharedMemSize);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("perscanFirstPass_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, perscanFirstPass_kernel invocatio overhead in "
           "average in GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, perscanFirstPass_kernel invocatio overhead in "
           "average in CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void perscan_kernel_handshake(int _HandShakeCPU_GPU,
                              cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 7;
  printf("Kid%d", kid);
  // Set the Argument values
  int sharedMemSize = 3000;
  int bit_isFull = 1;
  int numElementsPerBlock = 512;
  int base = 0;
  int OPERATOR = 0;
  int d_odataOffset = 0;
  // Set the Argument values
  size_t numThreadsPerBlock_x = 256;
  size_t globalWorkingSetSize = 32 * 64;
  int timer = DLL_genTimer(kid);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_getKernel("perscan_kernel", _HandShakeKernel);
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D3);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int),
                             (void *)&numElementsPerBlock);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int),
                             (void *)&bit_isFull);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_int), (void *)&base);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_int),
                             (void *)&d_odataOffset);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 7, sizeof(cl_int),
                             (void *)&OPERATOR);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 8, sizeof(cl_int),
                             (void *)&sharedMemSize);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("perscan_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, perscan_kernel invocatio overhead in average in GPU, "
           "%lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, perscan_kernel invocatio overhead in average in CPU, "
           "%lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void mapImpl_kernel_handshake(int _HandShakeCPU_GPU,
                              cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 14;
  printf("Kid%d", kid);
  int numRun = 8;
  int runSize = rLen / numRun;
  from = 0;
  to = runSize;
  int smallKey = rand() % 100;
  int largeKey = smallKey;
  int beginPos = 0;
  int value = 0x7f;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 64;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("mapImpl_kernel", _HandShakeKernel);

    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D5);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D6);

    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("mapImpl_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, mapImpl_kernel invocatio overhead in average in GPU, "
           "%lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, mapImpl_kernel invocatio overhead in average in CPU, "
           "%lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void memset_int_kernel_handshake(int _HandShakeCPU_GPU,
                                 cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 16;
  printf("Kid%d", kid);
  int numRun = 8;
  int runSize = rLen / numRun;
  from = 0;
  to = runSize;
  int smallKey = rand() % 100;
  int largeKey = smallKey;
  int beginPos = 0;
  int value = 0x7f;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("memset_int_kernel", _HandShakeKernel);

    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int), (void *)&value);

    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("memset_int_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, memset_int_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, memset_int_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void blockAddition_kernel_handshake(int _HandShakeCPU_GPU,
                                    cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 17;
  printf("Kid%d", kid);
  int numRun = 8;
  int runSize = rLen / numRun;
  from = 0;
  to = runSize;
  int smallKey = rand() % 100;
  int largeKey = smallKey;
  int beginPos = 0;
  int value = 0x7f;
  int blockSize = 256;
  /*private*/
  cl_event eventList[2];
  int index = 0;
  cl_kernel _Kernel;
  int FLAG_CPU_GPU;
  double burden;
  static cl_ulong usedLocalMemory; /**< Used local memory by _HandShakeKernel */
  ScanPara *SP;
  SP = (ScanPara *)malloc(sizeof(ScanPara));
  initScan(rLen, SP);

  /* Do block-wise sum */
  bScan_int(SP->gLength, &D1, &SP->outputBuffer[0], &SP->blockSumBuffer[0],
            &index, eventList, &_Kernel, &FLAG_CPU_GPU, &burden, SP, 0);
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  for (int i = 1; i < (int)SP->pass; i++) {
    bScan_int((cl_uint)(SP->gLength / pow((float)SP->blockSize, (float)i)),
              &SP->blockSumBuffer[i - 1], &SP->outputBuffer[i],
              &SP->blockSumBuffer[i], &index, eventList, &_Kernel,
              &FLAG_CPU_GPU, &burden, SP, 0);
  }
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  int tempLength =
      (int)(SP->gLength / pow((float)SP->blockSize, (float)SP->pass));

  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_int status;
    cl_getKernel("blockAddition_kernel", _HandShakeKernel);
    /* set the block size*/
    size_t globalThreads[1] = {static_cast<size_t>(tempLength)};
    size_t localThreads[1] = {static_cast<size_t>(SP->blockSize)};

    /*** Set appropriate arguments to the _HandShakeKernel ***/
    /* 1st argument to the _HandShakeKernel - inputBuffer */
    status = clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem),
                            (void *)&SP->tempBuffer);
    assert(status == CL_SUCCESS);

    /* 2nd argument to the _HandShakeKernel - SP->outputBuffer */
    status = clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem),
                            (void *)&SP->outputBuffer[SP->pass - 1]);
    assert(status == CL_SUCCESS);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("blockAddition_kernel invocatio overhead, %f\n", t);
#endif
  }
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  closeScan(SP);
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, blockAddition_kernel invocatio overhead in average "
           "in GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, blockAddition_kernel invocatio overhead in average "
           "in CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void prefixSum_kernel_handshake(int _HandShakeCPU_GPU,
                                cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 18;
  printf("Kid%d", kid);
  int numRun = 8;
  int runSize = rLen / numRun;
  from = 0;
  to = runSize;
  int smallKey = rand() % TEST_MAX;
  int largeKey = smallKey;
  int beginPos = 0;
  int value = 0x7f;
  int blockSize = 256;
  /*private*/
  cl_event eventList[2];
  int index = 0;
  cl_kernel _Kernel;
  int FLAG_CPU_GPU;
  double burden;
  static cl_ulong usedLocalMemory; /**< Used local memory by _HandShakeKernel */
  ScanPara *SP;
  SP = (ScanPara *)malloc(sizeof(ScanPara));
  initScan(rLen, SP);
  /* Do block-wise sum */
  bScan_int(SP->gLength, &D1, &SP->outputBuffer[0], &SP->blockSumBuffer[0],
            &index, eventList, &_Kernel, &FLAG_CPU_GPU, &burden, SP, 0);
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  for (int i = 1; i < (int)SP->pass; i++) {
    bScan_int((cl_uint)(SP->gLength / pow((float)SP->blockSize, (float)i)),
              &SP->blockSumBuffer[i - 1], &SP->outputBuffer[i],
              &SP->blockSumBuffer[i], &index, eventList, &_Kernel,
              &FLAG_CPU_GPU, &burden, SP, 0);
    clWaitForEvents(1, &eventList[(index - 1) % 2]);
  }
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  int tempLength =
      (int)(SP->gLength / pow((float)SP->blockSize, (float)SP->pass));

  size_t numThreadsPerBlock_x = 16;
  size_t globalWorkingSetSize = 32 * 64;
  cl_int status;
  int timer = DLL_genTimer(kid);
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_getKernel("prefixSum_kernel", _HandShakeKernel);
    /* Set appropriate arguments to the _HandShakeKernel */
    /* 1st argument to the _HandShakeKernel - SP->outputBuffer */
    status = clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem),
                            (void *)&SP->tempBuffer);
    assert(status == CL_SUCCESS);

    /* 2nd argument to the _HandShakeKernel - inputBuffer */
    status = clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem),
                            (void *)&SP->blockSumBuffer[SP->pass - 1]);
    assert(status == CL_SUCCESS);

    /* 3rd argument to the _HandShakeKernel - local memory */
    status = clSetKernelArg((*_HandShakeKernel), 2, tempLength * sizeof(cl_int),
                            NULL);
    assert(status == CL_SUCCESS);

    /* 4th argument to the _HandShakeKernel - SP->gLength */
    status = clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int),
                            (void *)&tempLength);
    assert(status == CL_SUCCESS);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("prefixSum_kernel invocation overhead, %f\n", t);
#endif
  }
  printf("prefixsum finish\n");
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  // closeScan(SP);
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, prefixSum_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, prefixSum_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void ScanLargeArrays_kernel_handshake(int _HandShakeCPU_GPU,
                                      cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 19;
  printf("Kid%d", kid);
  int numRun = 8;
  int runSize = rLen / numRun;
  from = 0;
  to = runSize;
  int smallKey = rand() % 100;
  int largeKey = smallKey;
  int beginPos = 0;
  int value = 0x7f;
  int blockSize = 256;
  /*private*/
  ScanPara *SP;
  SP = (ScanPara *)malloc(sizeof(ScanPara));
  initScan(rLen, SP);
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("ScanLargeArrays_kernel", _HandShakeKernel);
    /* Set appropriate arguments to the _HandShakeKernel */
    /* 1st argument to the _HandShakeKernel - SP->outputBuffer */
    cl_int status =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D6);
    /* 2nd argument to the _HandShakeKernel - inputBuffer */
    status |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D5);
    /* 3rd argument to the _HandShakeKernel - local memory */
    status |= clSetKernelArg((*_HandShakeKernel), 2, blockSize * sizeof(cl_int),
                             NULL);
    /* 4th argument to the _HandShakeKernel - block_size  */
    status |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int), &SP->blockSize);
    /* 5th argument to the _HandShakeKernel - SP->gLength  */
    status |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), &SP->gLength);
    /* 6th argument to the _HandShakeKernel - sum of blocks  */
    status |= clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_mem),
                             SP->blockSumBuffer);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("ScanLargeArrays_kernel invocatio overhead, %f\n", t);
#endif
  }
  closeScan(SP);
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, ScanLargeArrays_kernel invocatio overhead in average "
           "in GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, ScanLargeArrays_kernel invocatio overhead in average "
           "in CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}

void filterImpl_map_kernel_handshake(int _HandShakeCPU_GPU,
                                     cl_kernel *_HandShakeKernel) {
  if (_HandShakeCPU_GPU == 1) {
    exit(0);
  }
  double scaler = 1000;
  int kid = 20;
  printf("Kid%d sweep mode\n", kid);
  printf("forward + no spin loop\n");
  int beginPos = 0;
  int timer = DLL_genTimer(kid);
  size_t wasEntrySize = 4 * sizeof(cl_ulong); // 32 B

  // Paper §4.2 cache pressure model — per entry cost in shared L3:
  //   WAS struct (32B) + preload (M=2 × 64B cache line) = 160B
  // Paper threshold: total ≤ cache/4 (= 25% L3) for safety.
  cl_ulong l3CacheSize = 0;
  clGetDeviceInfo(Device[0], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong),
                  &l3CacheSize, NULL);
  const size_t kEntryStruct = 32;
  const size_t kPreloadPerEntry = 2 * 64;                       // 128 B
  const size_t kCostPerEntry = kEntryStruct + kPreloadPerEntry; // 160 B

  // Sweep axes (Method A focus: was_per_wi partitioning, helper sequential):
  //   - delta:    1792 (= 7×256 = 1 WG/main core, Method A)
  //               7168, 14336, ... (multiple WGs/core for comparison)
  //   - wassize:  K × delta where K = wassize/delta = was_per_wi
  //               (lookahead depth, paper §4.2 §4.4)
  struct SweepCfg {
    const char *label;
    int delta;
    int wassize;
  };
  struct SweepCfg cfgs[] = {
      //>>>CFGS_BEGIN
      {"M1 d=1792 noWAS7c", 1792, -1},
      {"M1 d=1792 noWAS8c", 1792, 0},
      {"M1 d=1792 w28", 1792, 28},
      {"M1 d=1792 w56", 1792, 56},
      {"M1 d=1792 w112", 1792, 112},
      {"M1 d=1792 w224", 1792, 224},
      {"M1 d=1792 w448", 1792, 448},
      {"M1 d=1792 w896", 1792, 896},
      {"M1 d=1792 w1792", 1792, 1792},
      {"M1 d=1792 w3584", 1792, 3584},
      {"M1 d=1792 w7168", 1792, 7168},
      {"M1 d=1792 w14336", 1792, 14336},
      {"M1 d=1792 w28672", 1792, 28672},
      {"M1 d=1792 w57344", 1792, 57344},
      //<<<CFGS_END
  };
  size_t n_cfgs = sizeof(cfgs) / sizeof(cfgs[0]);

  struct SweepResult {
    const char *label;
    int delta;
    int wassize;
    double total_sec;
    double avg_ms;
    cl_uint hits;
  } results[64];

  for (size_t c = 0; c < n_cfgs; c++) {
    // isolate one cfg for perf cache-miss measurement (FILTER_ONLY_W=<wassize>)
    if (getenv("FILTER_ONLY_W") &&
        cfgs[c].wassize != atoi(getenv("FILTER_ONLY_W")))
      continue;
    int wassize = cfgs[c].wassize;
    size_t globalWorkingSetSize = (size_t)cfgs[c].delta;
    size_t numThreadsPerBlock_x = 256;
    double sum = 0;
    cl_uint hits = 0;

    cl_mem was_buffer = NULL;
    cl_mem dummy_buffer = NULL;
    cl_mem last_tag_buffer = NULL;

    if (wassize > 0) {
      cl_int err;
      was_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE,
                                  wassize * wasEntrySize, NULL, &err);
      dummy_buffer = clCreateBuffer(
          Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 64, NULL, &err);
      if (err == CL_SUCCESS) {
        void *dmap = clEnqueueMapBuffer(CommandQueue[_HandShakeCPU_GPU],
                                        dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0,
                                        64, 0, NULL, NULL, &err);
        if (err == CL_SUCCESS && dmap) {
          memset(dmap, 0, 64);
          clEnqueueUnmapMemObject(CommandQueue[_HandShakeCPU_GPU], dummy_buffer,
                                  dmap, 0, NULL, NULL);
          clFinish(CommandQueue[_HandShakeCPU_GPU]);
        }
      }
      cl_ulong *wmap = (cl_ulong *)clEnqueueMapBuffer(
          CommandQueue[_HandShakeCPU_GPU], was_buffer, CL_TRUE, CL_MAP_WRITE, 0,
          wassize * wasEntrySize, 0, NULL, NULL, &err);
      if (err == CL_SUCCESS && wmap) {
        void *dmap2 =
            clEnqueueMapBuffer(CommandQueue[_HandShakeCPU_GPU], dummy_buffer,
                               CL_TRUE, CL_MAP_READ, 0, 8, 0, NULL, NULL, &err);
        cl_ulong dummyVal = (cl_ulong)(uintptr_t)dmap2;
        if (dmap2)
          clEnqueueUnmapMemObject(CommandQueue[_HandShakeCPU_GPU], dummy_buffer,
                                  dmap2, 0, NULL, NULL);
        for (int j = 0; j < wassize; j++) {
          wmap[j * 4 + 0] = dummyVal;
          wmap[j * 4 + 1] = dummyVal;
          wmap[j * 4 + 2] = (cl_ulong)-1;
          wmap[j * 4 + 3] = (cl_ulong)-1;
        }
        clEnqueueUnmapMemObject(CommandQueue[_HandShakeCPU_GPU], was_buffer,
                                wmap, 0, NULL, NULL);
        clFinish(CommandQueue[_HandShakeCPU_GPU]);
      }
      // last_tag: per-slot last seen pointer (as ulong). Init 0 (NULL — will
      // never collide with valid d_Rin pointers, so first real post triggers
      // preload).
      last_tag_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE,
                                       wassize * sizeof(cl_ulong), NULL, &err);
      if (err == CL_SUCCESS) {
        cl_ulong *lmap = (cl_ulong *)clEnqueueMapBuffer(
            CommandQueue[_HandShakeCPU_GPU], last_tag_buffer, CL_TRUE,
            CL_MAP_WRITE, 0, wassize * sizeof(cl_ulong), 0, NULL, NULL, &err);
        if (err == CL_SUCCESS && lmap) {
          memset(lmap, 0, wassize * sizeof(cl_ulong));
          clEnqueueUnmapMemObject(CommandQueue[_HandShakeCPU_GPU],
                                  last_tag_buffer, lmap, 0, NULL, NULL);
          clFinish(CommandQueue[_HandShakeCPU_GPU]);
        }
      }
    }

    cl_kernel wasKernel = NULL;
    if (wassize > 0 && was_buffer && dummy_buffer && last_tag_buffer &&
        PrefetchCommandQueue) {
      cl_int kerr;
      wasKernel = clCreateKernel(Program, "WAS_kernel", &kerr);
      if (kerr == CL_SUCCESS) {
        clSetKernelArg(wasKernel, 0, sizeof(cl_mem), (void *)&was_buffer);
        clSetKernelArg(wasKernel, 1, sizeof(cl_int), (void *)&wassize);
        clSetKernelArg(wasKernel, 2, sizeof(cl_mem), (void *)&dummy_buffer);
        clSetKernelArg(wasKernel, 3, sizeof(cl_mem), (void *)&last_tag_buffer);
        cl_int wmode = getenv("FILTER_MODE") ? atoi(getenv("FILTER_MODE")) : 6;
        clSetKernelArg(wasKernel, 4, sizeof(cl_int), (void *)&wmode);
        size_t wasGlobal = 1, wasLocal = 1;
        kerr = clEnqueueNDRangeKernel(PrefetchCommandQueue, wasKernel, 1, NULL,
                                      &wasGlobal, &wasLocal, 0, NULL, NULL);
        clFlush(PrefetchCommandQueue);
      }
    }

    // Queue selection by wassize:
    //   wassize == 0  -> noWAS on full 8 CUs (FullCPUCommandQueue)
    //   wassize <  0  -> noWAS on 7-CU MainCPUSubDevice (no swap) — control to
    //                    isolate the helper effect at equal main-core count
    //   wassize >  0  -> WAS: 7-CU main + helper on the prefetch CU
    cl_command_queue savedQ = CommandQueue[_HandShakeCPU_GPU];
    if (wassize == 0 && FullCPUCommandQueue) {
      CommandQueue[_HandShakeCPU_GPU] = FullCPUCommandQueue;
    }

    int selectionCount = getenv("FILTER_REPS") ? atoi(getenv("FILTER_REPS")) : 50;
    for (int it = 0; it < selectionCount; it++) {
      DLL_getTimer(timer);
      cl_getKernel((char *)"filterImpl_map_kernel", _HandShakeKernel);

      int smallKey = rand() % TEST_MAX;
      int largeKey = smallKey + 20000000;

      cl_int ciErr1 =
          clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
      ciErr1 |= clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int),
                               (void *)&beginPos);
      ciErr1 |=
          clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int), (void *)&rLen);
      ciErr1 |=
          clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D5);
      ciErr1 |= clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int),
                               (void *)&smallKey);
      ciErr1 |= clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_int),
                               (void *)&largeKey);
      ciErr1 |=
          clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_mem), (void *)&D3);
      ciErr1 |= clSetKernelArg((*_HandShakeKernel), 7, sizeof(cl_mem),
                               (void *)&was_buffer);
      ciErr1 |= clSetKernelArg((*_HandShakeKernel), 8, sizeof(cl_int),
                               (void *)&wassize);
      ciErr1 |= clSetKernelArg((*_HandShakeKernel), 9, sizeof(cl_mem),
                               (void *)&dummy_buffer);
      cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                      _HandShakeKernel, _HandShakeCPU_GPU);
      double t = DLL_getTimer(timer);
      sum += t;
    }

    // Restore the original queue before any WAS cleanup runs on it.
    CommandQueue[_HandShakeCPU_GPU] = savedQ;

    if (wasKernel && dummy_buffer) {
      cl_int err;
      cl_uint *dmap = (cl_uint *)clEnqueueMapBuffer(
          CommandQueue[_HandShakeCPU_GPU], dummy_buffer, CL_TRUE, CL_MAP_WRITE,
          0, 64, 0, NULL, NULL, &err);
      if (err == CL_SUCCESS && dmap) {
        dmap[1] = 1;
        clEnqueueUnmapMemObject(CommandQueue[_HandShakeCPU_GPU], dummy_buffer,
                                dmap, 0, NULL, NULL);
        clFinish(CommandQueue[_HandShakeCPU_GPU]);
      }
      if (PrefetchCommandQueue)
        clFinish(PrefetchCommandQueue);

      cl_uint *counterPtr = (cl_uint *)clEnqueueMapBuffer(
          CommandQueue[_HandShakeCPU_GPU], dummy_buffer, CL_TRUE, CL_MAP_READ,
          0, sizeof(cl_uint), 0, NULL, NULL, &err);
      if (err == CL_SUCCESS && counterPtr) {
        hits = *counterPtr;
        clEnqueueUnmapMemObject(CommandQueue[_HandShakeCPU_GPU], dummy_buffer,
                                counterPtr, 0, NULL, NULL);
        clFinish(CommandQueue[_HandShakeCPU_GPU]);
      }
      clReleaseKernel(wasKernel);
    }

    if (was_buffer)
      clReleaseMemObject(was_buffer);
    if (dummy_buffer)
      clReleaseMemObject(dummy_buffer);
    if (last_tag_buffer)
      clReleaseMemObject(last_tag_buffer);

    results[c].label = cfgs[c].label;
    results[c].delta = cfgs[c].delta;
    results[c].wassize = wassize;
    results[c].total_sec = sum;
    results[c].avg_ms = sum / selectionCount * scaler;
    results[c].hits = hits;

    // Paper §4.2 cache footprint: total = wassize × (struct + preload)
    double l3_pct = (l3CacheSize > 0) ? 100.0 * (double)wassize *
                                            kCostPerEntry / (double)l3CacheSize
                                      : 0.0;
    const char *paper_flag = (l3_pct > 25.0) ? " §4.2>L3/4" : "";

    printf("[KID20-SWEEP %2zu/%2zu] %-25s d=%6d w=%6d "
           "L3=%5.1f%% avg=%7.2fms hits=%u%s\n",
           c + 1, n_cfgs, cfgs[c].label, cfgs[c].delta, wassize, l3_pct,
           results[c].avg_ms, hits, paper_flag);
    fflush(stdout);
  }

  printf("\n[KID20-SWEEP RESULTS]  (paper §4.2: L3≤25%% safe)\n");
  printf("%-25s %6s %7s %7s %10s %12s\n", "label", "delta", "wassize", "L3%",
         "avg(ms)", "hits");
  printf("------------------------- ------ ------- --- ------- ---------- "
         "------------\n");
  size_t best = 0;
  for (size_t c = 0; c < n_cfgs; c++) {
    if (results[c].avg_ms < results[best].avg_ms)
      best = c;
    double l3_pct = (l3CacheSize > 0) ? 100.0 * (double)results[c].wassize *
                                            kCostPerEntry / (double)l3CacheSize
                                      : 0.0;
    const char *flag = (l3_pct > 25.0) ? " ⚠" : "";
    printf("%-25s %6d %7d %6.1f%% %10.3f %12u%s\n", results[c].label,
           results[c].delta, results[c].wassize, l3_pct, results[c].avg_ms,
           results[c].hits, flag);
  }
  printf("[KID20-SWEEP BEST] %s avg=%.2fms (vs BASE noWAS-d=131k=%.2fms)\n",
         results[best].label, results[best].avg_ms, results[0].avg_ms);

  AddCPUBurden[kid] = results[best].avg_ms;
  printf("filterImpl_map_kernel avg in CPU: %lf ms (best from sweep)\n",
         AddCPUBurden[kid]);
}
void filterImpl_write_kernel_handshake(int _HandShakeCPU_GPU,
                                       cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 23;
  printf("Kid%d", kid);
  int numRun = 8;
  int runSize = rLen / numRun;
  from = 0;
  to = runSize;
  int smallKey = rand() % 100;
  int largeKey = smallKey;
  int beginPos = 0;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("filterImpl_write_kernel", _HandShakeKernel);

    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D5);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D6);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int),
                             (void *)&beginPos);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_int), (void *)&rLen);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("filterImpl_write_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, filterImpl_write_kernel invocatio overhead in "
           "average in GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, filterImpl_write_kernel invocatio overhead in "
           "average in CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void optScatter_kernel_handshake(int _HandShakeCPU_GPU,
                                 cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 24;
  printf("Kid%d", kid);
  int numRun = 8;
  int runSize = rLen / numRun;
  from = 0;
  to = runSize;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("optScatter_kernel", _HandShakeKernel);
    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int), (void *)&from);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), (void *)&to);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_mem), (void *)&D3);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("optScatter_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, optScatter_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, optScatter_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void optGather_kernel_handshake(int _HandShakeCPU_GPU,
                                cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 25;
  printf("Kid%d", kid);
  int numRun = 8;
  int runSize = rLen / numRun;
  from = 0;
  to = runSize;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("optGather_kernel", _HandShakeKernel);
    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int), (void *)&from);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), (void *)&to);
    ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_mem), (void *)&D3);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_int), (void *)&pLen);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("optGather_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, optGather_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, optGather_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}

void partition_kernel_handshake(int _HandShakeCPU_GPU,
                                cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 26;
  printf("Kid%d", kid);
  int numPart = 1;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 64;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("partition_kernel", _HandShakeKernel);
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int),
                             (void *)&numPart);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D5);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_mem), (void *)&D6);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("partition_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, partition_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, partition_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}

void mapPart_kernel_handshake(int _HandShakeCPU_GPU,
                              cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 27;
  printf("Kid%d", kid);
  int numPart = 1;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("mapPart_kernel", _HandShakeKernel);
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int),
                             (void *)&numPart);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D5);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_mem), (void *)&D6);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("mapPart_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, mapPart_kernel invocatio overhead in average in GPU, "
           "%lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, mapPart_kernel invocatio overhead in average in CPU, "
           "%lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void countHist_kernel_handshake(int _HandShakeCPU_GPU,
                                cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 28;
  printf("Kid%d", kid);
  int numThreadPB = 256;
  size_t globalWorkingSetSize = 32 * 64;
  int numPart = 8;
  /*private*/
  cl_event eventList[2]; // now it seems this is a memory wasted method, change
                         // later!!!
  int index = 0;
  double burden;
  int FLAG_CPU_GPU = 0;
  // map->pid
  mapPart(D1, rLen, numPart, D6, SPLIT, numThreadPB, globalWorkingSetSize,
          &index, eventList, _HandShakeKernel, &FLAG_CPU_GPU, &burden, 0);

  // pid->write loc
  size_t numThreadsPerBlock_x = 0;
  if (numThreadPB == -1)
    numThreadsPerBlock_x = 1 << (log2((int)(SHARED_MEMORY_PER_PROCESSOR /
                                            (numPart * sizeof(int)))));
  else
    numThreadsPerBlock_x = numThreadPB;
  if (numThreadsPerBlock_x > 256)
    numThreadsPerBlock_x = 256;
  int sharedMemSize = numThreadsPerBlock_x * numPart * sizeof(int);

  // assert(numThreadsPerBlock_x>=16);
  int numThreadsPerBlock_y = 1;
  int numBlock_x;
  int numThreadBlock = -1;
  if (numThreadBlock == -1)
    numBlock_x = 512;
  else
    numBlock_x = numThreadBlock;
  // printf("numThreadsPerBlock_x, %d,sharedMemSize, %d,  numBlock_x, %d\n",
  // numThreadsPerBlock_x, sharedMemSize,numBlock_x);
  int numBlock_y = 1;
  int numThread = numBlock_x * numThreadsPerBlock_x;
  int numInPS = numThread * numPart;

  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    cl_getKernel("countHist_kernel", _HandShakeKernel);
    // Set the Argument values
    //(__global Record *d_R, int rLen,__global int *loc, int from, int to,
    //__global Record *d_S)
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D6);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int),
                             (void *)&numPart);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D5);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 4, sharedMemSize, NULL);

    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("countHist_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, countHist_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, countHist_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void writeHist_kernel_handshake(int _HandShakeCPU_GPU,
                                cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 29;
  printf("Kid%d", kid);
  int numPart = 8;
  int numThreadPB = 256;
  int numBlock = 32 * 64;
  size_t globalWorkingSetSize = 32 * 64;

  /*private*/
  cl_event eventList[2]; // now it seems this is a memory wasted method, change
                         // later!!!
  int index = 0;
  double burden;
  int FLAG_CPU_GPU = 0;

  int result = 0;
  int memSize = sizeof(Record) * rLen;
  int outSize = sizeof(Record) * rLen;
  void *Rin;
  HOST_MALLOC(Rin, memSize);
  void *h_bound;
  HOST_MALLOC(h_bound, numPart * sizeof(Record));
  generateRand((Record *)Rin, TEST_MAX, rLen, 0);
  void *Rout;
  HOST_MALLOC(Rout, outSize);
  CL_MALLOC(&D1, memSize);
  CL_MALLOC(&D2, outSize);
  CL_MALLOC(&D3, numPart * sizeof(Record));

  // copy to
  cl_writebuffer(D1, Rin, memSize, &index, eventList, &FLAG_CPU_GPU, &burden,
                 0);
  // map->pid
  CL_MALLOC(&D5, sizeof(int) * rLen);
  mapPart(D1, rLen, numPart, D5, SPLIT, numThreadPB, numBlock, &index,
          eventList, _HandShakeKernel, &FLAG_CPU_GPU, &burden, 0);

  /*void* tempResult;
  int totalSize=sizeof(int)*rLen;
  HOST_MALLOC(tempResult, totalSize);
  //cl_readbuffer(tempResult,D5,totalSize,_HandShakeCPU_GPU);*/

  // pid->write loc
  size_t numThreadsPerBlock_x = 0;
  if (numThreadPB == -1)
    numThreadsPerBlock_x = 1 << (log2((int)(SHARED_MEMORY_PER_PROCESSOR /
                                            (numPart * sizeof(int)))));
  else
    numThreadsPerBlock_x = numThreadPB;
  if (numThreadsPerBlock_x > 256)
    numThreadsPerBlock_x = 256;
  int sharedMemSize = numThreadsPerBlock_x * numPart * sizeof(int);

  // assert(numThreadsPerBlock_x>=16);
  int numThreadsPerBlock_y = 1;
  int numBlock_x;
  int numThreadBlock = -1;
  if (numThreadBlock == -1)
    numBlock_x = 512;
  else
    numBlock_x = numThreadBlock;
  // printf("numThreadsPerBlock_x, %d,sharedMemSize, %d,  numBlock_x, %d\n",
  // numThreadsPerBlock_x, sharedMemSize,numBlock_x);
  int numBlock_y = 1;
  int numThread = numBlock_x * numThreadsPerBlock_x;
  int numInPS = numThread * numPart;
  CL_MALLOC(&D6, sizeof(int) * numInPS);
  countHist_int(D5, rLen, numPart, D6, numThreadsPerBlock_x, numBlock_x,
                sharedMemSize, &index, eventList, _HandShakeKernel,
                &FLAG_CPU_GPU, &burden, 0);

  // prefix sum
  CL_MALLOC(&D7, sizeof(int) * numInPS);
  ScanPara *SP;
  SP = (ScanPara *)malloc(sizeof(ScanPara));
  initScan(numInPS, SP);
  scanImpl(D6, numInPS, D7, &index, eventList, _HandShakeKernel, &FLAG_CPU_GPU,
           &burden, SP, 0);
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  closeScan(SP);

  CL_MALLOC(&D6, sizeof(int) * rLen);
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    cl_getKernel("writeHist_kernel", _HandShakeKernel);
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D5);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int),
                             (void *)&numPart);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D7);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_mem), (void *)&D6);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 5, sharedMemSize, NULL);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("writeHist_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, writeHist_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, writeHist_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void getBound_kernel_handshake(int _HandShakeCPU_GPU,
                               cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 30;
  printf("Kid%d", kid);
  int numPart = 8;
  size_t numThreadsPerBlock_x = 256;
  size_t globalWorkingSetSize = 32 * 64;
  int sharedMemSize = numThreadsPerBlock_x * numPart * sizeof(int);
  int interval = 0;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    cl_getKernel("getBound_kernel", _HandShakeKernel);
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D5);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int),
                             (void *)&interval);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int), (void *)&rLen);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int),
                             (void *)&numPart);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_mem), (void *)&D6);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("getBound_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, getBound_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, getBound_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}

void BitonicSort_kernel_handshake(int _HandShakeCPU_GPU,
                                  cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 31;
  printf("Kid%d", kid);
  int stage = 0;
  int passOfStage = 0;
  cl_int sortAscending = 1; // 1: ascending order, 0: descending order
  size_t numThreadsPerBlock_x = 256;
  size_t globalWorkingSetSize = 32 * 64;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    cl_getKernel("BitonicSort_kernel", _HandShakeKernel);

    cl_int err =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    err |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_uint), (void *)&stage);
    err |= clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_uint),
                          (void *)&passOfStage);
    err |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_uint), (void *)&rLen);
    err |= clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_uint),
                          (void *)&sortAscending);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("BitonicSort_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, BitonicSort_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, BitonicSort_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}

void gpuNLJ_kernel_handshake(int _HandShakeCPU_GPU,
                             cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 38;
  printf("kid:%d\n", kid);
  int sStart = 0;
  int rLen = 256 * 1024;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("gpuNLJ_kernel", _HandShakeKernel);
    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D6);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D3);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int), (void *)&sStart);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), (void *)&rLen);
    ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_int), (void *)&rLen);
    ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_mem), (void *)&D5);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("gpuNLJ_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gpuNLJ_kernel invocatio overhead in average in GPU, "
           "%lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gpuNLJ_kernel invocatio overhead in average in CPU, "
           "%lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void nlj_write_kernel_handshake(int _HandShakeCPU_GPU,
                                cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 39;
  printf("Kid%d", kid);
  int sStart = 0;
  int rLen = 256 * 1024;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("nlj_write_kernel", _HandShakeKernel);
    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int), (void *)&sStart);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int), (void *)&rLen);
    ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), (void *)&rLen);
    ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_mem), (void *)&D5);
    ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_mem), (void *)&D3);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("nlj_write_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, nlj_write_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, nlj_write_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void gSearchTree_kernel_handshake(int _HandShakeCPU_GPU,
                                  cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 40;
  printf("Kid%d", kid);
  unsigned int nDataNodes;
  nDataNodes = uintCeilingDiv(rLen, TREE_NODE_SIZE);
  // #region Calculate parameters on host
  unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, nDataNodes);
  unsigned int nDirNodes = uintCeilingDiv(nDataNodes - 1, TREE_NODE_SIZE);
  unsigned int tree_size = nDirNodes + nDataNodes;
  unsigned int bottom_start =
      (uintPower(TREE_FANOUT, lvlDir) - 1) / TREE_NODE_SIZE;
  // #endregion
  unsigned int nNodesPerBlock = uintCeilingDiv(nDirNodes, BLCK_PER_GRID_create);
  unsigned int nKeysPerThread = uintCeilingDiv(pLen, THRD_PER_GRID_search);
  // #region Execute on device
  int numThreadPB = THRD_PER_BLCK_create;
  int numBlock = BLCK_PER_GRID_create;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("gSearchTree_kernel", _HandShakeKernel);

    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int),
                             (void *)&nDataNodes);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int),
                             (void *)&nDirNodes);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), (void *)&lvlDir);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_mem), (void *)&D3);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_mem), (void *)&D4);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 7, sizeof(cl_int), (void *)&pLen);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 8, sizeof(cl_int),
                             (void *)&nKeysPerThread);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 9, sizeof(cl_int),
                             (void *)&tree_size);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 10, sizeof(cl_int),
                             (void *)&bottom_start);
    int rLen = nDirNodes * nKeysPerThread / THRD_PER_GRID_search;
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("gSearchTree_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gSearchTree_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gSearchTree_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void gIndexJoin_kernel_handshake(int _HandShakeCPU_GPU,
                                 cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 41;
  printf("Kid%d", kid);
  unsigned int nDataNodes;
  nDataNodes = uintCeilingDiv(rLen, TREE_NODE_SIZE);
  // #region Calculate parameters on host
  unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, nDataNodes);
  unsigned int nDirNodes = uintCeilingDiv(nDataNodes - 1, TREE_NODE_SIZE);
  unsigned int tree_size = nDirNodes + nDataNodes;
  unsigned int bottom_start =
      (uintPower(TREE_FANOUT, lvlDir) - 1) / TREE_NODE_SIZE;
  // #endregion
  unsigned int nNodesPerBlock = uintCeilingDiv(nDirNodes, BLCK_PER_GRID_create);
  unsigned int nKeysPerThread = uintCeilingDiv(pLen, THRD_PER_GRID_search);
  int clusterSize = uintCeilingDiv(pLen, THRD_PER_GRID_join);
  // #region Execute on device
  int numThreadPB = THRD_PER_BLCK_create;
  int numBlock = BLCK_PER_GRID_create;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("gIndexJoin_kernel", _HandShakeKernel);

    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D5);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), (void *)&pLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_mem), (void *)&D6);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_int),
                             (void *)&clusterSize);
    int rLen = nDirNodes * nKeysPerThread / THRD_PER_GRID_search;
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("gIndexJoin_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gIndexJoin_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gIndexJoin_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void gJoinWithWrite_kernel_handshake(int _HandShakeCPU_GPU,
                                     cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 42;
  printf("Kid%d", kid);
  unsigned int nDataNodes;
  nDataNodes = uintCeilingDiv(rLen, TREE_NODE_SIZE);
  // #region Calculate parameters on host
  unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, nDataNodes);
  unsigned int nDirNodes = uintCeilingDiv(nDataNodes - 1, TREE_NODE_SIZE);
  unsigned int tree_size = nDirNodes + nDataNodes;
  unsigned int bottom_start =
      (uintPower(TREE_FANOUT, lvlDir) - 1) / TREE_NODE_SIZE;
  // #endregion
  unsigned int nNodesPerBlock = uintCeilingDiv(nDirNodes, BLCK_PER_GRID_create);
  unsigned int nKeysPerThread = uintCeilingDiv(pLen, THRD_PER_GRID_search);
  int clusterSize = uintCeilingDiv(pLen, THRD_PER_GRID_join);
  // #region Execute on device
  int numThreadPB = THRD_PER_BLCK_create;
  int numBlock = BLCK_PER_GRID_create;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("gJoinWithWrite_kernel", _HandShakeKernel);

    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D5);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), (void *)&pLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_mem), (void *)&D6);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_mem), (void *)&D3);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 7, sizeof(cl_int),
                             (void *)&clusterSize);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("gJoinWithWrite_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gJoinWithWrite_kernel invocatio overhead in average "
           "in GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gJoinWithWrite_kernel invocatio overhead in average "
           "in CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void gCreateIndex_kernel_handshake(int _HandShakeCPU_GPU,
                                   cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 43;
  printf("Kid%d", kid);
  unsigned int nDataNodes;
  nDataNodes = uintCeilingDiv(rLen, TREE_NODE_SIZE);
  // #region Calculate parameters on host
  unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, nDataNodes);
  unsigned int nDirNodes = uintCeilingDiv(nDataNodes - 1, TREE_NODE_SIZE);
  unsigned int tree_size = nDirNodes + nDataNodes;
  unsigned int bottom_start =
      (uintPower(TREE_FANOUT, lvlDir) - 1) / TREE_NODE_SIZE;
  // #endregion
  unsigned int nNodesPerBlock = uintCeilingDiv(nDirNodes, BLCK_PER_GRID_create);

  // #region Execute on device
  int numThreadPB = THRD_PER_BLCK_create;
  int numBlock = BLCK_PER_GRID_create;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("gCreateIndex_kernel", _HandShakeKernel);

    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), (void *)&D2);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int),
                             (void *)&nDirNodes);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int),
                             (void *)&tree_size);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int),
                             (void *)&bottom_start);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_int),
                             (void *)&nNodesPerBlock);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("gCreateIndex_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gCreateIndex_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gCreateIndex_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void quanMap_kernel_handshake(int _HandShakeCPU_GPU,
                              cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 44;
  printf("Kid%d", kid);
  size_t globalWorkingSetSize = 32 * 64;
  /*private*/
  int interval = SMJ_NUM_THREADS_PER_BLOCK;
  size_t numThreadsPerBlock_x = interval;
  size_t numThreadsPerBlock_y = 1;
  size_t numBlock_X = divRoundUp(rLen, interval);
  size_t numBlock_Y = 1;
  if (numBlock_X > NLJ_MAX_NUM_BLOCK_PER_DIM) {
    numBlock_Y = numBlock_X / NLJ_MAX_NUM_BLOCK_PER_DIM;
    if (numBlock_X % NLJ_MAX_NUM_BLOCK_PER_DIM != 0)
      numBlock_Y++;
    numBlock_X = NLJ_MAX_NUM_BLOCK_PER_DIM;
  }
  /////////////
  size_t thread[2] = {numThreadsPerBlock_x, numThreadsPerBlock_y};
  size_t grid[2] = {
      numBlock_X * numThreadsPerBlock_x * numBlock_Y * numThreadsPerBlock_y, 1};
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    cl_getKernel("quanMap_kernel", _HandShakeKernel);

    // Set the Argument values
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int),
                             (void *)&interval);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, interval * sizeof(Record), NULL);
    cl_launchKernel(1, grid, thread, _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("gCreateIndex_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, quanMap_kernel invocatio overhead in average in GPU, "
           "%lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, quanMap_kernel invocatio overhead in average in CPU, "
           "%lf\n",
           sum, AddCPUBurden[kid]);
  }
}

void gSearchTree_usingKeys_kernel_handshake(int _HandShakeCPU_GPU,
                                            cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 45;
  printf("Kid%d", kid);
  /*private*/
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  cl_event eventList[2];
  int index = 0;
  cl_kernel test_Kernel;
  int FLAG_CPU_GPU;
  double burden;
  int numResult = 0;

  int interval = SMJ_NUM_THREADS_PER_BLOCK;
  int numQuanR = divRoundUp(rLen, interval);
  CL_MALLOC(&D3, numQuanR * sizeof(int) * 2);
  getQuantile(D1, rLen, interval, D3, numQuanR, &index, eventList, &test_Kernel,
              &FLAG_CPU_GPU, &burden, 0);
  CL_MALLOC(&D4, numQuanR * sizeof(int) * 2);
  CUDA_CSSTree *tree;
  gpu_constructCSSTreeImpl(D2, rLen, &tree, &index, eventList, &test_Kernel,
                           &FLAG_CPU_GPU, &burden, 0);

  ///////////cuda_search_index_usingKeys////////////
  int nSearchKeys = numQuanR * 2;
  unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, tree->nDataNodes);
  unsigned int tree_size = tree->nDataNodes + tree->nDirNodes;
  unsigned int bottom_start =
      (uintPower(TREE_FANOUT, lvlDir) - 1) / TREE_NODE_SIZE;

  int numThreadPB = THRD_PER_BLCK_search;
  int numBlock = BLCK_PER_GRID_search;

  unsigned int nKeysPerThread =
      uintCeilingDiv(nSearchKeys, THRD_PER_GRID_search);

  size_t numThreadsPerBlock_x = 256;
  size_t globalWorkingSetSize = 32 * 64;
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  int timer = DLL_genTimer(kid);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_getKernel("gSearchTree_usingKeys_kernel", _HandShakeKernel);
    // Set the Argument values
    cl_int ciErr1 = clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem),
                                   (void *)&tree->data);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int),
                             (void *)&tree->nDataNodes);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem),
                             (void *)&tree->dir);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int),
                             (void *)&tree->nDirNodes);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_int), (void *)&lvlDir);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_mem), (void *)&D3);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_mem), (void *)&D4);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 7, sizeof(cl_int),
                             (void *)&nSearchKeys);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 8, sizeof(cl_int),
                             (void *)&nKeysPerThread);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 9, sizeof(cl_int),
                             (void *)&tree_size);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 10, sizeof(cl_int),
                             (void *)&bottom_start);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("gSearchTree_usingKeys_kernel invocatio overhead, %f\n", t);
#endif
  }
  printf("try to delete tree\n");
  delete tree;
  printf("start test gSearchTree_usingKeys_kernel\n");
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gSearchTree_usingKeys_kernel invocatio overhead in "
           "average in GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, gSearchTree_usingKeys_kernel invocatio overhead in "
           "average in CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void joinMBCount_kernel_handshake(int _HandShakeCPU_GPU,
                                  cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 46;
  printf("Kid%d", kid);
  /*private*/
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  cl_event eventList[2];
  int index = 0;
  cl_kernel test_Kernel;
  int FLAG_CPU_GPU;
  double burden;
  int numResult = 0;
  int interval = SMJ_NUM_THREADS_PER_BLOCK;
  int numQuanR = divRoundUp(rLen, interval);
  CL_MALLOC(&D4, numQuanR * sizeof(int) * 2);
  getQuantile(D1, rLen, interval, D4, numQuanR, &index, eventList, &test_Kernel,
              &FLAG_CPU_GPU, &burden, 0);
  CL_MALLOC(&D5, numQuanR * sizeof(int) * 2);
  CUDA_CSSTree *tree;
  gpu_constructCSSTreeImpl(D1, rLen, &tree, &index, eventList, &test_Kernel,
                           &FLAG_CPU_GPU, &burden, 0);
  cuda_search_index_usingKeys(
      tree->data, tree->nDataNodes, tree->dir, tree->nDirNodes, D4, D5,
      numQuanR * 2, &index, eventList, &test_Kernel, &FLAG_CPU_GPU, &burden, 0);
  int numThreadPerBlock = SMJ_NUM_THREADS_PER_BLOCK;
  int numBlock_X = numQuanR;
  int numBlock_Y = 1;
  if (numBlock_X > NLJ_MAX_NUM_BLOCK_PER_DIM) {
    numBlock_Y = numBlock_X / NLJ_MAX_NUM_BLOCK_PER_DIM;
    if (numBlock_X % NLJ_MAX_NUM_BLOCK_PER_DIM != 0)
      numBlock_Y++;
    numBlock_X = NLJ_MAX_NUM_BLOCK_PER_DIM;
  }
  dim3 threads_NLJ(numThreadPerBlock, 1, 1);
  dim3 grid_NLJ(numBlock_X, numBlock_Y, 1);
  int resultBuf = grid_NLJ.x * grid_NLJ.y * numThreadPerBlock;
  CL_MALLOC(&D6, sizeof(int) * resultBuf);
  CL_MALLOC(&D7, sizeof(int) * resultBuf);
  int h_n = 0;
  int h_sum = 0;
  size_t numThreadsPerBlock_x = 256;
  size_t globalWorkingSetSize = 32 * 64;
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  printf("start test joinMBCount_kernel\n");
  int timer = DLL_genTimer(kid);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    cl_getKernel("joinMBCount_kernel", _HandShakeKernel);
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D1);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_mem), (void *)&D5);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_int),
                             (void *)&numQuanR);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_mem), (void *)&D3);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("joinMBCount_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, joinMBCount_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, joinMBCount_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
  delete tree;
}
void joinMBWrite_kernel_handshake(int _HandShakeCPU_GPU,
                                  cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 47;
  printf("Kid%d", kid);
  /*private*/
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  cl_event eventList[2];
  int index = 0;
  cl_kernel test_Kernel;
  int FLAG_CPU_GPU;
  double burden;
  ////////////////////MJImpl/////////////////////////////
  int numResult = 0;

  int interval = SMJ_NUM_THREADS_PER_BLOCK;
  int numQuanR = divRoundUp(rLen, interval);
  CL_MALLOC(&D3, numQuanR * sizeof(int) * 2);
  getQuantile(D1, rLen, interval, D3, numQuanR, &index, eventList, &test_Kernel,
              &FLAG_CPU_GPU, &burden, 0);
  CL_MALLOC(&D4, numQuanR * sizeof(int) * 2);
  CUDA_CSSTree *tree;
  gpu_constructCSSTreeImpl(D1, rLen, &tree, &index, eventList, &test_Kernel,
                           &FLAG_CPU_GPU, &burden, 0);
  cuda_search_index_usingKeys(
      tree->data, tree->nDataNodes, tree->dir, tree->nDirNodes, D3, D4,
      numQuanR * 2, &index, eventList, &test_Kernel, &FLAG_CPU_GPU, &burden, 0);
  ///////////joinMatchingBlocks////////////////////////////////////////////////////
  int numResults = 0;
  int numThreadPerBlock = SMJ_NUM_THREADS_PER_BLOCK;
  int numBlock_X = numQuanR;
  int numBlock_Y = 1;
  if (numBlock_X > NLJ_MAX_NUM_BLOCK_PER_DIM) {
    numBlock_Y = numBlock_X / NLJ_MAX_NUM_BLOCK_PER_DIM;
    if (numBlock_X % NLJ_MAX_NUM_BLOCK_PER_DIM != 0)
      numBlock_Y++;
    numBlock_X = NLJ_MAX_NUM_BLOCK_PER_DIM;
  }
  dim3 threads_NLJ(numThreadPerBlock, 1, 1);
  dim3 grid_NLJ(numBlock_X, numBlock_Y, 1);
  int resultBuf = grid_NLJ.x * grid_NLJ.y * numThreadPerBlock;
  CL_MALLOC(&D5, sizeof(int) * resultBuf);
  CL_MALLOC(&D6, sizeof(int) * resultBuf);
  int h_n = 0;
  int h_sum = 0;
  joinMBCount(D1, rLen, D2, rLen, D4, numQuanR, D5, grid_NLJ, threads_NLJ,
              &index, eventList, &test_Kernel, &FLAG_CPU_GPU, &burden, 0);
  ScanPara *SP;
  SP = (ScanPara *)malloc(sizeof(ScanPara));
  initScan(resultBuf, SP);
  scanImpl(D5, resultBuf, D6, &index, eventList, &test_Kernel, &FLAG_CPU_GPU,
           &burden, SP, 0);
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  closeScan(SP);
  cl_readbuffer((void *)&h_n, D5, (resultBuf - 1) * sizeof(int), sizeof(int),
                &index, eventList, &FLAG_CPU_GPU, &burden, 0);
  cl_readbuffer((void *)&h_sum, D6, (resultBuf - 1) * sizeof(int), sizeof(int),
                &index, eventList, &FLAG_CPU_GPU, &burden, 0);
  clWaitForEvents(1, &eventList[(index - 1) % 2]);
  numResults = h_n + h_sum;
  CL_MALLOC(&D7, sizeof(Record) * numResults);
  size_t numThreadsPerBlock_x = 256;
  size_t globalWorkingSetSize = 32 * 64;
  printf("start test joinMBWrite_kernel\n");
  int timer = DLL_genTimer(kid);
  for (i = 0; i < Count; i++) {
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 2;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("joinMBWrite_kernel", _HandShakeKernel);
    // Set the Argument values
    // gpuNLJ_int_kernel(cl_mem d_temp, Record *d_R, Record *d_S, int sStart,
    // int rLen, int sLen, int *D5)
    cl_int ciErr1 =
        clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_mem), (void *)&D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_int), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_mem), (void *)&D4);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 5, sizeof(cl_int),
                             (void *)&numQuanR);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 6, sizeof(cl_mem), (void *)&D6);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 7, sizeof(cl_mem), (void *)&D7);
    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("joinMBWrite_kernel invocatio overhead, %f\n", t);
#endif
  }
  printf("try to delete tree\n");
  delete tree;
  printf("start test gSearchTree_usingKeys_kernel\n");
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, joinMBWrite_kernel invocatio overhead in average in "
           "GPU, %lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, joinMBWrite_kernel invocatio overhead in average in "
           "CPU, %lf\n",
           sum, AddCPUBurden[kid]);
  }
}

void build_kernel_handshake(int _HandShakeCPU_GPU,
                            cl_kernel *_HandShakeKernel) {
  double i;
  double sum = 0;
  double scaler = 1000;
  int kid = 49;
  printf("Kid%d", kid);
  cl_uint rHashTableBucketNum = 2 * 1024 * 1024;
  CL_MALLOC(&D2, rLen * sizeof(Record) + rHashTableBucketNum * sizeof(cl_uint));
  // #region Execute on device;
  for (i = 0; i < Count; i++) {
    int timer = DLL_genTimer(kid);
    DLL_getTimer(timer);
    size_t numThreadsPerBlock_x = 256;
    size_t globalWorkingSetSize = 32 * 64;
    cl_getKernel("build_kernel", _HandShakeKernel);

    // configure build _HandShakeKernel
    cl_int ciErr1 = clSetKernelArg((*_HandShakeKernel), 0, sizeof(cl_mem), &D1);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 1, sizeof(cl_mem), &D2);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 2, sizeof(cl_uint), (void *)&rLen);
    ciErr1 |=
        clSetKernelArg((*_HandShakeKernel), 3, sizeof(cl_uint), (void *)&rLen);
    ciErr1 |= clSetKernelArg((*_HandShakeKernel), 4, sizeof(cl_uint),
                             (void *)&rHashTableBucketNum);

    cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                    _HandShakeKernel, _HandShakeCPU_GPU);
    double t = DLL_getTimer(timer);
    sum += t;
#ifdef HandshakeDebug
    printf("build_kernel invocatio overhead, %f\n", t);
#endif
  }
  if (_HandShakeCPU_GPU) {
    AddGPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, build_kernel invocatio overhead in average in GPU, "
           "%lf\n",
           sum, AddGPUBurden[kid]);
  } else {
    AddCPUBurden[kid] = sum / Count * scaler;
    printf("sum is %lf\n, build_kernel invocatio overhead in average in CPU, "
           "%lf\n",
           sum, AddCPUBurden[kid]);
  }
}
void probe_kernel_handshake(int _HandShakeCPU_GPU,
                            cl_kernel *_HandShakeKernel) {
  // CPU-only: the WG-shared WAS post races under GPU SIMD; on PoCL CPU the
  // work-items of a work-group run sequentially, so it is race-free.
  if (_HandShakeCPU_GPU == 1)
    return;
  double scaler = 1000;
  int kid = 50;
  printf("Kid%d probe WAS sweep mode (CPU)\n", kid);
  cl_uint rHashTableBucketNum = 2 * 1024 * 1024;
  size_t wasEntrySize = 4 * sizeof(cl_ulong); // 32 B

  cl_ulong l3CacheSize = 0;
  clGetDeviceInfo(Device[0], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong),
                  &l3CacheSize, NULL);
  const size_t kCostPerEntry = 32 + 2 * 64; // struct + 2 preloaded lines

  // ---- buffers allocated ONCE; data + hash table regenerated each round ----
  int memSizeR = sizeof(Record) * rLen, memSizeS = sizeof(Record) * rLen;
  void *h_R, *h_S;
  HOST_MALLOC(h_R, memSizeR);
  HOST_MALLOC(h_S, memSizeS);
  CL_MALLOC(&D2, memSizeR);
  CL_MALLOC(&D3, memSizeS);
  size_t hashTableBytes =
      rLen * sizeof(Record) + rHashTableBucketNum * sizeof(cl_uint);
  CL_MALLOC(&D4, hashTableBytes);
  cl_uint resultsNum = rLen;
  CL_MALLOC(&D1, sizeof(Record) * resultsNum * 2);

  // ---- WAS sweep (mirrors filter kid=20) ----
  //   wassize <0 noWAS 7-CU control | =0 noWAS 8-CU | >0 WAS 7-CU + helper
  struct SweepCfg {
    const char *label;
    int delta;
    int wassize;
    int mode; // WAS helper variant: bit0=backward, bit1=spin, bit2=pause
  };
  // helper fixed at mode 3 (Backward + spin-loop + no-pause). Per delta:
  // noWAS7c, noWAS8c, then WAS with wassize = num_wg * wpw for wpw doubling
  // (1,2,4,8,...) up to wassize <= 57344.
  static char cfgLabels[160][28];
  struct SweepCfg cfgs[160];
  int nc = 0;
  int pmode = getenv("PROBE_MODE") ? atoi(getenv("PROBE_MODE")) : 3; // WAS helper mode
  const char *onlyW = getenv("PROBE_ONLY_W");
  if (onlyW) { // isolate ONE cfg for perf cache-miss measurement
    int w = atoi(onlyW), dlt = 1792;
    sprintf(cfgLabels[nc], "d%d only w%d", dlt, w);
    cfgs[nc].label = cfgLabels[nc];
    cfgs[nc].delta = dlt;
    cfgs[nc].wassize = w;
    cfgs[nc].mode = (w > 0 ? pmode : 0);
    nc++;
  } else {
    int deltas[] = {1792}; // d=1792 deep-dive
    int nd = (int)(sizeof(deltas) / sizeof(deltas[0]));
    for (int di = 0; di < nd; di++) {
      int dlt = deltas[di];
      int num_wg = dlt / 256;
      sprintf(cfgLabels[nc], "d%d noWAS7c", dlt);
      cfgs[nc].label = cfgLabels[nc];
      cfgs[nc].delta = dlt; cfgs[nc].wassize = -1; cfgs[nc].mode = 0; nc++;
      sprintf(cfgLabels[nc], "d%d noWAS8c", dlt);
      cfgs[nc].label = cfgLabels[nc];
      cfgs[nc].delta = dlt; cfgs[nc].wassize = 0; cfgs[nc].mode = 0; nc++;
      for (int wpw = 4; num_wg * wpw <= 57344; wpw *= 2) { // wassize 28..57344
        sprintf(cfgLabels[nc], "d%d wpw%d w%d", dlt, wpw, num_wg * wpw);
        cfgs[nc].label = cfgLabels[nc];
        cfgs[nc].delta = dlt; cfgs[nc].wassize = num_wg * wpw;
        cfgs[nc].mode = pmode; nc++;
      }
    }
  }
  size_t n_cfgs = (size_t)nc;
  struct SweepResult {
    const char *label;
    int delta, wassize;
    double avg_ms;
    cl_uint hits;
  } results[160];
  int timer = DLL_genTimer(kid);
  cl_uint zero = 0;

  // ---- Round-based design: each round regenerates the dataset + rebuilds the
  //      hash table, then runs every cfg (x PROBE_REPS). Per-cfg stats are
  //      accumulated (Welford) over all rounds x reps. Because every cfg sees
  //      the SAME dataset within a round, we also accumulate the PAIRED diff
  //      (best-WAS - noWAS8c) per delta across rounds (randomized-block test).
  int nRounds = getenv("PROBE_ROUNDS") ? atoi(getenv("PROBE_ROUNDS")) : 1;
  int nReps = getenv("PROBE_REPS") ? atoi(getenv("PROBE_REPS")) : 5;
  if (nRounds < 1) nRounds = 1;
  if (nReps < 1) nReps = 1;

  double cmean[160], cM2[160], cmin[160]; // per-cfg over all rounds x reps (raw)
  int cn[160];
  cl_uint chits[160];
  double pmean[160], pM2[160]; // paired (best-WAS - noWAS8c) per delta over rounds
  int pn[160];
  for (size_t c = 0; c < n_cfgs; c++) {
    cmean[c] = 0; cM2[c] = 0; cmin[c] = 1e30; cn[c] = 0; chits[c] = 0;
    pmean[c] = 0; pM2[c] = 0; pn[c] = 0;
  }

  for (int round = 0; round < nRounds; round++) {
    // fresh dataset (distinct R/S seeds; round 0 == legacy seeds 0/1)
    generateRand((Record *)h_R, TEST_MAX, rLen, 2 * round);
    generateRand((Record *)h_S, TEST_MAX, rLen, 2 * round + 1);
    cl_writebuffer(D2, h_R, memSizeR, 0);
    cl_writebuffer(D3, h_S, memSizeS, 0);
    { // zero the table so build_kernel atomic_inc bucket counts start fresh
      cl_uint fzero = 0;
      clEnqueueFillBuffer(CommandQueue[0], D4, &fzero, sizeof(cl_uint), 0,
                          hashTableBytes, 0, NULL, NULL);
      clFinish(CommandQueue[0]);
    }
    { // rebuild the hash table for this round
      size_t bg = 8192, bl = 256;
      cl_getKernel((char *)"build_kernel", _HandShakeKernel);
      clSetKernelArg(*_HandShakeKernel, 0, sizeof(cl_mem), &D2);
      clSetKernelArg(*_HandShakeKernel, 1, sizeof(cl_mem), &D4);
      clSetKernelArg(*_HandShakeKernel, 2, sizeof(cl_uint), (void *)&rLen);
      clSetKernelArg(*_HandShakeKernel, 3, sizeof(cl_uint), (void *)&rLen);
      clSetKernelArg(*_HandShakeKernel, 4, sizeof(cl_uint),
                     (void *)&rHashTableBucketNum);
      cl_launchKernel(1, &bg, &bl, _HandShakeKernel, 0);
    }
    { // warm-up: 2 untimed noWAS probes absorb first-touch cost after rebuild
      int negw = -1;
      cl_mem nullbuf = NULL;
      size_t wgs = 8192, wls = 256;
      for (int w = 0; w < 2; w++) {
        clEnqueueWriteBuffer(CommandQueue[0], D1, CL_TRUE, 0, sizeof(cl_uint),
                             &zero, 0, NULL, NULL);
        cl_getKernel((char *)"probe_kernel", _HandShakeKernel);
        clSetKernelArg(*_HandShakeKernel, 0, sizeof(cl_mem), &D4);
        clSetKernelArg(*_HandShakeKernel, 1, sizeof(cl_mem), &D3);
        clSetKernelArg(*_HandShakeKernel, 2, sizeof(cl_mem), &D1);
        clSetKernelArg(*_HandShakeKernel, 3, sizeof(cl_uint), (void *)&rLen);
        clSetKernelArg(*_HandShakeKernel, 4, sizeof(cl_uint), (void *)&rLen);
        clSetKernelArg(*_HandShakeKernel, 5, sizeof(cl_uint),
                       (void *)&rHashTableBucketNum);
        clSetKernelArg(*_HandShakeKernel, 6, sizeof(cl_uint), (void *)&resultsNum);
        clSetKernelArg(*_HandShakeKernel, 7, sizeof(cl_mem), (void *)&nullbuf);
        clSetKernelArg(*_HandShakeKernel, 8, sizeof(cl_int), (void *)&negw);
        clSetKernelArg(*_HandShakeKernel, 9, sizeof(cl_mem), (void *)&nullbuf);
        cl_launchKernel(1, &wgs, &wls, _HandShakeKernel, 0);
      }
    }

    double roundTime[160]; // this round's per-cfg avg (ms), for paired diff
    // randomized cfg execution order per round (removes cfg-position bias);
    // seeded by round so it's reproducible. roundTime/stats stay indexed by c.
    size_t order[160];
    for (size_t i = 0; i < n_cfgs; i++) order[i] = i;
    unsigned rng = (unsigned)round * 2654435761u + 12345u;
    for (size_t i = n_cfgs; i > 1; i--) {
      rng = rng * 1103515245u + 12345u;
      size_t j = rng % i;
      size_t tmp = order[i - 1]; order[i - 1] = order[j]; order[j] = tmp;
    }
    for (size_t oi = 0; oi < n_cfgs; oi++) {
      size_t c = order[oi];
      int wassize = cfgs[c].wassize;
      // diagnostic: skip all WAS cfgs so no helper ever launches (clean noWAS)
      if (getenv("PROBE_SKIP_WAS") && wassize > 0) { roundTime[c] = 1e9; continue; }
      size_t globalWorkingSetSize = (size_t)cfgs[c].delta;
      size_t numThreadsPerBlock_x = 256;
      double sum = 0;
      cl_uint hits = 0;
      cl_mem was_buffer = NULL, dummy_buffer = NULL, last_tag_buffer = NULL;

      if (wassize > 0) {
        cl_int err;
        was_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE,
                                    wassize * wasEntrySize, NULL, &err);
        dummy_buffer = clCreateBuffer(
            Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 64, NULL, &err);
        void *dmap = clEnqueueMapBuffer(CommandQueue[0], dummy_buffer, CL_TRUE,
                                        CL_MAP_WRITE, 0, 64, 0, NULL, NULL, &err);
        if (dmap) {
          memset(dmap, 0, 64);
          clEnqueueUnmapMemObject(CommandQueue[0], dummy_buffer, dmap, 0, NULL, NULL);
          clFinish(CommandQueue[0]);
        }
        cl_ulong *wmap = (cl_ulong *)clEnqueueMapBuffer(
            CommandQueue[0], was_buffer, CL_TRUE, CL_MAP_WRITE, 0,
            wassize * wasEntrySize, 0, NULL, NULL, &err);
        if (wmap) {
          void *dmap2 = clEnqueueMapBuffer(CommandQueue[0], dummy_buffer, CL_TRUE,
                                           CL_MAP_READ, 0, 8, 0, NULL, NULL, &err);
          cl_ulong dummyVal = (cl_ulong)(uintptr_t)dmap2; // empty-slot sentinel
          if (dmap2)
            clEnqueueUnmapMemObject(CommandQueue[0], dummy_buffer, dmap2, 0, NULL, NULL);
          for (int j = 0; j < wassize; j++) {
            wmap[j * 4 + 0] = dummyVal;   // p1 = dummy (empty)
            wmap[j * 4 + 1] = dummyVal;   // p2
            wmap[j * 4 + 2] = (cl_ulong)-1; // state1 (key) — overwritten on post
            wmap[j * 4 + 3] = (cl_ulong)-1; // state2 (val)
          }
          clEnqueueUnmapMemObject(CommandQueue[0], was_buffer, wmap, 0, NULL, NULL);
          clFinish(CommandQueue[0]);
        }
        last_tag_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE,
                                         wassize * sizeof(cl_ulong), NULL, &err);
        cl_ulong *lmap = (cl_ulong *)clEnqueueMapBuffer(
            CommandQueue[0], last_tag_buffer, CL_TRUE, CL_MAP_WRITE, 0,
            wassize * sizeof(cl_ulong), 0, NULL, NULL, &err);
        if (lmap) {
          memset(lmap, 0, wassize * sizeof(cl_ulong));
          clEnqueueUnmapMemObject(CommandQueue[0], last_tag_buffer, lmap, 0, NULL, NULL);
          clFinish(CommandQueue[0]);
        }
      }

      cl_kernel wasKernel = NULL;
      if (wassize > 0 && was_buffer && dummy_buffer && last_tag_buffer &&
          PrefetchCommandQueue && !getenv("PROBE_NO_HELPER")) {
        cl_int kerr;
        wasKernel = clCreateKernel(Program, "WAS_kernel", &kerr);
        if (kerr == CL_SUCCESS) {
          clSetKernelArg(wasKernel, 0, sizeof(cl_mem), (void *)&was_buffer);
          clSetKernelArg(wasKernel, 1, sizeof(cl_int), (void *)&wassize);
          clSetKernelArg(wasKernel, 2, sizeof(cl_mem), (void *)&dummy_buffer);
          clSetKernelArg(wasKernel, 3, sizeof(cl_mem), (void *)&last_tag_buffer);
          cl_int wmode = cfgs[c].mode;
          clSetKernelArg(wasKernel, 4, sizeof(cl_int), (void *)&wmode);
          size_t wg = 1, wl = 1;
          clEnqueueNDRangeKernel(PrefetchCommandQueue, wasKernel, 1, NULL, &wg, &wl,
                                 0, NULL, NULL);
          clFlush(PrefetchCommandQueue);
        }
      }

      cl_command_queue savedQ = CommandQueue[0];
      if (wassize == 0 && FullCPUCommandQueue)
        CommandQueue[0] = FullCPUCommandQueue;

      for (int it = 0; it < nReps; it++) {
        // reset the match counter (untimed) so it never overflows matchedTable.
        clEnqueueWriteBuffer(CommandQueue[0], D1, CL_TRUE, 0, sizeof(cl_uint),
                             &zero, 0, NULL, NULL);
        DLL_getTimer(timer);
        cl_getKernel((char *)"probe_kernel", _HandShakeKernel);
        clSetKernelArg(*_HandShakeKernel, 0, sizeof(cl_mem), &D4); // rHashTable
        clSetKernelArg(*_HandShakeKernel, 1, sizeof(cl_mem), &D3); // sTable
        clSetKernelArg(*_HandShakeKernel, 2, sizeof(cl_mem), &D1); // matchedTable
        clSetKernelArg(*_HandShakeKernel, 3, sizeof(cl_uint), (void *)&rLen);
        clSetKernelArg(*_HandShakeKernel, 4, sizeof(cl_uint), (void *)&rLen);
        clSetKernelArg(*_HandShakeKernel, 5, sizeof(cl_uint),
                       (void *)&rHashTableBucketNum);
        clSetKernelArg(*_HandShakeKernel, 6, sizeof(cl_uint), (void *)&resultsNum);
        clSetKernelArg(*_HandShakeKernel, 7, sizeof(cl_mem), (void *)&was_buffer);
        clSetKernelArg(*_HandShakeKernel, 8, sizeof(cl_int), (void *)&wassize);
        clSetKernelArg(*_HandShakeKernel, 9, sizeof(cl_mem), (void *)&dummy_buffer);
        cl_launchKernel(1, &globalWorkingSetSize, &numThreadsPerBlock_x,
                        _HandShakeKernel, 0);
        double t = DLL_getTimer(timer);
        sum += t;
        cn[c]++; // Welford over all rounds x reps
        double wd = t - cmean[c];
        cmean[c] += wd / cn[c];
        cM2[c] += wd * (t - cmean[c]);
        if (t < cmin[c]) cmin[c] = t;
      }
      CommandQueue[0] = savedQ;

      if (wasKernel && dummy_buffer) {
        cl_int err;
        cl_uint *dmap = (cl_uint *)clEnqueueMapBuffer(
            CommandQueue[0], dummy_buffer, CL_TRUE, CL_MAP_WRITE, 0, 64, 0, NULL,
            NULL, &err);
        if (dmap) {
          dmap[1] = 1; // stop the helper
          clEnqueueUnmapMemObject(CommandQueue[0], dummy_buffer, dmap, 0, NULL, NULL);
          clFinish(CommandQueue[0]);
        }
        if (PrefetchCommandQueue)
          clFinish(PrefetchCommandQueue);
        cl_uint *cp = (cl_uint *)clEnqueueMapBuffer(CommandQueue[0], dummy_buffer,
                                                    CL_TRUE, CL_MAP_READ, 0,
                                                    sizeof(cl_uint), 0, NULL, NULL, &err);
        if (cp) {
          hits = *cp;
          clEnqueueUnmapMemObject(CommandQueue[0], dummy_buffer, cp, 0, NULL, NULL);
          clFinish(CommandQueue[0]);
        }
        clReleaseKernel(wasKernel);
      }
      if (was_buffer)
        clReleaseMemObject(was_buffer);
      if (dummy_buffer)
        clReleaseMemObject(dummy_buffer);
      if (last_tag_buffer)
        clReleaseMemObject(last_tag_buffer);

      chits[c] += hits;
      roundTime[c] = (sum / nReps) * scaler; // this round's avg for this cfg
    } // cfg loop

    if (getenv("PROBE_RLOG"))
      fprintf(stderr,
              "[ROUND %2d] d1792 7c=%.1f 8c=%.1f wpw64=%.1f | "
              "d114688 8c=%.1f wpw4=%.1f\n",
              round, roundTime[0], roundTime[1], roundTime[5], roundTime[37],
              roundTime[39]);

    // paired diff for THIS round: for each noWAS8c anchor, best-WAS(same d) - it
    for (size_t c = 0; c < n_cfgs; c++) {
      if (cfgs[c].wassize != 0)
        continue;
      double bestWAS = 1e30;
      for (size_t d = 0; d < n_cfgs; d++)
        if (cfgs[d].wassize > 0 && cfgs[d].delta == cfgs[c].delta &&
            roundTime[d] < bestWAS)
          bestWAS = roundTime[d];
      double diff = bestWAS - roundTime[c]; // >0 => WAS slower than 8c
      pn[c]++;
      double pd = diff - pmean[c];
      pmean[c] += pd / pn[c];
      pM2[c] += pd * (diff - pmean[c]);
    }
  } // round loop

  // ---- per-cfg summary over all rounds x reps ----
  for (size_t c = 0; c < n_cfgs; c++) {
    int wassize = cfgs[c].wassize;
    results[c].label = cfgs[c].label;
    results[c].delta = cfgs[c].delta;
    results[c].wassize = wassize;
    results[c].avg_ms = cmean[c] * scaler;
    results[c].hits = chits[c];
    double std_ms = (cn[c] > 1 ? sqrt(cM2[c] / (cn[c] - 1)) : 0.0) * scaler;
    double min_ms = cmin[c] * scaler;
    double cv_pct =
        (results[c].avg_ms > 0 ? 100.0 * std_ms / results[c].avg_ms : 0.0);
    double l3_pct = (l3CacheSize > 0 && wassize > 0)
                        ? 100.0 * (double)wassize * kCostPerEntry / (double)l3CacheSize
                        : 0.0;
    printf("[KID50-PROBE %2zu/%2zu] %-22s d=%6d w=%6d L3=%5.1f%% "
           "avg=%8.2f ±%6.2f cv=%4.1f%% min=%8.2f n=%d hits=%u%s\n",
           c + 1, n_cfgs, cfgs[c].label, cfgs[c].delta, wassize, l3_pct,
           results[c].avg_ms, std_ms, cv_pct, min_ms, cn[c], chits[c],
           l3_pct > 25.0 ? " §4.2>L3/4" : "");
    fflush(stdout);
  }

  // ---- paired comparison: best-WAS - noWAS8c per delta (rounds as blocks) ----
  printf("[KID50-PROBE PAIRED] best-WAS minus noWAS8c (rounds=%d, +ms => WAS slower)\n",
         nRounds);
  for (size_t c = 0; c < n_cfgs; c++) {
    if (cfgs[c].wassize != 0 || pn[c] < 1)
      continue;
    double pstd = (pn[c] > 1 ? sqrt(pM2[c] / (pn[c] - 1)) : 0.0); // already ms
    double pm = pmean[c];                                          // already ms
    double se = (pn[c] > 0 ? pstd / sqrt((double)pn[c]) : 0.0);
    double tval = (se > 0 ? pm / se : 0.0);
    printf("  d=%6d  WAS-8c=%+7.2fms ±%6.2f  t=%+6.2f  %s\n", cfgs[c].delta, pm,
           pstd, tval, pm > 0 ? "8c faster (WAS loses)" : "WAS faster");
  }
  fflush(stdout);

  size_t best = 0;
  for (size_t c = 1; c < n_cfgs; c++)
    if (results[c].avg_ms < results[best].avg_ms)
      best = c;
  printf("[KID50-PROBE BEST] %s avg=%.2fms\n", results[best].label,
         results[best].avg_ms);
  AddCPUBurden[kid] = results[best].avg_ms;
}