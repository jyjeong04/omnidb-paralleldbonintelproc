#include "common.h"
#include "KernelScheduler.h"
#include "OpenCL_DLL.h"
#include "scheduler.h"

// OpenCL Vars---------0 for CPU, 1 for GPU
extern cl_context Context;                // OpenCL context
extern cl_program Program;                // OpenCL program
extern cl_command_queue CommandQueue[2];  // OpenCL command que
extern cl_platform_id Platform[2];        // OpenCL platform
extern cl_device_id Device[2];            // OpenCL device
extern cl_ulong totalLocalMemory[2];      /**< Max local memory allowed */
extern cl_ulong totalGlobalMemory[2];     /**< Max global memory allowed */
extern cl_ulong usedtotalGlobalMemory[2]; /**< Max global memory used */
extern pthread_mutex_t CPUBurdenCS;
extern pthread_mutex_t GPUBurdenCS;
extern cl_device_id allDevices[10];
extern double GPUBurden;
extern double CPUBurden;
static int totalNumDevice = 0;
extern double LoGPUBurden;
extern double LoCPUBurden;
extern double UpGPUBurden;
extern double UpCPUBurden;
cl_kernel Kernel[2]; // OpenCL kernel---------------->should been cancelled
                     // after all method update.
static int TAG_NO;

// ---- CPU-assisted prefetching via device fission (PE config) ----
cl_device_id PrefetchSubDevice = NULL;
cl_device_id MainCPUSubDevice = NULL;
cl_command_queue PrefetchCommandQueue = NULL;
cl_command_queue FullCPUCommandQueue = NULL; // 8-CU queue on parent Device[0] (no-helper phases)
int g_prefetchEnabled = 0;
static int g_prefetchWASSize = 0;


cl_device_id    DecomSubDevice = NULL;
cl_device_id    NonDecomSubDevice = NULL;
cl_command_queue DecomCommandQueue = NULL;
cl_command_queue NonDecomCommandQueue = NULL;
int g_decomEnabled = 0;
int g_decomCUs = 0;
int g_nonDecomCUs = 0;
extern cl_ulong totalGlobalMemory[2];     /**< Max global memory allowed */
extern cl_ulong usedtotalGlobalMemory[2]; /**< Max global memory used */
#define APU
void bufferchecking(cl_mem R_in, size_t size) {
  printf("checking size is %d\n", size);
  Record *R_out;
  R_out = (Record *)malloc(sizeof(Record) * size);
  CopyGPUToCPU(R_in, (void *)R_out, size);
  printf("checking: R_out [0].x:%d R_out[0].y:%d, size is %d\n", R_out->x,
         R_out->y, size);
  free(R_out);
}

void cl_init(cl_device_type TYPE) {
  int CPU_GPU;
  if (TYPE == CL_DEVICE_TYPE_CPU)
    CPU_GPU = 0;
  else
    CPU_GPU = 1;
  cl_int ciErr1;
  // Get an OpenCL platform
  cl_uint numPlatform;
  cl_platform_id tempPlatform[10];
  ciErr1 = clGetPlatformIDs(0, NULL, &numPlatform);
  ciErr1 = clGetPlatformIDs(numPlatform, tempPlatform, NULL);
  cl_uint p = 0;
  char buffer[100];
  size_t length = 0;
  for (p = 0; p < numPlatform; p++) {
    clGetPlatformInfo(tempPlatform[p], CL_PLATFORM_VENDOR, 100, buffer,
                      &length);
    // Look for PoCL (Portable Computing Language) which has both CPU and GPU
    if (strstr(buffer, "Portable Computing Language") != NULL ||
        strstr(buffer, "pocl") != NULL || strstr(buffer, "POCL") != NULL) {
      Platform[CPU_GPU] = tempPlatform[p];
      break;
    }
    // Fallback to Intel OpenCL if PoCL not found
    if (strstr(buffer, "Intel") != NULL) {
      Platform[CPU_GPU] = tempPlatform[p];
    }
  }

  clGetPlatformInfo(Platform[CPU_GPU], CL_PLATFORM_VENDOR, 100, buffer,
                    &length);
  printf("%s, length, %d\n", buffer, length);

  if (ciErr1 != CL_SUCCESS) {
    printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__,
           __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  // Get the number of devices
  cl_uint uiNumDevices;
  ciErr1 = clGetDeviceIDs(Platform[CPU_GPU], TYPE, 0, NULL, &uiNumDevices);

  if (ciErr1 != CL_SUCCESS) {
    printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__,
           __FILE__);
    cl_clean(EXIT_FAILURE);
  } else
    printf("found: %d device\n", uiNumDevices);
  totalNumDevice = uiNumDevices;
  // get all the device
  ciErr1 =
      clGetDeviceIDs(Platform[CPU_GPU], TYPE, totalNumDevice, allDevices, NULL);

  if (ciErr1 != CL_SUCCESS) {
    printf("Error in get all devices, Line %u in file %s !!!\n\n", __LINE__,
           __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  // use the last one. =----->why??
  printf("all devices: ");
  for (int i = 0; i < totalNumDevice; i++) {
    printf("D%d\t", allDevices[i]);
  }
  printf("\n");
  Device[CPU_GPU] = allDevices[totalNumDevice - 1];
  printf("Device[%d], %d\n", CPU_GPU, Device[CPU_GPU]);

  ciErr1 = clGetDeviceInfo(Device[CPU_GPU], CL_DEVICE_LOCAL_MEM_SIZE,
                           sizeof(cl_ulong), (void *)&totalLocalMemory[CPU_GPU],
                           NULL);
  printf("totalLocalMemory, %u\n", totalLocalMemory[CPU_GPU]);
  ciErr1 = clGetDeviceInfo(Device[CPU_GPU], CL_DEVICE_GLOBAL_MEM_SIZE,
                           sizeof(cl_ulong),
                           (void *)&totalGlobalMemory[CPU_GPU], NULL);
  printf("totalGlobalMemroy, %lu\n", totalGlobalMemory[CPU_GPU]);

  char *deviceName = NULL;
  size_t size = 0;
  ciErr1 = clGetDeviceInfo(Device[CPU_GPU], CL_DEVICE_NAME, 0, NULL, &size);
  deviceName = (char *)malloc(size);
  ciErr1 =
      clGetDeviceInfo(Device[CPU_GPU], CL_DEVICE_NAME, size, deviceName, &size);
  printf("deviceName is %s \n", deviceName);
}
void cl_init_common() {
  cl_int ciErr1;
  // Create the context
  Context = clCreateContext(0, 2, Device, NULL, NULL, &ciErr1);
  // shrLog("clCreateContext...\n");
  if (ciErr1 != CL_SUCCESS) {
    printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__,
           __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  int CPU_GPU;
  for (CPU_GPU = 0; CPU_GPU < 2; CPU_GPU++) {
    // Create a command-queue
    CommandQueue[CPU_GPU] =
        clCreateCommandQueue(Context, Device[CPU_GPU], 0, NULL);
    // shrLog("clCreateCommandQueue...\n");
    if (ciErr1 != CL_SUCCESS) {
      printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n",
             __LINE__, __FILE__);
      cl_clean(EXIT_FAILURE);
    }
  }
}
int convertToString(const char *filename, std::string &s) {
  size_t size;
  char *str;

  std::fstream f(filename, (std::fstream::in | std::fstream::binary));

  if (f.is_open()) {
    size_t fileSize;
    f.seekg(0, std::fstream::end);
    size = fileSize = (size_t)f.tellg();
    f.seekg(0, std::fstream::beg);

    str = new char[size + 1];
    if (!str) {
      f.close();
      return NULL;
    }

    f.read(str, fileSize);
    f.close();
    str[size] = '\0';

    s = str;
    delete[] str;
    return 0;
  }
  printf("Error: Failed to open file %s\n", filename);
  return 1;
}
char *append(const char *orig, char c) {
  size_t sz = strlen(orig);
  char *str = (char *)malloc(sz + 2);
  strcpy(str, orig);
  str[sz] = c;
  str[sz + 1] = '\0';
  return str;
}
char *append(char *orig, char *st) {
  size_t sz = strlen(st);
  int i;
  for (i = 0; i < sz; i++)
    orig = append(orig, st[i]);
  return orig;
}
void cl_prepareProgram(char *cSourceFile, char *dir) {
  cSourceFile = append(dir, cSourceFile);
  std::cout << "Read:" << cSourceFile << "\n";
  cl_int ciErr1;
  // convert kernel file into string
  std::string sourceStr;
  convertToString(cSourceFile, sourceStr);
  const char *source = sourceStr.c_str();
  size_t sourceSize[] = {strlen(source)};

  Program = ::clCreateProgramWithSource(Context, 1, &source, NULL, &ciErr1);
  // Create the program

  printf("clCreateProgramWithSource...\n");
  if (ciErr1 != CL_SUCCESS) {
    printf("Error %d in clCreateProgramWithSource, Line %u in file %s !!!\n\n",
           ciErr1, __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }

// Build the program with 'mad' Optimization option====>>>> what is "mad"
#ifdef MAC
  char *flags = "-cl-fast-relaxed-math -DMAC -D WORKGROUP_SIZE=256";
#else
  char *flags = "-cl-fast-relaxed-math";
#endif
  ciErr1 = clBuildProgram(Program, 2, Device, flags, NULL, NULL);
  printf("clBuildProgram...\n");
  if (ciErr1 == CL_BUILD_PROGRAM_FAILURE) {
    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(Program, Device[0], CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);

    // Allocate memory for the log
    char *log = (char *)malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(Program, Device[0], CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);

    // Print the log
    printf("%s\n", log);
  } else {
    printf("compilation success!\n");
  }
}
void cl_getKernel(char *kernelName, int CPU_GPU) {};
void cl_getKernel(char *kernelName, cl_kernel *Kernel) {
  cl_int ciErr1;
  (*Kernel) = clCreateKernel(Program, kernelName, &ciErr1);
  // shrLog("clCreateKernel (VectorAdd)...\n");
  if (ciErr1 != CL_SUCCESS) {
    printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__,
           __FILE__);
    cl_clean(EXIT_FAILURE);
  }
}
void cl_getKernelByKernelFunction(char *kernelName, cl_kernel *Kernel) {
  cl_int ciErr1;
  (*Kernel) = clCreateKernel(Program, kernelName, &ciErr1);
  // shrLog("clCreateKernel (VectorAdd)...\n");
  if (ciErr1 != CL_SUCCESS) {
    printf("Error % in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1,
           __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
}
void CL_CREATE(cl_mem *mem, cl_int size) { CL_MALLOC(mem, size); }
void CL_DESTORY(cl_mem *mem) {
  if (mem)
    clReleaseMemObject(*mem);
}
cl_int cl_malloc(cl_mem *mem, cl_mem_flags flag, cl_int size) {
  cl_int ciErr1;
#ifdef APU
  *mem = clCreateBuffer(Context, CL_MEM_ALLOC_HOST_PTR | flag, size, NULL,
                        &ciErr1);
#else
  *mem = clCreateBuffer(Context, flag, size, NULL, &ciErr1);
#endif
  return ciErr1;
}
void cl_readbuffer(void *to, cl_mem from, size_t size, int *index,
                   cl_event *eventList, int *Flag_CPU_GPU, double *burden,
                   int _CPU_GPU) {
  cl_readbuffer(to, from, 0, size, index, eventList, Flag_CPU_GPU, burden,
                _CPU_GPU);
}
void cl_readbuffer(void *to, cl_mem from, size_t offset, size_t size,
                   int *index, cl_event *eventList, int *Flag_CPU_GPU,
                   double *burden, int _CPU_GPU) {
  int preFlag = (*Flag_CPU_GPU);
  double preBurden = (*burden);
  int CPU_GPU = 0;
  CPU_GPU = cl_readbufferscheduler(size, Flag_CPU_GPU, burden, _CPU_GPU);
  (*Flag_CPU_GPU) = CPU_GPU;
  cl_int ciErr1;
  if (*index != 0) {
    ciErr1 = clEnqueueReadBuffer(CommandQueue[CPU_GPU], from, CL_TRUE, offset,
                                 size, to, 1, &eventList[(*index - 1) % 2],
                                 &eventList[(*index) % 2]);
    deschedule(preFlag, preBurden);

  } else
    ciErr1 = clEnqueueReadBuffer(CommandQueue[CPU_GPU], from, CL_TRUE, offset,
                                 size, to, 0, NULL, &eventList[*index]);
  (*index)++;

  // clEnqueueWriteBuffer(CommandQueue[CPU_GPU], to, CL_FALSE, 0, size, from, 0,
  // NULL, NULL);
  if (ciErr1 != CL_SUCCESS) {
    printf("Error %d in //cl_readbuffer, Line %u in file %s !!!\n\n", ciErr1,
           __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  clFlush(CommandQueue[CPU_GPU]);
}

void cl_writebuffer(cl_mem to, void *from, size_t size, int *index,
                    cl_event *eventList, int *Flag_CPU_GPU, double *burden,
                    int _CPU_GPU) {
  int preFlag = (*Flag_CPU_GPU);
  double preBurden = (*burden);
  int CPU_GPU = 0;
  CPU_GPU = cl_writebufferscheduler(size, Flag_CPU_GPU, burden, _CPU_GPU);
  cl_int ciErr1;
  (*Flag_CPU_GPU) = CPU_GPU;
  if (*index != 0) {
    ciErr1 = clEnqueueWriteBuffer(CommandQueue[CPU_GPU], to, CL_FALSE, 0, size,
                                  from, 1, &eventList[((*index) - 1) % 2],
                                  &eventList[(*index) % 2]);
    deschedule(preFlag, preBurden);
  } else
    ciErr1 = clEnqueueWriteBuffer(CommandQueue[CPU_GPU], to, CL_FALSE, 0, size,
                                  from, 0, NULL, &eventList[*index]);
  (*index)++;

  if (ciErr1 != CL_SUCCESS) {
    printf("ciErr1 is %d, Error in clEnqueueWriteBuffer, Line %u in file %s "
           "!!!\n\n",
           ciErr1, __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  clFlush(CommandQueue[CPU_GPU]);
}
void cl_copyBuffer(cl_mem dest, cl_mem src, size_t size, int *index,
                   cl_event *eventList, int *Flag_CPU_GPU, double *burden,
                   int _CPU_GPU) {
  int preFlag = (*Flag_CPU_GPU);
  double preBurden = (*burden);
  int CPU_GPU = 0;
  CPU_GPU = cl_copyBufferscheduler(size, Flag_CPU_GPU, burden, _CPU_GPU);
  cl_int ciErr1;
  (*Flag_CPU_GPU) = CPU_GPU;
  if (*index != 0) {
    ciErr1 = clEnqueueCopyBuffer(CommandQueue[CPU_GPU], src, dest, 0, 0, size,
                                 1, &eventList[((*index) - 1) % 2],
                                 &eventList[(*index) % 2]);
    deschedule(preFlag, preBurden);
  } else
    ciErr1 = clEnqueueCopyBuffer(CommandQueue[CPU_GPU], src, dest, 0, 0, size,
                                 0, NULL, &eventList[*index]);

  (*index)++;
  if (ciErr1 != CL_SUCCESS) {
    printf(" Error %d, in //cl_copyBuffer, Line %u in file %s !!!\n\n", ciErr1,
           __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  clFlush(CommandQueue[CPU_GPU]);
}
void cl_copyBuffer(cl_mem dest, int destOffset, cl_mem src, size_t size,
                   int *index, cl_event *eventList, int *Flag_CPU_GPU,
                   double *burden, int _CPU_GPU) {
  int preFlag = (*Flag_CPU_GPU);
  double preBurden = (*burden);
  int CPU_GPU = 0;
  CPU_GPU = cl_copyBufferscheduler(size, Flag_CPU_GPU, burden, _CPU_GPU);
  CPU_GPU = _CPU_GPU;
  cl_int ciErr1;
  (*Flag_CPU_GPU) = CPU_GPU;
  if (*index != 0) {
    ciErr1 = clEnqueueCopyBuffer(
        CommandQueue[CPU_GPU], src, dest, 0, destOffset, size, 1,
        &eventList[((*index) - 1) % 2], &eventList[(*index) % 2]);
    deschedule(preFlag, preBurden);
  } else
    ciErr1 = clEnqueueCopyBuffer(CommandQueue[CPU_GPU], src, dest, 0,
                                 destOffset, size, 0, NULL, &eventList[*index]);

  (*index)++;
  // clEnqueueWriteBuffer(CommandQueue[CPU_GPU], to, CL_FALSE, 0, size, from, 0,
  // NULL, NULL);
  if (ciErr1 != CL_SUCCESS) {
    printf("Error %d in cl_copyBuffer, Line %u in file %s !!!\n\n", ciErr1,
           __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  clFlush(CommandQueue[CPU_GPU]);
}
void cl_copyBuffer(cl_mem dest, int destOffset, cl_mem src, int srcOffset,
                   size_t size, int *index, cl_event *eventList,
                   int *Flag_CPU_GPU, double *burden, int _CPU_GPU) {
  int preFlag = (*Flag_CPU_GPU);
  double preBurden = (*burden);
  int CPU_GPU = 0;
  CPU_GPU = cl_copyBufferscheduler(size, Flag_CPU_GPU, burden, _CPU_GPU);
  cl_int ciErr1;
  (*Flag_CPU_GPU) = CPU_GPU;
  if (*index != 0) {
    ciErr1 = clEnqueueCopyBuffer(
        CommandQueue[CPU_GPU], src, dest, srcOffset, destOffset, size, 1,
        &eventList[((*index) - 1) % 2], &eventList[(*index) % 2]);
    deschedule(preFlag, preBurden);
  } else
    ciErr1 = clEnqueueCopyBuffer(CommandQueue[CPU_GPU], src, dest, srcOffset,
                                 destOffset, size, 0, NULL, &eventList[*index]);

  (*index)++;
  // clEnqueueWriteBuffer(CommandQueue[CPU_GPU], to, CL_FALSE, 0, size, from, 0,
  // NULL, NULL);
  if (ciErr1 != CL_SUCCESS) {
    printf("Error %d in cl_copyBuffer, Line %u in file %s !!!\n\n", ciErr1,
           __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  clFlush(CommandQueue[CPU_GPU]);
}
void cl_clean(int iExitCode) {
  if (g_decomEnabled) {
    cl_cleanup_decom();
  }
  if (g_prefetchEnabled) {
    cl_cleanup_prefetch();
  }
  // Cleanup allocated objects
  printf("Starting Cleanup...\n\n");
  int CPU_GPU;
  for (CPU_GPU = 0; CPU_GPU < 2; CPU_GPU++) {
    if (CommandQueue[CPU_GPU])
      clReleaseCommandQueue(CommandQueue[CPU_GPU]);
  }
  if (Program)
    clReleaseProgram(Program);
  if (Context)
    clReleaseContext(Context);
  exit(0);
}
// new

int floorPow2(int n) {
  int exp;
  frexp((float)n, &exp);
  return 1 << (exp - 1);
}
/*original*/
void cl_readbuffer(void *to, cl_mem from, size_t size, int CPU_GPU) {
  cl_int ciErr1;
  // bufferchecking(from,size);
  ciErr1 = clEnqueueReadBuffer(CommandQueue[CPU_GPU], from, CL_TRUE, NULL, size,
                               to, 0, NULL, NULL);
  // clEnqueueWriteBuffer(CommandQueue[CPU_GPU], to, CL_FALSE, 0, size, from, 0,
  // NULL, NULL);
  if (ciErr1 != CL_SUCCESS) {
    printf("Error %d in cl_readbuffer, Line %u in file %s !!!\n\n", ciErr1,
           __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  clFinish(CommandQueue[CPU_GPU]);
}
void CopyCPUToGPU(cl_mem to, void *from, size_t size) {
  cl_writebuffer(to, from, size, 0);
}
void CopyGPUToGPU(cl_mem from, cl_mem to, size_t size) {
  cl_copyBuffer(to, from, size, 0);
}

void CopyGPUToCPU(cl_mem from, void *to, size_t size) {
  cl_readbuffer(to, from, size, 0);
}
void cl_writebuffer(cl_mem to, void *from, size_t size, int CPU_GPU) {
  cl_int ciErr1;
  ciErr1 = clEnqueueWriteBuffer(CommandQueue[CPU_GPU], to, CL_FALSE, 0, size,
                                from, 0, NULL, NULL);
  if (ciErr1 != CL_SUCCESS) {
    printf("Error  %d in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n",
           ciErr1, __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  clFinish(CommandQueue[CPU_GPU]);
}
void cl_copyBuffer(cl_mem dest, cl_mem src, size_t size, int CPU_GPU) {
  cl_int ciErr1;
  ciErr1 = clEnqueueCopyBuffer(CommandQueue[CPU_GPU], src, dest, 0, 0, size, 0,
                               NULL, NULL);
  if (ciErr1 != CL_SUCCESS) {
    printf("Error %d in cl_copyBuffer, Line %u in file %s !!!\n\n", ciErr1,
           __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  clFinish(CommandQueue[CPU_GPU]);
}
void cl_launchKernel(cl_uint work_dim, const size_t *groups, size_t *threads,
                     cl_kernel *Kernel, int CPU_GPU) {
  if (work_dim == 1) {
    // printf("dim 1: G%d T%d\n", groups[0], threads[0]);
    if (CPU_GPU && threads[0] > 256) {
      threads[0] = 256;
      // printf("!!!exceed GPU limit:max work item is 256!");
    }
  } else if (work_dim == 2)
    printf("dim 2: G%d, %d T%d,%d\n", groups[0], groups[1], threads[0],
           threads[1]);

  cl_int ciErr1 =
      clEnqueueNDRangeKernel(CommandQueue[CPU_GPU], (*Kernel), work_dim, NULL,
                             groups, threads, 0, NULL, NULL);
  if (ciErr1 != CL_SUCCESS) {
    printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",
           ciErr1, __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
  ciErr1 = clFinish(CommandQueue[CPU_GPU]);
  if (ciErr1 != CL_SUCCESS) {
    printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",
           ciErr1, __LINE__, __FILE__);
    cl_clean(EXIT_FAILURE);
  }
}
void wait(int index, cl_event *eventList) {
  printf("index of %d Going to wait!\n", index);
  cl_int err = clWaitForEvents(1, &eventList[(index - 1) % 2]);
  printf("index of %d Finish wait! err is %d\n,", index, err);
}

///////////////////////////////////////////////////////////////////////////////
// Prefetching via Device Fission
// Per paper: "We fix prefetching on one CPU CU" — use clCreateSubDevices
// to partition the CPU device into a 1-CU prefetch helper and the remaining
// CUs for main query processing.
///////////////////////////////////////////////////////////////////////////////
void cl_init_prefetch() {
  cl_int err;

  // 1. Query the number of compute units on the CPU device
  cl_uint maxCUs = 0;
  err = clGetDeviceInfo(Device[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                        &maxCUs, NULL);
  if (err != CL_SUCCESS) {
    printf("[Prefetch] Error %d querying CL_DEVICE_MAX_COMPUTE_UNITS\n", err);
    cl_clean(err);
  }
  printf("[Prefetch] CPU Device has %u compute units\n", maxCUs);

  if (maxCUs < 2) {
    printf("[Prefetch] WARNING: CPU has only %u CU(s), cannot split for "
           "prefetching. Prefetching disabled.\n",
           maxCUs);
    g_prefetchEnabled = 0;
    cl_clean(CL_SUCCESS);
  }

  // 2. Check if device supports partitioning
  cl_uint maxSubDevices = 0;
  err = clGetDeviceInfo(Device[0], CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                        sizeof(cl_uint), &maxSubDevices, NULL);
  if (err != CL_SUCCESS || maxSubDevices < 2) {
    printf("[Prefetch] WARNING: Device does not support partition "
           "(maxSubDevices=%u, err=%d). Prefetching disabled.\n",
           maxSubDevices, err);
    g_prefetchEnabled = 0;
    cl_clean(CL_SUCCESS);
  }
  printf("[Prefetch] Device supports up to %u sub-devices\n", maxSubDevices);

  // 3. Partition the CPU device: 1 CU for prefetch, (maxCUs-1) for main work
  //    Using CL_DEVICE_PARTITION_BY_COUNTS to specify exact CU counts.
  cl_uint prefetchCUs = 1;
  cl_uint mainCUs = maxCUs - 1;

  cl_device_partition_property partitionProps[] = {
      CL_DEVICE_PARTITION_BY_COUNTS, (cl_device_partition_property)prefetchCUs,
      (cl_device_partition_property)mainCUs,
      CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0};

  cl_device_id subDevices[2];
  cl_uint numSubDevicesRet = 0;
  err = clCreateSubDevices(Device[0], partitionProps, 2, subDevices,
                           &numSubDevicesRet);
  if (err != CL_SUCCESS) {
    printf("[Prefetch] Error %d in clCreateSubDevices. "
           "Prefetching disabled.\n",
           err);
    g_prefetchEnabled = 0;
    cl_clean(err);
  }
  printf("[Prefetch] Created %u sub-devices (prefetch: 1 CU, main: %u CUs)\n",
         numSubDevicesRet, mainCUs);

  PrefetchSubDevice = subDevices[0]; // 1 CU — dedicated prefetch helper
  MainCPUSubDevice = subDevices[1];  // (maxCUs-1) CUs — main CPU work

  // 4. Create a command queue on the prefetch sub-device
  //    Reuse the existing shared Context (it was created with the parent
  //    device)
  PrefetchCommandQueue =
      clCreateCommandQueue(Context, PrefetchSubDevice, 0, &err);
  if (err != CL_SUCCESS) {
    printf("[Prefetch] Error %d creating PrefetchCommandQueue\n", err);
    g_prefetchEnabled = 0;
    clReleaseDevice(PrefetchSubDevice);
    clReleaseDevice(MainCPUSubDevice);
    PrefetchSubDevice = NULL;
    MainCPUSubDevice = NULL;
    cl_clean(err);
  }

  // 5. Optionally re-create the main CPU command queue using the main
  // sub-device
  //    so that main CPU work only uses (maxCUs-1) CUs and does not interfere
  //    with the prefetch CU.
  if (CommandQueue[0]) {
    clReleaseCommandQueue(CommandQueue[0]);
  }
  CommandQueue[0] = clCreateCommandQueue(Context, MainCPUSubDevice, 0, &err);
  if (err != CL_SUCCESS) {
    printf("[Prefetch] Error %d re-creating main CPU CommandQueue. "
           "Falling back to parent device.\n",
           err);
    // Fallback: re-create using original CPU device
    CommandQueue[0] = clCreateCommandQueue(Context, Device[0], 0, &err);
  }

  // 5b. Full 8-CU queue on the parent Device[0]. Used for CPU kernels that have NO
  //     concurrent prefetch helper (build, projection, noWAS probe): those can use all
  //     8 cores, since the prefetch core is only busy during the WAS probe. The WAS probe
  //     stays on the 7-CU CommandQueue[0] (helper on the 1-CU PrefetchSubDevice).
  FullCPUCommandQueue = clCreateCommandQueue(Context, Device[0], 0, &err);
  if (err != CL_SUCCESS) {
    printf("[Prefetch] Warning: Error %d creating FullCPUCommandQueue; "
           "no-helper CPU kernels will use the 7-CU queue.\n", err);
    FullCPUCommandQueue = NULL;
  } else {
    printf("[Prefetch] Created FullCPUCommandQueue on parent Device[0] (8 CUs) "
           "for no-helper CPU kernels\n");
  }

  // 6. Set default WAS size if not already configured
  if (g_prefetchWASSize <= 0) {
    cl_ulong cacheSize = 0;
    clGetDeviceInfo(Device[0], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                    sizeof(cl_ulong), &cacheSize, NULL);

    if (cacheSize > 0) {
      // === Zhou et al. VLDB 2005 §4.2 ===
      //
      //   "the expected amount of preloaded data for the entire work-ahead
      //    set should be smaller than the L2 cache size. Since the main
      //    thread may have older cache-resident data that is still being
      //    used, and in order to avoid conflict misses, the threshold
      //    should be lower, such as one quarter of the cache size."
      //
      // Paper assumes Pentium 4 SMT (helper+main share L2). Our setup uses
      // device fission (helper on 1 core, main on 7 cores), so the *shared*
      // cache is L3 — apply the L3/4 rule there.
      //
      // Per entry cache occupancy:
      //   - WAS struct itself              : 32 B (lives in WAS buffer)
      //   - 2 pointers × 64 B cache line   : 128 B (preloaded data)
      //   ────────────────────────────────────────────────
      //   total                            : 160 B
      //
      // Constraint:  N × 160 B ≤ cache / 4
      const size_t entrySize = 32;
      const int pointersPerEntry = 2; // p1 + p2 in WASEntry
      const size_t cacheLineSize = 64;
      const size_t preloadPerEntry = pointersPerEntry * cacheLineSize; // 128
      const size_t costPerEntry = entrySize + preloadPerEntry;         // 160

      int raw = (int)(cacheSize / (4 * costPerEntry));
      // Power of 2 round-down (allows bitmask wrap in WAS_kernel idx step)
      int pot = 1;
      while (pot * 2 <= raw)
        pot *= 2;
      g_prefetchWASSize = pot;

      // Paper-compliance accounting
      double l3_total_pct =
          100.0 * (double)pot * costPerEntry / (double)cacheSize;
      double l3_preload_pct =
          100.0 * (double)pot * preloadPerEntry / (double)cacheSize;
      printf("[Prefetch] L3 = %.1f MB, cost/entry = %zu B "
             "(struct %zu + preload %zu)\n",
             cacheSize / (1024.0 * 1024.0), costPerEntry, entrySize,
             preloadPerEntry);
      printf("[Prefetch] WAS=%d → preload=%.2f%% L3, total=%.2f%% L3 "
             "(paper §4.2 L3/4 = 25%% threshold)\n",
             pot, l3_preload_pct, l3_total_pct);
    }
  }

  printf("[Prefetch] Work-Ahead Set size = %d elements (power of 2)\n",
         g_prefetchWASSize);
  printf("[Prefetch] Device fission prefetching initialized successfully.\n");
}

void cl_cleanup_prefetch() {
  if (FullCPUCommandQueue) {
    clFinish(FullCPUCommandQueue);
    clReleaseCommandQueue(FullCPUCommandQueue);
    FullCPUCommandQueue = NULL;
  }
  if (PrefetchCommandQueue) {
    clFinish(PrefetchCommandQueue);
    clReleaseCommandQueue(PrefetchCommandQueue);
    PrefetchCommandQueue = NULL;
  }
  if (PrefetchSubDevice) {
    clReleaseDevice(PrefetchSubDevice);
    PrefetchSubDevice = NULL;
  }
  if (MainCPUSubDevice) {
    clReleaseDevice(MainCPUSubDevice);
    MainCPUSubDevice = NULL;
  }
  printf("[Prefetch] Cleanup complete.\n");
}

void cl_init_decom() {
  cl_int err;

  // n = number of CUs for decompression (D). Read once from DECOM_CU.
  int n = getenv("DECOM_CU") ? atoi(getenv("DECOM_CU")) : 0;

  cl_uint maxCUs = 0;
  err = clGetDeviceInfo(Device[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                        &maxCUs, NULL);
  if (err != CL_SUCCESS) {
    printf("[cPDE] Error %d querying CL_DEVICE_MAX_COMPUTE_UNITS\n", err);
    g_decomEnabled = 0;
    return;
  }
  printf("[cPDE] CPU Device has %u compute units\n", maxCUs);

  // P takes 1 CU up front if prefetch fission is active.
  int reserved = g_prefetchEnabled ? 1 : 0;
  int avail = (int)maxCUs - reserved;        // CUs left to split between D and E

  int execGpuPct = getenv("EXEC_GPU_PCT") ? atoi(getenv("EXEC_GPU_PCT")) : 0;
  if (execGpuPct < 0) execGpuPct = 0;
  if (execGpuPct > 100) execGpuPct = 100;
  bool eAllGpu = (execGpuPct >= 100);

  if (n < 1) n = 1;

  if (eAllGpu) {
    if (n > avail) n = avail;            // can't exceed available CUs
    int dCUs = avail;                    // D owns every CPU CU left after P
    DecomSubDevice = NULL;               // no fission needed for the {n,0} case
    NonDecomSubDevice = NULL;
    DecomCommandQueue = CommandQueue[0]; // full-CPU (or 7-CU PE-main) queue for D
    NonDecomCommandQueue = NULL;         // E is on GPU; no CPU E queue
    g_decomCUs = dCUs;
    g_nonDecomCUs = 0;
    g_decomEnabled = 1;
    printf("[cPDE] cDE-b ({%d,0}): D=%d CU on CommandQueue[0] (full CPU), "
           "E entirely on GPU (CommandQueue[1])\n", dCUs, dCUs);
    fprintf(stderr, "[cPDE] cDE-b: D=%d CU (CPU), E=0 CU (GPU)\n", dCUs);
    return;
  }

  if (n > avail - 1) {
    // need at least 1 CU left for E
    printf("[cPDE] WARNING: DECOM_CU=%d too large (avail=%d, reserved P=%d); "
           "clamping to %d.\n", n, avail, reserved, avail - 1);
    n = avail - 1;
  }
  if (n < 1 || avail < 2) {
    printf("[cPDE] WARNING: cannot split %d CU(s) into D+E. cPDE disabled.\n",
           avail);
    g_decomEnabled = 0;
    return;
  }
  int eCUs = avail - n;                        // CUs for E

  cl_uint maxSubDevices = 0;
  err = clGetDeviceInfo(Device[0], CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                        sizeof(cl_uint), &maxSubDevices, NULL);
  if (err != CL_SUCCESS || maxSubDevices < 2) {
    printf("[cPDE] WARNING: device does not support partition "
           "(maxSubDevices=%u, err=%d). cPDE disabled.\n", maxSubDevices, err);
    g_decomEnabled = 0;
    return;
  }

  cl_device_id parent = Device[0];
  cl_device_id subDevices[3];
  cl_uint numRet = 0;

  if (g_prefetchEnabled) {
    // 3-way: {1 (P), n (D), 7-n (E)}
    cl_device_partition_property props[] = {
        CL_DEVICE_PARTITION_BY_COUNTS,
        (cl_device_partition_property)1,
        (cl_device_partition_property)n,
        (cl_device_partition_property)eCUs,
        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0};
    err = clCreateSubDevices(parent, props, 3, subDevices, &numRet);
    if (err != CL_SUCCESS || numRet < 3) {
      printf("[cPDE] Error %d in clCreateSubDevices (3-way {1,%d,%d}). "
             "cPDE disabled.\n", err, n, eCUs);
      g_decomEnabled = 0;
      return;
    }
    DecomSubDevice    = subDevices[1];
    NonDecomSubDevice = subDevices[2];
    clReleaseDevice(subDevices[0]);
    printf("[cPDE] 3-way fission: P=1 CU, D=%d CU, E=%d CU "
           "(PE already owns its own P sub-device)\n", n, eCUs);
  } else {
    // 2-way: {n (D), 8-n (E)}
    cl_device_partition_property props[] = {
        CL_DEVICE_PARTITION_BY_COUNTS,
        (cl_device_partition_property)n,
        (cl_device_partition_property)eCUs,
        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0};
    err = clCreateSubDevices(parent, props, 2, subDevices, &numRet);
    if (err != CL_SUCCESS || numRet < 2) {
      printf("[cPDE] Error %d in clCreateSubDevices (2-way {%d,%d}). "
             "cPDE disabled.\n", err, n, eCUs);
      g_decomEnabled = 0;
      return;
    }
    DecomSubDevice    = subDevices[0];
    NonDecomSubDevice = subDevices[1];
    printf("[cPDE] 2-way fission: D=%d CU, E=%d CU\n", n, eCUs);
  }

  DecomCommandQueue = clCreateCommandQueue(Context, DecomSubDevice, 0, &err);
  if (err != CL_SUCCESS) {
    printf("[cPDE] Error %d creating DecomCommandQueue. cPDE disabled.\n", err);
    if (DecomSubDevice) clReleaseDevice(DecomSubDevice);
    if (NonDecomSubDevice) clReleaseDevice(NonDecomSubDevice);
    DecomSubDevice = NonDecomSubDevice = NULL;
    g_decomEnabled = 0;
    return;
  }
  NonDecomCommandQueue =
      clCreateCommandQueue(Context, NonDecomSubDevice, 0, &err);
  if (err != CL_SUCCESS) {
    printf("[cPDE] Error %d creating NonDecomCommandQueue. cPDE disabled.\n",
           err);
    clReleaseCommandQueue(DecomCommandQueue);
    DecomCommandQueue = NULL;
    clReleaseDevice(DecomSubDevice);
    clReleaseDevice(NonDecomSubDevice);
    DecomSubDevice = NonDecomSubDevice = NULL;
    g_decomEnabled = 0;
    return;
  }

  cl_uint dCU = 0, eCUq = 0;
  clGetDeviceInfo(DecomSubDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                  &dCU, NULL);
  clGetDeviceInfo(NonDecomSubDevice, CL_DEVICE_MAX_COMPUTE_UNITS,
                  sizeof(cl_uint), &eCUq, NULL);
  g_decomCUs = (int)dCU;
  g_nonDecomCUs = (int)eCUq;
  g_decomEnabled = 1;
  printf("[cPDE] decom CUs=%u (DecomCommandQueue), exec CUs=%u "
         "(NonDecomCommandQueue)\n", dCU, eCUq);
  fprintf(stderr, "[cPDE] decom CUs=%u, exec CUs=%u\n", dCU, eCUq);
}

void cl_cleanup_decom() {
  if (DecomCommandQueue && DecomCommandQueue == CommandQueue[0]) {
    clFinish(DecomCommandQueue);
    DecomCommandQueue = NULL;
  } else if (DecomCommandQueue) {
    clFinish(DecomCommandQueue);
    clReleaseCommandQueue(DecomCommandQueue);
    DecomCommandQueue = NULL;
  }
  if (NonDecomCommandQueue) {
    clFinish(NonDecomCommandQueue);
    clReleaseCommandQueue(NonDecomCommandQueue);
    NonDecomCommandQueue = NULL;
  }
  if (DecomSubDevice) {
    clReleaseDevice(DecomSubDevice);
    DecomSubDevice = NULL;
  }
  if (NonDecomSubDevice) {
    clReleaseDevice(NonDecomSubDevice);
    NonDecomSubDevice = NULL;
  }
  g_decomEnabled = 0;
  printf("[cPDE] Cleanup complete.\n");
}