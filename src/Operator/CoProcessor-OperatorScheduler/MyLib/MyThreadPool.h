#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include "stdafx.h"

#ifdef _WIN32
#include "windows.h"
#else
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#endif

#define NUM_CORE_CPU 8
#define MAX_RUN ((1<<NUM_CORE_CPU)-1)

#ifdef _WIN32
// Windows: original definitions
class MyThreadPool
{
public:
	MyThreadPool(void);
	~MyThreadPool(void);
	int numThread;
	int* masks;
	DWORD *dwThreadId;
	HANDLE *hThread;
	void ** tParam;
	int setMask(int tid, int maskValue);
	int assignTask(LPTHREAD_START_ROUTINE  pfnThreadProc);
	int assignParameter(int tid, void* pvParam);
	LPTHREAD_START_ROUTINE pfnThreadProc;
	int run(void);
	int init(int nThead, LPTHREAD_START_ROUTINE  pfn);
};
#else
// Linux: pthreads-based implementation
typedef void* (*ThreadFunc)(void*);

class MyThreadPool
{
public:
	MyThreadPool(void) : numThread(0), masks(NULL), threads(NULL), tParam(NULL), pfnThreadProc(NULL) {}
	~MyThreadPool(void) {}
	int numThread;
	int* masks;
	pthread_t *threads;
	void ** tParam;
	int setMask(int tid, int maskValue)
	{
		int mv = (1 << maskValue) % MAX_RUN;
		masks[tid] = mv;
		return 0;
	}
	int assignTask(ThreadFunc pfn)
	{
		pfnThreadProc = pfn;
		return 0;
	}
	int assignParameter(int tid, void* pvParam)
	{
		tParam[tid] = pvParam;
		return 0;
	}
	ThreadFunc pfnThreadProc;
	int run(void)
	{
		int i = 0;
		for (i = 0; i < numThread; i++)
		{
			pthread_create(&threads[i], NULL, pfnThreadProc, tParam[i]);
#ifdef __linux__
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			// Set affinity based on mask
			for (int c = 0; c < NUM_CORE_CPU; c++)
			{
				if (masks[i] & (1 << c))
					CPU_SET(c, &cpuset);
			}
			pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
#endif
		}
		for (i = 0; i < numThread; i++)
		{
			pthread_join(threads[i], NULL);
		}
		return 0;
	}
	int init(int nThread, ThreadFunc pfn)
	{
		numThread = nThread;
		threads = (pthread_t*)malloc(sizeof(pthread_t) * numThread);
		masks = (int*)malloc(sizeof(int) * numThread);
		for (int i = 0; i < numThread; i++)
			masks[i] = (1 << i) % MAX_RUN;
		tParam = (void**)malloc(sizeof(void*) * numThread);
		pfnThreadProc = pfn;
		return 0;
	}
};
#endif

#endif
