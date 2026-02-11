#ifndef THREAD_POOL_COP_H
#define THREAD_POOL_COP_H

//multiple threads for co-processing.

#ifdef _WIN32
#include "windows.h"
#else
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#endif

#define NUM_CORE_CPU 8
#define MAX_RUN ((1<<(NUM_CORE_CPU+1))-1)

#ifdef _WIN32
// Original Windows implementation
struct MyThreadPoolCop
{
public:
	void destory()
	{
		if(dwThreadId!=NULL) free(dwThreadId);
		if(hThread!=NULL) free(hThread);
		if(masks!=NULL) free(masks);
		if(tParam!=NULL) free(tParam);
		if(pfnThreadProc!=NULL) free(pfnThreadProc);
	}
	int numThread;
	int* masks;
	DWORD *dwThreadId;
	HANDLE *hThread;
	void ** tParam;
	LPTHREAD_START_ROUTINE* pfnThreadProc;

	int setCPUID(int tid, int cpuid)
	{
		int maskValue=(1<<cpuid)%MAX_RUN;
		masks[tid]=maskValue;
		return 0;
	}

	int assignTask(int tid, LPTHREAD_START_ROUTINE pfn)
	{
		pfnThreadProc[tid]=pfn;
		return 0;
	}

	int assignParameter(int tid, void* pvParam)
	{
		tParam[tid]=pvParam;
		return 0;
	}

	int run(void)
	{
		int i=0;
		for( i=0; i<numThread; i++ )
		{
			hThread[i] = CreateThread(NULL, 0, pfnThreadProc[i], tParam[i], 0, &dwThreadId[i]);
			SetThreadAffinityMask(hThread[i],masks[i]);
			if (hThread[i] == NULL)
				ExitProcess(i);
		}
		WaitForMultipleObjects(numThread, hThread, TRUE, INFINITE);
		return 0;
	}

	void create(int nThead)
	{
		numThread=nThead;
		dwThreadId=(DWORD*)malloc(sizeof(DWORD)*numThread);
		hThread=(HANDLE*)malloc(sizeof(HANDLE)*numThread);
		masks=(int*)malloc(sizeof(int)*numThread);
		for(int i=0;i<numThread;i++)
			masks[i]=(1<<i)%MAX_RUN;
		tParam=(void**)malloc(sizeof(void*)*numThread);
		pfnThreadProc=(LPTHREAD_START_ROUTINE*)malloc(sizeof(LPTHREAD_START_ROUTINE)*numThread);
	}
};

#else
// Linux pthreads implementation
typedef void* (*PThreadFunc)(void*);

struct MyThreadPoolCop
{
public:
	void destory()
	{
		if(threads!=NULL) free(threads);
		if(masks!=NULL) free(masks);
		if(tParam!=NULL) free(tParam);
		if(pfnThreadProc!=NULL) free(pfnThreadProc);
	}
	int numThread;
	int* masks;
	pthread_t *threads;
	void ** tParam;
	PThreadFunc* pfnThreadProc;

	int setCPUID(int tid, int cpuid)
	{
		int maskValue=(1<<cpuid)%MAX_RUN;
		masks[tid]=maskValue;
		return 0;
	}

	int assignTask(int tid, PThreadFunc pfn)
	{
		pfnThreadProc[tid]=pfn;
		return 0;
	}

	int assignParameter(int tid, void* pvParam)
	{
		tParam[tid]=pvParam;
		return 0;
	}

	int run(void)
	{
		int i=0;
		for( i=0; i<numThread; i++ )
		{
			int ret = pthread_create(&threads[i], NULL, pfnThreadProc[i], tParam[i]);
			if (ret != 0)
			{
				exit(i);
			}
#ifdef __linux__
			// Set thread affinity
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			for (int c = 0; c < NUM_CORE_CPU * 2; c++)
			{
				if (masks[i] & (1 << c))
					CPU_SET(c, &cpuset);
			}
			pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
#endif
		}
		// Wait until all threads have terminated.
		for(i=0; i<numThread; i++)
		{
			pthread_join(threads[i], NULL);
		}
		return 0;
	}

	void create(int nThead)
	{
		numThread=nThead;
		threads=(pthread_t*)malloc(sizeof(pthread_t)*numThread);
		masks=(int*)malloc(sizeof(int)*numThread);
		int i=0;
		for(i=0;i<numThread;i++)
			masks[i]=(1<<i)%MAX_RUN;
		tParam=(void**)malloc(sizeof(void*)*numThread);
		pfnThreadProc=(PThreadFunc*)malloc(sizeof(PThreadFunc)*numThread);
	}
};

#endif

#endif
