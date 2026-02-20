#ifndef MAP_IMPL_H
#define MAP_IMPL_H
#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#ifdef __APPLE__
#include </opt/homebrew/opt/libomp/include/omp.h>
#else
#include <omp.h>
#endif
#include "math.h"


//primitives.
#ifdef _WIN32
template <class T> DWORD WINAPI tp_mapImpl( LPVOID lpParam ) 
#else
template <class T> void* tp_mapImpl( void* lpParam ) 
#endif
{ 
    ws_map<T>* pData;
	pData = (ws_map<T>*)lpParam;
	Record *Rin=pData->Rin;
	T* Rout=pData->Rout;
	mapper_t mapFunc=pData->mapFunc;
	int startID=pData->startID;
	int endID=pData->endID;
	void* para=pData->para;
	int i=0;
	int sum=0;
	for(i=startID;i<endID;i++)
	{
		//mapFunc((void*)(Rin+i), para, (void*)(Rout+i));
		mapFunc((void*)(Rin+i), para, (((T*)Rout)+i));
	}
#ifdef _WIN32
	return sum;
#else
	return NULL;
#endif
} 


template <class T> void mapImpl_thread(Record *Rin, int Query_rLen, mapper_t mapFunc, void* para, T* Rout, int numThread)
{
	int result=0;
	MyThreadPool *pool=new MyThreadPool();
	pool->init(numThread, tp_mapImpl<T>);
	int i=0;
	ws_map<T>** pData=(ws_map<T>**)malloc(sizeof(ws_map<T>*)*numThread);
	int chunkSize=Query_rLen/numThread;
	if(Query_rLen%numThread!=0)
		chunkSize++;
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (ws_map<T>*) calloc(1, sizeof(ws_map<T>));

		if( pData[i]  == NULL )
			exit(2);

		// Generate unique data for each thread.
		pData[i]->Rin=Rin;
		pData[i]->Rout=Rout;
		pData[i]->para=para;
		pData[i]->mapFunc=mapFunc;
		pData[i]->startID=i*chunkSize;
		pData[i]->endID=(i+1)*chunkSize;
		if(pData[i]->endID>Query_rLen)
			pData[i]->endID=Query_rLen;
		pool->assignParameter(i, pData[i]);
	}
	pool->run();
	delete pool;
	for(i=0;i<numThread;i++)
		free(pData[i]);
	free(pData);
}

//open mp versions.
template <class T> void mapImpl_openmp(Record *Rin, int Query_rLen, mapper_t mapFunc, void* para, T* Rout)
{
	int i=0;
	omp_set_num_threads(1);
	#pragma omp parallel for
	for(i=0;i<Query_rLen;i++)
	{
		mapFunc((void*)(Rin+i), para, (void*)(Rout+i));
		//double value=sqrt(sqrt((double)Rin[i].value));
		//((int*)Rout)[i]=(int)value;
	}
}

#endif



