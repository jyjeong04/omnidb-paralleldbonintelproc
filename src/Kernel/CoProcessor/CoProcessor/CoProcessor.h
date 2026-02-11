#ifndef CO_PROCESSOR_H
#define CO_PROCESSOR_H
#include "../MyLib/QP_Utility.h"

#ifdef _WIN32
#include "windows.h"
#else
#include <pthread.h>
#include <stdlib.h>
// Linux compatibility: HANDLE is a pthread_mutex_t pointer
typedef pthread_mutex_t* HANDLE;

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef INFINITE
#define INFINITE 0xFFFFFFFF
#endif

// Windows API compatibility wrappers using pthreads
static inline HANDLE CreateMutexLinux() {
	HANDLE mtx = (HANDLE)malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(mtx, NULL);
	return mtx;
}
static inline void WaitForSingleObject(HANDLE mtx, unsigned int /*timeout_unused*/) {
	pthread_mutex_lock(mtx);
}
static inline void ReleaseMutex(HANDLE mtx) {
	pthread_mutex_unlock(mtx);
}
static inline void CloseHandle(HANDLE mtx) {
	pthread_mutex_destroy(mtx);
	free(mtx);
}
#define CreateMutex(a, b, c) CreateMutexLinux()

#endif /* !_WIN32 */

#include "../MyLib/CC_CSSTree.h"
#include "db.h"
#include "../TonyLib/OpenCL_DLL.h"
#include "../MianLib/GPU_Dll.h"
#include <vector>
using namespace std;

//processing the query in the device memory only. 
//#define GPUONLY_QP 1


void GPUDEBUG_Int(int* d_RIDList, int RIDLen, bool pGPUONLY_QP);
void GPUDEBUG_Record(Record* d_RIDList, int RIDLen, bool pGPUONLY_QP);


typedef enum{
	EXEC_CPU,//0
	EXEC_GPU,//1
}EXEC_MODE;




#define ADAPTIVE_DEBUG 1

#ifdef ADAPTIVE_DEBUG

#define ON_CPU(OP) cout<<OP<<" on the CPU"<<endl;
#define ON_GPU(OP) cout<<OP<<" on the GPU"<<endl;
#define ON_CPU_GPU(OP) cout<<OP<<" on both GPU and CPU"<<endl;
#define ON_GPUONLY(OP) cout<<OP<<" ON_GPUONLY"<<endl;


#define ON_CPU_DONE(OP) cout<<OP<<" on the CPU DONE!"<<endl;
#define ON_GPU_DONE(OP) cout<<OP<<" on the GPU DONE!"<<endl;
#define ON_CPU_GPU_DONE(OP) cout<<OP<<" on both GPU and CPU DONE!"<<endl;
#define ON_GPUONLY_DONE(OP) cout<<OP<<" ON_GPUONLY DONE!"<<endl;

#define DATA_FROM_GPU(num) cout<<"DATA_FROM_GPU, "<<num<<" bytes"<<endl;
#define DATA_TO_GPU(num) cout<<"DATA_TO_GPU, "<<num<<" bytes"<<endl;
#else

#define ON_CPU(OP) ;
#define ON_GPU(OP) ;
#define ON_CPU_GPU(OP) ;
#define ON_GPUONLY(OP) ;


#define ON_CPU_DONE(OP) ;
#define ON_GPU_DONE(OP) ;
#define ON_CPU_GPU_DONE(OP) ;
#define ON_GPUONLY_DONE(OP) ;

#define DATA_FROM_GPU(num) ;
#define DATA_TO_GPU(num) ;

#endif

/*
* here is the data structure for the sorting algorithm.
*/
#define CPU_SORT_BLOCKSIZE (8*1024*1024)
#define GPU_SORT_BLOCKSIZE (16*1024*1024)

struct ws_cosort
{
	Record* R;
	int Query_rLen;
	Record *Rout;
	int blockSize;
//shared parameters
	HANDLE dispatchMutex;
	HANDLE mergeMutex;
	int* curPosToSort;
	EXEC_MODE execMode;
	vector<int>* tempSizeVec;
	void init(Record *pR, int pRLen, int pBlockSize, HANDLE pDMutex, HANDLE pMMutex, int *curP, 
		EXEC_MODE eM,vector<int>* pSizeVec, Record *pRout)
	{
		R=pR;
		Query_rLen=pRLen;
		blockSize=pBlockSize;
		dispatchMutex=pDMutex;
		mergeMutex=pMMutex;
		curPosToSort=curP;
		execMode=eM;
		tempSizeVec=pSizeVec;
		Rout=pRout;
	}
	
};

//parameter for co-processing on the CPU and the GPU


/*
* here are the data structure for ninlj
*/
#define CPU_NINLJ_BLOCKSIZE (192*1024)
#define GPU_NINLJ_BLOCKSIZE (1024*1024)
struct ws_coninlj
{
	Record* R;
	int Query_rLen;
	Record* S;
	int sLen;
	Record *Rout;
	int blockSizeOnR;
//shared parameters
	HANDLE dispatchMutex;
	HANDLE mergeMutex;
	int* curPosToJoin;
	EXEC_MODE execMode;
	vector<Record**>* tempResultVec;
	vector<int>* tempSizeVec;//one int corresponds to one Record** in tempResultVec.
	void init(Record *pR, int pRLen, int pBlockSize, Record* pS, int pSLen, 
		HANDLE pDMutex, HANDLE pMMutex, int *curP, EXEC_MODE eM,
		vector<Record**>* pRVec, vector<int>* pSizeVec)
	{
		R=pR;
		Query_rLen=pRLen;
		S=pS;
		sLen=pSLen;
		blockSizeOnR=pBlockSize;
		dispatchMutex=pDMutex;
		mergeMutex=pMMutex;
		curPosToJoin=curP;
		execMode=eM;
		tempResultVec=pRVec;
		tempSizeVec=pSizeVec;
	}
	
};


/*
* here are the data structure for ninlj
*/
#define CPU_INLJ_BLOCKSIZE (1*1024*1024)
#define GPU_INLJ_BLOCKSIZE (6*1024*1024)

struct ws_coinlj
{
	Record* R;
	int Query_rLen;
	Record* S;
	int sLen;
	Record *Rout;
	int blockSizeOnR;
//shared parameters
	CC_CSSTree *cpu_tree;
	CUDA_CSSTree *gpu_tree;
	HANDLE dispatchMutex;
	HANDLE mergeMutex;
	int* curPosToJoin;
	EXEC_MODE execMode;
	vector<Record**>* tempResultVec;
	vector<int>* tempSizeVec;//one int corresponds to one Record** in tempResultVec.
	void init(Record *pR, int pRLen, int pBlockSize, Record* pS, int pSLen, 
		CC_CSSTree *cT, CUDA_CSSTree *gT,
		HANDLE pDMutex, HANDLE pMMutex, int *curP, EXEC_MODE eM,
		vector<Record**>* pRVec, vector<int>* pSizeVec)
	{
		R=pR;
		Query_rLen=pRLen;
		S=pS;
		sLen=pSLen;
		blockSizeOnR=pBlockSize;
		dispatchMutex=pDMutex;
		mergeMutex=pMMutex;
		curPosToJoin=curP;
		execMode=eM;
		tempResultVec=pRVec;
		tempSizeVec=pSizeVec;
		cpu_tree=cT;
		gpu_tree=gT;
	}
	
};

/*
* the data structures for hj.
*/

struct ws_hj
{
	Record* R;
	int rLen;
	Record* S;
	int sLen;
	int* curPartition;
//shared parameters
	int *RStartHist;
	int *SStartHist;
	int numPartition;
	HANDLE dispatchMutex;
	HANDLE mergeMutex;
	int* curPosToJoin;
	EXEC_MODE execMode;
	vector<Record**>* tempResultVec;
	vector<int>* tempSizeVec;//one int corresponds to one Record** in tempResultVec.
	void init(Record *pR, int pRLen,  Record* pS, int pSLen, 
		int *pRStartHist, int *pSStartHist, int pnumPartition,
		HANDLE pDMutex, HANDLE pMMutex, int *curP, EXEC_MODE eM,
		vector<Record**>* pRVec, vector<int>* pSizeVec)
	{
		R=pR;
		rLen=pRLen;
		S=pS;
		sLen=pSLen;
		dispatchMutex=pDMutex;
		mergeMutex=pMMutex;
		curPosToJoin=curP;
		execMode=eM;
		tempResultVec=pRVec;
		tempSizeVec=pSizeVec;
		RStartHist=pRStartHist;
		SStartHist=pSStartHist;
		numPartition=pnumPartition;
		curPartition=curP;
	}
	
};


//functions
void CO_Sort(Record *R, int Query_rLen, Record* Rout);
int CO_ninlj(Record *R, int Query_rLen, Record *S, int sLen, Record** Rout);
int MergeJoinResult(vector<Record**>* tempResultVec, vector<int>* tempSizeVec, Record** Rout);

int CO_inlj(Record *R, int Query_rLen, CC_CSSTree *cT,  CUDA_CSSTree* gt, Record *S, int sLen, Record** Rout);
int CO_Smj(Record *R, int Query_rLen, Record *S, int sLen, Record** Rout);
int CO_hj(Record *R, int Query_rLen, Record *S, int sLen, Record** Rout);

char* OpToString(OP_MODE op, EXEC_MODE eM);

void setHighPriority();

///////////////////////////////////////////////////////////////////////////////////
// GPUCopy_* wrapper functions
// These wrap the CL_* host-pointer functions with CPU_GPU=1 (GPU execution).
// The CL_* functions handle the host-to-device copy, kernel execution, and
// device-to-host copy internally.
///////////////////////////////////////////////////////////////////////////////////
inline int GPUCopy_hj(Record *R, int rLen, Record *S, int sLen, Record **Rout) {
	return CL_hj(R, rLen, S, sLen, Rout, 1);
}

inline int GPUCopy_ninlj(Record *R, int rLen, Record *S, int sLen, Record **Rout) {
	return CL_ninlj(R, rLen, S, sLen, Rout, 1);
}

inline int GPUCopy_inlj(Record *R, int rLen, CUDA_CSSTree *tree, Record *S, int sLen, Record **Rout) {
	// TonyLib CL_inlj takes CUDA_CSSTree** (pointer-to-pointer)
	CUDA_CSSTree *treePtr = tree;
	return CL_inlj(R, rLen, &treePtr, S, sLen, Rout, 1);
}

inline void GPUCopy_QuickSort(Record *R, int rLen, Record *Rout) {
	CL_QuickSort(R, rLen, Rout, 1);
}

inline void GPUCopy_bitonicSort(Record *R, int rLen, Record *Rout) {
	CL_bitonicSort(R, rLen, Rout, 128, 1024, 1);
}

inline void GPUCopy_Partition(Record *R, int rLen, int numPart, Record *Rout, int *startHist) {
	CL_Partition(R, rLen, numPart, Rout, startHist, -1, -1, 1);
}


#endif
