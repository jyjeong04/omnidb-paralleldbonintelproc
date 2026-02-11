#ifndef _GPU_DLL_MIANLIB_H_
#define _GPU_DLL_MIANLIB_H_

#include "cuCSSTree.h"
#include "../MyLib/hashTable.h"

// If TonyLib/OpenCL_DLL.h is already included (_GPU_DLL_H_ defined),
// skip CUDA_CSSTree and overlapping CL_* declarations to avoid conflicts.
// TonyLib uses cl_mem-based CUDA_CSSTree; MianLib uses raw pointer-based.
// When both headers coexist, TonyLib's definitions take precedence.

#ifndef _GPU_DLL_H_
// ---------- MianLib-only CUDA_CSSTree definition (raw pointers) ----------
typedef struct
{
	IDataNode* data;
	unsigned int nDataNodes;

	IDirectoryNode* dir;
	unsigned int nDirNodes;

	int search(int key, Record** Rout)
	{
		return 0;
	}
	void print()
	{
		
	}
} CUDA_CSSTree;
#endif /* !_GPU_DLL_H_ */


///////////////////////////////////////////////////////////////////////////////////
// Functions unique to MianLib (NOT declared in TonyLib/OpenCL_DLL.h)
// These are always declared regardless of TonyLib inclusion.
///////////////////////////////////////////////////////////////////////////////////

//data structure
void CL_BuildHashTable( Record* h_R, int rLen, int intBits, Bound* h_bound );

int CL_BuildTreeIndex( Record* h_Rin, int rLen, CUDA_CSSTree** tree, int CPU_GPU=0 );

int CL_HashSearch( Record* h_R, int rLen, Bound* h_bound, int inBits, Record* h_S, int sLen, Record** h_Rout, 
														  int numThread = 512, int CPU_GPU=0);

int CL_TreeSearch( Record* h_Rin, int rLen, CUDA_CSSTree* tree, Record* h_Sin, int sLen, Record** h_Rout, int CPU_GPU=0 );

//sort (unique to MianLib)
void CL_bitonicSort( Record* h_Rin, int rLen, Record* h_Rout, 
														int numThreadPB = 128, int numBlock = 1024, int CPU_GPU=0);

void CL_QuickSort( Record* h_Rin, int rLen, Record* h_Rout, int CPU_GPU=0);

//partition
void CL_Partition( Record* h_Rin, int rLen, int numPart, Record* d_Rout, int* d_startHist, 
														  int numThreadPB = -1, int numBlock = -1, int CPU_GPU=0);

void resetGPU();


#ifndef _GPU_DLL_H_
///////////////////////////////////////////////////////////////////////////////////
// Functions that overlap with TonyLib/OpenCL_DLL.h
// Only declared if TonyLib is NOT included (to avoid conflicting linkage/defaults)
///////////////////////////////////////////////////////////////////////////////////

//selection
int CL_PointSelection( Record* h_Rin, int rLen, int matchingKeyValue, Record** h_Rout, 
															  int numThreadPB = 32, int numBlock = 256,int CPU_GPU=0);

int CL_RangeSelection( Record* h_Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record** h_Rout, 
															  int numThreadPB = 64, int numBlock = 512,int CPU_GPU=0);

//projection
void CL_Projection( Record* h_Rin, int rLen, Record* h_projTable, int pLen, 
														   int numThread = 256, int numBlock = 256 ,int CPU_GPU=0);


//aggregation
int CL_AggMax( Record* h_Rin, int rLen, Record** d_Rout,
													  int numThread = 256, int numBlock = 1024 ,int CPU_GPU=0);

int CL_AggMin( Record* h_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 ,int CPU_GPU=0);

int CL_AggSum( Record* h_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 ,int CPU_GPU=0);

int CL_AggAvg( Record* h_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024,int CPU_GPU=0 );

//group by
int	CL_GroupBy( Record* h_Rin, int rLen, Record* h_Rout, int** h_startPos, 
					int numThread = 64, int numBlock = 1024,int CPU_GPU=0 );

//agg after group by
void CL_agg_max_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread = 256,int CPU_GPU=0 ); 

void CL_agg_min_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread = 256,int CPU_GPU=0 ); 

void CL_agg_sum_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread = 256,int CPU_GPU=0 ); 

void CL_agg_avg_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread = 256,int CPU_GPU=0 ); 

//sort
void CL_RadixSort( Record* h_Rin, int rLen, Record* h_Rout ,int CPU_GPU=0);

//join
int CL_ninlj( Record* h_R, int rLen, Record* h_S, int sLen, Record** h_Rout ,int CPU_GPU=0);

int	CL_smj( Record* h_R, int rLen, Record* h_S, int sLen, Record** h_Joinout ,int CPU_GPU=0);

int CL_hj( Record* h_Rin, int rLen, Record* d_Sin, int sLen, Record** h_Rout,int CPU_GPU=0 );

int CL_inlj( Record* h_Rin, int rLen, CUDA_CSSTree* h_tree, Record* h_Sin, int sLen, Record** h_Rout ,int CPU_GPU=0);

int CL_mj( Record* h_Rin, int rLen, Record* h_Sin, int sLen, Record** h_Joinout,int CPU_GPU=0 );

#endif /* !_GPU_DLL_H_ */

#endif /* _GPU_DLL_MIANLIB_H_ */
