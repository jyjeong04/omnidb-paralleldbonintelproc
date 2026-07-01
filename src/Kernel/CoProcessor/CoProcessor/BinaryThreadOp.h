#pragma once
#include "ThreadOp.h"
#include "../MyLib/CPU_Dll.h"

class BinaryThreadOp :
	public ThreadOp
{
public:
	cl_mem S;
	int sLen;
	int nsHJ;
	int nsNarrow;
	BinaryThreadOp(OP_MODE opt);
	void init(cl_mem p_R, int p_rLen, cl_mem p_S, int p_sLen);
	void initNSHJ(cl_mem p_R, int p_rLen, cl_mem p_S, int p_sLen, int narrow);
	~BinaryThreadOp(void);
	void execute(EXEC_MODE eM);
	ThreadOp* getNextOp(EXEC_MODE eM);
};

class IndexJoinThreadOp: public BinaryThreadOp
{
public:
	CUDA_CSSTree *h_tree;
	IndexJoinThreadOp(OP_MODE opt);
	void init(cl_mem p_R, int p_rLen, cl_mem p_S, int p_sLen);
	~IndexJoinThreadOp(void);
	void execute(EXEC_MODE eM);
	ThreadOp* getNextOp(EXEC_MODE eM);

};
