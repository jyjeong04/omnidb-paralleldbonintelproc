#include "BinaryThreadOp.h"
#include "../MyLib/CPU_Dll.h"
#include "CoProcessor.h"
#include "../TonyLib/OpenCL_DLL.h"
BinaryThreadOp::BinaryThreadOp(OP_MODE opt)
{
	optType=opt;
}


void BinaryThreadOp::init(cl_mem p_R, int p_rLen, cl_mem p_S, int p_sLen)
{
	R=p_R;
	Query_rLen=p_rLen;
	S=p_S;
	sLen=p_sLen;
	nsHJ=0;
	nsNarrow=0;
}

void BinaryThreadOp::initNSHJ(cl_mem p_R, int p_rLen, cl_mem p_S, int p_sLen, int narrow)
{
	R=p_R;
	Query_rLen=p_rLen;
	S=p_S;
	sLen=p_sLen;
	nsHJ=1;
	nsNarrow=narrow;
}

BinaryThreadOp::~BinaryThreadOp(void)
{
}


void BinaryThreadOp::execute(EXEC_MODE eM)
{
		//ON_GPU("BinaryThreadOp::execute");
		switch(optType)
		{
			case JOIN_NINLJ:
				{
					//numResult=CL_ninlj(R,Query_rLen,S,sLen,&Rout,eM);		
					cout<<"you should not be here"<<endl;
				}break;
			case JOIN_INLJ:
				{
					cout<<"you should not be here"<<endl;
				}break;
			case JOIN_SMJ:
				{			
					//printf("CL_smjOnly::execute\n");
					numResult=CL_smjOnly(R,Query_rLen,S,sLen,&Rout,eM);
					//printf("CL_smjOnly::finished\n");
				}break;
			case JOIN_HJ:
				{
					static int decomMode = -1;
					if (decomMode < 0)
						decomMode = (getenv("DECOM_MODE") && atoi(getenv("DECOM_MODE"))) ? 1 : 0;
					if(nsHJ && nsNarrow && decomMode)
					{
						// cPDE HJ: decompress (D) the NS-compressed R/S, then run the
						// build+probe (E) on the decompressed 8-byte data (CPU+GPU split,
						// optional P). Only the narrow/compressed path feeds cPDE.
						numResult=CL_hjOnly_cPDE(R,Query_rLen,S,sLen,&Rout,eM);
					}
					else if(nsHJ)
					{
						numResult=CL_hjOnly_ns(R,Query_rLen,S,sLen,&Rout,nsNarrow,eM);
					}
					else
					{
						//printf("CL_hjOnly::execute\n");
						numResult=CL_hjOnly(R,Query_rLen,S,sLen,&Rout,eM);
						//printf("CL_hjOnly::finished\n");
					}
				}break;
		}
		//ON_GPU_DONE("BinaryThreadOp::execute");
}


ThreadOp* BinaryThreadOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}
/*
* for the indexed nlj.
*/
IndexJoinThreadOp::IndexJoinThreadOp(OP_MODE opt):BinaryThreadOp(opt)
{
}


void IndexJoinThreadOp::init(cl_mem p_R, int p_rLen,cl_mem p_S, int p_sLen)
{
	R=p_R;
	Query_rLen=p_rLen;
	S=p_S;
	sLen=p_sLen;
}

IndexJoinThreadOp::~IndexJoinThreadOp(void)
{

}

void IndexJoinThreadOp::execute(EXEC_MODE eM)
{
	//printf("IndexJoinThreadOp::execute\n");
	//Kernel_bufferchecking(R,100);
	//Kernel_bufferchecking(S,100);
	numResult=CL_inljOnly(R,Query_rLen,&h_tree,S,sLen,&Rout,eM);
	//Kernel_bufferchecking(Rout,sizeof(Record)*numResult);
	//printf("IndexJoinThreadOp::execute done\n");
}

ThreadOp* IndexJoinThreadOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}
