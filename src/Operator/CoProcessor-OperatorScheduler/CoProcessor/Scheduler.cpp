#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "QueryPlanNode.h"
#include "QueryPlanTree.h"
#include "../MyLib/QP_Utility.h"
#include "time.h"
#include "assert.h"
#include "Database.h"
#include "SingularThreadOp.h"
#include "SortThreadOp.h"
#include "BinaryThreadOp.h"
#include "GroupByThreadOp.h"
#include <iostream>
#include "CoProcessorTest.h"
/////////////////////////////////////////////////////////////
extern double OP_LothresholdForGPUApp;
extern double OP_LothresholdForCPUApp;
extern double SpeedupGPUOverCPU_Operator[22];
extern double AddCPUBurden_OP[22];
extern double AddGPUBurden_OP[22];
extern double OP_UpCPUBurden;
extern double OP_LoCPUBurden;
extern double OP_UpGPUBurden;
extern double OP_LoGPUBurden;
extern double OP_CPUBurden;
extern double OP_GPUBurden;
extern pthread_mutex_t OP_CPUBurdenCS;
extern pthread_mutex_t OP_GPUBurdenCS;
extern pthread_mutex_t preEMCS;

//#define Greedy
static int preEm=EXEC_CPU;
void inline GPUBurdenINC(const double burden){
		pthread_mutex_lock(&(OP_GPUBurdenCS));
		OP_GPUBurden+=(burden);
		pthread_mutex_unlock(&(OP_GPUBurdenCS));
}
void inline CPUBurdenINC(const double burden){
		pthread_mutex_lock(&(OP_CPUBurdenCS));
		OP_CPUBurden+=(burden);
		pthread_mutex_unlock(&(OP_CPUBurdenCS));
}

EXEC_MODE OPScheduler(OP_MODE _optType)
{
/*OPERATOR SCHEDULER*/
	EXEC_MODE eM;
#ifdef Greedy
	if((OP_CPUBurden+AddCPUBurden_OP[_optType])<(OP_GPUBurden+AddGPUBurden_OP[_optType]))
	{
		eM=EXEC_CPU;
		CPUBurdenINC(AddCPUBurden_OP[_optType]);
	}else{
		eM=EXEC_GPU;
		GPUBurdenINC(AddGPUBurden_OP[_optType]);
	}
#else
		pthread_mutex_lock(&(preEMCS));
	 	if(preEm==EXEC_CPU){
			eM=EXEC_GPU;
			GPUBurdenINC(AddGPUBurden_OP[_optType]);
			preEm=EXEC_GPU;
		}
		else{
			eM=EXEC_CPU;
			CPUBurdenINC(AddCPUBurden_OP[_optType]);
			preEm=EXEC_CPU;
		}
		pthread_mutex_unlock(&(preEMCS));
#endif
	return eM;
}
