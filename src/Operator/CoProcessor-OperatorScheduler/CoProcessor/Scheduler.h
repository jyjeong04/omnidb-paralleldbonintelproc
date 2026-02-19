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
EXEC_MODE OPScheduler(OP_MODE _optType);