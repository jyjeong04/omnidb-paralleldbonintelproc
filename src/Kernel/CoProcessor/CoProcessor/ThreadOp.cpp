#include "ThreadOp.h"
#include "../MyLib/CPU_Dll.h"
#include <iostream>
#include "../TonyLib/OpenCL_DLL.h"
using namespace std;

ThreadOp::ThreadOp()
{
	isFinished=false;
	numResult=Query_rLen;
	Rout=NULL;
}

ThreadOp::~ThreadOp()
{
	clReleaseMemObject(this->R);
}


