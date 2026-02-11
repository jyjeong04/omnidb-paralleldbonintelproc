#ifndef HELPER_H
#define HELPER_H

#include <time.h>
#include <stdio.h>

// Timer functions (implemented in QP_Utility.cpp / KernelEngine)
int genTimer(int timerID);
double getTimer(int timerID);
double getTimerNoSync(int timer);
double endTime(char *info);
double endTimer( char* info, int* timer );

// Simple clock_t-based timer helpers used in Co_*.cpp files
// These are likely the original macros that were missing from the codebase
inline void startTimer(clock_t &t) { t = clock(); }

inline double endTimer(const char* info, clock_t t) {
    double elapsed = (double)(clock() - t) / CLOCKS_PER_SEC;
    // Uncomment for debug: printf("%s: %f sec\n", info, elapsed);
    return elapsed;
}

#endif
