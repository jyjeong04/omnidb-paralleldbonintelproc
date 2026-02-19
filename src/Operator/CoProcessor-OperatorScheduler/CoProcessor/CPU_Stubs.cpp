/**
 * CPU_Stubs.cpp
 *
 * Implementations of CPU-side database operations and GPU host-pointer wrapper
 * functions that were originally in a separate Windows DLL.
 *
 * These provide:
 * - CPU_* functions: OpenMP-based CPU implementations of sort, join, etc.
 * - CL_Partition, CL_QuickSort: host-pointer wrappers for GPU operations
 * - set_selfCPUID, set_thread_affinity, etc.: thread affinity helpers
 */

#include "CoProcessor.h"
#include "../MyLib/CPU_Dll.h"
#include "../MyLib/common.h"
#include "../MyLib/hashTable.h"
#include "../MyLib/CC_CSSTree.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <omp.h>

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#endif

// ---------- CC_CSSTree::search implementation ----------
// Search for key in CSS-tree, returns the position in the data array
int CC_CSSTree::search(int key) {
    int curNode = 0; // start at root
    int i, startKey, endKey;
    for (int lv = 0; lv < level; lv++) {
        startKey = curNode * blockSize;
        endKey = startKey + blockSize;
        // Linear search within node to find child pointer
        int childIdx = 0;
        for (i = startKey; i < endKey; i++) {
            if (ntree[i] == -1 || key <= ntree[i]) break;
            childIdx++;
        }
        if (lv < level - 1) {
            // Navigate to child node
            curNode = curNode * fanout + 1 + childIdx;
        } else {
            // At leaf level, compute data position
            int dataStart = (curNode - ((int)((pow((double)fanout, (double)(level-1)) - 1) / (fanout - 1)))) * blockSize;
            int pos = dataStart + childIdx;
            if (pos >= 0 && pos < numRecord && data[pos].value == key) {
                return pos;
            }
            // Try neighbors
            for (int d = -1; d <= 1; d += 2) {
                int np = pos + d;
                if (np >= 0 && np < numRecord && data[np].value == key) return np;
            }
            return -1; // not found
        }
    }
    return -1;
}

// ---------- buildHashTable implementation ----------
void buildHashTable(Record* h_R, int rLen, int intBits, Bound *h_bound) {
    int hashSize = (1 << intBits);
    // Initialize bounds
    for (int i = 0; i < hashSize; i++) {
        h_bound[i].start = -1;
        h_bound[i].end = -1;
    }
    // Simple approach: sort by hash value and set bounds
    int hashMask = hashSize - 1;
    // Count per bucket
    int* counts = (int*)calloc(hashSize, sizeof(int));
    for (int i = 0; i < rLen; i++) {
        int h = h_R[i].value & hashMask;
        counts[h]++;
    }
    // Set bounds
    int offset = 0;
    for (int i = 0; i < hashSize; i++) {
        if (counts[i] > 0) {
            h_bound[i].start = offset;
            h_bound[i].end = offset + counts[i];
            offset += counts[i];
        }
    }
    free(counts);
}

// ---------- HashSearch_omp implementation ----------
int HashSearch_omp(Record* R, int rLen, Bound *h_bound, int intBits, Record *S, int sLen, Record** Rout, int numThread) {
    int hashMask = (1 << intBits) - 1;
    // Count matches
    int count = 0;
    for (int i = 0; i < sLen; i++) {
        int h = S[i].value & hashMask;
        if (h_bound[h].start >= 0) {
            for (int j = h_bound[h].start; j < h_bound[h].end && j < rLen; j++) {
                if (R[j].value == S[i].value) count++;
            }
        }
    }
    if (count == 0) { *Rout = NULL; return 0; }
    *Rout = new Record[count];
    int idx = 0;
    for (int i = 0; i < sLen; i++) {
        int h = S[i].value & hashMask;
        if (h_bound[h].start >= 0) {
            for (int j = h_bound[h].start; j < h_bound[h].end && j < rLen; j++) {
                if (R[j].value == S[i].value) {
                    (*Rout)[idx].rid = R[j].rid;
                    (*Rout)[idx].value = R[j].value;
                    idx++;
                }
            }
        }
    }
    return count;
}

// ---------- Comparator for qsort ----------
static int cmp_record_value(const void *a, const void *b) {
    int va = ((const Record*)a)->value;
    int vb = ((const Record*)b)->value;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

// ---------- Thread affinity ----------
void set_selfCPUID(int id) {
#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

void set_thread_affinity(int id, int numThread) {
#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(id % NUM_CORE_PER_CPU, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

void set_CPU_affinity(int id, int numThread) {
    set_thread_affinity(id, numThread);
}

void set_numCoreForCPU(int numCore) {
    // Stub: could be used to limit cores
}

// ---------- CPU Sort ----------
void CPU_Sort(Record* Rin, int rLen, Record* Rout, int numThread) {
    memcpy(Rout, Rin, sizeof(Record) * rLen);
    qsort(Rout, rLen, sizeof(Record), cmp_record_value);
}

// ---------- CPU Point Selection ----------
int CPU_PointSelection(Record* Rin, int rLen, int matchingKeyValue, Record **Rout, int numThread) {
    // Count matches first
    int count = 0;
    for (int i = 0; i < rLen; i++) {
        if (Rin[i].value == matchingKeyValue) count++;
    }
    if (count == 0) { *Rout = NULL; return 0; }
    *Rout = new Record[count];
    int idx = 0;
    for (int i = 0; i < rLen; i++) {
        if (Rin[i].value == matchingKeyValue) {
            (*Rout)[idx++] = Rin[i];
        }
    }
    return count;
}

// ---------- CPU Range Selection ----------
int CPU_RangeSelection(Record* Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record **Rout, int numThread) {
    int count = 0;
    for (int i = 0; i < rLen; i++) {
        if (Rin[i].value >= rangeSmallKey && Rin[i].value <= rangeLargeKey) count++;
    }
    if (count == 0) { *Rout = NULL; return 0; }
    *Rout = new Record[count];
    int idx = 0;
    for (int i = 0; i < rLen; i++) {
        if (Rin[i].value >= rangeSmallKey && Rin[i].value <= rangeLargeKey) {
            (*Rout)[idx++] = Rin[i];
        }
    }
    return count;
}

// ---------- CPU Projection ----------
void CPU_Projection(Record* baseTable, int rLen, Record* projTable, int pLen, int numThread) {
    for (int i = 0; i < pLen; i++) {
        int rid = projTable[i].rid;
        if (rid >= 0 && rid < rLen) {
            projTable[i].value = baseTable[rid].value;
        }
    }
}

// ---------- CPU Aggregation ----------
int CPU_AggMax(Record *R, int rLen, int numThread) {
    int maxVal = R[0].value;
    for (int i = 1; i < rLen; i++) {
        if (R[i].value > maxVal) maxVal = R[i].value;
    }
    return maxVal;
}

int CPU_AggMin(Record *R, int rLen, int numThread) {
    int minVal = R[0].value;
    for (int i = 1; i < rLen; i++) {
        if (R[i].value < minVal) minVal = R[i].value;
    }
    return minVal;
}

int CPU_AggSum(Record *R, int rLen, int numThread) {
    int sum = 0;
    for (int i = 0; i < rLen; i++) {
        sum += R[i].value;
    }
    return sum;
}

int CPU_AggAvg(Record *R, int rLen, int numThread) {
    if (rLen == 0) return 0;
    return CPU_AggSum(R, rLen, numThread) / rLen;
}

// ---------- CPU Group By ----------
int CPU_GroupBy(Record*R, int rLen, Record* Rout, int** d_startPos, int numThread) {
    // Sort first
    memcpy(Rout, R, sizeof(Record) * rLen);
    qsort(Rout, rLen, sizeof(Record), cmp_record_value);
    // Find group boundaries
    int numGroups = 1;
    for (int i = 1; i < rLen; i++) {
        if (Rout[i].value != Rout[i-1].value) numGroups++;
    }
    *d_startPos = new int[numGroups + 1];
    (*d_startPos)[0] = 0;
    int g = 1;
    for (int i = 1; i < rLen; i++) {
        if (Rout[i].value != Rout[i-1].value) {
            (*d_startPos)[g++] = i;
        }
    }
    return numGroups;
}

// ---------- CPU Agg After Group By ----------
void CPU_agg_max_afterGroupBy(Record *Rin, int rLen, int* d_startPos, int numGroups,
                              Record * RinAggOrig, int* d_aggResults, int numThread) {
    for (int g = 0; g < numGroups; g++) {
        int start = d_startPos[g];
        int end = (g + 1 < numGroups) ? d_startPos[g+1] : rLen;
        int maxVal = Rin[start].value;
        for (int i = start + 1; i < end; i++) {
            if (Rin[i].value > maxVal) maxVal = Rin[i].value;
        }
        d_aggResults[g] = maxVal;
    }
}

void CPU_agg_min_afterGroupBy(Record *Rin, int rLen, int* d_startPos, int numGroups,
                              Record * RinAggOrig, int* d_aggResults, int numThread) {
    for (int g = 0; g < numGroups; g++) {
        int start = d_startPos[g];
        int end = (g + 1 < numGroups) ? d_startPos[g+1] : rLen;
        int minVal = Rin[start].value;
        for (int i = start + 1; i < end; i++) {
            if (Rin[i].value < minVal) minVal = Rin[i].value;
        }
        d_aggResults[g] = minVal;
    }
}

void CPU_agg_sum_afterGroupBy(Record *Rin, int rLen, int* d_startPos, int numGroups,
                              Record * RinAggOrig, int* d_aggResults, int numThread) {
    for (int g = 0; g < numGroups; g++) {
        int start = d_startPos[g];
        int end = (g + 1 < numGroups) ? d_startPos[g+1] : rLen;
        int sum = 0;
        for (int i = start; i < end; i++) {
            sum += Rin[i].value;
        }
        d_aggResults[g] = sum;
    }
}

void CPU_agg_avg_afterGroupBy(Record *Rin, int rLen, int* d_startPos, int numGroups,
                              Record * RinAggOrig, int* d_aggResults, int numThread) {
    for (int g = 0; g < numGroups; g++) {
        int start = d_startPos[g];
        int end = (g + 1 < numGroups) ? d_startPos[g+1] : rLen;
        int sum = 0;
        int cnt = end - start;
        for (int i = start; i < end; i++) {
            sum += Rin[i].value;
        }
        d_aggResults[g] = (cnt > 0) ? (sum / cnt) : 0;
    }
}

// ---------- CPU Data Structures ----------
void CPU_BuildHashTable(Record* h_R, int rLen, int intBits, Bound *h_bound) {
    ::buildHashTable(h_R, rLen, intBits, h_bound);
}

void CPU_BuildTreeIndex(Record* R, int rLen, CC_CSSTree** tree) {
    // Sort the data first (CSS tree requires sorted input)
    Record* sorted = new Record[rLen];
    memcpy(sorted, R, sizeof(Record) * rLen);
    qsort(sorted, rLen, sizeof(Record), cmp_record_value);
    *tree = new CC_CSSTree(sorted, rLen, CSS_TREE_FANOUT);
}

// ---------- CPU Access Methods ----------
int CPU_HashSearch(Record* R, int rLen, Bound *h_bound, int intBits, Record *S, int sLen, Record** Rout, int numThread) {
    return ::HashSearch_omp(R, rLen, h_bound, intBits, S, sLen, Rout, numThread);
}

int CPU_TreeSearch(Record *R, int rLen, CC_CSSTree *tree, Record *S, int sLen, Record** Rout, int numThread) {
    // Simple index lookup for each S record
    int count = 0;
    Record* tempOut = new Record[sLen]; // worst case
    for (int i = 0; i < sLen; i++) {
        int loc = tree->search(S[i].value);
        if (loc >= 0 && loc < rLen && R[loc].value == S[i].value) {
            tempOut[count].rid = R[loc].rid;
            tempOut[count].value = R[loc].value;
            count++;
        }
    }
    if (count > 0) {
        *Rout = new Record[count];
        memcpy(*Rout, tempOut, sizeof(Record) * count);
    } else {
        *Rout = NULL;
    }
    delete[] tempOut;
    return count;
}

// ---------- CPU Joins ----------
int CPU_ninlj(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread) {
    // Nested loop join: R join S on R.value == S.value
    // First pass: count results
    int count = 0;
    #pragma omp parallel for reduction(+:count) num_threads(numThread)
    for (int i = 0; i < rLen; i++) {
        for (int j = 0; j < sLen; j++) {
            if (R[i].value == S[j].value) {
                count++;
            }
        }
    }
    if (count == 0) { *Rout = NULL; return 0; }

    // Second pass: materialize
    *Rout = new Record[count];
    int idx = 0;
    for (int i = 0; i < rLen; i++) {
        for (int j = 0; j < sLen; j++) {
            if (R[i].value == S[j].value) {
                (*Rout)[idx].rid = R[i].rid;
                (*Rout)[idx].value = R[i].value;
                idx++;
            }
        }
    }
    return count;
}

int CPU_inlj(Record *R, int rLen, CC_CSSTree *tree, Record *S, int sLen, Record** Rout, int numThread) {
    // Indexed nested loop join using CSS tree on R
    int count = 0;
    Record* tempOut = new Record[sLen]; // worst case: each S tuple matches once
    for (int i = 0; i < sLen; i++) {
        int loc = tree->search(S[i].value);
        if (loc >= 0 && loc < rLen && R[loc].value == S[i].value) {
            tempOut[count].rid = R[loc].rid;
            tempOut[count].value = S[i].value;
            count++;
        }
    }
    if (count > 0) {
        *Rout = new Record[count];
        memcpy(*Rout, tempOut, sizeof(Record) * count);
    } else {
        *Rout = NULL;
    }
    delete[] tempOut;
    return count;
}

int CPU_smj(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread) {
    // Sort-merge join: both R and S should be sorted
    Record* sortedR = new Record[rLen];
    Record* sortedS = new Record[sLen];
    memcpy(sortedR, R, sizeof(Record) * rLen);
    memcpy(sortedS, S, sizeof(Record) * sLen);
    qsort(sortedR, rLen, sizeof(Record), cmp_record_value);
    qsort(sortedS, sLen, sizeof(Record), cmp_record_value);

    // Count matches
    int count = 0;
    int i = 0, j = 0;
    while (i < rLen && j < sLen) {
        if (sortedR[i].value < sortedS[j].value) { i++; }
        else if (sortedR[i].value > sortedS[j].value) { j++; }
        else {
            // Match: count all pairs with same value
            int val = sortedR[i].value;
            int ri = i, rj = j;
            while (ri < rLen && sortedR[ri].value == val) ri++;
            while (rj < sLen && sortedS[rj].value == val) rj++;
            count += (ri - i) * (rj - j);
            i = ri; j = rj;
        }
    }

    if (count == 0) { *Rout = NULL; delete[] sortedR; delete[] sortedS; return 0; }

    *Rout = new Record[count];
    int idx = 0;
    i = 0; j = 0;
    while (i < rLen && j < sLen) {
        if (sortedR[i].value < sortedS[j].value) { i++; }
        else if (sortedR[i].value > sortedS[j].value) { j++; }
        else {
            int val = sortedR[i].value;
            int ri = i;
            while (ri < rLen && sortedR[ri].value == val) {
                int rj = j;
                while (rj < sLen && sortedS[rj].value == val) {
                    (*Rout)[idx].rid = sortedR[ri].rid;
                    (*Rout)[idx].value = val;
                    idx++;
                    rj++;
                }
                ri++;
            }
            i = ri;
            while (j < sLen && sortedS[j].value == val) j++;
        }
    }
    delete[] sortedR;
    delete[] sortedS;
    return count;
}

int CPU_MergeJoin(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread) {
    // Merge join (assumes R and S are already sorted)
    int count = 0;
    int i = 0, j = 0;
    while (i < rLen && j < sLen) {
        if (R[i].value < S[j].value) { i++; }
        else if (R[i].value > S[j].value) { j++; }
        else {
            int val = R[i].value;
            int ri = i, rj = j;
            while (ri < rLen && R[ri].value == val) ri++;
            while (rj < sLen && S[rj].value == val) rj++;
            count += (ri - i) * (rj - j);
            i = ri; j = rj;
        }
    }

    if (count == 0) { *Rout = NULL; return 0; }
    *Rout = new Record[count];
    int idx = 0;
    i = 0; j = 0;
    while (i < rLen && j < sLen) {
        if (R[i].value < S[j].value) { i++; }
        else if (R[i].value > S[j].value) { j++; }
        else {
            int val = R[i].value;
            int ri = i;
            while (ri < rLen && R[ri].value == val) {
                int rj = j;
                while (rj < sLen && S[rj].value == val) {
                    (*Rout)[idx].rid = R[ri].rid;
                    (*Rout)[idx].value = val;
                    idx++;
                    rj++;
                }
                ri++;
            }
            i = ri;
            while (j < sLen && S[j].value == val) j++;
        }
    }
    return count;
}

int CPU_hj(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread) {
    // Hash join: build hash table on R, probe with S
    if (rLen == 0 || sLen == 0) { *Rout = NULL; return 0; }

    // Simple hash join using chaining
    int hashBits = 16;
    int hashSize = (1 << hashBits);
    int hashMask = hashSize - 1;

    // Build phase: create hash table buckets
    int* next = new int[rLen];
    int* buckets = new int[hashSize];
    memset(buckets, -1, sizeof(int) * hashSize);
    for (int i = 0; i < rLen; i++) {
        int h = R[i].value & hashMask;
        next[i] = buckets[h];
        buckets[h] = i;
    }

    // Probe phase: count
    int count = 0;
    for (int j = 0; j < sLen; j++) {
        int h = S[j].value & hashMask;
        int idx = buckets[h];
        while (idx >= 0) {
            if (R[idx].value == S[j].value) count++;
            idx = next[idx];
        }
    }

    if (count == 0) { *Rout = NULL; delete[] next; delete[] buckets; return 0; }

    // Probe phase: materialize
    *Rout = new Record[count];
    int outIdx = 0;
    for (int j = 0; j < sLen; j++) {
        int h = S[j].value & hashMask;
        int idx = buckets[h];
        while (idx >= 0) {
            if (R[idx].value == S[j].value) {
                (*Rout)[outIdx].rid = R[idx].rid;
                (*Rout)[outIdx].value = R[idx].value;
                outIdx++;
            }
            idx = next[idx];
        }
    }
    delete[] next;
    delete[] buckets;
    return count;
}

// ---------- CPU Partition ----------
void CPU_Partition(Record *Rin, int rLen, int numPart, Record* Rout, int* startHist, int numThread) {
    if (numPart <= 0 || rLen <= 0) return;

    // Find min/max for range partitioning
    int minVal = Rin[0].value;
    int maxVal = Rin[0].value;
    for (int i = 1; i < rLen; i++) {
        if (Rin[i].value < minVal) minVal = Rin[i].value;
        if (Rin[i].value > maxVal) maxVal = Rin[i].value;
    }

    int range = maxVal - minVal + 1;
    if (range <= 0) range = 1;

    // Count per partition
    int* counts = new int[numPart];
    memset(counts, 0, sizeof(int) * numPart);
    for (int i = 0; i < rLen; i++) {
        int p = (int)((long long)(Rin[i].value - minVal) * numPart / range);
        if (p >= numPart) p = numPart - 1;
        if (p < 0) p = 0;
        counts[p]++;
    }

    // Compute start positions
    startHist[0] = 0;
    for (int p = 1; p < numPart; p++) {
        startHist[p] = startHist[p-1] + counts[p-1];
    }

    // Output
    int* pos = new int[numPart];
    memcpy(pos, startHist, sizeof(int) * numPart);
    for (int i = 0; i < rLen; i++) {
        int p = (int)((long long)(Rin[i].value - minVal) * numPart / range);
        if (p >= numPart) p = numPart - 1;
        if (p < 0) p = 0;
        Rout[pos[p]++] = Rin[i];
    }

    delete[] counts;
    delete[] pos;
}

// ---------- CL_QuickSort (host pointer version) ----------
// Wraps a CPU-side quicksort since the shared library doesn't export this
void CL_QuickSort(Record* h_Rin, int rLen, Record* h_Rout, int CPU_GPU) {
    memcpy(h_Rout, h_Rin, sizeof(Record) * rLen);
    qsort(h_Rout, rLen, sizeof(Record), cmp_record_value);
}

// ---------- CL_bitonicSort (host pointer version) ----------
void CL_bitonicSort(Record* h_Rin, int rLen, Record* h_Rout,
                    int numThreadPB, int numBlock, int CPU_GPU) {
    // Fallback to qsort
    memcpy(h_Rout, h_Rin, sizeof(Record) * rLen);
    qsort(h_Rout, rLen, sizeof(Record), cmp_record_value);
}

// ---------- CL_Partition (host pointer version) ----------
void CL_Partition(Record* h_Rin, int rLen, int numPart, Record* d_Rout, int* d_startHist,
                  int numThreadPB, int numBlock, int CPU_GPU) {
    CPU_Partition(h_Rin, rLen, numPart, d_Rout, d_startHist, 1);
}

// ---------- CL_BuildHashTable (host pointer version) ----------
void CL_BuildHashTable(Record* h_R, int rLen, int intBits, Bound* h_bound) {
    buildHashTable(h_R, rLen, intBits, h_bound);
}

// ---------- CL_BuildTreeIndex (host pointer version) ----------
int CL_BuildTreeIndex(Record* h_Rin, int rLen, CUDA_CSSTree** tree, int CPU_GPU) {
    // Stub: CUDA_CSSTree uses cl_mem in TonyLib version, 
    // can't easily create one without OpenCL context
    fprintf(stderr, "CL_BuildTreeIndex: not yet implemented for host-only path\n");
    *tree = NULL;
    return 0;
}

// ---------- CL_HashSearch (host pointer version) ----------
int CL_HashSearch(Record* h_R, int rLen, Bound* h_bound, int inBits, Record* h_S, int sLen, Record** h_Rout,
                  int numThread, int CPU_GPU) {
    return HashSearch_omp(h_R, rLen, h_bound, inBits, h_S, sLen, h_Rout, numThread);
}

// ---------- CL_TreeSearch (host pointer version) ----------
int CL_TreeSearch(Record* h_Rin, int rLen, CUDA_CSSTree* tree, Record* h_Sin, int sLen, Record** h_Rout, int CPU_GPU) {
    fprintf(stderr, "CL_TreeSearch: not yet implemented for host-only path\n");
    *h_Rout = NULL;
    return 0;
}

// ---------- resetGPU ----------
void resetGPU() {
    // Stub: GPU reset handled by OpenCL runtime
}

// ---------- CPU_Sum (for testing) ----------
int CPU_Sum(int a, int b) {
    return a + b;
}

void testDLL(int argc, char **argv) {
    printf("testDLL: stub\n");
}
