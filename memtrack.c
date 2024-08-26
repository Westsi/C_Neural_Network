#include "memtrack.h"
#include <stdlib.h>

void* allocated[MEMTRACK_SIZE];
int allocedCnt = 0;

void registerAllocated(void* alloced) {
    allocated[allocedCnt++] = alloced;
}

void freeAll() {
    for (int i=0;i<allocedCnt;i++) {
        free(allocated[i]);
    }
}