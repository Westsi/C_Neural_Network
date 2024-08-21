#include "memtrack.h"
#include <stdlib.h>

void* allocated[65536];
int allocedCnt = 0;

void registerAllocated(void* alloced) {
    allocated[allocedCnt++] = alloced;
}

void freeAll() {
    for (int i=0;i<allocedCnt;i++) {
        free(allocated[i]);
    }
}