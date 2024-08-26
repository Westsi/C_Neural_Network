#ifndef MEMTRACK_H_
#define MEMTRACK_H_

#define MEMTRACK_SIZE 16777216

extern void* allocated[MEMTRACK_SIZE];
extern int allocedCnt;

void registerAllocated(void* alloced);
void freeAll();

#endif