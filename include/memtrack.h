#ifndef MEMTRACK_H_
#define MEMTRACK_H_

extern void* allocated[65536];
extern int allocedCnt;

void registerAllocated(void* alloced);
void freeAll();

#endif