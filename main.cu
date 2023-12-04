#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4
#define P 4

void MatrixInit(float *M, int n, int p){
    srand(time(NULL));
    float r = (float)rand() / (float)RAND_MAX;
    printf("%f\n", r);

}