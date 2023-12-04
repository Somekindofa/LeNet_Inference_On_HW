#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Initialisez les valeurs de la matrice de façon aléatoire entre -1 et 1
void MatrixInit(float *M, int n, int p){
    cudaMalloc ((void **) &M, n*p*sizeof(float));
    int i, j;
    for(i=0; i<n; i++){
        for(j=0; j<p; j++){ // M is a pointer to a 2D array
            M[i*p+j] = ((float)rand()/(float)(RAND_MAX)) * 2 - 1; // Generate random number between -1 and 1
        }
    }
};









































void MatrixPrint(float *M, int n, int p){
    int i, j;
    for(i=0; i<n; i++){
        for(j=0; j<p; j++){
            printf("%f ", M[i*p+j]);
        }
        printf("\n");
    }
    printf("\n");
};

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i, j;
    for(i=0; i<n; i++){
        for(j=0; j<p; j++){
            Mout[i*p+j] = M1[i*p+j] + M2[i*p+j];
        }
    }
};


__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<n && j<p){
        Mout[i*p+j] = M1[i*p+j] + M2[i*p+j];
    }
};
void MatrixMult(float *M1, float *M2, float *Mout, int n){
    int i, j, k;
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            Mout[i*n+j] = 0;
            for(k=0; k<n; k++){
                Mout[i*n+j] += M1[i*n+k] * M2[k*n+j];
            }
        }
    }
};

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<n && j<n){
        Mout[i*n+j] = 0;
        for(int k=0; k<n; k++){
            Mout[i*n+j] += M1[i*n+k] * M2[k*n+j];
        }
    }
};
