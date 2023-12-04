#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void MatrixInit(float *M, int n, int p){
    int i, j;
    for(i=0; i<n; i++){
        for(j=0; j<p; j++){
            // random floats bewteen -1 and 1
            float randomFloat = ((float)rand()/(float)(RAND_MAX)) * 2 - 1;
            M[i*p+j] = randomFloat;
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

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<n && j<p){
        Mout[i*p+j] = M1[i*p+j] + M2[i*p+j];
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < n && j < n){
        Mout[i*n+j] = 0;
        float sum = 0;
        for(int k=0; k<n; k++){
            sum += M1[i*n+k] * M2[k*n+j];
        }
        Mout[i*n+j] = sum;
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    int N, P;

    if (argc != 3) {
        printf("Usage: %s <number of rows (N)> <number of columns (P)>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    P = atoi(argv[2]);

    // Allocation
    /////////////////
    clock_t start_time_CPU = clock(); // start time measure for CPU

    float* M1   =(float*)malloc(N*P*sizeof(float));
    float* M2   =(float*)malloc(N*P*sizeof(float));
    float* Mout =(float*)malloc(N*P*sizeof(float));
    //////////////////

    // Init
    /////////////////
    MatrixInit(M1, N, P);
    MatrixInit(M2, N, P);
    /////////////////


    // CPU Functions
    /////////////////
    MatrixAdd(M1, M2, Mout, N, P);
    MatrixMult(M1, M2, Mout, N);

    clock_t end_time_CPU = clock(); // end time measure for CPU
    double execution_time_CPU = ((double) (end_time_CPU - start_time_CPU)) / CLOCKS_PER_SEC;
    /////////////////


    // GPU Functions
    /////////////////
    clock_t start_time_GPU = clock(); // start time measure for GPU

    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void**)&d_M1, sizeof(float)*N*P);
    cudaMalloc((void**)&d_M2, sizeof(float)*N*P);
    cudaMalloc((void**)&d_Mout, sizeof(float)*N*P);

    cudaMemcpy(d_M1, M1, sizeof(float)*N*P, cudaMemcpyHostToDevice);        // RAM --> GPU
    cudaMemcpy(d_M2, M2, sizeof(float)*N*P, cudaMemcpyHostToDevice);        // RAM --> GPU
    cudaMemcpy(d_Mout, Mout, sizeof(float)*N*P, cudaMemcpyHostToDevice);    // RAM --> GPU

    dim3 dimGrid(ceil(N/32.0), ceil(P/32.0));
    dim3 dimBlock(32, 32);


    cudaMatrixAdd<<<dimGrid, dimBlock>>>(d_M1, d_M2, d_Mout, N, P);
    cudaMemcpy(Mout, d_Mout, sizeof(float)*N*P, cudaMemcpyDeviceToHost);    // GPU --> RAM

    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    clock_t end_time_GPU = clock(); // end time measure for GPU
    double execution_time_GPU = ((double) (end_time_GPU - start_time_GPU)) / CLOCKS_PER_SEC;
    /////////////////


    // Print
    /////////////////
    // MatrixPrint(M1, N, P);
    // MatrixPrint(M2, N, P);
    // MatrixPrint(Mout, N, P);
    printf("Execution time CPU: %f\n", execution_time_CPU);
    printf("Execution time GPU: %f\n", execution_time_GPU);
    /////////////////

    return 0;
}
