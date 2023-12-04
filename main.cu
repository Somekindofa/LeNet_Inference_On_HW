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

void MatrixInit3D(float *M, int n, int p, int q){
    int i, j, k;
    for(i=0; i<n; i++){
        for(j=0; j<p; j++){
            for(k=0; k<q; k++){
                // random floats bewteen -1 and 1
                float randomFloat = ((float)rand()/(float)(RAND_MAX)) * 2 - 1;
                M[i*p*q + j*q + k] = randomFloat;
            }
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
};
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<n && j<p){
        Mout[i*p+j] = M1[i*p+j] + M2[i*p+j];
    }
};

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
};

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
};

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
    // clock_t start_time_CPU = clock(); // start time measure for CPU

    float* raw_data     = (float*)malloc(N*P*sizeof(float));
    float* C1_kernel    = (float*)malloc(6*5*5*sizeof(float));
    float* C1_data      = (float*)malloc(6*28*28*sizeof(float));
    float* S1_data      = (float*)malloc(6*14*14*sizeof(float));
    //////////////////

    // Init
    /////////////////
    MatrixInit(raw_data, N, P);
    MatrixInit(C1_kernel, N, P);
    float C1_data[6][28][28];
    float S1_data[6][14][14];

    // Initialize C1_data and S1_data to zeros
    cudaMemset(C1_data, 0, sizeof(float) * 6 * 28 * 28);
    cudaMemset(S1_data, 0, sizeof(float) * 6 * 14 * 14);
    /////////////////

    // GPU Functions
    /////////////////
    clock_t start_time_GPU = clock(); // start time measure for GPU

    float *d_raw, *d_kernel, *d_C1_data, *d_S1_data;
    cudaMalloc((void**)&d_raw, sizeof(float)*N*P);
    cudaMalloc((void**)&d_kernel, sizeof(float)*6*5*5);
    cudaMalloc((void**)&d_C1_data, sizeof(float)*6*28*28);
    cudaMalloc((void**)&d_S1_data, sizeof(float)*6*14*14);

    cudaMemcpy(d_raw, raw_data, sizeof(float)*N*P, cudaMemcpyHostToDevice);        // RAM --> GPU
    cudaMemcpy(d_kernel, C1_kernel, sizeof(float)*N*P, cudaMemcpyHostToDevice);        // RAM --> GPU
    cudaMemcpy(d_C1_data, C1_data, sizeof(float)*N*P, cudaMemcpyHostToDevice);    // RAM --> GPU

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
