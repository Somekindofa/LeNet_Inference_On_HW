#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <getopt.h>

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
    for(k=0; k<n; k++){
        for(i=0; i<p; i++){
            for(j=0; j<q; j++){
                // random floats bewteen -1 and 1
                float randomFloat = ((float)rand()/(float)(RAND_MAX)) * 2 - 1;
                M[k*p*q + i*q + j] = randomFloat;
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

void MatrixPrint3D(float *M, int n, int p, int q){
    int i, j, k;
    for(k=0; k<n; k++){
        printf("Channel %d\n", k);
        for(i=0; i<p; i++){
            for(j=0; j<q; j++){
                printf("%f ", M[k*p*q + i*q + j]);
            }
            printf("\n");
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

__global__ void cudaMatrixConvolve(float *raw, float *kernel, float *Mout, int n, int p, int kernelSize){
    // kernel is of size 6x5x5
    int k = blockIdx.x;
    int r = blockIdx.y;
    int c = blockIdx.z;

    float sum = 0;
    // compute the multiplication of the kernel and the input matrix
    for(int l=0; l<kernelSize; l++){
        for(int m=0; m<kernelSize; m++){
            sum += kernel[k*kernelSize*kernelSize + l*kernelSize + m] * raw[(r+l)*p + (c+m)];
        }
    }
    Mout[k*n*p + r*p + c] = sum; // assign the result to the output matrix
    printf("Mout[%d] = %f\n", k*n*p + r*p + c, Mout[k*n*p + r*p + c]);
}


int main(int argc, char *argv[]) {
    srand(time(NULL));

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int channels = atoi(argv[3]);
    int N = rows*cols;

    // Allocation
    /////////////////
    // clock_t start_time_CPU = clock(); // start time measure for CPU
    int kernel_size = 5;
    int N1 = rows - kernel_size + 1;
    printf("N1 = %d \n", N1);
    float* raw_data     = (float*)malloc(N*sizeof(float));
    float* C1_kernel    = (float*)malloc(channels*kernel_size*kernel_size*sizeof(float));
    float* C1_data      = (float*)malloc(channels*N1*N1*sizeof(float));
    float* S1_data      = (float*)malloc(channels*N1*(N1/4)*sizeof(float));
    //////////////////

    // Init
    /////////////////
    MatrixInit(raw_data, rows, cols);
    MatrixInit3D(C1_kernel, channels, kernel_size, kernel_size);
    MatrixPrint3D(C1_kernel, channels, kernel_size, kernel_size);
    MatrixPrint(raw_data, rows, cols);

    // GPU Functions
    /////////////////
    // clock_t start_time_GPU = clock(); 
    // start time measure for GPU

    float *d_raw, *d_C1_kernel, *d_C1_data, *d_S1_data;
    cudaMalloc((void**)&d_raw, sizeof(float)*N);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float)*channels*kernel_size*kernel_size);
    cudaMalloc((void**)&d_C1_data, sizeof(float)*channels*N1*N1);
    cudaMalloc((void**)&d_S1_data, sizeof(float)*channels*N1*(N1/4));

    cudaMemcpy(d_raw, raw_data, sizeof(float)*N, cudaMemcpyHostToDevice);         // RAM --> GPU
    cudaMemcpy(d_C1_kernel, C1_kernel,sizeof(float)*channels*kernel_size*kernel_size, cudaMemcpyHostToDevice);     // RAM --> GPU
    cudaMemcpy(d_C1_data, C1_data, sizeof(float)*channels*N1*N1, cudaMemcpyHostToDevice);      // RAM --> GPU
    cudaMemcpy(d_S1_data, S1_data, sizeof(float)*channels*N1*(N1/4), cudaMemcpyHostToDevice);      // RAM --> GPU

    dim3 dimGrid(channels, N1, N1);

    cudaMatrixConvolve<<<dimGrid, 1>>>(d_raw, d_C1_kernel, d_C1_data, rows, cols, kernel_size);
    cudaMemcpy(C1_data, d_C1_data, sizeof(float)*channels*N1*N1, cudaMemcpyDeviceToHost);      // GPU --> RAM

    cudaFree(d_raw);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    /////////////////


    // Print
    /////////////////
    // MatrixPrint(M1, N, P);
    // MatrixPrint(M2, N, P);
    MatrixPrint3D(C1_data, channels, N1, N1);
    // printf("Execution time CPU: %f\n", execution_time_CPU);
    // printf("Execution time GPU: %f\n", execution_time_GPU);

    free(raw_data);
    free(C1_kernel);
    free(C1_data);
    free(S1_data);
    /////////////////

    return 0;
}
