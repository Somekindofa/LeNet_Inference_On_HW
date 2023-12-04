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

void MatrixPrint3D(float *M, int n, int p, int d){
    int i, j, k;
    for(i=0; i<d; i++){
        printf("Matrix %d:\n", i+1);

        for(j=0; j<n; j++){
            for(k=0; k<p; k++){
                printf("%f ", M[i*p*p + j*p + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
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
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if(i < n && j < p && k < kernelSize){
        Mout[k*n*p + i*p + j] = 0;

        // compute the multiplication of the kernel and the input matrix
        float sum = 0;
        for(int q=0; q<kernelSize; q++){
            for(int l=0; l<kernelSize; l++){
                for(int m=0; m<kernelSize; m++){
                    sum += kernel[q*kernelSize*kernelSize + l*kernelSize + m] * raw[(i+q)*p*p + (j+l)*p + m];
                }
            }
        }
        Mout[k*n*p + i*p + j] = sum; // assign the result to the output matrix
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    int N, P;
    int input[2] = {0, 0};
    int kernel[3] = {0, 0, 0};

    if (argc != 3) {
        printf("Usage: %s <number of rows (N)> <number of columns (P)>\n", argv[0]);
        return 1;
    }

    // int opt;
    // while ((opt = getopt(argc, argv, "i:k:")) != -1) {
    //     switch (opt) {
    //         case 'i':
    //             sscanf(optarg, "(%d, %d)", &input[0], &input[1]);
    //             break;
    //         case 'k':
    //             sscanf(optarg, "(%d, %d, %d)", &kernel[0], &kernel[1], &kernel[2]);
    //             break;
    //         default:
    //             fprintf(stderr, "Usage: %s -i (x,y) -k (x,y)\n", argv[0]);
    //             exit(EXIT_FAILURE);
    //     }
    // }

    N = atoi(argv[1]);
    P = atoi(argv[2]);

    // Allocation
    /////////////////
    // clock_t start_time_CPU = clock(); // start time measure for CPU
    int kernel_size = 5;
    float* raw_data     = (float*)malloc(N*P*sizeof(float));
    float* C1_kernel    = (float*)malloc(6*kernel_size*kernel_size*sizeof(float));
    float* C1_data      = (float*)malloc(6*28*28*sizeof(float));
    float* S1_data      = (float*)malloc(6*14*14*sizeof(float));
    //////////////////

    // Init
    /////////////////
    MatrixInit(raw_data, N, P);
    MatrixInit3D(C1_kernel, 6, kernel_size, kernel_size);

    // Initialize C1_data and S1_data to zeros
    cudaMemset(C1_data, 0, sizeof(float) * 6 * 28 * 28);
    cudaMemset(S1_data, 0, sizeof(float) * 6 * 14 * 14);
    /////////////////

    // GPU Functions
    /////////////////
    // clock_t start_time_GPU = clock(); // start time measure for GPU

    float *d_raw, *d_kernel, *d_C1_data, *d_S1_data;
    cudaMalloc((void**)&d_raw, sizeof(float)*N*P);
    cudaMalloc((void**)&d_kernel, sizeof(float)*6*5*5);
    cudaMalloc((void**)&d_C1_data, sizeof(float)*6*28*28);
    cudaMalloc((void**)&d_S1_data, sizeof(float)*6*14*14);

    cudaMemcpy(d_raw, raw_data, sizeof(float)*N*P, cudaMemcpyHostToDevice);         // RAM --> GPU
    cudaMemcpy(d_kernel, C1_kernel, sizeof(float)*N*P, cudaMemcpyHostToDevice);     // RAM --> GPU
    cudaMemcpy(d_C1_data, C1_data, sizeof(float)*N*P, cudaMemcpyHostToDevice);      // RAM --> GPU

    dim3 dimGrid(28,28);    // 28x28 blocks of 1 thread each (2D) SO 28x28 = 784 threads
    dim3 dimBlock(6, 1, 1); // 6 blocks of 1 thread each (1D) SO 6 threads

    cudaMatrixConvolve<<<dimGrid, dimBlock>>>(d_raw, d_kernel, d_C1_data, N, P, kernel_size);
    cudaMemcpy(C1_data, d_C1_data, sizeof(float)*6*28*28, cudaMemcpyDeviceToHost);      // GPU --> RAM

    cudaFree(d_raw);
    cudaFree(d_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    /////////////////


    // Print
    /////////////////
    // MatrixPrint(M1, N, P);
    // MatrixPrint(M2, N, P);
    MatrixPrint3D(C1_data, N, P, 6);
    // printf("Execution time CPU: %f\n", execution_time_CPU);
    // printf("Execution time GPU: %f\n", execution_time_GPU);
    /////////////////

    return 0;
}
