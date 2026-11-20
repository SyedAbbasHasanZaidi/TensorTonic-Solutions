#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // Store N element-wise results in C; this kernel returns no value.
    // Assuming A,B,C is allocated correctly 
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if(i<N){
        C[i] = A[i] + B[i];
    }

    return;
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256; // 256 threads in per block 
    int blocks = (N + threads - 1) / threads; 
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}