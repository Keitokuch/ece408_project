
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE 16
#define H_out (H-K+1)
#define W_out (W-K+1)

namespace mxnet
{
namespace op
{

// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     */

// // An example use of these macros:
// // float a = y4d(0,0,0,0)
// // y4d(0,0,0,0) = a
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int b = blockIdx.z;
//     int m = blockIdx.x;
//     int h = blockIdx.y / W_grid * TILE + threadIdx.y;
//     int w = blockIdx.y % W_grid * TILE + threadIdx.x;
//     float acc = 0;
//     if (h < H_out && w < W_out) {
// 	    for (int c = 0; c < C; c++) {
// 	        for (int p = 0; p < K; p++) {
// 	            for (int q = 0; q < K; q++) {
// 		            acc += x4d(b,c,h+p,w+q) * k4d(m,c,p,q);
// 	            }
// 	        }
// 	    }
// 	    y4d(b,m,h,w) = acc;
//     }

// #undef y4d
// #undef x4d
// #undef k4d
// }

__global__ void unroll_input(float *x_unroll, const float *x,
    const int B, const int M, const int C, const int H, const int W, const int K) {
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll4d(i2, i1, i0) x_unroll[(i2) * (H_out * W_out * K * K * C) + (i1) * (H_out * W_out) + i0]

    int w_unroll = blockIdx.x * blockDim.x + threadIdx.x;
    int h_unroll = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    int c = blockIdx.y;
    int h = w_unroll / W_out;
    int w = w_unroll % W_out;
    int p = threadIdx.y / K;
    int q = threadIdx.y % K;
    if (b < B && c < C && h + p < H && w + q < W)
        x_unroll4d(b, h_unroll, w_unroll) = x4d(b, c, h + p, w + q);

#undef x4d
#undef x_unroll4d
}

__global__ void unroll_kernel(float *k_unroll, const float *k,
    const int B, const int M, const int C, const int H, const int W, const int K) {
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k_unroll4d(i1, i0) k_unroll[(i1) * (C * K * K) + i0]

    int w_unroll = blockIdx.x * blockDim.x + threadIdx.x;
    int h_unroll = blockIdx.y * blockDim.y + threadIdx.y;
    int m = h_unroll;
    int c = blockIdx.x;
    int p = threadIdx.x / K;
    int q = threadIdx.x % K;
    if (m < M && c < C && p < K && q < K)
        k_unroll4d(h_unroll, w_unroll) = k4d(m, c, p, q);

#undef k4d
#undef k_unroll4d
}

__global__ void matrixMultiplyShared(const float *k_unroll, const float *x_unroll, float *y,
    const int B, const int M, const int C, const int H, const int W, const int K) {
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    __shared__ float subTileM[TILE][TILE];
    __shared__ float subTileN[TILE][TILE];

    int numARows = M;
    int numAColumns = C * K * K;
    int numBRows = numAColumns;
    int numBColumns = H_out * W_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE + ty;
    int Col = bx * TILE + tx;

    int b = blockIdx.z;
    int m = Row;
    int h = Col / W_out;
    int w = Col % W_out;

    float Pvalue = 0;
    for (int i = 0; i < ceil(numAColumns/float(TILE)); i++) {
        if (Row < numARows && (i*TILE+tx) < numAColumns)
            subTileM[ty][tx] = k_unroll[Row*numAColumns+i*TILE+tx];
        if (b < B && (i*TILE+ty) < numBRows && Col < numBColumns)
            subTileN[ty][tx] = x_unroll[b*numBRows*numBColumns+(i*TILE+ty)*numBColumns+Col];
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            if ((i*TILE+k) < numAColumns)
                Pvalue += subTileM[ty][k] * subTileN[k][tx];
        __syncthreads();
    }
    if (b < B && Row < numCRows && Col < numCColumns)
        y4d(b, m, h, w) = Pvalue;

#undef y4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";
    float *x_unroll;
    float *k_unroll;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    // printf("B = %d, M = %d, C = %d, H = %d, W = %d, K = %d\n", B, M, C, H, W, K);

    cudaMalloc((void **)&x_unroll, sizeof(float)*H_out*W_out*K*K*C);
    cudaMalloc((void **)&k_unroll, sizeof(float)*M*K*K*C);

    // Set the kernel dimensions
    dim3 gridDim_1(ceil(H_out*W_out/float(K*K)), C, B);
    dim3 blockDim_1(K*K, K*K, 1);
    dim3 gridDim_2(C, ceil(M/float(K*K)), 1);
    dim3 blockDim_2(K*K, K*K, 1);
    dim3 gridDim_3(ceil(M/float(TILE)), ceil(H_out*W_out/float(TILE)), B);
    dim3 blockDim_3(TILE, TILE, 1);
    cudaStream_t s = 0;

    // Call the kernel
    unroll_input<<<gridDim_1, blockDim_1>>>(x_unroll, x.dptr_, B, M, C, H, W, K);
    cudaDeviceSynchronize();
    unroll_kernel<<<gridDim_2, blockDim_2>>>(k_unroll, w.dptr_, B, M, C, H, W, K);
    cudaDeviceSynchronize();
    matrixMultiplyShared<<<gridDim_3, blockDim_3>>>(k_unroll, x_unroll, y.dptr_, B, M, C, H, W, K);
    // unroll_input<<<gridDim_1, blockDim_1, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    cudaFree(x_unroll);
    cudaFree(k_unroll);
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif