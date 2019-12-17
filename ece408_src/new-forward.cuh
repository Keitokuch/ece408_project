#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define H_out (H - K + 1)
#define W_out (W - K + 1)
#define HALO_WIDTH (TILE_WIDTH + 4)

// #define ORIGINAL
// #define UNROLL_EXPLICIT
#define UNROLL_IMPLICIT
// #define CONSTANT
// #define SHARED

namespace mxnet
{
namespace op
{

#ifdef CONSTANT
__constant__ float deviceKernel[10000];
#endif

#ifdef ORIGINAL
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#ifdef CONSTANT
#define k4d(i3, i2, i1, i0) deviceKernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#else
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#endif


    int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int W_grid = ceil(1.0*W_out / TILE_WIDTH);

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
    float acc = 0;
    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += x4d(b,c,h+p,w+q) * k4d(m,c,p,q);
                }
            }
        }
        y4d(b,m,h,w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}
#endif


#ifdef UNROLL_EXPLICIT
#define CUDA_MAX_NUM_THREADS 1024
#define BATCH_SIZE 1000
__global__ void unroll_input(float *x_unroll, const float *x, const int b_index,
    const int B, const int M, const int C, const int H, const int W, const int K) {
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll4d(i2, i1, i0) x_unroll[(i2) * (H_out * W_out * K * K * C) + (i1) * (H_out * W_out) + i0]

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int W_unroll = H_out * W_out;
    int b = b_index * BATCH_SIZE + blockIdx.y;
    if (t < C * W_unroll) {
        int c = t / W_unroll;
        int w_unroll = t % W_unroll;
        int h = w_unroll / W_out;
        int w = w_unroll % W_out;
        int h_base = c * K * K;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = h_base + p * K + q;
                x_unroll4d(blockIdx.y, h_unroll, w_unroll) = x4d(b, c, h + p, w + q);
            }
        }
    }
#undef x4d
#undef x_unroll4d
}

__global__ void forward_kernel(const float *k_unroll, const float *x_unroll, float *y, const int b_index,
    const int B, const int M, const int C, const int H, const int W, const int K) {
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int numARows = M;
    int numAColumns = C * K * K;
    int numBRows = numAColumns;
    int numBColumns = H_out * W_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int m = Row;
    int h = Col / W_out;
    int w = Col % W_out;
    int b = b_index * BATCH_SIZE + blockIdx.z;
    int b_base = blockIdx.z * (H_out * W_out * K * K * C);

    float Pvalue = 0;
    for (int i = 0; i < ceil(numAColumns/float(TILE_WIDTH)); i++) {
        if (Row < numARows && (i*TILE_WIDTH+tx) < numAColumns)
            subTileM[ty][tx] = k_unroll[Row*numAColumns+i*TILE_WIDTH+tx];
        else
            subTileM[ty][tx] = 0;
        if ((i*TILE_WIDTH+ty) < numBRows && Col < numBColumns)
            subTileN[ty][tx] = x_unroll[b_base+(i*TILE_WIDTH+ty)*numBColumns+Col];
        else
            subTileN[ty][tx] = 0;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            Pvalue += subTileM[ty][k] * subTileN[k][tx];
        __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
        y4d(b, m, h, w) = Pvalue;

#undef y4d
}
#endif // #ifdef UNROLL_EXPLICIT



#ifdef UNROLL_IMPLICIT
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int numARows = M;
    int numAColumns = C * K * K;
    int numBRows = numAColumns;
    int numBColumns = H_out * W_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int b = blockIdx.z;
    int m = Row;
    int h = Col / W_out;
    int w = Col % W_out;

    float Pvalue = 0;
    for (int i = 0; i < ceil(numAColumns/float(TILE_WIDTH)); i++) {
        int temp_col = i * TILE_WIDTH + tx, temp_row = i * TILE_WIDTH + ty;
        int K_m = Row;
        int K_c = temp_col / (K * K);
        int K_h = (temp_col % (K * K)) / K, K_w = (temp_col % (K * K)) % K;
        if (Row < numARows && temp_col < numAColumns)
            subTileM[ty][tx] = k4d(K_m, K_c, K_h, K_w);
        else
            subTileM[ty][tx] = 0;

        int X_b = b;
        int X_c = temp_row / (K * K);
        int X_p = temp_row % (K * K) / K, X_q = (temp_row % (K * K)) % K;
        int X_h = Col / W_out, X_w = Col % W_out;
        if (temp_row < numBRows && Col < numBColumns)
            subTileN[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
        else
            subTileN[ty][tx] = 0;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            Pvalue += subTileM[ty][k] * subTileN[k][tx];
        __syncthreads();
    }

    if (Row < numCRows && Col < numCColumns)
        y4d(b, m, h, w) = Pvalue;

#undef y4d
#undef x4d
#undef k4d
}
#endif // #ifdef UNROLL_IMPLICIT



#ifdef SHARED
__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /* __shared__ float X_ds[12][TILE_WIDTH + 5 - 1][TILE_WIDTH + 5 - 1]; */
    __shared__ float X_ds[12 * HALO_WIDTH * HALO_WIDTH];
    int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int W_grid = ceil(1.0*W_out / TILE_WIDTH);


#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#ifdef CONSTANT
#define k4d(i3, i2, i1, i0) deviceKernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#else
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#endif

    int n, m, h, w, c, p, q;
    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    #pragma unroll
    for (c = 0; c < C; ++c) {
        if (h < H && w < W) {
            /* X_ds[c][threadIdx.y][threadIdx.x] = x4d(n, c, h, w); */
            X_ds[c * HALO_WIDTH * HALO_WIDTH + threadIdx.y * HALO_WIDTH + threadIdx.x] = x4d(n, c, h, w);
        } else {
            /* X_ds[c][threadIdx.y][threadIdx.x] = 0.0f; */
            X_ds[c * HALO_WIDTH * HALO_WIDTH + threadIdx.y * HALO_WIDTH + threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

    if (threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH && h < H_out && w < W_out) {
        float acc = 0;
        #pragma unroll
        for (c = 0; c < C; c++)
            #pragma unroll
            for (p = 0; p < K; ++p)
                #pragma unroll
                for (q = 0; q < K; ++q)
                    /* acc += X_ds[c][threadIdx.y + p][threadIdx.x + q] * k4d(m, c, p, q); */
                    acc += X_ds[c*HALO_WIDTH*HALO_WIDTH + (threadIdx.y+p)*HALO_WIDTH + (threadIdx.x+q)] * k4d(m, c, p, q);
        y4d(n, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}
#endif // #ifdef SHARED

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

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    /* printf("B = %d, M = %d, C = %d, H = %d, W = %d, K = %d\n", B, M, C, H, W, K); */
    cudaStream_t s = 0;


#ifdef CONSTANT
    cudaMemcpyToSymbol(deviceKernel, w.dptr_, M * C * K * K * sizeof(float));
#endif

    /* --------------------------- RUN KERNEL ------------------------------ */

#ifdef ORIGINAL
    /* ------------------------- Original ------------------------ */
    int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int W_grid = ceil(1.0*W_out / TILE_WIDTH);
    int Z = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);
    forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
#endif /* #ifdef ORIGINAL */


#ifdef UNROLL_EXPLICIT
    float *x_unroll;
    cudaMalloc((void **)&x_unroll, BATCH_SIZE*H_out*W_out*K*K*C*sizeof(float));

    // Set the kernel dimensions
    dim3 gridDim_1(ceil(C*H_out*W_out/float(CUDA_MAX_NUM_THREADS)), BATCH_SIZE, 1);
    dim3 blockDim_1(CUDA_MAX_NUM_THREADS, 1, 1);
    dim3 gridDim_2(ceil(H_out*W_out/float(TILE_WIDTH)), ceil(M/float(TILE_WIDTH)), BATCH_SIZE);
    dim3 blockDim_2(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    for (int b_index = 0; b_index < B / BATCH_SIZE; b_index++) {
        unroll_input<<<gridDim_1, blockDim_1>>>(x_unroll, x.dptr_, b_index, B, M, C, H, W, K);
        forward_kernel<<<gridDim_2, blockDim_2>>>(w.dptr_, x_unroll, y.dptr_, b_index, B, M, C, H, W, K);
    }
#endif /* #ifdef UNROLL_EXPLICIT  */


#ifdef UNROLL_IMPLICIT
    // Set the kernel dimensions
    dim3 gridDim_1(ceil(H_out*W_out/float(TILE_WIDTH)), ceil(M/float(TILE_WIDTH)), B);
    dim3 blockDim_1(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    forward_kernel<<<gridDim_1, blockDim_1>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
#endif /* #ifdef UNROLL_IMPLICIT  */


#ifdef SHARED
    int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int W_grid = ceil(1.0*W_out / TILE_WIDTH);
    int Z = H_grid * W_grid;
    /* ----------------------- SHARED --------------------- */
    int blockWidth = TILE_WIDTH + K - 1;
    dim3 blockDim_s(blockWidth, blockWidth, 1);
    dim3 gridDim_s(B, M, Z);
    forward_kernel<<<gridDim_s, blockDim_s, 0, s>>>(y.dptr_, x.dptr_, w.dptr_, B,M,C,H,W,K);
#endif /* #ifdef SHARED */

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

#ifdef UNROLL
    cudaFree(x_unroll);
#endif
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
