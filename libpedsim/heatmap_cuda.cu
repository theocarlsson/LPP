#include "ped_model.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "heatmap_cuda.h"
#include <cstdlib>
#include <iostream>
#include <chrono>

using namespace Ped;

#define BLOCK_SIZE 32

static cudaStream_t stream;
static bool initialized = false;
static int  *d_heatmap  = nullptr;
static int  *d_scaled   = nullptr;
static int  *d_blurred  = nullptr;
static int  *d_agentX   = nullptr;
static int  *d_agentY   = nullptr;

// -------------------- CUDA Kernels --------------------

__global__ void fadeKernel(int* heatmap, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size) return;
    int idx = y * size + x;
    heatmap[idx] = (int)roundf(heatmap[idx] * 0.8f);
}

__global__ void agentHeatKernel(int* heatmap, int* agentX, int* agentY,
                                int numAgents, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAgents) return;
    int x = agentX[i];
    int y = agentY[i];
    if (x < 0 || x >= size || y < 0 || y >= size) return;
    atomicAdd(&heatmap[y * size + x], 40);
}

__global__ void clampKernel(int* heatmap, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size) return;
    int idx = y * size + x;
    if (heatmap[idx] > 255) heatmap[idx] = 255;
}

__global__ void scaleKernel(int* heatmap, int* scaled, int size,
                            int scaledSize, int cellsize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= scaledSize || y >= scaledSize) return;
    int srcX = x / cellsize;
    int srcY = y / cellsize;
    scaled[y * scaledSize + x] = heatmap[srcY * size + srcX];
}

__global__ void blurKernel(int* scaled, int* blurred, int size) {
    __shared__ int tile[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;
    int sharedX = tx + 2;
    int sharedY = ty + 2;

    if (x < size && y < size)
        tile[sharedY][sharedX] = scaled[y * size + x];

    if (tx < 2 && x >= 2) tile[sharedY][tx] = scaled[y * size + x - 2];
    if (tx >= BLOCK_SIZE - 2 && x + 2 < size) tile[sharedY][sharedX + 2] = scaled[y * size + x + 2];
    if (ty < 2 && y >= 2) tile[ty][sharedX] = scaled[(y - 2) * size + x];
    if (ty >= BLOCK_SIZE - 2 && y + 2 < size) tile[sharedY + 2][sharedX] = scaled[(y + 2) * size + x];

    __syncthreads();

    if (x < 2 || y < 2 || x >= size - 2 || y >= size - 2) return;

    const int w[5][5] = {
        {1,4,7,4,1},
        {4,16,26,16,4},
        {7,26,41,26,7},
        {4,16,26,16,4},
        {1,4,7,4,1}
    };

    int sum = 0;
    for (int ky = -2; ky <= 2; ky++)
        for (int kx = -2; kx <= 2; kx++)
            sum += w[ky+2][kx+2] * tile[sharedY + ky][sharedX + kx];

    blurred[y * size + x] = 0x00FF0000 | ((sum / 273) << 24);
}

// -------------------- Async launch (no blocking) --------------------

void Model::setupHeatmapCuda()
{
    // Regular malloc for heatmap (uploaded to GPU once)
    int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
    heatmap = (int**)malloc(SIZE*sizeof(int*));
    for (int i = 0; i < SIZE; i++)
        heatmap[i] = hm + SIZE*i;

    // Pinned memory for blurred_heatmap - makes D2H truly async
    int *bhm;
    cudaMallocHost(&bhm, SCALED_SIZE*SCALED_SIZE*sizeof(int));
    blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
    for (int i = 0; i < SCALED_SIZE; i++)
        blurred_heatmap[i] = bhm + SCALED_SIZE*i;

    // scaled_heatmap not used in CUDA path
    scaled_heatmap = nullptr;
}

void Model::updateHeatmapCudaAsync(bool needsResult)
{
    int numAgents = (int)agents.size();

    if (!initialized) {
        cudaStreamCreate(&stream);
        cudaMalloc(&d_heatmap, SIZE * SIZE * sizeof(int));
        cudaMalloc(&d_scaled,  SCALED_SIZE * SCALED_SIZE * sizeof(int));
        cudaMalloc(&d_blurred, SCALED_SIZE * SCALED_SIZE * sizeof(int));
        cudaMalloc(&d_agentX,  numAgents * sizeof(int));
        cudaMalloc(&d_agentY,  numAgents * sizeof(int));
        cudaMemcpyAsync(d_heatmap, heatmap[0], SIZE * SIZE * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        cudaEventCreate((cudaEvent_t*)&ev0);
        cudaEventCreate((cudaEvent_t*)&ev1);
        cudaEventCreate((cudaEvent_t*)&ev2);
        cudaEventCreate((cudaEvent_t*)&ev3);
        cudaEventCreate((cudaEvent_t*)&ev4);
        cudaEventCreate((cudaEvent_t*)&ev5);

        initialized = true;
    }

    // h_agentX/h_agentY already filled by tick_cuda_impl
    cudaMemcpyAsync(d_agentX, h_agentX, numAgents * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_agentY, h_agentY, numAgents * sizeof(int), cudaMemcpyHostToDevice, stream);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid      ((SIZE        + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (SIZE        + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridScaled((SCALED_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (SCALED_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord((cudaEvent_t)ev0, stream);
    fadeKernel<<<grid, block, 0, stream>>>(d_heatmap, SIZE);
    cudaEventRecord((cudaEvent_t)ev1, stream);

    agentHeatKernel<<<(numAgents + 255) / 256, 256, 0, stream>>>(
        d_heatmap, d_agentX, d_agentY, numAgents, SIZE);
    clampKernel<<<grid, block, 0, stream>>>(d_heatmap, SIZE);
    cudaEventRecord((cudaEvent_t)ev2, stream);

    scaleKernel<<<gridScaled, block, 0, stream>>>(
        d_heatmap, d_scaled, SIZE, SCALED_SIZE, CELLSIZE);
    cudaEventRecord((cudaEvent_t)ev3, stream);

    blurKernel<<<gridScaled, block, 0, stream>>>(d_scaled, d_blurred, SCALED_SIZE);
    cudaEventRecord((cudaEvent_t)ev4, stream);

    if (needsResult)
        cudaMemcpyAsync(blurred_heatmap[0], d_blurred,
                        SCALED_SIZE * SCALED_SIZE * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
    cudaEventRecord((cudaEvent_t)ev5, stream);
}

void Model::syncHeatmapCuda()
{
    cudaStreamSynchronize(stream);

    static int count = 0;
    if (count++ < 5) {
        float f, a, s, b, d;
        cudaEventElapsedTime(&f, (cudaEvent_t)ev0, (cudaEvent_t)ev1);
        cudaEventElapsedTime(&a, (cudaEvent_t)ev1, (cudaEvent_t)ev2);
        cudaEventElapsedTime(&s, (cudaEvent_t)ev2, (cudaEvent_t)ev3);
        cudaEventElapsedTime(&b, (cudaEvent_t)ev3, (cudaEvent_t)ev4);
        cudaEventElapsedTime(&d, (cudaEvent_t)ev4, (cudaEvent_t)ev5);
        std::cerr << "  fade       : " << f << " ms\n"
                  << "  agent+clamp: " << a << " ms\n"
                  << "  scale      : " << s << " ms\n"
                  << "  blur       : " << b << " ms\n"
                  << "  D2H copy   : " << d << " ms\n"
                  << "  total GPU  : " << (f+a+s+b) << " ms\n" << std::flush;
    }
}