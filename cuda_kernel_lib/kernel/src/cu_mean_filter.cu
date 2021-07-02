
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cu_mean_filter.h"
#include <stdio.h>
#include <stdint.h>
#define BLOCK_SIZE 16

__global__ void mean_filter_kernel(uint8_t *input, uint8_t *output,
                                   uint32_t width, uint8_t height,
                                   uint32_t window)
{
    uint32_t window_radis = window >> 1;
    uint32_t block_offset = BLOCK_SIZE - 2 * window_radis;
    __shared__ uint8_t shared_buf[BLOCK_SIZE][BLOCK_SIZE];
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t input_idx = bx * block_offset + tx + (by * block_offset + ty) * width;
    shared_buf[ty][tx] = input[input_idx];
    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    
    if (tx < window_radis ||
        tx >= (BLOCK_SIZE - window_radis) ||
        ty < window_radis ||
        ty >= (BLOCK_SIZE - window_radis))
    {
        return;
    }

    uint32_t i, j, sum = 0;
    uint8_t *ptr_filter_adder = &shared_buf[ty - window_radis][tx - window_radis];
    for (i = 0; i < window; i++)
    {
        for (j = 0; j < window; j++)
        {
            uint8_t adder = ptr_filter_adder[i * BLOCK_SIZE + j];
            sum += *ptr_filter_adder;
        }
    }
    output[input_idx] = sum / (window * window);
}

int32_t cuda_mean_filter(uint8_t *input, uint8_t *output,
                         uint32_t width, uint32_t height,
                         uint32_t window)
{
    uint8_t *cu_input_buf = NULL;
    uint8_t *cu_output_buf = NULL;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&cu_input_buf, width * height * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&cu_output_buf, width * height * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(cu_input_buf, input, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    uint32_t window_radis = window >> 1;
    uint32_t block_dim_x = (width + (BLOCK_SIZE - 2 * window_radis)) / (BLOCK_SIZE - 2 * window_radis);
    uint32_t block_dim_y = (height + (BLOCK_SIZE - 2 * window_radis)) / (BLOCK_SIZE - 2 * window_radis);
    dim3 grid(block_dim_x, block_dim_y);

    mean_filter_kernel<<<grid, threads>>>(cu_input_buf, cu_output_buf, width, height, window);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return -1;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, cu_output_buf, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return -1;
    }

    return 0;
}

#if 0
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
#endif