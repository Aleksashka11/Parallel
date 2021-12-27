#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_fp16.h>
#include <vector>
#include "Lab3_Parallel.cuh"
#include <limits>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() 
#else
#define __syncthreads()
#endif

using namespace std;


//first task
__global__ void getChannelPixels_cuda(const IN_TYPE* in, int width, int height, unsigned int* out)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int NUM_BINS = 3;
    // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // grid dimensions
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    // linear thread index within 2D block
    int t = threadIdx.x + threadIdx.y * blockDim.x;

    // total threads in 2D block
    int nt = blockDim.x * blockDim.y;

    // linear block index within 2D grid
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    // initialize temporary accumulation array in global memory
    unsigned int* gmem = out + g * NUM_BINS;
    for (int i = t; i < 3 * width*height; i += nt) gmem[i] = 0;
    // process pixels
    // updates our block's partial histogram in global memory
    cudaEventRecord(start);
    for (int col = x; col < width; col += nx)
        for (int row = y; row < height; row += ny) {
            unsigned int r = (unsigned int)(256 * in[row * width + col].x);
            unsigned int g = (unsigned int)(256 * in[row * width + col].y);
            unsigned int b = (unsigned int)(256 * in[row * width + col].z);
            atomicAdd(&gmem[NUM_BINS * 0 + r], 1);
            atomicAdd(&gmem[NUM_BINS * 1 + g], 1);
            atomicAdd(&gmem[NUM_BINS * 2 + b], 1);
        }
    cudaEventRecord(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Value of blue channel:" << atomicAdd.b << endl;
	cout << "Value of green channel:" << atomicAdd.g << endl;
	cout << "Value of red channel:" << atomicAdd.r << endl;
	cout << "Time spent to count pixels with Cuda: " << milliseconds*0.0001 << " microseconds" << endl;
}


//second task
__device__ float atomicMinf(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(val));
    }
    return __int_as_float(old);
}


__global__ void min_reduce(const float* const d_array, float* d_min,
    const size_t elements)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    extern __shared__ float shared[];

    float FLOAT_MIN = std::numeric_limits<float>::min();

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -FLOAT_MIN;

    while (gid < elements) {
        shared[tid] = std::min(shared[tid], d_array[gid]);
        gid += gridDim.x * blockDim.x;
    }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;  // 1
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && gid < elements)
            shared[tid] = std::min(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMinf(d_min, shared[0]);
	cudaEventRecord(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Minimum for blue channel: " << d_min << endl;
	cout << "Time spent to find minimum in Blue channel with Cuda: " << milliseconds*0.0001 << " microseconds" << endl;
}

//third task

//CUDA Error Checker
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

//Allocate filter kernel in constant memory on device
__constant__ int filter_kernel[] =
{
  1, 1, 1,
  1, 1, 1,
  1, 1, 1
};

//Set block size
#define BLOCK_SIZE 16


__global__ void conv_globalMem(unsigned char* input,
	unsigned char* output,
	int width,
	int height,
	int inputWidthStep,
	int outputWidthStep,
	int radius, int weight) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;


	//Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height)) {

		//Allocate values
		int3 acc = make_int3(0, 0, 0);
		int3 val = make_int3(0, 0, 0);

		int output_tid = yIndex * outputWidthStep + (3 * xIndex);

		for (int i = -radius; i <= radius; i++) {
			for (int j = -radius; j <= radius; j++) {

				//Skip violations (which will lead to zero by default
				if ((xIndex + i < 0) || (xIndex + i >= width) || (yIndex + j < 0) || (yIndex + j >= height)) continue;

				//Get kernel value
				int temp = filter_kernel[i + radius + (j + radius) * ((radius << 1) + 1)];

				//Location of colored pixel in input
				int input_tid = (yIndex + j) * inputWidthStep + (3 * (xIndex + i));

				//Fetch the three channel values
				const unsigned char blue = input[input_tid];
				const unsigned char green = input[input_tid + 1];
				const unsigned char red = input[input_tid + 2];


				val.x = int(blue) * temp;
				val.y = int(green) * temp;
				val.z = int(red) * temp;

				//Perform cumulative sum
				acc.x += val.x;
				acc.y += val.y;
				acc.z += val.z;
			}
		}

		acc.x = acc.x / weight;
		acc.y = acc.y / weight;
		acc.z = acc.z / weight;

		output[output_tid] = static_cast<unsigned char>(acc.x);
		output[output_tid + 1] = static_cast<unsigned char>(acc.y);
		output[output_tid + 2] = static_cast<unsigned char>(acc.z);

		cudaEventRecord(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cout << "Convo;ution for the Blue Channel: " << acc.x << endl;
		cout << "Convolution for the Green Channel: " << acc.y << endl;
		cout << "Convolution for the Red Channel: " << acc.z << endl;
		cout << "Time spent to calculate convolution with Cuda: " << milliseconds*0.0001 << " microseconds" << endl;

	}

} //end of convGlobal


/*Convolution Wrapper*/

void convolution(const cv::Mat& input, cv::Mat& output) {

	//Calculate the bytes to be transferred
	const int inputBytes = input.step * input.rows;
	const int outputBytes = output.step * output.rows;

	//Instantiate device pointers
	unsigned char* d_input, * d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");

	//Calculate required threads and gridSize size to cover the whole image

	//Specify a reasonable blockSize size
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); //16x16 threads = 256 thread per block

	//Calculate gridSize size to cover the whole image
	const dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x, (input.rows + blockSize.y - 1) / blockSize.y);

	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//Launch the convolution kernel
	int radius = 2;
	int weight = 256;

	conv_globalMem << <gridSize, blockSize >> > (d_input, d_output, input.cols, input.rows, input.step, output.step, radius, weight);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	//Copy back data from destination  device memory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

}