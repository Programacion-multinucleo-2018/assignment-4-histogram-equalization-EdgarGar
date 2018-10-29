#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.h"
#include <cuda_runtime.h>

using namespace std;

__global__ void equalizer_kernel(unsigned char* input, unsigned char* output, int width, int height, int grayWidthStep, int *h_s)
{
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const int gray_tid = yIndex * grayWidthStep + xIndex;

	if ((xIndex < width) && (yIndex < height))
	{
		output[gray_tid] = h_s[input[gray_tid]];
	}
}

//Creación de historgrama
__global__ void createHistogram_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, float grayImageSize, int *h_s) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int color_tid = yIndex * colorWidthStep + xIndex;
	int xyIndex = threadIdx.x + threadIdx.y * blockDim.x;

	__shared__ int temp[256];

	if(xyIndex < 256) {
		temp[xyIndex] = 0;
	}
	__syncthreads();

	if(xIndex < width && yIndex < height) {
		atomicAdd(&temp[input[color_tid]], 1);
	}
	__syncthreads();

	if(xyIndex < 256) {
		atomicAdd(&h_s[xyIndex], temp[xyIndex]);
	}
}

__global__ void normalizeHistogram(unsigned char* input, unsigned char* output, float grayImageSize, int *h_s) {
	int xyIndex = threadIdx.x + threadIdx.y * blockDim.x;

	__shared__ int temporal[256];

	if(xyIndex < 256 && blockIdx.x == 0 && blockIdx.y == 0) {
		temporal[xyIndex] = 0;
		temporal[xyIndex] = h_s[xyIndex];
		__syncthreads();

		unsigned int normVar = 0;
		for(int i = 0; i <= xyIndex; i++) {
			normVar += temporal[i];
		}
		h_s[xyIndex] = normVar*255/grayImageSize;
	}
}

void equalizer(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	size_t grayBytes = output.step * output.rows;
	float grayImageSize = input.rows * input.cols;

	unsigned char *d_input, *d_output;


	int * h_s ;
	int * temp;
	size_t histogramBytes = sizeof(int) * 256;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, grayBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");
	//Malloc al histograma
	SAFE_CALL(cudaMalloc(&h_s, histogramBytes), "CUDA Malloc failed");
	SAFE_CALL(cudaMalloc(&temp, histogramBytes), "CUDA Malloc failed");


	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), grayBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), grayBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	int xBlock = 16;
	int yBlock = 16;
	// Specify a reasonable block size
	const dim3 block(xBlock, yBlock);

	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("equalizer_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	auto start_cpu =  chrono::high_resolution_clock::now();
	createHistogram_kernel <<<grid, block>>>(d_input, d_output, input.cols, input.rows, input.step, grayImageSize, h_s);
	normalizeHistogram <<<grid, block>>>(d_input, d_output, grayImageSize, h_s);
	equalizer_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, input.step, h_s);
	auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("La cantidad de tiempo que se tarda cada ejecucion es alrededor de: %f ms con bloque de %d y %d\n", duration_ms.count(), xBlock, yBlock);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
	SAFE_CALL(cudaFree(h_s), "CUDA Free FAiled");
}

int main(int argc, char *argv[])
{
	string imagePath;

	if(argc < 2)
		imagePath = "Images/dog2.jpeg";
  	else
  		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat temp(input.rows, input.cols, CV_8UC1);
	cv::Mat output(input.rows, input.cols, CV_8UC1);

	//Cambiamos el input a un gray
	cv::cvtColor(input, temp, CV_BGR2GRAY);

	//Call the wrapper function
	equalizer(temp, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	cv::resizeWindow("Input", 800, 600);
	cv::resizeWindow("Output", 800, 600);

	//Show the input and output
	imshow("Input", temp);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
