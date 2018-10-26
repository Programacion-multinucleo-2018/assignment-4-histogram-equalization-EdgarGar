//Para este la realizacion de este codigo se utilizaron algunas partes de la tarea
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

//
__global__ void saveImage(unsigned char* input, unsigned char* output, int width, int height, int step, int *tem)
{
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const int tid = yIndex * step + xIndex;

	if ((xIndex < width) && (yIndex < height))
	{
		output[tid] = temp[input[tid]];
	}
}

//Funcion para generar el histograma
__global__ void generateHistogram(unsigned char* input, unsigned char* output, int width, int height, int step, float imgSize, int *temp) {
	//Codigo de arriba de la funcion pasada
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int color_tid = yIndex * step + xIndex;

  __shared__ int temp[256];

	int xyIndex = threadIdx.x + threadIdx.y * blockDim.x;

	if(xyIndex < 256) {
		temp[xyIndex] = 0;
	}
  //Creacion de histograma
	__syncthreads();

	if(xIndex < width && yIndex < height) {
		atomicAdd(&temp[input[color_tid]], 1);
	}

	__syncthreads();

	if(xyIndex < 256) {
		atomicAdd(&temp[xyIndex], temp[xyIndex]);
	}
}

__global__ void equalizationPhases(unsigned char* input, unsigned char* output, float imgSize, int *temp) {
	int xyIndex = threadIdx.x + threadIdx.y * blockDim.x;
	__shared__ int pix[256];

	if(xyIndex < 256 && blockIdx.x == 0 && blockIdx.y == 0) {
    pix[xyIndex] = 0;
		pix[xyIndex] = temp[xyIndex];
		__syncthreads();
		unsigned int normVar = 0;
		for(int i = 0; i <= xyIndex; i++) {
			normVar += pix[i];
		}
		temp[xyIndex] = normVar/255;
	}
}

//Extracccion de codigo de la actividad 2
void equalizer(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	size_t grayBytes = output.step * output.rows;
	float imgSize = input.rows * input.cols;

	unsigned char *d_input, *d_output;

	int * temp;
	int * temp2;
	size_t histogramBytes = sizeof(int) * 256;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, grayBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");
	//Malloc al histograma
	SAFE_CALL(cudaMalloc(&temp, histogramBytes), "CUDA Malloc failed");
	SAFE_CALL(cudaMalloc(&temp2, histogramBytes), "CUDA Malloc failed");


	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), grayBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), grayBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	int xB = 16;
	int yB = 16;
	// Specify a reasonable block size
	const dim3 block(xB, yB);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("equalizer_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	auto start_cpu =  chrono::high_resolution_clock::now();
	generateHistogram <<<grid, block>>>(d_input, d_output, input.cols, input.rows, input.step, imgSize, temp);
	equalizationPhases <<<grid, block>>>(d_input, d_output, imgSize, temp);
	equalizer_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, input.step, temp);
	auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("elapsed %f ms con bloque de %d y %d\n", duration_ms.count(), xB, yB);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
	SAFE_CALL(cudaFree(temp), "CUDA Free FAiled");
}

//main para busqueda de la imagen
int main(int argc, char *argv[])
{
	string imagePath;

	if(argc < 2)
		imagePath = "Images/dog3.jpeg";
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

	cv::Mat temp(input.rows, input.cols, CV_8UC1);
	cv::Mat output(input.rows, input.cols, CV_8UC1);

	cv::cvtColor(input, temp, CV_BGR2GRAY);

	equalizer(temp, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", temp);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
