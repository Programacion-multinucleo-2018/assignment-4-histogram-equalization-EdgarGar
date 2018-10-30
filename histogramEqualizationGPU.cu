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

// Utilizamos un poco del codigo pasado de la tarea 2 (image blurring)
// input - input image one dimensional array
// ouput - output image one dimensional array
// width, height - width and height of the images
// colorWidthStep - number of color bytes (cols * colors)
__global__ void equalizer_kernel(unsigned char* input, unsigned char* output, int width, int height, int grayWidthStep, int *h_s)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Location of gray pixel in output del codigo de rgb_to_gray de class demos
	//Texel ID
	const int gray_tid = yIndex * grayWidthStep + xIndex;

	//Debugging
    // printf("width: %d", width);
    // printf("height: %d", height);

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		//Creamos e igualuamos nuestro index del histograma y luego se lo asignamos al output el valor en el gray pixel
		output[gray_tid] = h_s[input[gray_tid]];
	}
}

//Funcion para crear el histograma
__global__ void createHistogram_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, float grayImageSize, int *h_s) {
	//Codigo de arriba de la funcion pasada
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Texel id
	int color_tid = yIndex * colorWidthStep + xIndex;

	int xyIndex = threadIdx.x + threadIdx.y * blockDim.x;

	//Shared Int histogram en donde vamos a ir guardando los valores
	// __shared__ int h_sTable[256];
	__shared__ int temp[256];

	//Inicializamos el histograma en 0
	if(xyIndex < 256) {
		temp[xyIndex] = 0;
	}

	//Sync threads para ir creando el histograma
	__syncthreads();

	//Codigo de atomic class demos 9
	// if (p(data[i]))
	// atomicAdd(count, 1);
	if(xIndex < width && yIndex < height) { //Validacion
		atomicAdd(&temp[input[color_tid]], 1);
	}

	__syncthreads();

	//Se lo copiamos a nuestra variable con la memoria
	if(xyIndex < 256) {
		atomicAdd(&h_s[xyIndex], temp[xyIndex]);
	}

	// __syncthreads();

	// //Normalizamos el histograma
	// if(xyIndex < 256 && blockIdx.x == 0 && blockIdx.y == 0){
	// 	for(int i = 0; i < xyIndex; i++) {
	// 		h_s[xyIndex] += h_sTable[i];
	// 	}
	// 	h_s[xyIndex] = h_s[xyIndex]*(255/grayImageSize);
	// }

	// __syncthreads();
}

__global__ void normalizeHistogram(unsigned char* input, unsigned char* output, float grayImageSize, int *h_s) {
	int xyIndex = threadIdx.x + threadIdx.y * blockDim.x;

	//Creamos variable temporal para guardar el histograma original
	__shared__ int temporal[256];

	if(xyIndex < 256 && blockIdx.x == 0 && blockIdx.y == 0) { //Validacion
		//Inicializamos con 0
		temporal[xyIndex] = 0;
		//Guardamos histograma original
		temporal[xyIndex] = h_s[xyIndex];
		__syncthreads();

		unsigned int normVar = 0;
		for(int i = 0; i <= xyIndex; i++) {
			//Normalizar
			normVar += temporal[i];
		}
		// h_s[xyIndex] = normVar/255;
		// for(int i = 0; i <= xyIndex; i++) {
		// 	h_s[xyIndex] += temp[i];
		// }
		h_s[xyIndex] = normVar*255/grayImageSize;
	}
}
//Codigo de la tarea 2
void equalizer(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	//Gray bytes para el output
	size_t grayBytes = output.step * output.rows;
	float grayImageSize = input.rows * input.cols;

	unsigned char *d_input, *d_output;

	// El histograma lo tenemos que guardar en algun lugar y como tenemos que hacer malloc lo declaramos de tamano 256 * sizeof de int
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

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
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
	cv::Mat temp(input.rows, input.cols, CV_8UC1); //Creamos una matriz temporal para cambiar el input a una griz
	cv::Mat output(input.rows, input.cols, CV_8UC1); //Se tiene que cambiar a CV_8UC1 en vez de CV_8UC3

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
