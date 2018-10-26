////g++ -o exe histogramEqualization.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

void equalizationPha2(cv::Mat& input, cv::Mat& output, long * temp){
	int index;
	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			index = (int)input.at<uchar>(i,j);
			output.at<uchar>(i,j) = temp[index];
		}
	}
}

void createHistogram(cv::Mat& input, long *temp) {
	int index;
	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			index = (int)input.at<uchar>(i,j);
			temp[index]++;
		}
	}
}

void normalize(cv::Mat& input, long *temp) {
	long temp[256] = {};
	for(int i = 0; i < 256; i++) {
		temp[i] = temp[i];
	}
	//Reinicializamos en 0 el histograma orgiinal
	for(int i = 0; i < 256; i++) {
		temp[i] = 0;
	}
	for(int i = 0; i < 256; i++) {
		for(int j = 0; j <= i; j++) {
			temp[i] += temp[j];
		}
		int normalizeVar = (temp[i]*255) / (input.rows*input.cols);
		temp[i] = normalizeVar;
	}
}

void equalizationPha1(cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	long temp[256] = {};
	createHistogram(input, temp);
	normalize(input, temp);
	equalizationPha2(input, output, temp);
}

int main(int argc, char *argv[])
{
	string imagePath;

	if(argc < 2)
		imagePath = "Images/dog1.jpeg";
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
	auto start_cpu =  chrono::high_resolution_clock::now();
	equalizationPha1(temp, output);
	auto end_cpu =  chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("elapsed: %f ms\n", duration_ms.count());

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
