// Lab3_Parallel.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>	
#include <typeinfo>
#include "lab3.h"
//#include <cuda.h>
//#include <opencv2/core/cuda.hpp>
//#include "cuda_runtime.h"
//#include <cuda/std/atomic>
//#include "Lab3_Parallel.cu"




using namespace cv;

using namespace std;
using namespace std::chrono;



void getChannelPixels() {
	Mat image;
	image = imread("pict.jpg");
	int down_width = 10;
	int down_height = 29;
	Mat resize_down;
	resize(image, resize_down, Size(down_width, down_height), INTER_LINEAR);
	int x = image.at<Vec3b>(resize_down.cols, resize_down.rows)[0];
	int y = image.at<Vec3b>(resize_down.cols, resize_down.rows)[1];
	int z = image.at<Vec3b>(resize_down.cols, resize_down.rows)[2];
	//cout << "Width: " << resize_down.cols << endl;
	//cout << "Height: " << resize_down.rows << endl;
	cout << "Value of blue channel:" << x << endl;
	cout << "Value of green channel:" << y << endl;
	cout << "Value of red channel:" << z << endl;
}

void getMinimumBlue() {
	Mat image;
	Vec3d minimum;
	image = imread("pict.jpg");
	Mat resize_down;
	int down_width = 10;
	int down_height = 29;
	resize(image, resize_down, Size(down_width, down_height), INTER_LINEAR);
	Mat blue, green, red;
	vector<Mat> channels(3);
	split(resize_down, channels);
	int m = 1000;
	blue = channels[0];
#pragma omp parallel for reduction (min:m)
	for (int i = 0; i < blue.total(); i++) {
		int pix = blue.data[i];
		m = min(m, pix);
	}
	cout << "Minimum for blue channel: " << m << endl;
}

void calculateConvolution() {
	Mat image;
	image = imread("pict.jpg");
	Mat resize_down;
	int down_width = 10;
	int down_height = 29;
	resize(image, resize_down, Size(down_width, down_height), INTER_LINEAR);
	Mat blue, green, red;
	vector<Mat> channels(3);
	split(resize_down, channels);
	blue = channels[0];
	green = channels[1];
	red = channels[2];
	Mat kernel2 = Mat::ones(3, 3, CV_64F);
	kernel2 = kernel2 / 9;
	Mat img_b, img_g, img_r, img;
	filter2D(blue, img_b, -1, kernel2, Point(-1, -1), 0, 4);
	cout << "Convolution for the Blue Channel: " << round(sum(img_b)[0] / blue.total()) << endl;
	//cout << "Convolution Matrix for the Blue Channel: " << img_b << endl;
	filter2D(green, img_g, -1, kernel2, Point(-1, -1), 0, 4);
	cout << "Convolution for the Green Channel: " << round(sum(img_g)[0] / green.total()) << endl;
	//cout << "Convolution Matrix for the Green Channel: " << img_g << endl;
	filter2D(red, img_r, -1, kernel2, Point(-1, -1), 0, 4);
	cout << "Convolution for the Red Channel: " << round(sum(img_r)[0] / red.total()) << endl;
	//cout << "Convolution Matrix for the Red Channel: " << img_r << endl;
	filter2D(image, img, -1, kernel2, Point(-1, -1), 0, 4);
	imshow("Original", image);
	imshow("Kernel blur", img);
	imwrite("blur_kernel.jpg", img);
	waitKey();
	destroyAllWindows();
}



int main()
{
	//1st task
	auto start1 = high_resolution_clock::now();
	getChannelPixels()	;
	auto stop1 = high_resolution_clock::now();
	auto duration1 = duration_cast<microseconds>(stop1 - start1);
	cout << "Time spent to count pixels without Cuda: " << duration1.count() << " microseconds" << endl;



	Mat image;
	image = imread("pict.jpg");
	int down_width = 10;
	int down_height = 29;
	Mat resize_down;
	resize(image, resize_down, Size(down_width, down_height), INTER_LINEAR);
	getChannelPixels_cuda(image, down_width, down_height, resize_down);

	
	//2nd task
	auto start2 = high_resolution_clock::now();
	getMinimumBlue();
	auto stop2 = high_resolution_clock::now();
	auto duration2 = duration_cast<microseconds>(stop2 - start2);
	cout << "Time spent to find minimum in Blue channel without Cuda: " << duration2.count() << " microseconds" << endl;
	
	Mat blue, green, red;
	vector<Mat> channels(3);
	split(resize_down, channels);
	int m = 1000;
	blue = channels[0];
	minReduce(image, blue);
	
	//3rd task
	auto start3 = high_resolution_clock::now();
	calculateConvolution();
	auto stop3 = high_resolution_clock::now();
	auto duration3 = duration_cast<microseconds>(stop3 - start3);
	cout << "Time spent to calculate convolution without Cuda: " << duration3.count() << " microseconds" << endl;
	conv_globalMem(image, resize_down, down_width, down_height, 1, 1, 1, 9);
	system("pause");
	
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
