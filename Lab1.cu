#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

__global__ void cnt_Pixel(uchar *img, int r, int g, int b, int N, int* cnt)
{
	int i = 3 * (threadIdx.x + blockIdx.x*blockDim.x);
	if (i >= N) return;
	if (img[i] == b && img[i+1] == g && img[i+2] == r) {
		atomicAdd(cnt, 1);
	}
}

int main(void)
{
	Mat image;
	image = imread("Lab1pic.jpg", cv::IMREAD_COLOR);   // Read the file
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	int r, g, b;

	cout << "Enter RGB: ";
	cin >> r >> g >> b;

	int cntColorCPU = 0;

	float elapsedTimeCUDA, elapsedTimeCPU;
	clock_t startCPU;

	uchar* img_data = image.data;

	size_t N = image.rows * image.cols * 3;
	startCPU = clock();

	for (size_t i = 0; i < N; i += 3) {
		if (img_data[i] == b && img_data[i + 1] == g && img_data[i + 2] == r)
			++cntColorCPU;
	}
	elapsedTimeCPU = (double)(clock() - startCPU) / CLOCKS_PER_SEC;

	cout << "Count of pixel with color(CPU): " << cntColorCPU << endl;

	cout << "CPU cnt pixel time = " << elapsedTimeCPU * 1000 << " ms\n";
	cout << "CPU memory throughput = " << N * sizeof(uchar) / elapsedTimeCPU / 1024 / 1024 / 1024 << " Gb/s\n";

	cudaEvent_t startCUDA, stopCUDA;
	cudaEventCreate(&startCUDA);
	cudaEventCreate(&stopCUDA);

	uchar *host_img;
	int *cntColorCUDA;

	CHECK(cudaMalloc(&host_img, N * sizeof(uchar)));
	CHECK(cudaMalloc(&cntColorCUDA, sizeof(int)));

	CHECK(cudaMemcpy(host_img, img_data, N * sizeof(uchar), cudaMemcpyHostToDevice));
	CHECK(cudaMemset(cntColorCUDA, 0, sizeof(int)));

	cudaEventRecord(startCUDA, 0);
	cnt_Pixel <<<((N + 511)/512), 512 >>> (host_img, r, g, b, N, cntColorCUDA);

	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);
	CHECK(cudaGetLastError());

	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

	int cntCUDA;
	CHECK(cudaMemcpy(&cntCUDA, cntColorCUDA, sizeof(int), cudaMemcpyDeviceToHost));

	cout << "Count of pixel with color(GPU): " << cntCUDA << endl;
	cout << "CUDA cnt pixel time = " << elapsedTimeCUDA * 1000 << " ms\n";
	cout << "CUDA memory throughput = " << N * sizeof(uchar) / elapsedTimeCUDA / 1024 / 1024 / 1024 << " Gb/s\n";

	waitKey(0);
	return 0;
}