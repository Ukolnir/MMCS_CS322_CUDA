#include <cmath>
#include <iostream>

#define N 25600
#define M 1024

using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

void generateMatrix(int* matrix, int size) {
	srand(time(NULL));
	for (size_t i = 0; i < size; ++i) {
		matrix[i] = rand() % 100;
	}
}

void print(int* matrix, int n, int m) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			cout << matrix[i*m + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void compute(int* matrix, int n, int m, int* result) {
	for (int i = 0; i < n; ++i) {
		result[i] = 0;
		int sz = i * m;
		for (int j = 1; j < m; ++j) {
			if (matrix[sz + j - 1] > matrix[sz + j])
				++result[i];
		}
	}
}

bool checkResult(int* resultCPU, int* resultGPU, int n) {
	for(int i = 0; i < N; ++i){
		if(resultCPU[i] != resultGPU[i]){
      cout << "Wrong in str: " << i << " inCPU: " << resultCPU[i] << " inGPU: " << resultGPU[i] << endl;
			return false;
    }
	}
	return true;
}

__global__ void fastComputeCUDA(int* matrix, int n, int m, int* result)
{
  int idxStr = threadIdx.x + blockIdx.x*blockDim.x;
	if(idxStr >= n) return;
	int res = 0;
  int temp0 = matrix[idxStr*m];

  __shared__ int cache[256][33];
  for(int k = 0; k < m/32; ++k){
      //32 потока читают одну строчку
      for(int s = 0; s < 256/8; ++s){
        int row = threadIdx.x / 32 + s*8;
        int col = k * 32 + threadIdx.x % 32;
        cache[row][threadIdx.x % 32]
        = matrix[(blockIdx.x*256 + row) * m + col];
      }

    __syncthreads();

    for(int idx = 0; idx < 32; ++idx){
      int tempC = cache[threadIdx.x][idx];
  		if(temp0 > tempC)
  			++res;
      temp0 = tempC;
  	}
  }
	result[idxStr] = res;
}

int main(void) {
	float elapsedTimeCUDA, elapsedTimeCPU;
	clock_t startCPU;

	int* matrixDEVICE;
	int* resultDEVICE;
	int* resultHOST = new int[N];
	int* matrixHOST = new int[N*M];
	int* resultCPU = new int[N];

	generateMatrix(matrixHOST, N*M);
	startCPU = clock();
	compute(matrixHOST, N, M, resultCPU);
	elapsedTimeCPU = (double)(clock() - startCPU) / CLOCKS_PER_SEC;

	cout << "CPU time = " << elapsedTimeCPU * 1000 << " ms\n";
	cout << "CPU memory throughput = " << N * M * 4 / elapsedTimeCPU / 1024 / 1024 / 1024 << " Gb/s\n";

	cudaEvent_t startCUDA, stopCUDA;
	cudaEventCreate(&startCUDA);
	cudaEventCreate(&stopCUDA);

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	CHECK(cudaMalloc(&matrixDEVICE, N * M * 4));
	CHECK(cudaMemcpy(matrixDEVICE, matrixHOST, N * M * 4, cudaMemcpyHostToDevice));

	CHECK(cudaMalloc(&resultDEVICE, N * 4));

	cudaEventRecord(startCUDA, 0);
	fastComputeCUDA <<<((N + 255)/256), 256 >>> (matrixDEVICE, N, M, resultDEVICE);
	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);
	CHECK(cudaGetLastError());
	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

	CHECK(cudaMemcpy(resultHOST, resultDEVICE, N * 4, cudaMemcpyDeviceToHost));

	cout << (checkResult(resultCPU, resultHOST, N) ? "Result is correct" : "Result isn't correct") << endl;

	cout << "CUDA time = " << elapsedTimeCUDA << " ms\n";
	cout << "CUDA memory throughput = " << N * M * 4 / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";

	//waitKey(0);
	return 0;
}
