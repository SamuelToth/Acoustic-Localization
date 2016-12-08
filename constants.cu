#ifndef __lcl_constants
#define __lcl_constants

//#include <thrust/device_vector.h>
#include <cuda.h>
#include <vector>
#include <cufft.h>
#include <cufftXt.h>

typedef float decimal;

struct MicData {
  cufftDoubleComplex** micData;
  int* waveLengths;
  int numberOfBatches;
};

struct Distances {
  decimal* distances;
};

struct Coordinate {
  decimal x;
  decimal y;
};

struct FftResult {
  decimal frequency;
  decimal offset;
};

struct FftBatch {
  FftResult* fftResults;
  unsigned int size;  
};

struct WavePair {
	int waveIdx1;
	int waveIdx2;
	decimal offset;
};


struct WavePairContainer{
	int firstFFT;
	int secondFFT;
	WavePair* wavePairArray;
	int wavePairCount;
	//thrust::device_vector<WavePair> wavePairArray;
};

struct WaveMatches {
  std::vector<bool*> matches;
  std::vector<unsigned int> widths;
  std::vector<unsigned int> heights;
  std::vector<int> widthBatches;
  std::vector<int> heightBatches;
};

struct GpuWaveMatches {
  bool** matches;
  unsigned int matchesCount;
  
  unsigned int* widths;
  unsigned int widthsCount;
  
  unsigned int* heights;
  unsigned int heightsCount;
  
  int* widthBatches;
  unsigned int widthBatchesCount;
  
  int* heightBatches;
  unsigned int heightBatchesCount;
  
};

void GpuWaveMatchesToHost(GpuWaveMatches* h_gpuWaveMatches, GpuWaveMatches* d_gpuWaveMatches)
{
  h_gpuWaveMatches = (GpuWaveMatches*)malloc(sizeof(GpuWaveMatches));
  cudaMemcpy(h_gpuWaveMatches, d_gpuWaveMatches, sizeof(GpuWaveMatches), cudaMemcpyDeviceToHost);
  
  cudaMemcpy(h_gpuWaveMatches->widths, d_gpuWaveMatches->widths, sizeof(unsigned int) * h_gpuWaveMatches->widthsCount, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gpuWaveMatches->heights, d_gpuWaveMatches->heights, sizeof(unsigned int) * h_gpuWaveMatches->heightsCount, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gpuWaveMatches->widthBatches, d_gpuWaveMatches->widthBatches, sizeof(int) * h_gpuWaveMatches->widthBatchesCount, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gpuWaveMatches->heightBatches, d_gpuWaveMatches->heightBatches, sizeof(int) * h_gpuWaveMatches->heightBatchesCount, cudaMemcpyDeviceToHost);
  
  //copy matches
  cudaMemcpy(h_gpuWaveMatches->matches, d_gpuWaveMatches->matches, sizeof(bool*) * h_gpuWaveMatches->matchesCount, cudaMemcpyDeviceToHost);
  for (unsigned int i = 0; i < h_gpuWaveMatches->matchesCount; i++){
    cudaMemcpy(h_gpuWaveMatches->matches[i], d_gpuWaveMatches->matches[i], sizeof(bool) * h_gpuWaveMatches->widths[i] * h_gpuWaveMatches->heights[i], cudaMemcpyDeviceToHost);
  }
}

void freeGpuWaveMatches(GpuWaveMatches* gpuMatches)
{
  for (unsigned int i = 0; i < gpuMatches->matchesCount; i++)
  {
    cudaFree(gpuMatches->matches[i]);
  }
  cudaFree(gpuMatches->widths);
  cudaFree(gpuMatches->heights);
  cudaFree(gpuMatches->widthBatches);
  cudaFree(gpuMatches->heightBatches);
}

void WaveMatchesToGpu(const WaveMatches& matches, GpuWaveMatches* gpuMatches)
{
  //allocate memory for the GpuWaveMatches struct
  cudaMalloc(&gpuMatches, sizeof(GpuWaveMatches));
  
  //copy the the matches array and all match matrix
  bool** gpuMatchesArray;
  cudaMalloc(&gpuMatchesArray, sizeof(bool*) * matches.matches.size());
  for (unsigned int i = 0; i < matches.matches.size(); i++)
  {
    bool* gpuMatchMatrix;
    cudaMalloc(&gpuMatchMatrix, sizeof(bool) * matches.widths[i] * matches.heights[i]);
    cudaMemcpy(gpuMatchMatrix, matches.matches[i], sizeof(bool) * matches.widths[i] * matches.heights[i], cudaMemcpyHostToDevice);
    gpuMatchesArray[i] = gpuMatchMatrix;
  }
  gpuMatches->matches = gpuMatchesArray;
  gpuMatches->matchesCount = matches.matches.size();
  
  //copy the stored widths
  unsigned int* gpuWidths;
  cudaMalloc(&gpuWidths, sizeof(unsigned int) * matches.widths.size());
  cudaMemcpy(gpuWidths, &matches.widths[0], sizeof(unsigned int) * matches.widths.size(), cudaMemcpyHostToDevice);
  gpuMatches->widths = gpuWidths;
  gpuMatches->widthsCount = matches.widths.size();
  
  //copy the stored heights
  unsigned int* gpuHeights;
  cudaMalloc(&gpuHeights, sizeof(unsigned int) * matches.heights.size());
  cudaMemcpy(gpuHeights, &matches.heights[0], sizeof(unsigned int) * matches.heights.size(), cudaMemcpyHostToDevice);
  gpuMatches->heights = gpuHeights;
  gpuMatches->heightsCount = matches.heights.size();
  
  //copy stored widthBatches
  int* gpuWidthBatches;
  cudaMalloc(&gpuWidthBatches, sizeof(int) * matches.widthBatches.size());
  cudaMemcpy(gpuWidthBatches, &matches.widthBatches[0], sizeof(int) * matches.widthBatches.size(), cudaMemcpyHostToDevice);
  gpuMatches->widthBatches = gpuWidthBatches;
  gpuMatches->widthBatchesCount = matches.widthBatches.size();
  
  //copy stored heightBatches
  int* gpuHeightBatches;
  cudaMalloc(&gpuHeightBatches, sizeof(int) * matches.heightBatches.size());
  cudaMemcpy(gpuHeightBatches, &matches.heightBatches[0], sizeof(int) * matches.heightBatches.size(), cudaMemcpyHostToDevice);
  gpuMatches->heightBatches = gpuHeightBatches;
  gpuMatches->heightBatchesCount = matches.heightBatches.size();
}
#endif



