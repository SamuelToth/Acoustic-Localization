//#include <thrust/device_vector.h>
#include <cuda.h>
#include <vector>

typedef float decimal;

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




