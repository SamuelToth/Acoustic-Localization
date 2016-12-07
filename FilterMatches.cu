#include "constants.cu"

#include<vector>
#include<iostream>
#include<stdio.h>


#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>

 
__global__
void buildHistogramForTriples(const GpuWaveMatches* allMatches,
                              unsigned int** matchHistograms)
{
  // 3D block 1D grid
  int matchIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int widthIdx = blockIdx.x * blockDim.y + threadIdx.y;
  int heightIdx = blockIdx.x * blockDim.z + threadIdx.z;
  
  //return if index is bad
  if (matchIdx >= allMatches->matchesCount)
  {
    return;
  }
  
  if (widthIdx >= allMatches->widths[matchIdx] 
        || heightIdx >= allMatches->heights[matchIdx])
  {
    return;
  }
  
  //find matrix and batch information from allMatches
  unsigned int flatMatrixPosition = allMatches->widths[matchIdx] * widthIdx + heightIdx;
  int widthBatchNum = allMatches->widthBatches[matchIdx];
  int heightBatchNum = allMatches->heightBatches[matchIdx];
  unsigned int matrixVal = (unsigned int)allMatches->matches[matchIdx][flatMatrixPosition];
  
  
  //set histogram values
  atomicAdd(&matchHistograms[widthBatchNum][widthIdx], matrixVal);
  atomicAdd(&matchHistograms[heightBatchNum][heightIdx], matrixVal);
}


__global__
void removeNonTripleMatches(GpuWaveMatches* allMatches,
                            const unsigned int * const * const matchHistograms)
{
  // 3D block 1D grid
  int matchIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int widthIdx = blockIdx.x * blockDim.y + threadIdx.y;
  int heightIdx = blockIdx.x * blockDim.z + threadIdx.z;
  
  //return if index is bad
  if (matchIdx >= allMatches->matchesCount)
  {
    return;
  }
  
  if (widthIdx >= allMatches->widths[matchIdx] 
        || heightIdx >= allMatches->heights[matchIdx])
  {
    return;
  }
  
  //find matrix and batch information from allMatches
  unsigned int flatMatrixPosition = allMatches->widths[matchIdx] * widthIdx + heightIdx;
  int widthBatchNum = allMatches->widthBatches[matchIdx];
  int heightBatchNum = allMatches->heightBatches[matchIdx]; 
  
  //set value based on histogram
  if (matchHistograms[widthBatchNum][widthIdx] < 3 || matchHistograms[heightBatchNum][heightIdx] < 3)
  {
    allMatches->matches[matchIdx][flatMatrixPosition] = 0;
  }
}



/*filterForTriples
 *Removes all matches with frequencies that aren't found across
 * at least three results
 *
 *allMatches: WaveMatches struct containing all the match matricies for each
 * FftBatch
 *
 *MatchHistograms: an array of histograms (one for each FftBatch) for tracking
 * the number of matches for each frequency
 */
 void filterForTriples(WaveMatches& allMatches,
                      unsigned int** matchHistograms,
                      GpuWaveMatches* d_outMatches)
{
  //move waveMatches to teh gpu
  GpuWaveMatches* gpuWaveMatches = NULL;
  WaveMatchesToGpu(allMatches, gpuWaveMatches);
  
  //determine kernel dimentions
  const int maxThreadsPerBlock = 512;
  unsigned int* maxWidth = thrust::max_element(&allMatches.widths[0], &allMatches.widths[0] + allMatches.widths.size());
  unsigned int* maxHeight = thrust::max_element(&allMatches.heights[0], &allMatches.heights[0] + allMatches.heights.size());
  double widthHeightRatio = *maxWidth / *maxHeight;
  unsigned int blockSizeIntX = allMatches.matches.size() % maxThreadsPerBlock;
  unsigned int blockSizeIntY = (maxThreadsPerBlock - blockSizeIntX) * widthHeightRatio;
  unsigned int blockSizeIntZ = blockSizeIntY;
  if (widthHeightRatio > 1)
  {
    blockSizeIntY = (maxThreadsPerBlock - blockSizeIntX) / widthHeightRatio;
  }
  else
  {
    blockSizeIntY = (maxThreadsPerBlock - blockSizeIntX) * widthHeightRatio;
  }
  blockSizeIntZ = (maxThreadsPerBlock - blockSizeIntX) - blockSizeIntY;
  unsigned int gridSizeInt = (*maxWidth * *maxHeight * allMatches.matches.size()) / (blockSizeIntY * blockSizeIntX * blockSizeIntZ) + 1;
  dim3 blockSize(blockSizeIntX, blockSizeIntY, blockSizeIntZ);
  dim3 gridSize(gridSizeInt);
  
  //fill histograms
  buildHistogramForTriples<<<gridSize, blockSize>>>(gpuWaveMatches,matchHistograms);
  
  //remove frequencies that dont match at least three times
  removeNonTripleMatches<<<gridSize, blockSize>>>(gpuWaveMatches, matchHistograms);
  
  d_outMatches = gpuWaveMatches;
}




__global__
void matrixToWavePair(bool* d_waveMatches,
                      const unsigned int* const outputPositions,
                      unsigned int matrixSize,
                      unsigned int matrixWidth,
                      unsigned int matrixHeight,
                      WavePair* d_wavePairs,
                      unsigned int pairCount)
{
  int thid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  int thidX = blockIdx.x *blockDim.x + threadIdx.x;
  int thidY = blockIdx.x *blockDim.y + threadIdx.y;
  if (thid >= matrixSize || thidX >= matrixWidth || thidY >= matrixHeight)
  {
    return;
  }
  
  unsigned int position = matrixWidth * thidX + thidY;
  if (d_waveMatches[position])
  {
    //TODO; ensure that waveidx1 will always be the width of the matrix
    d_wavePairs[outputPositions[position]].waveIdx1 = thidX;
    d_wavePairs[outputPositions[position]].waveIdx2 = thidY;
  }
}




void findWavePairs(FftBatch* batches,
              unsigned int batchCount,
              GpuWaveMatches* d_waveMatches
              WavePairContainer* wpContainers)
{
  GpuWaveMatches* h_waveMatches;
  GpuWaveMatchesToHost(h_waveMatches, d_waveMatches);
  
  for (unsigned int i = 0; i < h_waveMatches->matchesCount i++)
  {
    //determine the number of wavePairs and their positions in the output array
    unsigned int matrixSize = waveMatches->widths[i] * waveMatches->heights[i];
    bool* scanResult = (bool*)malloc(sizeof(bool) * matrixSize);
    thrust::exclusive_scan(h_waveMatches->matches[i], h_waveMatches->matches[i] + matrixSize, scanResult);
    unsigned int total = scanResult[matrixSize - 1] + h_waveMatches->matches[i][matrixSize - 1];
    
    //create wavePairContainer
    wpContainers[i].wavePairCount = total;
    wpContainers[i].firstFFT = h_waveMatches.widthBatches[i];
    wpContainers[i].secondFFT = h_waveMatches.heightBatches[i];
    wpContainers[i].wavePairArray = (WavePair*)malloc(sizeof(WavePair) * total);
    
    //populate teh wavePairArray
    WavePair* d_wavePairs;
    cudaMalloc(&d_wavePairs, sizeof(WavePair) * total);
    
    

    free(scanResult);
    cudaFree(d_wavePairs);
    
  }
  //TODO: free h_waveMatches;
}


/*filterMatches: removes all invalid matches. Returns valid matches in wave pair
 * containers
 *
 *batches: array of Fft data taken from input mics. Each batch is data
 * from one mic.
 *
 *batchCount: the number of batches
 *
 *allMatches: Raw match data
 *
 *wavePairContainers: output. Pairs of waves from different mics
 * with the same frequency
 *
 *containerCount: the number of wavePairContainers to output
 */
void filterMatches(FftBatch* batches,
                   unsigned int batchCount,
                   WaveMatches* allMatches,
                   WavePairContainer* wavePairContainers,
                   unsigned int containerCount)
{
  //Create a histogram for each WaveMatch width
  unsigned int** d_matchHistograms;
  cudaMalloc(&d_matchHistograms, sizeof(unsigned int*) * batchCount);
  for (unsigned int i = 0; i < batchCount; i++)
  {
    unsigned int* d_matchHistogram;
    cudaMalloc(&d_matchHistogram, sizeof(unsigned int) * batches[i].size);
    cudaMemset(d_matchHistogram, 0, sizeof(unsigned int) * batches[i].size);
    d_matchHistograms[i] = d_matchHistogram;
  }
  
  GpuWaveMatches* d_waveMatches;
  filterForTriples(*allMatches, d_matchHistograms, d_waveMatches);
  
  

  //free device waveMatches
  freeGpuWaveMatches(d_waveMatches);
  
  //free histogram memory
  for (unsigned int i = 0; i < batchCount; i++)
  {
    cudaFree(d_matchHistograms[i]);
  }
  cudaFree(d_matchHistograms);
}


int main()
{

}
