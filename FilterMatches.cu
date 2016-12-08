#ifndef __lcl_FilterMatches
#define __lcl_FilterMatches

#include "constants.cu"

#include<vector>
#include<iostream>
#include<stdio.h>


#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>

/*
__global__ buildHistogramForTriples(+Seq)
__global__ removeNonTripleMatches(+Seq)
filterForTriples
__global__ matrixToWavePair(+Seq)
findWavePairs
filterMatches
*/

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

void buildHistogramForTriplesSeq(const GpuWaveMatches* allMatches,
                              unsigned int** matchHistograms)
{
  for(int count=0; count<allMatches->matchesCount; count++){
    for(int x=0; x<allMatches->widths[count]; x++){
      for(int y=0; y<allMatches->heights[count]; y++){
        unsigned int flatMatrixPosition = allMatches->widths[count] * x + y;
        int widthBatchNum = allMatches->widthBatches[count];
        int heightBatchNum = allMatches->heightBatches[count];
        unsigned int matrixVal = (unsigned int)allMatches->matches[count][flatMatrixPosition];
        matchHistograms[heightBatchNum][y] += matrixVal;
        matchHistograms[widthBatchNum][x] += matrixVal;
      }
    }
  }
  return;
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


void removeNonTripleMatchesSeq(GpuWaveMatches* allMatches,
                               const unsigned int * const * const matchHistograms)
{
  for(int count=0; count<allMatches->matchesCount; count++){
    for(int x=0; x<allMatches->widths[count]; x++){
      for(int y=0; y<allMatches->heights[count]; y++){
        unsigned int flatMatrixPosition = allMatches->widths[count] * x + y;
        int widthBatchNum = allMatches->widthBatches[count];
        int heightBatchNum = allMatches->heightBatches[count];
        if(matchHistograms[widthBatchNum][x] < 3 || matchHistograms[heightBatchNum][y] <3){
          allMatches->matches[count][flatMatrixPosition] = 0;
        }
      }
    }
  }
  return;
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
  
  printf(";alskdjf;aslkdjfa;lskdjf\n");fflush(NULL);
  
  //determine kernel dimentions
  const int maxThreadsPerBlock = 512;
  unsigned int* maxWidth = thrust::max_element(&allMatches.widths[0], &allMatches.widths[0] + allMatches.widths.size());
  unsigned int* maxHeight = thrust::max_element(&allMatches.heights[0], &allMatches.heights[0] + allMatches.heights.size());
  double widthHeightRatio = *maxWidth / *maxHeight;
  printf(";alskdjf;aslkdjfa;lskdjf\n");fflush(NULL);
  unsigned int blockSizeIntX = 10;//allMatches.matches.size() % maxThreadsPerBlock;
  unsigned int blockSizeIntY = 10;//(maxThreadsPerBlock - blockSizeIntX) * widthHeightRatio;
  unsigned int blockSizeIntZ = 10;//blockSizeIntY;
  printf(";alskdjf;aslkdjfa;lskdjf\n");fflush(NULL);
  if (widthHeightRatio > 1)
  {
    blockSizeIntY = (maxThreadsPerBlock - blockSizeIntX) / widthHeightRatio;
  }
  else
  {
    blockSizeIntY = (maxThreadsPerBlock - blockSizeIntX) * widthHeightRatio;
  }
  printf(";alskdjf;aslkdjfa;lskdjf\n");fflush(NULL);
  blockSizeIntZ = 10;//(maxThreadsPerBlock - blockSizeIntX) - blockSizeIntY;
  unsigned int gridSizeInt = (*maxWidth * *maxHeight * allMatches.matches.size()) / (blockSizeIntY * blockSizeIntX * blockSizeIntZ) + 1;
  printf(";alskdjf;aslkdjfa;lskdjf\n");fflush(NULL);
  dim3 blockSize(blockSizeIntX, blockSizeIntY, blockSizeIntZ);
  dim3 gridSize(gridSizeInt);
  
  
  printf("!!!!!!");fflush(NULL);
  
  //fill histograms
  buildHistogramForTriples<<<gridSize, blockSize>>>(gpuWaveMatches,matchHistograms);
  
  //remove frequencies that dont match at least three times
  removeNonTripleMatches<<<gridSize, blockSize>>>(gpuWaveMatches, matchHistograms);
  
  d_outMatches = gpuWaveMatches;
}




__global__
void matrixToWavePair(bool* d_waveMatches,
                      const int* const outputPositions,
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

void matrixToWavePairSeq(bool* h_waveMatches,
                         const int* const outputPositions,
                         unsigned int matrixSize,
                         unsigned int matrixWidth,
                         unsigned int matrixHeight,
                         WavePair* h_wavePairs,
                         unsigned int pairCount)
{
  for (int x=0; x<matrixWidth; x++){
    for (int y=0; y<matrixHeight; y++){
      if(h_waveMatches[matrixWidth*x+y]){
        h_wavePairs[matrixWidth*x+y].waveIdx1=x;
        h_wavePairs[matrixWidth*x+y].waveIdx2=y;
      }
    }
  }
  return;
}


void findWavePairs(FftBatch* batches,
              unsigned int batchCount,
              GpuWaveMatches* d_waveMatches,
              WavePairContainer* wpContainers)
{
  GpuWaveMatches* h_waveMatches = NULL;
  GpuWaveMatchesToHost(h_waveMatches, d_waveMatches);
  
  for (unsigned int i = 0; i < h_waveMatches->matchesCount; i++)
  {
    //determine the number of wavePairs and their positions in the output array
    unsigned int matrixSize = h_waveMatches->widths[i] * h_waveMatches->heights[i];
    int* scanResult = (int*)malloc(sizeof(int) * matrixSize);
    thrust::exclusive_scan(h_waveMatches->matches[i], h_waveMatches->matches[i] + matrixSize, scanResult);
    unsigned int total = scanResult[matrixSize - 1] + h_waveMatches->matches[i][matrixSize - 1];
    int* d_scanResult;
    cudaMalloc(&d_scanResult, sizeof(int) * matrixSize);
    cudaMemcpy(d_scanResult, scanResult, sizeof(int) * matrixSize, cudaMemcpyHostToDevice);
    
    //create wavePairContainer
    wpContainers[i].wavePairCount = total;
    wpContainers[i].firstFFT = h_waveMatches->widthBatches[i];
    wpContainers[i].secondFFT = h_waveMatches->heightBatches[i];
    wpContainers[i].wavePairArray = (WavePair*)malloc(sizeof(WavePair) * total);
    
    //populate teh wavePairArray
    WavePair* d_wavePairs;
    cudaMalloc(&d_wavePairs, sizeof(WavePair) * total);
    
    //determine kernel dimentions
    int blockSizeIntX;
    int blockSizeIntY;
    int gridSizeInt;
    const int maxBlockSize = 512;
    double widthHeightRatio = h_waveMatches->widths[i]/h_waveMatches->heights[i];
    if (widthHeightRatio > 1)
    {
      blockSizeIntY = maxBlockSize / widthHeightRatio;
      blockSizeIntX = maxBlockSize - blockSizeIntY;
    }
    else
    {
      blockSizeIntX = maxBlockSize * widthHeightRatio;    
      blockSizeIntY = maxBlockSize - blockSizeIntX;
    }
    gridSizeInt = matrixSize / maxBlockSize + 1;
    
    dim3 blockSize(blockSizeIntX, blockSizeIntY);
    dim3 gridSize(gridSizeInt);    
    
    //prodece wavePairs
    matrixToWavePair<<<gridSize, blockSize>>>(d_waveMatches->matches[i],
                      d_scanResult,
                      matrixSize,
                      h_waveMatches->widths[i],
                      h_waveMatches->heights[i],
                      d_wavePairs,
                      total);
    cudaMemcpy(wpContainers[i].wavePairArray, d_wavePairs, sizeof(WavePair) * total, cudaMemcpyDeviceToHost);             
                  

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
  unsigned int** h_matchHistograms = (unsigned int**)malloc(sizeof(unsigned int*) * batchCount);
  cudaMalloc(&d_matchHistograms, sizeof(unsigned int*) * batchCount);
  for (unsigned int i = 0; i < batchCount; i++)
  {
    printf("HALP PLS\r\n");
    unsigned int* d_matchHistogram;
    cudaMalloc(&d_matchHistogram, sizeof(unsigned int) * batches[i].size);
    cudaMemset(d_matchHistogram, 0, sizeof(unsigned int) * batches[i].size);
    
    h_matchHistograms[i] = d_matchHistogram;
    cudaMemcpy(d_matchHistograms, h_matchHistograms, sizeof(unsigned int) * batchCount, cudaMemcpyHostToDevice);
    //d_matchHistograms[i] = d_matchHistogram;
  }
  
  printf("a\r\n");
  
  GpuWaveMatches* d_waveMatches;
  filterForTriples(*allMatches, d_matchHistograms, d_waveMatches);
  
  printf("b\r\n");

  //free device waveMatches
  freeGpuWaveMatches(d_waveMatches);
  
  printf("c\r\n");
  
  //free histogram memory
  for (unsigned int i = 0; i < batchCount; i++)
  {
    cudaFree(d_matchHistograms[i]);
  }
  cudaFree(d_matchHistograms);
}


/*int main()
{

}*/
#endif
