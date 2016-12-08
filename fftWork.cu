#ifndef __lcl_fftWork
#define __lcl_fftWork

#define NX 512//256
#define BATCH 1

#include "constants.cu"

#include <cufft.h>
#include <cufftXt.h>
#include <cufftw.h>
#include <cmath>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <iostream>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>


__global__
void getFftResults(int oldArraySize, int* goodIndexes, bool* d_validFrequencies, FftResult* fftResults, cufftDoubleComplex* rawCufftResults)
{
  int thid = blockIdx.x *blockDim.x + threadIdx.x;
  if (thid >= oldArraySize)
  {
    return;
  }
  
  if(d_validFrequencies[thid])
  {
    fftResults[goodIndexes[thid]].frequency = thid;
    fftResults[goodIndexes[thid]].offset = rawCufftResults[thid].y;
    
  }
}


__global__
void trueIfGreater(bool* results, const double* const fftReals, unsigned int size, double flagValue)
{
  int thid = blockIdx.x *blockDim.x + threadIdx.x;
  if (thid >= size)
  {
    return;
  }
  
  results[thid] = (abs(fftReals[thid]) > flagValue);
}


int getFftBatch(FftBatch* batch, cufftDoubleComplex* h_data)
{
  cufftHandle plan;// = cufftCreate();
  cufftDoubleComplex *data;
  cufftDoubleComplex *outData;
  cudaMalloc((void**)&data, sizeof(cufftDoubleComplex)*(NX)*BATCH);               //example code: cudaMalloc((void**)&data, sizeof(cufftComplex)*(NX/2+1)*BATCH);  
  cudaMemcpy(data, h_data, sizeof(cufftDoubleComplex)*NX*BATCH, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&outData, sizeof(cufftDoubleComplex)*(NX / 2 + 1)*BATCH);
  

  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n"); 
    return 1;
  }

  if (cufftPlan1d(&plan, NX, CUFFT_D2Z, BATCH) != CUFFT_SUCCESS){ 
    fprintf(stderr, "CUFFT error: Plan creation failed"); 
    return 1;	
  }

	/* Use the CUFFT plan to transform the signal in place. */ 
  if (cufftExecD2Z(plan, (cufftDoubleReal*)data, outData) != CUFFT_SUCCESS){ 
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed"); 
    return 1;	
  }

  if (cudaDeviceSynchronize() != cudaSuccess){ 
    fprintf(stderr, "Cuda error: Failed to synchronize\n"); 
    return 1;
  }
  
  int outputSize = (NX / 2 + 1)*BATCH;
  
  //cudaMemcpy(h_data, outData, sizeof(cufftDoubleComplex)*(NX / 2 + 1)*BATCH, cudaMemcpyDeviceToHost);
  thrust::host_vector<cufftDoubleComplex> rawFft(outData, outData + outputSize);
  

  thrust::host_vector<double> fftReals(outputSize);
  //double* fftReals = (double*)malloc(sizeof(double) * outputSize);
 
  double rawSum = 0;
  for (unsigned int i = 0; i < outputSize; i++)
  {
    rawSum += rawFft[i].x;
    fftReals[i] = rawFft[i].x;
  }
  double rawAverage = rawSum / outputSize;
  
  thrust::device_vector<double> d_fftReals(fftReals);
  //thrust::device_vector<double> d_validReals(outputSize);
  
  bool* d_validFrequencies;
  cudaMalloc(&d_validFrequencies, sizeof(bool) * outputSize);
  

  
  
  int blockSizeInt = 1024;
  int gridSizeInt = outputSize / 1024 + 1;
  
  trueIfGreater<<<gridSizeInt, blockSizeInt>>>(d_validFrequencies, thrust::raw_pointer_cast(d_fftReals.data()), outputSize, rawAverage * 5);
  
  int* goodIndexes = (int*)malloc(sizeof(int) * outputSize);
  thrust::exclusive_scan(d_validFrequencies, d_validFrequencies + outputSize, goodIndexes);
  int goodVals = thrust::reduce(d_validFrequencies, d_validFrequencies + outputSize);
  
  FftResult* d_fftResults;
  cudaMalloc(&d_fftResults, sizeof(FftResult) * goodVals);
  
  batch->size = goodVals;
  batch->fftResults = (FftResult*)malloc(sizeof(FftResult) * goodVals);
  
  int* d_goodIndexes;
  cudaMalloc(&d_goodIndexes, sizeof(int) * outputSize);
  cudaMemcpy(d_goodIndexes, goodIndexes, sizeof(int) * outputSize, cudaMemcpyHostToDevice);
  
  getFftResults<<<gridSizeInt, blockSizeInt>>>(outputSize, d_goodIndexes,d_validFrequencies, d_fftResults, outData);
  
  cudaMemcpy(batch->fftResults, d_fftResults, sizeof(FftResult) * goodVals, cudaMemcpyDeviceToHost);
  
  cudaFree(d_fftResults);
  //void getFftResults<<<gridSizeInt, blockSizeInt>>>(outputSize, goodIndexes, d_validFrequencies, d_fftResults, outData);
  
  //bool* h_validFrequencies = (bool*)malloc(sizeof(bool) * outputSize);
  //cudaMemcpy(h_validFrequencies, d_validFrequencies, sizeof(bool) * outputSize, cudaMemcpyDeviceToHost);
  
  /*for (unsigned int i = 0 i < outputSize; i++)
  {
    if (h_validFrequencies[i])
    {
      
    }
  }*/
  
  cudaFree(d_goodIndexes);
  cufftDestroy(plan); 
  cudaFree(data);
  cudaFree(outData);
  
  return 0;
}

#endif
