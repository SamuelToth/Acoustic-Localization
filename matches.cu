#include "constants.cu"
#include "distances.cu"

#include<vector>
#include<iostream>
#include<stdio.h>


#define MAX_THREADS 1024
#define FREQUENCY_RANGE 4


/**findMatches
  *Finds all the matching wave pairs between two input sources.
  *Output does not filter unlikely results and false posatives
  *
  *batch1: array containing the FftResults from one input device
  *batch1Count: the number of elements in batch1
  *batch2: array containing the FftResults from one input device
  *batch2Count: the number of elements in batch2
  *matchMatrix 1D bool array representing a 2d bool matrix
  */

//find matches sequentially
void findMatchesSeq(FftResult* batch1, 
                 unsigned int batch1Count, 
                 FftResult* batch2,
                 unsigned int batch2Count,
                 bool* matchMatrix) {
  
  //Find all matching waves between two sources
//  FftResult wave1 = batch1[absoluteIndex];
  for (unsigned int i = 0; i < batch2Count; i++) {
    //find the waves are within an acceptable range there is a match
    for (unsigned int k=0; k<batch1Count; k++) {
      if (abs(batch1[k].frequency - batch2[i].frequency) <= FREQUENCY_RANGE){
        //create wavePair
        matchMatrix[i + k] = true;
        //printf("i: %i j: %i\n", absoluteIndex, i);
      }
      else { //else not a match
        matchMatrix[i + k] = false;
      }
    }
  }
  return;
}

__global__
void findMatches(FftResult* batch1, 
                 unsigned int batch1Count, 
                 FftResult* batch2,
                 unsigned int batch2Count,
                 bool* matchMatrix) {
                 
  //get threads position and return early if out of bounds
  int absoluteIndex = blockIdx.x *blockDim.x + threadIdx.x;  
  if (absoluteIndex > batch1Count) {
    return;
  }
  
  //init matchMatrix to false
  for (unsigned int i = 0; i < batch2Count; i++) {
    matchMatrix[absoluteIndex * batch1Count + i] = false;
  }
  __syncthreads();
  
  //Find all matching waves between two sources
  FftResult wave1 = batch1[absoluteIndex];
  for (unsigned int i = 0; i < batch2Count; i++) {
    //find the waves are within an acceptable range there is a match
    if (abs(wave1.frequency - batch2[i].frequency) <= FREQUENCY_RANGE){
      //create wavePair
      matchMatrix[absoluteIndex * batch1Count + i] = true;
      //printf("i: %i j: %i\n", absoluteIndex, i);
    }
  }
}

__global__
void findMatches2d(FftResult* batch1, 
                 unsigned int batch1Count, 
                 FftResult* batch2,
                 unsigned int batch2Count,
                 bool* matchMatrix) {
                 
  int absoluteIndex = blockIdx.x * blockDim.x * blockDim.y
      + threadIdx.y * blockDim.x + threadIdx.x;
      
  //init matchMatrix to false
  //for (unsigned int i = 0; i < batch2Count; i++) {
    matchMatrix[absoluteIndex] = false;
  //}
  __syncthreads();
  
  int crossBlockXIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int crossBlockYIndex = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (abs(batch1[crossBlockXIndex].frequency - batch2[crossBlockYIndex].frequency) <= FREQUENCY_RANGE) {
    matchMatrix[absoluteIndex] = true;
  }
  
}


WaveMatches findAllMatches(FftBatch* batches, unsigned int batchCount) {
  
  //store matrix for each batch compare
  WaveMatches matches;  
  
  for (unsigned int i = 0; i < batchCount; i++) {
    for (unsigned int j = i + 1; j < batchCount; j++) {
      //create return matrix for findMatches kernal on device
      bool* d_matchMatrix;  
      cudaMalloc(&d_matchMatrix, sizeof(bool) * batches[i].size * batches[j].size);
      
      //for better performance make the larger batch the first batch in the kernal call
      FftBatch bigBatch;
      FftBatch littleBatch;
      
      if (batches[i].size > batches[j].size) {
        bigBatch = batches[i];
        littleBatch = batches[j];
      } else {
        bigBatch = batches[j];
        littleBatch = batches[i];
      }
      
      bool* h_matchMatrix = (bool *)malloc(bigBatch.size * littleBatch.size * sizeof(bool));

      //Move Fft results to kernal
      FftResult* d_batch1;
      FftResult* d_batch2;
      cudaMalloc(&d_batch1, sizeof(FftResult) * bigBatch.size);
      cudaMalloc(&d_batch2, sizeof(FftResult) * littleBatch.size);
      
      cudaMemcpy(d_batch1, bigBatch.fftResults, 
                 sizeof(FftResult) * bigBatch.size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_batch2, littleBatch.fftResults, 
                 sizeof(FftResult) * littleBatch.size, cudaMemcpyHostToDevice);
      
      //call find matches kernal
      int threads = bigBatch.size % MAX_THREADS;
      int blocks = bigBatch.size / MAX_THREADS + 1;
      dim3 bDim(bigBatch.size, littleBatch.size, 0);
      //findMatches2d<<<threads, bDim>>>(d_batch1, bigBatch.size, 
                                       //d_batch2, littleBatch.size, 
                                       //d_matchMatrix);
      findMatchesSeq(bigBatch.fftResults, bigBatch.size, littleBatch.fftResults,littleBatch.size, h_matchMatrix);
      
      //copy matchMatrix to host and store in return vector
      //bool* matchMatrix = (bool*)malloc(sizeof(bool) * bigBatch.size * littleBatch.size);
      
      //cudaMemcpy(matchMatrix, d_matchMatrix, sizeof(bool) * bigBatch.size *littleBatch.size,cudaMemcpyDeviceToHost);
                 
      matches.matches.push_back(h_matchMatrix);
      matches.widths.push_back(bigBatch.size);
      matches.heights.push_back(littleBatch.size);
       
            
      //free memory
      cudaFree(d_batch1);
      cudaFree(d_batch2);
      cudaFree(d_matchMatrix);
            
    }
  }

  return matches;
}

int main(){

  FftBatch batch1;
  FftResult results1[5];
  FftBatch batch2;
  FftResult results2[3];
  FftBatch batch3;
  FftResult results3[2];
  
  for (unsigned int i = 0; i < 5; i++) {
    FftResult result;
    result.frequency = i;
    result.offset = i * 2;
    results1[i] = result;
  }
  batch1.fftResults = results1;
  batch1.size = 5;
  
  for (unsigned int i = 0; i < 3; i++) {
    FftResult result;
    result.frequency = i + 1;
    result.offset = i * 4;
    results2[i] = result;
  }
  batch2.fftResults = results2;
  batch2.size = 3;
  
  for (unsigned int i = 0; i < 2; i++) {
    FftResult result;
    result.frequency = i + 3;
    result.offset = i;
    results3[i] = result;
  }
  batch3.fftResults = results3;
  batch3.size = 2;
  
  FftBatch batches[3];
  batches[0] = batch1;
  batches[1] = batch2;
  batches[2] = batch3;
  
  WaveMatches matches = findAllMatches(batches, 3);
  

  std::cout << "matches: " << matches.matches.size() << std::endl;
  for (unsigned int i = 0; i < matches.matches.size(); i++) {
    //std::cout << "width: " << matches.widths[i] << std::endl;
    //std::cout << "height: " << matches.heights[i] << std::endl; 
    
    //std::cout << matches.matches[0][1][0] << std::endl;
    
    
    for (unsigned int j = 0; j < matches.widths[i]; j++) {
      for (unsigned int k = 0; k < matches.heights[i]; k++) {
        if (matches.matches[matches.widths[i] * j + k]){
          std::cout<<"("<<j<<","<<k<<")"<<std::endl;
        }
      
      }
    }
  }


  return 0;
}
