#ifndef __lcl_main
#define __lcl_main

#define NX 512//256
#define BATCH 1

#include "constants.cu"
#include "matches.cu"
#include "FilterMatches.cu"
#include "fftWork.cu"

#include <cufft.h>
#include <cufftXt.h>
#include <cufftw.h>
#include <cmath>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "constants.cu"
#include "matches.cu"
#include "FilterMatches.cu"

#include <iostream>

MicData generateData()
{
  //create array of batches(microphones)
  int numberOfBatches = -1;
  std::cout<<"Number of microphones: ";
  std::cin>>numberOfBatches;

  cufftDoubleComplex** micDataArray = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*)*numberOfBatches);
  //length of each mic's array should be the same as int* waveLengths

  //create array to store waves
  int numberOfWaves = -1;
  std::cout<<"Number of waves: ";
  std::cin>>numberOfWaves;

  //int** wavesArray = (int**)malloc(sizeof(int*) * numberOfWaves);
  
  int* waveLengths = (int*)malloc(sizeof(int) * numberOfWaves);
  for(int batchNum=0; batchNum<numberOfBatches; batchNum++){
    //initialize h_data
    cufftDoubleComplex* h_data = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*NX);
    for (unsigned int i = 0; i < NX; i++){
      h_data[i].x = 0;
      h_data[i].y = 0.0;
    }
    srand (time(NULL));

    /* generate secret number between 1 and 10: */
    bool addNoise = 0;
    for (unsigned int i = 0; i < numberOfWaves; i++)
    {
      double freq = 0;
    
      std::cout<< "Wave frequency (whole numbers only): ";
      std::cin>>freq;
      std::cout<<"add noise? (0 = no, 1 = yes): ";
      std::cin>>addNoise;
      for (unsigned int i = 0; i < NX; i++){
        h_data[i].x += sin( 2 * M_PI * freq * (double)i / NX);
        double noise = 0;
        if (addNoise)
        {
          noise = (double)rand() / RAND_MAX;
          if (rand() % 2){
            noise *= -1;
          }
        }
        h_data[i].x += noise;     
      }
    }
  micDataArray[batchNum]=h_data;
  }
  MicData micData = {micDataArray, waveLengths, numberOfBatches};
  return micData;
}


int main() {


  MicData h_micData = generateData();

  FftBatch* fftBatches = (FftBatch*)malloc(sizeof(FftBatch) * h_micData.numberOfBatches);
  for (unsigned int i = 0; i < h_micData.numberOfBatches; i++)
  {
    printf("3.%i\r\n", i);
    getFftBatch(&fftBatches[i], h_micData.micData[i]);//TODO: this must take a size parameter if batches can have different sizes
  }

  //find all matches across FFT batches
  WaveMatches matches = findAllMatches(fftBatches, h_micData.numberOfBatches);
  
  //allocate memory for wave pair containers
  WavePairContainer* wavePairContainers;
  wavePairContainers = (WavePairContainer*)malloc(sizeof(WavePairContainer) * matches.matches.size());
  
  
  //Filter matches into wavePairContainers
  filterMatches(fftBatches,
                h_micData.numberOfBatches, 
                &matches,
                wavePairContainers,
                matches.matches.size());
  
 
  for(int i=0; i<h_micData.numberOfBatches; i++){
    //free(MicData[i]->h_data);//allocated within generate data
  }
  
  free (fftBatches);
  return 0;
}

#endif
