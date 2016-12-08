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

#include <iostream>


cufftDoubleComplex* generateData()
{
  //create array to store waves
  std::cout<< "Number of waves: ";
  int numberOfWaves = 0;
  std::cin>>numberOfWaves;
  int** wavesArray = (int**)malloc(sizeof(int*) * numberOfWaves);
  
  int* waveLengths = (int*)malloc(sizeof(int) * numberOfWaves);
  
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
  
  return h_data;
}


int main() {

  cufftDoubleComplex* h_data = generateData();
  
  //handleCufft(freq);


  /*
  //find all matches across FFT batches
  WaveMatches matches = findAllMatches(batches, 3);
  
  //allocate memory for wave pair containers
  WavePairContainer* wavePairContainers;
  wavePairContainers = (WavePairContainer*)malloc(sizeof(WavePairContainer * matches.matches.size()));
  
  //Filter matches into wavePairContainers
  filterMatches(batches,
                batchCount, 
                &allMatches,
                wavePairContainers,
                matches.matches.size());
  */
  
  
  free(h_data);//allocated within generate data
  return 0;
}

#endif
