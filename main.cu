#ifndef __lcl_main
#define __lcl_main

#define NX 512//256
#define BATCH 1

#include <cufft.h>
#include <cufftXt.h>
#include <cufftw.h>
#include <cmath>

#include <iostream>

int handleCufft(double freq)
{
  //std::cout << "===========input:==========="<<std::endl;
  //gen data
  cufftDoubleComplex* h_data = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*NX);
  for (unsigned int i = 0; i < NX; i++){
    h_data[i].x = sin( 2 * M_PI * freq * (double)i / NX);
    h_data[i].y = 0.0;
    //std::cout<<i<<": "<<h_data[i].x<< "+i * " << h_data[i].y <<std::endl;
    //std::cout<<h_data[i].x<<std::endl;
  }

  //std::cout<<std::endl;
  //std::cin.get();
  
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
  
  cudaMemcpy(h_data, outData, sizeof(cufftDoubleComplex)*(NX / 2 + 1)*BATCH, cudaMemcpyDeviceToHost);
  
  //std::cin.get();

  //std::cout<<"~~~~~~~output~~~~~~~"<<std::endl;
   for (unsigned int i = 0; i < NX / 2 + 1; i++) {
    //std::cout << i << ": " <<h_data[i].x << "+ i*" << h_data[i].y << std::endl;
    std::cout << h_data[i].x << std::endl;
  }
  
  std::cin.get();
  std::cin.get();
  
  for (unsigned int i = 0; i < NX / 2 + 1; i++) {
    std::cout << i << ": " <<h_data[i].x << " + i*" << h_data[i].y << std::endl;
    //std::cout << h_data[i].x << std::endl;
  }
  
  cufftDestroy(plan); 
  cudaFree(data);
  cudaFree(outData);
  
  free(h_data);
  return 0;
}


int main() {

  double freq;
  std::cout << "freq: ";
  std::cin >> freq;
  
  handleCufft(freq);

  
  return 0;
}

#endif
