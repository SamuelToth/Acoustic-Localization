#include "constants.cu"
#include <math.h>

//numFFTPairs is number of mics which have a set of matches
__global__
void findCoordinate(int numFFTPairs, Coordinate* micCoordinates,
                    WavePairContainer* filteredPairs, Coordinate* newPoint) {
/*
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= numFFTPairs){
    return;
  }
  __shared__ int addCounter=0; //for averageing later
  int referenceMic=filteredPairs[0].firstFFT;
  
  for(int i=1; i<=2;i++){
    if(tid+i<numFFTPairs){
      int mic1 = filteredPairs[tid].firstFFT;
      int mic2 = filteredPairs[tid].secondFFT;
      //distance between 2 mics
      decimal pairDistance = sqrt(pow(micCoordinates[mic2].x - micCoordinates[mic1].x ,2)+pow(micCoordinates[mic2].y-micCoordinates[mic1].y,2));
      //phase^2 = (x-x1)^2 + (y-y1)^2
      decimal mic1r=filteredPairs[tid].firstFFT.offset;
      decimal mic2r=filteredPairs[tid].secondFFT.offset;
      
      //stackoverflow.com
      //make sure that the circles intersect
      if(mic1r + mic2r < pairDistance || //if dist > r1+r2
         fabs(mic1r - mic2r) > pairDistance || //if dist < |r1-r2|
         (pairDistance==0 && mic1r==mic2)) //if (dist==0 && r1==r2)
        return;
      
      decimal a = (mic1r*mic1r - mic2r*mic2r + pairDistance*pairDistance)/(2*pairDistance);
      //find altitude of triangle formed by source and two mics
      decimal altitude = sqrt(mic1r*mic1r - a*a);

      Coordinate subPoint = {micCoordinates[mic2].x - micCoordinates[mic1].x,
                             micCoordinates[mic2].y - micCoordinates[mic1].y};
      
      subPoint.x = subPoint.x*(a/pairDistance);
      subPoint.y = subPoint.y*(a/pairDistance);
      subPoint.x = subPoint.x+micCoordinates[mic2].x;
      subPoint.y = subPoint.y+micCoordinates[mic2].y;
    
      decimal x3,y3,x4,y4; //two possible points
      x3 = subPoint.x + altitude*(micCoordinates[mic2].y-micCoordinates[mic1].y)/pairDistance;
      y3 = subPoint.y - altitude*(micCoordinates[mic2].x-micCoordinates[mic1].x)/pairDistance;
      x4 = subPoint.x - altitude*(micCoordinates[mic2].y-micCoordinates[mic1].y)/pairDistance;
      y4 = subPoint.y + altitude*(micCoordinates[mic2].x-micCoordinates[mic1].x)/pairDistance;

      decimal referenceDist = sqrt(pow(micCoordinates[referenceMic].x - x3 ,2)+pow(micCoordinates[referenceMic].y-y3,2));
      if(referenceMic.offset+10>referenceDist1 && referenceMic.offset-10<referenceDist){ //dist from refmic to point3 vs offset
        decimal goodX = x3; //choose which of the two points it is closest to
        decimal goodY = y3;     
      } else {
        decimal goodx = x4;
        decimal goody = y4;
      }
      
      atomicAdd(&addCounter, 1);
      atomicAdd(newPoint.x, goodx);
      atomicAdd(newPoint.y, goody);
    }
  }
  if(tid==0){
    newPoint.x = newPoint.x/addCounter; //average the coordinates
    newPoint.y = newPoint.y/addCounter; //average the coordinates
 */ 
}
  /*
  decimal micDistances[numFFTPairs];

  micDistances[tid]=0;
  referenceMic = filteredPairs[0].firstFFT;
  int mic1 = filteredPairs[idx].firstFFT;
  int mic2 = filteredPairs[idx].secondFFT;
  if (mic1 == referenceMic || mic2 == referenceMic){
     tempDistances = sqrt(micCoordinates[mic1].x * micCoordinates[mic1].x + micCoordinates[mic2].y * micCoordinates[mic2].y);

    (mic1 != referenceMic) ? micDistances[mic1]=tempDistance : micDistances[mic2]=tempDistance;
  }*/
  //speed of sound is about 34320 centimeters per second (ma.
  
  //go through list of matches of first mic, and take the join of the matches of the rest of the mics.

  
