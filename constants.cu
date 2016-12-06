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
	//thrust::device_vector<WavePair> wavePairArray;
};

struct WaveMatches {
  std::vector<bool*> matches;
  std::vector<unsigned int> widths;
  std::vector<unsigned int> heights;
};
