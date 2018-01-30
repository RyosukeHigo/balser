// ====================================================================== //
//  pvcore -- simple parallel computer vision library
//  Copyright (C) 2012  Niklas Bergström
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
// ====================================================================== //

#include "pvcore/histogram.h"

// AVX/SSE
#include <immintrin.h>

// TBB
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/mutex.h>

#include <cmath>
#include <iostream>

#include "histogram/generate_probim.cpp"
#include "histogram/create_histogram.cpp"
#include "histogram/create_histogram_polar.cpp"



void normalizeHistogram_helper(float* _hist, unsigned int _histx,
							   unsigned int _histy, unsigned int _histz) {
	
	// Normalize
	float ssum = 0.0f;
	for( unsigned int i=0; i<_histz; ++i ) {
		for( unsigned int j=0; j<_histy; ++j ) {
			for( unsigned int k=0; k<_histx; ++k ) {
				ssum += _hist[i*_histy*_histx+j*_histx+k] + 1e-7f;
			}
		}
	}
	
	for( unsigned int i=0; i<_histz; ++i ) {
		for( unsigned int j=0; j<_histy; ++j ) {
			for( unsigned int k=0; k<_histx; ++k ) {
				_hist[i*_histy*_histx+j*_histx+k] = (_hist[i*_histy*_histx+j*_histx+k] + 1e-7f)/ssum;
			}
		}
	}
	
}


void reduceHistogram_helper(float* thist,
							float* _hist,
							unsigned int _histx,
							unsigned int _histy,
							unsigned int _histz,
							unsigned int _histdim,
							unsigned int _threads) {
	
	
    // Sum histograms
    // (Could be done using parallel reduction)
    unsigned int i=0;
#ifdef USE_AVX1
    for(  ;i<_histdim; i+=8 ) {
        float *th = thist+i;
        __m256 h1 = _mm256_loadu_ps( th );
        for( unsigned int j=1; j<_threads; ++j ) {
            const __m256 h2 = _mm256_loadu_ps( th+j*_histdim );
            h1 = _mm256_add_ps( h1, h2 );
        }
        _mm256_storeu_ps( _hist+i, h1 );
    }
    i-=8;
#endif
    
    // Do the rest
    for(  ;i<_histdim; i+=1 ) {
        _hist[i] = thist[i];
        for( unsigned int j=1; j<_threads; ++j ) {
            _hist[i] += thist[j*_histdim+i];
        }
    }
	
	
	normalizeHistogram_helper(_hist, _histx, _histy, _histz);
    
}



namespace pvcore {
    
    template <typename T>
    void __initHistogramFromImageCPU(const unsigned char* _src,     // Image
                                     const T* _mask,    // Mask
                                     float* _hist,                  // Histogram
                                     unsigned int _width,           // Image width
                                     unsigned int _height,          // Image height
                                     unsigned int _pitch,           // Image pitch
                                     unsigned int _srcchannels,     // Number of channels in image
                                     unsigned char _maskid,         // Id of valid pixels in mask
                                     unsigned int _histx,           // Histogram x-dimension
                                     unsigned int _histy,           // Histogram y-dimension
                                     unsigned int _histz,           // Histogram z-dimension
                                     unsigned int _histchannels,    // Number of histogram channels
                                     unsigned int _threads // Number of threads
                                     ) {
        
        // Compute total histogram dimension
        int histdim = _histx*_histy*_histz;
        
        // One full histogram per thread
        float* thist = new float[histdim*_threads];
        memset(thist, 0, histdim*_threads*4);
        
        // Fill all histograms
       tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1),
                          CREATE_HISTOGRAM_TBB<T>( _src, _mask, _maskid,
                                                  _width, _height, _pitch, _srcchannels,
                                                  _histx, _histy, _histz,
                                                  thist, _threads,
                                                  _histchannels,
                                                  1.0f/(float)_height) );
        
        reduceHistogram_helper(thist, _hist, _histx, _histy, _histz, histdim, _threads);
        
        delete [] thist;
    }
    template void __initHistogramFromImageCPU<unsigned char>(const unsigned char*, const unsigned char*, float*, unsigned int, unsigned int,unsigned int,
                                                     unsigned int, unsigned char, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    template void __initHistogramFromImageCPU<short>(const unsigned char*, const short*, float*, unsigned int, unsigned int,unsigned int,
                                                     unsigned int, unsigned char, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    

	
	
	
	void __initHistogramFromPolarImageCPU(const unsigned char* _src,     // Image
										  const short* _mask,            // Mask
										  float* _hist,                  // Histogram
										  unsigned int _width,           // Image width
										  unsigned int _height,          // Image height
										  unsigned int _pitch,           // Image pitch
										  unsigned int _srcchannels,     // Number of channels in image
										  const short* _maxr,            // Mask info
										  bool _fg,                      // Foreground or background of mask
										  unsigned int _histx,           // Histogram x-dimension
										  unsigned int _histy,           // Histogram y-dimension
										  unsigned int _histz,           // Histogram z-dimension
										  unsigned int _histchannels,    // Number of histogram channels
										  unsigned int _threads			 // Number of threads
	) {
		
		
#ifdef USE_TBB
		// Compute total histogram dimension
		int histdim = _histx*_histy*_histz;
		
		// One full histogram per thread
		float* thist = new float[histdim*_threads];
		memset(thist, 0, histdim*_threads*4);
		
		// Fill all histograms
		tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1),
						  CREATE_HISTOGRAM_POLAR_TBB( _src, _mask, _maxr, _fg,
													 _width, _height, _pitch, _srcchannels,
													 _histx, _histy, _histz,
													 thist, _threads,
													 _histchannels,
													 1.0f/(float)_height) );
		
		// Sum and normalize histograms
		reduceHistogram_helper(thist, _hist, _histx, _histy, _histz, histdim, _threads);
		
		delete [] thist;
#else
		if( _fg ) {
			short* zeros = new short[_height];
			memset(zeros,0,_height*sizeof(short));
			
			_create_histogram_polar_helper(_src, _mask,
										   _maxr,
										   _width,
										   _pitch,
										   _srcchannels,
										   _histx,
										   _histy,
										   _histz,
										   _hist,
										   _histchannels,
										   1.0f/(float)_height,
										   0, _height,
										   zeros, _mask);
			
			delete [] zeros;
		} else {
			_create_histogram_polar_helper(_src, _mask,
										   _maxr,
										   _width,
										   _pitch,
										   _srcchannels,
										   _histx,
										   _histy,
										   _histz,
										   _hist,
										   _histchannels,
										   1.0f/(float)_height,
										   0, _height,
										   _mask, _maxr);
		}
		
		normalizeHistogram_helper(_hist, _histx, _histy, _histz);
#endif

	}
		
		
    void __generateProbabilityImagePolarCPU(const unsigned char* _src,     // Image
                                            const float* _hist,            // Histogram
                                            float* _dest,                  // Probability image
                                            unsigned int _width,           // Image width
                                            unsigned int _height,          // Image height
                                            unsigned int _pitch,           // Image pitch
                                            unsigned int _srcchannels,     // Number of channels in image
                                            const short* _maxr,            // Mask info
                                            unsigned int _histx,           // Histogram x-dimension
                                            unsigned int _histy,           // Histogram y-dimension
                                            unsigned int _histz,           // Histogram z-dimension
                                            unsigned int _histchannels,    // Number of histogram channels
                                            unsigned int _threads          // Number of threads
                                            ) {
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0,_threads,1 ),
                          GENERATE_PROBIM_TBB(_src, _hist, _dest,
                                              _width, _height, _pitch, _srcchannels,
                                              _histx, _histy, _histz,
                                              _histchannels, _maxr, _threads) );
        
    }
    
    
    void __generateProbabilityImageCPU(const unsigned char* _src,     // Image
                                       const float* _hist,            // Histogram
                                       float* _dest,                  // Probability image
                                       unsigned int _width,           // Image width
                                       unsigned int _height,          // Image height
                                       unsigned int _pitch,           // Image pitch
                                       unsigned int _srcchannels,     // Number of channels in image
                                       unsigned int _histx,           // Histogram x-dimension
                                       unsigned int _histy,           // Histogram y-dimension
                                       unsigned int _histz,           // Histogram z-dimension
                                       unsigned int _histchannels,    // Number of histogram channels
                                       unsigned int _threads          // Number of threads
                                            ) {
        
        
        // Fill all histograms
        tbb::parallel_for(tbb::blocked_range<size_t>(0,_threads,1 ),
                          GENERATE_PROBIM_TBB(_src, _hist, _dest,
                                              _width, _height, _pitch, _srcchannels,
                                              _histx, _histy, _histz,
                                              _histchannels, NULL, _threads) );

        
        float idimx = (float)_histx/256.0f;
        float idimy = (float)_histy/256.0f;
        float idimz = (float)_histz/256.0f;
        
        unsigned int dpitch = _pitch/_srcchannels;
        
        switch( _histchannels ) {
            case 1:
                for( unsigned int i=0; i<_height; ++i ) {
                    unsigned int idx = _pitch*i;
                    unsigned int j=0;
                    for( ; j<_width; ++j ) {
                        unsigned char c1 = _src[idx]; idx+=_srcchannels;
                        unsigned int idx1 = (unsigned int)((float)(c1)*idimx);
                        
                        _dest[i*dpitch+j] = _hist[idx1];
                    }
                    //memset( &(dest[i*dpitch+j]), 0, (width-j)*sizeof(float) );
                    
                }
                break;
                
            case 2:
                for( unsigned int i=0; i<_height; ++i ) {
                    unsigned int idx = _pitch*i;
                    unsigned int j=0;
                    for( ; j<_width; ++j ) {
                        unsigned char c1 = _src[idx]; idx+=1;
                        unsigned char c2 = _src[idx]; idx+=2;
                        unsigned int idx1 = (unsigned int)((float)(c1)*idimx);
                        unsigned int idx2 = (unsigned int)((float)(c2)*idimy);
                        
                        float p = _hist[idx2*_histx+idx1];
                        
                        _dest[i*dpitch+j] = p;
                    }
                    //memset( &(dest[i*dpitch+j]), 0, (width-j)*sizeof(float) );
                    
                }
                break;
                
            case 3:
            {
                for( unsigned int i=0; i<_height; ++i ) {
                    unsigned int idx = _pitch*i;
                    unsigned int j=0;
                    for( ; j<_width; ++j ) {
                        unsigned char c1 = _src[idx]; idx+=1;
                        unsigned char c2 = _src[idx]; idx+=1;
                        unsigned char c3 = _src[idx]; idx+=1;
                        unsigned int idx1 = (unsigned int)((float)(c1)*idimx);
                        unsigned int idx2 = (unsigned int)((float)(c2)*idimy);
                        unsigned int idx3 = (unsigned int)((float)(c3)*idimz);
                        
                        float p = _hist[(idx3*_histy + idx2)*_histx + idx1];
                        
                        _dest[i*dpitch+j] = p;
                    }
                    //memset( &(dest[i*dpitch+j]), 0, (width-j)*sizeof(float) );
                    
                }
                break;
            }                

        }
        

        
    }
  
    
} // namespace pvcore
