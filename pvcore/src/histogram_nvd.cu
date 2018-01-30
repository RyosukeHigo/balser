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


#include "pvcore/colorconversion.h"

#include "pvcore/common.h"


// Each thread block fills one shared memory histogram, which are later
// reduced into one final histogram
__global__ void _initHistogram2DFromPolarImageFG(const unsigned char* _src,     // Image
                                                 const short* _mask,            // Mask
                                                 float* _hist,                  // Histogram
                                                 unsigned int _pitch,           // Image pitch
                                                 unsigned int _srcchannels,     // Number of channels in image
                                                 unsigned int _histx,           // Histogram x-dimension
                                                 unsigned int _histy,           // Histogram y-dimension
                                                 unsigned int _histz           // Histogram z-dimension
                                                 ) {
    
    // Image indices
    int y = _pitch*(blockDim.y+blockIdx.y + threadIdx.y);
    int x = 3*(blockDim.x+blockIdx.x + threadIdx.x);
    
    if( x < _mask[y] ) {
        // Histogram indices
        short xidx = (short)_src[y+x]*_histx >> 8;
        short yidx = (short)_src[y+x+1]*_histy >> 8;
        short zidx = (short)_src[y+x+2]*_histz >> 8;
        
        // Histogram index
        int hidx = zidx*_histy*_histx + yidx*_histx + xidx;
        
        _hist[hidx] += x*1.0f;
    }
    
}


// Histogram size: x*y*z*sizeof(float)
// 


void __initHistogramFromPolarImageGPU(const unsigned char* _src,     // Image
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
                                      dim3& _blockDim                // Number of threads
                                      ) {
    
    
    
    
    
}