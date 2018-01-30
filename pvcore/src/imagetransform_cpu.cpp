// ====================================================================== //
//  pvtools -- simple parallel computer vision library
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

#include "pvcore/imagetransform.h"

#include <algorithm>



///////////////////////////////////////////////
//////////////////// DEFINES //////////////////
///////////////////////////////////////////////
#define CAST_8U(t) convert_uchar_rte(t)

#define inv255 (0.003921568627451f);


#include "imagetransform/cart2pol.cpp"
#include "imagetransform/gradmag.cpp"
#include "imagetransform/flip.cpp"


namespace pvcore {
    
    // External function
    void __cart2polCPU(const unsigned char* _src,
                       unsigned char* _dest,
                       float _xc, float _yc,
                       unsigned int _width,
                       unsigned int _height,
                       unsigned int _nChannels,
                       unsigned int _pitchs,
                       unsigned int _maxrad,
                       unsigned int _theta,
                       unsigned int _pitchd,
                       short* _maxr,
                       unsigned int _nthreads) {
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0,_nthreads,1 ),
                          CART2POL_8U_TBB(_src, _dest, _xc, _yc,
                                          _width, _height, _nChannels, _pitchs,
                                          _maxrad, _theta, _pitchd, _maxr, _nthreads ) );
        
    };
    
    
    void __gradmagCPU(const unsigned char* _src,
                      unsigned char* _dest,
                      unsigned int _width,
                      unsigned int _height,
                      unsigned int _pitch,
                      unsigned int _nthreads ) {
    }
    
    
    void __flipHorizontalCPU(const unsigned char* _src,
                             unsigned char* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _nChannels,
                             unsigned int _pitch,
                             unsigned int _nthreads )  {
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0,_nthreads,1 ),
                          FLIPH_8U_TBB(_src, _dest, _width, _height, _nChannels,
                                       _pitch, _nthreads) );
        
        
    }
    
    
    void __flipVerticalCPU(const unsigned char* _src,
                           unsigned char* _dest,
                           unsigned int _width,
                           unsigned int _height,
                           unsigned int _nChannels,
                           unsigned int _pitch,
                           unsigned int _nthreads
                           ) {
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0,_nthreads,1 ),
                          FLIPV_8U_TBB(_src, _dest, _width, _height, _nChannels,
                                       _pitch, _nthreads) );
        
    }
} // namespace pvcore