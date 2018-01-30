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

#ifndef _COREIMAGETRANSFORM_H_
#define _COREIMAGETRANSFORM_H_

// TBB
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// CUDA
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif // USE_CUDA

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
                       unsigned int _nthreads);
    
    
    void __gradmagCPU(const unsigned char* _src,
                      unsigned char* _dest,
                      unsigned int _width,
                      unsigned int _height,
                      unsigned int _pitch,
                      unsigned int _nthreads );
    
    
    void __flipHorizontalCPU(const unsigned char* _src,
                             unsigned char* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _nChannels,
                             unsigned int _pitch,
                             unsigned int _nthreads );
    
    
    void __flipVerticalCPU(const unsigned char* _src,
                           unsigned char* _dest,
                           unsigned int _width,
                           unsigned int _height,
                           unsigned int _nChannels,
                           unsigned int _pitch,
                           unsigned int _nthreads);
    
#ifdef USE_CUDA
    void __transposeGPU(const unsigned char* _src,
                        unsigned char* _dest,
                        unsigned int _width,
                        unsigned int _height,
                        unsigned int _pitchs,
                        unsigned int _pitchd,
                        dim3& _blockDim);
    
    void __cart2polGPU(const unsigned char* _src,
                       unsigned char* _dest,
                       short* _maxr,
                       float _xc, float _yc,
                       unsigned int _width,
                       unsigned int _height,
                       unsigned int _pitchs,
                       unsigned int _pitchd,
                       unsigned int _theta,
                       unsigned int _nChannels,
                       dim3& _blockDim
                       );
    
    void __flipHorizontalGPU(const unsigned char* _src,
                             unsigned char* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _nChannels,
                             unsigned int _pitch,
                             dim3& _blockSize
                             );
    
    void __flipVerticalGPU(const unsigned char* _src,
                           unsigned char* _dest,
                           unsigned int _width,
                           unsigned int _height,
                           unsigned int _nChannels,
                           unsigned int _pitch,
                           dim3& _blockSize
                           );
#endif // USE_CUDA
    
    
} // pvcore

#endif // _COREIMAGETRANSFORM_H_
