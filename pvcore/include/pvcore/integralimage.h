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

#ifndef _COREINTEGRALIMAGE_H_
#define _COREINTEGRALIMAGE_H_

#include "pvcore/common.h"


namespace pvcore {
    
    void __integralImageCPU(const unsigned char* _src,
                            float* _dest,
                            int _width,
                            int _height,
                            int _srcpitch,
                            int _destpitch,
                            unsigned int _threads );

    void __integralImageCPU(const unsigned char* _src,
                            int* _dest,
                            int _srcpitch,
                            int _destpitch,
                            int _width,
                            int _height,
                            unsigned int _nthreads );
#ifdef USE_CUDA
    cudaError_t __integralImageCUDA(const unsigned char* _src,
                                    unsigned char* _dest,
                                    unsigned int _width,
                                    unsigned int _height,
                                    unsigned int _pitchs,
                                    unsigned int _pitchd,
                                    unsigned int _kernel,
                                    dim3 _blockDim,
                                    unsigned char _threshold = 0);
#endif // USE_CUDA
    
#ifdef USE_OPENCL
    cl_int __integralImageCL(const cl::Buffer* _src,
                             cl::Buffer* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _pitchs,
                             unsigned int _pitchd,
                             unsigned int _kernel,
                             dim3 _blockDim,
                             unsigned char _threshold = 0);
#endif // USE_OPENCL
    
} // pvcore

#endif // _COREINTEGRALIMAGE_H_
