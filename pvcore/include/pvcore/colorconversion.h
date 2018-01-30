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

#ifndef _CORECOLORCONVERSION_H_
#define _CORECOLORCONVERSION_H_

#include "pvcore/common.h"


enum eKernels {
    // RGB <-> Luv
    rgb2luv_8u = 0,
    rgb2luv_32f,
    luv2rgb_8u,
    luv2rgb_32f,
    
    // RGB <-> HSV
    rgb2hsv_8u,
    rgbx2hsvx_8u,
    hsv2rgb_8u,
    
    // RGB  -> Gray
    rgb2gray_8u,
    rgb2gray_32f,
    rgb2gray_8uto32f,
    rgb2gray_32fto8u,
    
    // RGB <-> BGR
    rgb2bgr_32f,
    rgbx2bgrx_32f,
    rgb2bgr_8u,
    rgbx2bgrx_8u,
    
    // GRAY -> JET
    gray2jet_32f,
    gray2jet_8u,
    
    // BAYER format -> RGB
    bayergr2rgbx_8u,
    bayergr2rgb_8u,
    bayerbg2rgbx_8u,
    bayerbg2rgb_8u,
    bayergb2rgbx_8u,
    bayergb2rgb_8u,
    bayerrg2rgbx_8u,
    bayerrg2rgb_8u,
    
    // # kernels
    kNoKernels
};


namespace pvcore {

	void __convertColorWBCPU(const unsigned char* _src,
							 unsigned char* _dest,
							 unsigned int _width,
							 unsigned int _height,
							 unsigned int _pitchs,
							 unsigned int _pitchd,
							 unsigned int _kernel,
							 short _rcoeff, short _gcoeff, short _bcoeff,
							 unsigned int _threads );

    void __convertColorCPU(const unsigned char* _src,
                           unsigned char* _dest,
                           unsigned int _width,
                           unsigned int _height,
                           unsigned int _pitchs,
                           unsigned int _pitchd,
                           unsigned int _kernel,
                           unsigned int _threads);

#ifdef USE_CUDA
    cudaError_t __convertColorGPU(const unsigned char* _src,
                                  unsigned char* _dest,
                                  unsigned int _width,
                                  unsigned int _height,
                                  unsigned int _pitchs,
                                  unsigned int _pitchd,
                                  unsigned int _kernel,
                                  dim3 _blockDim);
#endif // USE_CUDA

#ifdef USE_OPENCL
    cl_int __convertColorGPU(const cl::Buffer* _src,
                             cl::Buffer* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _pitchs,
                             unsigned int _pitchd,
                             unsigned int _kernel,
                             dim3 _blockDim,
                             std::vector<cl::Device>& _devices = g_cl_devices );
#endif // USE_OPENCL
    
} // pvcore

#endif // _CORECOLORCONVERSION_H_
