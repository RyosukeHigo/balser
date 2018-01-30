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

#include "pvcore/cl.h"

#include "pvcore/colorconversion.h"


#include <iostream>

namespace pvcore {
    
    // Programs, one for each device type (CPU, NVIDIA, AMD, INTEL)
    std::map< cl::Device*, cl::Program> programs;
    
    // Kernels, one for each device type (CPU, NVIDIA, AMD, INTEL)
    std::map< cl::Device*, std::vector<cl::Kernel> > kernels_vec;
    
    
    // Loading kernels
    cl_int loadKernels( std::vector<cl::Device>& _devices ) {
        std::string flags = "";
        
        cl_int err = pvcore::CLLoadProgram( _devices, programs, "colorconversion", flags );
        
        for( int i=0; i<_devices.size(); ++i ) {
            cl::Device& device = _devices[i];
            std::vector<cl::Kernel>* kernels = &kernels_vec[&device];
            cl::Program* program = &programs[&device];
            
            (*kernels)[rgb2gray_8u] = cl::Kernel(*program, "rgb2gray_8u", &err );
            (*kernels)[rgb2hsv_8u] = cl::Kernel(*program, "rgb2hsv_8u", &err );
            (*kernels)[rgb2bgr_8u] = cl::Kernel(*program, "rgb2bgr_8u", &err );
            (*kernels)[bayergr2rgb_8u] = cl::Kernel(*program, "bayergr2rgb_8u", &err );
            (*kernels)[bayerrg2rgb_8u] = cl::Kernel(*program, "bayerrg2rgb_8u", &err );
            (*kernels)[bayergb2rgb_8u] = cl::Kernel(*program, "bayergb2rgb_8u", &err );
            (*kernels)[bayerbg2rgb_8u] = cl::Kernel(*program, "bayerbg2rgb_8u", &err );
        }
		
        if( err == CL_SUCCESS ) {
            std::cout << "Successfully loaded kernels\n";
        }
        
        return err;
    }
    
    
    cl_int __convertColorGPU(const cl::Buffer* _src,
                             cl::Buffer* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _pitchs,
                             unsigned int _pitchd,
                             unsigned int _kernel,
                             dim3 _blockDim,
                             std::vector<cl::Device>& _devices ) {
        
        cl::Device& device = _devices[0];
        
        static bool kernelsLoaded;
        
        cl_int err = CL_SUCCESS;
        
        if( !kernelsLoaded ) {
            for( int i=0; i<_devices.size(); ++i ) {
                kernels_vec[&_devices[i]].resize(kNoKernels);
            }
            err = loadKernels(_devices);
            if( err != CL_SUCCESS ) {
                return err;
            }
            kernelsLoaded = true;
        }
        
        std::vector<cl::Kernel>* kernels = &kernels_vec[&device];

        // To modify kernel parameters
        int pixels_per_x = 4;
        int pixels_per_y = 1;
        int padding = 0;
        int channels = 1;
        
        // Local memory
        if( _kernel == bayergr2rgbx_8u ||
           _kernel == bayergr2rgb_8u ||
           _kernel == bayerrg2rgbx_8u ||
           _kernel == bayerrg2rgb_8u ||
           _kernel == bayergb2rgbx_8u ||
           _kernel == bayergb2rgb_8u ||
           _kernel == bayerbg2rgbx_8u ||
           _kernel == bayerbg2rgb_8u) {
            padding = 1;
            pixels_per_y = 2;
        }
        if( _kernel == rgb2hsv_8u ||
           _kernel == rgb2bgr_8u) {
            channels = 3;
        }
        
        
        // Modify kernel parameters
        size_t lWorksize[3] = {(size_t)_blockDim.x,(size_t)_blockDim.y,1};
        size_t gWorksize[3] = {GLOBAL_SIZE(_width, lWorksize[0])/pixels_per_x,GLOBAL_SIZE(_height, lWorksize[1])/pixels_per_y,1};
        
        size_t pitchl = (2*padding+lWorksize[0])*channels;
        cl::LocalSpaceArg local_memsize = cl::Local(4*pitchl*(2*lWorksize[1]+2*padding));
        
        
        unsigned int argidx = 0;
        err  = (*kernels)[_kernel].setArg(argidx++, *_src);
        err |= (*kernels)[_kernel].setArg(argidx++, *_dest);
        err |= (*kernels)[_kernel].setArg(argidx++, local_memsize);
        err |= (*kernels)[_kernel].setArg(argidx++, _width);
        err |= (*kernels)[_kernel].setArg(argidx++, _height);
        err |= (*kernels)[_kernel].setArg(argidx++, _pitchs/4);
        err |= (*kernels)[_kernel].setArg(argidx++, _pitchd/4);
        err |= (*kernels)[_kernel].setArg(argidx++, pitchl);
        
        if( err != CL_SUCCESS ) {
            std::cout << "Error setting arguments for kernel " << _kernel << std::endl;
            return err;
        }
        
        err = g_cl_command_queue[&device]->enqueueNDRangeKernel((*kernels)[_kernel],
                                                                cl::NullRange,
                                                                cl::NDRange(gWorksize[0],gWorksize[1],gWorksize[2]),
                                                                cl::NDRange(lWorksize[0],lWorksize[1],lWorksize[2]) );

        
        if( err != CL_SUCCESS ) {
            std::cout << "Error enqueuing kernel" << std::endl;
            return err;
        }
        
        
        /*switch( _kernel ) {
         // ================ RGB2BGR ==============
         case rgb2bgr_8u:
         gridDim.x = GLOBAL_SIZE( _width/4, _blockDim.x );
         gridDim.y = GLOBAL_SIZE( _height, _blockDim.y );
         _rgb2bgr_8u<<< gridDim,_blockDim, shmSize >>>( (uchar4*)_src,(uchar4*)_dest, _width/4, _height, _pitchs/4, _pitchd/4, 3*_blockDim.x );
         break;
         
         default:
         break;
         }*/
        
        return err;
        
    }
    
} // namespace pvcore

