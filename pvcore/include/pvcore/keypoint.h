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

#ifndef _KEYPOINT_H_
#define _KEYPOINT_H_

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <npp.h>
#endif // USE_CUDA

#include <cstdio>
#include <cstdlib>

namespace pvcore {
    
    
    struct keypt {
        short x;
        short y;
    };
    
    struct keypt_vel {
        float x;
        float y;
    };
    
    struct featurept {
        unsigned char element[16];
    };
    
    struct opticalFlowStruct {
        // Keypoints array (location of keypoints)
        keypt* keypoints;
        // Keypoint velocities
        keypt_vel* keypoints_vel;
        // Number of keypoints
        int nkeypoints;
        // Keypoints image
        float* keypoints_img;
        // Number of rows in keypoints image
        size_t keypoints_img_pitch;
        // Corresponding features to keypoints
        featurept* features;
        // Number of rows in feature image
        size_t features_pitch;
        
        size_t keypointsImgBytesPerRow() {
            return keypoints_img_pitch*sizeof(float);
        }

        size_t featuresBytesPerRow() {
            return features_pitch*sizeof(featurept);
        }

        void print(int n) {

#ifdef USE_CUDA
	    keypt* keypoints_h = (keypt*)malloc(nkeypoints*sizeof(keypt));
            cudaMemcpy(keypoints_h, keypoints, nkeypoints*sizeof(keypt), cudaMemcpyDeviceToHost);
            
            for( int i=0; i<n; ++i ) {
                printf("%d, %d\n",keypoints_h[i].x,keypoints_h[i].y);
            }
            free(keypoints_h);
#endif
	    
        }
    };
    
    struct predictionStruct {
        // Keypoints array (location of keypoints)
        keypt* keypoints;
        int nkeypoints;
        featurept* features;
        size_t features_pitch;
        
        size_t featuresBytesPerRow() {
            return features_pitch*sizeof(featurept);
        }
        
        void print(int n) {
#ifdef USE_CUDA
            keypt* keypoints_h = (keypt*)malloc(nkeypoints*sizeof(keypt));
            cudaMemcpy(keypoints_h, keypoints, nkeypoints*sizeof(keypt), cudaMemcpyDeviceToHost);
            
            for( int i=0; i<n; ++i ) {
                printf("%d, %d\n",keypoints_h[i].x,keypoints_h[i].y);
            }
            free(keypoints_h);
#endif
        }
    };
    

    
#ifdef USE_CUDA
    cudaError_t __initOpticalFlowStructGPU(opticalFlowStruct* _src,
                                           int _width,
                                           int _height,
                                           int _nscales);
    
    cudaError_t __initPredictionStructGPU(predictionStruct* _src,
                                          int _width,
                                          int _height,
                                          int _nscales);
    
    
    cudaError_t __generateImagePyramid(const unsigned char* _src,
                                       unsigned int _width,
                                       unsigned int _height,
                                       unsigned int _pitchs,
                                       unsigned int _nscales,
                                       dim3 _blockDim);
    
    
    cudaError_t __predictPointsGPU(opticalFlowStruct* _opticalFlow,
                                   predictionStruct* _prediction,
                                   int _width, int _height, dim3 _blockDim);
    
    
    cudaError_t __matchGPU(opticalFlowStruct* _opticalFlow,
                           predictionStruct* _tkeypoints,
                           int _width, int _height, dim3 _blockDim);
    
    
    cudaError_t __extractFeaturesGPU(predictionStruct* _prediction,
                                     int _width,
                                     int _height,
                                     dim3 _blockDim);
    
    /** \brief Extracts keypoints given a specific type
     *  Extracts keypoints in an image and puts the results in 
     *  
     *  \param _kptStruct     Struct holding info about current keypoint locations, velocities, strength of keypoints
     *                        and feature points.
     *  \param _width         Width of image
     *  \param _height        Height of image
     *  \param _keypointType  Type of keypoint to extract (semidense, etc.)
     *  \param _blockDim      Block size for the cuda calls
     *
     *  \return               The state of the cuda calls
     */
    cudaError_t __extractKeypointsGPU(opticalFlowStruct* _kptStruct,
                                      unsigned int _width,
                                      unsigned int _height,
                                      unsigned int _keypointType,
                                      dim3 _blockDim );    
#endif // USE_CUDA


} // namespace pvcore


#endif // _KEYPOINT_H_
