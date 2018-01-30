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

#ifndef _COREHISTOGRAM_H_
#define _COREHISTOGRAM_H_

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif // USE_CUDA


namespace pvcore {
    
    /** \brief Creates histogram from an image
     *  The function takes an image together with a mask and creates
     *  a histogram based on valid pixels in the mask
     *
     *  \param _src The image
     *  \param _mask The mask,
     *  \param _hist The histogram being created
     *  \param _width Image width
     *  \param _height Image height
     *  \param _pitch Image row pitch (for alignment)
     *  \param _srcchannels Number of channels per pixel in the image
     *  \param _maskid Which value valid pixels should have in the mask
     *  \param _histx Histogram x-dimension, corresponding to first channel of image
     *  \param _histy Histogram y-dimension, corresponding to second channel of image
     *  \param _histz Histogram z-dimension, corresponding to third channel of image
     *  \param _histchannels User defined number of components in histogram
     *  \param _threads Number of threads to be used in TBB
     */
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
                                     );
    
    
    /** \brief Creates histogram from an image
     *  The function takes an image together with a mask and creates
     *  a histogram based on valid pixels in the mask
     *
     *  \param _src The image
     *  \param _mask The mask, stored as column indices for each row in the image
     *  \param _hist The histogram being created
     *  \param _width Image width
     *  \param _height Image height
     *  \param _pitch Image row pitch (for alignment)
     *  \param _srcchannels Number of channels per pixel in the image
     *  \param _maxr For each row, which is the max width
     *  \param _fg Tells whether to look to the left of the mask (true) or to the right (false)
     *  \param _histx Histogram x-dimension, corresponding to first channel of image
     *  \param _histy Histogram y-dimension, corresponding to second channel of image
     *  \param _histz Histogram z-dimension, corresponding to third channel of image
     *  \param _histchannels User defined number of components in histogram
     *  \param _threads Number of threads to be used in TBB
     */
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
                                          unsigned int _threads // Number of threads
                                          );
	
	
    /** \brief Creates histogram from an image
     *  The function takes an image together with a mask and creates
     *  a histogram based on valid pixels in the mask
     *
     *  \param _src The image
     *  \param _hist The histogram being created
     *  \param _dest Resulting probability image
     *  \param _width Image width
     *  \param _height Image height
     *  \param _pitch Image row pitch (for alignment)
     *  \param _srcchannels Number of channels per pixel in the image
     *  \param _maxr For each row, which is the max width
     *  \param _histx Histogram x-dimension, corresponding to first channel of image
     *  \param _histy Histogram y-dimension, corresponding to second channel of image
     *  \param _histz Histogram z-dimension, corresponding to third channel of image
     *  \param _histchannels User defined number of components in histogram
     *  \param _threads Number of threads to be used in TBB
     */
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
                                            );


    /** \brief Creates histogram from an image
     *  The function takes an image together with a mask and creates
     *  a histogram based on valid pixels in the mask
     *
     *  \param _src The image
     *  \param _hist The histogram being created
     *  \param _dest Resulting probability image
     *  \param _width Image width
     *  \param _height Image height
     *  \param _pitch Image row pitch (for alignment)
     *  \param _srcchannels Number of channels per pixel in the image
     *  \param _maxr For each row, which is the max width
     *  \param _histx Histogram x-dimension, corresponding to first channel of image
     *  \param _histy Histogram y-dimension, corresponding to second channel of image
     *  \param _histz Histogram z-dimension, corresponding to third channel of image
     *  \param _histchannels User defined number of components in histogram
     *  \param _threads Number of threads to be used in TBB
     */
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
                                       );
    

#ifdef USE_CUDA
    
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
                                          dim3& _blockDim // Number of threads
                                          );

    
#endif // USE_CUDA


} // namespace pvcore


#endif // _COREHISTOGRAM_H_
