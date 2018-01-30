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

#ifndef _COREFORMATTRANSFORM_H_
#define _COREFORMATTRANSFORM_H_

#include "pvcore/common.h"

#include <xmmintrin.h>
#include <immintrin.h>

namespace pvcore {
    
    // Alternatives:
    /*
     4-3
     3-4
     2-1
     1-2
     
    or
     
     x-x with changed type
     */
    
    template <class T1, class T2>
    void transformSame(const T1* _src,
                       T2* _dest,
                       int _width, int _height,
                       int _srcpitch, int _destpitch,
                       int _nchannels,
                       unsigned int _threads) {
        
        const __m128 scale = (sizeof(T1) == 4 ? _mm_set_ps(255.f, 255.f, 255.f, 255.f) :
                              _mm_set_ps(1.f/255.f, 1.f/255.f, 1.f/255.f, 1.f/255.f) );
        
        const __m128i zero8 = _mm_set1_epi8(0);
        
        int swidth = (_srcpitch/_nchannels) >> 4;
        
        const T1* src;
        T2* dest;
        
        for( int y=0; y<_height; ++y ) {
            src = _src + y*_srcpitch;
            dest = _dest + y*_destpitch;
            
            for( int x=0; x<swidth; ++x ) {
                // Float src image
                if( sizeof(T1) == 4 ) {

                    __m128 srcv1 = _mm_load_ps((const float*)src); src+=4;
                    __m128 srcv2 = _mm_load_ps((const float*)src); src+=4;
                    __m128 srcv3 = _mm_load_ps((const float*)src); src+=4;
                    __m128 srcv4 = _mm_load_ps((const float*)src); src+=4;
                    
                    __m128 srcv1scale = _mm_mul_ps(srcv1, scale);
                    __m128 srcv2scale = _mm_mul_ps(srcv2, scale);
                    __m128 srcv3scale = _mm_mul_ps(srcv3, scale);
                    __m128 srcv4scale = _mm_mul_ps(srcv4, scale);
                    
                    __m128i srcv1i = _mm_cvtps_epi32(srcv1scale);
                    __m128i srcv2i = _mm_cvtps_epi32(srcv2scale);
                    __m128i srcv3i = _mm_cvtps_epi32(srcv3scale);
                    __m128i srcv4i = _mm_cvtps_epi32(srcv4scale);
                    
                    __m128i srcv1sh = _mm_packs_epi32(srcv1i,srcv2i);
                    __m128i srcv2sh = _mm_packs_epi32(srcv3i,srcv4i);
                    
                    __m128i destv = _mm_packus_epi16(srcv1sh, srcv2sh);
                    
                    _mm_store_si128((__m128i*)dest, destv); dest+=16;
                    
                } else {

                    for( int i=0; i<_nchannels; ++i ) {
                        __m128i srcv = _mm_load_si128((const __m128i*)src); src+=16;
                        
                        __m128i srcv1sh = _mm_unpacklo_epi8(srcv, zero8);
                        __m128i srcv2sh = _mm_unpackhi_epi8(srcv, zero8);
                        
                        __m128i srcv1i = _mm_unpacklo_epi16(srcv1sh, zero8);
                        __m128i srcv2i = _mm_unpackhi_epi16(srcv1sh, zero8);
                        __m128i srcv3i = _mm_unpacklo_epi16(srcv2sh, zero8);
                        __m128i srcv4i = _mm_unpackhi_epi16(srcv2sh, zero8);
                        
                        __m128 srcv1 = _mm_cvtepi32_ps(srcv1i);
                        __m128 srcv2 = _mm_cvtepi32_ps(srcv2i);
                        __m128 srcv3 = _mm_cvtepi32_ps(srcv3i);
                        __m128 srcv4 = _mm_cvtepi32_ps(srcv4i);
                        
                        __m128 srcv1scale = _mm_mul_ps(srcv1, scale);
                        __m128 srcv2scale = _mm_mul_ps(srcv2, scale);
                        __m128 srcv3scale = _mm_mul_ps(srcv3, scale);
                        __m128 srcv4scale = _mm_mul_ps(srcv4, scale);
                        
                        _mm_store_ps((float*)dest, srcv1scale); dest+=4;
                        _mm_store_ps((float*)dest, srcv2scale); dest+=4;
                        _mm_store_ps((float*)dest, srcv3scale); dest+=4;
                        _mm_store_ps((float*)dest, srcv4scale); dest+=4;
                    }
                    
                }
            }
        }
        
        
    }
    
    
    
    template <class T1, class T2>
    void __formatTransformCPU(const T1* _src,
                              T2* _dest,
                              int _width, int _height,
                              int _srcpitch, int _destpitch,
                              int s1, int s2,
                              unsigned int _threads) {
        
        float factor = (sizeof(T1) == sizeof(T2) ? 1.0f :
                        (sizeof(T1)>sizeof(T2) ? 255.0f : 1.0f/255.0f));
        
        if( s1 == s2 ) {
            transformSame<T1,T2>(_src, _dest, _width, _height, _srcpitch, _destpitch, s1, _threads);
            return;
        }
        

        T2 alpha = (sizeof(T2) == 4 ? 1.0f : 255 );
        
        const T1* tsrc;
        T2* tdest;
        for( int y=0; y<_height; ++y ) {
            tsrc = _src + y*_srcpitch;
            tdest = _dest + y*_destpitch;
            for( int x=0; x<_width; ++x ) {
                *tdest = static_cast<T2>(factor* *tsrc); tdest++; tsrc++;
                // Should be optimized by compiler
                if( s1 == 1 && s2 == 2 ) {
                    *tdest = alpha; tdest++;
                }
                if( s1 == 2 && s2 == 1 ) {
                    tsrc++;
                }
                if( s1 >= 2 && s2 >= 2 ) {
                    *tdest = static_cast<T2>(factor* *tsrc); tdest++; tsrc++;
                }
                if( s1 >= 3 && s2 >= 3 ) {
                    *tdest = static_cast<T2>(factor* *tsrc); tdest++; tsrc++;
                }
                if( s1 == 3 && s2 == 4 ) {
                    *tdest = alpha; tdest++;
                }
                if( s1 == 4 && s2 == 3 ) {
                    tsrc++;
                }
                if( s1 >= 4 && s2 >= 4 ) {
                    *tdest = static_cast<T2>(factor* *tsrc); tdest++; tsrc++;
                }
            }
        }
        
        
        
    }
    
#ifdef USE_CUDA
    
#endif // USE_CUDA
    
#ifdef USE_OPENCL
    
#endif // USE_OPENCL
    
} // pvcore

#endif // _COREFORMATTRANSFORM_H_
