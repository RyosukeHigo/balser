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

#include "pvcore/integralimage.h"

#include "pvcore/common.h"

// SSE
#include <immintrin.h>

// TBB
#include <tbb/tbb.h>

#include <iostream>
#include <algorithm>

#ifdef _WIN32
#undef max
#undef min
#endif

#define SIMPLE 1


//void bobo( const unsigned char* _src,
//          int* _sum,
//          int _srcpitch,
//          int _destpitch,
//          int _xdim,
//          int _ydim  ) {
//    
//    int x,y;
//    
//    x = 0;
//    const unsigned char* src;
//    int* sum;
//    for( ; x < _srcpitch; x+=16 ) {
//        src = _src+x;
//        sum = _sum+_destpitch+1+x;
//        __m128i srcv = _mm_load_si128((const __m128i*)src);
//        
//        __m128i srcv1sh = _mm_unpacklo_epi8(srcv, _mm_set1_epi8(0));
//        __m128i srcv2sh = _mm_unpackhi_epi8(srcv, _mm_set1_epi8(0));
//        
//        __m128i srcv1i = _mm_unpacklo_epi16(srcv1sh, _mm_set1_epi8(0));
//        _mm_storeu_si128((__m128i*)sum,srcv1i);
//        
//        __m128i srcv2i = _mm_unpackhi_epi16(srcv1sh, _mm_set1_epi8(0));
//        _mm_storeu_si128((__m128i*)sum+1,srcv2i);
//        
//        __m128i srcv3i = _mm_unpacklo_epi16(srcv2sh, _mm_set1_epi8(0));
//        _mm_storeu_si128((__m128i*)sum+2,srcv3i);
//        
//        __m128i srcv4i = _mm_unpackhi_epi16(srcv2sh, _mm_set1_epi8(0));
//        _mm_storeu_si128((__m128i*)sum+3,srcv4i);
//        
//        
//        src += _srcpitch;
//        sum += _destpitch;
//        
//        for( y = 1; y < _ydim; y++, src += _srcpitch, sum += _destpitch )
//        {
//            srcv = _mm_load_si128((const __m128i*)src);
//            
//            srcv1sh = _mm_unpacklo_epi8(srcv, _mm_set1_epi8(0));
//            srcv2sh = _mm_unpackhi_epi8(srcv, _mm_set1_epi8(0));
//            
//            __m128i srcv1ib = _mm_unpacklo_epi16(srcv1sh, _mm_set1_epi8(0));
//            srcv1i = _mm_add_epi32(srcv1i, srcv1ib);
//            _mm_storeu_si128((__m128i*)sum,srcv1i);
//            
//            __m128i srcv2ib = _mm_unpackhi_epi16(srcv1sh, _mm_set1_epi8(0));
//            srcv2i = _mm_add_epi32(srcv2i, srcv2ib);
//            _mm_storeu_si128((__m128i*)sum+1,srcv2i);
//            
//            __m128i srcv3ib = _mm_unpacklo_epi16(srcv2sh, _mm_set1_epi8(0));
//            srcv3i = _mm_add_epi32(srcv3i, srcv3ib);
//            _mm_storeu_si128((__m128i*)sum+2,srcv3i);
//            
//            __m128i srcv4ib = _mm_unpackhi_epi16(srcv2sh, _mm_set1_epi8(0));
//            srcv4i = _mm_add_epi32(srcv4i, srcv4ib);
//            _mm_storeu_si128((__m128i*)sum+3,srcv4i);
//            
//        }
//    }
//    
//}
//


inline void transpose8_ps(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3,
                          __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7) {
//inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
//                          __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {

    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(_mm256_castsi256_ps(row0), _mm256_castsi256_ps(row1));
    __t1 = _mm256_unpackhi_ps(_mm256_castsi256_ps(row0), _mm256_castsi256_ps(row1));
    __t2 = _mm256_unpacklo_ps(_mm256_castsi256_ps(row2), _mm256_castsi256_ps(row3));
    __t3 = _mm256_unpackhi_ps(_mm256_castsi256_ps(row2), _mm256_castsi256_ps(row3));
    __t4 = _mm256_unpacklo_ps(_mm256_castsi256_ps(row4), _mm256_castsi256_ps(row5));
    __t5 = _mm256_unpackhi_ps(_mm256_castsi256_ps(row4), _mm256_castsi256_ps(row5));
    __t6 = _mm256_unpacklo_ps(_mm256_castsi256_ps(row6), _mm256_castsi256_ps(row7));
    __t7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(row6), _mm256_castsi256_ps(row7));
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
	
}


tbb::mutex mtx;

class INTEGRAL_FIRST_STEP_32S_TBB {
    
    const unsigned char* _src;
    int* _sum;
    int _srcpitch;
    int _destpitch;
    int _xdim, _ydim;
    float wi_per_thread;
    
public:
    INTEGRAL_FIRST_STEP_32S_TBB( const unsigned char* __src,
                                int* __sum,
                                int __srcpitch,
                                int __destpitch,
                                int __xdim,
                                int __ydim,
                                float _wi_per_thread) : _src(__src), _sum(__sum), _srcpitch(__srcpitch),
    _destpitch(__destpitch), _xdim(__xdim), _ydim(__ydim), wi_per_thread(_wi_per_thread)
    {}
    
    inline void operator()( const tbb::blocked_range<size_t>& r ) const {
        
#ifdef USE_AVX2
        int y;
        
        int ystart = 8*GET_START(r.begin(), wi_per_thread);//(int)wi_per_thread*(int)r.begin();
        int yend = 8*GET_STOP(r.end(), wi_per_thread);//(int)wi_per_thread*(int)r.end();
        
//        mtx.lock();
//        std::cout << ystart << " " << yend << std::endl;
//        mtx.unlock();
        
        __m256i idxv = _mm256_set_epi32(7*_srcpitch, 6*_srcpitch, 5*_srcpitch, 4*_srcpitch, 3*_srcpitch, 2*_srcpitch, 1*_srcpitch, 0);
        __m256i shufflev = _mm256_set_epi8(0x0f, 0x0b, 0x07, 0x03,
                                           0x0e, 0x0a, 0x06, 0x02,
                                           0x0d, 0x09, 0x05, 0x01,
                                           0x0c, 0x08, 0x04, 0x00,
                                           0x0f, 0x0b, 0x07, 0x03,
                                           0x0e, 0x0a, 0x06, 0x02,
                                           0x0d, 0x09, 0x05, 0x01,
                                           0x0c, 0x08, 0x04, 0x00);
        
        
        for( y=ystart; y<yend; y+=8 ) {

            const unsigned char* tsrc = _src + y*_srcpitch;
            int* tdest = _sum + _destpitch*(y+1) + 1;
            __m256i sum = _mm256_set1_epi32(0);
            
            __m256i srcv_1 = _mm256_i32gather_epi32((int*)tsrc, idxv, 1); tsrc += 4;
            __m256i srcv_2 = _mm256_i32gather_epi32((int*)tsrc, idxv, 1); tsrc += 4;
            
            __m256i src_sh_1 = _mm256_shuffle_epi8(srcv_1, shufflev);
            __m256i src_sh_2 = _mm256_shuffle_epi8(srcv_2, shufflev);
            
            __m256i src_sh_1_lo = _mm256_unpacklo_epi8(src_sh_1, _mm256_set1_epi8(0));
            __m256i src_sh_1_hi = _mm256_unpackhi_epi8(src_sh_1, _mm256_set1_epi8(0));
            
            __m256i src_sh_2_lo = _mm256_unpacklo_epi8(src_sh_2, _mm256_set1_epi8(0));
            __m256i src_sh_2_hi = _mm256_unpackhi_epi8(src_sh_2, _mm256_set1_epi8(0));
            
            __m256i srcv0i = _mm256_unpacklo_epi16(src_sh_1_lo, _mm256_set1_epi8(0));
            __m256i sum0 = _mm256_add_epi32(sum, srcv0i);
            
            __m256i srcv1i = _mm256_unpackhi_epi16(src_sh_1_lo, _mm256_set1_epi8(0));
            __m256i sum1 = _mm256_add_epi32(sum0, srcv1i);
            
            __m256i srcv2i = _mm256_unpacklo_epi16(src_sh_1_hi, _mm256_set1_epi8(0));
            __m256i sum2 = _mm256_add_epi32(sum1, srcv2i);
            
            __m256i srcv3i = _mm256_unpackhi_epi16(src_sh_1_hi, _mm256_set1_epi8(0));
            __m256i sum3 = _mm256_add_epi32(sum2, srcv3i);
            
            __m256i srcv4i = _mm256_unpacklo_epi16(src_sh_2_lo, _mm256_set1_epi8(0));
            __m256i sum4 = _mm256_add_epi32(sum3, srcv4i);
            
            __m256i srcv5i = _mm256_unpackhi_epi16(src_sh_2_lo, _mm256_set1_epi8(0));
            __m256i sum5 = _mm256_add_epi32(sum4, srcv5i);
            
            __m256i srcv6i = _mm256_unpacklo_epi16(src_sh_2_hi, _mm256_set1_epi8(0));
            __m256i sum6 = _mm256_add_epi32(sum5, srcv6i);
            
            __m256i srcv7i = _mm256_unpackhi_epi16(src_sh_2_hi, _mm256_set1_epi8(0));
            __m256i sum7 = _mm256_add_epi32(sum6, srcv7i);
            
            sum = sum7;
            
            // TRANSPOSE
            transpose8_ps(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
            
            
            for( int x = 8; x<_srcpitch; x+=8 ) {
                
                __m256i srcv_1 = _mm256_i32gather_epi32((int*)tsrc, idxv, 1); tsrc += 4;
                __m256i srcv_2 = _mm256_i32gather_epi32((int*)tsrc, idxv, 1); tsrc += 4;
                
                __m256i src_sh_1 = _mm256_shuffle_epi8(srcv_1, shufflev);
                __m256i src_sh_2 = _mm256_shuffle_epi8(srcv_2, shufflev);
                
                __m256i src_sh_1_lo = _mm256_unpacklo_epi8(src_sh_1, _mm256_set1_epi8(0));
                __m256i src_sh_1_hi = _mm256_unpackhi_epi8(src_sh_1, _mm256_set1_epi8(0));
                
                __m256i src_sh_2_lo = _mm256_unpacklo_epi8(src_sh_2, _mm256_set1_epi8(0));
                __m256i src_sh_2_hi = _mm256_unpackhi_epi8(src_sh_2, _mm256_set1_epi8(0));
                
                __m256i srcv0i = _mm256_unpacklo_epi16(src_sh_1_lo, _mm256_set1_epi8(0));
                _mm256_storeu_si256((__m256i*)tdest, sum0);
                sum0 = _mm256_add_epi32(sum, srcv0i);
                
                __m256i srcv1i = _mm256_unpackhi_epi16(src_sh_1_lo, _mm256_set1_epi8(0));
                _mm256_storeu_si256((__m256i*)(tdest+_destpitch),   sum1);
                sum1 = _mm256_add_epi32(sum0, srcv1i);
                
                __m256i srcv2i = _mm256_unpacklo_epi16(src_sh_1_hi, _mm256_set1_epi8(0));
                _mm256_storeu_si256((__m256i*)(tdest+_destpitch*2), sum2);
                sum2 = _mm256_add_epi32(sum1, srcv2i);
                
                __m256i srcv3i = _mm256_unpackhi_epi16(src_sh_1_hi, _mm256_set1_epi8(0));
                _mm256_storeu_si256((__m256i*)(tdest+_destpitch*3), sum3);
                sum3 = _mm256_add_epi32(sum2, srcv3i);
                
                __m256i srcv4i = _mm256_unpacklo_epi16(src_sh_2_lo, _mm256_set1_epi8(0));
                _mm256_storeu_si256((__m256i*)(tdest+_destpitch*4), sum4);
                sum4 = _mm256_add_epi32(sum3, srcv4i);
                
                __m256i srcv5i = _mm256_unpackhi_epi16(src_sh_2_lo, _mm256_set1_epi8(0));
                _mm256_storeu_si256((__m256i*)(tdest+_destpitch*5), sum5);
                sum5 = _mm256_add_epi32(sum4, srcv5i);
                
                __m256i srcv6i = _mm256_unpacklo_epi16(src_sh_2_hi, _mm256_set1_epi8(0));
                _mm256_storeu_si256((__m256i*)(tdest+_destpitch*6), sum6);
                sum6 = _mm256_add_epi32(sum5, srcv6i);
                
                __m256i srcv7i = _mm256_unpackhi_epi16(src_sh_2_hi, _mm256_set1_epi8(0));
                _mm256_storeu_si256((__m256i*)(tdest+_destpitch*7), sum7);
                sum7 = _mm256_add_epi32(sum6, srcv7i);
                
                sum = sum7;
                
                // TRANSPOSE
				transpose8_ps(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
				
                tdest += 8;
            }
            
            _mm256_storeu_si256((__m256i*)tdest, sum0);
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch),   sum1);
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*2), sum2);
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*3), sum3);
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*4), sum4);
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*5), sum5);
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*6), sum6);
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*7), sum7);
            
        }
#else
        int x,y;
        
        x = (int)wi_per_thread*(int)r.begin()*16;
        int xend = (int)wi_per_thread*(int)r.end()*16;
        
        const unsigned char* src;
        int* sum;
        for( ; x < xend; x+=16 ) {
            src = _src+x;
            sum = _sum+_destpitch+1+x;
            __m128i srcv = _mm_load_si128((const __m128i*)src);
            
            __m128i srcv1sh = _mm_unpacklo_epi8(srcv, _mm_set1_epi8(0));
            __m128i srcv2sh = _mm_unpackhi_epi8(srcv, _mm_set1_epi8(0));
            
            __m128i srcv1i = _mm_unpacklo_epi16(srcv1sh, _mm_set1_epi8(0));
            _mm_storeu_si128((__m128i*)sum,srcv1i);
            
            __m128i srcv2i = _mm_unpackhi_epi16(srcv1sh, _mm_set1_epi8(0));
            _mm_storeu_si128((__m128i*)sum+1,srcv2i);
            
            __m128i srcv3i = _mm_unpacklo_epi16(srcv2sh, _mm_set1_epi8(0));
            _mm_storeu_si128((__m128i*)sum+2,srcv3i);
            
            __m128i srcv4i = _mm_unpackhi_epi16(srcv2sh, _mm_set1_epi8(0));
            _mm_storeu_si128((__m128i*)sum+3,srcv4i);
            
            
            src += _srcpitch;
            sum += _destpitch;
            
            for( y = 1; y < _ydim; y++, src += _srcpitch, sum += _destpitch )
            {
                srcv = _mm_load_si128((const __m128i*)src);
                
                srcv1sh = _mm_unpacklo_epi8(srcv, _mm_set1_epi8(0));
                srcv2sh = _mm_unpackhi_epi8(srcv, _mm_set1_epi8(0));
                
                __m128i srcv1ib = _mm_unpacklo_epi16(srcv1sh, _mm_set1_epi8(0));
                srcv1i = _mm_add_epi32(srcv1i, srcv1ib);
                _mm_storeu_si128((__m128i*)sum,srcv1i);
                
                __m128i srcv2ib = _mm_unpackhi_epi16(srcv1sh, _mm_set1_epi8(0));
                srcv2i = _mm_add_epi32(srcv2i, srcv2ib);
                _mm_storeu_si128((__m128i*)sum+1,srcv2i);
                
                __m128i srcv3ib = _mm_unpacklo_epi16(srcv2sh, _mm_set1_epi8(0));
                srcv3i = _mm_add_epi32(srcv3i, srcv3ib);
                _mm_storeu_si128((__m128i*)sum+2,srcv3i);
                
                __m128i srcv4ib = _mm_unpackhi_epi16(srcv2sh, _mm_set1_epi8(0));
                srcv4i = _mm_add_epi32(srcv4i, srcv4ib);
                _mm_storeu_si128((__m128i*)sum+3,srcv4i);
                
            }
        }
#endif
        
    }
};


class INTEGRAL_FIRST_STEP_32F_TBB {
    
    const unsigned char* src;
    float* sum;
    int srcpitch;
    int destpitch;
    int xdim, ydim;
    
public:
#if SIMPLE
    INTEGRAL_FIRST_STEP_32F_TBB( const unsigned char* _src,
                                float* _sum,
                                int _srcpitch,
                                int _destpitch,
                                int _xdim,
                                int _ydim ) : src(_src), sum(_sum), srcpitch(_srcpitch),
    destpitch(_destpitch), xdim(_xdim), ydim(_ydim)
    {}
    
    
    inline void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        size_t x,y;
        
        x = r.begin();
        const unsigned char* csrc;
        float* csum;
        for( ; x < r.end()-15; x+=16 ) {
            csrc = src+x;
            csum = sum+destpitch+1+x;
            __m128i srcv = _mm_load_si128((const __m128i*)csrc);
            
            __m128i srcv1sh = _mm_unpacklo_epi8(srcv, _mm_set1_epi8(0));
            __m128i srcv2sh = _mm_unpackhi_epi8(srcv, _mm_set1_epi8(0));
            
            __m128 srcv1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(srcv1sh, _mm_set1_epi8(0)));
            __m128 srcv2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(srcv1sh, _mm_set1_epi8(0)));
            __m128 srcv3 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(srcv2sh, _mm_set1_epi8(0)));
            __m128 srcv4 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(srcv2sh, _mm_set1_epi8(0)));
            
            _mm_storeu_ps(csum,srcv1);
            _mm_storeu_ps(csum+4,srcv2);
            _mm_storeu_ps(csum+8,srcv3);
            _mm_storeu_ps(csum+12,srcv4);
            
            csrc += srcpitch;
            csum += destpitch;
            
            for( y = 1; y < ydim; y++, csrc += srcpitch, csum += destpitch )
            {
                srcv = _mm_load_si128((const __m128i*)csrc);
                
                srcv1sh = _mm_unpacklo_epi8(srcv, _mm_set1_epi8(0));
                srcv2sh = _mm_unpackhi_epi8(srcv, _mm_set1_epi8(0));
                
                __m128 srcv1b = _mm_cvtepi32_ps(_mm_unpacklo_epi16(srcv1sh, _mm_set1_epi8(0)));
                __m128 srcv2b = _mm_cvtepi32_ps(_mm_unpackhi_epi16(srcv1sh, _mm_set1_epi8(0)));
                
                __m128 srcv3b = _mm_cvtepi32_ps(_mm_unpacklo_epi16(srcv2sh, _mm_set1_epi8(0)));
                __m128 srcv4b = _mm_cvtepi32_ps(_mm_unpackhi_epi16(srcv2sh, _mm_set1_epi8(0)));
                
                srcv1 = _mm_add_ps(srcv1,srcv1b);
                _mm_storeu_ps(csum,srcv1);
                srcv2 = _mm_add_ps(srcv2,srcv2b);
                _mm_storeu_ps(csum+4,srcv2);
                srcv3 = _mm_add_ps(srcv3,srcv3b);
                _mm_storeu_ps(csum+8,srcv3);
                srcv4 = _mm_add_ps(srcv4,srcv4b);
                _mm_storeu_ps(csum+12,srcv4);
            }
        }
    }
    
#else
    INTEGRAL_FIRST_STEP_32F_TBB( const unsigned char* _src,
                                float* _sum,
                                int _srcpitch,
                                int _destpitch,
                                int _xdim,
                                int _ydim ) : src(_src), sum(_sum), srcpitch(_srcpitch),
    destpitch(_destpitch), xdim(_xdim), ydim(_ydim)
    {}

    inline void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        int x,y;
        
        x = (int)r.begin();
        const unsigned char* csrc;
        float* csum;
        for( ; x < r.end()-15; x+=16 ) {
            // Get the correct addresses
            csrc = src + x;
            csum = sum + x*destpitch/4;
            
            // Loads block of bytes according to:
            // 0 1 2 3 4 5 6 7 8 9 a b c d e f
            
            // Unpacks them in float vectors:
            // [0 1 2 3] [4 5 6 7] etc.
            
            // Adds with previous vector
            
            // Permutes each block of 4 vectors and stores
            
            // Looking blockwise, the before and after matrices will look:
            // Before: [A B C] => [A' B' C']^T
            //         [D E F]    [D' E' F']
            
            // Load the source vectors
            __m128i srcv = _mm_load_si128((const __m128i*)csrc);
            
            // Unpack vectors 0
            __m128i srcv0sh = _mm_unpacklo_epi8(srcv, _mm_set1_epi8(0));
            __m128i srcv1sh = _mm_unpackhi_epi8(srcv, _mm_set1_epi8(0));
            
            __m128 srcv0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(srcv0sh, _mm_set1_epi8(0)));
            __m128 srcv1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(srcv0sh, _mm_set1_epi8(0)));
            __m128 srcv2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(srcv1sh, _mm_set1_epi8(0)));
            __m128 srcv3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(srcv1sh, _mm_set1_epi8(0)));
            
            // Step forward in source vector
            csrc += srcpitch;
            
            __m128 srcv0b;
            __m128 srcv1b;
            __m128 srcv2b;
            __m128 srcv3b;
            
            for( y = 1; y < ydim; y+=1, csrc += srcpitch )
            {
                // Load the source vectors
                __m128i srcvb = _mm_load_si128((const __m128i*)csrc);
                
                // Unpack
                __m128i srcv0shb = _mm_unpacklo_epi8(srcvb, _mm_set1_epi8(0));
                __m128i srcv1shb = _mm_unpackhi_epi8(srcvb, _mm_set1_epi8(0));
                
                srcv0b = _mm_cvtepi32_ps(_mm_unpacklo_epi16(srcv0shb, _mm_set1_epi8(0)));
                srcv1b = _mm_cvtepi32_ps(_mm_unpackhi_epi16(srcv0shb, _mm_set1_epi8(0)));
                srcv2b = _mm_cvtepi32_ps(_mm_unpacklo_epi16(srcv1shb, _mm_set1_epi8(0)));
                srcv3b = _mm_cvtepi32_ps(_mm_unpackhi_epi16(srcv1shb, _mm_set1_epi8(0)));
                
                
                // Adds the previous vectors to vectors b
                srcv0b = _mm_add_ps(srcv0, srcv0b);
                srcv1b = _mm_add_ps(srcv1, srcv1b);
                srcv2b = _mm_add_ps(srcv2, srcv2b);
                srcv3b = _mm_add_ps(srcv3, srcv3b);
                
                // Now we can transpose and store previous vectors
                //_MM_TRANSPOSE4_PS(srcv0, srcv1, srcv2, srcv3);
                
                _mm_store_ps(csum,srcv0); csum+=destpitch;
                srcv0 = srcv0b;
                _mm_store_ps(csum,srcv1); csum+=destpitch;
                srcv1 = srcv1b;
                _mm_store_ps(csum,srcv2); csum+=destpitch;
                srcv2 = srcv2b;
                _mm_store_ps(csum,srcv3);
                srcv3 = srcv3b;
                csum += (-3*destpitch+4);
            }
            
            _MM_TRANSPOSE4_PS(srcv0b, srcv0b, srcv0b, srcv0b);
            
            _mm_store_ps(csum,srcv0b); csum +=destpitch;
            _mm_store_ps(csum,srcv1b); csum +=destpitch;
            _mm_store_ps(csum,srcv2b); csum +=destpitch;
            _mm_store_ps(csum,srcv3b);
            
        }
    }
#endif
};




class INTEGRAL_SECOND_STEP_32S_TBB {
    
    int* _sum;
    int _destpitch;
    int _xdim, _ydim;
    float wi_per_thread;
    
public:
    INTEGRAL_SECOND_STEP_32S_TBB(int* __sum,
                                 int __destpitch,
                                 int __xdim,
                                 int __ydim,
                                 float _wi_per_thread = 1) : _sum(__sum), _destpitch(__destpitch), _xdim(__xdim), _ydim(__ydim),  wi_per_thread(_wi_per_thread)
    {}
    
    inline void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        
        
#ifdef USE_AVX2
        int xstart = 8*GET_START(r.begin(), wi_per_thread);//(int)wi_per_thread*(int)r.begin()*8;
        int xend = 8*GET_STOP(r.end(), wi_per_thread);//(int)wi_per_thread*(int)r.end()*8;
                
        for( int x = xstart; x<xend; x+=8 ) {
            int* tdest = _sum + _destpitch + x;
            __m256i sum = _mm256_set1_epi32(0);
            for( int y=0; y<_ydim; y++ ) {
                __m256i srcv = _mm256_load_si256((const __m256i*)tdest);
                
                sum = _mm256_add_epi32(sum, srcv);
                
                _mm256_store_si256((__m256i*)tdest, sum); tdest += _destpitch;
            }
        }
#else
        int yend = std::min((int)r.end(),_ydim);
        
        int* sum = _sum+(r.begin()+1)*_destpitch+1;
        for( int y = (int)r.begin(); y < yend; y++, sum += _destpitch ) {
            int s = sum[-1] = 0;
            for( int x = 0; x < _xdim; x+=1 ) {
                s += sum[x];
                sum[x] = s;
            }
        }
#endif

    }
    
 
};



#ifdef USE_AVX2
void first_step_gather(const unsigned char* _src,
                       int* _dest,
                       int _srcpitch,
                       int _destpitch,
                       int _width,
                       int _height) {
    
    /*
     FIRST STEP
     4x4 unsigned chars per lane
     - shuffle and unpack
     - integral add
     - store
     
     gather load:
     i0a i0b i0c i0d | i1a i1b i1c i1d | i2a i2b i2c i2d | i3a i3b i3c i3d
     
     shuffle:
     i0a i1a i2a i3a | i0b i1b i2b i3b | i0c i1c i2c i3c | i0d i1d i2d i3d
     
     unpack x 2
     i0a | i1a | i2a | i3a
     ...
     ...
     i0d | i1d | i2d | i3d
     
     integral sum
     i0a | i1a | i2a | i3a
     i0a+i0b | ... | i3a+i3b
     ...
     i0a+i0b+i0c+i0d | i1a+i1b+i1c+i1d | i2a+i2b+i2c+i2d | i3a+i3b+i3c+i3d
     
     store to memory, keep last sum in register, add with next load
    */

    __m256i idxv = _mm256_set_epi32(7*_srcpitch, 6*_srcpitch, 5*_srcpitch, 4*_srcpitch, 3*_srcpitch, 2*_srcpitch, 1*_srcpitch, 0);
    __m256i shufflev = _mm256_set_epi8(0x0f, 0x0b, 0x07, 0x03,
                                       0x0e, 0x0a, 0x06, 0x02,
                                       0x0d, 0x09, 0x05, 0x01,
                                       0x0c, 0x08, 0x04, 0x00,
                                       0x0f, 0x0b, 0x07, 0x03,
                                       0x0e, 0x0a, 0x06, 0x02,
                                       0x0d, 0x09, 0x05, 0x01,
                                       0x0c, 0x08, 0x04, 0x00);
    
    
    for( int y=0; y<_height; y+=8 ) {
        const unsigned char* tsrc = _src + y*_srcpitch;
        int* tdest = _dest + _destpitch*(y+1) + 1;
        __m256i sum = _mm256_set1_epi32(0);
        
        __m256i srcv_1 = _mm256_i32gather_epi32((int*)tsrc, idxv, 1); tsrc += 4;
        __m256i srcv_2 = _mm256_i32gather_epi32((int*)tsrc, idxv, 1); tsrc += 4;
        
        __m256i src_sh_1 = _mm256_shuffle_epi8(srcv_1, shufflev);
        __m256i src_sh_2 = _mm256_shuffle_epi8(srcv_2, shufflev);
        
        __m256i src_sh_1_lo = _mm256_unpacklo_epi8(src_sh_1, _mm256_set1_epi8(0));
        __m256i src_sh_1_hi = _mm256_unpackhi_epi8(src_sh_1, _mm256_set1_epi8(0));
        
        __m256i src_sh_2_lo = _mm256_unpacklo_epi8(src_sh_2, _mm256_set1_epi8(0));
        __m256i src_sh_2_hi = _mm256_unpackhi_epi8(src_sh_2, _mm256_set1_epi8(0));
        
        __m256i srcv0i = _mm256_unpacklo_epi16(src_sh_1_lo, _mm256_set1_epi8(0));
        __m256i sum0 = _mm256_add_epi32(sum, srcv0i);
        
        __m256i srcv1i = _mm256_unpackhi_epi16(src_sh_1_lo, _mm256_set1_epi8(0));
        __m256i sum1 = _mm256_add_epi32(sum0, srcv1i);
        
        __m256i srcv2i = _mm256_unpacklo_epi16(src_sh_1_hi, _mm256_set1_epi8(0));
        __m256i sum2 = _mm256_add_epi32(sum1, srcv2i);
        
        __m256i srcv3i = _mm256_unpackhi_epi16(src_sh_1_hi, _mm256_set1_epi8(0));
        __m256i sum3 = _mm256_add_epi32(sum2, srcv3i);
        
        __m256i srcv4i = _mm256_unpacklo_epi16(src_sh_2_lo, _mm256_set1_epi8(0));
        __m256i sum4 = _mm256_add_epi32(sum3, srcv4i);
        
        __m256i srcv5i = _mm256_unpackhi_epi16(src_sh_2_lo, _mm256_set1_epi8(0));
        __m256i sum5 = _mm256_add_epi32(sum4, srcv5i);
        
        __m256i srcv6i = _mm256_unpacklo_epi16(src_sh_2_hi, _mm256_set1_epi8(0));
        __m256i sum6 = _mm256_add_epi32(sum5, srcv6i);
        
        __m256i srcv7i = _mm256_unpackhi_epi16(src_sh_2_hi, _mm256_set1_epi8(0));
        __m256i sum7 = _mm256_add_epi32(sum6, srcv7i);
        
        sum = sum7;
        
        // TRANSPOSE
        transpose8_ps(sum0,sum1,sum2,sum3,
					  sum4,sum5,sum6,sum7);
        
        
        for( int x=8; x<_srcpitch; x+=8 ) {
            
            __m256i srcv_1 = _mm256_i32gather_epi32((int*)tsrc, idxv, 1); tsrc += 4;
            __m256i srcv_2 = _mm256_i32gather_epi32((int*)tsrc, idxv, 1); tsrc += 4;
            
            __m256i src_sh_1 = _mm256_shuffle_epi8(srcv_1, shufflev);
            __m256i src_sh_2 = _mm256_shuffle_epi8(srcv_2, shufflev);
            
            __m256i src_sh_1_lo = _mm256_unpacklo_epi8(src_sh_1, _mm256_set1_epi8(0));
            __m256i src_sh_1_hi = _mm256_unpackhi_epi8(src_sh_1, _mm256_set1_epi8(0));
            
            __m256i src_sh_2_lo = _mm256_unpacklo_epi8(src_sh_2, _mm256_set1_epi8(0));
            __m256i src_sh_2_hi = _mm256_unpackhi_epi8(src_sh_2, _mm256_set1_epi8(0));
            
            __m256i srcv0i = _mm256_unpacklo_epi16(src_sh_1_lo, _mm256_set1_epi8(0));
            _mm256_storeu_si256((__m256i*)tdest, sum0);
            sum0 = _mm256_add_epi32(sum, srcv0i);
            
            __m256i srcv1i = _mm256_unpackhi_epi16(src_sh_1_lo, _mm256_set1_epi8(0));
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch),   sum1);
            sum1 = _mm256_add_epi32(sum0, srcv1i);
            
            __m256i srcv2i = _mm256_unpacklo_epi16(src_sh_1_hi, _mm256_set1_epi8(0));
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*2), sum2);
            sum2 = _mm256_add_epi32(sum1, srcv2i);
            
            __m256i srcv3i = _mm256_unpackhi_epi16(src_sh_1_hi, _mm256_set1_epi8(0));
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*3), sum3);
            sum3 = _mm256_add_epi32(sum2, srcv3i);
            
            __m256i srcv4i = _mm256_unpacklo_epi16(src_sh_2_lo, _mm256_set1_epi8(0));
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*4), sum4);
            sum4 = _mm256_add_epi32(sum3, srcv4i);
            
            __m256i srcv5i = _mm256_unpackhi_epi16(src_sh_2_lo, _mm256_set1_epi8(0));
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*5), sum5);
            sum5 = _mm256_add_epi32(sum4, srcv5i);
            
            __m256i srcv6i = _mm256_unpacklo_epi16(src_sh_2_hi, _mm256_set1_epi8(0));
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*6), sum6);
            sum6 = _mm256_add_epi32(sum5, srcv6i);
            
            __m256i srcv7i = _mm256_unpackhi_epi16(src_sh_2_hi, _mm256_set1_epi8(0));
            _mm256_storeu_si256((__m256i*)(tdest+_destpitch*7), sum7);
            sum7 = _mm256_add_epi32(sum6, srcv7i);
            
            sum = sum7;
            
            // TRANSPOSE
			transpose8_ps(sum0,sum1,sum2,sum3,
						  sum4,sum5,sum6,sum7);
            
            tdest += 8;
        }
        
        _mm256_storeu_si256((__m256i*)tdest, sum0);
        _mm256_storeu_si256((__m256i*)(tdest+_destpitch),   sum1);
        _mm256_storeu_si256((__m256i*)(tdest+_destpitch*2), sum2);
        _mm256_storeu_si256((__m256i*)(tdest+_destpitch*3), sum3);
        _mm256_storeu_si256((__m256i*)(tdest+_destpitch*4), sum4);
        _mm256_storeu_si256((__m256i*)(tdest+_destpitch*5), sum5);
        _mm256_storeu_si256((__m256i*)(tdest+_destpitch*6), sum6);
        _mm256_storeu_si256((__m256i*)(tdest+_destpitch*7), sum7);
        
    }
}


void second_step_gather(int* _dest,
                        int _destpitch,
                        int _width,
                        int _height) {
    /*
     SECOND STEP
     
     gather load:
     i0a | i0ab | i0abc | i0abcd
     i1a | i1ab | i1abc | i1abcd
     i2a | i2ab | i2abc | i2abcd
     i3a | i3ab | i3abc | i3abcd
     
     integral sum:
     i0a | i0ab | i0abc | i0abcd
     i01a | i01ab | i01abc | i01abcd
     ...
     i0123a | i0123ab | i0123abc | i0123abcd
     
     store to memory, keep last sum in register, add with next load
     */
    
//    __m256i idxv = _mm256_set_epi32(7*(_height+1), 6*(_height+1), 5*(_height+1), 4*(_height+1), 3*(_height+1), 2*(_height+1), 1*(_height+1), 0);
//    
//    for( int y=0; y<_destpitch; y+=8 ) {
//        int* tsrc = _src + y*(_height+1) + 1;
//        int* tdest = _dest + y + _destpitch;
//        __m256i sum = _mm256_set1_epi32(0);
//        for( int x=0; x<_height; x++ ) {
////            std::cout << "tsrc[0-7]; " << tsrc[0] << " " << tsrc[1*(_height+1)] << " " << tsrc[2*(_height+1)] << " " << tsrc[3*(_height+1)] << " " << tsrc[4*(_height+1)] << " " << tsrc[5*(_height+1)] << " " << tsrc[6*(_height+1)] << " " << tsrc[7*(_height+1)] << " " << std::endl;
//            __m256i srcv = _mm256_i32gather_epi32(tsrc, idxv, 4); tsrc += 1;
//            
//            sum = _mm256_add_epi32(sum, srcv);
//            _mm256_storeu_si256((__m256i*)tdest, sum); tdest += _destpitch;
//            
//        }
//    }
    
    
    for( int x=0; x<_destpitch; x+=8 ) {
        int* tdest = _dest + _destpitch + x;
        __m256i sum = _mm256_set1_epi32(0);
        for( int y=0; y<_height; y++ ) {
            __m256i srcv = _mm256_load_si256((const __m256i*)tdest);

            sum = _mm256_add_epi32(sum, srcv);

            _mm256_store_si256((__m256i*)tdest, sum); tdest += _destpitch;
        }
    }

}
#endif // USE_AVX2


class INTEGRAL_SECOND_STEP_32F_TBB {
    
    float* src;
    float* sum;
    int srcpitch;
    int destpitch;
    int xdim, ydim; // Dimensions of the original image (not source image here)
    
public:
#if SIMPLE
    INTEGRAL_SECOND_STEP_32F_TBB(float* _sum,
                                 unsigned int _destpitch,
                                 unsigned int _xdim,
                                 unsigned int _ydim) : sum(_sum), destpitch(_destpitch), xdim(_xdim), ydim(_ydim)
    {}
    
    inline void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        int yend = std::min((int)r.end(),ydim);
        
        float* csum = sum+(r.begin()+1)*destpitch+1;
        for( size_t y = r.begin(); y < yend; y++, csum += destpitch ) {
            float s = csum[-1] = 0.0f;
            for( int x = 0; x < xdim; x+=1 ) {
                s += csum[x];
                csum[x] = s;
            }
        }
    }
    
#else
    INTEGRAL_SECOND_STEP_32F_TBB(float* _src,
                                 float* _sum,
                                 int _srcpitch,
                                 int _destpitch,
                                 int _xdim,
                                 int _ydim) : src(_src), sum(_sum), srcpitch(_srcpitch), destpitch(_destpitch), xdim(_xdim), ydim(_ydim)
    {}
    
   inline void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        int x,y;
        
        x = (int)r.begin();
        float* csrc;
        float* csum;
        
        for( ; x < r.end()-15; x+=16 ) {
            // Get the correct addresses
            csrc = src+x;
            csum = sum+destpitch+1+x*destpitch/4;
            
            // Loads block of bytes according to:
            // 0 1 2 3 4 5 6 7 8 9 a b c d e f
            
            // Unpacks them in float vectors:
            // [0 1 2 3] [4 5 6 7] etc.
            
            // Adds with previous vector
            
            // Permutes each block of 4 vectors and stores
            
            // Looking blockwise, the before and after matrices will look:
            // Before: [A B C] => [A' B' C']^T
            //         [D E F]    [D' E' F']
            
            // Load the source vectors
            __m128 srcv0 = _mm_load_ps(csrc); csrc+=4;
            __m128 srcv1 = _mm_load_ps(csrc); csrc+=4;
            __m128 srcv2 = _mm_load_ps(csrc); csrc+=4;
            __m128 srcv3 = _mm_load_ps(csrc);
            
            _MM_TRANSPOSE4_PS(srcv0, srcv1, srcv2, srcv3);
            
            srcv1 = _mm_add_ps(srcv0, srcv1);
            srcv2 = _mm_add_ps(srcv1, srcv2);
            srcv3 = _mm_add_ps(srcv2, srcv3);
            
            // Step forward in source vector
            csrc += srcpitch - 12;
            
            __m128 srcv0b;
            __m128 srcv1b;
            __m128 srcv2b;
            __m128 srcv3b;
            
            
            for( y = 1; y < ydim; y+=1, csrc += srcpitch-12 )
            {
                // Load the source vectors
                srcv0b = _mm_load_ps(csrc); csrc+=4;
                srcv1b = _mm_load_ps(csrc); csrc+=4;
                srcv2b = _mm_load_ps(csrc); csrc+=4;
                srcv3b = _mm_load_ps(csrc);
                
                _MM_TRANSPOSE4_PS(srcv0b, srcv1b, srcv2b, srcv3b);
                
                // Adds the previous vectors to vectors b
                srcv0b = _mm_add_ps(srcv3, srcv0b);
                srcv1b = _mm_add_ps(srcv0b, srcv1b);
                srcv2b = _mm_add_ps(srcv1b, srcv2b);
                srcv3b = _mm_add_ps(srcv2b, srcv3b);
                
                
                _MM_TRANSPOSE4_PS(srcv0, srcv1, srcv2, srcv3);
                
                // Store
                _mm_storeu_ps(csum,srcv0); csum+=destpitch;
                srcv0 = srcv0b;
                _mm_storeu_ps(csum,srcv1); csum+=destpitch;
                srcv1 = srcv1b;
                _mm_storeu_ps(csum,srcv2); csum+=destpitch;
                srcv2 = srcv2b;
                _mm_storeu_ps(csum,srcv3); csum+=(-3*destpitch+4);
                srcv3 = srcv3b;
                
            }
            
            _MM_TRANSPOSE4_PS(srcv0b, srcv0b, srcv0b, srcv0b);
            
            _mm_storeu_ps(csum,srcv0b); csum+=destpitch;
            _mm_storeu_ps(csum,srcv1b); csum+=destpitch;
            _mm_storeu_ps(csum,srcv2b); csum+=destpitch;
            _mm_storeu_ps(csum,srcv3b);
        }
   }
#endif
        
};

    
    
    void pvcore::__integralImageCPU(const unsigned char* _src,
                                    float* _dest,
                                    int _srcpitch,
                                    int _destpitch,
                                    int _width,
                                    int _height,
                                    unsigned int _nthreads ) {
        
#if SIMPLE
        memset( _dest, 0, _destpitch*sizeof(float) );
        

        tbb::parallel_for( tbb::blocked_range<size_t>(0,
                                                      _width,
                                                      _width/_nthreads ),
                          INTEGRAL_FIRST_STEP_32F_TBB( _src, _dest, _srcpitch, _destpitch, _width, _height ) );
        
        tbb::parallel_for( tbb::blocked_range<size_t>(0,
                                                      _height,
                                                      _height/_nthreads ),
                          INTEGRAL_SECOND_STEP_32F_TBB( _dest, _destpitch, _width, _height ) );
        
#else
        memset( _dest, 0, _destpitch*sizeof(float) );
        
        static float* tmp = NULL;
        static int tmppitch = -1;
        static int tmpheight = -1;
        
        if( 16*((4*_height+15)/16) != tmppitch ||
           _srcpitch/4 != tmpheight ) {
            tmppitch = 16*((4*_height+15)/16);
            tmpheight = _srcpitch/4;
            if( tmp != NULL ) {
                delete [] tmp;
            }
            tmp = new float[tmppitch*tmpheight];
        }
        
        tbb::parallel_for( tbb::blocked_range<size_t>(0,
                                                      _width,
                                                      _width/_nthreads ),
                          INTEGRAL_FIRST_STEP_32F_TBB( _src, tmp, _srcpitch, tmppitch, _width, _height ) );
        
        
        tbb::parallel_for( tbb::blocked_range<size_t>(0,
                                                      tmppitch,
                                                      tmppitch/_nthreads ),
                          INTEGRAL_SECOND_STEP_32F_TBB( tmp, _dest, tmppitch, _destpitch, tmppitch, tmpheight ) );
        
#endif
    }
    
    
    void pvcore::__integralImageCPU(const unsigned char* _src,
                                    int* _dest,
                                    int _srcpitch,
                                    int _destpitch,
                                    int _width,
                                    int _height,
                                    unsigned int _nthreads ) {
        
        
        memset( _dest, 0, _destpitch*sizeof(int) );
        
        // # work items in total
#ifdef USE_AVX2
        int wi = _srcpitch / 8;
#else
        int wi = _srcpitch / 16;
#endif
        
        // # work items per thread
        float wi_per_thread = wi/(float)_nthreads;
        
//        first_step_gather(_src, _dest, _srcpitch, _destpitch, _width, _height);
//
//        second_step_gather(_dest, _destpitch, _width, _height);
        
        
        
#ifdef USE_AVX2
        // Parallelize over rows
        tbb::parallel_for( tbb::blocked_range<size_t>(0,
                                                      _nthreads,
                                                      1 ),
                          INTEGRAL_FIRST_STEP_32S_TBB( _src, _dest, _srcpitch, _destpitch, _width, _height, (float)(_height/8.f)/(float)_nthreads ) );
        // Parallelize over columns
        tbb::parallel_for( tbb::blocked_range<size_t>(0,
                                                      _nthreads,
                                                      1 ),
                          INTEGRAL_SECOND_STEP_32S_TBB( _dest, _destpitch, _width, _height, (float)(_destpitch/8.f)/(float)_nthreads ) );
//        second_step_gather(_dest, _destpitch, _width, _height);
#else
        tbb::parallel_for( tbb::blocked_range<size_t>(0,
                                                      _nthreads,
                                                      1 ),
                          INTEGRAL_FIRST_STEP_32S_TBB( _src, _dest, _srcpitch, _destpitch, _width, _height, wi_per_thread) );
        tbb::parallel_for( tbb::blocked_range<size_t>(0,
                                                      _height,
                                                      _height/_nthreads ),
                          INTEGRAL_SECOND_STEP_32S_TBB( _dest, _destpitch, _width, _height, wi_per_thread ) );
#endif
    }
    
