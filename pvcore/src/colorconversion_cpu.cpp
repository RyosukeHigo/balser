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

#include <algorithm>
#include <cmath>

// SSE
#include <immintrin.h>

// TBB
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>


#include "colorconversion/common.cpp"

#include "colorconversion/rgb2bgr_8u.cpp"
#include "colorconversion/rgb2hsv_8u.cpp"
#include "colorconversion/rgb2gray_8u.cpp"
#include "colorconversion/rgb2luv.cpp"

#include "colorconversion/bayer2rgb.cpp"
#include "colorconversion/bayer2rgb_wb.cpp"

#define USE_TBB (1)


template <class T1, class T2>
void gray2jet(const T1* _src,
              T2* _dest,
              unsigned int _size) {
    
    for( unsigned int i=0; i<_size; ++i ) {
        int val = 3*(int)(255.0f*_src[i]);
        _dest[3*i+0] = (T2)k_jet[val+0];
        _dest[3*i+1] = (T2)k_jet[val+1];
        _dest[3*i+2] = (T2)k_jet[val+2];
    }
    
}

#ifdef USE_TBB
#	define optimizedCall(T,args) \
		tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1), \
		CVTCOLOR_TBB<T> args)
#	define optimizedCallBayer(LAYOUT,HAS_ALPHA,args) \
		tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1), \
		BAYER2RGB_8U_TBB<LAYOUT,HAS_ALPHA> args)
#	define optimizedCallBayerWB(LAYOUT,HAS_ALPHA,args) \
		tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1), \
		BAYER2RGB_WB_8U_TBB<LAYOUT,HAS_ALPHA> args)
#else
#	define optimizedCall(T,args) \
		sequentialCall args
#	define optimizedCallBayer(LAYOUT,HAS_ALPHA,args) \
		_bayer2rgb_8u<LAYOUT,HAS_ALPHA> args
#	define optimizedCallBayerWB(LAYOUT,HAS_ALPHA,args) \
		tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1), \
		BAYER2RGB_WB_8U_TBB<LAYOUT,HAS_ALPHA> args)
#endif


void pvcore::__convertColorWBCPU(const unsigned char* _src,
								 unsigned char* _dest,
								 unsigned int _width,
								 unsigned int _height,
								 unsigned int _pitchs,
								 unsigned int _pitchd,
								 unsigned int _kernel,
								 short _rcoeff, short _gcoeff, short _bcoeff,
								 unsigned int _threads ) {

    switch( _kernel ) {
		      case bayergr2rgb_8u:
			optimizedCallBayerWB(GRBG,false, ( _src, _dest,
											_width, _height,
											_pitchs, _pitchd,
											_rcoeff, _gcoeff, _bcoeff,
											_threads ));
            break;
            
        case bayergr2rgbx_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<GRBG,true>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd,
														   _threads ) );
            break;
            
        case bayerrg2rgb_8u:
			optimizedCallBayerWB(RGGB,false, ( _src, _dest,
											_width, _height,
											_pitchs, _pitchd,
											_rcoeff, _gcoeff, _bcoeff,
											_threads ));
/*            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<RGGB,false>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );*/
            break;

        case bayerrg2rgbx_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<RGGB,true>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );
            break;

        case bayergb2rgb_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<GBRG,false>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );
            break;
            
        case bayergb2rgbx_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<GBRG,true>( _src, _dest,
                                                          _width, _height,
                                                          _pitchs, _pitchd, _threads ) );
            break;
            
        case bayerbg2rgb_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<BGGR,false>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );
            break;
            
        case bayerbg2rgbx_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<BGGR,true>( _src, _dest,
                                                          _width, _height,
                                                          _pitchs, _pitchd, _threads ) );
            break;
            
        default:
            break;
	}
}



void pvcore::__convertColorCPU(const unsigned char* _src,
                               unsigned char* _dest,
                               unsigned int _width,
                               unsigned int _height,
                               unsigned int _pitchs,
                               unsigned int _pitchd,
                               unsigned int _kernel,
                               unsigned int _threads ) {
    
    switch( _kernel ) {
            
            // =============================================================
            // ===================== GRAY2COLORMAP =========================
            // =============================================================
        case gray2jet_8u:
            // Not Working
            gray2jet<unsigned char, unsigned char>( _src, _dest, _width*_height );
            break;
			
			
            // =============================================================
            // ======================== RGB2GRAY ===========================
            // =============================================================
        case rgb2gray_8u:
			if( _pitchd / _width == 3) {
				optimizedCall(unsigned char, ( _rgb2grayc_8u, _src, _dest,
														_width, _height,
														_pitchs, _pitchd, _threads ) );
			} else {
				optimizedCall(unsigned char, ( _rgb2gray_8u, _src, _dest,
														_width, _height,
														_pitchs, _pitchd, _threads ) );
			}

			
            // =============================================================
            // ======================== RGB2HSV ============================
            // =============================================================
		case rgb2hsv_8u:
			optimizedCall(unsigned char, (_rgb2hsv_8u, _src, _dest,
														  _width, _height,
														  _pitchs, _pitchd, _threads) );
            break;
        case hsv2rgb_8u:
			optimizedCall(unsigned char, (_hsv2rgb_8u, _src, _dest,
												   _width, _height,
												   _pitchs, _pitchd, _threads) );
            break;
			
			
            // =============================================================
            // ======================== RGB2LUV ============================
            // =============================================================
        case rgb2luv_8u:
			optimizedCall(unsigned char, (_rgb2luv_8u, _src, _dest,
										  _width, _height,
										  _pitchs, _pitchd, _threads ) );
            break;

        case luv2rgb_8u:
			optimizedCall(unsigned char, (_luv2rgb_8u, _src, _dest,
										  _width, _height,
										  _pitchs, _pitchd, _threads ) );
            break;
						  
        case rgb2luv_32f:
			optimizedCall(float, (_rgb2luv_32f, _src, _dest,
								  _width, _height,
								  _pitchs, _pitchd, _threads ) );
            break;
            
            // =============================================================
            // ======================== RGB2BGR ============================
            // =============================================================
        case rgb2bgr_8u:
			optimizedCall(unsigned char, (_rgb2bgr_8u, _src, _dest,
										   _width, _height,
										   _pitchs, _pitchd, _threads ) );
            break;
            
        case rgbx2bgrx_8u:
			optimizedCall(unsigned char, (_rgbx2bgrx_8u, _src, _dest,
										   _width, _height,
										   _pitchs, _pitchd, _threads ) );
            break;
            
		case rgb2bgr_32f:
			optimizedCall(float, (_rgb2bgr_32f, _src, _dest,
												  _width, _height,
												  _pitchs, _pitchd, _threads ) );
			break;
			
		case rgbx2bgrx_32f:
			optimizedCall(float, (_rgbx2bgrx_32f, _src, _dest,
												  _width, _height,
												  _pitchs, _pitchd, _threads ) );
			break;
			
            // =============================================================
            // ======================= BAYER2RGB ===========================
            // =============================================================
        case bayergr2rgb_8u:
			optimizedCallBayer(GRBG,false, ( _src, _dest,
											_width, _height,
											_pitchs, _pitchd, _threads ));
            break;
            
        case bayergr2rgbx_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<GRBG,true>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );
            break;
            
        case bayerrg2rgb_8u:
			optimizedCallBayer(RGGB,false, ( _src, _dest,
											_width, _height,
											_pitchs, _pitchd, _threads ));
/*            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<RGGB,false>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );*/
            break;

        case bayerrg2rgbx_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<RGGB,true>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );
            break;

        case bayergb2rgb_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<GBRG,false>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );
            break;
            
        case bayergb2rgbx_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<GBRG,true>( _src, _dest,
                                                          _width, _height,
                                                          _pitchs, _pitchd, _threads ) );
            break;
            
        case bayerbg2rgb_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<BGGR,false>( _src, _dest,
                                                           _width, _height,
                                                           _pitchs, _pitchd, _threads ) );
            break;
            
        case bayerbg2rgbx_8u:
            tbb::parallel_for( tbb::blocked_range<size_t>(0,_threads,1 ),
                              BAYER2RGB_8U_TBB<BGGR,true>( _src, _dest,
                                                          _width, _height,
                                                          _pitchs, _pitchd, _threads ) );
            break;
            
        default:
            break;
    }
    
    
    
    
}

/*void gray2jet_8u(const  uchar* _src,
 uchar* _dest,
 unsigned int _size) {
 
 size_t idx4 = 4*get_global_id(0);
 size_t idx12 = 3*idx4;
 if( idx4 < _size ) {
 
 int val = 3*(int)(_src[idx4]);
 _dest[idx12]   = CAST_8U(255.0f*k_jet[val]);
 _dest[idx12+1] = CAST_8U(255.0f*k_jet[val+1]);
 _dest[idx12+2] = CAST_8U(255.0f*k_jet[val+2]);
 val = 3*(int)(_src[idx4+1]);
 _dest[idx12+3]   = CAST_8U(255.0f*k_jet[val]);
 _dest[idx12+4] = CAST_8U(255.0f*k_jet[val+1]);
 _dest[idx12+5] = CAST_8U(255.0f*k_jet[val+2]);
 val = 3*(int)(_src[idx4+2]);
 _dest[idx12+6]   = CAST_8U(255.0f*k_jet[val]);
 _dest[idx12+7] = CAST_8U(255.0f*k_jet[val+1]);
 _dest[idx12+8] = CAST_8U(255.0f*k_jet[val+2]);
 val = 3*(int)(_src[idx4+3]);
 _dest[idx12+9]   = CAST_8U(255.0f*k_jet[val]);
 _dest[idx12+10] = CAST_8U(255.0f*k_jet[val+1]);
 _dest[idx12+11] = CAST_8U(255.0f*k_jet[val+2]);
 }
 
 }
 
 
 ///////////////////////////////////////////////
 ///////////////// RGB --> GRAY ////////////////
 ///////////////////////////////////////////////
 void rgba2gray_8u(const  uchar* _src,
 uchar* _dest,
 unsigned int _size) {
 
 const unsigned int idx4 = 4*get_global_id(0);
 const unsigned int idx16 = idx4*4;
 
 if( idx4 < _size ) {
 
 _dest[idx4]   = CAST_8U(0.30f*_src[idx16]    + 0.59f*_src[idx16+1]  + 0.11f*_src[idx16+2]);
 _dest[idx4+1] = CAST_8U(0.30f*_src[idx16+4]  + 0.59f*_src[idx16+5]  + 0.11f*_src[idx16+6]);
 _dest[idx4+2] = CAST_8U(0.30f*_src[idx16+8]  + 0.59f*_src[idx16+9]  + 0.11f*_src[idx16+10]);
 _dest[idx4+3] = CAST_8U(0.30f*_src[idx16+13] + 0.59f*_src[idx16+14] + 0.11f*_src[idx16+15]);
 
 }
 }
 
 void rgb2gray_8u(const  uchar* _src,
 uchar* _dest,
 unsigned int _size) {
 
 const unsigned int idx4 = 4*get_global_id(0);
 const unsigned int idx12 = idx4*3;
 
 if( idx4 < _size ) {
 
 _dest[idx4] = CAST_8U(0.30f*_src[idx12] + 0.59f*_src[idx12+1] + 0.11f*_src[idx12+2]);
 _dest[idx4+1] = CAST_8U(0.30f*_src[idx12+3] + 0.59f*_src[idx12+4] + 0.11f*_src[idx12+5]);
 _dest[idx4+2] = CAST_8U(0.30f*_src[idx12+6] + 0.59f*_src[idx12+7] + 0.11f*_src[idx12+8]);
 _dest[idx4+3] = CAST_8U(0.30f*_src[idx12+9] + 0.59f*_src[idx12+10] + 0.11f*_src[idx12+11]);
 
 }
 }
 
 void rgb2gray_32fto8u(const  float *_src,
 uchar *_dest,
 unsigned int _size) {
 
 const unsigned int idx4 = 4*get_global_id(0);
 const unsigned int idx12 = idx4*3;
 
 if( idx4 < _size ) {
 
 _dest[idx4] = CAST_8U(255.0f*(0.30f*_src[idx12] + 0.59f*_src[idx12+1] + 0.11f*_src[idx12+2]) );
 _dest[idx4+1] = CAST_8U(255.0f*(0.30f*_src[idx12+3] + 0.59f*_src[idx12+4] + 0.11f*_src[idx12+5]) );
 _dest[idx4+2] = CAST_8U(255.0f*(0.30f*_src[idx12+6] + 0.59f*_src[idx12+7] + 0.11f*_src[idx12+8]) );
 _dest[idx4+3] = CAST_8U(255.0f*(0.30f*_src[idx12+9] + 0.59f*_src[idx12+10] + 0.11f*_src[idx12+11]) );
 
 }
 
 }
 
 void rgb2gray_32f(const  float *_src,
 float *_dest,
 unsigned int _size) {
 
 const size_t idx4 = 4*get_global_id(0);
 const size_t idx12 = idx4*3;
 
 if( idx4 < _size ) {
 
 _dest[idx4] = 0.30f*_src[idx12] + 0.59f*_src[idx12+1] + 0.11f*_src[idx12+2];
 _dest[idx4+1] = 0.30f*_src[idx12+3] + 0.59f*_src[idx12+4] + 0.11f*_src[idx12+5];
 _dest[idx4+2] = 0.30f*_src[idx12+6] + 0.59f*_src[idx12+7] + 0.11f*_src[idx12+8];
 _dest[idx4+3] = 0.30f*_src[idx12+9] + 0.59f*_src[idx12+10] + 0.11f*_src[idx12+11];
 
 }
 
 }
 
 void rgb2gray_8uto32f(const  uchar* _src,
 float* _dest,
 unsigned int _size) {
 
 const size_t idx4 = 4*get_global_id(0);
 const size_t idx12 = idx4*3;
 
 if( idx4 < (unsigned int)_size ) {
 
 _dest[idx4]   = ( 0.30f*_src[idx12]   + 0.59f*_src[idx12+1]  + 0.11f*_src[idx12+2]  ) *inv255;
 _dest[idx4+1] = ( 0.30f*_src[idx12+3] + 0.59f*_src[idx12+4]  + 0.11f*_src[idx12+5]  ) *inv255;
 _dest[idx4+2] = ( 0.30f*_src[idx12+6] + 0.59f*_src[idx12+7]  + 0.11f*_src[idx12+8]  ) *inv255;
 _dest[idx4+3] = ( 0.30f*_src[idx12+9] + 0.59f*_src[idx12+10] + 0.11f*_src[idx12+11] ) *inv255;
 
 }
 
 }
 
 
 ///////////////////////////////////////////////
 /////////////// KERNEL FUNCTIONS //////////////
 ///////////////////////////////////////////////
 void rgb2gray_8u(const  uchar3* _src,
 uchar4* _dest,
 unsigned int _size) {
 
 const unsigned int idx = get_global_id(0);
 const unsigned int idx4 = idx*4;
 
 if( idx < _size ) {
 
 uchar3 p1 = vload3( idx4, ( const uchar*)_src);
 uchar3 p2 = vload3( idx4+1, ( const uchar*)_src);
 uchar3 p3 = vload3( idx4+2, ( const uchar*)_src);
 uchar3 p4 = vload3( idx4+3, ( const uchar*)_src);
 
 float4 pf1 = (float4)((float)p1.x,(float)p2.x,(float)p3.x,(float)p4.x);
 float4 pf2 = (float4)((float)p1.y,(float)p2.y,(float)p3.y,(float)p4.y);
 float4 pf3 = (float4)((float)p1.z,(float)p2.z,(float)p3.z,(float)p4.z);
 
 pf1 *= (float4)0.30f;
 pf2 *= (float4)0.59f;
 pf3 *= (float4)0.11f;
 
 float4 res = pf1+pf2+pf3;
 
 uchar r1 = CAST_8U(res.x);
 uchar r2 = CAST_8U(res.y);
 uchar r3 = CAST_8U(res.z);
 uchar r4 = CAST_8U(res.w);
 
 uchar4 o = (uchar4)( r1,r2,r3,r4 );
 vstore4(o, (size_t)idx, ( uchar*)_dest);
 }
 }
 
 
 
 ///////////////////////////////////////////////
 ///////////////// BGR <--> Lab ////////////////
 ///////////////////////////////////////////////
 
 // BGR -> Lab
 void bgr2lab( float* src,
 float* dest,
 unsigned int width,
 unsigned int height) {
 
 size_t globalX = get_global_id(0);
 size_t globalY = get_global_id(1);
 
 size_t global_memid = 3*(globalY*_width + globalX);
 
 float r = (float)src[global_memid+2];
 float g = (float)src[global_memid+1];
 float b = (float)src[global_memid];
 float x,y,z;
 
 float t = 0.008856;
 
 x = 0.433953f*r + 0.376219f*g + 0.189828f*b;
 y = 0.212671f*r + 0.715160f*g + 0.072169f*b;
 z = 0.017758f*r + 0.109477f*g + 0.872766f*b;
 
 x /= 0.950456f;
 y /= 1.0f;
 z /= 1.088754f;
 
 float y3 = rootn(y,3);
 
 bool xt = x>t;
 bool yt = y>t;
 bool zt = z>t;
 
 
 if(xt) {
 x = rootn(x,3);
 } else {
 x = 7.787f*x + 0.13793103448275862f;
 }
 
 if(yt) {
 y = y3;
 } else {
 y = 7.787f*y + 0.13793103448275862f;
 }
 
 if(zt) {
 z = rootn(z,3);
 } else {
 z = 7.787f*z + 0.13793103448275862f;
 }
 
 if(yt) {
 dest[global_memid] = (116.0f*y3 - 16.0f) / 100.0f;
 } else {
 dest[global_memid] = (903.3*y) / 100.0f;
 }
 dest[global_memid+1] = 500.0f * (x-y) / 256.0f + 0.5f;
 dest[global_memid+2] = 200.0f * (y-z) / 256.0f + 0.5f;
 
 }
 
 ///////////////////////////////////////////////
 //////////////// BGR <--> YCrCb////////////////
 ///////////////////////////////////////////////
 
 void bgr2ycrcb( float* src,
 float* dest,
 unsigned int width,
 unsigned int height) {
 
 size_t globalX = get_global_id(0);
 size_t globalY = get_global_id(1);
 
 size_t global_memid = 3*(globalY*_width + globalX);
 
 const float yuvYr_32f = 0.299f;//cscGr_32f;
 const float yuvYg_32f = 0.587f;//cscGg_32f;
 const float yuvYb_32f = 0.114f;//cscGb_32f;
 
 const float yuvCr_32f = 0.713f;
 const float yuvCb_32f = 0.564f;
 
 float r = (float)src[global_memid+2];///255.0f;
 float g = (float)src[global_memid+1];///255.0f;
 float b = (float)src[global_memid];///255.0f;
 float y;
 
 y = b*yuvYb_32f + g*yuvYg_32f + r*yuvYr_32f;
 r = (r - y)*yuvCr_32f + 0.5f;
 b = (b - y)*yuvCb_32f + 0.5f;
 
 dest[global_memid] = y;//0.299*r + 0.587f*g + 0.114f*b;
 dest[global_memid+1] = r;//r - 0.331264f*g + 0.5f*b;
 dest[global_memid+2] = b;//0.5f*r - 0.418688f*g - 0.81312f*b;
 
 }
 
 ///////////////////////////////////////////////
 //////////////// RGB <--> BGR /////////////////
 ///////////////////////////////////////////////
 void rgb2bgr_8u(const  uchar* _src,
 uchar* _dest,
 unsigned int _size) {
 
 if( get_global_id(0) < _size ) {
 size_t idx = get_global_id(0)*3;
 uchar t = _src[idx];
 _dest[idx] = _src[idx+2];
 _dest[idx+2] = t;
 }
 
 }
 
 void rgb2bgr_32f(const  float* _src,
 float* _dest,
 unsigned int _size) {
 
 if( get_global_id(0) < _size ) {
 size_t idx = get_global_id(0)*3;
 uchar t = _src[idx];
 _dest[idx] = _src[idx+2];
 _dest[idx+2] = t;
 }
 
 }*/