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

///////////////////////////////////////////////
/////////////////// CONSTANTS /////////////////
///////////////////////////////////////////////

#include "pvcore/colorconversion.h"

#include "pvcore/common.h"

texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> tex_u8;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> tex_u8_4;

#define inv8255 (1.f/(256.f*255.f));


///////////////////////////////////////////////
/////////////// Bayer <--> RGBX ///////////////
///////////////////////////////////////////////
// Working
__global__ void _bayer2rgbx_8u_tex(uchar4* _dest,
                                   unsigned int _width,
                                   unsigned int _height,
                                   unsigned int _pitch,
                                   unsigned int _shmpitch) {
    
    // Coordinates for bayer image (texture)
    unsigned int x = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    unsigned int y = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    
    extern __shared__ uchar4 shm_u8_4[];
    
    // LAYOUT:
    // G R G R
    // B G B G
    // G R G R
    // B G B G
    
    unsigned int xm1 = x-1;
    unsigned int xp1 = x+1;
    unsigned int xp2 = x+2;
    unsigned int ym1 = y-1;
    unsigned int yp1 = y+1;
    unsigned int yp2 = y+2;
    
    
    // Writing phase easy to speed up through shared memory
    if( x < _width && y < _height ) {
        // rgb pixel x,y
        int shmidx = 2*threadIdx.y*_shmpitch+2*threadIdx.x;
        shm_u8_4[shmidx].x = (tex2D(tex_u8,xm1,y) + tex2D(tex_u8,xp1,y) + 1) >> 1;
        shm_u8_4[shmidx].y = tex2D(tex_u8,x,y);
        shm_u8_4[shmidx].z = (tex2D(tex_u8,x,ym1) + tex2D(tex_u8,x,yp1) + 1) >> 1;
        shm_u8_4[shmidx].w = 255;
        
        // rgb pixel x+1,y
        shmidx++;
        shm_u8_4[shmidx].x = tex2D(tex_u8,xp1,y);
        shm_u8_4[shmidx].y = (tex2D(tex_u8,x,y) + tex2D(tex_u8,xp1,ym1) + tex2D(tex_u8,xp2,y) + tex2D(tex_u8,xp1,yp1) + 2) >> 2;
        shm_u8_4[shmidx].z = (tex2D(tex_u8,x,ym1) + tex2D(tex_u8,xp2,ym1) + tex2D(tex_u8,x,yp1) + tex2D(tex_u8,xp2,yp1) + 2) >> 2;
        shm_u8_4[shmidx].w = 255;
        
        // rgb pixel x,y+1
        shmidx += _shmpitch-1;
        shm_u8_4[shmidx].x = (tex2D(tex_u8,xm1,y) + tex2D(tex_u8,xp1,y) + tex2D(tex_u8,xm1,yp2) + tex2D(tex_u8,xp1,yp2) + 2) >> 2;
        shm_u8_4[shmidx].y = (tex2D(tex_u8,xm1,yp1) + tex2D(tex_u8,x,y) + tex2D(tex_u8,xp1,yp1) + tex2D(tex_u8,x,yp2) + 2) >> 2;
        shm_u8_4[shmidx].z = tex2D(tex_u8,x,yp1);
        shm_u8_4[shmidx].w = 255;
        
        shmidx++;
        // rgb pixel x+1,y+1
        shm_u8_4[shmidx].x = (tex2D(tex_u8,xp1,y) + tex2D(tex_u8,xp1,yp2) + 1) >> 1;
        shm_u8_4[shmidx].y = tex2D(tex_u8,xp1,yp1);
        shm_u8_4[shmidx].z = (tex2D(tex_u8,x,yp1) + tex2D(tex_u8,xp2,yp1) + 1) >> 1;
        shm_u8_4[shmidx].w = 255;
        
        __syncthreads();
        
        _dest[y*_pitch+2*blockIdx.x*blockDim.x + threadIdx.x] = shm_u8_4[2*threadIdx.y*_shmpitch + threadIdx.x];
        _dest[y*_pitch+2*blockIdx.x*blockDim.x + blockDim.x + threadIdx.x] = shm_u8_4[2*threadIdx.y*_shmpitch + blockDim.x + threadIdx.x];
        _dest[(y+1)*_pitch+2*blockIdx.x*blockDim.x + threadIdx.x] = shm_u8_4[(2*threadIdx.y+1)*_shmpitch+threadIdx.x];
        _dest[(y+1)*_pitch+2*blockIdx.x*blockDim.x + blockDim.x + threadIdx.x] = shm_u8_4[(2*threadIdx.y+1)*_shmpitch + blockDim.x + threadIdx.x];
        
    }
    
    // WORKING
    /*if( x < _width && y < _height ) {
        // rgb pixel x,y
        uchar4 rgbpix[4];
        rgbpix[0].x = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x+1,y) + 1) >> 1;
        rgbpix[0].y = tex2D(tex_u8,x,y);
        rgbpix[0].z = (tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x,y+1) + 1) >> 1;
        rgbpix[0].w = 255;
        
        // rgb pixel x+1,y
        rgbpix[1].x = tex2D(tex_u8,x+1,y);
        rgbpix[1].y = (tex2D(tex_u8,x,y) + tex2D(tex_u8,x+1,y-1) + tex2D(tex_u8,x+2,y) + tex2D(tex_u8,x+1,y+1) + 2) >> 2;
        rgbpix[1].z = (tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x+2,y-1) + tex2D(tex_u8,x,y+1) + tex2D(tex_u8,x+2,y+1) + 2) >> 2;
        rgbpix[1].w = 255;
        
        // rgb pixel x,y+1
        rgbpix[2].x =(tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x+1,y) + tex2D(tex_u8,x-1,y+2) + tex2D(tex_u8,x+1,y+2) + 2) >> 2;
        rgbpix[2].y = (tex2D(tex_u8,x-1,y+1) + tex2D(tex_u8,x,y) + tex2D(tex_u8,x+1,y+1) + tex2D(tex_u8,x,y+2) + 2) >> 2;
        rgbpix[2].z = tex2D(tex_u8,x,y+1);
        rgbpix[2].w = 255;
        
        // rgb pixel x+1,y+1
        rgbpix[3].x = (tex2D(tex_u8,x+1,y) + tex2D(tex_u8,x+1,y+2) + 1) >> 1;
        rgbpix[3].y = tex2D(tex_u8,x+1,y+1);
        rgbpix[3].z = (tex2D(tex_u8,x,y+1) + tex2D(tex_u8,x+2,y+1) + 1) >> 1;
        rgbpix[3].w = 255;
        
        
        _dest[y*_pitch+x] = rgbpix[0];
        _dest[y*_pitch+x+1] = rgbpix[1];
        _dest[(y+1)*_pitch+x] = rgbpix[2];
        _dest[(y+1)*_pitch+x+1] = rgbpix[3];
        
    }*/
    
    
}


// Not working
__global__ void _bayer2rgb_8u_tex(uchar4* _dest,
                                  unsigned int _width,
                                  unsigned int _height,
                                  unsigned int _pitch) {
    
    // Coordinates for bayer image (texture)
    unsigned int x = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    unsigned int y = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    
    
    
    // LAYOUT:
    // G R G R G R G R G R G R
    // B G B G B G B G B G B G
    // G R G R G R G R G R G R
    // B G B G B G B G B G B G
    
    // Each thread processes four pixels in two rows
    if( x < _width && y < _height ) {
        uchar4 rgbpix[3];
        // Row 1, pixel 1
        rgbpix[0].x = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x+1,y) + 1) >> 1;
        rgbpix[0].y = tex2D(tex_u8,x,y);
        rgbpix[0].z = (tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x,y+1) + 1) >> 1;
        
        x++;
        // Row 1, pixel 2
        rgbpix[0].w = tex2D(tex_u8,x,y);
        rgbpix[1].x = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x+1,y) + tex2D(tex_u8,x,y+1) + 2) >> 2;
        rgbpix[1].y = (tex2D(tex_u8,x-1,y-1) + tex2D(tex_u8,x+1,y-1) + tex2D(tex_u8,x-1,y+1) + tex2D(tex_u8,x+1,y+1) + 2) >> 2;
        
        x++;
        // Row 1, pixel 3
        rgbpix[1].z = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x+1,y) + 1) >> 1;
        rgbpix[1].w = tex2D(tex_u8,x,y);
        rgbpix[2].x = (tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x,y+1) + 1) >> 1;
        
        x++;
        // Row 1, pixel 4
        rgbpix[2].y = tex2D(tex_u8,x,y);
        rgbpix[2].z = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x+1,y) + tex2D(tex_u8,x,y+1) + 2) >> 2;
        rgbpix[2].w = (tex2D(tex_u8,x-1,y-1) + tex2D(tex_u8,x+1,y-1) + tex2D(tex_u8,x-1,y+1) + tex2D(tex_u8,x+1,y+1) + 2) >> 2;
    
        x -= 3;
        
        _dest[y*_pitch+3*x/4]   = rgbpix[0];
        _dest[y*_pitch+3*x/4+1] = rgbpix[1];
        _dest[y*_pitch+3*x/4+2] = rgbpix[2];
        

        y++;
        // rgb pixel x,y+1
        rgbpix[0].x = (tex2D(tex_u8,x-1,y-1) + tex2D(tex_u8,x+1,y-1) + tex2D(tex_u8,x-1,y+1) + tex2D(tex_u8,x+1,y+1) + 2) >> 2;
        rgbpix[0].y = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x+1,y) + tex2D(tex_u8,x,y+1) + 2) >> 2;
        rgbpix[0].z = tex2D(tex_u8,x,y);
        
        // rgb pixel x+1,y+1
        x++;
        rgbpix[0].w = (tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x,y+1) + 1) >> 1;
        rgbpix[1].x = tex2D(tex_u8,x,y);
        rgbpix[1].y = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x+1,y) + 1) >> 1;
        
        // rgb pixel x,y+1
        x++;
        rgbpix[1].z = (tex2D(tex_u8,x-1,y-1) + tex2D(tex_u8,x+1,y-1) + tex2D(tex_u8,x-1,y+1) + tex2D(tex_u8,x+1,y+1) + 2) >> 2;
        rgbpix[1].w = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x+1,y) + tex2D(tex_u8,x,y+1) + 2) >> 2;
        rgbpix[2].x = tex2D(tex_u8,x,y);
        
        // rgb pixel x+1,y+1
        x++;
        rgbpix[2].y = (tex2D(tex_u8,x,y-1) + tex2D(tex_u8,x,y+1) + 1) >> 1;
        rgbpix[2].z = tex2D(tex_u8,x,y);
        rgbpix[2].w = (tex2D(tex_u8,x-1,y) + tex2D(tex_u8,x+1,y) + 1) >> 1;
        
        x -= 3;
        
        _dest[y*_pitch+3*x/4] = rgbpix[0];
        _dest[y*_pitch+3*x/4+1] = rgbpix[1];
        _dest[y*_pitch+3*x/4+2] = rgbpix[2];
        
        
        
    } else {
        
    }
    
    
    
    
    
}

///////////////////////////////////////////////
//////////////// RGB <--> BGR /////////////////
///////////////////////////////////////////////
// Working
__global__ void _rgb2bgr_8u(const uchar4* _src,
                            uchar4* _dest,
                            unsigned int _width,
                            unsigned int _height,
                            unsigned int _pitchs,
                            unsigned int _pitchd,
                            unsigned int _shmwidth) {
    
    unsigned int x =  blockIdx.x*blockDim.x; // address to block x on row 2*y
    unsigned int y =  (blockIdx.y*blockDim.y+threadIdx.y); // address to row 2*y
    unsigned int x3 = 3*x;
    unsigned int t3 = threadIdx.y*_shmwidth+3*threadIdx.x;
    
    extern __shared__ uchar4 shm[];
    
    if( blockIdx.y*blockDim.y+threadIdx.y < _height ) {
        // Read three uchar 4, equal to 4 pixels
        shm[threadIdx.y*_shmwidth + threadIdx.x] = _src[y*_pitchs+x3+threadIdx.x];
        shm[threadIdx.y*_shmwidth + threadIdx.x+blockDim.x] = _src[y*_pitchs+x3+blockDim.x+threadIdx.x];
        shm[threadIdx.y*_shmwidth + threadIdx.x+2*blockDim.x] = _src[y*_pitchs+x3+2*blockDim.x+threadIdx.x];
    
        __syncthreads();
        
        // Pixel 1
        unsigned char t = shm[t3].x;
        shm[t3].x = shm[t3].z;
        shm[t3].y = shm[t3].y;
        shm[t3].z = t;

        // Pixel 2
        t = shm[t3].w;
        shm[t3].w = shm[t3+1].y;
        shm[t3+1].x = shm[t3+1].x;
        shm[t3+1].y = t;
        
        // Pixel 3
        t = shm[t3+1].z;
        shm[t3+1].z = shm[t3+2].x;
        shm[t3+1].w = shm[t3+1].w;
        shm[t3+2].x = t;
        
        // Pixel 4
        t = shm[t3+2].y;
        shm[t3+2].y = shm[t3+2].w;
        shm[t3+2].z = shm[t3+2].z;
        shm[t3+2].w = t;
        
        __syncthreads();
        
        _dest[y*_pitchd+x3+threadIdx.x] = shm[threadIdx.y*_shmwidth + threadIdx.x];
        _dest[y*_pitchd+x3+threadIdx.x+blockDim.x] = shm[threadIdx.y*_shmwidth + threadIdx.x+blockDim.x];
        _dest[y*_pitchd+x3+threadIdx.x+2*blockDim.x] = shm[threadIdx.y*_shmwidth + threadIdx.x+2*blockDim.x];
        
    }
}

// Working
__global__ void _rgbx2bgrx_8u(const uchar4* _src,
                              uchar4* _dest,
                              unsigned int _width,
                              unsigned int _height,
                              unsigned int _pitchs,
                              unsigned int _pitchd) {
    
    unsigned int x =  blockIdx.x*blockDim.x+threadIdx.x; // address to block x on row 2*y
    unsigned int y =  (blockIdx.y*blockDim.y+threadIdx.y); // address to row 2*y
    
    if( blockIdx.y*blockDim.y+threadIdx.y < _height ) {
        // Read three uchar 4, equal to 4 pixels
        
        uchar4 pix = _src[y*_pitchs+x];
        unsigned char t = pix.x;
        pix.x = pix.z;
        pix.z = t;
        _dest[y*_pitchd+x] = pix;
        
    }
}

// Working?
__global__ void _rgb2bgr_32f(const float* _src,
                             float* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _pitchs,
                             unsigned int _pitchd) {
    
    int x =  3*(blockIdx.x*blockDim.x+threadIdx.x); // address to block x on row 2*y
    int y =  (blockIdx.y*blockDim.y+threadIdx.y); // address to row 2*y
    
    if( blockIdx.y*blockDim.y+threadIdx.y < _height ) {
        _dest[y*_pitchd+x+2] = _src[y*_pitchs+x];
        _dest[y*_pitchd+x] = _src[y*_pitchs+x+2];
    }
    
}



__global__ void _rgbx2bgrx_32f(const float* _src,
                               float* _dest,
                               unsigned int _width,
                               unsigned int _height,
                               unsigned int _pitchs,
                               unsigned int _pitchd) {
    
    int x =  4*(blockIdx.x*blockDim.x+threadIdx.x); // address to block x on row 2*y
    int y =  (blockIdx.y*blockDim.y+threadIdx.y); // address to row 2*y
    
    if( blockIdx.y*blockDim.y+threadIdx.y < _height ) {
        _dest[y*_pitchd+x+2] = _src[y*_pitchs+x];
        _dest[y*_pitchd+x] = _src[y*_pitchs+x+2];
    }
    
}



// Currently only working for exact block and grid dimensions
__global__ void _rgb2gray_8u( const uchar4* _src, uchar4* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _pitchs,
                             unsigned int _pitchd,
                             unsigned int _shmwidth) {
    
    unsigned int x =  blockIdx.x*blockDim.x; // address to block x on row 2*y
    unsigned int y =  (blockIdx.y*blockDim.y+threadIdx.y)*_pitchs; // address to row 2*y
    unsigned int x3 = 3*x;
    unsigned int t3 = threadIdx.y*_shmwidth+3*threadIdx.x;
    
    extern __shared__ uchar4 shm[];
    
    unsigned int r = 0x0000004D;
    unsigned int g = 0x00000096;
    unsigned int b = 0x0000001D;
    
    if( blockIdx.y*blockDim.y+threadIdx.y < _height ) {
        
        shm[threadIdx.y*_shmwidth + threadIdx.x] = _src[y+x3+threadIdx.x];
        shm[threadIdx.y*_shmwidth + threadIdx.x+blockDim.x] = _src[y+x3+blockDim.x+threadIdx.x];
        shm[threadIdx.y*_shmwidth + threadIdx.x+2*blockDim.x] = _src[y+x3+2*blockDim.x+threadIdx.x];
        
        __syncthreads();

        // Layout: || r|g|b|r || g|b|r|g || b|r|g|b || r|g|b|r || ..
        // Each thread processes three consecutive uchar4, i.e. four pixels
        unsigned int res1 = (unsigned int)shm[t3].x*r + shm[t3].y*g + shm[t3].z*b;
        unsigned int res2 = (unsigned int)shm[t3].w*r + shm[t3+1].x*g + shm[t3+1].y*b;
        unsigned int res3 = (unsigned int)shm[t3+1].z*r + shm[t3+1].w*g + shm[t3+2].x*b;
        unsigned int res4 = (unsigned int)shm[t3+2].y*r + shm[t3+2].z*g + shm[t3+2].w*b;

        // Store results
        uchar4 res;
        res.x = res1 >> 8;
        res.y = res2 >> 8;
        res.z = res3 >> 8;
        res.w = res4 >> 8;
        
        _dest[(blockIdx.y*blockDim.y+threadIdx.y)*_pitchd+x+threadIdx.x] = res;

    }
    
}


// Currently only working for exact block and grid dimensions
__global__ void _rgb2gray_8uto32f( const uchar4* _src, float* _dest,
                                  unsigned int _width,
                                  unsigned int _height,
                                  unsigned int _pitchs,
                                  unsigned int _pitchd,
                                  unsigned int _shmwidth) {
    
    unsigned int x =  blockIdx.x*blockDim.x; // address to block x
    unsigned int y =  (blockIdx.y*blockDim.y+threadIdx.y)*_pitchs; // address to row y
    unsigned int x3 = 3*x;
    unsigned int t3 = threadIdx.y*_shmwidth+3*threadIdx.x;
    
    extern __shared__ uchar4 shm[];
    
    unsigned int r = 0x0000004D;
    unsigned int g = 0x00000096;
    unsigned int b = 0x0000001D;
    
    if( blockIdx.y*blockDim.y+threadIdx.y < _height ) {
        
        shm[threadIdx.y*_shmwidth + threadIdx.x] = _src[y+x3+threadIdx.x];
        shm[threadIdx.y*_shmwidth + threadIdx.x+blockDim.x] = _src[y+x3+blockDim.x+threadIdx.x];
        shm[threadIdx.y*_shmwidth + threadIdx.x+2*blockDim.x] = _src[y+x3+2*blockDim.x+threadIdx.x];
        
        __syncthreads();
        
        // Layout: || r|g|b|r || g|b|r|g || b|r|g|b || r|g|b|r || ..
        // Each thread processes three consecutive uchar4, i.e. four pixels
        unsigned int res1 = (float)(shm[t3].x*r + shm[t3].y*g + shm[t3].z*b);
        unsigned int res2 = (float)(shm[t3].w*r + shm[t3+1].x*g + shm[t3+1].y*b);
        unsigned int res3 = (float)(shm[t3+1].z*r + shm[t3+1].w*g + shm[t3+2].x*b);
        unsigned int res4 = (float)(shm[t3+2].y*r + shm[t3+2].z*g + shm[t3+2].w*b);
        
        // Store results
        float res;
        res = res1*inv8255;
        _dest[(blockIdx.y*blockDim.y+threadIdx.y)*_pitchd+4*(x+threadIdx.x)] = res;
        res = res2 * inv8255;
        _dest[(blockIdx.y*blockDim.y+threadIdx.y)*_pitchd+4*(x+threadIdx.x)+1] = res;
        res = res3 * inv8255;
        _dest[(blockIdx.y*blockDim.y+threadIdx.y)*_pitchd+4*(x+threadIdx.x)+2] = res;
        res = res4 * inv8255;
        _dest[(blockIdx.y*blockDim.y+threadIdx.y)*_pitchd+4*(x+threadIdx.x)+3] = res;
        
        
    }
    
}

__device__ void _rgb2hsv_helper( float r, float g, float b,
                                float *h, float *s, float *v ) {
    
    int M = max( r , max(g,b) );
    int m = min( r , min(g,b) );
    float C = (float)(M-m);
    float Cinv = (C==0 ? 1.0f : 1.0f/C);
    
    *h = ((C==0) ? 0.0f:
          ((M==r) ? (g-b)*Cinv : ((M==g) ? (b-r)*Cinv+2 : (r-g)*Cinv+4) ));
    
    *h = ((*h<0) ? *h+6.0f : *h);
    
    *h *= 180.0f/6.0f;
    *v = M;
    *s = 255.0f*(C == 0.0f ? 0.0f : C / *v);
    
}


__global__ void _rgb2hsv_8u(const uchar4* _src, uchar4* _dest,
                            unsigned int _width,
                            unsigned int _height,
                            unsigned int _pitchs,
                            unsigned int _pitchd,
                            unsigned int _shmwidth) {
    
    unsigned int x =  blockIdx.x*blockDim.x; // address to block x on row 2*y
    unsigned int y =  (blockIdx.y*blockDim.y+threadIdx.y)*_pitchs; // address to row 2*y
    unsigned int x3 = 3*x;
    unsigned int t3 = threadIdx.y*_shmwidth+3*threadIdx.x;
    
    extern __shared__ uchar4 shm[];
    
    if( blockIdx.y*blockDim.y+threadIdx.y < _height ) {
        
        shm[threadIdx.y*_shmwidth + threadIdx.x] = _src[y+x3+threadIdx.x];
        shm[threadIdx.y*_shmwidth + threadIdx.x+blockDim.x] = _src[y+x3+blockDim.x+threadIdx.x];
        shm[threadIdx.y*_shmwidth + threadIdx.x+2*blockDim.x] = _src[y+x3+2*blockDim.x+threadIdx.x];
        
        __syncthreads();
        
        float h,s,v;
        
        _rgb2hsv_helper(shm[t3].x,shm[t3].y,shm[t3].z,
                        &h,&s,&v);
        shm[t3].x = h; shm[t3].y = s; shm[t3].z = v;
        
        _rgb2hsv_helper(shm[t3].w,shm[t3+1].x,shm[t3+1].y,
                        &h,&s,&v);
        shm[t3].w = h; shm[t3+1].x = s; shm[t3+1].y = v;

        _rgb2hsv_helper(shm[t3+1].z,shm[t3+1].w,shm[t3+2].x,
                        &h,&s,&v);
        shm[t3+1].z = h; shm[t3+1].w = s; shm[t3+2].x = v;

        _rgb2hsv_helper(shm[t3+2].y,shm[t3+2].z,shm[t3+2].w,
                        &h,&s,&v);
        shm[t3+2].y = h; shm[t3+2].z = s; shm[t3+2].w = v;

        __syncthreads();
        
        _dest[y+x3+threadIdx.x] = shm[threadIdx.y*_shmwidth + threadIdx.x];
        _dest[y+x3+blockDim.x+threadIdx.x] = shm[threadIdx.y*_shmwidth + threadIdx.x+blockDim.x];
        _dest[y+x3+2*blockDim.x+threadIdx.x] = shm[threadIdx.y*_shmwidth + threadIdx.x+2*blockDim.x];
        
    }

    
}

__global__ void _rgbx2hsvx_8u( const uchar4* _src, uchar4* _dest,
                              unsigned int _width, unsigned int _height,
                              unsigned int _pitchs, unsigned int _pitchd) {
    
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
    

    if( y < _height ) {
        
        uchar4 tmp = _src[y*_pitchs+x];
        
        int r = (int)tmp.x;
        int g = (int)tmp.y;
        int b = (int)tmp.z;
        
        int M = max( r , max(g,b) );
        int m = min( r , min(g,b) );
        float C = (float)(M-m);
        float Cinv = (C==0 ? 1.0f : 1.0f/C);
        
        float H = ((C==0) ? 0.0f:
                   ((M==r) ? (g-b)*Cinv : ((M==g) ? (b-r)*Cinv+2 : (r-g)*Cinv+4) ));
        
        H = ((H<0) ? H+6.0f : H);
        
        H *= 255/6;
        float V = M;
        float S = 255.0f*(C == 0 ? 0 : C/V);
        
        tmp.x = (unsigned char)H;
        tmp.y = (unsigned char)S;
        tmp.z = (unsigned char)V;
        
        _dest[y*_pitchd+x] = tmp;
        
    }
    
}



///////////////////////////////////////////////
///////////////// BGR <--> Lab ////////////////
///////////////////////////////////////////////
/*__global__ void _rgb2lab_8u( const unsigned char* _src,
 unsigned char* _dest,
 unsigned int _size ) {
 
 unsigned int x3 = 3*(blockDim.x*blockIdx.x);
 
 float r = (float)_src[x3];
 float g = (float)_src[x3+1];
 float b = (float)_src[x3+2];
 float x,y,z;
 
 float t = 0.0088565f;
 
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
 _dest[x3] = (116.0f*y3 - 16.0f) / 100.0f;
 } else {
 _dest[x3] = (903.3*y) / 100.0f;
 }
 _dest[x3+1] = 500.0f * (x-y) / 256.0f + 0.5f;
 _dest[x3+2] = 200.0f * (y-z) / 256.0f + 0.5f;
 
 }*/




/*__constant float k_8u_32f[256] = {
 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
 60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f,
 70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
 80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f,
 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f,
 100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f,
 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f,
 120.0f, 121.0f, 122.0f, 123.0f, 124.0f, 125.0f, 126.0f, 127.0f, 128.0f, 129.0f,
 130.0f, 131.0f, 132.0f, 133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f, 139.0f,
 140.0f, 141.0f, 142.0f, 143.0f, 144.0f, 145.0f, 146.0f, 147.0f, 148.0f, 149.0f,
 150.0f, 151.0f, 152.0f, 153.0f, 154.0f, 155.0f, 156.0f, 157.0f, 158.0f, 159.0f,
 160.0f, 161.0f, 162.0f, 163.0f, 164.0f, 165.0f, 166.0f, 167.0f, 168.0f, 169.0f,
 170.0f, 171.0f, 172.0f, 173.0f, 174.0f, 175.0f, 176.0f, 177.0f, 178.0f, 179.0f,
 180.0f, 181.0f, 182.0f, 183.0f, 184.0f, 185.0f, 186.0f, 187.0f, 188.0f, 189.0f,
 190.0f, 191.0f, 192.0f, 193.0f, 194.0f, 195.0f, 196.0f, 197.0f, 198.0f, 199.0f,
 200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f, 206.0f, 207.0f, 208.0f, 209.0f,
 210.0f, 211.0f, 212.0f, 213.0f, 214.0f, 215.0f, 216.0f, 217.0f, 218.0f, 219.0f,
 220.0f, 221.0f, 222.0f, 223.0f, 224.0f, 225.0f, 226.0f, 227.0f, 228.0f, 229.0f,
 230.0f, 231.0f, 232.0f, 233.0f, 234.0f, 235.0f, 236.0f, 237.0f, 238.0f, 239.0f,
 240.0f, 241.0f, 242.0f, 243.0f, 244.0f, 245.0f, 246.0f, 247.0f, 248.0f, 249.0f,
 250.0f, 251.0f, 252.0f, 253.0f, 254.0f, 255.0f};
 
 __constant float k_post_coeffs[] = { 2.55f, 0.0f, 0.72033898305084743f,
 96.525423728813564f, 0.99609375f, 139.453125f };
 
 __constant float k_pre_coeffs[] = { 0.39215686274509809f, 0.0f, 1.388235294117647f,
 -134.0f, 1.003921568627451f, -140.0f };
 
 
 __constant float k_jet[] = {
 0.000, 0.000, 0.516,
 0.000, 0.000, 0.531,
 0.000, 0.000, 0.547,
 0.000, 0.000, 0.562,
 0.000, 0.000, 0.578,
 0.000, 0.000, 0.594,
 0.000, 0.000, 0.609,
 0.000, 0.000, 0.625,
 0.000, 0.000, 0.641,
 0.000, 0.000, 0.656,
 0.000, 0.000, 0.672,
 0.000, 0.000, 0.688,
 0.000, 0.000, 0.703,
 0.000, 0.000, 0.719,
 0.000, 0.000, 0.734,
 0.000, 0.000, 0.750,
 0.000, 0.000, 0.766,
 0.000, 0.000, 0.781,
 0.000, 0.000, 0.797,
 0.000, 0.000, 0.812,
 0.000, 0.000, 0.828,
 0.000, 0.000, 0.844,
 0.000, 0.000, 0.859,
 0.000, 0.000, 0.875,
 0.000, 0.000, 0.891,
 0.000, 0.000, 0.906,
 0.000, 0.000, 0.922,
 0.000, 0.000, 0.938,
 0.000, 0.000, 0.953,
 0.000, 0.000, 0.969,
 0.000, 0.000, 0.984,
 0.000, 0.000, 1.000,
 0.000, 0.016, 1.000,
 0.000, 0.031, 1.000,
 0.000, 0.047, 1.000,
 0.000, 0.062, 1.000,
 0.000, 0.078, 1.000,
 0.000, 0.094, 1.000,
 0.000, 0.109, 1.000,
 0.000, 0.125, 1.000,
 0.000, 0.141, 1.000,
 0.000, 0.156, 1.000,
 0.000, 0.172, 1.000,
 0.000, 0.188, 1.000,
 0.000, 0.203, 1.000,
 0.000, 0.219, 1.000,
 0.000, 0.234, 1.000,
 0.000, 0.250, 1.000,
 0.000, 0.266, 1.000,
 0.000, 0.281, 1.000,
 0.000, 0.297, 1.000,
 0.000, 0.312, 1.000,
 0.000, 0.328, 1.000,
 0.000, 0.344, 1.000,
 0.000, 0.359, 1.000,
 0.000, 0.375, 1.000,
 0.000, 0.391, 1.000,
 0.000, 0.406, 1.000,
 0.000, 0.422, 1.000,
 0.000, 0.438, 1.000,
 0.000, 0.453, 1.000,
 0.000, 0.469, 1.000,
 0.000, 0.484, 1.000,
 0.000, 0.500, 1.000,
 0.000, 0.516, 1.000,
 0.000, 0.531, 1.000,
 0.000, 0.547, 1.000,
 0.000, 0.562, 1.000,
 0.000, 0.578, 1.000,
 0.000, 0.594, 1.000,
 0.000, 0.609, 1.000,
 0.000, 0.625, 1.000,
 0.000, 0.641, 1.000,
 0.000, 0.656, 1.000,
 0.000, 0.672, 1.000,
 0.000, 0.688, 1.000,
 0.000, 0.703, 1.000,
 0.000, 0.719, 1.000,
 0.000, 0.734, 1.000,
 0.000, 0.750, 1.000,
 0.000, 0.766, 1.000,
 0.000, 0.781, 1.000,
 0.000, 0.797, 1.000,
 0.000, 0.812, 1.000,
 0.000, 0.828, 1.000,
 0.000, 0.844, 1.000,
 0.000, 0.859, 1.000,
 0.000, 0.875, 1.000,
 0.000, 0.891, 1.000,
 0.000, 0.906, 1.000,
 0.000, 0.922, 1.000,
 0.000, 0.938, 1.000,
 0.000, 0.953, 1.000,
 0.000, 0.969, 1.000,
 0.000, 0.984, 1.000,
 0.000, 1.000, 1.000,
 0.016, 1.000, 0.984,
 0.031, 1.000, 0.969,
 0.047, 1.000, 0.953,
 0.062, 1.000, 0.938,
 0.078, 1.000, 0.922,
 0.094, 1.000, 0.906,
 0.109, 1.000, 0.891,
 0.125, 1.000, 0.875,
 0.141, 1.000, 0.859,
 0.156, 1.000, 0.844,
 0.172, 1.000, 0.828,
 0.188, 1.000, 0.812,
 0.203, 1.000, 0.797,
 0.219, 1.000, 0.781,
 0.234, 1.000, 0.766,
 0.250, 1.000, 0.750,
 0.266, 1.000, 0.734,
 0.281, 1.000, 0.719,
 0.297, 1.000, 0.703,
 0.312, 1.000, 0.688,
 0.328, 1.000, 0.672,
 0.344, 1.000, 0.656,
 0.359, 1.000, 0.641,
 0.375, 1.000, 0.625,
 0.391, 1.000, 0.609,
 0.406, 1.000, 0.594,
 0.422, 1.000, 0.578,
 0.438, 1.000, 0.562,
 0.453, 1.000, 0.547,
 0.469, 1.000, 0.531,
 0.484, 1.000, 0.516,
 0.500, 1.000, 0.500,
 0.516, 1.000, 0.484,
 0.531, 1.000, 0.469,
 0.547, 1.000, 0.453,
 0.562, 1.000, 0.438,
 0.578, 1.000, 0.422,
 0.594, 1.000, 0.406,
 0.609, 1.000, 0.391,
 0.625, 1.000, 0.375,
 0.641, 1.000, 0.359,
 0.656, 1.000, 0.344,
 0.672, 1.000, 0.328,
 0.688, 1.000, 0.312,
 0.703, 1.000, 0.297,
 0.719, 1.000, 0.281,
 0.734, 1.000, 0.266,
 0.750, 1.000, 0.250,
 0.766, 1.000, 0.234,
 0.781, 1.000, 0.219,
 0.797, 1.000, 0.203,
 0.812, 1.000, 0.188,
 0.828, 1.000, 0.172,
 0.844, 1.000, 0.156,
 0.859, 1.000, 0.141,
 0.875, 1.000, 0.125,
 0.891, 1.000, 0.109,
 0.906, 1.000, 0.094,
 0.922, 1.000, 0.078,
 0.938, 1.000, 0.062,
 0.953, 1.000, 0.047,
 0.969, 1.000, 0.031,
 0.984, 1.000, 0.016,
 1.000, 1.000, 0.000,
 1.000, 0.984, 0.000,
 1.000, 0.969, 0.000,
 1.000, 0.953, 0.000,
 1.000, 0.938, 0.000,
 1.000, 0.922, 0.000,
 1.000, 0.906, 0.000,
 1.000, 0.891, 0.000,
 1.000, 0.875, 0.000,
 1.000, 0.859, 0.000,
 1.000, 0.844, 0.000,
 1.000, 0.828, 0.000,
 1.000, 0.812, 0.000,
 1.000, 0.797, 0.000,
 1.000, 0.781, 0.000,
 1.000, 0.766, 0.000,
 1.000, 0.750, 0.000,
 1.000, 0.734, 0.000,
 1.000, 0.719, 0.000,
 1.000, 0.703, 0.000,
 1.000, 0.688, 0.000,
 1.000, 0.672, 0.000,
 1.000, 0.656, 0.000,
 1.000, 0.641, 0.000,
 1.000, 0.625, 0.000,
 1.000, 0.609, 0.000,
 1.000, 0.594, 0.000,
 1.000, 0.578, 0.000,
 1.000, 0.562, 0.000,
 1.000, 0.547, 0.000,
 1.000, 0.531, 0.000,
 1.000, 0.516, 0.000,
 1.000, 0.500, 0.000,
 1.000, 0.484, 0.000,
 1.000, 0.469, 0.000,
 1.000, 0.453, 0.000,
 1.000, 0.438, 0.000,
 1.000, 0.422, 0.000,
 1.000, 0.406, 0.000,
 1.000, 0.391, 0.000,
 1.000, 0.375, 0.000,
 1.000, 0.359, 0.000,
 1.000, 0.344, 0.000,
 1.000, 0.328, 0.000,
 1.000, 0.312, 0.000,
 1.000, 0.297, 0.000,
 1.000, 0.281, 0.000,
 1.000, 0.266, 0.000,
 1.000, 0.250, 0.000,
 1.000, 0.234, 0.000,
 1.000, 0.219, 0.000,
 1.000, 0.203, 0.000,
 1.000, 0.188, 0.000,
 1.000, 0.172, 0.000,
 1.000, 0.156, 0.000,
 1.000, 0.141, 0.000,
 1.000, 0.125, 0.000,
 1.000, 0.109, 0.000,
 1.000, 0.094, 0.000,
 1.000, 0.078, 0.000,
 1.000, 0.062, 0.000,
 1.000, 0.047, 0.000,
 1.000, 0.031, 0.000,
 1.000, 0.016, 0.000,
 1.000, 0.000, 0.000,
 0.984, 0.000, 0.000,
 0.969, 0.000, 0.000,
 0.953, 0.000, 0.000,
 0.938, 0.000, 0.000,
 0.922, 0.000, 0.000,
 0.906, 0.000, 0.000,
 0.891, 0.000, 0.000,
 0.875, 0.000, 0.000,
 0.859, 0.000, 0.000,
 0.844, 0.000, 0.000,
 0.828, 0.000, 0.000,
 0.812, 0.000, 0.000,
 0.797, 0.000, 0.000,
 0.781, 0.000, 0.000,
 0.766, 0.000, 0.000,
 0.750, 0.000, 0.000,
 0.734, 0.000, 0.000,
 0.719, 0.000, 0.000,
 0.703, 0.000, 0.000,
 0.688, 0.000, 0.000,
 0.672, 0.000, 0.000,
 0.656, 0.000, 0.000,
 0.641, 0.000, 0.000,
 0.625, 0.000, 0.000,
 0.609, 0.000, 0.000,
 0.594, 0.000, 0.000,
 0.578, 0.000, 0.000,
 0.562, 0.000, 0.000,
 0.547, 0.000, 0.000,
 0.531, 0.000, 0.000,
 0.516, 0.000, 0.000,
 0.500, 0.000, 0.000
 };
 
 
 ///////////////////////////////////////////////
 //////////////////// DEFINES //////////////////
 ///////////////////////////////////////////////
 #define CAST_8U(t) (uchar)(!(((int)t) & ~255) ? (t) : (t) > 0 ? 255 : 0)
 
 #define inv255 (.003921568627451f);
 
 
 __kernel void gray2jet_float(const __global float* _src,
 __global float* _dest,
 unsigned int _size) {
 
 unsigned int idx = get_global_id(0);
 if( idx < _size ) {
 int val = 3*(int)(255.0f*_src[idx]);
 _dest[3*idx]   = k_jet[val];
 _dest[3*idx+1] = k_jet[val+1];
 _dest[3*idx+2] = k_jet[val+2];
 }
 
 }
 
 __kernel void gray2jet_8u(const __global uchar* _src,
 __global uchar* _dest,
 unsigned int _size) {
 
 unsigned int idx = get_global_id(0);
 if( idx < _size ) {
 int val = 3*_src[idx];
 _dest[3*idx]   = k_jet[val];
 _dest[3*idx+1] = k_jet[val+1];
 _dest[3*idx+2] = k_jet[val+2];
 }
 
 }
 */
 
 
/*
 ///////////////////////////////////////////////
 ///////////////// RGB --> GRAY ////////////////
 ///////////////////////////////////////////////
 
 __kernel void rgb2gray_32fto8u(const __global float *src,
 __global uchar *dest,
 __local  float *localimg,
 unsigned int img_size) {
 
 const int localX = get_local_id(0);
 const int lsizeX = get_local_size(0);
 const int globalX = get_group_id(0)*3*lsizeX + localX;
 const int global_memid = get_global_id(0);
 
 localimg[localX] = src[globalX];
 localimg[localX+lsizeX] = src[globalX+lsizeX];
 localimg[localX+2*lsizeX] = src[globalX+2*lsizeX];
 
 // Synchronize
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(global_memid < img_size) {
 int localX3 = 3*localX;
 float t = (0.30f * localimg[localX3] +
 0.59f * localimg[localX3+1] +
 0.11f * localimg[localX3+2])*255.0f;
 dest[global_memid] = CAST_8U(t);
 }
 
 }
 
 __kernel void rgb2gray_32f(const __global float *src,
 __global float *dest,
 __local  float *localimg,
 unsigned int img_size) {
 
 const int localX = get_local_id(0);
 const int lsizeX = get_local_size(0);
 const int globalX = get_group_id(0)*3*lsizeX + localX;
 const int global_memid = get_global_id(0);
 
 localimg[localX] = src[globalX];
 localimg[localX+lsizeX] = src[globalX+lsizeX];
 localimg[localX+2*lsizeX] = src[globalX+2*lsizeX];
 
 // Synchronize
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(global_memid < img_size) {
 int localX3 = 3*localX;
 dest[global_memid] = (0.30f * localimg[localX3] +
 0.59f * localimg[localX3+1] +
 0.11f * localimg[localX3+2]);
 
 }
 
 }
 
 __kernel void rgb2gray_8uto32f(const __global uchar* _src,
 __global float* _dest,
 __local  float* _shm,
 unsigned int _size) {
 
 const int lx = get_local_id(0);
 const int lszx = get_local_size(0);
 const int x = get_group_id(0)*3*lszx + lx;
 const int idx = get_global_id(0);
 
 if( idx < _size ) {
 _shm[lx] = (float)_src[x]*inv255;
 _shm[lx+lszx] = (float)_src[x+lszx]*inv255;
 _shm[lx+2*lszx] = (float)_src[x+2*lszx]*inv255;
 }
 
 // Synchronize
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if( idx < _size ) {
 int lx3 = 3*lx;
 _dest[idx] = (0.30f * _shm[lx3] +
 0.59f * _shm[lx3+1] +
 0.11f * _shm[lx3+2]);
 }
 
 }
 
 
 ///////////////////////////////////////////////
 ///////////////// RGB <--> Luv ////////////////
 ///////////////////////////////////////////////
 inline void rgb2luv(const float* r, const float* g, const float* b,
 float* L, float* u, float* v) {
 
 float x,y,z,t;
 
 x = *r*0.412453f + *g*0.357580f + *b*0.180423f;
 y = *r*0.212671f + *g*0.715160f + *b*0.072169f;
 z = *r*0.019334f + *g*0.119193f + *b*0.950227f;
 
 if( x == 0.0f && y == 0.0f && z == 0.0f) {
 *L = *u = *v = 0.0f;
 } else {
 if( y > 0.008856f ) {
 *L = 116.0f*cbrt(y) - 16.0f;
 } else {
 *L = 903.3f*y;
 }
 
 t = 1.0f / (x + 15.0f*y + 3.0f*z);
 *u = 4.0f*x*t;
 *v = 9.0f*y*t;
 
 *u = 13.0f*(*L)*(*u - 0.19793943f);
 *v = 13.0f*(*L)*(*v - 0.46831096f);
 }
 
 }
 
 
 inline void luv2rgb(const float* L, const float* u, const float* v,
 float* r, float* g, float* b) {
 
 float x,y,z,t;
 float _L;
 
 if( *L >= 8 ) {
 t = (*L + 16.0f)*0.008620689655172f;
 y = t*t*t;
 _L = *L;
 } else {
 y = *L * 0.001107051920735f;
 _L = ( *L > 0.001f ? *L : 0.001f );
 }
 
 float u1,v1;
 t = 1.0f/(13.0f * (_L));
 u1 = (*u)*t + 0.19793943f;
 v1 = (*v)*t + 0.46831096;
 x = 2.25f * u1 * y / v1 ;
 z = (12.0f - 3.0f*u1 - 20.0f*v1) * y / (4.0f*v1);
 
 *r =  3.240479f*x - 1.53715f*y  - 0.498535f*z;
 *g = -0.969256f*x + 1.875991f*y + 0.041556f*z;
 *b =  0.055648f*x - 0.204043f*y + 1.057311f*z;
 
 }
 
 
 ///////////////////////////////////////////////
 /////////////// KERNEL FUNCTIONS //////////////
 ///////////////////////////////////////////////
 
 // Optimization is possible for Compute Device < 2.0
 
 __kernel void rgb2luv_8u(const __global uchar* src,
 __global uchar* dest,
 __local  uchar* localimg,
 unsigned int img_size) {
 
 const int localX = get_local_id(0);
 const int lsizeX = get_local_size(0);
 const int globalX = get_group_id(0)*3*lsizeX + localX;
 const int global_memid = 3*get_global_id(0);
 
 localimg[localX] = src[globalX];
 localimg[localX+lsizeX] = src[globalX+lsizeX];
 localimg[localX+2*lsizeX] = src[globalX+2*lsizeX];
 
 // Synchronize
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(get_global_id(0) < img_size) {
 int localX3 = 3*localX;
 
 uchar r = localimg[localX3];
 uchar g = localimg[localX3+1];
 uchar b = localimg[localX3+2];
 
 float rf = k_8u_32f[(int)r]*0.0039215686274509803f;
 float gf = k_8u_32f[(int)g]*0.0039215686274509803f;
 float bf = k_8u_32f[(int)b]*0.0039215686274509803f;
 
 float L,u,v;
 
 rgb2luv(&rf, &gf, &bf, &L, &u, &v);
 
 int Li = k_post_coeffs[0]*L + k_post_coeffs[1];
 int ui = k_post_coeffs[2]*u + k_post_coeffs[3];
 int vi = k_post_coeffs[4]*v + k_post_coeffs[5];
 
 dest[global_memid] = CAST_8U(Li);
 dest[global_memid+1] = CAST_8U(ui);
 dest[global_memid+2] = CAST_8U(vi);
 
 }
 }
 
 __kernel void luv2rgb_8u(const __global uchar* src,
 __global uchar* dest,
 __local  uchar* localimg,
 unsigned int img_size) {
 
 const int localX = get_local_id(0);
 const int lsizeX = get_local_size(0);
 const int globalX = get_group_id(0)*3*lsizeX + localX;
 const int global_memid = 3*get_global_id(0);
 
 localimg[localX] = (float)src[globalX];
 localimg[localX+lsizeX] = (float)src[globalX+lsizeX];
 localimg[localX+2*lsizeX] = (float)src[globalX+2*lsizeX];
 
 // Synchronize
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(get_global_id(0) < img_size) {
 int localX3 = 3*localX;
 
 uchar L = localimg[localX3];
 uchar u = localimg[localX3+1];
 uchar v = localimg[localX3+2];
 
 float Lf = k_8u_32f[(int)L]*k_pre_coeffs[0] + k_pre_coeffs[1];
 float uf = k_8u_32f[(int)u]*k_pre_coeffs[2] + k_pre_coeffs[3];
 float vf = k_8u_32f[(int)v]*k_pre_coeffs[4] + k_pre_coeffs[5];
 
 float r,g,b;
 
 luv2rgb(&Lf, &uf, &vf, &r, &g, &b);
 
 int ri = (int)(255.0f*r + (r >= 0.0f ? 0.5f : -0.5f));
 ri = (ri < 0.0f ? 0.0f : ri);
 int gi = (int)(255.0f*g + (g >= 0.0f ? 0.5f : -0.5f));
 gi = (gi < 0.0f ? 0.0f : gi);
 int bi = (int)(255.0f*b + (b >= 0.0f ? 0.5f : -0.5f));
 bi = (bi < 0.0f ? 0.0f : bi);
 
 dest[global_memid] = (uchar)ri;
 dest[global_memid+1] = (uchar)gi;
 dest[global_memid+2] = (uchar)bi;
 }
 
 }
 
 __kernel void rgb2luv_32f(const __global float* src,
 __global float* dest,
 __local float*  localimg,
 unsigned int img_size) {
 
 const int localX = get_local_id(0);
 const int lsizeX = get_local_size(0);
 const int globalX = get_group_id(0)*3*lsizeX + localX;
 const int global_memid = 3*get_global_id(0);
 
 localimg[localX] = (float)src[globalX];
 localimg[localX+lsizeX] = (float)src[globalX+lsizeX];
 localimg[localX+2*lsizeX] = (float)src[globalX+2*lsizeX];
 
 // Synchronize
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(get_global_id(0) < img_size) {
 int localX3 = 3*localX;
 
 float r = localimg[localX3];
 float g = localimg[localX3+1];
 float b = localimg[localX3+2];
 
 float L,u,v;
 
 rgb2luv(&r, &g, &b, &L, &u, &v);
 
 dest[global_memid] = L;
 dest[global_memid+1] = u;
 dest[global_memid+2] = v;
 }
 
 }
 
 __kernel void luv2rgb_32f(const __global float* src,
 __global float* dest,
 __local  float* localimg,
 unsigned int img_size) {
 
 const int localX = get_local_id(0);
 const int lsizeX = get_local_size(0);
 const int globalX = get_group_id(0)*3*lsizeX + localX;
 const int global_memid = 3*get_global_id(0);
 
 localimg[localX] = src[globalX];
 localimg[localX+lsizeX] = src[globalX+lsizeX];
 localimg[localX+2*lsizeX] = src[globalX+2*lsizeX];
 
 // Synchronize
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(get_global_id(0) < img_size) {
 int localX3 = 3*localX;
 
 float L = localimg[localX3];
 float u = localimg[localX3+1];
 float v = localimg[localX3+2];
 
 float r, g, b;
 
 luv2rgb(&L, &u, &v, &r, &g, &b);
 
 dest[global_memid] = r;
 dest[global_memid+1] = g;
 dest[global_memid+2] = b;
 }
 
 }
 
 ///////////////////////////////////////////////
 //////////////// BGR <--> YCrCb////////////////
 ///////////////////////////////////////////////
 
 __kernel void bgr2ycrcb(__global float* src,
 __global float* dest,
 unsigned int width,
 unsigned int height) {
 
 unsigned int globalX = get_global_id(0);
 unsigned int globalY = get_global_id(1);
 
 unsigned int global_memid = 3*mad24(globalY,width,globalX);
 
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
 */

#include <cstdio>

namespace pvcore {
    
    cudaError_t __convertColorGPU(const unsigned char* _src,
                                  unsigned char* _dest,
                                  unsigned int _width,
                                  unsigned int _height,
                                  unsigned int _pitchs,
                                  unsigned int _pitchd,
                                  unsigned int _kernel,
                                  dim3 _blockDim ) {


        unsigned int size = _width*_height;
        
        dim3 gridDim;
        gridDim.z = 1;
        
        gridDim.x = GLOBAL_SIZE( _width, _blockDim.x );
        gridDim.y = GLOBAL_SIZE( _height, _blockDim.y );
        
        size_t shmSize = 3*_blockDim.x*_blockDim.y*sizeof(uchar4);
        
        switch( _kernel ) {
                // ================ RGB2GRAY ==============
            case rgb2gray_8u:
                gridDim.x = GLOBAL_SIZE( _width/4, _blockDim.x );
                gridDim.y = GLOBAL_SIZE( _height, _blockDim.y );
                _rgb2gray_8u<<< gridDim,_blockDim, shmSize >>>( (uchar4*)_src,(uchar4*)_dest, _width/4, _height, _pitchs/4, _pitchd/4, 3*_blockDim.x );
                break;
                
            case rgb2gray_8uto32f:
                gridDim.x = GLOBAL_SIZE( _width/4, _blockDim.x );
                gridDim.y = GLOBAL_SIZE( _height, _blockDim.y );
                _rgb2gray_8uto32f<<<gridDim,_blockDim,shmSize >>>( (uchar4*)_src, (float*)_dest, _width/4, _height, _pitchs/4, _pitchd, 3*_blockDim.x );
                break;
                
                // ================ RGB2HSV ==============
            case rgb2hsv_8u:
                gridDim.x = GLOBAL_SIZE( _width/4, _blockDim.x );
                gridDim.y = GLOBAL_SIZE( _height, _blockDim.y );
                _rgb2hsv_8u<<< gridDim,_blockDim, shmSize >>>( (uchar4*)_src,(uchar4*)_dest, _width/4, _height, _pitchs/4, _pitchd/4, 3*_blockDim.x );
                break;
                
            case rgbx2hsvx_8u:
                _rgbx2hsvx_8u<<< gridDim,_blockDim, shmSize >>>( (uchar4*)_src,(uchar4*) _dest, _width, _height, _pitchs/4, _pitchd/4 );
                break;
                // ================ RGB2BGR ==============
            case rgb2bgr_8u:
                gridDim.x = GLOBAL_SIZE( _width/4, _blockDim.x );
                gridDim.y = GLOBAL_SIZE( _height, _blockDim.y );
                _rgb2bgr_8u<<< gridDim,_blockDim, shmSize >>>( (uchar4*)_src,(uchar4*)_dest, _width/4, _height, _pitchs/4, _pitchd/4, 3*_blockDim.x );
                break;
                
            case rgbx2bgrx_8u:
                gridDim.x = GLOBAL_SIZE( _width, _blockDim.x );
                gridDim.y = GLOBAL_SIZE( _height, _blockDim.y );
                _rgbx2bgrx_8u<<< gridDim,_blockDim, shmSize >>>( (uchar4*)_src,(uchar4*)_dest, _width, _height, _pitchs/4, _pitchd/4 );
                break;
                
            case rgb2bgr_32f:
                _rgb2bgr_32f<<< gridDim,_blockDim >>>( (float*)_src,(float*)_dest, _width, _height,
                                                      _pitchs, _pitchd );
                break;
                
            case rgbx2bgrx_32f:
                _rgbx2bgrx_32f<<< gridDim,_blockDim >>>( (float*)_src,(float*)_dest, _width, _height,
                                                      _pitchs, _pitchd );
                break;
                // ================ BAYER2RGB ==============
            case bayergr2rgbx_8u:
            case bayergr2rgb_8u:
            {
                cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
                cudaBindTexture2D( 0 ,&tex_u8, _src, &channelDesc, _width, _height, _pitchs );
                tex_u8.addressMode[0] = cudaAddressModeClamp;
                tex_u8.addressMode[1] = cudaAddressModeClamp;
                tex_u8.filterMode = cudaFilterModePoint;
                tex_u8.normalized = false;
                gridDim.y = GLOBAL_SIZE( _height/2, _blockDim.y );
                switch( _kernel ) {
                    case bayergr2rgbx_8u:
                        gridDim.x = GLOBAL_SIZE( _width/2, _blockDim.x );
                        _bayer2rgbx_8u_tex<<<gridDim,_blockDim,(2*_blockDim.x+1)*2*_blockDim.y*sizeof(uchar4)>>>((uchar4*)_dest,_width,_height,_pitchd/4, 2*_blockDim.x+1);
                        break;
                    case bayergr2rgb_8u:
                        gridDim.x = GLOBAL_SIZE( _width/4, _blockDim.x );
                        _bayer2rgb_8u_tex<<<gridDim,_blockDim>>>((uchar4*)_dest,_width,_height,_pitchd/4);
                        break;
                }
                cudaUnbindTexture(tex_u8);
            }
                break;

            default:
                break;
        }
        
        return cudaSuccess;
        
    }
    
}




