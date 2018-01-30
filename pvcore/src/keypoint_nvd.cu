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

#include "pvcore/keypoint.h"

#include "pvcore/cuda_functions.h"

#include "pvcore/common.h"

#include <math_constants.h>



texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> tex_8u;
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex_32u;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_32f;

texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex_32u_0;
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex_32u_1;

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> tex_8u4_0;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> tex_8u4_1;

__global__ void saliency_3(float* _dest,
                           int _width,
                           int _height,
                           int _pitchd,
                           float _threshold) {
    
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    
    if ( y < _height && x < _width ) {
        int ip = tex2D(tex_8u,x,y);
        // Get local contrast
        int contrast = abs(ip - tex2D(tex_8u,x,y-3));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+1,y-3)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+2,y-2)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+3,y-1)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+3,y)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+3,y+1)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+2,y+2)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+1,y+3)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x,y+3)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-1,y+3)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-2,y+2)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-3,y+1)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-3,y)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-3,y-1)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-2,y-2)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-1,y-3)));
        
        float contrastf = contrast * (1.f/255.f);
        
        int saliency = abs(tex2D(tex_8u,x,y-3) + tex2D(tex_8u,x,y+3) - 2*ip);
        saliency = min(saliency,abs(tex2D(tex_8u,x+1,y-3) + tex2D(tex_8u,x-1,y+3) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+2,y-2) + tex2D(tex_8u,x-2,y+2) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+3,y-1) + tex2D(tex_8u,x-3,y+1) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+3,y) +   tex2D(tex_8u,x-3,y) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+3,y+1) + tex2D(tex_8u,x-3,y-1) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+2,y+2) + tex2D(tex_8u,x-2,y-2) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+1,y+3) + tex2D(tex_8u,x-1,y-3) - 2*ip));
        
        float saliencyf = saliency * (1.f/255.f);
        
        _dest[y*_pitchd+x] = contrastf < _threshold ? 0.0f : saliencyf/contrastf;
        
    }
}

__global__ void saliency_6(float* _dest,
                           int _width,
                           int _height,
                           int _pitchd,
                           float _threshold) {
    
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    
    if ( y < _height && x < _width ) {
        int ip = tex2D(tex_8u,x,y);
        // Get local contrast
        int contrast = abs(ip - tex2D(tex_8u,x,y-6));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+2,y-6)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+4,y-4)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+6,y-2)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+6,y  )));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+6,y+2)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+4,y+4)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x+2,y+6)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x  ,y+6)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-2,y+6)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-4,y+4)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-6,y+2)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-6,y  )));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-6,y-2)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-4,y-4)));
        contrast = max(contrast,abs(ip - tex2D(tex_8u,x-2,y-6)));
        
        float contrastf = contrast * (1.f/255.f);
        
        int saliency = abs(tex2D(tex_8u,x,y-6) + tex2D(tex_8u,x,y+6) - 2*ip);
        saliency = min(saliency,abs(tex2D(tex_8u,x+2,y-6) + tex2D(tex_8u,x-2,y+6) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+4,y-4) + tex2D(tex_8u,x-4,y+4) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+6,y-2) + tex2D(tex_8u,x-6,y+2) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+6,y)   + tex2D(tex_8u,x-6,y)   - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+6,y+2) + tex2D(tex_8u,x-6,y-2) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+4,y+4) + tex2D(tex_8u,x-4,y-4) - 2*ip));
        saliency = min(saliency,abs(tex2D(tex_8u,x+2,y+6) + tex2D(tex_8u,x-2,y-6) - 2*ip));
        
        float saliencyf = saliency * (1.f/255.f);
        
        float res = contrastf < _threshold ? 0.0f : saliencyf/contrastf;
        
        _dest[y*_pitchd+x] = fmax(_dest[y*_pitchd+x],res);
        
    }
}


__global__ void nonmax_sup(float* _dest,
                           int _width,
                           int _height,
                           int _pitchd) {
    
    
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    
    float pixelval = tex2D(tex_32f,x,y);
    
    float tmp =    tex2D(tex_32f,x-1,y-1);
    tmp = fmax(tmp,tex2D(tex_32f,x  ,y-1));
    tmp = fmax(tmp,tex2D(tex_32f,x+1,y-1));
    tmp = fmax(tmp,tex2D(tex_32f,x-1,y));
    tmp = fmax(tmp,tex2D(tex_32f,x+1,y));
    tmp = fmax(tmp,tex2D(tex_32f,x-1,y+1));
    tmp = fmax(tmp,tex2D(tex_32f,x  ,y+1));
    tmp = fmax(tmp,tex2D(tex_32f,x+1,y+1));
    
    _dest[y*_pitchd+x] = (pixelval >= tmp ? (pixelval > 0.25 ? pixelval : 0.0f) : 0.0f);
    
}


/**
 *  \brief Extracts features for all points in image tex_32u
 *
 *  \param _dest Destination image of feature points (unsigned char [16])
 *  \param _width Width of image
 *  \param _height Height of image
 *  \param _pitch Elements per row of _dest
 */
__global__ void weak_features_half(unsigned int* _dest,
                                   int _width,
                                   int _height,
                                   int _pitchd) {
    
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Allocate shared memory for 4x16x16 16-byte feature vectors, and add
    // extra byte for padding of
    __shared__ unsigned int shm[4*4*16*16+16*16];
    
    // Read contributions for eight pixels for four consecutive pixels
    unsigned int s0 = (tex2D(tex_32u, x  , y-3) & 0xF0F0F0F0) >> 4;
    unsigned int s1 = (tex2D(tex_32u, x+1, y-3) & 0xF0F0F0F0) >> 4;
    unsigned int s2 = (tex2D(tex_32u, x+2, y-2) & 0xF0F0F0F0) >> 4;
    unsigned int s3 = (tex2D(tex_32u, x+3, y-1) & 0xF0F0F0F0) >> 4;
    
    s0 |= tex2D(tex_32u, x+3, y  ) & 0xF0F0F0F0;
    s1 |= tex2D(tex_32u, x+3, y+1) & 0xF0F0F0F0;
    s2 |= tex2D(tex_32u, x+2, y+2) & 0xF0F0F0F0;
    s3 |= tex2D(tex_32u, x+1, y+3) & 0xF0F0F0F0;
    
    // After reading the contributions are stored in columns of bytes
    // in the unsigned integers. We need to transpose
    transpose(&s0,&s1,&s2,&s3);
    
    // Store 4 feature vectors in every row of shm
    shm[16*threadIdx.y*17+17*threadIdx.x] = s0;
    shm[16*threadIdx.y*17+17*threadIdx.x+4] = s1;
    shm[16*threadIdx.y*17+17*threadIdx.x+8] = s2;
    shm[16*threadIdx.y*17+17*threadIdx.x+12] = s3;
    
    // Read contributions for eight pixels for four consecutive pixels
    s0 = (tex2D(tex_32u, x  , y+3) & 0xF0F0F0F0) >> 4;
    s1 = (tex2D(tex_32u, x-1, y+3) & 0xF0F0F0F0) >> 4;
    s2 = (tex2D(tex_32u, x-2, y+2) & 0xF0F0F0F0) >> 4;
    s3 = (tex2D(tex_32u, x-3, y+1) & 0xF0F0F0F0) >> 4;
    
    s0 |= tex2D(tex_32u, x-3, y  ) & 0xF0F0F0F0;
    s1 |= tex2D(tex_32u, x-3, y-1) & 0xF0F0F0F0;
    s2 |= tex2D(tex_32u, x-2, y-2) & 0xF0F0F0F0;
    s3 |= tex2D(tex_32u, x-1, y-3) & 0xF0F0F0F0;
    
    // After reading the contributions are stored in columns of bytes
    // in the unsigned integers. We need to transpose
    transpose(&s0,&s1,&s2,&s3);
    
    // Store 4 feature vectors in every row of shm
    shm[16*threadIdx.y*17+17*threadIdx.x+1] = s0;
    shm[16*threadIdx.y*17+17*threadIdx.x+5] = s1;
    shm[16*threadIdx.y*17+17*threadIdx.x+9] = s2;
    shm[16*threadIdx.y*17+17*threadIdx.x+13] = s3;
    
    // Read contributions for eight pixels for four consecutive pixels
    s0 = (tex2D(tex_32u, x  , y-6) & 0xF0F0F0F0) >> 4;
    s1 = (tex2D(tex_32u, x+2, y-6) & 0xF0F0F0F0) >> 4;
    s2 = (tex2D(tex_32u, x+4, y-4) & 0xF0F0F0F0) >> 4;
    s3 = (tex2D(tex_32u, x+6, y-2) & 0xF0F0F0F0) >> 4;
    
    s0 |= tex2D(tex_32u, x+6, y  ) & 0xF0F0F0F0;
    s1 |= tex2D(tex_32u, x+6, y+2) & 0xF0F0F0F0;
    s2 |= tex2D(tex_32u, x+4, y+4) & 0xF0F0F0F0;
    s3 |= tex2D(tex_32u, x+2, y+6) & 0xF0F0F0F0;
    
    // After reading the contributions are stored in columns of bytes
    // in the unsigned integers. We need to transpose
    transpose(&s0,&s1,&s2,&s3);
    
    // Store 4 feature vectors in every row of shm
    shm[16*threadIdx.y*17+17*threadIdx.x+2] = s0;
    shm[16*threadIdx.y*17+17*threadIdx.x+6] = s1;
    shm[16*threadIdx.y*17+17*threadIdx.x+10] = s2;
    shm[16*threadIdx.y*17+17*threadIdx.x+14] = s3;
    
    // Read contributions for eight pixels for four consecutive pixels
    s0 = (tex2D(tex_32u, x  , y+6) & 0xF0F0F0F0) >> 4;
    s1 = (tex2D(tex_32u, x-2, y+6) & 0xF0F0F0F0) >> 4;
    s2 = (tex2D(tex_32u, x-4, y+4) & 0xF0F0F0F0) >> 4;
    s3 = (tex2D(tex_32u, x-6, y+2) & 0xF0F0F0F0) >> 4;
    
    s0 |= tex2D(tex_32u, x-6, y  ) & 0xF0F0F0F0;
    s1 |= tex2D(tex_32u, x-6, y-2) & 0xF0F0F0F0;
    s2 |= tex2D(tex_32u, x-4, y-4) & 0xF0F0F0F0;
    s3 |= tex2D(tex_32u, x-2, y-6) & 0xF0F0F0F0;
    
    // After reading the contributions are stored in columns of bytes
    // in the unsigned integers. We need to transpose
    transpose(&s0,&s1,&s2,&s3);
    
    // Store 4 feature vectors in every row of shm
    shm[16*threadIdx.y*17+17*threadIdx.x+3] = s0;
    shm[16*threadIdx.y*17+17*threadIdx.x+7] = s1;
    shm[16*threadIdx.y*17+17*threadIdx.x+11] = s2;
    shm[16*threadIdx.y*17+17*threadIdx.x+15] = s3;
    
    __syncthreads();
    
    // Store shared memory in _dest
#pragma unroll
    for (int i=0; i<16; ++i) {
        _dest[y*_pitchd+16*blockIdx.x*blockDim.x + i*blockDim.x + threadIdx.x] = shm[17*16*threadIdx.y + 17*i + threadIdx.x];
    }
    
}


__global__ void weak_features_full(unsigned int* _dest,
                                   int _width,
                                   int _height,
                                   int _pitchd) {
    
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Allocate shared memory for 4x16x16 16-byte feature vectors, and add
    // extra byte for padding of
    __shared__ unsigned int shm[4*4*16*16+16*16];
    
    // Read contributions for eight pixels for four consecutive pixels
    unsigned int s0 = tex2D(tex_32u, x  , y-3);
    unsigned int s1 = tex2D(tex_32u, x+2, y-2);
    unsigned int s2 = tex2D(tex_32u, x+3, y  );
    unsigned int s3 = tex2D(tex_32u, x+2, y+2);
    
    // After reading the contributions are stored in columns of bytes
    // in the unsigned integers. We need to transpose
    transpose(&s0,&s1,&s2,&s3);
    
    // Store 4 feature vectors in every row of shm
    shm[16*threadIdx.y*17+17*threadIdx.x] = s0;
    shm[16*threadIdx.y*17+17*threadIdx.x+4] = s1;
    shm[16*threadIdx.y*17+17*threadIdx.x+8] = s2;
    shm[16*threadIdx.y*17+17*threadIdx.x+12] = s3;
    
    // Read contributions for eight pixels for four consecutive pixels
    s0 = tex2D(tex_32u, x  , y+3);
    s1 = tex2D(tex_32u, x-2, y+2);
    s2 = tex2D(tex_32u, x-3, y  );
    s3 = tex2D(tex_32u, x-2, y-2);
    
    // After reading the contributions are stored in columns of bytes
    // in the unsigned integers. We need to transpose
    transpose(&s0,&s1,&s2,&s3);
    
    // Store 4 feature vectors in every row of shm
    shm[16*threadIdx.y*17+17*threadIdx.x+1] = s0;
    shm[16*threadIdx.y*17+17*threadIdx.x+5] = s1;
    shm[16*threadIdx.y*17+17*threadIdx.x+9] = s2;
    shm[16*threadIdx.y*17+17*threadIdx.x+13] = s3;
    
    // Read contributions for eight pixels for four consecutive pixels
    s0 = tex2D(tex_32u, x  , y-6);
    s1 = tex2D(tex_32u, x+4, y-4);
    s2 = tex2D(tex_32u, x+6, y  );
    s3 = tex2D(tex_32u, x+4, y+4);
    
    // After reading the contributions are stored in columns of bytes
    // in the unsigned integers. We need to transpose
    transpose(&s0,&s1,&s2,&s3);
    
    // Store 4 feature vectors in every row of shm
    shm[16*threadIdx.y*17+17*threadIdx.x+2] = s0;
    shm[16*threadIdx.y*17+17*threadIdx.x+6] = s1;
    shm[16*threadIdx.y*17+17*threadIdx.x+10] = s2;
    shm[16*threadIdx.y*17+17*threadIdx.x+14] = s3;
    
    // Read contributions for eight pixels for four consecutive pixels
    s0 = tex2D(tex_32u, x  , y+6);
    s1 = tex2D(tex_32u, x-4, y+4);
    s2 = tex2D(tex_32u, x-6, y  );
    s3 = tex2D(tex_32u, x-4, y-4);
    
    // After reading the contributions are stored in columns of bytes
    // in the unsigned integers. We need to transpose
    transpose(&s0,&s1,&s2,&s3);
    
    // Store 4 feature vectors in every row of shm
    shm[16*threadIdx.y*17+17*threadIdx.x+3] = s0;
    shm[16*threadIdx.y*17+17*threadIdx.x+7] = s1;
    shm[16*threadIdx.y*17+17*threadIdx.x+11] = s2;
    shm[16*threadIdx.y*17+17*threadIdx.x+15] = s3;
    
    __syncthreads();
    
    // Store shared memory in _dest
#pragma unroll
    for (int i=0; i<16; ++i) {
        _dest[y*_pitchd+16*blockIdx.x*blockDim.x + i*blockDim.x + threadIdx.x] = shm[17*16*threadIdx.y + 17*i + threadIdx.x];
    }
    
}


__global__ void count_points(float* _pts,
                             unsigned char* _count,
                             float _threshold,
                             int _pitch) {
    
	__shared__ unsigned char shm[16*8];
    
	// Global indices
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = 2*blockIdx.y*blockDim.y + threadIdx.y;
    
    int idx = y*_pitch + x;
    int reductionoffset = 8*_pitch;
    
	// Thread block indices
	int shmidx = threadIdx.y*16+threadIdx.x;
	// Read two bytes per thread
	shm[shmidx] = (_pts[idx] > _threshold ? 1 : 0) + (_pts[idx+reductionoffset] > _threshold ? 1 : 0);
	
	__syncthreads();
    
	// 64 numbers to add
	if( shmidx < 64 ) {
		shm[shmidx] += shm[shmidx+64];
		__syncthreads();
        
		if( shmidx < 32 ) {
			// 64 numbers to add
			shm[shmidx] += shm[shmidx+32];
            __syncthreads();
            
			if( shmidx < 16 ) {
				// 32 numbers to add
				shm[shmidx] += shm[shmidx+16];
                
				if( shmidx < 8 ) {
					// 16 numbers to add
					shm[shmidx] += shm[shmidx+8];
                    
					if( shmidx < 4 ) {
						// 8 numbers to add
						shm[shmidx] += shm[shmidx+4];
                        
						if( shmidx < 2 ) {
							// 4 numbers to add
							shm[shmidx] += shm[shmidx+2];
                            
							if( shmidx == 0 ) {
								// 2 numbers to add
								_count[blockIdx.y*gridDim.x+blockIdx.x] = (shm[0] + shm[1]);
							}
						}
					}
				}
			}
		}
	}
    
}



// _edges holds the binary image
// _ptsperblock holds the block's index of the point list and the number of points
__global__ void createPointList(float* _src,
								int* _ptsperblock,
								pvcore::keypt* _pointlist,
                                int _pitch) {
    
	// Index of the block - will give how many points should be stored in _pointlist
	int blockidx = blockIdx.y*gridDim.x + blockIdx.x;
	int ptidx = _ptsperblock[blockidx];
    // Number of points to save
	int npts  = _ptsperblock[blockidx+1] - ptidx;
    
	// Thread block indices (0-127)
	int shmidx = threadIdx.y*16 + threadIdx.x;
    int direction = threadIdx.y*16 + threadIdx.x;
    
	
    // No need to sort if no points in block
	if( npts == 0 ) {
		return;
	}
    
    int reductionoffset = 8*_pitch;
    
	__shared__ float shm_points[16*17];
	__shared__ unsigned char ptlist[16*17];
    
    
	// Indices to global point
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = 2*blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*_pitch + x;
    
    
	// Shared memory holding a copy of the feature point image
	shm_points[shmidx] = _src[idx];
	shm_points[shmidx+128] = _src[idx+reductionoffset];
    // Shared memory holding the (local) indices of the feature points
	ptlist[shmidx] = shmidx;
	ptlist[shmidx+128] = shmidx+128;
	__syncthreads();
    
	// Sort 256 numbers, largest first !!POTENTIAL BANK CONFLICTS!!
	// Implements bitonic merge sort
	// Loop
	for( int i=1; i < 256; i <<= 1 ) {
		// Outer loop determines sorting direction
		int sortdir = (direction & i) > 0 ? 0 : 1;
        
		// Inner loop
		for( int j=i; j > 0; j >>= 1 ) {
			// New index
			int mask = 0x0FFFFFFF * j;
			int tidx = ((shmidx&mask) << 1) + (shmidx & ~mask);
			atomicSort(shm_points,ptlist,tidx,j,j*sortdir);
			__syncthreads();
		}
	}
    
	// Save points in list
	if( shmidx < npts ) {
		_pointlist[ptidx + shmidx].x = blockIdx.x * blockDim.x + (ptlist[shmidx] & 0x0F);
		_pointlist[ptidx + shmidx].y = 2*blockIdx.y * blockDim.y + ((ptlist[shmidx] & 0xF0) >> 4);
	}
    
    //	_edges[idx] = shm_points[shmidx];
    //	_edges[idx+reductionoffset] = shm_points[shmidx+128];
    
}


__device__ inline int bin_dist(unsigned int* _src,
                               unsigned int* _ref) {
    
    
    int t = __popc( _src[0] ^ _ref[0] );
    t += __popc( _src[1] ^ _ref[1] );
    t += __popc( _src[2] ^ _ref[2] );
    t += __popc( _src[3] ^ _ref[3] );
    
    return t;
}

__device__ inline unsigned long L2_dist(uchar4* _src,
                                        uchar4* _ref,
                                        int _npts) {
    
    unsigned long sum = 0;
    for (int i=0; i<_npts; ++i) {
        sum += (_src[i].x-_ref[i].x) * (_src[i].x-_ref[i].x);
        sum += (_src[i].y-_ref[i].y) * (_src[i].y-_ref[i].y);
        sum += (_src[i].z-_ref[i].z) * (_src[i].z-_ref[i].z);
        sum += (_src[i].w-_ref[i].w) * (_src[i].w-_ref[i].w);
    }
    
    return sum;
}


template <int _area>
__global__ void best_L2_match(const pvcore::keypt* _srcpts,
                              pvcore::keypt* _destpts) {
    
    pvcore::keypt pt = _srcpts[blockDim.x*blockIdx.x + threadIdx.x];
    
    uchar4 src[4], ref[4];
    
    ref[0] = tex2D(tex_8u4_0, 4*pt.x,   pt.y);
    ref[1] = tex2D(tex_8u4_0, 4*pt.x+1, pt.y);
    ref[2] = tex2D(tex_8u4_0, 4*pt.x+2, pt.y);
    ref[3] = tex2D(tex_8u4_0, 4*pt.x+3, pt.y);
    
    pvcore::keypt destpt;
    destpt.x = pt.x;
    destpt.y = pt.y;
    
    unsigned long dist = 255*255*16+1;
#pragma unroll
    for (int y=-_area; y<=_area; ++y) {
        unsigned long t;
#pragma unroll
        for (int x=-_area; x<=_area; ++x) {
            src[0] = tex2D(tex_8u4_1, 4*(pt.x+x),   pt.y+y);
            src[1] = tex2D(tex_8u4_1, 4*(pt.x+x)+1, pt.y+y);
            src[2] = tex2D(tex_8u4_1, 4*(pt.x+x)+2, pt.y+y);
            src[3] = tex2D(tex_8u4_1, 4*(pt.x+x)+3, pt.y+y);
            t = L2_dist(src,ref,4);
            if (t < dist) {
                dist = t;
                destpt.x = pt.x+x;
                destpt.y = pt.y+y;
            }
        }
    }
    
    _destpts[blockDim.x*blockIdx.x + threadIdx.x] = destpt;
    
}


template <int _area>
__global__ void best_bin_match(const pvcore::keypt* _srcpts,
                               pvcore::keypt* _destpts) {
    
    pvcore::keypt pt = _srcpts[blockDim.x*blockIdx.x + threadIdx.x];
    
    unsigned int src[4], ref[4];
    
    ref[0] = tex2D(tex_32u_0, 4*pt.x,   pt.y);
    ref[1] = tex2D(tex_32u_0, 4*pt.x+1, pt.y);
    ref[2] = tex2D(tex_32u_0, 4*pt.x+2, pt.y);
    ref[3] = tex2D(tex_32u_0, 4*pt.x+3, pt.y);
    
    pvcore::keypt destpt;
    destpt.x = pt.x;
    destpt.y = pt.y;

    int dist = 129;
#pragma unroll
    for (int y=-_area; y<=_area; ++y) {
#pragma unroll
        for (int x=-_area; x<=_area; ++x) {
            src[0] = tex2D(tex_32u_1, 4*(pt.x+x),   pt.y+y);
            src[1] = tex2D(tex_32u_1, 4*(pt.x+x)+1, pt.y+y);
            src[2] = tex2D(tex_32u_1, 4*(pt.x+x)+2, pt.y+y);
            src[3] = tex2D(tex_32u_1, 4*(pt.x+x)+3, pt.y+y);
            int t = bin_dist(src,ref);
            if (t <= dist) {
                dist = t;
                destpt.x = pt.x+x;
                destpt.y = pt.y+y;
            }
        }
    }
    
    _destpts[blockDim.x*blockIdx.x + threadIdx.x] = destpt;
    
}



template <typename T>
__global__ void zero(T* _dest, int _pitch) {
    _dest[(blockDim.y*blockIdx.y+threadIdx.y)*_pitch + blockDim.x*blockIdx.x+threadIdx.x] = (T)0;
}


__global__ void zeroVel(pvcore::keypt_vel* _vel, int _pitch) {
    _vel[(blockDim.y*blockIdx.y+threadIdx.y)*_pitch + blockDim.x*blockIdx.x+threadIdx.x].x = 0.0f;
    _vel[(blockDim.y*blockIdx.y+threadIdx.y)*_pitch + blockDim.x*blockIdx.x+threadIdx.x].y = 0.0f;
}

__global__ void fill_features(pvcore::keypt* _features,
                              float* _dest,
                              int _nfeatures,
                              int _pitchd) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if( idx < _nfeatures ) {
        _dest[_features[idx].y*_pitchd + _features[idx].x] = 1.0f;
    }
    
}


__global__ void predict_points(pvcore::keypt* _srcpt, pvcore::keypt_vel* _srcvel, pvcore::keypt* _destpt, int _npts) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (idx < _npts) {
        _destpt[idx].x = _srcpt[idx].x + _srcvel[idx].x;
        _destpt[idx].y = _srcpt[idx].y + _srcvel[idx].y;
    }
    
}


__global__ void add_even(const float* _src,
                         float* _dest,
                         int _width,
                         int _height,
                         int _pitchd) {
    
    int y = (blockDim.y*blockIdx.y + threadIdx.y)*2;
    int x = (blockDim.x*blockIdx.x + threadIdx.x)*2;
    
    __shared__ float shm[18*18];
    
    if (y>0 && y<_height-1 && x>0 && x<_width-1) {
        float sum = 0.0f;
        sum += tex2D( tex_32f, x-1, y-1);
        sum += tex2D( tex_32f, x  , y-1);
        sum += tex2D( tex_32f, x+1, y-1);
        sum += tex2D( tex_32f, x-1, y  );
        sum += tex2D( tex_32f, x  , y  );
        sum += tex2D( tex_32f, x+1, y  );
        sum += tex2D( tex_32f, x-1, y+1);
        sum += tex2D( tex_32f, x  , y+1);
        sum += tex2D( tex_32f, x+1, y+1);
        
        if (sum == 0) {
            _dest[y*_pitchd+x] = _src[y*_pitchd+x];
        }
    }
    
}


__global__ void add_odd(const float* _src,
                         float* _dest,
                         int _width,
                         int _height,
                         int _pitchd) {
    
    int y = (blockDim.y*blockIdx.y + threadIdx.y)*2+1;
    int x = (blockDim.x*blockIdx.x + threadIdx.x)*2+1;
    
    __shared__ float shm[18*18];
    
    if (y<_height-1 && x<_width-1) {
        float sum = 0.0f;
        sum += tex2D( tex_32f, x-1, y-1);
        sum += tex2D( tex_32f, x  , y-1);
        sum += tex2D( tex_32f, x+1, y-1);
        sum += tex2D( tex_32f, x-1, y  );
        sum += tex2D( tex_32f, x  , y  );
        sum += tex2D( tex_32f, x+1, y  );
        sum += tex2D( tex_32f, x-1, y+1);
        sum += tex2D( tex_32f, x  , y+1);
        sum += tex2D( tex_32f, x+1, y+1);
        
        if (sum == 0) {
            _dest[y*_pitchd+x] = _src[y*_pitchd+x];
        }
    }
    
}


namespace pvcore {
    
    // The results from this method will differ. Consider implementing own
    template <typename T>
    cudaError_t __resizeGPU(const T* _src,
                            T* _dest,
                            unsigned int _width,
                            unsigned int _height,
                            unsigned int _pitchs,
                            unsigned int _pitchd,
                            double _scale) {
        
        NppiSize osz; osz.width = _width; osz.height = _height;
        NppiRect oroi; oroi.x = 0; oroi.y = 0; oroi.width = _width; oroi.height = _height;
        NppiRect droi; droi.x = 1; droi.y = 1; droi.width = _width*_scale; droi.height = _height*_scale;
        
        switch (sizeof(T)) {
            case 4:
                if (_pitchs/_width == 1) {
                    nppiResizeSqrPixel_32f_C1R((const Npp32f*)_src, osz, _pitchs*4, oroi, (Npp32f*)_dest, _pitchd*4, droi, _scale, _scale, 0, 0, NPPI_INTER_LINEAR);
                } else if (_pitchs/_width == 3) {
                    nppiResizeSqrPixel_32f_C3R((const Npp32f*)_src, osz, _pitchs*4, oroi, (Npp32f*)_dest, _pitchd*4, droi, _scale, _scale, 0, 0, NPPI_INTER_LINEAR);
                } else if (_pitchs/_width == 4) {
                    nppiResizeSqrPixel_32f_C4R((const Npp32f*)_src, osz, _pitchs*4, oroi, (Npp32f*)_dest, _pitchd*4, droi, _scale, _scale, 0, 0, NPPI_INTER_LINEAR);
                }
                break;
                
            case 1:
                if (_pitchs/_width == 1) {
                    nppiResizeSqrPixel_8u_C1R((const Npp8u*)_src, osz, _pitchs, oroi, (Npp8u*)_dest, _pitchd, droi, _scale, _scale, 0, 0, NPPI_INTER_LINEAR);
                } else if (_pitchs/_width == 3) {
                    nppiResizeSqrPixel_8u_C3R((const Npp8u*)_src, osz, _pitchs, oroi, (Npp8u*)_dest, _pitchd, droi, _scale, _scale, 0, 0, NPPI_INTER_LINEAR);
                } else if (_pitchs/_width == 4) {
                    nppiResizeSqrPixel_8u_C4R((const Npp8u*)_src, osz, _pitchs, oroi, (Npp8u*)_dest, _pitchd, droi, _scale, _scale, 0, 0, NPPI_INTER_LINEAR);
                }
                break;
                
            default:
                break;
        }
        
        return cudaSuccess;
    }
    
    
    // The results from this method will differ. Consider implementing own
    template <typename T>
    cudaError_t __filterGaussGPU(const T* _src,
                                 T* _dest,
                                 unsigned int _width,
                                 unsigned int _height,
                                 unsigned int _pitchs,
                                 unsigned int _pitchd,
                                 int _mask) {
        
        NppiSize osz; osz.width = _width; osz.height = _height;
        
        NppiMaskSize mask_size = NPP_MASK_SIZE_3_X_3;
        if (_mask == 5) {
            mask_size = NPP_MASK_SIZE_5_X_5;
        }
        
        switch (sizeof(T)) {
            case 4:
                if (_pitchs/_width == 1) {
                    nppiFilterGauss_32f_C1R((const Npp32f*)_src, _pitchs*4, (Npp32f*)_dest, _pitchd*4, osz, mask_size);
                } else if (_pitchs/_width == 3) {
                    nppiFilterGauss_32f_C3R((const Npp32f*)_src, _pitchs*4, (Npp32f*)_dest, _pitchd*4, osz, mask_size);
                } else if (_pitchs/_width == 4) {
                    nppiFilterGauss_32f_C4R((const Npp32f*)_src, _pitchs*4, (Npp32f*)_dest, _pitchd*4, osz, mask_size);
                }
                break;
                
            case 1:
                if (_pitchs/_width == 1) {
                    nppiFilterGauss_8u_C1R((const Npp8u*)_src, _pitchs, (Npp8u*)_dest, _pitchd, osz, mask_size);
                } else if (_pitchs/_width == 3) {
                    nppiFilterGauss_8u_C3R((const Npp8u*)_src, _pitchs, (Npp8u*)_dest, _pitchd, osz, mask_size);
                } else if (_pitchs/_width == 4) {
                    nppiFilterGauss_8u_C4R((const Npp8u*)_src, _pitchs, (Npp8u*)_dest, _pitchd, osz, mask_size);
                }
                break;
                
            default:
                break;
        }
        
        return cudaSuccess;
    }
    
    
    struct semiDenseKeypointStruct {
        static int nscales;
        static bool inited;
        
        // Filtered source images
        unsigned char* scale3;
        unsigned char* scale6;
        
        // Saliency image
        float* saliency;
        
        // Number of feature points in cuda grid
        unsigned char* ptcount;
        unsigned char* ptcount_h;
        
        // Accumulated feature points in the grid
        int* idxvector;
        int* idxvector_h;
        
        // Keypoints
        float* keypoints_img;
        keypt* keypoints;
        int nkeypoints;
        
        // Pitch for
        size_t fpitch, ipitch;
        size_t fpitch_h, ipitch_h;
        
        pvcore::keypt*  featurepts_h;
    };
    
    int semiDenseKeypointStruct::nscales = 1;
    bool semiDenseKeypointStruct::inited  = false;
    
    semiDenseKeypointStruct* gKeypoints = NULL;
    
    
    // Initializes device and host memory for semi-dense optical flow
    cudaError_t __initKeypointBuffer(int _width, int _height, int _nscales, dim3 _blockDim) {
        
        printf("initing keypoint buffer with %d scales\n",_nscales);
        
        if (semiDenseKeypointStruct::inited) {
            printf("Already inited\n");
            return cudaSuccess;
        }
        
        semiDenseKeypointStruct::nscales = _nscales;
        
        gKeypoints = new semiDenseKeypointStruct[_nscales];
        
        dim3 gridDim; gridDim.z = 1;
        
        for( int i=0; i<_nscales; ++i ) {
            int div = 1 << i;
            size_t ipitch, fpitch;
            cudaMallocPitch((void**)&gKeypoints[i].scale3, &ipitch, _width/div, _height/div);
            cudaMallocPitch((void**)&gKeypoints[i].scale6, &ipitch, _width/div, _height/div);
            
            cudaMallocPitch((void**)&gKeypoints[i].saliency,    &fpitch, 4*_width/div, _height/div);
            
            cudaMallocPitch((void**)&gKeypoints[i].keypoints_img, &fpitch, 4*_width/div, _height/div);
            cudaMalloc((void**)&gKeypoints[i].keypoints, 4*_width*_height/(div*div) );
            gKeypoints[i].nkeypoints = 0;
            
            
            gridDim.x = GLOBAL_SIZE( _width/div,  _blockDim.x );
            gridDim.y = GLOBAL_SIZE( _height/div, _blockDim.y );
            cudaMalloc((void**)&gKeypoints[i].ptcount, gridDim.x*gridDim.y);
            gKeypoints[i].ptcount_h = new unsigned char[gridDim.x*gridDim.y];
            
            cudaMalloc((void**)&gKeypoints[i].idxvector, (gridDim.x*gridDim.y+1)*sizeof(int));
            gKeypoints[i].idxvector_h = new int[gridDim.x*gridDim.y+1];
            
            gKeypoints[i].featurepts_h = new pvcore::keypt[_width/div*_height/div];
            
            gKeypoints[i].ipitch = ipitch;
            gKeypoints[i].fpitch = fpitch;
        }
        
        semiDenseKeypointStruct::inited = true;
        
        return cudaSuccess;
    }
    
    
    // Releases device and host memory for semi-dense optical flow
    void __freeKeypointBuffer() {
        for( int i=0; i<semiDenseKeypointStruct::nscales; ++i ) {
            cudaFree(gKeypoints[i].scale3);
            cudaFree(gKeypoints[i].scale6);
            
            cudaFree(gKeypoints[i].saliency);
            
            cudaFree(gKeypoints[i].ptcount);
            delete [] gKeypoints[i].ptcount_h;
            
            cudaFree(gKeypoints[i].idxvector);
            delete [] gKeypoints[i].idxvector_h;
            
            delete [] gKeypoints[i].featurepts_h;
            
        }
        semiDenseKeypointStruct::inited = false;
    }
    
    
    cudaError_t __initOpticalFlowStructGPU(opticalFlowStruct* _src,
                                           int _width,
                                           int _height,
                                           int _nscales) {
        
        //*_src = new opticalFlowStruct[_nscales];
        
        cudaError_t err = cudaSuccess;
        
        for( int i=0; i<_nscales; ++i ) {
            int div = 1 << i;
            // Init keypoint struct (Maximum number of keypoints is as many as there are pixels in the image)
            err = cudaMalloc((void**)&(_src[i].keypoints), sizeof(keypt)*_width/div*_height/div);
            err = cudaMalloc((void**)&(_src[i].keypoints_vel), sizeof(keypt_vel)*_width/div*_height/div);
            _src[i].nkeypoints = 0;
            err = cudaMallocPitch((void**)&(_src[i].keypoints_img),
                                  &(_src[i].keypoints_img_pitch),
                                  _width*sizeof(float)/div, _height/div);
            
            _src[i].keypoints_img_pitch /= sizeof(float);
            
            err = cudaMallocPitch((void**)&(_src[i].features), &(_src[i].features_pitch),
                            _width*sizeof(featurept)/div, _height/div);
            _src[i].features_pitch /= sizeof(featurept);
            
            dim3 blockDim; blockDim.x = blockDim.y = 16; blockDim.z = 1;
            dim3 gridDim; gridDim.z = 1;
            gridDim.x = GLOBAL_SIZE(_width/div,blockDim.x);
            gridDim.y = GLOBAL_SIZE(_height/div,blockDim.y);
            zeroVel<<<gridDim,blockDim>>>(_src[i].keypoints_vel,_width/div);
        }
        
        return err;
    }
    
    
    cudaError_t __initPredictionStructGPU(predictionStruct* _src,
                                          int _width,
                                          int _height,
                                          int _nscales) {
        //*_src = new predictionStruct[_nscales];
        
        for( int i=0; i<_nscales; ++i ) {
            int div = 1 << i;
            cudaMalloc((void**)&_src[i].keypoints,_width/div*_height/div*sizeof(keypt));
            _src[i].nkeypoints = 0;
            cudaMallocPitch((void**)&_src[i].features, &_src[i].features_pitch,
                            _width*sizeof(featurept)/div, _height/div);
            _src[i].features_pitch /= sizeof(featurept);
        }
        
        return cudaSuccess;
    }

    
    cudaError_t __matchGPU(opticalFlowStruct* _opticalFlow,
                           predictionStruct* _prediction,
                           int _width, int _height, dim3 _blockDim) {

        
        _blockDim.y = 1;
        _blockDim.x = 128;
        int nscales = semiDenseKeypointStruct::nscales;
        
        dim3 gridDim; gridDim.y = gridDim.z = 1;
        
        // Features
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);
        tex_8u4_0.addressMode[0] = cudaAddressModeClamp;
        tex_8u4_0.addressMode[1] = cudaAddressModeClamp;
        tex_8u4_0.filterMode = cudaFilterModePoint;
        tex_8u4_0.normalized = false;
        tex_8u4_1.addressMode[0] = cudaAddressModeClamp;
        tex_8u4_1.addressMode[1] = cudaAddressModeClamp;
        tex_8u4_1.filterMode = cudaFilterModePoint;
        tex_8u4_1.normalized = false;

//        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);
//        tex_32u_0.addressMode[0] = cudaAddressModeClamp;
//        tex_32u_0.addressMode[1] = cudaAddressModeClamp;
//        tex_32u_0.filterMode = cudaFilterModePoint;
//        tex_32u_0.normalized = false;
//        tex_32u_1.addressMode[0] = cudaAddressModeClamp;
//        tex_32u_1.addressMode[1] = cudaAddressModeClamp;
//        tex_32u_1.filterMode = cudaFilterModePoint;
//        tex_32u_1.normalized = false;
        
        for (int i=0; i<nscales; ++i) {
            int div = 1 << i;
            
            if (_opticalFlow[i].nkeypoints == 0) {
                continue;
            }
            gridDim.x = GLOBAL_SIZE(_opticalFlow[i].nkeypoints, _blockDim.x);
            
            cudaMemcpy( gKeypoints[i].featurepts_h, _prediction[i].keypoints,
                       sizeof(pvcore::keypt)*_opticalFlow[i].nkeypoints,cudaMemcpyDeviceToHost);
            
            printf("#npoints: %d\n",_opticalFlow[i].nkeypoints);
            
            for (int j=0; j<10; ++j) {
                printf("kp %d before: %d %d\n",j,gKeypoints[i].featurepts_h[j].x,gKeypoints[i].featurepts_h[j].y);
            }
            
            
            cudaBindTexture2D( 0, &tex_8u4_0, _prediction[i].features, &channelDesc,
                              _width/(4*div), _height/div,
                              _prediction[i].featuresBytesPerRow()/4);
            cudaBindTexture2D( 0, &tex_8u4_1, _prediction[i].features, &channelDesc,
                              _width/(4*div), _height/div,
                              _prediction[i].featuresBytesPerRow()/4);
            best_L2_match<3><<<gridDim,_blockDim>>>(_prediction[i].keypoints, _opticalFlow[i].keypoints);
            
            
//            cudaBindTexture2D( 0, &tex_32u_0, _opticalFlow[i].features, &channelDesc,
//                              _width/(4*div), _height/div,
//                              _prediction[i].featuresBytesPerRow()/4);
//            cudaBindTexture2D( 0, &tex_32u_1, _prediction[i].features, &channelDesc,
//                              _width/(4*div), _height/div,
//                              _prediction[i].featuresBytesPerRow()/4);
//            best_bin_match<3><<<gridDim,_blockDim>>>(_prediction[i].keypoints, _opticalFlow[i].keypoints);
            

            cudaMemcpy( gKeypoints[i].featurepts_h, _opticalFlow[i].keypoints,
                       sizeof(pvcore::keypt)*_opticalFlow[i].nkeypoints,cudaMemcpyDeviceToHost);
            for (int j=0; j<10; ++j) {
                printf("kp %d after:  %d %d\n",j,gKeypoints[i].featurepts_h[j].x,gKeypoints[i].featurepts_h[j].y);
            }
            
            // Now we have matched, so swap features in prediction and optical flow
            featurept* tmp = _opticalFlow[i].features;
            _opticalFlow[i].features = _prediction[i].features;
            _prediction[i].features = tmp;
            
        }
        
        return cudaSuccess;
    }

    
    cudaError_t __predictPointsGPU(opticalFlowStruct* _opticalFlow,
                                   predictionStruct* _prediction,
                                   int _width, int _height, dim3 _blockDim) {
        
        _blockDim.y = 1;
        _blockDim.x = 128;
        
        int nscales = semiDenseKeypointStruct::nscales;
        
        dim3 gridDim; gridDim.y = gridDim.z = 1;
        
        for (int i=0; i<nscales; ++i) {
            if (_opticalFlow[i].nkeypoints == 0) {
                continue;
            }
            gridDim.x = GLOBAL_SIZE(_opticalFlow[i].nkeypoints, _blockDim.x);
            predict_points<<<gridDim,_blockDim>>>(_opticalFlow[i].keypoints, _opticalFlow[i].keypoints_vel,
                                                  _prediction[i].keypoints,  _opticalFlow[i].nkeypoints);
            _prediction[i].nkeypoints = _opticalFlow[i].nkeypoints;
        }
     
        return cudaSuccess;
    }

    
    
    
    cudaError_t __generateImagePyramid(const unsigned char* _src,
                                       unsigned int _width,
                                       unsigned int _height,
                                       unsigned int _pitchs,
                                       unsigned int _nscales,
                                       dim3 _blockDim) {
        
        if (!semiDenseKeypointStruct::inited) {
            __initKeypointBuffer(_width,_height,_nscales,_blockDim);
        }
        
        int nscales = semiDenseKeypointStruct::nscales;
        
        // Resize and gauss filter to different levels
        const unsigned char* timg = _src;
        for (int i=1; i<nscales; ++i) {
            __resizeGPU(timg, gKeypoints[i].scale3, _width,_height,
                        gKeypoints[i-1].ipitch, gKeypoints[i].ipitch,0.5);
            timg = gKeypoints[i].scale3;
        }
        __filterGaussGPU(_src,gKeypoints[0].scale3, _width, _height, gKeypoints[0].ipitch,gKeypoints[0].ipitch,3);
        __filterGaussGPU(gKeypoints[0].scale3, gKeypoints[0].scale6, _width, _height,
                         gKeypoints[0].ipitch, gKeypoints[0].ipitch,3);
        for (int i=1; i<nscales; ++i) {
            int div = 1 << i;
            __filterGaussGPU(gKeypoints[i].scale3, gKeypoints[i].scale3, _width/div,_height/div,
                             gKeypoints[i].ipitch, gKeypoints[i].ipitch,3);
            __filterGaussGPU(gKeypoints[i].scale3, gKeypoints[i].scale6, _width/div,_height/div,
                             gKeypoints[i].ipitch, gKeypoints[i].ipitch,3);
        }
        
        return cudaSuccess;
    }
    
    
    
    cudaError_t __extractFeaturesGPU(predictionStruct* _prediction,
                                     int _width,
                                     int _height,
                                     dim3 _blockDim) {
        
        int nscales = semiDenseKeypointStruct::nscales;
        
        // Features
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);
        tex_32u.addressMode[0] = cudaAddressModeClamp;
        tex_32u.addressMode[1] = cudaAddressModeClamp;
        tex_32u.filterMode = cudaFilterModePoint;
        tex_32u.normalized = false;
        dim3 gridDim; gridDim.z = 1;
        for (int i=0; i<nscales; ++i) {
            int div = 1 << i;
            gridDim.x = GLOBAL_SIZE( _width/(div*4), _blockDim.x );
            gridDim.y = GLOBAL_SIZE( _height/div,  _blockDim.y );
            cudaBindTexture2D( 0, &tex_32u, gKeypoints[i].scale3, &channelDesc,
                              _width/(4*div), _height/div, gKeypoints[i].ipitch );
            weak_features_full<<<gridDim,_blockDim>>>((unsigned int*)_prediction[i].features,
                                                      _width/(div*4), _height/div, _prediction[i].featuresBytesPerRow()/4);
        }
        
        return cudaSuccess;
    }
    
    
    
    // Extracts keypoints semi-dense optical flow
    cudaError_t __extractKeypointsGPU(opticalFlowStruct* _opticalFlow,
                                      unsigned int _width,
                                      unsigned int _height,
                                      unsigned int _keypointType,
                                      dim3 _blockDim) {
        
        int nscales = semiDenseKeypointStruct::nscales;
        
        // Compute saliency images
        float threshold = (float)_keypointType*0.01f;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
        tex_8u.addressMode[0] = cudaAddressModeClamp;
        tex_8u.addressMode[1] = cudaAddressModeClamp;
        tex_8u.filterMode = cudaFilterModePoint;
        tex_8u.normalized = false;
        
        dim3 gridDim; gridDim.z = 1;
        for (int i=0; i<nscales; ++i) {
            int div = 1 << i;
            gridDim.x = GLOBAL_SIZE( _width/div, _blockDim.x );
            gridDim.y = GLOBAL_SIZE( _height/div, _blockDim.y );
            cudaBindTexture2D( 0, &tex_8u, gKeypoints[i].scale3, &channelDesc,
                              _width/div, _height/div, gKeypoints[i].ipitch );
            saliency_3<<< gridDim,_blockDim>>>(gKeypoints[i].saliency,
                                               _width/div, _height/div,
                                               gKeypoints[i].fpitch/4, threshold);
            
            cudaBindTexture2D( 0, &tex_8u, gKeypoints[i].scale6, &channelDesc,
                              _width/div, _height/div, gKeypoints[i].ipitch );
            saliency_6<<< gridDim,_blockDim>>>(gKeypoints[i].saliency,
                                               _width/div, _height/div,
                                               gKeypoints[i].fpitch/4, threshold);
        }
        
        
        // Non max suppression
        channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
        tex_32f.addressMode[0] = cudaAddressModeClamp;
        tex_32f.addressMode[1] = cudaAddressModeClamp;
        tex_32f.filterMode = cudaFilterModePoint;
        tex_32f.normalized = false;
        for (int i=0; i<nscales; ++i) {
            int div = 1 << i;
            gridDim.x = GLOBAL_SIZE( _width/div, _blockDim.x );
            gridDim.y = GLOBAL_SIZE( _height/div, _blockDim.y );
            cudaBindTexture2D( 0, &tex_32f, gKeypoints[i].saliency, &channelDesc,
                              _width/div, _height/div, gKeypoints[i].fpitch );
            nonmax_sup<<< gridDim,_blockDim>>>(gKeypoints[i].keypoints_img, _width/div, _height/div,
                                               gKeypoints[i].fpitch);
            
            // Add new keypoints to optical flow
            gridDim.x = GLOBAL_SIZE( _width/(2*div), _blockDim.x );
            gridDim.y = GLOBAL_SIZE( _height/(2*div), _blockDim.y );
//            add_even<<< gridDim, _blockDim>>>();
//            add_odd<<< gridDim, _blockDim>>>();
        }
        
        for (int i=0; i<nscales; ++i) {
            int div = 1 << i;
            _blockDim.y = 16;
            gridDim.x = GLOBAL_SIZE( _width/div,  _blockDim.x );
            gridDim.y = GLOBAL_SIZE( _height/div, _blockDim.y );
            _blockDim.y = 8;
            
            // Count points
            count_points<<<gridDim,_blockDim>>>(_opticalFlow[i].keypoints_img, gKeypoints[i].ptcount,
                                                0.3f, _opticalFlow[i].keypoints_img_pitch);
            // Accumulate points
            cudaMemcpy(gKeypoints[i].ptcount_h, gKeypoints[i].ptcount,
                       gridDim.x*gridDim.y,cudaMemcpyDeviceToHost);
            _opticalFlow[i].nkeypoints = 0;
            gKeypoints[i].idxvector_h[0] = 0;
            for( int j=0; j<gridDim.x*gridDim.y; ++j ) {
                _opticalFlow[i].nkeypoints += gKeypoints[i].ptcount_h[j];
                gKeypoints[i].idxvector_h[j+1] = _opticalFlow[i].nkeypoints;
            }
            cudaMemcpy(gKeypoints[i].idxvector, gKeypoints[i].idxvector_h,
                       (gridDim.x*gridDim.y+1)*sizeof(int), cudaMemcpyHostToDevice );
            
            // Create points
            createPointList<<<gridDim,_blockDim>>>(_opticalFlow[i].keypoints_img, gKeypoints[i].idxvector,
                                                   _opticalFlow[i].keypoints, _opticalFlow[i].keypoints_img_pitch);
            cudaMemcpy( gKeypoints[i].featurepts_h, _opticalFlow[i].keypoints,
                       sizeof(pvcore::keypt)*_opticalFlow[i].nkeypoints,cudaMemcpyDeviceToHost);
            
        }
        
        return cudaSuccess;
    }
    
    
} // namespace pvcore

