#include "pvcore/imagetransform.h"

#include "pvcore/cuda_functions.h"

#include "pvcore/common.h"

#include <cstdio>

// Texture reference for 2D float texture
texture<uchar4, 2, cudaReadModeElementType> tex_u8_4;

#define M_PI 3.1415926

// Constant float holding cosine and sine values
__constant__ float cossin_d[720];
__constant__ short maxr_d[360];


__global__ void _fliph( const unsigned char* _src, unsigned char* _dest,
                       unsigned int _width, unsigned int _height,
                       unsigned int _channels, unsigned int _pitch ) {
    
    
}



__global__ void _cart2pol(uchar4* _dest,
                          float _xc, float _yc,
                          float _widthi,
                          float _heighti,
                          unsigned int _theta,
                          unsigned int _pitchd,
                          unsigned int _nChannels) {
    
    // Which pixels in the polar image should be filled
    unsigned int r = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int t = blockDim.y*blockIdx.y + threadIdx.y;
    
    float x = (_xc + r*cossin_d[t])*_widthi;
    float y = (_yc + r*cossin_d[t+_theta])*_heighti;
    
    if( t < _theta && r < maxr_d[t] ) {
        _dest[t*_pitchd + r] = tex2D( tex_u8_4, x, y );
    }
    
    
}



__global__ void transpose(const unsigned int* _src,
                          unsigned int* _dest,
                          unsigned int _width,
                          unsigned int _height,
                          unsigned int _pitchs,
                          unsigned int _pitchd) {
    
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    
    
    unsigned int s0 = _src[4*y*_pitchs+x];
    unsigned int s1 = _src[(4*y+1)*_pitchs+x];
    unsigned int s2 = _src[(4*y+2)*_pitchs+x];
    unsigned int s3 = _src[(4*y+3)*_pitchs+x];
    
    transpose(&s0,&s1,&s2,&s3);
    
    _dest[4*x*_pitchd+y] = s0;
    _dest[(4*x+1)*_pitchd+y] = s1;
    _dest[(4*x+2)*_pitchd+y] = s2;
    _dest[(4*x+3)*_pitchd+y] = s3;
    
}



namespace pvcore {
    
    
    
    void __transposeGPU(const unsigned char* _src,
                        unsigned char* _dest,
                        unsigned int _width,
                        unsigned int _height,
                        unsigned int _pitchs,
                        unsigned int _pitchd,
                        dim3& _blockDim) {
        
        dim3 gridDim;
        gridDim.z = 1;
        
        gridDim.x = GLOBAL_SIZE( _width/4, _blockDim.x );
        gridDim.y = GLOBAL_SIZE( _height/4, _blockDim.y );
        
        transpose<<<gridDim, _blockDim >>>( (const unsigned int*)_src, (unsigned int*)_dest,
                                           _width/4, _height, _pitchs/4, _pitchd/4 );
        
    }
    
    
    
    void __cart2polGPU(const unsigned char* _src,
                       unsigned char* _dest,
                       short* _maxr,
                       float _xc, float _yc,
                       unsigned int _width,
                       unsigned int _height,
                       unsigned int _pitchs,
                       unsigned int _pitchd,
                       unsigned int _theta,
                       unsigned int _nChannels,
                       dim3& _blockDim
                       ) {
        
        static bool first = true;
        
        static float cossin[720];

        if( first ) {
            
            for( int i=0; i<360; ++i ) {
                cossin[i] = cos(i*2*M_PI/360.0f);
                cossin[i+360] = sin(i*2*M_PI/360.0f);
            }
            
            cudaMemcpyToSymbol( cossin_d, cossin, 720*sizeof(float), 0, cudaMemcpyHostToDevice );

            // Compute max radius
            for( int i=0; i<_theta/4; ++i ) {
                float cost = cossin[i];
                float sint = cossin[i+_theta];
                
                if( sint == 0 ) {
                    _maxr[i] = _width-_xc;
                } else {
                    _maxr[i] = min((_width-_xc)/cost,(_height-_yc)/sint);
                }
            }
            for( int i=_theta/4; i<_theta/2; ++i ) {
                float cost = cossin[i];
                float sint = cossin[i+_theta];
                
                if( fabs(cost) < 0.00001 ) {
                    _maxr[i] = _height-_yc;
                } else {
                    _maxr[i] = min(-_xc/cost,(_height-_yc)/sint);
                }
            }
            for( int i=_theta/2; i<3*_theta/4; ++i ) {
                float cost = cossin[i];
                float sint = cossin[i+_theta];
                
                if( fabs(sint) < 0.00001 ) {
                    _maxr[i] = _xc;
                } else {
                    _maxr[i] = min(-_xc/cost,-_yc/sint);
                }
            }
            for( int i=3*_theta/4; i<_theta; ++i ) {
                float cost = cossin[i];
                float sint = cossin[i+_theta];
                
                if( fabs(cost) < 0.00001 ) {
                    _maxr[i] = _yc;
                } else {
                    _maxr[i] = min((_width-_xc)/cost,-_yc/sint);
                }
            }
            cudaMemcpyToSymbol( maxr_d, _maxr, 360*sizeof(short), 0, cudaMemcpyHostToDevice );
            
            first = false;
        }
        // Work sizes
        dim3 gridDim; gridDim.z = 1;
        
        
        
        
        unsigned int maxx = max(abs(_width-_xc),abs(_xc-_width));
        unsigned int maxy = max(abs(_height-_yc),abs(_yc-_height));
        unsigned int maxr = sqrt(maxx*maxx + maxy*maxy);
        
        gridDim.x = GLOBAL_SIZE( maxr, _blockDim.x );
        gridDim.y = GLOBAL_SIZE( _theta, _blockDim.y );
        
        
        // Texture setup
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);
        
        tex_u8_4.addressMode[0] = cudaAddressModeBorder;
        tex_u8_4.addressMode[1] = cudaAddressModeBorder;
        // cudaFilterModeLinear only works for floating point
        tex_u8_4.filterMode = cudaFilterModePoint;
        tex_u8_4.normalized = true;
        
        cudaBindTexture2D( 0, tex_u8_4, _src, channelDesc, _width, _height, _pitchs );
        
        // Kernel
        _cart2pol<<<gridDim,_blockDim>>>((uchar4*)_dest,_xc, _yc, 1.0f/(float)_width,1.0f/(float)_height,_theta,_pitchd/4, _nChannels);
        
        cudaUnbindTexture( tex_u8_4 );
        
    }
    
    
    void __flipHorizontalGPU(const unsigned char* _src,
                             unsigned char* _dest,
                             unsigned int _width,
                             unsigned int _height,
                             unsigned int _pitch,
                             unsigned int _nChannels,
                             dim3& _blockDim
                             ) {
        
        dim3 gridDim;
        gridDim.z = 1;
        
        gridDim.x = GLOBAL_SIZE( _width/2, _blockDim.x );
        gridDim.y = GLOBAL_SIZE( _height, _blockDim.y );
        
        size_t shmSize = 6*_blockDim.x*sizeof(uchar4);
        
        
        _fliph<<< gridDim, _blockDim >>>( _src, _dest, _width, _height, _nChannels, _pitch );
        
        
    }
    
    void __flipVerticalGPU(const unsigned char* _src,
                           unsigned char* _dest,
                           unsigned int _width,
                           unsigned int _height,
                           unsigned int _nChannels,
                           unsigned int _pitch,
                           dim3& _blockSize
                           ) {
        
    }
    
} // namespace pvcore