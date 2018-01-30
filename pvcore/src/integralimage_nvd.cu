#include "pvcore/integralimage.h"

#include <cstdio>

// Texture reference for 2D float texture
__global__ void _integralimage_vert_pass_1( const unsigned char* _src, float* _dest,
                                           int _srcpitch, int _destpitch,
                                           int _width, int _height,
                                           int _shmpitch) {
    
    extern __shared__ float shm_f32[];
    
    int ybase = 2*blockDim.y*blockIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
        
    // Load pairwise summed rows into shared mem
    int idx = ybase + 2*threadIdx.y*_srcpitch + x;

    // First step sum (0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15)
    int shmidx = 2*threadIdx.y*_shmpitch + threadIdx.x;
    shm_f32[shmidx] = (float)_src[idx];
    shm_f32[shmidx+_shmpitch] = shm_f32[shmidx] + _src[idx+_srcpitch];
    
    __syncthreads();

    // Second step sum (1,2 1,3 5,6 5,7 9,10 9,11 13,14 13,15)
    shmidx = ((0xFFFFFFFE & (int)threadIdx.y)*4+2+(0x00000001&(int)threadIdx.y))*_shmpitch + threadIdx.x;
    shm_f32[shmidx] += shm_f32[shmidx - (0x00000001&(int)threadIdx.y)*_shmpitch];
    
    __syncthreads();
    // Third step sum (3,4 3,5 3,6 3,7 11,12 11,13 11,14 11,15)
    shmidx = ((0xFFFFFFFC & (int)threadIdx.y)*8+4+(0x00000003&(int)threadIdx.y))*_shmpitch + threadIdx.x;
    shm_f32[shmidx] += shm_f32[shmidx - (0x00000003&(int)threadIdx.y)*_shmpitch];

    __syncthreads();

    // Fourth step sum (7,8 7,9 7,10 7,11 7,12 7,13 7,14 7,15)
    shmidx = 8+(0x00000007&(int)threadIdx.y)*_shmpitch + threadIdx.x;
    shm_f32[shmidx] += shm_f32[shmidx - (0x00000007&(int)threadIdx.y)*_shmpitch];

    __syncthreads();
    
    // At this point blockDim.x x 16 pixels have been added vertically now let's write them back in _dest
    shmidx = 2*threadIdx.y*_shmpitch + threadIdx.x;
    _dest[idx] = shm_f32[shmidx];
    _dest[_srcpitch] = shm_f32[shmidx+_shmpitch];
}


__global__ void _integralimage_vert_pass_2(float* _dest,
                                           int _destpitch,
                                           int _width, int _height,
                                           int _ymask, int _offset) {
    
    int addidxy = 2*_offset*(_ymask & (int)blockIdx.y) + _offset - 1;

    int addidx = (blockDim.y*addidxy)*_destpitch;
    
    // Determine which block should be accessed and add the row in that block
    int blocky = 2*_offset*(_ymask & (int)blockIdx.y) + _offset + ((0xFFFFFFFF ^ _ymask) & blockIdx.y);
    int y = blockDim.y*blocky+threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;

    if( y < _height ) {
        _dest[y*_destpitch+x] += _dest[addidx+x];
    }
    
}


namespace pvcore {
    
    void __integralImageCUDA(const unsigned char* _src,
                             float* _dest,
                             int _srcpitch,
                             int _destpitch,
                             int _width,
                             int _height,
                             dim3& _blockDim
                             ) {
        
        _blockDim.x = 32; _blockDim.y = 8; _blockDim.z = 1;
        dim3 gridDim;
        gridDim.x = GLOBAL_SIZE(_width, _blockDim.x);
        gridDim.y = GLOBAL_SIZE(_height, _blockDim.y);
        
        size_t shmsize = (_blockDim.x+1)*_blockDim.y*sizeof(float);
        
        _integralimage_vert_pass_1<<<gridDim,_blockDim,shmsize>>>(_src, _dest,
                                                                  _srcpitch, _destpitch,
                                                                  _width,_height,
                                                                  _blockDim.x+1);
        
        int ymask = static_cast<int>(0xFFFFFFFF);
        int offset = 1;
        
        _integralimage_vert_pass_2<<<gridDim,_blockDim>>>(_dest, _destpitch,
                                                          _width, _height,
                                                          ymask, offset);
        
    }
    
} // namespace pvcore