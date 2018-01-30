//
//  core_common.h
//  
//
//  Created by Niklas Bergström on 2013-05-09.
//
//

#ifndef _CORECOMMON_H_
#define _CORECOMMON_H_

#include <cstdio>
#include <vector>
#include <string>
#include <cmath>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCKTYPE dim3
#elif !defined(BLOCKTYPE)
typedef struct dim3 { int x, y, z; } dim3;
#define BLOCKTYPE dim3
#endif

#ifdef USE_OPENCL
#include "pvcore/cl.h"
#endif // USE_OPENCL

/**
 A macro that returns the least integer so
 that \f$y*z \leq x \f$. Used for computing the global worksizes
 in relation to the local worksize and work size
 */
#ifdef USE_CUDA
#define GLOBAL_SIZE(total_size,workgroup_size) ( (total_size+(workgroup_size-1))/(workgroup_size) )
#else
#define GLOBAL_SIZE(total_size,workgroup_size) ( workgroup_size*((total_size+(workgroup_size-1))/(workgroup_size)) )
#endif

// Determining row interval to be processed
#define GET_START(THREAD,BLOCKSIZE)  ((int)ceil((THREAD)*(BLOCKSIZE)))
#define GET_STOP(THREAD,BLOCKSIZE)   ((int)(floor((THREAD)*(BLOCKSIZE)) == (THREAD)*(BLOCKSIZE) ? (THREAD)*(BLOCKSIZE) : (THREAD)*(BLOCKSIZE)+1))


// Set alignment
#ifdef USE_AVX2
#define VEC_ALIGN 32
#else
#define VEC_ALIGN 16
#endif


// Marcros for
#if defined(__APPLE__) || defined(__unix__)
#define ALIGNED_BUFFER(TYPE,NAME,SIZE) TYPE NAME[SIZE] __attribute__((aligned(VEC_ALIGN)))
#elif defined(_WIN32)
#define ALIGNED_BUFFER(TYPE,NAME,SIZE) __declspec(align(VEC_ALIGN)) TYPE NAME[SIZE]
#endif






/*void adjustNumberOfBlocks( const dim3& _blockDim,
                          dim3& _gridDim,
                          const unsigned int* _size ) {
 
    _gridDim.x = GLOBAL_SIZE( _size[0], _blockDim.x );
    _gridDim.y = GLOBAL_SIZE( _size[1], _blockDim.y );
    
}

void adjustNumberOfBlocksFlat( const dim3& _blockDim,
                              dim3& _gridDim,
                              const unsigned int* _size ) {
    
    _gridDim.x = GLOBAL_SIZE( _size[0]*_size[1], _blockDim.x );
    _gridDim.y = 1;
    _gridDim.z = 1;
    
}*/

#endif // _CORECOMMON_H_
