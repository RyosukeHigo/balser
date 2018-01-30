#ifndef _CUDAFUNCTIONS_H_
#define _CUDAFUNCTIONS_H_

__device__ inline void transpose(unsigned int _src0,
                                 unsigned int _src1,
                                 unsigned int _src2,
                                 unsigned int _src3,
                                 unsigned int* _dest0,
                                 unsigned int* _dest1,
                                 unsigned int* _dest2,
                                 unsigned int* _dest3) {
    
    
    unsigned int t0 = __byte_perm(_src0, _src1, 0x5140 );
    unsigned int t1 = __byte_perm(_src0, _src1, 0x7362 );
    unsigned int t2 = __byte_perm(_src2, _src3, 0x5140 );
    unsigned int t3 = __byte_perm(_src2, _src3, 0x7362 );
    
    *_dest0 = __byte_perm(t0, t2, 0x5410 );
    *_dest1 = __byte_perm(t0, t2, 0x7632 );
    *_dest2 = __byte_perm(t1, t3, 0x5410 );
    *_dest3 = __byte_perm(t1, t3, 0x7632 );
    
}



__device__ inline void transpose(unsigned int* _src0,
                                 unsigned int* _src1,
                                 unsigned int* _src2,
                                 unsigned int* _src3) {
    
    
    unsigned int t0 = __byte_perm(*_src0, *_src1, 0x5140 );
    unsigned int t1 = __byte_perm(*_src0, *_src1, 0x7362 );
    unsigned int t2 = __byte_perm(*_src2, *_src3, 0x5140 );
    unsigned int t3 = __byte_perm(*_src2, *_src3, 0x7362 );
    
    *_src0 = __byte_perm(t0, t2, 0x5410 );
    *_src1 = __byte_perm(t0, t2, 0x7632 );
    *_src2 = __byte_perm(t1, t3, 0x5410 );
    *_src3 = __byte_perm(t1, t3, 0x7632 );
    
}


/**
 *  \brief Helper for bitonic sort algorithm
 *  
 *  \param _values Pointer to the value array that is being sorted
 *  \param _indices Pointer to the corresponding index array
 *  \param _idx Index of the element currently sorted
 *  \param _offset Offset to the index. Which are we comparing with now
 *  \param _sortDirection The sort direction
 */
__device__ inline void atomicSort(float* _values,
								  unsigned char* _indices,
								  int _idx,
								  int _offset,
								  int _sortDirection) {
    
	float tvalue;
	unsigned char tidx;
    
	// int offsetadjust;
	if( _values[_idx+_sortDirection] > _values[_idx+(_offset-_sortDirection)] ) {
		tvalue = _values[_idx+_sortDirection];
		_values[_idx+_sortDirection] = _values[_idx+(_offset-_sortDirection)];
		_values[_idx+(_offset-_sortDirection)] = tvalue;
		tidx = _indices[_idx+_sortDirection];
		_indices[_idx+_sortDirection] = _indices[_idx+(_offset-_sortDirection)];
		_indices[_idx+(_offset-_sortDirection)] = tidx;
	}
    
}



#endif // _CUDAFUNCTIONS_H_