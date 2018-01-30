#include <mutex>


void _create_histogram_polar_helper(const unsigned char* _src, const short* _mask,
									const short* _maxr,
									unsigned int _width,
									unsigned int _pitch, unsigned int _srcchannels,
									unsigned int _histdimx,
									unsigned int _histdimy,
									unsigned int _histdimz,
									float* _hist,
									unsigned int _histchannels,
									float _invtheta,
									unsigned int _start, unsigned int _stop,
									const short* _left, const short* _right,
									unsigned int _threadId=0) {
	
	// Select histogram for current thread
	float* hist = _hist + _histdimx*_histdimy*_histdimz*_threadId;
	
	float idimx = (float)_histdimx/256.0f;
	float idimy = (float)_histdimy/256.0f;
	float idimz = (float)_histdimz/256.0f;
	
	switch( _histchannels ) {
		case 1:
#ifdef USE_SSE
			
#else
			for( unsigned int i=_start; i<_stop; ++i ) {
				for( unsigned int j=_left[i]; j<_right[i]; ++j ) {
					float c1 = (float)_src[(i*_pitch + _srcchannels*j)];
					unsigned int idx1 = std::floor(c1*idimx);
					
					hist[idx1] += j*_invtheta;
				}
				/*if( _mask->uncertainty() == 0 ) {
				 continue;
				 }
				 float factor = 1.0f/_mask->uncertainty();
				 for( ; j<mdata[i]+_mask->uncertainty(); ++j ) {
				 unsigned char c1 = data[nChannels*(i*width+j)+0];
				 unsigned int idx1 = std::floor((float)(c1)*idimx);
				 
				 hist[idx1] += (1.0f-(j-mdata[i]))*factor*i*invtheta;
				 }*/
			}
#endif
			break;
			
		case 2:
#ifdef USE_SSE
		{
			__m128i dimmul = _mm_set_epi16(_histdimy, _histdimx, _histdimy, _histdimx, _histdimy, _histdimx, _histdimy, _histdimx);
			__m128i adjustv = _mm_set_epi16(_histdimx, 1, _histdimx, 1, _histdimx, 1, _histdimx, 1);
			
			unsigned char __buff[16+15];
			size_t align = ((size_t)__buff) % 16;
			unsigned long long *buff = (unsigned long long*)(__buff+align);
			
			buff[0] = 0x8004800380018000ull; buff[1] = 0x800a800980078006ull;
			const __m128i shuffle = _mm_load_si128((const __m128i*)buff);
			
			for( unsigned int i =_start; i<_stop; ++i ) {
				unsigned int j =_left[i]*_srcchannels;
				unsigned int r2end = _right[i]*_srcchannels;
				for( ; j<r2end; j+=4*_srcchannels ) {
					
					unsigned int idx = (i*_pitch + j);
					__m128i c123u8 = _mm_loadu_si128((const __m128i*)(_src+idx));
					
					__m128i c123s16 = _mm_shuffle_epi8(c123u8, shuffle);
					
					__m128i idx123s16 = _mm_srli_epi16(_mm_mullo_epi16(c123s16, dimmul),8);
					
					__m128i idxadjusts16 = _mm_mullo_epi16(idx123s16, adjustv);
					
					__m128i idxsum1u16 = _mm_add_epi16(idxadjusts16, _mm_srli_epi32(idxadjusts16, 16));
					
					hist[_mm_extract_epi16(idxsum1u16, 0)] += j*_invtheta;
					
					if( j + _srcchannels < r2end ) {
						hist[_mm_extract_epi16(idxsum1u16, 2)] += (j+_srcchannels)*_invtheta;
					}
					if( j + 2*_srcchannels < r2end ) {
						hist[_mm_extract_epi16(idxsum1u16, 4)] += (j+2*_srcchannels)*_invtheta;
					}
					if( j + 3*_srcchannels < r2end ) {
						hist[_mm_extract_epi16(idxsum1u16, 6)] += (j+3*_srcchannels)*_invtheta;
					}
					
				}
			}
		}
#else
		{
			for( unsigned int i=_start; i<_stop; ++i ) {
				int j=_left[i]*_srcchannels;
				int rightend = _right[i]*_srcchannels;
				for( ; j<rightend; j+=_srcchannels ) {
					unsigned int idx = (i*_pitch + j);
					short idx1s = ((short)_src[idx+0]*_histdimx >> 8);
					short idx2s = ((short)_src[idx+1]*_histdimy >> 8);
					
					hist[idx2s*_histdimx + idx1s] += j*_invtheta;
				}
				/*if( _mask->uncertainty() == 0 ) {
				 continue;
				 }
				 float factor = 1.0f/_mask->uncertainty();
				 for( ; j<mdata[i]+_mask->uncertainty(); ++j ) {
				 unsigned char c1 = data[3*(i*width+j)+0];
				 unsigned char c2 = data[3*(i*width+j)+1];
				 unsigned int idx1 = std::floor((float)(c1)*idimx);
				 unsigned int idx2 = std::floor((float)(c2)*idimy);
				 
				 hist[idx2*dimx + idx1] += (1.0f-(j-mdata[i]))*factor*i*invtheta;
				 }*/
			}
		}
#endif
			
			break;
			
		case 3:
#ifdef USE_SSE
		{
			__m128i dimmul = _mm_set_epi16(0, 0, _histdimz, _histdimy, _histdimx, _histdimz, _histdimy, _histdimx);
			__m128i adjustv = _mm_set_epi16(0, 0, _histdimy*_histdimx, _histdimx, 1, _histdimy*_histdimx, _histdimx, 1);
			__m128i zeros = _mm_set1_epi16(0);
			for( unsigned int i=_start; i<_stop; ++i ) {
				unsigned int j=_left[i]*_srcchannels;
				unsigned int r2end = _right[i]*_srcchannels;
				for( ; j<r2end; j+=2*_srcchannels ) {
					
					unsigned int idx = (i*_pitch + j);
					__m128i c123u8 = _mm_loadu_si128((const __m128i*)(_src+idx));
					__m128i c123s16 = _mm_unpacklo_epi8(c123u8, zeros);
					
					__m128i idx123s16 = _mm_srli_epi16(_mm_mullo_epi16(c123s16, dimmul),8);
					
					__m128i idxadjusts16 = _mm_mullo_epi16(idx123s16, adjustv);
					
					__m128i idxsum1u16 = _mm_add_epi16(idxadjusts16, _mm_srli_epi32(idxadjusts16, 16));
					__m128i idxsum2u16 = _mm_add_epi16(idxsum1u16, _mm_srli_epi64(idxsum1u16, 32));
					
					
					int idx0 = _mm_extract_epi16(idxsum2u16, 0);
					
					hist[idx0] += j*_invtheta;
					
					if( j+_srcchannels < r2end ) {
						hist[_mm_extract_epi16(idxsum2u16, 4)] += (j+_srcchannels)*_invtheta;
					}
					
				}
			}
		}
#else
		{
			for( unsigned int i=_start; i<_stop; ++i ) {
				int j=_left[i]*_srcchannels;
				int r2end = _right[i]*_srcchannels;
				for( ; j<r2end; j+=_srcchannels ) {
					unsigned int idx = (i*_pitch + j);
					float c1 = (float)_src[idx+0];
					float c2 = (float)_src[idx+1];
					float c3 = (float)_src[idx+2];
					
					unsigned int idx1 = std::floor(c1*idimx);
					unsigned int idx2 = std::floor(c2*idimy);
					unsigned int idx3 = std::floor(c3*idimz);
					
					hist[idx3*_histdimy*_histdimx + idx2*_histdimx + idx1] += j*_invtheta;
				}
			}
		}
#endif
			break;
			
	}
}


class CREATE_HISTOGRAM_POLAR_TBB {
	
	const unsigned char* src;
	const short* mask;
	const short* maxr;
	bool leftOf;
	unsigned int width;
	unsigned int height;
	unsigned int pitch;
	unsigned int srcchannels;
	unsigned int dimx;
	unsigned int dimy;
	unsigned int dimz;
	float* hist;
	unsigned int threads;
	unsigned int histchannels;
	float invtheta;
	
public:
	void operator()( const tbb::blocked_range<size_t>& r ) const {
		
		
		// Sort out indices
		unsigned int height_p_thread = (height/threads);
		
		unsigned int start = (unsigned int)r.begin()*height_p_thread;
		unsigned int stop  = (unsigned int)( (threads == r.end()) ? height : r.end()*height_p_thread );
		
//		std::cout << "start: " << start << " stop: " << stop << std::endl;
		
		if (leftOf) {
			short* zeros = new short[height];
			memset(zeros,0,height*sizeof(short));
			
			_create_histogram_polar_helper(src, mask,
										   maxr,
										   width,
										   pitch, srcchannels,
										   dimx,
										   dimy,
										   dimz,
										   hist,
										   histchannels,
										   invtheta,
										   start, stop,
										   zeros, mask,
										   (unsigned int)r.begin() );

			delete [] zeros;
		} else {
			_create_histogram_polar_helper(src, mask,
										   maxr,
										   width,
										   pitch, srcchannels,
										   dimx,
										   dimy,
										   dimz,
										   hist,
										   histchannels,
										   invtheta,
										   start, stop,
										   mask, maxr,
										   (unsigned int)r.begin() );
		}
		
	}
	
	CREATE_HISTOGRAM_POLAR_TBB(const unsigned char* _src, const short* _mask,
							   const short* _maxr, bool _leftOf,
							   unsigned int _width, unsigned int _height,
							   unsigned int _pitch, unsigned int _srcchannels,
							   unsigned int _histdimx,
							   unsigned int _histdimy,
							   unsigned int _histdimz,
							   float* _hist, unsigned int _threads,
							   unsigned int _histchannels, float _invtheta) :
	
	src(_src), mask(_mask), maxr(_maxr), leftOf(_leftOf), width(_width), height(_height),
	pitch(_pitch), srcchannels(_srcchannels),
	dimx(_histdimx), dimy(_histdimy), dimz(_histdimz), hist(_hist), threads(_threads), histchannels(_histchannels), invtheta(_invtheta) {
	}
	
};


class CREATE_HISTOGRAM_POLAR_TBB2 {
	
	const unsigned char* src;
	const short* mask;
	const short* maxr;
	bool leftOf;
	unsigned int width;
	unsigned int height;
	unsigned int pitch;
	unsigned int srcchannels;
	unsigned int dimx;
	unsigned int dimy;
	unsigned int dimz;
	float* hist;
	unsigned int threads;
	unsigned int histchannels;
	float invtheta;
	
public:
	void operator()( const tbb::blocked_range<size_t>& r ) const {
		
		
		// Sort out indices
		unsigned int height_p_thread = (height/threads);
		
		unsigned int start = (unsigned int)r.begin()*height_p_thread;
		unsigned int stop  = (unsigned int)( (threads == r.end()) ? height : r.end()*height_p_thread );
		
		if (leftOf) {
			short* zeros = new short[height];
			memset(zeros,0,height*sizeof(short));
			
			_create_histogram(start, stop,
							  zeros, mask,
							  (unsigned int)r.begin() );
			delete [] zeros;
		} else {
			_create_histogram(start, stop,
							  mask, maxr,
							  (unsigned int)r.begin() );
		}
		
	}
	
	CREATE_HISTOGRAM_POLAR_TBB2(const unsigned char* _src, const short* _mask,
							   const short* _maxr, bool _leftOf,
							   unsigned int _width, unsigned int _height,
							   unsigned int _pitch, unsigned int _srcchannels,
							   unsigned int _histdimx,
							   unsigned int _histdimy,
							   unsigned int _histdimz,
							   float* _hist, unsigned int _threads,
							   unsigned int _histchannels, float _invtheta) :
	
	src(_src), mask(_mask), maxr(_maxr), leftOf(_leftOf), width(_width), height(_height),
	pitch(_pitch), srcchannels(_srcchannels),
	dimx(_histdimx), dimy(_histdimy), dimz(_histdimz), hist(_hist), threads(_threads), histchannels(_histchannels), invtheta(_invtheta) {
	}
	
	
private:
	inline void _create_histogram(unsigned int _h1, unsigned int _h2,
								  const short* _r1, const short* _r2,
								  unsigned int _threadId) const {
		
		// Select histogram for current thread
		float* _hist = hist + dimx*dimy*dimz*_threadId;
		
		float idimx = (float)dimx/256.0f;
		float idimy = (float)dimy/256.0f;
		float idimz = (float)dimz/256.0f;
		switch( histchannels ) {
			case 1:
				for( unsigned int i=_h1; i<_h2; ++i ) {
					for( unsigned int j=(unsigned int)_r1[i]; j<(unsigned int)_r2[i]; ++j ) {
						float c1 = (float)src[(i*pitch + srcchannels*j)];
						unsigned int idx1 = (unsigned int)std::floor(c1*idimx);
						
						_hist[idx1] += j*invtheta;
					}
					/*if( _mask->uncertainty() == 0 ) {
					 continue;
					 }
					 float factor = 1.0f/_mask->uncertainty();
					 for( ; j<mdata[i]+_mask->uncertainty(); ++j ) {
					 unsigned char c1 = data[nChannels*(i*width+j)+0];
					 unsigned int idx1 = std::floor((float)(c1)*idimx);
					 
					 hist[idx1] += (1.0f-(j-mdata[i]))*factor*i*invtheta;
					 }*/
				}
				break;
				
			case 2:
				for( unsigned int i=_h1; i<_h2; ++i ) {
					unsigned int j=_r1[i]*srcchannels;
					unsigned int r2end = _r2[i]*srcchannels;
					for( ; j<r2end; j+=srcchannels ) {
						unsigned int idx = (i*pitch + j);
						float c1 = (float)src[idx+0];
						float c2 = (float)src[idx+1];
						unsigned int idx1 = (unsigned int)std::floor(c1*idimx);
						unsigned int idx2 = (unsigned int)std::floor(c2*idimy);
						_hist[idx2*dimx + idx1] += j*invtheta;
					}
					
					
					/*if( _mask->uncertainty() == 0 ) {
					 continue;
					 }
					 float factor = 1.0f/_mask->uncertainty();
					 for( ; j<mdata[i]+_mask->uncertainty(); ++j ) {
					 unsigned char c1 = data[3*(i*width+j)+0];
					 unsigned char c2 = data[3*(i*width+j)+1];
					 unsigned int idx1 = std::floor((float)(c1)*idimx);
					 unsigned int idx2 = std::floor((float)(c2)*idimy);
					 
					 hist[idx2*dimx + idx1] += (1.0f-(j-mdata[i]))*factor*i*invtheta;
					 }*/
				}
				
				break;
				
			case 3:
				__m128i dimmul = _mm_set_epi16(0, 0, dimz, dimy, dimx, dimz, dimy, dimx);
				__m128i adjustv = _mm_set_epi16(0, 0, dimy*dimx, dimx, 1, dimy*dimx, dimx, 1);
				__m128i zeros = _mm_set1_epi16(0);
				for( unsigned int i=_h1; i<_h2; ++i ) {
					unsigned int j=_r1[i]*srcchannels;
					unsigned int r2end = _r2[i]*srcchannels;
					for( ; j<r2end; j+=2*srcchannels ) {
						
						unsigned int idx = (i*pitch + j);
						__m128i c123u8 = _mm_loadu_si128((const __m128i*)(src+idx));
						__m128i c123s16 = _mm_unpacklo_epi8(c123u8, zeros);
						
						__m128i idx123s16 = _mm_srli_epi16(_mm_mullo_epi16(c123s16, dimmul),8);
						
						
						__m128i idxadjusts16 = _mm_mullo_epi16(idx123s16, adjustv);
						
						__m128i idxsum1u16 = _mm_add_epi16(idxadjusts16, _mm_srli_epi32(idxadjusts16, 16));
						__m128i idxsum2u16 = _mm_add_epi16(idxsum1u16, _mm_srli_epi64(idxsum1u16, 32));
						
						
						int idx0 = _mm_extract_epi16(idxsum2u16, 0);
						
						_hist[idx0] += j*invtheta;
						
						if( j+srcchannels < r2end ) {
							_hist[_mm_extract_epi16(idxsum2u16, 4)] += (j+srcchannels)*invtheta;
						}
						
					}
				}
				break;
				
				//            case 3:
				//                for( unsigned int i=_h1; i<_h2; ++i ) {
				//					int j=_r1[i]*srcchannels;
				//					int r2end = _r2[i]*srcchannels;
				//					for( ; j<r2end; j+=srcchannels ) {
				//                        unsigned int idx = (i*pitch + j);
				//                        float c1 = (float)src[idx+0];
				//                        float c2 = (float)src[idx+1];
				//                        float c3 = (float)src[idx+2];
				//
				//                        unsigned int idx1 = std::floor(c1*idimx);
				//                        unsigned int idx2 = std::floor(c2*idimy);
				//                        unsigned int idx3 = std::floor(c3*idimz);
				//
				//                        _hist[idx3*dimy*dimx + idx2*dimx + idx1] += j*invtheta;
				//                    }
				//                }
				//                break;
				
		}
		
	}
	
};

