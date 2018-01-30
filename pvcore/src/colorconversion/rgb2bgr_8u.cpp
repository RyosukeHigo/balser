
namespace pvcore {
	
	
	void _rgb2bgr_8u(const unsigned char* _src, unsigned char* _dest, unsigned int _width,
					 unsigned int _pitchs, unsigned int _pitchd,
					 unsigned int _start, unsigned int _stop )  {

#ifdef USE_SSE
		ALIGNED_BUFFER(unsigned long long, buff, 14);
		
		// Shuffles r and b bytes
		buff[0] = 0x0708030405000102ull; buff[1] = 0x800c0d0e090a0b06ull;
		buff[2] = 0x0506070203048000ull; buff[3] = 0x0f800b0c0d08090aull;
		buff[4] = 0x0904050601020380ull; buff[5] = 0x0d0e0f0a0b0c0708ull;
		
		buff[6]  = 0x8080808080800f80ull; buff[7]  = 0x8080808080808080ull;
		buff[8]  = 0x8080808080808080ull; buff[9]  = 0x0180808080808080ull;
		buff[10] = 0x808080808080800eull; buff[11] = 0x8080808080808080ull;
		buff[12] = 0x8080808080808080ull; buff[13] = 0x8000808080808080ull;
		
		// Read masks
		const __m128i shuffle0 = _mm_load_si128((const __m128i*)buff+0);
		const __m128i shuffle1 = _mm_load_si128((const __m128i*)buff+1);
		const __m128i shuffle2 = _mm_load_si128((const __m128i*)buff+2);
		
		const __m128i shufflea = _mm_load_si128((const __m128i*)buff+3);
		const __m128i shuffleb = _mm_load_si128((const __m128i*)buff+4);
		const __m128i shufflec = _mm_load_si128((const __m128i*)buff+5);
		const __m128i shuffled = _mm_load_si128((const __m128i*)buff+6);
		
		// This is the number of 3*16 blocks assigned to the thread
		const unsigned int width = (_pitchs/3) >> 4;
		
		// Get start positions for buffers
		const unsigned char* tsrc;
		unsigned char* tdest;
		
		for( unsigned int y=_start; y<_stop; ++y ) {
			tsrc = _src+(y*_pitchs);
			tdest = _dest+(y*_pitchd);
			for( unsigned int x=0; x<width; ++x ) {
				const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc += VEC_ALIGN;
				const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc += VEC_ALIGN;
				const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc += VEC_ALIGN;
				
				const __m128i v0p = v0;
				const __m128i v1p = v1;
				const __m128i v2p = v2;
				
				// Shuffle bits within each vector
				__m128i r0 = _mm_shuffle_epi8( v0, shuffle0 );
				__m128i r1 = _mm_shuffle_epi8( v1, shuffle1 );
				__m128i r2 = _mm_shuffle_epi8( v2, shuffle2 );
				
				// Some bits should go in other vectors
				r0 = _mm_or_si128( r0, _mm_shuffle_epi8( v1p, shuffleb ));
				r1 = _mm_or_si128( r1, _mm_shuffle_epi8( v0p, shufflea ));
				r1 = _mm_or_si128( r1, _mm_shuffle_epi8( v2p, shuffled ));
				r2 = _mm_or_si128( r2, _mm_shuffle_epi8( v1p, shufflec ));
				
				_mm_store_si128( (__m128i*)tdest, r0 ); tdest += VEC_ALIGN;
				_mm_store_si128( (__m128i*)tdest, r1 ); tdest += VEC_ALIGN;
				_mm_store_si128( (__m128i*)tdest, r2 ); tdest += VEC_ALIGN;
			}
		}
#else
		// Get start positions for buffers
		const unsigned char* tsrc;
		unsigned char* tdest;

		for( unsigned int y=_start; y<_stop; ++y ) {
			tsrc = _src+(y*_pitchs);
			tdest = _dest+(y*_pitchd);
			for( unsigned int x=0; x<_width; ++x ) {
				unsigned char t = tsrc[3*x];
				tdest[3*x] = tsrc[3*x+2];
				tdest[3*x+2] = t;
			}
		}
#endif
		
	}
	

	
	
    // =============================================================
    // ====================== RGBX2BGRX_8U =========================
    // =============================================================
	void _rgbx2bgrx_8u(const unsigned char* _src, unsigned char* _dest, unsigned int _width,
					   unsigned int _pitchs, unsigned int _pitchd,
					   unsigned int _start, unsigned int _stop ) {
		
		ALIGNED_BUFFER(unsigned long long, buff, 2);
		
		// Shuffles r and b bytes
		buff[0] = 0x0704050603000102ull; buff[1] = 0x0f0c0d0e0b08090aull;
		
		// Read masks
		const __m128i shuffle = _mm_load_si128((const __m128i*)buff);
		
		// This is the number of 3*16 blocks assigned to the thread
		const unsigned int widthz = (_pitchs/4) >> 2;
		
		// Get start positions for buffers
		const unsigned char* tsrc;
		unsigned char* tdest;
		
		for( unsigned int y=_start; y<=_stop; ++y ) {
			tsrc = _src+(y*_pitchs);
			tdest = _dest+(y*_pitchd);
			for( unsigned int x=0; x<widthz; ++x ) {
				// Load source vector
				const __m128i v = _mm_load_si128((const __m128i*)tsrc); tsrc += VEC_ALIGN;
				
				// Shuffle bits within each vector
				__m128i r = _mm_shuffle_epi8( v, shuffle );
				
				// Store result vector
				_mm_store_si128( (__m128i*)tdest, r ); tdest += VEC_ALIGN;
			}
		}
	}

	
    
    // =============================================================
    // ====================== RGB2BGR_32F ==========================
    // =============================================================
	// NOT WORKING
	void _rgb2bgr_32f(const float* _src, float* _dest, unsigned int _width,
					  unsigned int _pitchs, unsigned int _pitchd,
					  unsigned int _start, unsigned int _stop) {
		
		
		// This is the number of 3*16 blocks assigned to the thread
		const unsigned int widthz = (_pitchs/3) >> 2;
		
		// Get start positions for buffers
		const float* tsrc;
		float* tdest;
		
		for( unsigned int y=_start; y<=_stop; ++y ) {
			tsrc = _src+(y*_pitchs);
			tdest = _dest+(y*_pitchd);
			for( unsigned int x=0; x<widthz; ++x ) {
				const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc+=4;
				const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc+=4;
				const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc+=4;
				
				// Shuffle bits within each vector
				__m128i r0t = _mm_castps_si128(_mm_shuffle_ps( _mm_castsi128_ps(v0), _mm_castsi128_ps(v0), _MM_SHUFFLE(3,0,1,2) ) );
				__m128i r0 = _mm_castps_si128(_mm_shuffle_ps( _mm_castsi128_ps(r0t), _mm_castsi128_ps(v1), _MM_SHUFFLE(3,2,1,0) ));
				__m128i r1t = _mm_castps_si128(_mm_shuffle_ps( _mm_castsi128_ps(v1), _mm_castsi128_ps(v1), _MM_SHUFFLE(3,2,1,0) ));
				__m128i r2 = _mm_castps_si128(_mm_shuffle_ps( _mm_castsi128_ps(v2), _mm_castsi128_ps(v2), _MM_SHUFFLE(3,2,1,0) ));
				
				//
				_mm_store_si128( (__m128i*)tdest, r0 ); tdest+=4;
				_mm_store_si128( (__m128i*)tdest, r1t ); tdest+=4;
				_mm_store_si128( (__m128i*)tdest, r2 ); tdest+=4;
			}
		}
	}
	

	// =============================================================
	// ====================== RGBX2BGRX_32F ========================
	// =============================================================
	void _rgbx2bgrx_32f(const float* _src, float* _dest, unsigned int _width,
						unsigned int _pitchs, unsigned int _pitchd,
						unsigned int _start, unsigned int _stop) {
		
		
		const unsigned int widthz = (_pitchs/8);
		
		// Get start positions for buffers
		const float* tsrc;
		float* tdest;
		
		for( unsigned int y=_start; y<=_stop; ++y ) {
			tsrc = _src+(y*_pitchs);
			tdest = _dest+(y*_pitchd);
			for( unsigned int x=0; x<widthz; ++x ) {
				
#ifdef USE_AVX1
				const __m256 v0 = _mm256_load_ps(tsrc); tsrc+=8;
				
				__m256 r0 = _mm256_permute_ps(v0,0xc6);
				
				_mm256_store_ps(tdest, r0 ); tdest+=8;
#else // NOT TESTED
				
				const __m128 v0 = _mm_load_ps(tsrc); tsrc+=4;
				const __m128 v1 = _mm_load_ps(tsrc); tsrc+=4;
				
				//__m128 r0 = _mm_shuffle_ps(v0,0xc6);
				//__m128 r1 = _mm_shuffle_ps(v1,0xc6);
				
				//_mm_store_ps(tdest, r0 ); tdest+=4;
				//_mm_store_ps(tdest, r1 ); tdest+=4;
#endif
				
			}
		}
	}
	

} // namespace pvcore
