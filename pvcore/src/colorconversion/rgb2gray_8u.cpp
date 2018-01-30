#include <pvcore/sse_helpers.h>
namespace pvcore {
	
	void _rgb2gray_8u(const unsigned char* _src, unsigned char* _dest, unsigned int _width,
					  unsigned int _pitchs, unsigned int _pitchd,
					  unsigned int _start, unsigned int _stop) {
		
#ifdef USE_AVX2
		{
			// Generate buffers
			GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO16(buff_int2plan);
			
			// Load buffers to sse registers
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW(buff_int2plan);
			
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH(buff_int2plan);
			
			
			// Create coefficients for color mixing
			ALIGNED_BUFFER(unsigned long long, redc, 4);
			ALIGNED_BUFFER(unsigned long long, greenc, 4);
			ALIGNED_BUFFER(unsigned long long, bluec, 4);
			redc[0] = redc[1] = redc[2] = redc[3] = 0x001d001d001d001d;
			greenc[0] = greenc[1] = greenc[2] = greenc[3] = 0x0096009600960096;
			bluec[0] = bluec[1] = bluec[2] = bluec[3] = 0x004d004d004d004d;
			
			__m256i red_coeff   = _mm256_load_si256((const __m256i*)redc);
			__m256i green_coeff = _mm256_load_si256((const __m256i*)greenc);
			__m256i blue_coeff  = _mm256_load_si256((const __m256i*)bluec);
			
			const unsigned char* tsrc;
			unsigned char* tdest;
			
			for( unsigned int y=_start; y<_stop; ++y ) {
				tsrc = _src + y*_pitchs;
				tdest = _dest + y*_pitchd;
				for( unsigned int x=0; x<_pitchs; x+=3*32 ) {
					
					// We process RGB data in blocks of 3 x 16byte blocks:
					const __m256i v0 = _mm256_loadu2_m128i((const __m128i*)tsrc+3, (const __m128i*)tsrc); tsrc += 16;
					const __m256i v1 = _mm256_loadu2_m128i((const __m128i*)tsrc+3, (const __m128i*)tsrc); tsrc += 16;
					const __m256i v2 = _mm256_loadu2_m128i((const __m128i*)tsrc+3, (const __m128i*)tsrc); tsrc += 64;
					
					SHUFFLE_RGB2PLANAR_LOW(v0, v1, v2, red_lo, green_lo, blue_lo);
					
					const __m256i gray_lo = _mm256_adds_epu16(_mm256_mullo_epi16(red_lo, red_coeff),
															  _mm256_adds_epu16(_mm256_mullo_epi16(green_lo, green_coeff),
																				_mm256_mullo_epi16(blue_lo, blue_coeff) ));
					
					const __m256i gray_lo_shift = _mm256_srli_epi16(gray_lo,8);
					
					SHUFFLE_RGB2PLANAR_HIGH(v0, v1, v2, red_hi, green_hi, blue_hi);
					
					const __m256i gray_hi = _mm256_adds_epu16(_mm256_mullo_epi16(red_hi, red_coeff),
															  _mm256_adds_epu16(_mm256_mullo_epi16(green_hi, green_coeff),
																				_mm256_mullo_epi16(blue_hi,  blue_coeff) ));
					
					const __m256i gray_hi_shift = _mm256_srli_epi16(gray_hi,8);
					
					const __m256i gray_packed = _mm256_packus_epi16(gray_lo_shift,gray_hi_shift);
					
					_mm256_store_si256((__m256i*)tdest,gray_packed);
					
					tdest+=32;
				}
			}
		}
#else
		{
			// Generate buffers
			GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO16(buff_int2plan);
			
			// Load buffers to sse registers
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_16(buff_int2plan);
			
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_16(buff_int2plan);
			
			// Create coefficients for color mixing
			ALIGNED_BUFFER(unsigned long long, redc, 2);
			ALIGNED_BUFFER(unsigned long long, greenc, 2);
			ALIGNED_BUFFER(unsigned long long, bluec, 2);
			redc[0] = redc[1] = 0x001d001d001d001d;
			greenc[0] = greenc[1] = 0x0096009600960096;
			bluec[0] = bluec[1] = 0x004d004d004d004d;
			
			__m128i red_coeff   = _mm_load_si128((const __m128i*)redc);
			__m128i green_coeff = _mm_load_si128((const __m128i*)greenc);
			__m128i blue_coeff  = _mm_load_si128((const __m128i*)bluec);
			
			const unsigned char* tsrc;
			unsigned char* tdest;
			
			for( int y=_start; y<_stop; ++y ) {
				tsrc = _src + y*_pitchs;
				tdest = _dest + y*_pitchd;
				for( int x=0; x<_pitchs; x+=3*16 ) {
					
					// We process RGB data in blocks of 3 x 16byte blocks:
					const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					
					SHUFFLE_2X_RGB2PLANAR_LOW(v0, v1, v2, red_lo, green_lo, blue_lo);
					
					const __m128i gray_lo = _mm_adds_epu16(_mm_mullo_epi16(red_lo, red_coeff),
														   _mm_adds_epu16(_mm_mullo_epi16(green_lo, green_coeff),
																		  _mm_mullo_epi16(blue_lo, blue_coeff) ));
					
					const __m128i gray_lo_shift = _mm_srli_epi16(gray_lo,8);
					
					SHUFFLE_2X_RGB2PLANAR_HIGH(v0, v1, v2, red_hi, green_hi, blue_hi);
					
					const __m128i gray_hi = _mm_adds_epu16(_mm_mullo_epi16(red_hi, red_coeff),
														   _mm_adds_epu16(_mm_mullo_epi16(green_hi, green_coeff),
																		  _mm_mullo_epi16(blue_hi,  blue_coeff) ));
					
					const __m128i gray_hi_shift = _mm_srli_epi16(gray_hi,8);
					
					const __m128i gray_packed = _mm_packus_epi16(gray_lo_shift,gray_hi_shift);
					
					_mm_store_si128((__m128i*)tdest,gray_packed);
					
					tdest+=16;
				}
			}
		}
#endif
		
	}
	
	
	void _rgb2grayc_8u(const unsigned char* _src, unsigned char* _dest, unsigned int _width,
					   unsigned int _pitchs, unsigned int _pitchd,
					   unsigned int _start, unsigned int _stop) {
		
#ifdef USE_AVX2
		{
			// Generate buffers
			GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO16(buff_int2plan);
			
			// Load buffers to sse registers
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW(buff_int2plan);
			
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH(buff_int2plan);
			
			ALIGNED_BUFFER(unsigned long long, redc, 4);
			ALIGNED_BUFFER(unsigned long long, greenc, 4);
			ALIGNED_BUFFER(unsigned long long, bluec, 4);
			redc[0] = redc[1] = redc[2] = redc[3] = 0x001d001d001d001d;
			greenc[0] = greenc[1] = greenc[2] = greenc[3] = 0x0096009600960096;
			bluec[0] = bluec[1] = bluec[2] = bluec[3] = 0x004d004d004d004d;
			
			__m256i red_coeff   = _mm256_load_si256((const __m256i*)redc);
			__m256i green_coeff = _mm256_load_si256((const __m256i*)greenc);
			__m256i blue_coeff  = _mm256_load_si256((const __m256i*)bluec);
			
			const unsigned char* tsrc;
			unsigned char* tdest;
			
			for( unsigned int y=_start; y<_stop; ++y ) {
				tsrc = _src + y*_pitchs;
				tdest = _dest + y*_pitchd;
				for( unsigned int x=0; x<_pitchs; x+=3*32 ) {
					
					// We process RGB data in blocks of 3 x 16byte blocks:
					const __m256i v0 = _mm256_loadu2_m128i((const __m128i*)tsrc+3, (const __m128i*)tsrc); tsrc += 16;
					const __m256i v1 = _mm256_loadu2_m128i((const __m128i*)tsrc+3, (const __m128i*)tsrc); tsrc += 16;
					const __m256i v2 = _mm256_loadu2_m128i((const __m128i*)tsrc+3, (const __m128i*)tsrc); tsrc += 64;
					
					SHUFFLE_RGB2PLANAR_LOW(v0, v1, v2, red_lo, green_lo, blue_lo);
					
					const __m256i gray_lo = _mm256_adds_epu16(_mm256_mullo_epi16(red_lo, red_coeff),
															  _mm256_adds_epu16(_mm256_mullo_epi16(green_lo, green_coeff),
																				_mm256_mullo_epi16(blue_lo, blue_coeff) ));
					
					const __m256i gray_lo_shift = _mm256_srli_epi16(gray_lo,8);
					
					SHUFFLE_RGB2PLANAR_HIGH(v0, v1, v2, red_hi, green_hi, blue_hi);
					
					const __m256i gray_hi = _mm256_adds_epu16(_mm256_mullo_epi16(red_hi, red_coeff),
															  _mm256_adds_epu16(_mm256_mullo_epi16(green_hi, green_coeff),
																				_mm256_mullo_epi16(blue_hi,  blue_coeff) ));
					
					const __m256i gray_hi_shift = _mm256_srli_epi16(gray_hi,8);
					
					const __m256i gray_packed = _mm256_packus_epi16(gray_lo_shift,gray_hi_shift);
					
					_mm256_store_si256((__m256i*)tdest,gray_packed);
					
					tdest+=32;
				}
			}
		}
#else
		{
			// Generate buffers
			GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO16(buff_int2plan);
			
			// Load buffers to sse registers
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_16(buff_int2plan);
			
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_16(buff_int2plan);
			
			// Create coefficients for color mixing
			
			
			// gray to 3 channel gray
			ALIGNED_BUFFER(unsigned long long, buff, 6);
			buff[0] = 0x0202010101000000ull; buff[1] = 0x0504040403030302ull;
			buff[2] = 0x0707070606060505ull; buff[3] = 0x0a0a090909080808ull;
			buff[4] = 0x0d0c0c0c0b0b0b0aull; buff[5] = 0x0f0f0f0e0e0e0d0dull;
			
			__m128i gray_v0     = _mm_load_si128((const __m128i*)buff+0);
			__m128i gray_v1     = _mm_load_si128((const __m128i*)buff+1);
			__m128i gray_v2     = _mm_load_si128((const __m128i*)buff+2);
			
			
			ALIGNED_BUFFER(unsigned long long, redc, 2);
			ALIGNED_BUFFER(unsigned long long, greenc, 2);
			ALIGNED_BUFFER(unsigned long long, bluec, 2);
			redc[0] = redc[1] = 0x001d001d001d001d;
			greenc[0] = greenc[1] = 0x0096009600960096;
			bluec[0] = bluec[1] = 0x004d004d004d004d;
			
			__m128i red_coeff   = _mm_load_si128((const __m128i*)redc);
			__m128i green_coeff = _mm_load_si128((const __m128i*)greenc);
			__m128i blue_coeff  = _mm_load_si128((const __m128i*)bluec);
			
			const unsigned char* tsrc;
			unsigned char* tdest;
			
			for( int y=_start; y<_stop; ++y ) {
				tsrc = _src + y*_pitchs;
				tdest = _dest + y*_pitchd;
				for( int x=0; x<_pitchs; x+=3*16 ) {
					
					// We process RGB data in blocks of 3 x 16byte blocks:
					const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					
					SHUFFLE_2X_RGB2PLANAR_LOW(v0, v1, v2, red_lo, green_lo, blue_lo);
					
					const __m128i gray_lo = _mm_adds_epu16(_mm_mullo_epi16(red_lo, red_coeff),
														   _mm_adds_epu16(_mm_mullo_epi16(green_lo, green_coeff),
																		  _mm_mullo_epi16(blue_lo, blue_coeff) ));
					
					const __m128i gray_lo_shift = _mm_srli_epi16(gray_lo,8);
					
					SHUFFLE_2X_RGB2PLANAR_HIGH(v0, v1, v2, red_hi, green_hi, blue_hi);
					
					const __m128i gray_hi = _mm_adds_epu16(_mm_mullo_epi16(red_hi, red_coeff),
														   _mm_adds_epu16(_mm_mullo_epi16(green_hi, green_coeff),
																		  _mm_mullo_epi16(blue_hi,  blue_coeff) ));
					
					const __m128i gray_hi_shift = _mm_srli_epi16(gray_hi,8);
					
					const __m128i gray_packed = _mm_packus_epi16(gray_lo_shift,gray_hi_shift);
					
					_mm_store_si128((__m128i*)tdest,_mm_shuffle_epi8(gray_packed,gray_v0)); tdest += 16;
					_mm_store_si128((__m128i*)tdest,_mm_shuffle_epi8(gray_packed,gray_v1)); tdest += 16;
					_mm_store_si128((__m128i*)tdest,_mm_shuffle_epi8(gray_packed,gray_v2)); tdest += 16;
					
				}
			}
		}
#endif
		
	}
	
	
} // namespace pvcore

