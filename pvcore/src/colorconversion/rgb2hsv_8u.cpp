#include <algorithm>

#include <pvcore/sse_helpers.h>

namespace pvcore {
	
	// =============================================================
	// ======================= RGB2HSV_8U ==========================
	// =============================================================
	void _rgb2hsv_8u(const unsigned char* _src, unsigned char* _dest, unsigned int _width,
					 unsigned int _pitchs, unsigned int _pitchd,
					 unsigned int _start, unsigned int _stop) {
		
#ifdef USE_AVX2
		{
			GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO16(buff_int2plan);
			GENERATE_BUFFERS_PLANAR_TO_INTERLEAVED(buff_plan2int);

			
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW(buff_int2plan);
			LOAD_BUFFERS_PLANAR_TO_INTERLEAVED(buff_plan2int);
			
			// Get start positions for buffers
			const unsigned char* tsrc  = NULL;
			unsigned char* tdest = NULL;
			
			// Constants
			const __m256i zeros = _mm256_set1_epi16(0);
			const __m256  two55  = _mm256_set1_ps( 255.0f );
			const __m256  four25 = _mm256_set1_ps(30.0f);
			const __m256i ones  = _mm256_set1_epi8( (char)0xff );
			//const __m128i thirty = _mm_set1_epi16( 30 );
			
			// Start computing Hue
			
			//         C = max(r,g,b) - min(r,g,b)
			//         _H = C==0 ? 0
			//         = g-b ? max(r,g,b) == r
			//         = b-r ? max(r,g,b) == g
			//         = r-g ? max(r,g,b) == b
			
			//         _H = _H < 0 ? _H+6C : _H
			//         H = 255*_H/(6C)
			
			for( unsigned int y=_start; y<_stop; ++y ) {
				tsrc  = _src+(y*_pitchs);
				tdest = _dest+(y*_pitchd);
				for( unsigned int x=0; x<_pitchs; x+=3*32 ) {
					
					// Read image
					const __m256i v0 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 16;
					const __m256i v1 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 16;
					const __m256i v2 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 64;
					
					// Final HSV vectors
					__m256i HSV0 = zeros;
					__m256i HSV1 = zeros;
					__m256i HSV2 = zeros;
					
					{
						
						// ======== LOW 8 bytes =======
						// Shuffle colors
						SHUFFLE_RGB2PLANAR_LOW(v0, v1, v2, reds_low, greens_low, blues_low);
						
						const __m256i maxlow = _mm256_max_epu8( reds_low, _mm256_max_epu8( blues_low, greens_low ) );
						const __m256i minlow = _mm256_min_epu8( reds_low, _mm256_min_epu8( blues_low, greens_low ) );
						
						const __m256i Clow = _mm256_sub_epi16(maxlow,minlow); // Unsigned 16-bit integer
						
						const __m256i Clow2 = _mm256_slli_epi16(Clow,1); // Unsigned 16-bit integer
						const __m256i Clow4 = _mm256_slli_epi16(Clow,2); // Unsigned 16-bit integer
						const __m256i Clow6 = _mm256_add_epi16(Clow2,Clow4); // Unsigned 16-bit integer
						
						const __m256i rmax = _mm256_subs_epi16( greens_low, blues_low  ); // Has Value between -Clow and Clow
						const __m256i gmax = _mm256_add_epi16(_mm256_subs_epi16( blues_low , reds_low   ), Clow2); // Has Value between Clow and 3*Clow
						const __m256i bmax = _mm256_add_epi16(_mm256_subs_epi16( reds_low  , greens_low ), Clow4); // Has Value between 3*Clow and 5*Clow
						
						
						// Has values between -Clow and 5*Clow
						__m256i Hlowt = _mm256_and_si256(_mm256_cmpeq_epi16(maxlow, reds_low) , rmax);
						Hlowt = _mm256_or_si256(Hlowt, _mm256_and_si256( _mm256_andnot_si256(_mm256_cmpeq_epi16(maxlow, reds_low),
																							 _mm256_cmpeq_epi16(maxlow, greens_low)),
																		gmax ) );
						Hlowt = _mm256_or_si256(Hlowt, _mm256_and_si256( _mm256_andnot_si256(_mm256_or_si256(_mm256_cmpeq_epi16(maxlow, reds_low),
																											 _mm256_cmpeq_epi16(maxlow, greens_low)),
																							 _mm256_cmpeq_epi16(maxlow, blues_low)),
																		bmax ) );
						
						
						// Has values in range [0 .. 6*Clow[
						// Note that 0 and 6*Clow is the same color
						// The hue could take value 6*Clow, but this is prevented by the order of assignment,
						// That is, if Hlow = 0, this means that r=g=max(r,g,b) and b=min(r,g,b)
						// For Hlow = 6, then the situation must be the same, but this is, as said above,
						// prevented by the order of assignment
						__m256i Hlow = _mm256_or_si256(_mm256_and_si256( _mm256_add_epi16( Hlowt, Clow6 ),
																		_mm256_cmpgt_epi16( zeros, Hlowt ) ),
													   _mm256_and_si256( Hlowt ,
																		_mm256_cmpgt_epi16( Hlowt, zeros )) ); // Has values between - and 6*Clow
						
						
						// To fit between 0_255 every number in Hlow should now be multiplied by 255/(6*Clow) = 85/(2*Clow). Hlow is a number between 0 and 6*255
						// To fit between 0_179 every number in Hlow should now be multiplied by 180/(6*Clow) = 30/Clow. Hlow is a number between 0 and 6*255
						
						// Needs conversion to float to perform division - Convert first H to float (needs two float registers)
						const __m256i xlo = _mm256_unpacklo_epi16(Hlow, zeros);
						const __m256i xhi = _mm256_unpackhi_epi16(Hlow, zeros);
						const __m256 xlof = _mm256_cvtepi32_ps(xlo);
						const __m256 xhif = _mm256_cvtepi32_ps(xhi);
						
						// Then convert C to float (needs two float registers)
						const __m256i clo = _mm256_unpacklo_epi16(Clow, zeros);
						const __m256i chi = _mm256_unpackhi_epi16(Clow, zeros);
						const __m256 clof = _mm256_cvtepi32_ps(clo);
						const __m256 chif = _mm256_cvtepi32_ps(chi);
						
						// Multiply H with factor depending on C
						const __m256 factorlo = _mm256_div_ps( four25 , clof );
						const __m256 factorhi = _mm256_div_ps( four25 , chif );
						const __m256 xlof_div = _mm256_mul_ps(xlof,factorlo);
						const __m256 xhif_div = _mm256_mul_ps(xhif,factorhi);
						
						// Finally convert back to integer
						const __m256i HUE_low = _mm256_packus_epi16(_mm256_cvtps_epi32(xlof_div),
																	_mm256_cvtps_epi32(xhif_div));
						
						// HUE
						HSV0 = _mm256_shuffle_epi8( HUE_low, packh_1 );
						HSV1 = _mm256_shuffle_epi8( HUE_low, packh_2 );
						
						
						// VALUE
						HSV0 = _mm256_or_si256( HSV0, _mm256_shuffle_epi8(maxlow, packv_1));
						HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8(maxlow, packv_2));
						
						
						// SATURATION
						// Depends on C and V - first convert V to
						const __m256i mlo = _mm256_unpacklo_epi16(maxlow, zeros);
						const __m256i mhi = _mm256_unpackhi_epi16(maxlow, zeros);
						const __m256 mlof = _mm256_cvtepi32_ps(mlo);
						const __m256 mhif = _mm256_cvtepi32_ps(mhi);
						
						// Perform C/V
						const __m256 slof = _mm256_mul_ps( _mm256_div_ps( clof, mlof ), two55 );
						const __m256 shif = _mm256_mul_ps( _mm256_div_ps( chif, mhif ), two55 );
						const __m256i slo = _mm256_cvtps_epi32(slof);
						const __m256i shi = _mm256_cvtps_epi32(shif);
						const __m256i SAT_low = _mm256_and_si256( _mm256_xor_si256(_mm256_cmpeq_epi16(Clow,zeros), // Finds C==0
																				   ones), // Flips bits, i.e. C!=0
																 _mm256_packus_epi16(slo, shi)); // Selects SAT s.t. C!=0
						
						HSV0 = _mm256_or_si256( HSV0, _mm256_shuffle_epi8(SAT_low, packs_1));
						HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8(SAT_low, packs_2));
					}
					
					{
						LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH(buff_int2plan);
						
						// ======== HIGH 8 bytes =======
						// Shuffle colors
						SHUFFLE_RGB2PLANAR_HIGH(v0, v1, v2, reds_high, greens_high, blues_high);
						
						const __m256i maxhigh = _mm256_max_epu8( reds_high, _mm256_max_epu8( blues_high, greens_high ) );
						const __m256i minhigh = _mm256_min_epu8( reds_high, _mm256_min_epu8( blues_high, greens_high ) );
						
						const __m256i Chigh = _mm256_sub_epi16(maxhigh,minhigh); // Unsigned 16-bit integer
						
						const __m256i Chigh2 = _mm256_slli_epi16(Chigh,1); // Unsigned 16-bit integer
						const __m256i Chigh4 = _mm256_slli_epi16(Chigh,2); // Unsigned 16-bit integer
						const __m256i Chigh6 = _mm256_add_epi16(Chigh2,Chigh4); // Unsigned 16-bit integer
						
						const __m256i rmaxh = _mm256_subs_epi16( greens_high, blues_high  ); // Has Value between -Chigh and Chigh
						const __m256i gmaxh = _mm256_add_epi16( _mm256_subs_epi16( blues_high , reds_high   ), Chigh2); // Has Value between Chigh and 3*Chigh
						const __m256i bmaxh = _mm256_add_epi16( _mm256_subs_epi16( reds_high  , greens_high ), Chigh4); // Has Value between 3*Chigh and 5*Chigh
						
						
						// Has values in range [-Chigh .. 5*Chigh[
						__m256i Hhight = _mm256_and_si256(_mm256_cmpeq_epi16(maxhigh, reds_high) , rmaxh);
						Hhight = _mm256_or_si256(Hhight, _mm256_and_si256 (_mm256_andnot_si256(_mm256_cmpeq_epi16(maxhigh, reds_high),
																							   _mm256_cmpeq_epi16(maxhigh, greens_high)),
																		   gmaxh ) );
						Hhight = _mm256_or_si256(Hhight, _mm256_and_si256 (_mm256_andnot_si256(_mm256_or_si256(_mm256_cmpeq_epi16(maxhigh, reds_high),
																											   _mm256_cmpeq_epi16(maxhigh, greens_high)),
																							   _mm256_cmpeq_epi16(maxhigh, blues_high)),
																		   bmaxh ) );
						
						// Has values in range [0 .. 6*Chigh[
						// Note that 0 and 6*Chigh is the same color
						// The hue could take value 6*Chigh, but this is prevented by the order of assignment,
						// That is, if Hhigh = 0, this means that r=g=max(r,g,b) and b=min(r,g,b)
						// For Hhigh = 6, then the situation must be the same, but this is, as said above,
						// prevented by the order of assignment
						__m256i Hhigh = _mm256_or_si256(_mm256_and_si256( _mm256_add_epi16( Hhight, Chigh6 ),
																		 _mm256_cmpgt_epi16( zeros, Hhight ) ),
														_mm256_and_si256( Hhight ,
																		 _mm256_cmpgt_epi16( Hhight, zeros )) );
						
						
						// To fit between 0_255 every number in Hhigh should now be multiplied by 255/(6*Chigh) = 85/(2*Chigh). Hhigh is a number between 0 and 6*255
						
						// Needs conversion to float to perform division - Convert first H to float (needs two float registers)
						const __m256i xloh = _mm256_unpacklo_epi16(Hhigh, zeros );
						const __m256i xhih = _mm256_unpackhi_epi16(Hhigh, zeros );
						const __m256 xlohf = _mm256_cvtepi32_ps(xloh);
						const __m256 xhihf = _mm256_cvtepi32_ps(xhih);
						
						// Then convert C to float (needs two float registers)
						const __m256i cloh = _mm256_unpacklo_epi16(Chigh, zeros);
						const __m256i chih = _mm256_unpackhi_epi16(Chigh, zeros);
						const __m256 clofh = _mm256_cvtepi32_ps(cloh);
						const __m256 chifh = _mm256_cvtepi32_ps(chih);
						
						// Multiply H with factor depending on C
						const __m256 factorloh = _mm256_div_ps( four25 , clofh );
						const __m256 factorhih = _mm256_div_ps( four25 , chifh );
						const __m256 xlofh_div = _mm256_mul_ps(xlohf,factorloh);
						const __m256 xhifh_div = _mm256_mul_ps(xhihf,factorhih);
						
						// Finally convert back to integer
						const __m256i HUE_high = _mm256_packus_epi16(_mm256_cvtps_epi32(xlofh_div),
																	 _mm256_cvtps_epi32(xhifh_div));
						
						// HUE
						HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8( HUE_high, packh_3 ) );
						HSV2 = _mm256_shuffle_epi8( HUE_high, packh_4 ); // FINAL
						
						
						// VALUE
						HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8(maxhigh, packv_3));
						HSV2 = _mm256_or_si256( HSV2, _mm256_shuffle_epi8(maxhigh, packv_4));
						
						
						// SATURATION
						// Depends on C and V - first convert V to
						const __m256i mloh = _mm256_unpacklo_epi16(maxhigh, zeros);
						const __m256i mhih = _mm256_unpackhi_epi16(maxhigh, zeros);
						const __m256 mlofh = _mm256_cvtepi32_ps(mloh);
						const __m256 mhifh = _mm256_cvtepi32_ps(mhih);
						
						// Perform C/V
						const __m256 slofh = _mm256_mul_ps( _mm256_div_ps( clofh, mlofh ), two55 );
						const __m256 shifh = _mm256_mul_ps( _mm256_div_ps( chifh, mhifh ), two55 );
						const __m256i sloh = _mm256_cvtps_epi32(slofh);
						const __m256i shih = _mm256_cvtps_epi32(shifh);
						const __m256i SAT_high = _mm256_and_si256( _mm256_xor_si256(_mm256_cmpeq_epi16(Chigh,zeros), // Finds C==0
																					ones), // Flips bits, i.e. C!=0
																  _mm256_packus_epi16(sloh, shih)); // Selects SAT s.t. C!=0
						
						HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8(SAT_high, packs_3));
						HSV2 = _mm256_or_si256( HSV2, _mm256_shuffle_epi8(SAT_high, packs_4));
					}
					
					// STORE IN DEST
					_mm256_storeu2_m128i( (__m128i*)tdest+3, (__m128i*)tdest, HSV0 ); tdest += 16;
					_mm256_storeu2_m128i( (__m128i*)tdest+3, (__m128i*)tdest, HSV1 ); tdest += 16;
					_mm256_storeu2_m128i( (__m128i*)tdest+3, (__m128i*)tdest, HSV2 ); tdest += 64;
					
				}
			}
		}
#elif defined( USE_SSE )
		{
			// GENERATE BUFFERS FOR SHUFFLING
			GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO16(buff_int2plan);
			GENERATE_BUFFERS_PLANAR_TO_INTERLEAVED_16TO8(buff_plan2int);	
			
			// Read masks
			LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_16(buff_int2plan);
			LOAD_BUFFERS_PLANAR_TO_INTERLEAVED_16(buff_plan2int);
			

			
			
			// This is the number of 3*16 blocks assigned to the thread
			const unsigned int width = (_pitchs/3) >> 4;
			
			// Get start positions for buffers
			const unsigned char* tsrc  = _src+(_start*_pitchs);
			unsigned char* tdest = _dest+(_start*_pitchd);
			
			// Constants
			const __m128i zeros = _mm_set1_epi16(0);
			const __m128  two55  = _mm_set1_ps( 255.0f );
			const __m128  four25 = _mm_set1_ps(30.0f);
			const __m128i ones  = _mm_set1_epi8( 0xff );
			//const __m128i thirty = _mm_set1_epi16( 30 );
			
			// Start computing Hue
			/*
			 C = max(r,g,b) - min(r,g,b)
			 _H = C==0 ? 0
			 = g-b ? max(r,g,b) == r
			 = b-r ? max(r,g,b) == g
			 = r-g ? max(r,g,b) == b
			 
			 _H = _H < 0 ? _H+6C : _H
			 H = 255*_H/(6C)
			 
			 */
			for( int y=_start; y<_stop; ++y ) {
				tsrc  = _src+(y*_pitchs);
				tdest = _dest+(y*_pitchd);
				for( int x=0; x<width; ++x ) {
					
					// Read image
					const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
					
					// Final HSV vectors
					__m128i HSV0 = zeros;
					__m128i HSV1 = zeros;
					__m128i HSV2 = zeros;
					
					{
						
						// ======== LOW 8 bytes =======
						// Shuffle colors
						SHUFFLE_2X_RGB2PLANAR_LOW(v0,v1,v2,reds_low,greens_low,blues_low);
						
						const __m128i maxlow = _mm_max_epu8( reds_low, _mm_max_epu8( blues_low, greens_low ) );
						const __m128i minlow = _mm_min_epu8( reds_low, _mm_min_epu8( blues_low, greens_low ) );
						
						const __m128i Clow = _mm_sub_epi16(maxlow,minlow); // Unsigned 16-bit integer
						
						const __m128i Clow2 = _mm_slli_epi16(Clow,1); // Unsigned 16-bit integer
						const __m128i Clow4 = _mm_slli_epi16(Clow,2); // Unsigned 16-bit integer
						const __m128i Clow6 = _mm_add_epi16(Clow2,Clow4); // Unsigned 16-bit integer
						
						const __m128i rmax = _mm_subs_epi16( greens_low, blues_low  ); // Has Value between -Clow and Clow
						const __m128i gmax = _mm_add_epi16(_mm_subs_epi16( blues_low , reds_low   ), Clow2); // Has Value between Clow and 3*Clow
						const __m128i bmax = _mm_add_epi16(_mm_subs_epi16( reds_low  , greens_low ), Clow4); // Has Value between 3*Clow and 5*Clow
						
						
						// Has values between -Clow and 5*Clow
						__m128i Hlowt = _mm_and_si128(_mm_cmpeq_epi16(maxlow, reds_low) , rmax);
						Hlowt = _mm_or_si128(Hlowt, _mm_and_si128( _mm_andnot_si128(_mm_cmpeq_epi16(maxlow, reds_low),
																					_mm_cmpeq_epi16(maxlow, greens_low)),
																  gmax ) );
						Hlowt = _mm_or_si128(Hlowt, _mm_and_si128( _mm_andnot_si128(_mm_or_si128(_mm_cmpeq_epi16(maxlow, reds_low),
																								 _mm_cmpeq_epi16(maxlow, greens_low)),
																					_mm_cmpeq_epi16(maxlow, blues_low)),
																  bmax ) );
						
						
						// Has values in range [0 .. 6*Clow[
						// Note that 0 and 6*Clow is the same color
						// The hue could take value 6*Clow, but this is prevented by the order of assignment,
						// That is, if Hlow = 0, this means that r=g=max(r,g,b) and b=min(r,g,b)
						// For Hlow = 6, then the situation must be the same, but this is, as said above,
						// prevented by the order of assignment
						__m128i Hlow = _mm_or_si128(_mm_and_si128( _mm_add_epi16( Hlowt, Clow6 ),
																  _mm_cmpgt_epi16( zeros, Hlowt ) ),
													_mm_and_si128( Hlowt ,
																  _mm_cmpgt_epi16( Hlowt, zeros )) ); // Has values between - and 6*Clow
						
						
						// To fit between 0_255 every number in Hlow should now be multiplied by 255/(6*Clow) = 85/(2*Clow). Hlow is a number between 0 and 6*255
						// To fit between 0_179 every number in Hlow should now be multiplied by 180/(6*Clow) = 30/Clow. Hlow is a number between 0 and 6*255
						
						// Needs conversion to float to perform division - Convert first H to float (needs two float registers)
						const __m128i xlo = _mm_unpacklo_epi16(Hlow, zeros);
						const __m128i xhi = _mm_unpackhi_epi16(Hlow, zeros);
						const __m128 xlof = _mm_cvtepi32_ps(xlo);
						const __m128 xhif = _mm_cvtepi32_ps(xhi);
						
						// Then convert C to float (needs two float registers)
						const __m128i clo = _mm_unpacklo_epi16(Clow, zeros);
						const __m128i chi = _mm_unpackhi_epi16(Clow, zeros);
						const __m128 clof = _mm_cvtepi32_ps(clo);
						const __m128 chif = _mm_cvtepi32_ps(chi);
						
						// Multiply H with factor depending on C
						const __m128 factorlo = _mm_div_ps( four25 , clof );
						const __m128 factorhi = _mm_div_ps( four25 , chif );
						const __m128 xlof_div = _mm_mul_ps(xlof,factorlo);
						const __m128 xhif_div = _mm_mul_ps(xhif,factorhi);
						
						// Finally convert back to integer
						const __m128i HUE_low = _mm_packus_epi16(_mm_cvtps_epi32(xlof_div),
																 _mm_cvtps_epi32(xhif_div));
						
						// HUE
						HSV0 = _mm_shuffle_epi8( HUE_low, packh_1 );
						HSV1 = _mm_shuffle_epi8( HUE_low, packh_2 );
						
						
						// VALUE
						HSV0 = _mm_or_si128( HSV0, _mm_shuffle_epi8(maxlow, packv_1));
						HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8(maxlow, packv_2));
						
						
						// SATURATION
						// Depends on C and V - first convert V to
						const __m128i mlo = _mm_unpacklo_epi16(maxlow, zeros);
						const __m128i mhi = _mm_unpackhi_epi16(maxlow, zeros);
						const __m128 mlof = _mm_cvtepi32_ps(mlo);
						const __m128 mhif = _mm_cvtepi32_ps(mhi);
						
						// Perform C/V
						const __m128 slof = _mm_mul_ps( _mm_div_ps( clof, mlof ), two55 );
						const __m128 shif = _mm_mul_ps( _mm_div_ps( chif, mhif ), two55 );
						const __m128i slo = _mm_cvtps_epi32(slof);
						const __m128i shi = _mm_cvtps_epi32(shif);
						const __m128i SAT_low = _mm_and_si128( _mm_xor_si128(_mm_cmpeq_epi16(Clow,zeros), // Finds C==0
																			 ones), // Flips bits, i.e. C!=0
															  _mm_packus_epi16(slo, shi)); // Selects SAT s.t. C!=0
						
						HSV0 = _mm_or_si128( HSV0, _mm_shuffle_epi8(SAT_low, packs_1));
						HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8(SAT_low, packs_2));
					}
					
					{
						LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_16(buff_int2plan);
						
						// ======== HIGH 8 bytes =======
						// Shuffle colors
						SHUFFLE_2X_RGB2PLANAR_HIGH(v0,v1,v2,reds_high,greens_high,blues_high);
						
						const __m128i maxhigh = _mm_max_epu8( reds_high, _mm_max_epu8( blues_high, greens_high ) );
						const __m128i minhigh = _mm_min_epu8( reds_high, _mm_min_epu8( blues_high, greens_high ) );
						
						const __m128i Chigh = _mm_sub_epi16(maxhigh,minhigh); // Unsigned 16-bit integer
						
						const __m128i Chigh2 = _mm_slli_epi16(Chigh,1); // Unsigned 16-bit integer
						const __m128i Chigh4 = _mm_slli_epi16(Chigh,2); // Unsigned 16-bit integer
						const __m128i Chigh6 = _mm_add_epi16(Chigh2,Chigh4); // Unsigned 16-bit integer
						
						const __m128i rmaxh = _mm_subs_epi16( greens_high, blues_high  ); // Has Value between -Chigh and Chigh
						const __m128i gmaxh = _mm_add_epi16( _mm_subs_epi16( blues_high , reds_high   ), Chigh2); // Has Value between Chigh and 3*Chigh
						const __m128i bmaxh = _mm_add_epi16( _mm_subs_epi16( reds_high  , greens_high ), Chigh4); // Has Value between 3*Chigh and 5*Chigh
						
						
						// Has values in range [-Chigh .. 5*Chigh[
						__m128i Hhight = _mm_and_si128(_mm_cmpeq_epi16(maxhigh, reds_high) , rmaxh);
						Hhight = _mm_or_si128(Hhight, _mm_and_si128 (_mm_andnot_si128(_mm_cmpeq_epi16(maxhigh, reds_high),
																					  _mm_cmpeq_epi16(maxhigh, greens_high)),
																	 gmaxh ) );
						Hhight = _mm_or_si128(Hhight, _mm_and_si128 (_mm_andnot_si128(_mm_or_si128(_mm_cmpeq_epi16(maxhigh, reds_high),
																								   _mm_cmpeq_epi16(maxhigh, greens_high)),
																					  _mm_cmpeq_epi16(maxhigh, blues_high)),
																	 bmaxh ) );
						
						// Has values in range [0 .. 6*Chigh[
						// Note that 0 and 6*Chigh is the same color
						// The hue could take value 6*Chigh, but this is prevented by the order of assignment,
						// That is, if Hhigh = 0, this means that r=g=max(r,g,b) and b=min(r,g,b)
						// For Hhigh = 6, then the situation must be the same, but this is, as said above,
						// prevented by the order of assignment
						__m128i Hhigh = _mm_or_si128(_mm_and_si128( _mm_add_epi16( Hhight, Chigh6 ),
																   _mm_cmpgt_epi16( zeros, Hhight ) ),
													 _mm_and_si128( Hhight ,
																   _mm_cmpgt_epi16( Hhight, zeros )) );
						
						
						// To fit between 0_255 every number in Hhigh should now be multiplied by 255/(6*Chigh) = 85/(2*Chigh). Hhigh is a number between 0 and 6*255
						
						// Needs conversion to float to perform division - Convert first H to float (needs two float registers)
						const __m128i xloh = _mm_unpacklo_epi16(Hhigh, zeros );
						const __m128i xhih = _mm_unpackhi_epi16(Hhigh, zeros );
						const __m128 xlohf = _mm_cvtepi32_ps(xloh);
						const __m128 xhihf = _mm_cvtepi32_ps(xhih);
						
						// Then convert C to float (needs two float registers)
						const __m128i cloh = _mm_unpacklo_epi16(Chigh, zeros);
						const __m128i chih = _mm_unpackhi_epi16(Chigh, zeros);
						const __m128 clofh = _mm_cvtepi32_ps(cloh);
						const __m128 chifh = _mm_cvtepi32_ps(chih);
						
						// Multiply H with factor depending on C
						const __m128 factorloh = _mm_div_ps( four25 , clofh );
						const __m128 factorhih = _mm_div_ps( four25 , chifh );
						const __m128 xlofh_div = _mm_mul_ps(xlohf,factorloh);
						const __m128 xhifh_div = _mm_mul_ps(xhihf,factorhih);
						
						// Finally convert back to integer
						const __m128i HUE_high = _mm_packus_epi16(_mm_cvtps_epi32(xlofh_div),
																  _mm_cvtps_epi32(xhifh_div));
						
						// HUE
						HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8( HUE_high, packh_3 ) );
						HSV2 = _mm_shuffle_epi8( HUE_high, packh_4 ); // FINAL
						
						
						// VALUE
						HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8(maxhigh, packv_3));
						HSV2 = _mm_or_si128( HSV2, _mm_shuffle_epi8(maxhigh, packv_4));
						
						
						// SATURATION
						// Depends on C and V - first convert V to
						const __m128i mloh = _mm_unpacklo_epi16(maxhigh, zeros);
						const __m128i mhih = _mm_unpackhi_epi16(maxhigh, zeros);
						const __m128 mlofh = _mm_cvtepi32_ps(mloh);
						const __m128 mhifh = _mm_cvtepi32_ps(mhih);
						
						// Perform C/V
						const __m128 slofh = _mm_mul_ps( _mm_div_ps( clofh, mlofh ), two55 );
						const __m128 shifh = _mm_mul_ps( _mm_div_ps( chifh, mhifh ), two55 );
						const __m128i sloh = _mm_cvtps_epi32(slofh);
						const __m128i shih = _mm_cvtps_epi32(shifh);
						const __m128i SAT_high = _mm_and_si128( _mm_xor_si128(_mm_cmpeq_epi16(Chigh,zeros), // Finds C==0
																			  ones), // Flips bits, i.e. C!=0
															   _mm_packus_epi16(sloh, shih)); // Selects SAT s.t. C!=0
						
						HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8(SAT_high, packs_3));
						HSV2 = _mm_or_si128( HSV2, _mm_shuffle_epi8(SAT_high, packs_4));
					}
					
					// STORE IN DEST
					
					_mm_store_si128( (__m128i*)tdest, HSV0 ); tdest += 16;
					_mm_store_si128( (__m128i*)tdest, HSV1 ); tdest += 16;
					_mm_store_si128( (__m128i*)tdest, HSV2 ); tdest += 16;
					
				}
			}
		}
#else
		{
			//         C = max(r,g,b) - min(r,g,b)
			//         _H = C==0 ? 0
			//         = g-b ? max(r,g,b) == r
			//         = b-r ? max(r,g,b) == g
			//         = r-g ? max(r,g,b) == b
			
			//         _H = _H < 0 ? _H+6C : _H
			//         H = 255*_H/(6C)
			
			const unsigned char* tsrc;
			unsigned char* tdest;
			for( int y=_start; y<=_stop; ++y ) {
				tsrc = _src+(y*_pitchs);
				tdest = _dest+(y*_pitchd);
				for( int x=0; x<_width; ++x ) {
					
					float maxRGB = std::max(tsrc[3*x],std::max(tsrc[3*x+1],tsrc[3*x+2]));
					float minRGB = std::min(tsrc[3*x],std::min(tsrc[3*x+1],tsrc[3*x+2]));
					
					if( minRGB == maxRGB ) {
						tdest[3*x] = 0;
						tdest[3*x+1] = 0;
						tdest[3*x+2] = maxRGB;
						continue;
					}
					
					float r = tsrc[3*x];
					float g = tsrc[3*x+1];
					float b = tsrc[3*x+2];
					
					// Colors other than black-gray-white
					float C = maxRGB - minRGB;
					
					int H = (int)(0.5f*((r == maxRGB) ? 60.f*(g-b)/C :
										((g == maxRGB) ? 120.f + 60.f*(b-r)/C :
										 240.f + 60.f*(r-g)/C)));
					if(H < 0) H = H + 180;
					
					tdest[3*x+0] = (unsigned char)H;
					tdest[3*x+1] = (unsigned char)((255.f*(maxRGB - minRGB))/maxRGB);
					tdest[3*x+2] = maxRGB;
					
				}
			}
		}
#endif
	}
	
	
	
	
	// =============================================================
	// ======================= HSV2RGB_8U ==========================
	// =============================================================
	void _hsv2rgb_8u(const unsigned char* _src, unsigned char* _dest, unsigned int _width,
					 unsigned int _pitchs, unsigned int _pitchd,
					 unsigned int _start, unsigned int _stop) {
		
		// Get start positions for buffers
		const unsigned char* tsrc;
		unsigned char* tdest;
		
		for( unsigned int y=_start; y<_stop; ++y ) {
			tsrc = _src+(y*_pitchs);
			tdest = _dest+(y*_pitchd);
			for( unsigned int x=0; x<_width; ++x ) {
				
				// _src has value in range [0 .. 255] corresponding to degrees [0 .. 359]
				// These should be put in a range of [0..5]
				float h_ = ((float)*tsrc)/30.0f; ++tsrc;
				// s and v should take values in range [0 .. 1]
				float s = ((float)*tsrc)/255.0f; ++tsrc;
				float v = ((float)*tsrc)/255.0f; ++tsrc;
				
				// wiki:
				// c = v*s
				// x = c(1 - abs(h'%2-1)) \in {0,c} -> x = c or 0
				
				int switchi = (int)h_;
				float f = h_ - switchi;
				float pv = v * (1 - s);
				float qv = v * (1 - s * f);
				float tv = v * (1 - s * (1 - f));
				
				//float c = v*s;
				//float x = c*(1-abs(((int)h_)%2-1));
				
				if( v == 0 ) {
					*tdest = 0; ++tdest;
					*tdest = 0; ++tdest;
					*tdest = 0; ++tdest;
					continue;
				}
				if( s == 0 ) {
					*tdest = (unsigned char)(255.0f*v); ++tdest;
					*tdest = (unsigned char)(255.0f*v); ++tdest;
					*tdest = (unsigned char)(255.0f*v); ++tdest;
					continue;
				}
				
				float r = 0.0f;
				float g = 0.0f;
				float b = 0.0f;
				switch( switchi ) {
					case 0:
						r = v;
						g = tv;
						b = pv;
						break;
					case 1:
						r = qv;
						g = v;
						b = pv;
						break;
					case 2:
						r = pv;
						g = v;
						b = tv;
						break;
					case 3:
						r = pv;
						g = qv;
						b = v;
						break;
					case 4:
						r = tv;
						g = pv;
						b = v;
						break;
					case 5:
						r = v;
						g = pv;
						b = qv;
						break;
				}
				
				*tdest = (unsigned char)(255.0f*r); ++tdest;
				*tdest = (unsigned char)(255.0f*g); ++tdest;
				*tdest = (unsigned char)(255.0f*b); ++tdest;
				
			}
		}
		
	}
	
	//class RGB2HSV_8U_TBB {
	//
	//	const unsigned char *src;
	//	unsigned char *dest;
	//	unsigned int width;
	//	unsigned int height;
	//	unsigned int pitchs;
	//	unsigned int pitchd;
	//	unsigned int threads;
	//
	//private:
	//	void _rgb2hsv_8u(unsigned int, unsigned int) const;
	//
	//public:
	//	void operator()( const tbb::blocked_range<size_t>& r ) const {
	//
	//
	//		// Let's make the last thread do the least work
	//		float blockSize = (float)height/(float)threads;
	//
	//		unsigned int start = GET_START(r.begin(),blockSize);
	//		unsigned int stop  = GET_STOP(r.end(),blockSize);
	//
	//		_rgb2hsv_8u( start, stop );
	//	}
	//
	//	RGB2HSV_8U_TBB( const unsigned char* _src, unsigned char* _dest, unsigned int _width, unsigned int _height, unsigned int _pitchs, unsigned int _pitchd, unsigned int _threads ) :
	//	src(_src), dest(_dest), width(_width), height(_height), pitchs(_pitchs), pitchd(_pitchd), threads(_threads) {
	//
	//	}
	//
	//};
	//
	//
	//class HSV2RGB_8U_TBB {
	//
	//	const unsigned char *src;
	//	unsigned char *dest;
	//	unsigned int width;
	//	unsigned int height;
	//	unsigned int pitchs;
	//	unsigned int pitchd;
	//	unsigned int threads;
	//
	//private:
	//	void _hsv2rgb_8u(unsigned int, unsigned int ) const;
	//
	//public:
	//	void operator()( const tbb::blocked_range<size_t>& r ) const {
	//		_hsv2rgb_8u( 0, height );
	//	}
	//
	//	HSV2RGB_8U_TBB( const unsigned char* _src, unsigned char* _dest, unsigned int _width, unsigned int _height, unsigned int _pitchs, unsigned int _pitchd, unsigned int _threads  ) :
	//	src(_src), dest(_dest), width(_width), height(_height), pitchs(_pitchs), pitchd(_pitchd), threads(_threads) {
	//
	//	}
	//
	//};
	//
	//
	//
	//void RGB2HSV_8U_TBB::_rgb2hsv_8u(unsigned int _start,
	//								 unsigned int _stop) const {
	//
	//#ifdef USE_AVX2
	//	{
	//		ALIGNED_BUFFER(unsigned long long, buff, 96);
	//
	//		buff[0]  = 0x8009800680038000ull; buff[1]  = 0x80808080800f800cull; buff[2]  = 0x8009800680038000ull; buff[3]  = 0x80808080800f800cull;    // Puts red colors from v0 in low
	//		buff[4]  = 0x800a800780048001ull; buff[5]  = 0x808080808080800dull; buff[6]  = 0x800a800780048001ull; buff[7]  = 0x808080808080800dull;    // Puts green colors from v0 in low
	//		buff[8]  = 0x800b800880058002ull; buff[9]  = 0x808080808080800eull; buff[10] = 0x800b800880058002ull; buff[11] = 0x808080808080800eull;    // Puts blue colors from v0 in low
	//
	//		buff[12] = 0x8080808080808080ull; buff[13] = 0x8005800280808080ull; buff[14] = 0x8080808080808080ull; buff[15] = 0x8005800280808080ull;    // Puts red colors from v1 in low
	//		buff[16] = 0x8080808080808080ull; buff[17] = 0x8006800380008080ull; buff[18] = 0x8080808080808080ull; buff[19] = 0x8006800380008080ull;    // Puts green colors from v1 in low
	//		buff[20] = 0x8080808080808080ull; buff[21] = 0x8007800480018080ull; buff[22] = 0x8080808080808080ull; buff[23] = 0x8007800480018080ull;    // Puts blue colors from v1 in low
	//
	//		buff[24] = 0x8080800e800b8008ull; buff[25] = 0x8080808080808080ull; buff[26] = 0x8080800e800b8008ull; buff[27] = 0x8080808080808080ull;    // Puts red colors from v1 in high
	//		buff[28] = 0x8080800f800c8009ull; buff[29] = 0x8080808080808080ull; buff[30] = 0x8080800f800c8009ull; buff[31] = 0x8080808080808080ull;    // Puts green colors from v1 in high
	//		buff[32] = 0x80808080800d800aull; buff[33] = 0x8080808080808080ull; buff[34] = 0x80808080800d800aull; buff[35] = 0x8080808080808080ull;    // Puts blue colors from v1 in high
	//
	//		buff[36] = 0x8001808080808080ull; buff[37] = 0x800d800a80078004ull; buff[38] = 0x8001808080808080ull; buff[39] = 0x800d800a80078004ull;    // Puts red colors from v2 in high
	//		buff[40] = 0x8002808080808080ull; buff[41] = 0x800e800b80088005ull; buff[42] = 0x8002808080808080ull; buff[43] = 0x800e800b80088005ull;    // Puts green colors from v2 in high
	//		buff[44] = 0x8003800080808080ull; buff[45] = 0x800f800c80098006ull; buff[46] = 0x8003800080808080ull; buff[47] = 0x800f800c80098006ull;    // Puts blue colors from v2 in high
	//
	//		buff[48] = 0x8004808002808000ull; buff[49] = 0x0a80800880800680ull; buff[50] = 0x8004808002808000ull; buff[51] = 0x0a80800880800680ull;    // Packs hlo-bits into final HSV0-vector
	//		buff[52] = 0x80800e80800c8080ull; buff[53] = 0x8080808080808080ull; buff[54] = 0x80800e80800c8080ull; buff[55] = 0x8080808080808080ull;    // Packs hlo-bits into final HSV1-vector
	//		buff[56] = 0x8080808080808080ull; buff[57] = 0x8004808002808000ull; buff[58] = 0x8080808080808080ull; buff[59] = 0x8004808002808000ull;    // Packs hhi-bits into final HSV1-vector
	//		buff[60] = 0x0a80800880800680ull; buff[61] = 0x80800e80800c8080ull; buff[62] = 0x0a80800880800680ull; buff[63] = 0x80800e80800c8080ull;    // Packs hhi-bits into final HSV2-vector
	//
	//		buff[64] = 0x0480800280800080ull; buff[65] = 0x8080088080068080ull; buff[66] = 0x0480800280800080ull; buff[67] = 0x8080088080068080ull;    // Packs slo-bits into final HSV0-vector
	//		buff[68] = 0x800e80800c80800aull; buff[69] = 0x8080808080808080ull; buff[70] = 0x800e80800c80800aull; buff[71] = 0x8080808080808080ull;    // Packs slo-bits into final HSV1-vector
	//		buff[72] = 0x8080808080808080ull; buff[73] = 0x0480800280800080ull; buff[74] = 0x8080808080808080ull; buff[75] = 0x0480800280800080ull;    // Packs shi-bits into final HSV1-vector
	//		buff[76] = 0x8080088080068080ull; buff[77] = 0x800e80800c80800aull; buff[78] = 0x8080088080068080ull; buff[79] = 0x800e80800c80800aull;    // Packs shi-bits into final HSV2-vector
	//
	//		buff[80] = 0x8080028080008080ull; buff[81] = 0x8008808006808004ull; buff[82] = 0x8080028080008080ull; buff[83] = 0x8008808006808004ull;    // Packs vlo-bits into final HSV0-vector
	//		buff[84] = 0x0e80800c80800a80ull; buff[85] = 0x8080808080808080ull; buff[86] = 0x0e80800c80800a80ull; buff[87] = 0x8080808080808080ull;    // Packs vlo-bits into final HSV1-vector
	//		buff[88] = 0x8080808080808080ull; buff[89] = 0x8080028080008080ull; buff[90] = 0x8080808080808080ull; buff[91] = 0x8080028080008080ull;    // Packs vhi-bits into final HSV1-vector
	//		buff[92] = 0x8008808006808004ull; buff[93] = 0x0e80800c80800a80ull; buff[94] = 0x8008808006808004ull; buff[95] = 0x0e80800c80800a80ull;    // Packs vhi-bits into final HSV2-vector
	//
	//		const __m256i mLowRv0 = _mm256_load_si256((const __m256i*)buff+0);
	//		const __m256i mLowRv1 = _mm256_load_si256((const __m256i*)buff+3);
	//		const __m256i mLowGv0 = _mm256_load_si256((const __m256i*)buff+1);
	//		const __m256i mLowGv1 = _mm256_load_si256((const __m256i*)buff+4);
	//		const __m256i mLowBv0 = _mm256_load_si256((const __m256i*)buff+2);
	//		const __m256i mLowBv1 = _mm256_load_si256((const __m256i*)buff+5);
	//
	//		const __m256i packh_1 = _mm256_load_si256((const __m256i*)buff+12);
	//		const __m256i packh_2 = _mm256_load_si256((const __m256i*)buff+13);
	//		const __m256i packh_3 = _mm256_load_si256((const __m256i*)buff+14);
	//		const __m256i packh_4 = _mm256_load_si256((const __m256i*)buff+15);
	//
	//		const __m256i packs_1 = _mm256_load_si256((const __m256i*)buff+16);
	//		const __m256i packs_2 = _mm256_load_si256((const __m256i*)buff+17);
	//		const __m256i packs_3 = _mm256_load_si256((const __m256i*)buff+18);
	//		const __m256i packs_4 = _mm256_load_si256((const __m256i*)buff+19);
	//
	//		const __m256i packv_1 = _mm256_load_si256((const __m256i*)buff+20);
	//		const __m256i packv_2 = _mm256_load_si256((const __m256i*)buff+21);
	//		const __m256i packv_3 = _mm256_load_si256((const __m256i*)buff+22);
	//		const __m256i packv_4 = _mm256_load_si256((const __m256i*)buff+23);
	//
	//		// Get start positions for buffers
	//		const unsigned char* tsrc  = NULL;
	//		unsigned char* tdest = NULL;
	//
	//		// Constants
	//		const __m256i zeros = _mm256_set1_epi16(0);
	//		const __m256  two55  = _mm256_set1_ps( 255.0f );
	//		const __m256  four25 = _mm256_set1_ps(30.0f);
	//		const __m256i ones  = _mm256_set1_epi8( 0xff );
	//		//const __m128i thirty = _mm_set1_epi16( 30 );
	//
	//		// Start computing Hue
	//		/*
	//		 C = max(r,g,b) - min(r,g,b)
	//		 _H = C==0 ? 0
	//		 = g-b ? max(r,g,b) == r
	//		 = b-r ? max(r,g,b) == g
	//		 = r-g ? max(r,g,b) == b
	//
	//		 _H = _H < 0 ? _H+6C : _H
	//		 H = 255*_H/(6C)
	//
	//		 */
	//		for( int y=_start; y<_stop; ++y ) {
	//			tsrc  = src+(y*pitchs);
	//			tdest = dest+(y*pitchd);
	//			for( int x=0; x<pitchs; x+=3*32 ) {
	//
	//				// Read image
	//				const __m256i v0 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 16;
	//				const __m256i v1 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 16;
	//				const __m256i v2 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 64;
	//
	//				// Final HSV vectors
	//				__m256i HSV0 = zeros;
	//				__m256i HSV1 = zeros;
	//				__m256i HSV2 = zeros;
	//
	//				{
	//
	//					// ======== LOW 8 bytes =======
	//					// Shuffle colors
	//					const __m256i reds_low   = _mm256_or_si256( _mm256_shuffle_epi8(v0,mLowRv0) , _mm256_shuffle_epi8(v1,mLowRv1) );
	//					const __m256i greens_low = _mm256_or_si256( _mm256_shuffle_epi8(v0,mLowGv0) , _mm256_shuffle_epi8(v1,mLowGv1) );
	//					const __m256i blues_low  = _mm256_or_si256( _mm256_shuffle_epi8(v0,mLowBv0) , _mm256_shuffle_epi8(v1,mLowBv1) );
	//
	//					const __m256i maxlow = _mm256_max_epu8( reds_low, _mm256_max_epu8( blues_low, greens_low ) );
	//					const __m256i minlow = _mm256_min_epu8( reds_low, _mm256_min_epu8( blues_low, greens_low ) );
	//
	//					const __m256i Clow = _mm256_sub_epi16(maxlow,minlow); // Unsigned 16-bit integer
	//
	//					const __m256i Clow2 = _mm256_slli_epi16(Clow,1); // Unsigned 16-bit integer
	//					const __m256i Clow4 = _mm256_slli_epi16(Clow,2); // Unsigned 16-bit integer
	//					const __m256i Clow6 = _mm256_add_epi16(Clow2,Clow4); // Unsigned 16-bit integer
	//
	//					const __m256i rmax = _mm256_subs_epi16( greens_low, blues_low  ); // Has Value between -Clow and Clow
	//					const __m256i gmax = _mm256_add_epi16(_mm256_subs_epi16( blues_low , reds_low   ), Clow2); // Has Value between Clow and 3*Clow
	//					const __m256i bmax = _mm256_add_epi16(_mm256_subs_epi16( reds_low  , greens_low ), Clow4); // Has Value between 3*Clow and 5*Clow
	//
	//
	//					// Has values between -Clow and 5*Clow
	//					__m256i Hlowt = _mm256_and_si256(_mm256_cmpeq_epi16(maxlow, reds_low) , rmax);
	//					Hlowt = _mm256_or_si256(Hlowt, _mm256_and_si256( _mm256_andnot_si256(_mm256_cmpeq_epi16(maxlow, reds_low),
	//																						 _mm256_cmpeq_epi16(maxlow, greens_low)),
	//																	gmax ) );
	//					Hlowt = _mm256_or_si256(Hlowt, _mm256_and_si256( _mm256_andnot_si256(_mm256_or_si256(_mm256_cmpeq_epi16(maxlow, reds_low),
	//																										 _mm256_cmpeq_epi16(maxlow, greens_low)),
	//																						 _mm256_cmpeq_epi16(maxlow, blues_low)),
	//																	bmax ) );
	//
	//
	//					// Has values in range [0 .. 6*Clow[
	//					// Note that 0 and 6*Clow is the same color
	//					// The hue could take value 6*Clow, but this is prevented by the order of assignment,
	//					// That is, if Hlow = 0, this means that r=g=max(r,g,b) and b=min(r,g,b)
	//					// For Hlow = 6, then the situation must be the same, but this is, as said above,
	//					// prevented by the order of assignment
	//					__m256i Hlow = _mm256_or_si256(_mm256_and_si256( _mm256_add_epi16( Hlowt, Clow6 ),
	//																	_mm256_cmpgt_epi16( zeros, Hlowt ) ),
	//												   _mm256_and_si256( Hlowt ,
	//																	_mm256_cmpgt_epi16( Hlowt, zeros )) ); // Has values between - and 6*Clow
	//
	//
	//					// To fit between 0_255 every number in Hlow should now be multiplied by 255/(6*Clow) = 85/(2*Clow). Hlow is a number between 0 and 6*255
	//					// To fit between 0_179 every number in Hlow should now be multiplied by 180/(6*Clow) = 30/Clow. Hlow is a number between 0 and 6*255
	//
	//					// Needs conversion to float to perform division - Convert first H to float (needs two float registers)
	//					const __m256i xlo = _mm256_unpacklo_epi16(Hlow, zeros);
	//					const __m256i xhi = _mm256_unpackhi_epi16(Hlow, zeros);
	//					const __m256 xlof = _mm256_cvtepi32_ps(xlo);
	//					const __m256 xhif = _mm256_cvtepi32_ps(xhi);
	//
	//					// Then convert C to float (needs two float registers)
	//					const __m256i clo = _mm256_unpacklo_epi16(Clow, zeros);
	//					const __m256i chi = _mm256_unpackhi_epi16(Clow, zeros);
	//					const __m256 clof = _mm256_cvtepi32_ps(clo);
	//					const __m256 chif = _mm256_cvtepi32_ps(chi);
	//
	//					// Multiply H with factor depending on C
	//					const __m256 factorlo = _mm256_div_ps( four25 , clof );
	//					const __m256 factorhi = _mm256_div_ps( four25 , chif );
	//					const __m256 xlof_div = _mm256_mul_ps(xlof,factorlo);
	//					const __m256 xhif_div = _mm256_mul_ps(xhif,factorhi);
	//
	//					// Finally convert back to integer
	//					const __m256i HUE_low = _mm256_packus_epi16(_mm256_cvtps_epi32(xlof_div),
	//																_mm256_cvtps_epi32(xhif_div));
	//
	//					// HUE
	//					HSV0 = _mm256_shuffle_epi8( HUE_low, packh_1 );
	//					HSV1 = _mm256_shuffle_epi8( HUE_low, packh_2 );
	//
	//
	//					// VALUE
	//					HSV0 = _mm256_or_si256( HSV0, _mm256_shuffle_epi8(maxlow, packv_1));
	//					HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8(maxlow, packv_2));
	//
	//
	//					// SATURATION
	//					// Depends on C and V - first convert V to
	//					const __m256i mlo = _mm256_unpacklo_epi16(maxlow, zeros);
	//					const __m256i mhi = _mm256_unpackhi_epi16(maxlow, zeros);
	//					const __m256 mlof = _mm256_cvtepi32_ps(mlo);
	//					const __m256 mhif = _mm256_cvtepi32_ps(mhi);
	//
	//					// Perform C/V
	//					const __m256 slof = _mm256_mul_ps( _mm256_div_ps( clof, mlof ), two55 );
	//					const __m256 shif = _mm256_mul_ps( _mm256_div_ps( chif, mhif ), two55 );
	//					const __m256i slo = _mm256_cvtps_epi32(slof);
	//					const __m256i shi = _mm256_cvtps_epi32(shif);
	//					const __m256i SAT_low = _mm256_and_si256( _mm256_xor_si256(_mm256_cmpeq_epi16(Clow,zeros), // Finds C==0
	//																			   ones), // Flips bits, i.e. C!=0
	//															 _mm256_packus_epi16(slo, shi)); // Selects SAT s.t. C!=0
	//
	//					HSV0 = _mm256_or_si256( HSV0, _mm256_shuffle_epi8(SAT_low, packs_1));
	//					HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8(SAT_low, packs_2));
	//				}
	//
	//				{
	//					const __m256i mHighRv0 = _mm256_load_si256( (const __m256i*)buff+6 );
	//					const __m256i mHighRv1 = _mm256_load_si256( (const __m256i*)buff+9 );
	//					const __m256i mHighGv0 = _mm256_load_si256( (const __m256i*)buff+7 );
	//					const __m256i mHighGv1 = _mm256_load_si256( (const __m256i*)buff+10);
	//					const __m256i mHighBv0 = _mm256_load_si256( (const __m256i*)buff+8 );
	//					const __m256i mHighBv1 = _mm256_load_si256( (const __m256i*)buff+11);
	//
	//					// ======== HIGH 8 bytes =======
	//					// Shuffle colors
	//					const __m256i reds_high   = _mm256_or_si256( _mm256_shuffle_epi8(v1,mHighRv0) , _mm256_shuffle_epi8(v2,mHighRv1) );
	//					const __m256i greens_high = _mm256_or_si256( _mm256_shuffle_epi8(v1,mHighGv0) , _mm256_shuffle_epi8(v2,mHighGv1) );
	//					const __m256i blues_high  = _mm256_or_si256( _mm256_shuffle_epi8(v1,mHighBv0) , _mm256_shuffle_epi8(v2,mHighBv1) );
	//
	//					const __m256i maxhigh = _mm256_max_epu8( reds_high, _mm256_max_epu8( blues_high, greens_high ) );
	//					const __m256i minhigh = _mm256_min_epu8( reds_high, _mm256_min_epu8( blues_high, greens_high ) );
	//
	//					const __m256i Chigh = _mm256_sub_epi16(maxhigh,minhigh); // Unsigned 16-bit integer
	//
	//					const __m256i Chigh2 = _mm256_slli_epi16(Chigh,1); // Unsigned 16-bit integer
	//					const __m256i Chigh4 = _mm256_slli_epi16(Chigh,2); // Unsigned 16-bit integer
	//					const __m256i Chigh6 = _mm256_add_epi16(Chigh2,Chigh4); // Unsigned 16-bit integer
	//
	//					const __m256i rmaxh = _mm256_subs_epi16( greens_high, blues_high  ); // Has Value between -Chigh and Chigh
	//					const __m256i gmaxh = _mm256_add_epi16( _mm256_subs_epi16( blues_high , reds_high   ), Chigh2); // Has Value between Chigh and 3*Chigh
	//					const __m256i bmaxh = _mm256_add_epi16( _mm256_subs_epi16( reds_high  , greens_high ), Chigh4); // Has Value between 3*Chigh and 5*Chigh
	//
	//
	//					// Has values in range [-Chigh .. 5*Chigh[
	//					__m256i Hhight = _mm256_and_si256(_mm256_cmpeq_epi16(maxhigh, reds_high) , rmaxh);
	//					Hhight = _mm256_or_si256(Hhight, _mm256_and_si256 (_mm256_andnot_si256(_mm256_cmpeq_epi16(maxhigh, reds_high),
	//																						   _mm256_cmpeq_epi16(maxhigh, greens_high)),
	//																	   gmaxh ) );
	//					Hhight = _mm256_or_si256(Hhight, _mm256_and_si256 (_mm256_andnot_si256(_mm256_or_si256(_mm256_cmpeq_epi16(maxhigh, reds_high),
	//																										   _mm256_cmpeq_epi16(maxhigh, greens_high)),
	//																						   _mm256_cmpeq_epi16(maxhigh, blues_high)),
	//																	   bmaxh ) );
	//
	//					// Has values in range [0 .. 6*Chigh[
	//					// Note that 0 and 6*Chigh is the same color
	//					// The hue could take value 6*Chigh, but this is prevented by the order of assignment,
	//					// That is, if Hhigh = 0, this means that r=g=max(r,g,b) and b=min(r,g,b)
	//					// For Hhigh = 6, then the situation must be the same, but this is, as said above,
	//					// prevented by the order of assignment
	//					__m256i Hhigh = _mm256_or_si256(_mm256_and_si256( _mm256_add_epi16( Hhight, Chigh6 ),
	//																	 _mm256_cmpgt_epi16( zeros, Hhight ) ),
	//													_mm256_and_si256( Hhight ,
	//																	 _mm256_cmpgt_epi16( Hhight, zeros )) );
	//
	//
	//					// To fit between 0_255 every number in Hhigh should now be multiplied by 255/(6*Chigh) = 85/(2*Chigh). Hhigh is a number between 0 and 6*255
	//
	//					// Needs conversion to float to perform division - Convert first H to float (needs two float registers)
	//					const __m256i xloh = _mm256_unpacklo_epi16(Hhigh, zeros );
	//					const __m256i xhih = _mm256_unpackhi_epi16(Hhigh, zeros );
	//					const __m256 xlohf = _mm256_cvtepi32_ps(xloh);
	//					const __m256 xhihf = _mm256_cvtepi32_ps(xhih);
	//
	//					// Then convert C to float (needs two float registers)
	//					const __m256i cloh = _mm256_unpacklo_epi16(Chigh, zeros);
	//					const __m256i chih = _mm256_unpackhi_epi16(Chigh, zeros);
	//					const __m256 clofh = _mm256_cvtepi32_ps(cloh);
	//					const __m256 chifh = _mm256_cvtepi32_ps(chih);
	//
	//					// Multiply H with factor depending on C
	//					const __m256 factorloh = _mm256_div_ps( four25 , clofh );
	//					const __m256 factorhih = _mm256_div_ps( four25 , chifh );
	//					const __m256 xlofh_div = _mm256_mul_ps(xlohf,factorloh);
	//					const __m256 xhifh_div = _mm256_mul_ps(xhihf,factorhih);
	//
	//					// Finally convert back to integer
	//					const __m256i HUE_high = _mm256_packus_epi16(_mm256_cvtps_epi32(xlofh_div),
	//																 _mm256_cvtps_epi32(xhifh_div));
	//
	//					// HUE
	//					HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8( HUE_high, packh_3 ) );
	//					HSV2 = _mm256_shuffle_epi8( HUE_high, packh_4 ); // FINAL
	//
	//
	//					// VALUE
	//					HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8(maxhigh, packv_3));
	//					HSV2 = _mm256_or_si256( HSV2, _mm256_shuffle_epi8(maxhigh, packv_4));
	//
	//
	//					// SATURATION
	//					// Depends on C and V - first convert V to
	//					const __m256i mloh = _mm256_unpacklo_epi16(maxhigh, zeros);
	//					const __m256i mhih = _mm256_unpackhi_epi16(maxhigh, zeros);
	//					const __m256 mlofh = _mm256_cvtepi32_ps(mloh);
	//					const __m256 mhifh = _mm256_cvtepi32_ps(mhih);
	//
	//					// Perform C/V
	//					const __m256 slofh = _mm256_mul_ps( _mm256_div_ps( clofh, mlofh ), two55 );
	//					const __m256 shifh = _mm256_mul_ps( _mm256_div_ps( chifh, mhifh ), two55 );
	//					const __m256i sloh = _mm256_cvtps_epi32(slofh);
	//					const __m256i shih = _mm256_cvtps_epi32(shifh);
	//					const __m256i SAT_high = _mm256_and_si256( _mm256_xor_si256(_mm256_cmpeq_epi16(Chigh,zeros), // Finds C==0
	//																				ones), // Flips bits, i.e. C!=0
	//															  _mm256_packus_epi16(sloh, shih)); // Selects SAT s.t. C!=0
	//
	//					HSV1 = _mm256_or_si256( HSV1, _mm256_shuffle_epi8(SAT_high, packs_3));
	//					HSV2 = _mm256_or_si256( HSV2, _mm256_shuffle_epi8(SAT_high, packs_4));
	//				}
	//
	//				// STORE IN DEST
	//				_mm256_storeu2_m128i( (__m128i*)tdest+3, (__m128i*)tdest, HSV0 ); tdest += 16;
	//				_mm256_storeu2_m128i( (__m128i*)tdest+3, (__m128i*)tdest, HSV1 ); tdest += 16;
	//				_mm256_storeu2_m128i( (__m128i*)tdest+3, (__m128i*)tdest, HSV2 ); tdest += 64;
	//
	//			}
	//		}
	//	}
	//
	//#else
	//	ALIGNED_BUFFER(unsigned long long, buff, 48);
	//
	//	buff[0]  = 0x8009800680038000ull; buff[1]  = 0x80808080800f800cull;    // Puts red colors from v0 in low
	//	buff[4]  = 0x800a800780048001ull; buff[5]  = 0x808080808080800dull;    // Puts green colors from v0 in low
	//	buff[8]  = 0x800b800880058002ull; buff[9]  = 0x808080808080800eull;    // Puts blue colors from v0 in low
	//
	//	buff[2]  = 0x8080808080808080ull; buff[3]  = 0x8005800280808080ull;    // Puts red colors from v1 in low
	//	buff[6]  = 0x8080808080808080ull; buff[7]  = 0x8006800380008080ull;    // Puts green colors from v1 in low
	//	buff[10] = 0x8080808080808080ull; buff[11] = 0x8007800480018080ull;    // Puts blue colors from v1 in low
	//
	//	buff[12] = 0x8080800e800b8008ull; buff[13] = 0x8080808080808080ull;    // Puts red colors from v1 in high
	//	buff[16] = 0x8080800f800c8009ull; buff[17] = 0x8080808080808080ull;    // Puts green colors from v1 in high
	//	buff[20] = 0x80808080800d800aull; buff[21] = 0x8080808080808080ull;    // Puts blue colors from v1 in high
	//
	//	buff[14] = 0x8001808080808080ull; buff[15] = 0x800d800a80078004ull;    // Puts red colors from v2 in high
	//	buff[18] = 0x8002808080808080ull; buff[19] = 0x800e800b80088005ull;    // Puts green colors from v2 in high
	//	buff[22] = 0x8003800080808080ull; buff[23] = 0x800f800c80098006ull;    // Puts blue colors from v2 in high
	//
	//	buff[24] = 0x8004808002808000ull; buff[25] = 0x0a80800880800680ull;    // Packs hlo-bits into final HSV0-vector
	//	buff[26] = 0x80800e80800c8080ull; buff[27] = 0x8080808080808080ull;    // Packs hlo-bits into final HSV1-vector
	//	buff[28] = 0x8080808080808080ull; buff[29] = 0x8004808002808000ull;    // Packs hhi-bits into final HSV1-vector
	//	buff[30] = 0x0a80800880800680ull; buff[31] = 0x80800e80800c8080ull;    // Packs hhi-bits into final HSV2-vector
	//
	//	buff[32] = 0x0480800280800080ull; buff[33] = 0x8080088080068080ull;    // Packs slo-bits into final HSV0-vector
	//	buff[34] = 0x800e80800c80800aull; buff[35] = 0x8080808080808080ull;    // Packs slo-bits into final HSV1-vector
	//	buff[36] = 0x8080808080808080ull; buff[37] = 0x0480800280800080ull;    // Packs shi-bits into final HSV1-vector
	//	buff[38] = 0x8080088080068080ull; buff[39] = 0x800e80800c80800aull;    // Packs shi-bits into final HSV2-vector
	//
	//	buff[40] = 0x8080028080008080ull; buff[41] = 0x8008808006808004ull;    // Packs vlo-bits into final HSV0-vector
	//	buff[42] = 0x0e80800c80800a80ull; buff[43] = 0x8080808080808080ull;    // Packs vlo-bits into final HSV1-vector
	//	buff[44] = 0x8080808080808080ull; buff[45] = 0x8080028080008080ull;    // Packs vhi-bits into final HSV1-vector
	//	buff[46] = 0x8008808006808004ull; buff[47] = 0x0e80800c80800a80ull;    // Packs vhi-bits into final HSV2-vector
	//
	//	// Read masks
	//	const __m128i mLowRv0 = _mm_load_si128((const __m128i*)buff+0);
	//	const __m128i mLowRv1 = _mm_load_si128((const __m128i*)buff+1);
	//	const __m128i mLowGv0 = _mm_load_si128((const __m128i*)buff+2);
	//	const __m128i mLowGv1 = _mm_load_si128((const __m128i*)buff+3);
	//	const __m128i mLowBv0 = _mm_load_si128((const __m128i*)buff+4);
	//	const __m128i mLowBv1 = _mm_load_si128((const __m128i*)buff+5);
	//
	//	const __m128i packh_1 = _mm_load_si128((const __m128i*)buff+12);
	//	const __m128i packh_2 = _mm_load_si128((const __m128i*)buff+13);
	//	const __m128i packh_3 = _mm_load_si128((const __m128i*)buff+14);
	//	const __m128i packh_4 = _mm_load_si128((const __m128i*)buff+15);
	//
	//	const __m128i packs_1 = _mm_load_si128((const __m128i*)buff+16);
	//	const __m128i packs_2 = _mm_load_si128((const __m128i*)buff+17);
	//	const __m128i packs_3 = _mm_load_si128((const __m128i*)buff+18);
	//	const __m128i packs_4 = _mm_load_si128((const __m128i*)buff+19);
	//
	//	const __m128i packv_1 = _mm_load_si128((const __m128i*)buff+20);
	//	const __m128i packv_2 = _mm_load_si128((const __m128i*)buff+21);
	//	const __m128i packv_3 = _mm_load_si128((const __m128i*)buff+22);
	//	const __m128i packv_4 = _mm_load_si128((const __m128i*)buff+23);
	//
	//	// This is the number of 3*16 blocks assigned to the thread
	//	const unsigned int widthz = (pitchs/3) >> 4;
	//
	//	// Get start positions for buffers
	//	const unsigned char* tsrc  = src+(_start*pitchs);
	//	unsigned char* tdest = dest+(_start*pitchd);
	//
	//	// Constants
	//	const __m128i zeros = _mm_set1_epi16(0);
	//	const __m128  two55  = _mm_set1_ps( 255.0f );
	//	const __m128  four25 = _mm_set1_ps(30.0f);
	//	const __m128i ones  = _mm_set1_epi8( 0xff );
	//	//const __m128i thirty = _mm_set1_epi16( 30 );
	//
	//	// Start computing Hue
	//	/*
	//	 C = max(r,g,b) - min(r,g,b)
	//	 _H = C==0 ? 0
	//	 = g-b ? max(r,g,b) == r
	//	 = b-r ? max(r,g,b) == g
	//	 = r-g ? max(r,g,b) == b
	//
	//	 _H = _H < 0 ? _H+6C : _H
	//	 H = 255*_H/(6C)
	//
	//	 */
	//	for( int y=_start; y<_stop; ++y ) {
	//		tsrc  = src+(y*pitchs);
	//		tdest = dest+(y*pitchd);
	//		for( int x=0; x<widthz; ++x ) {
	//
	//			// Read image
	//			const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
	//			const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
	//			const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
	//
	//			// Final HSV vectors
	//			__m128i HSV0 = zeros;
	//			__m128i HSV1 = zeros;
	//			__m128i HSV2 = zeros;
	//
	//			{
	//
	//				// ======== LOW 8 bytes =======
	//				// Shuffle colors
	//				const __m128i reds_low   = _mm_or_si128( _mm_shuffle_epi8(v0,mLowRv0) , _mm_shuffle_epi8(v1,mLowRv1) );
	//				const __m128i greens_low = _mm_or_si128( _mm_shuffle_epi8(v0,mLowGv0) , _mm_shuffle_epi8(v1,mLowGv1) );
	//				const __m128i blues_low  = _mm_or_si128( _mm_shuffle_epi8(v0,mLowBv0) , _mm_shuffle_epi8(v1,mLowBv1) );
	//
	//				const __m128i maxlow = _mm_max_epu8( reds_low, _mm_max_epu8( blues_low, greens_low ) );
	//				const __m128i minlow = _mm_min_epu8( reds_low, _mm_min_epu8( blues_low, greens_low ) );
	//
	//				const __m128i Clow = _mm_sub_epi16(maxlow,minlow); // Unsigned 16-bit integer
	//
	//				const __m128i Clow2 = _mm_slli_epi16(Clow,1); // Unsigned 16-bit integer
	//				const __m128i Clow4 = _mm_slli_epi16(Clow,2); // Unsigned 16-bit integer
	//				const __m128i Clow6 = _mm_add_epi16(Clow2,Clow4); // Unsigned 16-bit integer
	//
	//				const __m128i rmax = _mm_subs_epi16( greens_low, blues_low  ); // Has Value between -Clow and Clow
	//				const __m128i gmax = _mm_add_epi16(_mm_subs_epi16( blues_low , reds_low   ), Clow2); // Has Value between Clow and 3*Clow
	//				const __m128i bmax = _mm_add_epi16(_mm_subs_epi16( reds_low  , greens_low ), Clow4); // Has Value between 3*Clow and 5*Clow
	//
	//
	//				// Has values between -Clow and 5*Clow
	//				__m128i Hlowt = _mm_and_si128(_mm_cmpeq_epi16(maxlow, reds_low) , rmax);
	//				Hlowt = _mm_or_si128(Hlowt, _mm_and_si128( _mm_andnot_si128(_mm_cmpeq_epi16(maxlow, reds_low),
	//																			_mm_cmpeq_epi16(maxlow, greens_low)),
	//														  gmax ) );
	//				Hlowt = _mm_or_si128(Hlowt, _mm_and_si128( _mm_andnot_si128(_mm_or_si128(_mm_cmpeq_epi16(maxlow, reds_low),
	//																						 _mm_cmpeq_epi16(maxlow, greens_low)),
	//																			_mm_cmpeq_epi16(maxlow, blues_low)),
	//														  bmax ) );
	//
	//
	//				// Has values in range [0 .. 6*Clow[
	//				// Note that 0 and 6*Clow is the same color
	//				// The hue could take value 6*Clow, but this is prevented by the order of assignment,
	//				// That is, if Hlow = 0, this means that r=g=max(r,g,b) and b=min(r,g,b)
	//				// For Hlow = 6, then the situation must be the same, but this is, as said above,
	//				// prevented by the order of assignment
	//				__m128i Hlow = _mm_or_si128(_mm_and_si128( _mm_add_epi16( Hlowt, Clow6 ),
	//														  _mm_cmpgt_epi16( zeros, Hlowt ) ),
	//											_mm_and_si128( Hlowt ,
	//														  _mm_cmpgt_epi16( Hlowt, zeros )) ); // Has values between - and 6*Clow
	//
	//
	//				// To fit between 0_255 every number in Hlow should now be multiplied by 255/(6*Clow) = 85/(2*Clow). Hlow is a number between 0 and 6*255
	//				// To fit between 0_179 every number in Hlow should now be multiplied by 180/(6*Clow) = 30/Clow. Hlow is a number between 0 and 6*255
	//
	//				// Needs conversion to float to perform division - Convert first H to float (needs two float registers)
	//				const __m128i xlo = _mm_unpacklo_epi16(Hlow, zeros);
	//				const __m128i xhi = _mm_unpackhi_epi16(Hlow, zeros);
	//				const __m128 xlof = _mm_cvtepi32_ps(xlo);
	//				const __m128 xhif = _mm_cvtepi32_ps(xhi);
	//
	//				// Then convert C to float (needs two float registers)
	//				const __m128i clo = _mm_unpacklo_epi16(Clow, zeros);
	//				const __m128i chi = _mm_unpackhi_epi16(Clow, zeros);
	//				const __m128 clof = _mm_cvtepi32_ps(clo);
	//				const __m128 chif = _mm_cvtepi32_ps(chi);
	//
	//				// Multiply H with factor depending on C
	//				const __m128 factorlo = _mm_div_ps( four25 , clof );
	//				const __m128 factorhi = _mm_div_ps( four25 , chif );
	//				const __m128 xlof_div = _mm_mul_ps(xlof,factorlo);
	//				const __m128 xhif_div = _mm_mul_ps(xhif,factorhi);
	//
	//				// Finally convert back to integer
	//				const __m128i HUE_low = _mm_packus_epi16(_mm_cvtps_epi32(xlof_div),
	//														 _mm_cvtps_epi32(xhif_div));
	//
	//				// HUE
	//				HSV0 = _mm_shuffle_epi8( HUE_low, packh_1 );
	//				HSV1 = _mm_shuffle_epi8( HUE_low, packh_2 );
	//
	//
	//				// VALUE
	//				HSV0 = _mm_or_si128( HSV0, _mm_shuffle_epi8(maxlow, packv_1));
	//				HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8(maxlow, packv_2));
	//
	//
	//				// SATURATION
	//				// Depends on C and V - first convert V to
	//				const __m128i mlo = _mm_unpacklo_epi16(maxlow, zeros);
	//				const __m128i mhi = _mm_unpackhi_epi16(maxlow, zeros);
	//				const __m128 mlof = _mm_cvtepi32_ps(mlo);
	//				const __m128 mhif = _mm_cvtepi32_ps(mhi);
	//
	//				// Perform C/V
	//				const __m128 slof = _mm_mul_ps( _mm_div_ps( clof, mlof ), two55 );
	//				const __m128 shif = _mm_mul_ps( _mm_div_ps( chif, mhif ), two55 );
	//				const __m128i slo = _mm_cvtps_epi32(slof);
	//				const __m128i shi = _mm_cvtps_epi32(shif);
	//				const __m128i SAT_low = _mm_and_si128( _mm_xor_si128(_mm_cmpeq_epi16(Clow,zeros), // Finds C==0
	//																	 ones), // Flips bits, i.e. C!=0
	//													  _mm_packus_epi16(slo, shi)); // Selects SAT s.t. C!=0
	//
	//				HSV0 = _mm_or_si128( HSV0, _mm_shuffle_epi8(SAT_low, packs_1));
	//				HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8(SAT_low, packs_2));
	//			}
	//
	//			{
	//				const __m128i mHighRv0 = _mm_load_si128( (const __m128i*)buff+6 );
	//				const __m128i mHighRv1 = _mm_load_si128( (const __m128i*)buff+7 );
	//				const __m128i mHighGv0 = _mm_load_si128( (const __m128i*)buff+8 );
	//				const __m128i mHighGv1 = _mm_load_si128( (const __m128i*)buff+9 );
	//				const __m128i mHighBv0 = _mm_load_si128( (const __m128i*)buff+10);
	//				const __m128i mHighBv1 = _mm_load_si128( (const __m128i*)buff+11);
	//
	//				// ======== HIGH 8 bytes =======
	//				// Shuffle colors
	//				const __m128i reds_high   = _mm_or_si128( _mm_shuffle_epi8(v1,mHighRv0) , _mm_shuffle_epi8(v2,mHighRv1) );
	//				const __m128i greens_high = _mm_or_si128( _mm_shuffle_epi8(v1,mHighGv0) , _mm_shuffle_epi8(v2,mHighGv1) );
	//				const __m128i blues_high  = _mm_or_si128( _mm_shuffle_epi8(v1,mHighBv0) , _mm_shuffle_epi8(v2,mHighBv1) );
	//
	//				const __m128i maxhigh = _mm_max_epu8( reds_high, _mm_max_epu8( blues_high, greens_high ) );
	//				const __m128i minhigh = _mm_min_epu8( reds_high, _mm_min_epu8( blues_high, greens_high ) );
	//
	//				const __m128i Chigh = _mm_sub_epi16(maxhigh,minhigh); // Unsigned 16-bit integer
	//
	//				const __m128i Chigh2 = _mm_slli_epi16(Chigh,1); // Unsigned 16-bit integer
	//				const __m128i Chigh4 = _mm_slli_epi16(Chigh,2); // Unsigned 16-bit integer
	//				const __m128i Chigh6 = _mm_add_epi16(Chigh2,Chigh4); // Unsigned 16-bit integer
	//
	//				const __m128i rmaxh = _mm_subs_epi16( greens_high, blues_high  ); // Has Value between -Chigh and Chigh
	//				const __m128i gmaxh = _mm_add_epi16( _mm_subs_epi16( blues_high , reds_high   ), Chigh2); // Has Value between Chigh and 3*Chigh
	//				const __m128i bmaxh = _mm_add_epi16( _mm_subs_epi16( reds_high  , greens_high ), Chigh4); // Has Value between 3*Chigh and 5*Chigh
	//
	//
	//				// Has values in range [-Chigh .. 5*Chigh[
	//				__m128i Hhight = _mm_and_si128(_mm_cmpeq_epi16(maxhigh, reds_high) , rmaxh);
	//				Hhight = _mm_or_si128(Hhight, _mm_and_si128 (_mm_andnot_si128(_mm_cmpeq_epi16(maxhigh, reds_high),
	//																			  _mm_cmpeq_epi16(maxhigh, greens_high)),
	//															 gmaxh ) );
	//				Hhight = _mm_or_si128(Hhight, _mm_and_si128 (_mm_andnot_si128(_mm_or_si128(_mm_cmpeq_epi16(maxhigh, reds_high),
	//																						   _mm_cmpeq_epi16(maxhigh, greens_high)),
	//																			  _mm_cmpeq_epi16(maxhigh, blues_high)),
	//															 bmaxh ) );
	//
	//				// Has values in range [0 .. 6*Chigh[
	//				// Note that 0 and 6*Chigh is the same color
	//				// The hue could take value 6*Chigh, but this is prevented by the order of assignment,
	//				// That is, if Hhigh = 0, this means that r=g=max(r,g,b) and b=min(r,g,b)
	//				// For Hhigh = 6, then the situation must be the same, but this is, as said above,
	//				// prevented by the order of assignment
	//				__m128i Hhigh = _mm_or_si128(_mm_and_si128( _mm_add_epi16( Hhight, Chigh6 ),
	//														   _mm_cmpgt_epi16( zeros, Hhight ) ),
	//											 _mm_and_si128( Hhight ,
	//														   _mm_cmpgt_epi16( Hhight, zeros )) );
	//
	//
	//				// To fit between 0_255 every number in Hhigh should now be multiplied by 255/(6*Chigh) = 85/(2*Chigh). Hhigh is a number between 0 and 6*255
	//
	//				// Needs conversion to float to perform division - Convert first H to float (needs two float registers)
	//				const __m128i xloh = _mm_unpacklo_epi16(Hhigh, zeros );
	//				const __m128i xhih = _mm_unpackhi_epi16(Hhigh, zeros );
	//				const __m128 xlohf = _mm_cvtepi32_ps(xloh);
	//				const __m128 xhihf = _mm_cvtepi32_ps(xhih);
	//
	//				// Then convert C to float (needs two float registers)
	//				const __m128i cloh = _mm_unpacklo_epi16(Chigh, zeros);
	//				const __m128i chih = _mm_unpackhi_epi16(Chigh, zeros);
	//				const __m128 clofh = _mm_cvtepi32_ps(cloh);
	//				const __m128 chifh = _mm_cvtepi32_ps(chih);
	//
	//				// Multiply H with factor depending on C
	//				const __m128 factorloh = _mm_div_ps( four25 , clofh );
	//				const __m128 factorhih = _mm_div_ps( four25 , chifh );
	//				const __m128 xlofh_div = _mm_mul_ps(xlohf,factorloh);
	//				const __m128 xhifh_div = _mm_mul_ps(xhihf,factorhih);
	//
	//				// Finally convert back to integer
	//				const __m128i HUE_high = _mm_packus_epi16(_mm_cvtps_epi32(xlofh_div),
	//														  _mm_cvtps_epi32(xhifh_div));
	//
	//				// HUE
	//				HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8( HUE_high, packh_3 ) );
	//				HSV2 = _mm_shuffle_epi8( HUE_high, packh_4 ); // FINAL
	//
	//
	//				// VALUE
	//				HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8(maxhigh, packv_3));
	//				HSV2 = _mm_or_si128( HSV2, _mm_shuffle_epi8(maxhigh, packv_4));
	//
	//
	//				// SATURATION
	//				// Depends on C and V - first convert V to
	//				const __m128i mloh = _mm_unpacklo_epi16(maxhigh, zeros);
	//				const __m128i mhih = _mm_unpackhi_epi16(maxhigh, zeros);
	//				const __m128 mlofh = _mm_cvtepi32_ps(mloh);
	//				const __m128 mhifh = _mm_cvtepi32_ps(mhih);
	//
	//				// Perform C/V
	//				const __m128 slofh = _mm_mul_ps( _mm_div_ps( clofh, mlofh ), two55 );
	//				const __m128 shifh = _mm_mul_ps( _mm_div_ps( chifh, mhifh ), two55 );
	//				const __m128i sloh = _mm_cvtps_epi32(slofh);
	//				const __m128i shih = _mm_cvtps_epi32(shifh);
	//				const __m128i SAT_high = _mm_and_si128( _mm_xor_si128(_mm_cmpeq_epi16(Chigh,zeros), // Finds C==0
	//																	  ones), // Flips bits, i.e. C!=0
	//													   _mm_packus_epi16(sloh, shih)); // Selects SAT s.t. C!=0
	//
	//				HSV1 = _mm_or_si128( HSV1, _mm_shuffle_epi8(SAT_high, packs_3));
	//				HSV2 = _mm_or_si128( HSV2, _mm_shuffle_epi8(SAT_high, packs_4));
	//			}
	//
	//			// STORE IN DEST
	//
	//			_mm_store_si128( (__m128i*)tdest, HSV0 ); tdest += 16;
	//			_mm_store_si128( (__m128i*)tdest, HSV1 ); tdest += 16;
	//			_mm_store_si128( (__m128i*)tdest, HSV2 ); tdest += 16;
	//
	//		}
	//	}
	//#endif
	//}
	//
	//
	//
	//
	//void HSV2RGB_8U_TBB::_hsv2rgb_8u(unsigned int _start,
	//								 unsigned int _stop) const {
	//
	//
	//	// Get start positions for buffers
	//	const unsigned char* tsrc;
	//	unsigned char* tdest;
	//
	//	for( int y=_start; y<_stop; ++y ) {
	//		tsrc = src+(y*pitchs);
	//		tdest = dest+(y*pitchd);
	//		for( int x=0; x<width; ++x ) {
	//
	//			// _src has value in range [0 .. 255] corresponding to degrees [0 .. 359]
	//			// These should be put in a range of [0..5]
	//			float h_ = ((float)*tsrc)/30.0f; ++tsrc;
	//			// s and v should take values in range [0 .. 1]
	//			float s = ((float)*tsrc)/255.0f; ++tsrc;
	//			float v = ((float)*tsrc)/255.0f; ++tsrc;
	//			
	//			// wiki:
	//			// c = v*s
	//			// x = c(1 - abs(h'%2-1)) \in {0,c} -> x = c or 0
	//			
	//			int switchi = (int)h_;
	//			double f = h_ - switchi;
	//			double pv = v * (1 - s);
	//			double qv = v * (1 - s * f);
	//			double tv = v * (1 - s * (1 - f));
	//			
	//			//float c = v*s;
	//			//float x = c*(1-abs(((int)h_)%2-1));
	//			
	//			if( v == 0 ) {
	//				*tdest = 0.0f; ++tdest;
	//				*tdest = 0.0f; ++tdest;
	//				*tdest = 0.0f; ++tdest;
	//				continue;
	//			}
	//			if( s == 0 ) {
	//				*tdest = 255.0f*v; ++tdest;
	//				*tdest = 255.0f*v; ++tdest;
	//				*tdest = 255.0f*v; ++tdest;
	//				continue;
	//			}
	//			
	//			float r = 0.0f;
	//			float g = 0.0f;
	//			float b = 0.0f;
	//			switch( switchi ) {
	//				case 0:
	//					r = v;
	//					g = tv;
	//					b = pv;
	//					break;
	//				case 1:
	//					r = qv;
	//					g = v;
	//					b = pv;
	//					break;
	//				case 2:
	//					r = pv;
	//					g = v;
	//					b = tv;
	//					break;
	//				case 3:
	//					r = pv;
	//					g = qv;
	//					b = v;
	//					break;
	//				case 4:
	//					r = tv;
	//					g = pv;
	//					b = v;
	//					break;
	//				case 5:
	//					r = v;
	//					g = pv;
	//					b = qv;
	//					break;
	//			}
	//			
	//			*tdest = (unsigned char)255.0f*r; ++tdest;
	//			*tdest = (unsigned char)255.0f*g; ++tdest;
	//			*tdest = (unsigned char)255.0f*b; ++tdest;
	//			
	//		}
	//	}
	//	
	//	
	//}
	
	
} // namespace pvcore