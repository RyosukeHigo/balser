#include "stdafx.h"
#include "calcCOG.h"

void CALC_COG_TBB::_calc_cog(unsigned int _start, unsigned int _stop){
	int i;
	//	オリジナルor改訂版
#if 0
#if 0
	for (int y = _start; y <=_stop; y++){
		for (int x = 0; x < width; x++){
			i = width*y+x;
			if (src[i] > 250){
				mom0++;
				mom10 += (i%width);
				mom01 += (i / width);
			}
		}
	}
#else
	int out;
	int mom_0;

	const __m128i _true = _mm_set1_epi8(1);
	const __m128i _false = _mm_set1_epi8(0);
	__m128i _mom0;
	__m128i _mom10_1, _mom10_2, _mom10_3, _mom10_4;
	__m128i _mom01_1, _mom01_2, _mom01_3, _mom01_4;


	for (int y = _start; y <= _stop; y++){
		const __m128i yPos = _mm_set1_epi32(y);

		for (int x = 0; x < width; x += 16){
			mom_0 = 0;
			i = width*y + x;
			const __m128i xPos1 = _mm_set_epi32(x + 3, x + 2, x + 1, x);
			const __m128i xPos2 = _mm_set_epi32(x + 7, x + 6, x + 5, x + 4);
			const __m128i xPos3 = _mm_set_epi32(x + 11, x + 10, x + 9, x + 8);
			const __m128i xPos4 = _mm_set_epi32(x + 15, x + 14, x + 13, x + 12);

			//Load image
			const __m128i srcData = _mm_load_si128((const __m128i*) (src + i));

			//mom0
			//Mask image (Calc mom0) 255->1, 0->0
			_mom0 = _mm_blendv_epi8(_false, _true, srcData);

			//Output
			for (int i = 0; i < 16; i++){
				mom_0 += _mom0.m128i_u8[i];
			}

			//Divide 8bit -> 32bit convert
			__m128i m1 = _mm_cvtepu8_epi32(_mom0);
			_mom0 = _mm_srli_si128(_mom0, 4);
			__m128i m2 = _mm_cvtepu8_epi32(_mom0);
			_mom0 = _mm_srli_si128(_mom0, 4);
			__m128i m3 = _mm_cvtepu8_epi32(_mom0);
			_mom0 = _mm_srli_si128(_mom0, 4);
			__m128i m4 = _mm_cvtepu8_epi32(_mom0);

			//Calc mom10
			_mom10_1 = _mm_mullo_epi32(m1, xPos1);
			_mom10_2 = _mm_mullo_epi32(m2, xPos2);
			_mom10_3 = _mm_mullo_epi32(m3, xPos3);
			_mom10_4 = _mm_mullo_epi32(m4, xPos4);

			_mom10_1 = _mm_add_epi32(_mom10_1, _mom10_2);
			_mom10_3 = _mm_add_epi32(_mom10_3, _mom10_4);
			_mom10_1 = _mm_add_epi32(_mom10_1, _mom10_3);

			_mom10_1 = _mm_hadd_epi32(_mom10_1, _mom10_1);
			_mom10_1 = _mm_hadd_epi32(_mom10_1, _mom10_1);

			//Calc mom01
			_mom01_1 = _mm_mullo_epi32(m1, yPos);
			_mom01_2 = _mm_mullo_epi32(m2, yPos);
			_mom01_3 = _mm_mullo_epi32(m3, yPos);
			_mom01_4 = _mm_mullo_epi32(m4, yPos);

			_mom01_1 = _mm_add_epi32(_mom01_1, _mom01_2);
			_mom01_3 = _mm_add_epi32(_mom01_3, _mom01_4);
			_mom01_1 = _mm_add_epi32(_mom01_1, _mom01_3);

			_mom01_1 = _mm_hadd_epi32(_mom01_1, _mom01_1);
			_mom01_1 = _mm_hadd_epi32(_mom01_1, _mom01_1);


			//Output
			mom0 += mom_0;
			mom10 += _mom10_1.m128i_u32[0];
			mom01 += _mom01_1.m128i_u32[0];

		}
	}
#endif

#else	//	Not Use original
	int out;
	int mom_0;

	//	const __m128i _true = _mm_set1_epi8(1);
	//	const __m128i _false = _mm_set1_epi8(0);
	__m128i _mom0, _mom0_t1, _mom0_t2;
	__m128i _mom10_1, _mom10_2;	// , _mom10_3, _mom10_4;
	__m128i _mom01_1, _mom01_2;	// , _mom01_3, _mom01_4;

	__m128i _mom00_sum = _mm_set1_epi32(0);
	__m128i _mom10_sum = _mm_set1_epi32(0);
	__m128i _mom01_sum = _mm_set1_epi32(0);

	const __m128i _ones_16 = _mm_set1_epi16(1);
	const __m128i _true = _mm_set1_epi8(1);
	const __m128i _false = _mm_set1_epi8(0);

	int j;
	for (int y = _start; y < _stop; y++){
		const __m128i yPos = _mm_set1_epi16(y);
		j = width*y;
		for (int x = 0; x < width; x += 16){
			if ((x + 16 <= xstart) || (xstop <= x)){
				continue;
			}
			mom_0 = 0;
			i = j + x;
			//	静的クラス変数として、使用する値を用意しておく
			const __m128i xPos1 = _mm_set_epi16(x + 7, x + 6, x + 5, x + 4, x + 3, x + 2, x + 1, x);
			const __m128i xPos2 = _mm_set_epi16(x + 15, x + 14, x + 13, x + 12, x + 11, x + 10, x + 9, x + 8);
			//Load image
			const __m128i srcData = _mm_load_si128((const __m128i*) (src + i));

			//mom0
			//Mask image (Calc mom0) 255->1, 0->0
			_mom0 = _mm_blendv_epi8(_false, _true, srcData);

			//Output
			//			for (int i = 0; i < 16; i++){
			//				mom_0 += _mom0.m128i_u8[i];
			//			}

			//Divide 8bit -> 16bit convert
			__m128i m1 = _mm_cvtepu8_epi16(_mom0);
			_mom0 = _mm_srli_si128(_mom0, 8);
			__m128i m2 = _mm_cvtepu8_epi16(_mom0);

			//	m00の結果を溜め込んでおく
			_mom0_t1 = _mm_madd_epi16(m1, _ones_16);
			_mom00_sum = _mm_add_epi32(_mom00_sum, _mom0_t1);	//	結果を溜め込んでおく
			_mom0_t2 = _mm_madd_epi16(m2, _ones_16);
			_mom00_sum = _mm_add_epi32(_mom00_sum, _mom0_t2);	//	結果を溜め込んでおく
			//Calc mom10
			_mom10_1 = _mm_madd_epi16(m1, xPos1);
			_mom10_2 = _mm_madd_epi16(m2, xPos2);

			_mom10_1 = _mm_hadd_epi32(_mom10_1, _mom10_2);
			_mom10_sum = _mm_add_epi32(_mom10_sum, _mom10_1);	//	結果を溜め込んでおく

			//Calc mom01
			_mom01_1 = _mm_madd_epi16(m1, yPos);
			_mom01_2 = _mm_madd_epi16(m2, yPos);

			_mom01_1 = _mm_hadd_epi32(_mom01_1, _mom01_2);
			_mom01_sum = _mm_add_epi32(_mom01_sum, _mom01_1);	//	結果を溜め込んでおく

			//Output
			//			mom0 += mom_0;

		}
	}
	_mom00_sum = _mm_hadd_epi32(_mom00_sum, _mom00_sum);
	_mom00_sum = _mm_hadd_epi32(_mom00_sum, _mom00_sum);

	_mom10_sum = _mm_hadd_epi32(_mom10_sum, _mom10_sum);
	_mom10_sum = _mm_hadd_epi32(_mom10_sum, _mom10_sum);

	_mom01_sum = _mm_hadd_epi32(_mom01_sum, _mom01_sum);
	_mom01_sum = _mm_hadd_epi32(_mom01_sum, _mom01_sum);

	mom0 += _mom00_sum.m128i_u32[0];
	mom10 += _mom10_sum.m128i_u32[0];
	mom01 += _mom01_sum.m128i_u32[0];

	#endif

}