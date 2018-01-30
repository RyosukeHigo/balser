#include <iostream>

/*
i0a i0b i0c i0d
i1a i1b i1c i1d
i2a i2b i2c i2d
i3a i3b i3c i3d
 
 <unpack> x 2
i0a 0 0 0
i0b 0 0 0
i0c 0 0 0
i0d 0 0 0
 */
///////////////////////////////////////////////
//////////////////// DEFINES //////////////////
///////////////////////////////////////////////
#define CAST_8U(t) convert_uchar_rte(t)

#define BUILD_128BIT_CONST(_name,idx,B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15) _name[idx] = 0x##B7##B6##B5##B4##B3##B2##B1##B0##ull; _name[idx+1] = 0x##B15##B14##B13##B12##B11##B10##B9##B8##ull;

#define inv255 (0.003921568627451f);

// =============================================================
// ====================== CVTCOLOR COMMON ======================
// =============================================================
template <typename T>
class CVTCOLOR_TBB {
	
	void (*function)(const T* _src, T* _dest, unsigned int _width,
					 unsigned int _pitchs, unsigned int _pitchd,
					 unsigned int _start, unsigned int _stop );
	const unsigned char *src;
	unsigned char *dest;
	unsigned int width;
	unsigned int height;
	unsigned int pitchs;
	unsigned int pitchd;
	unsigned int threads;
	
public:
	inline void operator()( const tbb::blocked_range<size_t>& r ) const {
		
		// Let's make the last thread do the least work
		float blockSize = (float)height/(float)threads;
		
		unsigned int start = GET_START(r.begin(),blockSize);
		unsigned int stop  = GET_STOP(r.end(),blockSize);
		
		
		function((const T*)src, (T*)dest, width,
				 pitchs, pitchd,
				 start, stop );
	}
	
	CVTCOLOR_TBB(void(*_function)(const T* _src, T* _dest, unsigned int _width,
								  unsigned int _pitchs, unsigned int _pitchd,
								  unsigned int _start, unsigned int _stop ),
				 const unsigned char* _src, unsigned char* _dest,
				 unsigned int _width, unsigned int _height,
				 unsigned int _pitchs, unsigned int _pitchd, unsigned int _threads ) :
	function(_function), src(_src), dest(_dest), width(_width), height(_height), pitchs(_pitchs), pitchd(_pitchd), threads(_threads) {
		
	}
};


template <typename T>
void sequentialCall(void(*_function)(const T* _src, T* _dest, unsigned int _width,
									 unsigned int _pitchs, unsigned int _pitchd,
									 unsigned int _start, unsigned int _stop ),
					const unsigned char* _src, unsigned char* _dest,
					unsigned int _width, unsigned int _height,
					unsigned int _pitchs, unsigned int _pitchd, unsigned int _threads ) {
	_function((const T*)_src, (T*)_dest, _width, _pitchs, _pitchd, 0, _height);
}



///////////////////////////////////////////////
/////////////// HELPER FUNCTIONS //////////////
///////////////////////////////////////////////

// Takes mod of 8 words in _src (assumes values between -127 and 127
// OBS! This does not give an accurate mod since mod3(-3) = -3
inline __m128i mod3( const __m128i& _src ) {
    
    __m128i f = _mm_set1_epi16(0x000f);
    
    // If _src[i] < 0
    __m128i zero = _mm_set1_epi16(0);
    __m128i signv = _mm_cmplt_epi16(_src, zero);
    
    __m128i mthree = _mm_set1_epi16(-3);
    __m128i modv = _mm_and_si128(signv, mthree);
    
    __m128i shift = _mm_srai_epi16(_src, 4);
    __m128i masked =  _mm_and_si128(_src, f);
    
    __m128i newsrc = _mm_add_epi16(shift,masked);
    shift = _mm_srai_epi16(newsrc, 2);
    
    __m128i three = _mm_set1_epi16(0x00000003);
    newsrc = _mm_add_epi16(shift, _mm_and_si128(newsrc, three));
    shift = _mm_srai_epi16(newsrc, 2);
    newsrc = _mm_add_epi16(shift, _mm_and_si128(newsrc, three));
    shift = _mm_srai_epi16(newsrc, 2);
    newsrc = _mm_add_epi16(shift, _mm_and_si128(newsrc, three));
    
    __m128i eq3 = _mm_cmpeq_epi16(newsrc, three);
    newsrc = _mm_andnot_si128(eq3,newsrc);
    
    return _mm_add_epi16(newsrc, modv);
    
}

#ifdef USE_AVX2
inline __m256i mod3( const __m256i& _src ) {
    
    __m256i f = _mm256_set1_epi16(0x000f);
    
    // If _src[i] < 0
    __m256i signv = _mm256_cmpgt_epi16(_mm256_set1_epi16(0), _src);
    
    __m256i mthree = _mm256_set1_epi16(-3);
    __m256i modv = _mm256_and_si256(signv, mthree);
    
    __m256i shift = _mm256_srai_epi16(_src, 4);
    __m256i masked =  _mm256_and_si256(_src, f);
    
    __m256i newsrc = _mm256_add_epi16(shift,masked);
    shift = _mm256_srai_epi16(newsrc, 2);
    
    __m256i three = _mm256_set1_epi16(0x00000003);
    newsrc = _mm256_add_epi16(shift, _mm256_and_si256(newsrc, three));
    shift = _mm256_srai_epi16(newsrc, 2);
    newsrc = _mm256_add_epi16(shift, _mm256_and_si256(newsrc, three));
    shift = _mm256_srai_epi16(newsrc, 2);
    newsrc = _mm256_add_epi16(shift, _mm256_and_si256(newsrc, three));
    
    __m256i eq3 = _mm256_cmpeq_epi16(newsrc, three);
    newsrc = _mm256_andnot_si256(eq3,newsrc);
    
    return _mm256_add_epi16(newsrc, modv);
}
#endif // USE_AVX2


// Takes cube root of each element in _src and stores in _dest
// Assumes non-negative numbers
// Needs 32 bit integer operations, so AVX2 is needed
// void cubeRoot_p( const __m256& _src, __m256& _dest ) {}
// Takes two __m128 
void cubeRoot_p( const __m128* _src, __m128* _dest ) {
    
    __m128i srci1 = _mm_castps_si128(_src[0]);
    __m128i valuepart1 = _mm_and_si128(srci1, _mm_set1_epi32(0x7fffffff));
    __m128i signpart1  = _mm_and_si128(srci1, _mm_set1_epi32(0x80000000));

    __m128i srci2 = _mm_castps_si128(_src[1]);
    __m128i valuepart2 = _mm_and_si128(srci2, _mm_set1_epi32(0x7fffffff));
    __m128i signpart2  = _mm_and_si128(srci2, _mm_set1_epi32(0x80000000));
    
    __m128i exponent1 = _mm_srai_epi32(valuepart1, 23);
    __m128i exponent2 = _mm_srai_epi32(valuepart2, 23);
    
    __m128i exp16 = _mm_packs_epi32(exponent1, exponent2);
    
    exp16 = _mm_sub_epi16(exp16, _mm_set1_epi16(127)); // ex
    
    // mod 3
    __m128i expmod = mod3(exp16); // shx

    __m128i geqzero = _mm_cmpgt_epi16(expmod, _mm_set1_epi16(-1));
    __m128i expsub = _mm_or_si128(_mm_andnot_si128(geqzero,expmod),
                                  _mm_and_si128(geqzero,_mm_sub_epi16(expmod, _mm_set1_epi16(3))) ); // shx

    // Unpack
    __m128i expsub1 = _mm_srai_epi32(_mm_unpacklo_epi16(expsub, expsub), 16);
    __m128i expsub2 = _mm_srai_epi32(_mm_unpackhi_epi16(expsub, expsub), 16);
    
    exp16 = _mm_sub_epi16(exp16, expsub);
     
    // Unpack
    __m128i exp1 = _mm_srai_epi32(_mm_unpacklo_epi16(exp16, exp16), 16);
    __m128i exp2 = _mm_srai_epi32(_mm_unpackhi_epi16(exp16, exp16), 16);
    
    // Divide by three
    __m128i factor = _mm_set1_epi32(21846);
    __m128i exp1d3mul = _mm_mullo_epi32(exp1, factor);
    __m128i exp1d3 = _mm_srai_epi32(exp1d3mul ,16); //ex/3
    __m128i exp2d3 = _mm_srai_epi32(_mm_mullo_epi32(exp2, _mm_set1_epi32(21846)) ,16);
    
    __m128i lt = _mm_cmplt_epi32(exp1d3, _mm_set1_epi32(0));
    exp1d3 = _mm_or_si128(_mm_and_si128(lt, _mm_add_epi32(exp1d3, _mm_set1_epi32(1))),
                          _mm_andnot_si128(lt, exp1d3));
    lt = _mm_cmplt_epi32(exp2d3, _mm_set1_epi32(0));
    exp2d3 = _mm_or_si128(_mm_and_si128(lt, _mm_add_epi32(exp2d3, _mm_set1_epi32(1))),
                          _mm_andnot_si128(lt, exp2d3));

    // v.i = (ix & ((1<<23)-1)) | ((shx + 127)<<23);
    
    __m128i restored1 = _mm_or_si128(_mm_and_si128(valuepart1, _mm_set1_epi32( (1<<23)-1 )),
                                     _mm_slli_epi32(_mm_add_epi32(expsub1, _mm_set1_epi32(127)), 23) );
    __m128i restored2 = _mm_or_si128(_mm_and_si128(valuepart2, _mm_set1_epi32( (1<<23)-1 )),
                                     _mm_slli_epi32(_mm_add_epi32(expsub2, _mm_set1_epi32(127)), 23) );
    
    __m128 restoreds = _mm_castsi128_ps(restored1);
    __m128 mul1 = _mm_mul_ps(_mm_set_ps1(45.2548339756803022511987494f), restoreds);
    __m128 mul2 = _mm_mul_ps(_mm_add_ps(mul1, _mm_set_ps1(192.2798368355061050458134625f)), restoreds);
    __m128 mul3 = _mm_mul_ps(_mm_add_ps(mul2, _mm_set_ps1(119.1654824285581628956914143f)), restoreds);
    __m128 mul4 = _mm_mul_ps(_mm_add_ps(mul3, _mm_set_ps1(13.43250139086239872172837314f)), restoreds);
    __m128 nom1 = _mm_add_ps(mul4, _mm_set_ps1(0.1636161226585754240958355063f));
    
    mul1 = _mm_mul_ps(_mm_set_ps1(14.80884093219134573786480845f), restoreds);
    mul2 = _mm_mul_ps(_mm_add_ps(mul1, _mm_set_ps1(151.9714051044435648658557668f)), restoreds);
    mul3 = _mm_mul_ps(_mm_add_ps(mul2, _mm_set_ps1(168.5254414101568283957668343f)), restoreds);
    mul4 = _mm_mul_ps(_mm_add_ps(mul3, _mm_set_ps1(33.9905941350215598754191872f)), restoreds);
    __m128 denom1 = _mm_add_ps(mul4, _mm_set_ps1(1.0f));

    __m128 res1 = _mm_div_ps(nom1, denom1);

    restoreds = _mm_castsi128_ps(restored2);
    mul1 = _mm_mul_ps(_mm_set_ps1(45.2548339756803022511987494f), restoreds);
    mul2 = _mm_mul_ps(_mm_add_ps(mul1, _mm_set_ps1(192.2798368355061050458134625f)), restoreds);
    mul3 = _mm_mul_ps(_mm_add_ps(mul2, _mm_set_ps1(119.1654824285581628956914143f)), restoreds);
    mul4 = _mm_mul_ps(_mm_add_ps(mul3, _mm_set_ps1(13.43250139086239872172837314f)), restoreds);
    nom1 = _mm_add_ps(mul4, _mm_set_ps1(0.1636161226585754240958355063f));
    
    mul1 = _mm_mul_ps(_mm_set_ps1(14.80884093219134573786480845f), restoreds);
    mul2 = _mm_mul_ps(_mm_add_ps(mul1, _mm_set_ps1(151.9714051044435648658557668f)), restoreds);
    mul3 = _mm_mul_ps(_mm_add_ps(mul2, _mm_set_ps1(168.5254414101568283957668343f)), restoreds);
    mul4 = _mm_mul_ps(_mm_add_ps(mul3, _mm_set_ps1(33.9905941350215598754191872f)), restoreds);
    denom1 = _mm_add_ps(mul4, _mm_set_ps1(1.0f));
    
    __m128 res2 = _mm_div_ps(nom1, denom1);

    __m128i res1i = _mm_castps_si128(res1);
    res1i = _mm_add_epi32(_mm_add_epi32(res1i, _mm_slli_epi32(exp1d3, 23)),
                          signpart1);
    
    _dest[0] = _mm_castsi128_ps(res1i);
    
    
    __m128i res2i = _mm_castps_si128(res2);
    res2i = _mm_add_epi32(_mm_add_epi32(res2i, _mm_slli_epi32(exp2d3, 23)),
                          signpart2);
    
    _dest[1] = _mm_castsi128_ps(res2i);
    
}


#ifdef USE_AVX2
void cubeRoot_p( const __m256* _src, __m256* _dest ) {
    
    __m256i srci1 = _mm256_castps_si256(_src[0]);
    __m256i valuepart1 = _mm256_and_si256(srci1, _mm256_set1_epi32(0x7fffffff));
    __m256i signpart1  = _mm256_and_si256(srci1, _mm256_set1_epi32(0x80000000));
    
    __m256i srci2 = _mm256_castps_si256(_src[1]);
    __m256i valuepart2 = _mm256_and_si256(srci2, _mm256_set1_epi32(0x7fffffff));
    __m256i signpart2  = _mm256_and_si256(srci2, _mm256_set1_epi32(0x80000000));
    
    __m256i exponent1 = _mm256_srai_epi32(valuepart1, 23);
    __m256i exponent2 = _mm256_srai_epi32(valuepart2, 23);
    
    __m256i exp16 = _mm256_packs_epi32(exponent1, exponent2);
    
    exp16 = _mm256_sub_epi16(exp16, _mm256_set1_epi16(127)); // ex
    
    // mod 3
    __m256i expmod = mod3(exp16); // shx
    
    __m256i geqzero = _mm256_cmpgt_epi16(expmod, _mm256_set1_epi16(-1));
    __m256i expsub = _mm256_or_si256(_mm256_andnot_si256(geqzero,expmod),
                                  _mm256_and_si256(geqzero,_mm256_sub_epi16(expmod, _mm256_set1_epi16(3))) ); // shx
    
    // Unpack
    __m256i expsub1 = _mm256_srai_epi32(_mm256_unpacklo_epi16(expsub, expsub), 16);
    __m256i expsub2 = _mm256_srai_epi32(_mm256_unpackhi_epi16(expsub, expsub), 16);
    
    exp16 = _mm256_sub_epi16(exp16, expsub);
    
    // Unpack
    __m256i exp1 = _mm256_srai_epi32(_mm256_unpacklo_epi16(exp16, exp16), 16);
    __m256i exp2 = _mm256_srai_epi32(_mm256_unpackhi_epi16(exp16, exp16), 16);
    
    // Divide by three
    __m256i factor = _mm256_set1_epi32(21846);
    __m256i exp1d3mul = _mm256_mullo_epi32(exp1, factor);
    __m256i exp1d3 = _mm256_srai_epi32(exp1d3mul ,16); //ex/3
    __m256i exp2d3 = _mm256_srai_epi32(_mm256_mullo_epi32(exp2, _mm256_set1_epi32(21846)) ,16);
    
    __m256i lt = _mm256_cmpgt_epi32(_mm256_set1_epi32(0), exp1d3);
    exp1d3 = _mm256_or_si256(_mm256_and_si256(lt, _mm256_add_epi32(exp1d3, _mm256_set1_epi32(1))),
                             _mm256_andnot_si256(lt, exp1d3));
    lt = _mm256_cmpgt_epi32(_mm256_set1_epi32(0), exp2d3);
    exp2d3 = _mm256_or_si256(_mm256_and_si256(lt, _mm256_add_epi32(exp2d3, _mm256_set1_epi32(1))),
                             _mm256_andnot_si256(lt, exp2d3));
    
    // v.i = (ix & ((1<<23)-1)) | ((shx + 127)<<23);
    
    __m256i restored1 = _mm256_or_si256(_mm256_and_si256(valuepart1, _mm256_set1_epi32( (1<<23)-1 )),
                                        _mm256_slli_epi32(_mm256_add_epi32(expsub1, _mm256_set1_epi32(127)), 23) );
    __m256i restored2 = _mm256_or_si256(_mm256_and_si256(valuepart2, _mm256_set1_epi32( (1<<23)-1 )),
                                        _mm256_slli_epi32(_mm256_add_epi32(expsub2, _mm256_set1_epi32(127)), 23) );
    
    __m256 restoreds = _mm256_castsi256_ps(restored1);
    __m256 fma1 = _mm256_fmadd_ps(_mm256_set1_ps(45.2548339756803022511987494f), restoreds, _mm256_set1_ps(192.2798368355061050458134625f));
    __m256 fma2 = _mm256_fmadd_ps(fma1, restoreds, _mm256_set1_ps(119.1654824285581628956914143f));
    __m256 fma3 = _mm256_fmadd_ps(fma2, restoreds, _mm256_set1_ps(13.43250139086239872172837314f));
    __m256 nom1 = _mm256_fmadd_ps(fma3, restoreds, _mm256_set1_ps(0.1636161226585754240958355063f));

    fma1 = _mm256_fmadd_ps(_mm256_set1_ps(14.80884093219134573786480845f), restoreds, _mm256_set1_ps(151.9714051044435648658557668f));
    fma2 = _mm256_fmadd_ps(fma1, restoreds, _mm256_set1_ps(168.5254414101568283957668343f));
    fma3 = _mm256_fmadd_ps(fma2, restoreds, _mm256_set1_ps(33.9905941350215598754191872f));
    __m256 denom1 = _mm256_fmadd_ps(fma3, restoreds, _mm256_set1_ps(1.0f));
    
    __m256 res1 = _mm256_div_ps(nom1, denom1);
    
    restoreds = _mm256_castsi256_ps(restored2);
    fma1 = _mm256_fmadd_ps(_mm256_set1_ps(45.2548339756803022511987494f), restoreds, _mm256_set1_ps(192.2798368355061050458134625f));
    fma2 = _mm256_fmadd_ps(fma1, restoreds, _mm256_set1_ps(119.1654824285581628956914143f));
    fma3 = _mm256_fmadd_ps(fma2, restoreds, _mm256_set1_ps(13.43250139086239872172837314f));
    nom1 = _mm256_fmadd_ps(fma3, restoreds, _mm256_set1_ps(0.1636161226585754240958355063f));
    
    fma1 = _mm256_fmadd_ps(_mm256_set1_ps(14.80884093219134573786480845f), restoreds, _mm256_set1_ps(151.9714051044435648658557668f));
    fma2 = _mm256_fmadd_ps(fma1, restoreds, _mm256_set1_ps(168.5254414101568283957668343f));
    fma3 = _mm256_fmadd_ps(fma2, restoreds, _mm256_set1_ps(33.9905941350215598754191872f));
    denom1 = _mm256_fmadd_ps(fma3, restoreds, _mm256_set1_ps(1.0f));
    
    __m256 res2 = _mm256_div_ps(nom1, denom1);
    
    __m256i res1i = _mm256_castps_si256(res1);
    res1i = _mm256_add_epi32(_mm256_add_epi32(res1i, _mm256_slli_epi32(exp1d3, 23)),
                             signpart1);
    
    _dest[0] = _mm256_castsi256_ps(res1i);
    
    
    __m256i res2i = _mm256_castps_si256(res2);
    res2i = _mm256_add_epi32(_mm256_add_epi32(res2i, _mm256_slli_epi32(exp2d3, 23)),
                             signpart2);
    
    _dest[1] = _mm256_castsi256_ps(res2i);
    
}
#endif // USE_AVX2


union Cv32sufp {
    float f;
    int i;
    unsigned int s;
};


float cubeRoot(float value) {
    
    float fr;
    Cv32sufp v, m;
    int ix, s;
    int ex, shx;
    
    v.f = value;
    // _src as integer
    ix = v.i & 0x7fffffff;
    // Sign of _src
    s = v.i & 0x80000000;
    // Exponent minus 127: Gives (stored as unsigned integer. 127 -> 0)
    ex = (ix >> 23) - 127;
    
    // exponent of cube root
    // Modulo can be done in parallel using mod(4^k*a + b,3) == mod(a+b,3)
    shx = ex % 3;
    //std::cout << ex << " " << shx << std::endl;

    shx -= shx >= 0 ? 3 : 0;
    //std::cout << ex << "," << shx << std::endl;
    
    
    ex = (ex - shx) / 3;
    //std::cout << ex << "," << shx << std::endl;
    
    // exponent restored
    v.i = (ix & ((1<<23)-1)) | ((shx + 127)<<23);
    fr = v.f;
    
    // 0.125 <= fr < 1.0
    //Use quartic rational polynomial with error < 2^(-24)
    fr = (float)(((((45.2548339756803022511987494 * fr +
                     192.2798368355061050458134625) * fr +
                    119.1654824285581628956914143) * fr +
                   13.43250139086239872172837314) * fr +
                  0.1636161226585754240958355063)/
                 ((((14.80884093219134573786480845 * fr +
                     151.9714051044435648658557668) * fr +
                    168.5254414101568283957668343) * fr +
                   33.9905941350215598754191872) * fr +
                  1.0));
    
    // fr *= 2^ex * sign
    m.f = value;
    v.f = fr;
    v.i = (v.i + (ex << 23) + s) & (m.i*2 != 0 ? -1 : 0);
    return v.f;
    
}


///////////////////////////////////////////////
////////////////// CONSTANTS //////////////////
///////////////////////////////////////////////
const float k_8u_32f[256] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
    20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
    30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
    40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
    50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
    60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f,
    70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
    80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f,
    90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f,
    100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f,
    110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f,
    120.0f, 121.0f, 122.0f, 123.0f, 124.0f, 125.0f, 126.0f, 127.0f, 128.0f, 129.0f,
    130.0f, 131.0f, 132.0f, 133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f, 139.0f,
    140.0f, 141.0f, 142.0f, 143.0f, 144.0f, 145.0f, 146.0f, 147.0f, 148.0f, 149.0f,
    150.0f, 151.0f, 152.0f, 153.0f, 154.0f, 155.0f, 156.0f, 157.0f, 158.0f, 159.0f,
    160.0f, 161.0f, 162.0f, 163.0f, 164.0f, 165.0f, 166.0f, 167.0f, 168.0f, 169.0f,
    170.0f, 171.0f, 172.0f, 173.0f, 174.0f, 175.0f, 176.0f, 177.0f, 178.0f, 179.0f,
    180.0f, 181.0f, 182.0f, 183.0f, 184.0f, 185.0f, 186.0f, 187.0f, 188.0f, 189.0f,
    190.0f, 191.0f, 192.0f, 193.0f, 194.0f, 195.0f, 196.0f, 197.0f, 198.0f, 199.0f,
    200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f, 206.0f, 207.0f, 208.0f, 209.0f,
    210.0f, 211.0f, 212.0f, 213.0f, 214.0f, 215.0f, 216.0f, 217.0f, 218.0f, 219.0f,
    220.0f, 221.0f, 222.0f, 223.0f, 224.0f, 225.0f, 226.0f, 227.0f, 228.0f, 229.0f,
    230.0f, 231.0f, 232.0f, 233.0f, 234.0f, 235.0f, 236.0f, 237.0f, 238.0f, 239.0f,
    240.0f, 241.0f, 242.0f, 243.0f, 244.0f, 245.0f, 246.0f, 247.0f, 248.0f, 249.0f,
    250.0f, 251.0f, 252.0f, 253.0f, 254.0f, 255.0f};

const float k_post_coeffs_8u[] = { 2.55f, 0.0f, 0.72033898305084743f,
    96.525423728813564f, 0.99609375f, 139.453125f };
const float k_post_coeffs_32f[] = { 2.55f/255.f, 0.0f/255.f, 0.72033898305084743f/255.f,
    96.525423728813564f/255.f, 0.99609375f/255.f, 139.453125f/255.f };

const float k_pre_coeffs[] = { 0.39215686274509809f, 0.0f, 1.388235294117647f,
    -134.0f, 1.003921568627451f, -140.0f };

const double k_jet[] = {
    0.000, 0.000, 0.516,
    0.000, 0.000, 0.531,
    0.000, 0.000, 0.547,
    0.000, 0.000, 0.562,
    0.000, 0.000, 0.578,
    0.000, 0.000, 0.594,
    0.000, 0.000, 0.609,
    0.000, 0.000, 0.625,
    0.000, 0.000, 0.641,
    0.000, 0.000, 0.656,
    0.000, 0.000, 0.672,
    0.000, 0.000, 0.688,
    0.000, 0.000, 0.703,
    0.000, 0.000, 0.719,
    0.000, 0.000, 0.734,
    0.000, 0.000, 0.750,
    0.000, 0.000, 0.766,
    0.000, 0.000, 0.781,
    0.000, 0.000, 0.797,
    0.000, 0.000, 0.812,
    0.000, 0.000, 0.828,
    0.000, 0.000, 0.844,
    0.000, 0.000, 0.859,
    0.000, 0.000, 0.875,
    0.000, 0.000, 0.891,
    0.000, 0.000, 0.906,
    0.000, 0.000, 0.922,
    0.000, 0.000, 0.938,
    0.000, 0.000, 0.953,
    0.000, 0.000, 0.969,
    0.000, 0.000, 0.984,
    0.000, 0.000, 1.000,
    0.000, 0.016, 1.000,
    0.000, 0.031, 1.000,
    0.000, 0.047, 1.000,
    0.000, 0.062, 1.000,
    0.000, 0.078, 1.000,
    0.000, 0.094, 1.000,
    0.000, 0.109, 1.000,
    0.000, 0.125, 1.000,
    0.000, 0.141, 1.000,
    0.000, 0.156, 1.000,
    0.000, 0.172, 1.000,
    0.000, 0.188, 1.000,
    0.000, 0.203, 1.000,
    0.000, 0.219, 1.000,
    0.000, 0.234, 1.000,
    0.000, 0.250, 1.000,
    0.000, 0.266, 1.000,
    0.000, 0.281, 1.000,
    0.000, 0.297, 1.000,
    0.000, 0.312, 1.000,
    0.000, 0.328, 1.000,
    0.000, 0.344, 1.000,
    0.000, 0.359, 1.000,
    0.000, 0.375, 1.000,
    0.000, 0.391, 1.000,
    0.000, 0.406, 1.000,
    0.000, 0.422, 1.000,
    0.000, 0.438, 1.000,
    0.000, 0.453, 1.000,
    0.000, 0.469, 1.000,
    0.000, 0.484, 1.000,
    0.000, 0.500, 1.000,
    0.000, 0.516, 1.000,
    0.000, 0.531, 1.000,
    0.000, 0.547, 1.000,
    0.000, 0.562, 1.000,
    0.000, 0.578, 1.000,
    0.000, 0.594, 1.000,
    0.000, 0.609, 1.000,
    0.000, 0.625, 1.000,
    0.000, 0.641, 1.000,
    0.000, 0.656, 1.000,
    0.000, 0.672, 1.000,
    0.000, 0.688, 1.000,
    0.000, 0.703, 1.000,
    0.000, 0.719, 1.000,
    0.000, 0.734, 1.000,
    0.000, 0.750, 1.000,
    0.000, 0.766, 1.000,
    0.000, 0.781, 1.000,
    0.000, 0.797, 1.000,
    0.000, 0.812, 1.000,
    0.000, 0.828, 1.000,
    0.000, 0.844, 1.000,
    0.000, 0.859, 1.000,
    0.000, 0.875, 1.000,
    0.000, 0.891, 1.000,
    0.000, 0.906, 1.000,
    0.000, 0.922, 1.000,
    0.000, 0.938, 1.000,
    0.000, 0.953, 1.000,
    0.000, 0.969, 1.000,
    0.000, 0.984, 1.000,
    0.000, 1.000, 1.000,
    0.016, 1.000, 0.984,
    0.031, 1.000, 0.969,
    0.047, 1.000, 0.953,
    0.062, 1.000, 0.938,
    0.078, 1.000, 0.922,
    0.094, 1.000, 0.906,
    0.109, 1.000, 0.891,
    0.125, 1.000, 0.875,
    0.141, 1.000, 0.859,
    0.156, 1.000, 0.844,
    0.172, 1.000, 0.828,
    0.188, 1.000, 0.812,
    0.203, 1.000, 0.797,
    0.219, 1.000, 0.781,
    0.234, 1.000, 0.766,
    0.250, 1.000, 0.750,
    0.266, 1.000, 0.734,
    0.281, 1.000, 0.719,
    0.297, 1.000, 0.703,
    0.312, 1.000, 0.688,
    0.328, 1.000, 0.672,
    0.344, 1.000, 0.656,
    0.359, 1.000, 0.641,
    0.375, 1.000, 0.625,
    0.391, 1.000, 0.609,
    0.406, 1.000, 0.594,
    0.422, 1.000, 0.578,
    0.438, 1.000, 0.562,
    0.453, 1.000, 0.547,
    0.469, 1.000, 0.531,
    0.484, 1.000, 0.516,
    0.500, 1.000, 0.500,
    0.516, 1.000, 0.484,
    0.531, 1.000, 0.469,
    0.547, 1.000, 0.453,
    0.562, 1.000, 0.438,
    0.578, 1.000, 0.422,
    0.594, 1.000, 0.406,
    0.609, 1.000, 0.391,
    0.625, 1.000, 0.375,
    0.641, 1.000, 0.359,
    0.656, 1.000, 0.344,
    0.672, 1.000, 0.328,
    0.688, 1.000, 0.312,
    0.703, 1.000, 0.297,
    0.719, 1.000, 0.281,
    0.734, 1.000, 0.266,
    0.750, 1.000, 0.250,
    0.766, 1.000, 0.234,
    0.781, 1.000, 0.219,
    0.797, 1.000, 0.203,
    0.812, 1.000, 0.188,
    0.828, 1.000, 0.172,
    0.844, 1.000, 0.156,
    0.859, 1.000, 0.141,
    0.875, 1.000, 0.125,
    0.891, 1.000, 0.109,
    0.906, 1.000, 0.094,
    0.922, 1.000, 0.078,
    0.938, 1.000, 0.062,
    0.953, 1.000, 0.047,
    0.969, 1.000, 0.031,
    0.984, 1.000, 0.016,
    1.000, 1.000, 0.000,
    1.000, 0.984, 0.000,
    1.000, 0.969, 0.000,
    1.000, 0.953, 0.000,
    1.000, 0.938, 0.000,
    1.000, 0.922, 0.000,
    1.000, 0.906, 0.000,
    1.000, 0.891, 0.000,
    1.000, 0.875, 0.000,
    1.000, 0.859, 0.000,
    1.000, 0.844, 0.000,
    1.000, 0.828, 0.000,
    1.000, 0.812, 0.000,
    1.000, 0.797, 0.000,
    1.000, 0.781, 0.000,
    1.000, 0.766, 0.000,
    1.000, 0.750, 0.000,
    1.000, 0.734, 0.000,
    1.000, 0.719, 0.000,
    1.000, 0.703, 0.000,
    1.000, 0.688, 0.000,
    1.000, 0.672, 0.000,
    1.000, 0.656, 0.000,
    1.000, 0.641, 0.000,
    1.000, 0.625, 0.000,
    1.000, 0.609, 0.000,
    1.000, 0.594, 0.000,
    1.000, 0.578, 0.000,
    1.000, 0.562, 0.000,
    1.000, 0.547, 0.000,
    1.000, 0.531, 0.000,
    1.000, 0.516, 0.000,
    1.000, 0.500, 0.000,
    1.000, 0.484, 0.000,
    1.000, 0.469, 0.000,
    1.000, 0.453, 0.000,
    1.000, 0.438, 0.000,
    1.000, 0.422, 0.000,
    1.000, 0.406, 0.000,
    1.000, 0.391, 0.000,
    1.000, 0.375, 0.000,
    1.000, 0.359, 0.000,
    1.000, 0.344, 0.000,
    1.000, 0.328, 0.000,
    1.000, 0.312, 0.000,
    1.000, 0.297, 0.000,
    1.000, 0.281, 0.000,
    1.000, 0.266, 0.000,
    1.000, 0.250, 0.000,
    1.000, 0.234, 0.000,
    1.000, 0.219, 0.000,
    1.000, 0.203, 0.000,
    1.000, 0.188, 0.000,
    1.000, 0.172, 0.000,
    1.000, 0.156, 0.000,
    1.000, 0.141, 0.000,
    1.000, 0.125, 0.000,
    1.000, 0.109, 0.000,
    1.000, 0.094, 0.000,
    1.000, 0.078, 0.000,
    1.000, 0.062, 0.000,
    1.000, 0.047, 0.000,
    1.000, 0.031, 0.000,
    1.000, 0.016, 0.000,
    1.000, 0.000, 0.000,
    0.984, 0.000, 0.000,
    0.969, 0.000, 0.000,
    0.953, 0.000, 0.000,
    0.938, 0.000, 0.000,
    0.922, 0.000, 0.000,
    0.906, 0.000, 0.000,
    0.891, 0.000, 0.000,
    0.875, 0.000, 0.000,
    0.859, 0.000, 0.000,
    0.844, 0.000, 0.000,
    0.828, 0.000, 0.000,
    0.812, 0.000, 0.000,
    0.797, 0.000, 0.000,
    0.781, 0.000, 0.000,
    0.766, 0.000, 0.000,
    0.750, 0.000, 0.000,
    0.734, 0.000, 0.000,
    0.719, 0.000, 0.000,
    0.703, 0.000, 0.000,
    0.688, 0.000, 0.000,
    0.672, 0.000, 0.000,
    0.656, 0.000, 0.000,
    0.641, 0.000, 0.000,
    0.625, 0.000, 0.000,
    0.609, 0.000, 0.000,
    0.594, 0.000, 0.000,
    0.578, 0.000, 0.000,
    0.562, 0.000, 0.000,
    0.547, 0.000, 0.000,
    0.531, 0.000, 0.000,
    0.516, 0.000, 0.000,
    0.500, 0.000, 0.000
};


