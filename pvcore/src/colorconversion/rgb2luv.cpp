
#ifdef USE_AVX2
// takes vectors of size 2 (__m256[2])
inline void rgb2luv_helper(const __m256* r, const __m256* g, const __m256* b,
						   __m256* L, __m256* u, __m256* v ) {

    __m256 y[2];
	
	// First convert rgb to xyz-form
	// TODO: Replace constants with defines
	__m256 x0;
	x0 = _mm256_add_ps(_mm256_mul_ps(r[0],_mm256_set1_ps(0.412453f)),
					   _mm256_add_ps(_mm256_mul_ps(g[0],_mm256_set1_ps(0.357580f)),
									 _mm256_mul_ps(b[0],_mm256_set1_ps(0.180423f))));
	y[0] = _mm256_add_ps(_mm256_mul_ps(r[0],_mm256_set1_ps(0.212671f)),
						 _mm256_add_ps(_mm256_mul_ps(g[0],_mm256_set1_ps(0.715160f)),
									   _mm256_mul_ps(b[0],_mm256_set1_ps(0.072169f))));
	__m256 z0;
	z0 = _mm256_add_ps(_mm256_mul_ps(r[0],_mm256_set1_ps(0.019334f)),
					   _mm256_add_ps(_mm256_mul_ps(g[0],_mm256_set1_ps(0.119193f)),
									 _mm256_mul_ps(b[0],_mm256_set1_ps(0.950227f))));
	
	__m256 x1 = _mm256_add_ps(_mm256_mul_ps(r[1],_mm256_set1_ps(0.412453f)),
							  _mm256_add_ps(_mm256_mul_ps(g[1],_mm256_set1_ps(0.357580f)),
											_mm256_mul_ps(b[1],_mm256_set1_ps(0.180423f))));
	y[1] = _mm256_add_ps(_mm256_mul_ps(r[1],_mm256_set1_ps(0.212671f)),
						 _mm256_add_ps(_mm256_mul_ps(g[1],_mm256_set1_ps(0.715160f)),
									   _mm256_mul_ps(b[1],_mm256_set1_ps(0.072169f))));
	__m256 z1 = _mm256_add_ps(_mm256_mul_ps(r[1],_mm256_set1_ps(0.019334f)),
							  _mm256_add_ps(_mm256_mul_ps(g[1],_mm256_set1_ps(0.119193f)),
											_mm256_mul_ps(b[1],_mm256_set1_ps(0.950227f))));
	
	
	// ======================== Convert xyz to Luv
	// L-channel
	__m256 cube[2];
	cubeRoot_p(y, cube);
	__m256 Lbig0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(116.0f),cube[0]),_mm256_set1_ps(16.0f));
	__m256 Lsmall0 = _mm256_mul_ps(y[0], _mm256_set1_ps(903.3f));
	
	__m256 Lbig1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(116.0f),cube[1]),_mm256_set1_ps(16.0f));
	__m256 Lsmall1 = _mm256_mul_ps(y[1], _mm256_set1_ps(903.3f));
	// Select
	__m256 ygt0 = _mm256_cmp_ps(y[0], _mm256_set1_ps(0.008856f),_CMP_GT_OS);
	__m256 ygt1 = _mm256_cmp_ps(y[1], _mm256_set1_ps(0.008856f),_CMP_GT_OS);
	
	L[0] = _mm256_or_ps(_mm256_and_ps(ygt0, Lbig0), _mm256_andnot_ps(ygt0, Lsmall0));
	L[1] = _mm256_or_ps(_mm256_and_ps(ygt1, Lbig1), _mm256_andnot_ps(ygt1, Lsmall1));
	
	// u- and v- channels
	__m256 t0 = _mm256_add_ps(x0, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(15.0f), y[0]),
												_mm256_mul_ps(_mm256_set1_ps(3.0f), z0)));
	__m256 ti0 = _mm256_div_ps(_mm256_set1_ps(1.0f),t0);
	
	u[0] = _mm256_mul_ps(_mm256_set1_ps(4.0f), _mm256_mul_ps(x0, ti0));
	v[0] = _mm256_mul_ps(_mm256_set1_ps(9.0f), _mm256_mul_ps(y[0], ti0));
	
	u[0] = _mm256_mul_ps(_mm256_set1_ps(13.0f), _mm256_mul_ps(L[0], _mm256_sub_ps(u[0], _mm256_set1_ps(0.19793943f))));
	v[0] = _mm256_mul_ps(_mm256_set1_ps(13.0f), _mm256_mul_ps(L[0], _mm256_sub_ps(v[0], _mm256_set1_ps(0.46831096f))));
	
	__m256 t1 = _mm256_add_ps(x1, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(15.0f), y[1]),
												_mm256_mul_ps(_mm256_set1_ps(3.0f), z1)));
	__m256 ti1 = _mm256_div_ps(_mm256_set1_ps(1.0f),t1);
	
	u[1] = _mm256_mul_ps(_mm256_set1_ps(4.0f), _mm256_mul_ps(x1, ti1));
	v[1] = _mm256_mul_ps(_mm256_set1_ps(9.0f), _mm256_mul_ps(y[1], ti1));
	
	u[1] = _mm256_mul_ps(_mm256_set1_ps(13.0f), _mm256_mul_ps(L[1], _mm256_sub_ps(u[1], _mm256_set1_ps(0.19793943f))));
	v[1] = _mm256_mul_ps(_mm256_set1_ps(13.0f), _mm256_mul_ps(L[1], _mm256_sub_ps(v[1], _mm256_set1_ps(0.46831096f))));
	
}
#endif

// takes vectors of size 2 (__m128[2])
inline void rgb2luv_helper(const __m128* r, const __m128* g, const __m128* b,
						   __m128* L, __m128* u, __m128* v ) {
	
	__m128 y[2];
	
	// First convert rgb to xyz-form
	// TODO: Replace constants with defines
	__m128 x0;
	x0 = _mm_add_ps(_mm_mul_ps(r[0],_mm_set1_ps(0.412453f)),
					_mm_add_ps(_mm_mul_ps(g[0],_mm_set1_ps(0.357580f)),
							   _mm_mul_ps(b[0],_mm_set1_ps(0.180423f))));
	y[0] = _mm_add_ps(_mm_mul_ps(r[0],_mm_set1_ps(0.212671f)),
					  _mm_add_ps(_mm_mul_ps(g[0],_mm_set1_ps(0.715160f)),
								 _mm_mul_ps(b[0],_mm_set1_ps(0.072169f))));
	__m128 z0;
	z0 = _mm_add_ps(_mm_mul_ps(r[0],_mm_set1_ps(0.019334f)),
					_mm_add_ps(_mm_mul_ps(g[0],_mm_set1_ps(0.119193f)),
							   _mm_mul_ps(b[0],_mm_set1_ps(0.950227f))));
	
	__m128 x1 = _mm_add_ps(_mm_mul_ps(r[1],_mm_set1_ps(0.412453f)),
						   _mm_add_ps(_mm_mul_ps(g[1],_mm_set1_ps(0.357580f)),
									  _mm_mul_ps(b[1],_mm_set1_ps(0.180423f))));
	y[1] = _mm_add_ps(_mm_mul_ps(r[1],_mm_set1_ps(0.212671f)),
					  _mm_add_ps(_mm_mul_ps(g[1],_mm_set1_ps(0.715160f)),
								 _mm_mul_ps(b[1],_mm_set1_ps(0.072169f))));
	__m128 z1 = _mm_add_ps(_mm_mul_ps(r[1],_mm_set1_ps(0.019334f)),
						   _mm_add_ps(_mm_mul_ps(g[1],_mm_set1_ps(0.119193f)),
									  _mm_mul_ps(b[1],_mm_set1_ps(0.950227f))));
	
	
	// ======================== Convert xyz to Luv
	// L-channel
	__m128 cube[2];
	cubeRoot_p(y, cube);
	__m128 Lbig0 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(116.0f),cube[0]),_mm_set1_ps(16.0f));
	__m128 Lsmall0 = _mm_mul_ps(y[0], _mm_set1_ps(903.3f));
	
	__m128 Lbig1 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(116.0f),cube[1]),_mm_set1_ps(16.0f));
	__m128 Lsmall1 = _mm_mul_ps(y[1], _mm_set1_ps(903.3f));
	// Select
	__m128 ygt0 = _mm_cmpgt_ps(y[0], _mm_set1_ps(0.008856f));
	__m128 ygt1 = _mm_cmpgt_ps(y[1], _mm_set1_ps(0.008856f));
	
	L[0] = _mm_or_ps(_mm_and_ps(ygt0, Lbig0), _mm_andnot_ps(ygt0, Lsmall0));
	L[1] = _mm_or_ps(_mm_and_ps(ygt1, Lbig1), _mm_andnot_ps(ygt1, Lsmall1));
	
	// u- and v- channels
	__m128 t0 = _mm_add_ps(x0, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(15.0f), y[0]),
										  _mm_mul_ps(_mm_set1_ps(3.0f), z0)));
	__m128 ti0 = _mm_div_ps(_mm_set1_ps(1.0f),t0);
	
	u[0] = _mm_mul_ps(_mm_set1_ps(4.0f), _mm_mul_ps(x0, ti0));
	v[0] = _mm_mul_ps(_mm_set1_ps(9.0f), _mm_mul_ps(y[0], ti0));
	
	u[0] = _mm_mul_ps(_mm_set1_ps(13.0f), _mm_mul_ps(L[0], _mm_sub_ps(u[0], _mm_set1_ps(0.19793943f))));
	v[0] = _mm_mul_ps(_mm_set1_ps(13.0f), _mm_mul_ps(L[0], _mm_sub_ps(v[0], _mm_set1_ps(0.46831096f))));
	
	__m128 t1 = _mm_add_ps(x1, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(15.0f), y[1]),
										  _mm_mul_ps(_mm_set1_ps(3.0f), z1)));
	__m128 ti1 = _mm_div_ps(_mm_set1_ps(1.0f),t1);
	
	u[1] = _mm_mul_ps(_mm_set1_ps(4.0f), _mm_mul_ps(x1, ti1));
	v[1] = _mm_mul_ps(_mm_set1_ps(9.0f), _mm_mul_ps(y[1], ti1));
	
	u[1] = _mm_mul_ps(_mm_set1_ps(13.0f), _mm_mul_ps(L[1], _mm_sub_ps(u[1], _mm_set1_ps(0.19793943f))));
	v[1] = _mm_mul_ps(_mm_set1_ps(13.0f), _mm_mul_ps(L[1], _mm_sub_ps(v[1], _mm_set1_ps(0.46831096f))));
	
	
}


void _rgb2luv_32f(const float* _src, float* _dest, unsigned int _width,
				  unsigned int _pitchs, unsigned int _pitchd,
				  unsigned int _start, unsigned int _stop)  {
	
	const unsigned int width = (_pitchs/3) >> 3;
	
	float Ls[8], us[8], vs[8];
	
	for( unsigned int h=_start; h<_stop; ++h ) {
		const float *tsrc = _src + _pitchs*h;
		float *tdest = _dest + _pitchd*h;
		for( unsigned int w=0; w<width; ++w ) {
			
			__m128 reds[2],greens[2],blues[2];
			__m128 L[2],u[2],v[2];
			
			// Load float values
			reds[0]   = _mm_set_ps(tsrc[0],tsrc[3],tsrc[6],tsrc[9]);
			greens[0] = _mm_set_ps(tsrc[1],tsrc[4],tsrc[7],tsrc[10]);
			blues[0]  = _mm_set_ps(tsrc[2],tsrc[5],tsrc[8],tsrc[11]); tsrc += 12;
			
			reds[1]   = _mm_set_ps(tsrc[0],tsrc[3],tsrc[6],tsrc[9]);
			greens[1] = _mm_set_ps(tsrc[1],tsrc[4],tsrc[7],tsrc[10]);
			blues[1]  = _mm_set_ps(tsrc[2],tsrc[5],tsrc[8],tsrc[11]); tsrc += 12;
			
			rgb2luv_helper(reds, greens, blues, L, u, v);
			
			L[0] = _mm_add_ps(_mm_mul_ps(L[0], _mm_set1_ps(k_post_coeffs_32f[0])), _mm_set1_ps(k_post_coeffs_32f[1]));
			L[1] = _mm_add_ps(_mm_mul_ps(L[1], _mm_set1_ps(k_post_coeffs_32f[0])), _mm_set1_ps(k_post_coeffs_32f[1]));
			u[0] = _mm_add_ps(_mm_mul_ps(u[0], _mm_set1_ps(k_post_coeffs_32f[2])), _mm_set1_ps(k_post_coeffs_32f[3]));
			u[1] = _mm_add_ps(_mm_mul_ps(u[1], _mm_set1_ps(k_post_coeffs_32f[2])), _mm_set1_ps(k_post_coeffs_32f[3]));
			v[0] = _mm_add_ps(_mm_mul_ps(v[0], _mm_set1_ps(k_post_coeffs_32f[4])), _mm_set1_ps(k_post_coeffs_32f[5]));
			v[1] = _mm_add_ps(_mm_mul_ps(v[1], _mm_set1_ps(k_post_coeffs_32f[4])), _mm_set1_ps(k_post_coeffs_32f[5]));
			
			
			_mm_store_ps(Ls,L[0]);
			_mm_store_ps(us,u[0]);
			_mm_store_ps(vs,v[0]);
			
			_mm_store_ps(Ls+4,L[1]);
			_mm_store_ps(us+4,u[1]);
			_mm_store_ps(vs+4,v[1]);
			
			tdest[0] = Ls[0]; tdest[3] = Ls[1]; tdest[6] = Ls[2]; tdest[9] = Ls[3];
			tdest[1] = us[0]; tdest[4] = us[1]; tdest[7] = us[2]; tdest[10] = us[3];
			tdest[2] = vs[0]; tdest[5] = vs[1]; tdest[8] = vs[2]; tdest[11] = vs[3];
			tdest += 12;
			
			tdest[0] = Ls[4]; tdest[3] = Ls[5]; tdest[6] = Ls[6]; tdest[9] = Ls[7];
			tdest[1] = us[4]; tdest[4] = us[5]; tdest[7] = us[6]; tdest[10] = us[7];
			tdest[2] = vs[4]; tdest[5] = vs[5]; tdest[8] = vs[6]; tdest[11] = vs[7];
			tdest += 12;
			
		}
	}
	
}


void _rgb2luv_8u(const unsigned char* _src, unsigned char* _dest, unsigned int _width,
				 unsigned int _pitchs, unsigned int _pitchd,
				 unsigned int _start, unsigned int _stop) {
#ifdef USE_AVX2
	{
		// Change this to struct for better readability?
		ALIGNED_BUFFER(unsigned long long, buff, 144);
		
		// Shuffle bytes for processing
		// Reads 16 rgb-values and stores in 3x4 __m128i vectors
		buff[0]  = 0x8080800380808000ull; buff[1]  = 0x8080800980808006ull; buff[2]  = 0x8080800380808000ull; buff[3]  = 0x8080800980808006ull;    // Puts red colors from v0 in r1
		buff[4]  = 0x8080800480808001ull; buff[5]  = 0x8080800a80808007ull; buff[6]  = 0x8080800480808001ull; buff[7]  = 0x8080800a80808007ull;    // Puts green colors from v0 in g1
		buff[8]  = 0x8080800580808002ull; buff[9]  = 0x8080800b80808008ull; buff[10] = 0x8080800580808002ull; buff[11] = 0x8080800b80808008ull;    // Puts blue colors from v0 in b1
		
		buff[12] = 0x8080800f8080800cull; buff[13] = 0x8080808080808080ull; buff[14] = 0x8080800f8080800cull; buff[15] = 0x8080808080808080ull;    // Puts red colors from v0 in r2
		buff[16] = 0x8080808080808080ull; buff[17] = 0x8080800580808002ull; buff[18] = 0x8080808080808080ull; buff[19] = 0x8080800580808002ull;    // Puts red colors from v1 in r2
		buff[20] = 0x808080808080800dull; buff[21] = 0x8080808080808080ull; buff[22] = 0x808080808080800dull; buff[23] = 0x8080808080808080ull;    // Puts green colors from v0 in g2
		buff[24] = 0x8080800080808080ull; buff[25] = 0x8080800680808003ull; buff[26] = 0x8080800080808080ull; buff[27] = 0x8080800680808003ull;    // Puts green colors from v1 in g2
		buff[28] = 0x808080808080800eull; buff[29] = 0x8080808080808080ull; buff[30] = 0x808080808080800eull; buff[31] = 0x8080808080808080ull;    // Puts blue colors from v0 in b2
		buff[32] = 0x8080800180808080ull; buff[33] = 0x8080800780808004ull; buff[34] = 0x8080800180808080ull; buff[35] = 0x8080800780808004ull;    // Puts blue colors from v1 in b2
		
		buff[36] = 0x8080800b80808008ull; buff[37] = 0x808080808080800eull; buff[38] = 0x8080800b80808008ull; buff[39] = 0x808080808080800eull;    // Puts red colors from v1 in r3
		buff[40] = 0x8080808080808080ull; buff[41] = 0x8080800180808080ull; buff[42] = 0x8080808080808080ull; buff[43] = 0x8080800180808080ull;    // Puts red colors from v2 in r3
		buff[44] = 0x8080800c80808009ull; buff[45] = 0x808080808080800full; buff[46] = 0x8080800c80808009ull; buff[47] = 0x808080808080800full;    // Puts green colors from v1 in g3
		buff[48] = 0x8080808080808080ull; buff[49] = 0x8080800280808080ull; buff[50] = 0x8080808080808080ull; buff[51] = 0x8080800280808080ull;    // Puts green colors from v2 in g3
		buff[52] = 0x8080800d8080800aull; buff[53] = 0x8080808080808080ull; buff[54] = 0x8080800d8080800aull; buff[55] = 0x8080808080808080ull;    // Puts blue colors from v1 in b3
		buff[56] = 0x8080808080808080ull; buff[57] = 0x8080800380808000ull; buff[58] = 0x8080808080808080ull; buff[59] = 0x8080800380808000ull;    // Puts blue colors from v2 in b3
		
		buff[60] = 0x8080800780808004ull; buff[61] = 0x8080800d8080800aull; buff[62] = 0x8080800780808004ull; buff[63] = 0x8080800d8080800aull;    // Puts red colors from v2 in r4
		buff[64] = 0x8080800880808005ull; buff[65] = 0x8080800e8080800bull; buff[66] = 0x8080800880808005ull; buff[67] = 0x8080800e8080800bull;    // Puts green colors from v2 in g4
		buff[68] = 0x8080800980808006ull; buff[69] = 0x8080800f8080800cull; buff[70] = 0x8080800980808006ull; buff[71] = 0x8080800f8080800cull;    // Puts blue colors from v2 in b4
		
		// Shuffle bytes for storing
		buff[72] = 0x8008808004808000ull; buff[73] = 0x8080808080800c80ull; buff[74] = 0x8008808004808000ull; buff[75] = 0x8080808080800c80ull;    // Puts L0 in dest0
		buff[76] = 0x0880800480800080ull; buff[77] = 0x80808080800c8080ull; buff[78] = 0x0880800480800080ull; buff[79] = 0x80808080800c8080ull;    // Puts u0 in dest0
		buff[80] = 0x8080048080008080ull; buff[81] = 0x808080800c808008ull; buff[82] = 0x8080048080008080ull; buff[83] = 0x808080800c808008ull;    // Puts v0 in dest0
		
		buff[84] = 0x8080808080808080ull; buff[85] = 0x0480800080808080ull; buff[86] = 0x8080808080808080ull; buff[87] = 0x0480800080808080ull;    // Puts L1 in dest0
		buff[88] = 0x8080808080808080ull; buff[89] = 0x8080008080808080ull; buff[90] = 0x8080808080808080ull; buff[91] = 0x8080008080808080ull;    // Puts u1 in dest0
		buff[92] = 0x8080808080808080ull; buff[93] = 0x8000808080808080ull; buff[94] = 0x8080808080808080ull; buff[95] = 0x8000808080808080ull;    // Puts v1 in dest0
		
		buff[96]  = 0x80800c8080088080ull; buff[97]  = 0x8080808080808080ull; buff[98]  = 0x80800c8080088080ull; buff[99]  = 0x8080808080808080ull;    // Puts L1 in dest1
		buff[100] = 0x800c808008808004ull; buff[101] = 0x8080808080808080ull; buff[102] = 0x800c808008808004ull; buff[103] = 0x8080808080808080ull;    // Puts u1 in dest1
		buff[104] = 0x0c80800880800480ull; buff[105] = 0x8080808080808080ull; buff[106] = 0x0c80800880800480ull; buff[107] = 0x8080808080808080ull;    // Puts v1 in dest1
		
		buff[108] = 0x8080808080808080ull; buff[109] = 0x8008808004808000ull; buff[110] = 0x8080808080808080ull; buff[111] = 0x8008808004808000ull;    // Puts L2 in dest1
		buff[112] = 0x8080808080808080ull; buff[113] = 0x0880800480800080ull; buff[114] = 0x8080808080808080ull; buff[115] = 0x0880800480800080ull;    // Puts u2 in dest1
		buff[116] = 0x8080808080808080ull; buff[117] = 0x8080048080008080ull; buff[118] = 0x8080808080808080ull; buff[119] = 0x8080048080008080ull;    // Puts v2 in dest1
		
		buff[120] = 0x8080808080800c80ull; buff[121] = 0x8080808080808080ull; buff[122] = 0x8080808080800c80ull; buff[123] = 0x8080808080808080ull;    // Puts L2 in dest2
		buff[124] = 0x80808080800c8080ull; buff[125] = 0x8080808080808080ull; buff[126] = 0x80808080800c8080ull; buff[127] = 0x8080808080808080ull;    // Puts u2 in dest2
		buff[128] = 0x808080800c808008ull; buff[129] = 0x8080808080808080ull; buff[130] = 0x808080800c808008ull; buff[131] = 0x8080808080808080ull;    // Puts v2 in dest2
		
		buff[132] = 0x0480800080808080ull; buff[133] = 0x80800c8080088080ull; buff[134] = 0x0480800080808080ull; buff[135] = 0x80800c8080088080ull;    // Puts L3 in dest2
		buff[136] = 0x8080008080808080ull; buff[137] = 0x800c808008808004ull; buff[138] = 0x8080008080808080ull; buff[139] = 0x800c808008808004ull;    // Puts u3 in dest2
		buff[140] = 0x8000808080808080ull; buff[141] = 0x0c80800880800480ull; buff[142] = 0x8000808080808080ull; buff[143] = 0x0c80800880800480ull;    // Puts v3 in dest2
		
		
		// ============================ FIRST 8 COLORS
		const __m256i shuffle_r0 = _mm256_load_si256((const __m256i*)buff+0);
		const __m256i shuffle_g0 = _mm256_load_si256((const __m256i*)buff+1);
		const __m256i shuffle_b0 = _mm256_load_si256((const __m256i*)buff+2);
		
		const __m256i shuffle_r1a = _mm256_load_si256((const __m256i*)buff+3);
		const __m256i shuffle_r1b = _mm256_load_si256((const __m256i*)buff+4);
		const __m256i shuffle_g1a = _mm256_load_si256((const __m256i*)buff+5);
		const __m256i shuffle_g1b = _mm256_load_si256((const __m256i*)buff+6);
		const __m256i shuffle_b1a = _mm256_load_si256((const __m256i*)buff+7);
		const __m256i shuffle_b1b = _mm256_load_si256((const __m256i*)buff+8);
		
		const __m256i shuffle_r2a = _mm256_load_si256((const __m256i*)buff+9);
		const __m256i shuffle_r2b = _mm256_load_si256((const __m256i*)buff+10);
		const __m256i shuffle_g2a = _mm256_load_si256((const __m256i*)buff+11);
		const __m256i shuffle_g2b = _mm256_load_si256((const __m256i*)buff+12);
		const __m256i shuffle_b2a = _mm256_load_si256((const __m256i*)buff+13);
		const __m256i shuffle_b2b = _mm256_load_si256((const __m256i*)buff+14);
		
		const __m256i shuffle_r3 = _mm256_load_si256((const __m256i*)buff+15);
		const __m256i shuffle_g3 = _mm256_load_si256((const __m256i*)buff+16);
		const __m256i shuffle_b3 = _mm256_load_si256((const __m256i*)buff+17);
		
		
		const __m256i shuffle_L0 = _mm256_load_si256((const __m256i*)buff+18);
		const __m256i shuffle_u0 = _mm256_load_si256((const __m256i*)buff+19);
		const __m256i shuffle_v0 = _mm256_load_si256((const __m256i*)buff+20);
		
		const __m256i shuffle_L1a = _mm256_load_si256((const __m256i*)buff+21);
		const __m256i shuffle_L1b = _mm256_load_si256((const __m256i*)buff+24);
		const __m256i shuffle_u1a = _mm256_load_si256((const __m256i*)buff+22);
		const __m256i shuffle_u1b = _mm256_load_si256((const __m256i*)buff+25);
		const __m256i shuffle_v1a = _mm256_load_si256((const __m256i*)buff+23);
		const __m256i shuffle_v1b = _mm256_load_si256((const __m256i*)buff+26);
		
		const __m256i shuffle_L2a = _mm256_load_si256((const __m256i*)buff+27);
		const __m256i shuffle_L2b = _mm256_load_si256((const __m256i*)buff+30);
		const __m256i shuffle_u2a = _mm256_load_si256((const __m256i*)buff+28);
		const __m256i shuffle_u2b = _mm256_load_si256((const __m256i*)buff+31);
		const __m256i shuffle_v2a = _mm256_load_si256((const __m256i*)buff+29);
		const __m256i shuffle_v2b = _mm256_load_si256((const __m256i*)buff+32);
		
		const __m256i shuffle_L3 = _mm256_load_si256((const __m256i*)buff+33);
		const __m256i shuffle_u3 = _mm256_load_si256((const __m256i*)buff+34);
		const __m256i shuffle_v3 = _mm256_load_si256((const __m256i*)buff+35);
		
		for( unsigned int h=_start; h<_stop; ++h ) {
			const unsigned char *tsrc = _src + _pitchs*h;
			unsigned char *tdest = _dest + _pitchd*h;
			for( unsigned int w=0; w<_pitchs; w+=3*32 ) {
				
				// Load src into vectors
				const __m256i v0 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 16;
				const __m256i v1 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 16;
				const __m256i v2 = _mm256_loadu2_m128i((const __m128i*)tsrc+3,(const __m128i*)tsrc); tsrc += 64;
				
				__m256i redsi[2], greensi[2], bluesi[2];
				__m256 reds[2], greens[2], blues[2];
				__m256 L[2], u[2], v[2];
				
				// Put colors in place in integer vectors
				redsi[0]   = _mm256_shuffle_epi8(v0,shuffle_r0);
				greensi[0] = _mm256_shuffle_epi8(v0,shuffle_g0);
				bluesi[0]  = _mm256_shuffle_epi8(v0,shuffle_b0);
				
				redsi[1]   = _mm256_or_si256( _mm256_shuffle_epi8(v0,shuffle_r1a) , _mm256_shuffle_epi8(v1,shuffle_r1b) );
				greensi[1] = _mm256_or_si256( _mm256_shuffle_epi8(v0,shuffle_g1a) , _mm256_shuffle_epi8(v1,shuffle_g1b) );
				bluesi[1]  = _mm256_or_si256( _mm256_shuffle_epi8(v0,shuffle_b1a) , _mm256_shuffle_epi8(v1,shuffle_b1b) );
				
				// Normalized float values
				reds[0]   = _mm256_mul_ps(_mm256_cvtepi32_ps(redsi[0]),  _mm256_set1_ps(0.0039215686274509803f));
				reds[1]   = _mm256_mul_ps(_mm256_cvtepi32_ps(redsi[1]),  _mm256_set1_ps(0.0039215686274509803f));
				greens[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(greensi[0]),_mm256_set1_ps(0.0039215686274509803f));
				greens[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(greensi[1]),_mm256_set1_ps(0.0039215686274509803f));
				blues[0]  = _mm256_mul_ps(_mm256_cvtepi32_ps(bluesi[0]), _mm256_set1_ps(0.0039215686274509803f));
				blues[1]  = _mm256_mul_ps(_mm256_cvtepi32_ps(bluesi[1]), _mm256_set1_ps(0.0039215686274509803f));
				
				// Do actual conversion
				rgb2luv_helper(reds, greens, blues, L, u, v);
				
				// Post process
				L[0] = _mm256_add_ps(_mm256_mul_ps(L[0], _mm256_set1_ps(k_post_coeffs_8u[0])), _mm256_set1_ps(k_post_coeffs_8u[1]));
				L[1] = _mm256_add_ps(_mm256_mul_ps(L[1], _mm256_set1_ps(k_post_coeffs_8u[0])), _mm256_set1_ps(k_post_coeffs_8u[1]));
				u[0] = _mm256_add_ps(_mm256_mul_ps(u[0], _mm256_set1_ps(k_post_coeffs_8u[2])), _mm256_set1_ps(k_post_coeffs_8u[3]));
				u[1] = _mm256_add_ps(_mm256_mul_ps(u[1], _mm256_set1_ps(k_post_coeffs_8u[2])), _mm256_set1_ps(k_post_coeffs_8u[3]));
				v[0] = _mm256_add_ps(_mm256_mul_ps(v[0], _mm256_set1_ps(k_post_coeffs_8u[4])), _mm256_set1_ps(k_post_coeffs_8u[5]));
				v[1] = _mm256_add_ps(_mm256_mul_ps(v[1], _mm256_set1_ps(k_post_coeffs_8u[4])), _mm256_set1_ps(k_post_coeffs_8u[5]));
				
				// Convert to integer
				__m256i Li[4], ui[4], vi[4];
				Li[0] = _mm256_cvtps_epi32( L[0] );
				Li[1] = _mm256_cvtps_epi32( L[1] );
				ui[0] = _mm256_cvtps_epi32( u[0] );
				ui[1] = _mm256_cvtps_epi32( u[1] );
				vi[0] = _mm256_cvtps_epi32( v[0] );
				vi[1] = _mm256_cvtps_epi32( v[1] );
				
				
				
				__m256i dest0 = _mm256_shuffle_epi8(Li[0],shuffle_L0);
				dest0 = _mm256_or_si256( dest0, _mm256_shuffle_epi8(ui[0],shuffle_u0) );
				dest0 = _mm256_or_si256( dest0, _mm256_shuffle_epi8(vi[0],shuffle_v0) );
				
				dest0 = _mm256_or_si256( dest0, _mm256_shuffle_epi8(Li[1],shuffle_L1a) );
				dest0 = _mm256_or_si256( dest0, _mm256_shuffle_epi8(ui[1],shuffle_u1a) );
				dest0 = _mm256_or_si256( dest0, _mm256_shuffle_epi8(vi[1],shuffle_v1a) );
				
				__m256i dest1 = _mm256_shuffle_epi8(Li[1],shuffle_L1b);
				dest1 = _mm256_or_si256( dest1, _mm256_shuffle_epi8(ui[1],shuffle_u1b) );
				dest1 = _mm256_or_si256( dest1, _mm256_shuffle_epi8(vi[1],shuffle_v1b) );
				
				
				// ============================ SECOND 8 COLORS
				redsi[0]   = _mm256_or_si256( _mm256_shuffle_epi8(v1,shuffle_r2a) , _mm256_shuffle_epi8(v2,shuffle_r2b) );
				greensi[0] = _mm256_or_si256( _mm256_shuffle_epi8(v1,shuffle_g2a) , _mm256_shuffle_epi8(v2,shuffle_g2b) );
				bluesi[0]  = _mm256_or_si256( _mm256_shuffle_epi8(v1,shuffle_b2a) , _mm256_shuffle_epi8(v2,shuffle_b2b) );
				
				redsi[1]   = _mm256_shuffle_epi8(v2,shuffle_r3);
				greensi[1] = _mm256_shuffle_epi8(v2,shuffle_g3);
				bluesi[1]  = _mm256_shuffle_epi8(v2,shuffle_b3);
				
				
				// Normalized float values
				reds[0]   = _mm256_mul_ps(_mm256_cvtepi32_ps(redsi[0]),_mm256_set1_ps(0.0039215686274509803f));
				reds[1]   = _mm256_mul_ps(_mm256_cvtepi32_ps(redsi[1]),_mm256_set1_ps(0.0039215686274509803f));
				greens[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(greensi[0]),_mm256_set1_ps(0.0039215686274509803f));
				greens[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(greensi[1]),_mm256_set1_ps(0.0039215686274509803f));
				blues[0]  = _mm256_mul_ps(_mm256_cvtepi32_ps(bluesi[0]),_mm256_set1_ps(0.0039215686274509803f));
				blues[1]  = _mm256_mul_ps(_mm256_cvtepi32_ps(bluesi[1]),_mm256_set1_ps(0.0039215686274509803f));
				
				rgb2luv_helper(reds, greens, blues, L, u, v);
				
				L[0] = _mm256_add_ps(_mm256_mul_ps(L[0], _mm256_set1_ps(k_post_coeffs_8u[0])), _mm256_set1_ps(k_post_coeffs_8u[1]));
				L[1] = _mm256_add_ps(_mm256_mul_ps(L[1], _mm256_set1_ps(k_post_coeffs_8u[0])), _mm256_set1_ps(k_post_coeffs_8u[1]));
				u[0] = _mm256_add_ps(_mm256_mul_ps(u[0], _mm256_set1_ps(k_post_coeffs_8u[2])), _mm256_set1_ps(k_post_coeffs_8u[3]));
				u[1] = _mm256_add_ps(_mm256_mul_ps(u[1], _mm256_set1_ps(k_post_coeffs_8u[2])), _mm256_set1_ps(k_post_coeffs_8u[3]));
				v[0] = _mm256_add_ps(_mm256_mul_ps(v[0], _mm256_set1_ps(k_post_coeffs_8u[4])), _mm256_set1_ps(k_post_coeffs_8u[5]));
				v[1] = _mm256_add_ps(_mm256_mul_ps(v[1], _mm256_set1_ps(k_post_coeffs_8u[4])), _mm256_set1_ps(k_post_coeffs_8u[5]));
				
				Li[2] = _mm256_cvtps_epi32( L[0] );
				Li[3] = _mm256_cvtps_epi32( L[1] );
				ui[2] = _mm256_cvtps_epi32( u[0] );
				ui[3] = _mm256_cvtps_epi32( u[1] );
				vi[2] = _mm256_cvtps_epi32( v[0] );
				vi[3] = _mm256_cvtps_epi32( v[1] );
				
				
				dest1 = _mm256_or_si256( dest1, _mm256_shuffle_epi8(Li[2],shuffle_L2a) );
				dest1 = _mm256_or_si256( dest1, _mm256_shuffle_epi8(ui[2],shuffle_u2a) );
				dest1 = _mm256_or_si256( dest1, _mm256_shuffle_epi8(vi[2],shuffle_v2a) );
				
				__m256i dest2 = _mm256_shuffle_epi8(ui[2],shuffle_u2b);
				dest2 = _mm256_or_si256( dest2, _mm256_shuffle_epi8(ui[3],shuffle_u3) );
				
				dest2 = _mm256_or_si256( dest2, _mm256_shuffle_epi8(Li[2],shuffle_L2b) );
				dest2 = _mm256_or_si256( dest2, _mm256_shuffle_epi8(Li[3],shuffle_L3) );
				dest2 = _mm256_or_si256( dest2, _mm256_shuffle_epi8(vi[2],shuffle_v2b) );
				dest2 = _mm256_or_si256( dest2, _mm256_shuffle_epi8(vi[3],shuffle_v3) );
				
				_mm256_storeu2_m128i((__m128i*)tdest+3, (__m128i*)tdest, dest0); tdest+=16;
				_mm256_storeu2_m128i((__m128i*)tdest+3, (__m128i*)tdest, dest1); tdest+=16;
				_mm256_storeu2_m128i((__m128i*)tdest+3, (__m128i*)tdest, dest2); tdest+=64;
				
			}
		}
	}
#else
	{
		GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO32(buff_int2plan);
		
		GENERATE_BUFFERS_PLANAR_TO_INTERLEAVED_32TO8(buff_plan2int);
		
		const unsigned int width = (_pitchs/3) >> 4;
		
		for( int h=_start; h<_stop; ++h ) {
			const unsigned char *tsrc = _src + _pitchd*h;
			unsigned char *tdest = _dest + _pitchd*h;
			for( int w=0; w<width; ++w ) {
				
				// Load src into vectors
				const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc += VEC_ALIGN;
				const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc += VEC_ALIGN;
				const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc += VEC_ALIGN;
				
				// ============================ FIRST 8 COLORS
				LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_32(buff_int2plan);
				
				//__m128i redsi[2], greensi[2], bluesi[2];
				__m128 reds[2], greens[2], blues[2];
				__m128 L[2], u[2], v[2];
				
				// Put colors in place in integer vectors
				SHUFFLE_RGB2PLANAR_LOW(v0, redsi0, greensi0, bluesi0)
				
				SHUFFLE_2X_RGB2PLANAR_LOW(v0, v1, v2, redsi1, greensi1, bluesi1);
				
				// Normalized float values
				reds[0]   = _mm_mul_ps(_mm_cvtepi32_ps(redsi0),  _mm_set1_ps(0.0039215686274509803f));
				reds[1]   = _mm_mul_ps(_mm_cvtepi32_ps(redsi1),  _mm_set1_ps(0.0039215686274509803f));
				greens[0] = _mm_mul_ps(_mm_cvtepi32_ps(greensi0),_mm_set1_ps(0.0039215686274509803f));
				greens[1] = _mm_mul_ps(_mm_cvtepi32_ps(greensi1),_mm_set1_ps(0.0039215686274509803f));
				blues[0]  = _mm_mul_ps(_mm_cvtepi32_ps(bluesi0), _mm_set1_ps(0.0039215686274509803f));
				blues[1]  = _mm_mul_ps(_mm_cvtepi32_ps(bluesi1), _mm_set1_ps(0.0039215686274509803f));
				
				// Do actual conversion
				rgb2luv_helper(reds, greens, blues, L, u, v);
				
				// Post process
				L[0] = _mm_add_ps(_mm_mul_ps(L[0], _mm_set1_ps(k_post_coeffs_8u[0])), _mm_set1_ps(k_post_coeffs_8u[1]));
				L[1] = _mm_add_ps(_mm_mul_ps(L[1], _mm_set1_ps(k_post_coeffs_8u[0])), _mm_set1_ps(k_post_coeffs_8u[1]));
				u[0] = _mm_add_ps(_mm_mul_ps(u[0], _mm_set1_ps(k_post_coeffs_8u[2])), _mm_set1_ps(k_post_coeffs_8u[3]));
				u[1] = _mm_add_ps(_mm_mul_ps(u[1], _mm_set1_ps(k_post_coeffs_8u[2])), _mm_set1_ps(k_post_coeffs_8u[3]));
				v[0] = _mm_add_ps(_mm_mul_ps(v[0], _mm_set1_ps(k_post_coeffs_8u[4])), _mm_set1_ps(k_post_coeffs_8u[5]));
				v[1] = _mm_add_ps(_mm_mul_ps(v[1], _mm_set1_ps(k_post_coeffs_8u[4])), _mm_set1_ps(k_post_coeffs_8u[5]));
				
				// Convert to integer
				__m128i Li[4], ui[4], vi[4];
				Li[0] = _mm_cvtps_epi32( L[0] );
				Li[1] = _mm_cvtps_epi32( L[1] );
				ui[0] = _mm_cvtps_epi32( u[0] );
				ui[1] = _mm_cvtps_epi32( u[1] );
				vi[0] = _mm_cvtps_epi32( v[0] );
				vi[1] = _mm_cvtps_epi32( v[1] );
				
				
				LOAD_BUFFERS_PLANAR_TO_INTERLEAVED_LOW_32(buff_plan2int);
				
				
				__m128i dest0 = _mm_shuffle_epi8(Li[0],shuffle_L0);
				dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(ui[0],shuffle_u0) );
				dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(vi[0],shuffle_v0) );
				
				dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(Li[1],shuffle_L1a) );
				dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(ui[1],shuffle_u1a) );
				dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(vi[1],shuffle_v1a) );
				
				__m128i dest1 = _mm_shuffle_epi8(Li[1],shuffle_L1b);
				dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(ui[1],shuffle_u1b) );
				dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(vi[1],shuffle_v1b) );
				
				
				// ============================ SECOND 8 COLORS
				LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_32(buff_int2plan);
				
				SHUFFLE_2X_RGB2PLANAR_HIGH(v0, v1, v2, redsi2, greensi2, bluesi2);
				
				SHUFFLE_RGB2PLANAR_HIGH(v2, redsi3, greensi3, bluesi3);
				
				
				// Normalized float values
				reds[0]   = _mm_mul_ps(_mm_cvtepi32_ps(redsi2),_mm_set1_ps(0.0039215686274509803f));
				reds[1]   = _mm_mul_ps(_mm_cvtepi32_ps(redsi3),_mm_set1_ps(0.0039215686274509803f));
				greens[0] = _mm_mul_ps(_mm_cvtepi32_ps(greensi2),_mm_set1_ps(0.0039215686274509803f));
				greens[1] = _mm_mul_ps(_mm_cvtepi32_ps(greensi3),_mm_set1_ps(0.0039215686274509803f));
				blues[0]  = _mm_mul_ps(_mm_cvtepi32_ps(bluesi2),_mm_set1_ps(0.0039215686274509803f));
				blues[1]  = _mm_mul_ps(_mm_cvtepi32_ps(bluesi3),_mm_set1_ps(0.0039215686274509803f));
				
				rgb2luv_helper(reds, greens, blues, L, u, v);
				
				L[0] = _mm_add_ps(_mm_mul_ps(L[0], _mm_set1_ps(k_post_coeffs_8u[0])), _mm_set1_ps(k_post_coeffs_8u[1]));
				L[1] = _mm_add_ps(_mm_mul_ps(L[1], _mm_set1_ps(k_post_coeffs_8u[0])), _mm_set1_ps(k_post_coeffs_8u[1]));
				u[0] = _mm_add_ps(_mm_mul_ps(u[0], _mm_set1_ps(k_post_coeffs_8u[2])), _mm_set1_ps(k_post_coeffs_8u[3]));
				u[1] = _mm_add_ps(_mm_mul_ps(u[1], _mm_set1_ps(k_post_coeffs_8u[2])), _mm_set1_ps(k_post_coeffs_8u[3]));
				v[0] = _mm_add_ps(_mm_mul_ps(v[0], _mm_set1_ps(k_post_coeffs_8u[4])), _mm_set1_ps(k_post_coeffs_8u[5]));
				v[1] = _mm_add_ps(_mm_mul_ps(v[1], _mm_set1_ps(k_post_coeffs_8u[4])), _mm_set1_ps(k_post_coeffs_8u[5]));
				
				Li[2] = _mm_cvtps_epi32( L[0] );
				Li[3] = _mm_cvtps_epi32( L[1] );
				ui[2] = _mm_cvtps_epi32( u[0] );
				ui[3] = _mm_cvtps_epi32( u[1] );
				vi[2] = _mm_cvtps_epi32( v[0] );
				vi[3] = _mm_cvtps_epi32( v[1] );
				
				LOAD_BUFFERS_PLANAR_TO_INTERLEAVED_HIGH_32(buff_plan2int);
				
				dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(Li[2],shuffle_L2a) );
				dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(ui[2],shuffle_u2a) );
				dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(vi[2],shuffle_v2a) );
				
				__m128i dest2 = _mm_shuffle_epi8(ui[2],shuffle_u2b);
				dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(ui[3],shuffle_u3) );
				
				dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(Li[2],shuffle_L2b) );
				dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(Li[3],shuffle_L3) );
				dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(vi[2],shuffle_v2b) );
				dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(vi[3],shuffle_v3) );
				
				_mm_store_si128((__m128i*)tdest, dest0); tdest+=VEC_ALIGN;
				_mm_store_si128((__m128i*)tdest, dest1); tdest+=VEC_ALIGN;
				_mm_store_si128((__m128i*)tdest, dest2); tdest+=VEC_ALIGN;
				
			}
		}
	}
#endif
	
}





inline void _luv2rgb_helper(const float* L, const float* u, const float* v,
							float* r, float* g, float* b) {
	
	float x,y,z,t;
	float _L;
	
	if( *L >= 8 ) {
		t = (*L + 16.0f)*0.008620689655172f;
		y = t*t*t;
		_L = *L;
	} else {
		y = *L * 0.001107051920735f;
		_L = ( *L > 0.001f ? *L : 0.001f );
	}
	
	float u1,v1;
	t = 1.0f/(13.0f * (_L));
	u1 = (*u)*t + 0.19793943f;
	v1 = (*v)*t + 0.46831096f;
	x = 2.25f * u1 * y / v1 ;
	z = (12.0f - 3.0f*u1 - 20.0f*v1) * y / (4.0f*v1);
	
	*r =  3.240479f*x - 1.53715f*y  - 0.498535f*z;
	*g = -0.969256f*x + 1.875991f*y + 0.041556f*z;
	*b =  0.055648f*x - 0.204043f*y + 1.057311f*z;
	
}


void _luv2rgb_8u(const unsigned char* _src, unsigned char* _dest, unsigned int _width,
				 unsigned int _pitchs, unsigned int _pitchd,
				 unsigned int _start, unsigned int _stop) {
	
	
	for( unsigned int h=_start; h<_stop; ++h ) {
		const unsigned char* tsrc = _src+h*_pitchs;
		unsigned char* tdest = _dest+h*_pitchd;
		for( unsigned int w=0; w<_width; ++w ) {
			unsigned char L = tsrc[0];
			unsigned char u = tsrc[1];
			unsigned char v = tsrc[2]; tsrc+=3;
			
			float Lf = k_8u_32f[(int)L]*k_pre_coeffs[0] + k_pre_coeffs[1];
			float uf = k_8u_32f[(int)u]*k_pre_coeffs[2] + k_pre_coeffs[3];
			float vf = k_8u_32f[(int)v]*k_pre_coeffs[4] + k_pre_coeffs[5];
			
			float r,g,b;
			
			_luv2rgb_helper(&Lf, &uf, &vf, &r, &g, &b);
			
			int ri = (int)(255.0f*r + (r >= 0.0f ? 0.5f : -0.5f));
			ri = (ri < 0 ? 0 : ri);
			int gi = (int)(255.0f*g + (g >= 0.0f ? 0.5f : -0.5f));
			gi = (gi < 0 ? 0 : gi);
			int bi = (int)(255.0f*b + (b >= 0.0f ? 0.5f : -0.5f));
			bi = (bi < 0 ? 0 : bi);
			
			tdest[0] = static_cast<unsigned char>(ri>255?255:ri);
			tdest[1] = static_cast<unsigned char>(gi>255?255:gi);
			tdest[2] = static_cast<unsigned char>(bi>255?255:bi); tdest+=3;
		}
	}
	
}

