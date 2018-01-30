#ifndef _SSEHELPERS_H_
#define _SSEHELPERS_H_

#define USE_SSE (1)

#define SOURCE_RGB ( 2)
#define SOURCE_BGR (-2)

// =============================================================
// ================== LOAD SOURCE SHUFFLES =====================
// =============================================================
#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_16(buff) \
LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_16_(buff,SOURCE_RGB)

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_16_BGR(buff) \
LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_16_(buff,SOURCE_BGR)

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_16(buff) \
LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_16_(buff,SOURCE_RGB)

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_16_BGR(buff) \
LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_16_(buff,SOURCE_BGR)

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_32(buff) \
LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_32_(buff,SOURCE_RGB)

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_32_BGR(buff) \
LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_32_(buff,SOURCE_BGR)

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_32(buff) \
LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_32_(buff,SOURCE_RGB)

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_32_BGR(buff) \
LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_32_(buff,SOURCE_BGR)



#ifdef USE_AVX2

#define GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO16(buff) \
ALIGNED_BUFFER(unsigned long long, buff, 48); \
\
buff[0]  = 0x8009800680038000ull; buff[1]  = 0x80808080800f800cull; buff[2]  = 0x8009800680038000ull; buff[3]  = 0x80808080800f800cull;  \
buff[4]  = 0x800a800780048001ull; buff[5]  = 0x808080808080800dull; buff[6]  = 0x800a800780048001ull; buff[7]  = 0x808080808080800dull;  \
buff[8]  = 0x800b800880058002ull; buff[9]  = 0x808080808080800eull; buff[10] = 0x800b800880058002ull; buff[11] = 0x808080808080800eull;  \
\
buff[12] = 0x8080808080808080ull; buff[13] = 0x8005800280808080ull; buff[14] = 0x8080808080808080ull; buff[15] = 0x8005800280808080ull;  \
buff[16] = 0x8080808080808080ull; buff[17] = 0x8006800380008080ull; buff[18] = 0x8080808080808080ull; buff[19] = 0x8006800380008080ull;  \
buff[20] = 0x8080808080808080ull; buff[21] = 0x8007800480018080ull; buff[22] = 0x8080808080808080ull; buff[23] = 0x8007800480018080ull;  \
\
buff[24] = 0x8080800e800b8008ull; buff[25] = 0x8080808080808080ull; buff[26] = 0x8080800e800b8008ull; buff[27] = 0x8080808080808080ull;  \
buff[28] = 0x8080800f800c8009ull; buff[29] = 0x8080808080808080ull; buff[30] = 0x8080800f800c8009ull; buff[31] = 0x8080808080808080ull;  \
buff[32] = 0x80808080800d800aull; buff[33] = 0x8080808080808080ull; buff[34] = 0x80808080800d800aull; buff[35] = 0x8080808080808080ull;  \
\
buff[36] = 0x8001808080808080ull; buff[37] = 0x800d800a80078004ull; buff[38] = 0x8001808080808080ull; buff[39] = 0x800d800a80078004ull;  \
buff[40] = 0x8002808080808080ull; buff[41] = 0x800e800b80088005ull; buff[42] = 0x8002808080808080ull; buff[43] = 0x800e800b80088005ull;  \
buff[44] = 0x8003800080808080ull; buff[45] = 0x800f800c80098006ull; buff[46] = 0x8003800080808080ull; buff[47] = 0x800f800c80098006ull;  \

	
#define GENERATE_BUFFERS_PLANAR_TO_INTERLEAVED(buff) \
ALIGNED_BUFFER(unsigned long long, buff, 48); \
\
buff[0]  = 0x8004808002808000ull; buff[1]  = 0x0a80800880800680ull; buff[2]  = 0x8004808002808000ull; buff[3]  = 0x0a80800880800680ull; \
buff[4]  = 0x80800e80800c8080ull; buff[5]  = 0x8080808080808080ull; buff[6]  = 0x80800e80800c8080ull; buff[7]  = 0x8080808080808080ull; \
buff[8]  = 0x8080808080808080ull; buff[9]  = 0x8004808002808000ull; buff[10] = 0x8080808080808080ull; buff[11] = 0x8004808002808000ull; \
buff[12] = 0x0a80800880800680ull; buff[13] = 0x80800e80800c8080ull; buff[14] = 0x0a80800880800680ull; buff[15] = 0x80800e80800c8080ull; \
\
buff[16] = 0x0480800280800080ull; buff[17] = 0x8080088080068080ull; buff[18] = 0x0480800280800080ull; buff[19] = 0x8080088080068080ull; \
buff[20] = 0x800e80800c80800aull; buff[21] = 0x8080808080808080ull; buff[22] = 0x800e80800c80800aull; buff[23] = 0x8080808080808080ull; \
buff[24] = 0x8080808080808080ull; buff[25] = 0x0480800280800080ull; buff[26] = 0x8080808080808080ull; buff[27] = 0x0480800280800080ull; \
buff[28] = 0x8080088080068080ull; buff[29] = 0x800e80800c80800aull; buff[30] = 0x8080088080068080ull; buff[31] = 0x800e80800c80800aull; \
\
buff[32] = 0x8080028080008080ull; buff[33] = 0x8008808006808004ull; buff[34] = 0x8080028080008080ull; buff[35] = 0x8008808006808004ull; \
buff[36] = 0x0e80800c80800a80ull; buff[37] = 0x8080808080808080ull; buff[38] = 0x0e80800c80800a80ull; buff[39] = 0x8080808080808080ull; \
buff[40] = 0x8080808080808080ull; buff[41] = 0x8080028080008080ull; buff[42] = 0x8080808080808080ull; buff[43] = 0x8080028080008080ull; \
buff[44] = 0x8008808006808004ull; buff[45] = 0x0e80800c80800a80ull; buff[46] = 0x8008808006808004ull; buff[47] = 0x0e80800c80800a80ull; \

	
#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW(buff) \
const __m256i mLowRv0 = _mm256_load_si256((const __m256i*)buff+0); \
const __m256i mLowGv0 = _mm256_load_si256((const __m256i*)buff+1); \
const __m256i mLowBv0 = _mm256_load_si256((const __m256i*)buff+2); \
const __m256i mLowRv1 = _mm256_load_si256((const __m256i*)buff+3); \
const __m256i mLowGv1 = _mm256_load_si256((const __m256i*)buff+4); \
const __m256i mLowBv1 = _mm256_load_si256((const __m256i*)buff+5);
	
#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH(buff) \
const __m256i mHighRv0 = _mm256_load_si256( (const __m256i*)buff+6 ); \
const __m256i mHighGv0 = _mm256_load_si256( (const __m256i*)buff+7 ); \
const __m256i mHighBv0 = _mm256_load_si256( (const __m256i*)buff+8 ); \
const __m256i mHighRv1 = _mm256_load_si256( (const __m256i*)buff+9 ); \
const __m256i mHighGv1 = _mm256_load_si256( (const __m256i*)buff+10); \
const __m256i mHighBv1 = _mm256_load_si256( (const __m256i*)buff+11);
	
	
#define LOAD_BUFFERS_PLANAR_TO_INTERLEAVED(buff) \
const __m256i packh_1 = _mm256_load_si256((const __m256i*)buff+0); \
const __m256i packh_2 = _mm256_load_si256((const __m256i*)buff+1); \
const __m256i packh_3 = _mm256_load_si256((const __m256i*)buff+2); \
const __m256i packh_4 = _mm256_load_si256((const __m256i*)buff+3); \
\
const __m256i packs_1 = _mm256_load_si256((const __m256i*)buff+4); \
const __m256i packs_2 = _mm256_load_si256((const __m256i*)buff+5); \
const __m256i packs_3 = _mm256_load_si256((const __m256i*)buff+6); \
const __m256i packs_4 = _mm256_load_si256((const __m256i*)buff+7); \
\
const __m256i packv_1 = _mm256_load_si256((const __m256i*)buff+8); \
const __m256i packv_2 = _mm256_load_si256((const __m256i*)buff+9); \
const __m256i packv_3 = _mm256_load_si256((const __m256i*)buff+10); \
const __m256i packv_4 = _mm256_load_si256((const __m256i*)buff+11); \

#define SHUFFLE_RGB2PLANAR_LOW(v0,v1,v2,r,g,b) \
const __m256i r = _mm256_or_si256( _mm256_shuffle_epi8(v0,mLowRv0) , _mm256_shuffle_epi8(v1,mLowRv1) ); \
const __m256i g = _mm256_or_si256( _mm256_shuffle_epi8(v0,mLowGv0) , _mm256_shuffle_epi8(v1,mLowGv1) ); \
const __m256i b = _mm256_or_si256( _mm256_shuffle_epi8(v0,mLowBv0) , _mm256_shuffle_epi8(v1,mLowBv1) );
	
#define SHUFFLE_RGB2PLANAR_HIGH(v0,v1,v2,r,g,b) \
const __m256i r = _mm256_or_si256( _mm256_shuffle_epi8(v1,mHighRv0) , _mm256_shuffle_epi8(v2,mHighRv1) ); \
const __m256i g = _mm256_or_si256( _mm256_shuffle_epi8(v1,mHighGv0) , _mm256_shuffle_epi8(v2,mHighGv1) ); \
const __m256i b = _mm256_or_si256( _mm256_shuffle_epi8(v1,mHighBv0) , _mm256_shuffle_epi8(v2,mHighBv1) );


#elif defined(USE_SSE)

// =============================================================
// ================== FILL SHUFFLE BUFFERS =====================
// =============================================================
#define GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO16(buff) \
ALIGNED_BUFFER(unsigned long long, buff, 24); \
\
buff[0]  = 0x8009800680038000ull; buff[1]  = 0x80808080800f800cull; \
buff[4]  = 0x800a800780048001ull; buff[5]  = 0x808080808080800dull; \
buff[8]  = 0x800b800880058002ull; buff[9]  = 0x808080808080800eull; \
\
buff[2]  = 0x8080808080808080ull; buff[3]  = 0x8005800280808080ull; \
buff[6]  = 0x8080808080808080ull; buff[7]  = 0x8006800380008080ull; \
buff[10] = 0x8080808080808080ull; buff[11] = 0x8007800480018080ull; \
\
buff[12] = 0x8080800e800b8008ull; buff[13] = 0x8080808080808080ull; \
buff[16] = 0x8080800f800c8009ull; buff[17] = 0x8080808080808080ull; \
buff[20] = 0x80808080800d800aull; buff[21] = 0x8080808080808080ull; \
\
buff[14] = 0x8001808080808080ull; buff[15] = 0x800d800a80078004ull; \
buff[18] = 0x8002808080808080ull; buff[19] = 0x800e800b80088005ull; \
buff[22] = 0x8003800080808080ull; buff[23] = 0x800f800c80098006ull;


#define GENERATE_BUFFERS_INTERLEAVED_TO_PLANAR_8TO32(buff) \
ALIGNED_BUFFER(unsigned long long, buff, 36); \
\
buff[0]  = 0x8080800380808000ull; buff[1]  = 0x8080800980808006ull;  /* Puts red colors from v0 in r1 */   \
buff[2]  = 0x8080800480808001ull; buff[3]  = 0x8080800a80808007ull;  /* Puts green colors from v0 in g1 */ \
buff[4]  = 0x8080800580808002ull; buff[5]  = 0x8080800b80808008ull;  /* Puts blue colors from v0 in b1 */  \
\
buff[6]  = 0x8080800f8080800cull; buff[7]  = 0x8080808080808080ull;  /* Puts red colors from v0 in r2 */   \
buff[8] =  0x8080808080808080ull; buff[9] =  0x8080800580808002ull;  /* Puts red colors from v1 in r2 */   \
buff[10] = 0x808080808080800dull; buff[11] = 0x8080808080808080ull;  /* Puts green colors from v0 in g2 */ \
buff[12] = 0x8080800080808080ull; buff[13] = 0x8080800680808003ull;  /* Puts green colors from v1 in g2 */ \
buff[14] = 0x808080808080800eull; buff[15] = 0x8080808080808080ull;  /* Puts blue colors from v0 in b2 */  \
buff[16] = 0x8080800180808080ull; buff[17] = 0x8080800780808004ull;  /* Puts blue colors from v1 in b2 */  \
\
buff[18] = 0x8080800b80808008ull; buff[19] = 0x808080808080800eull;  /* Puts red colors from v1 in r3 */   \
buff[20] = 0x8080808080808080ull; buff[21] = 0x8080800180808080ull;  /* Puts red colors from v2 in r3 */   \
buff[22] = 0x8080800c80808009ull; buff[23] = 0x808080808080800full;  /* Puts green colors from v1 in g3 */ \
buff[24] = 0x8080808080808080ull; buff[25] = 0x8080800280808080ull;  /* Puts green colors from v2 in g3 */ \
buff[26] = 0x8080800d8080800aull; buff[27] = 0x8080808080808080ull;  /* Puts blue colors from v1 in b3 */  \
buff[28] = 0x8080808080808080ull; buff[29] = 0x8080800380808000ull;  /* Puts blue colors from v2 in b3 */  \
\
buff[30] = 0x8080800780808004ull; buff[31] = 0x8080800d8080800aull;  /* Puts red colors from v2 in r4 */   \
buff[32] = 0x8080800880808005ull; buff[33] = 0x8080800e8080800bull;  /* Puts green colors from v2 in g4 */ \
buff[34] = 0x8080800980808006ull; buff[35] = 0x8080800f8080800cull;  /* Puts blue colors from v2 in b4 */

#define GENERATE_BUFFERS_PLANAR_TO_INTERLEAVED_16TO8(buff) \
ALIGNED_BUFFER(unsigned long long, buff, 24); \
\
buff[0]  = 0x8004808002808000ull; buff[1]  = 0x0a80800880800680ull; \
buff[2]  = 0x80800e80800c8080ull; buff[3]  = 0x8080808080808080ull; \
buff[4]  = 0x8080808080808080ull; buff[5]  = 0x8004808002808000ull; \
buff[6]  = 0x0a80800880800680ull; buff[7]  = 0x80800e80800c8080ull; \
\
buff[8]  = 0x0480800280800080ull; buff[9]  = 0x8080088080068080ull; \
buff[10] = 0x800e80800c80800aull; buff[11] = 0x8080808080808080ull; \
buff[12] = 0x8080808080808080ull; buff[13] = 0x0480800280800080ull; \
buff[14] = 0x8080088080068080ull; buff[15] = 0x800e80800c80800aull; \
\
buff[16] = 0x8080028080008080ull; buff[17] = 0x8008808006808004ull; \
buff[18] = 0x0e80800c80800a80ull; buff[19] = 0x8080808080808080ull; \
buff[20] = 0x8080808080808080ull; buff[21] = 0x8080028080008080ull; \
buff[22] = 0x8008808006808004ull; buff[23] = 0x0e80800c80800a80ull; \


#define GENERATE_BUFFERS_PLANAR_TO_INTERLEAVED_32TO8(buff) \
ALIGNED_BUFFER(unsigned long long, buff, 36); \
\
buff[0]  = 0x8008808004808000ull; buff[1]  = 0x8080808080800c80ull;    /* Puts L0 in dest0 */ \
buff[2]  = 0x0880800480800080ull; buff[3]  = 0x80808080800c8080ull;    /* Puts u0 in dest0 */ \
buff[4]  = 0x8080048080008080ull; buff[5]  = 0x808080800c808008ull;    /* Puts v0 in dest0 */ \
\
buff[6]  = 0x8080808080808080ull; buff[7]  = 0x0480800080808080ull;    /* Puts L1 in dest0 */ \
buff[8]  = 0x8080808080808080ull; buff[9]  = 0x8080008080808080ull;    /* Puts u1 in dest0 */ \
buff[10] = 0x8080808080808080ull; buff[11] = 0x8000808080808080ull;    /* Puts v1 in dest0 */ \
\
buff[12] = 0x80800c8080088080ull; buff[13] = 0x8080808080808080ull;    /* Puts L1 in dest1 */ \
buff[14] = 0x800c808008808004ull; buff[15] = 0x8080808080808080ull;    /* Puts u1 in dest1 */ \
buff[16] = 0x0c80800880800480ull; buff[17] = 0x8080808080808080ull;    /* Puts v1 in dest1 */ \
\
buff[18] = 0x8080808080808080ull; buff[19] = 0x8008808004808000ull;    /* Puts L2 in dest1 */ \
buff[20] = 0x8080808080808080ull; buff[21] = 0x0880800480800080ull;    /* Puts u2 in dest1 */ \
buff[22] = 0x8080808080808080ull; buff[23] = 0x8080048080008080ull;    /* Puts v2 in dest1 */ \
\
buff[24] = 0x8080808080800c80ull; buff[25] = 0x8080808080808080ull;    /* Puts L2 in dest2 */ \
buff[26] = 0x80808080800c8080ull; buff[27] = 0x8080808080808080ull;    /* Puts u2 in dest2 */ \
buff[28] = 0x808080800c808008ull; buff[29] = 0x8080808080808080ull;    /* Puts v2 in dest2 */ \
\
buff[30] = 0x0480800080808080ull; buff[31] = 0x80800c8080088080ull;    /* Puts L3 in dest2 */ \
buff[32] = 0x8080008080808080ull; buff[33] = 0x800c808008808004ull;    /* Puts u3 in dest2 */ \
buff[34] = 0x8000808080808080ull; buff[35] = 0x0c80800880800480ull;    /* Puts v3 in dest2 */


// =============================================================
// ================== LOAD SOURCE SHUFFLES =====================
// =============================================================
#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_16_(buff,shift) \
const __m128i red_lo_v0   = _mm_load_si128((const __m128i*)buff+2-shift); \
const __m128i red_lo_v1   = _mm_load_si128((const __m128i*)buff+3-shift); \
const __m128i green_lo_v0 = _mm_load_si128((const __m128i*)buff+2); \
const __m128i green_lo_v1 = _mm_load_si128((const __m128i*)buff+3); \
const __m128i blue_lo_v0  = _mm_load_si128((const __m128i*)buff+2+shift); \
const __m128i blue_lo_v1  = _mm_load_si128((const __m128i*)buff+3+shift);

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_16_(buff,shift) \
const __m128i red_hi_v1   = _mm_load_si128( (const __m128i*)buff+8-shift ); \
const __m128i red_hi_v2   = _mm_load_si128( (const __m128i*)buff+9-shift ); \
const __m128i green_hi_v1 = _mm_load_si128( (const __m128i*)buff+8 ); \
const __m128i green_hi_v2 = _mm_load_si128( (const __m128i*)buff+9 ); \
const __m128i blue_hi_v1  = _mm_load_si128( (const __m128i*)buff+8+shift); \
const __m128i blue_hi_v2  = _mm_load_si128( (const __m128i*)buff+9+shift);

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_LOW_32_(buff,shift) \
const __m128i red_lo   = _mm_load_si128((const __m128i*)buff+1-shift/2); \
const __m128i green_lo = _mm_load_si128((const __m128i*)buff+1); \
const __m128i blue_lo  = _mm_load_si128((const __m128i*)buff+1+shift/2); \
\
const __m128i red_lo_v0   = _mm_load_si128((const __m128i*)buff+5-shift); \
const __m128i red_lo_v1   = _mm_load_si128((const __m128i*)buff+6-shift); \
const __m128i green_lo_v0 = _mm_load_si128((const __m128i*)buff+5); \
const __m128i green_lo_v1 = _mm_load_si128((const __m128i*)buff+6); \
const __m128i blue_lo_v0  = _mm_load_si128((const __m128i*)buff+5+shift); \
const __m128i blue_lo_v1  = _mm_load_si128((const __m128i*)buff+6+shift); \

#define LOAD_BUFFERS_INTERLEAVED_TO_PLANAR_HIGH_32_(buff,shift) \
const __m128i red_hi_v1   = _mm_load_si128((const __m128i*)buff+11-shift); \
const __m128i red_hi_v2   = _mm_load_si128((const __m128i*)buff+12-shift); \
const __m128i green_hi_v1 = _mm_load_si128((const __m128i*)buff+11); \
const __m128i green_hi_v2 = _mm_load_si128((const __m128i*)buff+12); \
const __m128i blue_hi_v1  = _mm_load_si128((const __m128i*)buff+11+shift); \
const __m128i blue_hi_v2  = _mm_load_si128((const __m128i*)buff+12+shift); \
\
const __m128i red_hi   = _mm_load_si128((const __m128i*)buff+16-shift/2); \
const __m128i green_hi = _mm_load_si128((const __m128i*)buff+16); \
const __m128i blue_hi  = _mm_load_si128((const __m128i*)buff+16+shift/2);


// =============================================================
// ===================== LOAD DEST SHUFFLES ====================
// =============================================================
#define LOAD_BUFFERS_PLANAR_TO_INTERLEAVED_16(buff) \
const __m128i packh_1 = _mm_load_si128((const __m128i*)buff+0); \
const __m128i packh_2 = _mm_load_si128((const __m128i*)buff+1); \
const __m128i packh_3 = _mm_load_si128((const __m128i*)buff+2); \
const __m128i packh_4 = _mm_load_si128((const __m128i*)buff+3); \
\
const __m128i packs_1 = _mm_load_si128((const __m128i*)buff+4); \
const __m128i packs_2 = _mm_load_si128((const __m128i*)buff+5); \
const __m128i packs_3 = _mm_load_si128((const __m128i*)buff+6); \
const __m128i packs_4 = _mm_load_si128((const __m128i*)buff+7); \
\
const __m128i packv_1 = _mm_load_si128((const __m128i*)buff+8); \
const __m128i packv_2 = _mm_load_si128((const __m128i*)buff+9); \
const __m128i packv_3 = _mm_load_si128((const __m128i*)buff+10); \
const __m128i packv_4 = _mm_load_si128((const __m128i*)buff+11); \


#define LOAD_BUFFERS_PLANAR_TO_INTERLEAVED_LOW_32(buff) \
const __m128i shuffle_L0 = _mm_load_si128((const __m128i*)buff+0); \
const __m128i shuffle_u0 = _mm_load_si128((const __m128i*)buff+1); \
const __m128i shuffle_v0 = _mm_load_si128((const __m128i*)buff+2); \
\
const __m128i shuffle_L1a = _mm_load_si128((const __m128i*)buff+3); \
const __m128i shuffle_L1b = _mm_load_si128((const __m128i*)buff+6); \
const __m128i shuffle_u1a = _mm_load_si128((const __m128i*)buff+4); \
const __m128i shuffle_u1b = _mm_load_si128((const __m128i*)buff+7); \
const __m128i shuffle_v1a = _mm_load_si128((const __m128i*)buff+5); \
const __m128i shuffle_v1b = _mm_load_si128((const __m128i*)buff+8);


#define LOAD_BUFFERS_PLANAR_TO_INTERLEAVED_HIGH_32(buff) \
const __m128i shuffle_L2a = _mm_load_si128((const __m128i*)buff+9); \
const __m128i shuffle_L2b = _mm_load_si128((const __m128i*)buff+12); \
const __m128i shuffle_u2a = _mm_load_si128((const __m128i*)buff+10); \
const __m128i shuffle_u2b = _mm_load_si128((const __m128i*)buff+13); \
const __m128i shuffle_v2a = _mm_load_si128((const __m128i*)buff+11); \
const __m128i shuffle_v2b = _mm_load_si128((const __m128i*)buff+14); \
\
const __m128i shuffle_L3 = _mm_load_si128((const __m128i*)buff+15); \
const __m128i shuffle_u3 = _mm_load_si128((const __m128i*)buff+16); \
const __m128i shuffle_v3 = _mm_load_si128((const __m128i*)buff+17); \


#define SHUFFLE_RGB2PLANAR_LOW(v,r,g,b) \
const __m128i r = _mm_shuffle_epi8(v,red_lo); \
const __m128i g = _mm_shuffle_epi8(v,green_lo); \
const __m128i b = _mm_shuffle_epi8(v,blue_lo);

#define SHUFFLE_RGB2PLANAR_HIGH(v,r,g,b) \
const __m128i r = _mm_shuffle_epi8(v,red_hi); \
const __m128i g = _mm_shuffle_epi8(v,green_hi); \
const __m128i b = _mm_shuffle_epi8(v,blue_hi);


#define SHUFFLE_2X_RGB2PLANAR_LOW(v0,v1,v2,r,g,b) \
const __m128i r = _mm_or_si128( _mm_shuffle_epi8(v0,red_lo_v0) , _mm_shuffle_epi8(v1,red_lo_v1) ); \
const __m128i g = _mm_or_si128( _mm_shuffle_epi8(v0,green_lo_v0) , _mm_shuffle_epi8(v1,green_lo_v1) ); \
const __m128i b = _mm_or_si128( _mm_shuffle_epi8(v0,blue_lo_v0) , _mm_shuffle_epi8(v1,blue_lo_v1) );

#define SHUFFLE_2X_RGB2PLANAR_HIGH(v0,v1,v2,r,g,b) \
const __m128i r = _mm_or_si128( _mm_shuffle_epi8(v1,red_hi_v1) , _mm_shuffle_epi8(v2,red_hi_v2) ); \
const __m128i g = _mm_or_si128( _mm_shuffle_epi8(v1,green_hi_v1) , _mm_shuffle_epi8(v2,green_hi_v2) ); \
const __m128i b = _mm_or_si128( _mm_shuffle_epi8(v1,blue_hi_v1) , _mm_shuffle_epi8(v2,blue_hi_v2) );

#endif

#endif // _SSEHELPERS_H_
