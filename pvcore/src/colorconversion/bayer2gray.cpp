
#include <iostream>

#include <immintrin.h>

namespace pvcore {
    
    typedef int BAYER_LAYOUT;
#define RGGB 0
#define BGGR 1
#define GRBG 2
#define GBRG 3
    
    
    // =============================================================
    // ====================== BAYER2GRAY_8U =========================
    // =============================================================
    
    template <BAYER_LAYOUT LAYOUT, bool HAS_ALPHA>
    class BAYER2GRAY_8U_TBB {
        
        const unsigned char *src;
        unsigned char *dest;
        unsigned int width;
        unsigned int height;
        unsigned int pitchs;
        unsigned int pitchd;
        unsigned int threads;
        
    private:
        void _bayer2gray_8u(unsigned int, unsigned int) const;
        
    public:
        inline void operator()( const tbb::blocked_range<size_t>& r ) const {
            
            // Let's make the last thread do the least work
            unsigned int blockSize = ((height-2)+threads-1)/threads;
            
            if( blockSize % 2 == 1 ) blockSize += 1;
            
            unsigned int start = (unsigned int)r.begin()*blockSize+1;
            
            if( start >= height-2 ) return;
            
            unsigned int stop  = (unsigned int)r.end()*blockSize;
            if( stop >= height-1 ) {
                stop = height-1;
            }
            
            _bayer2gray_8u( start, stop );
        }
        
        BAYER2GRAY_8U_TBB( const unsigned char* _src, unsigned char* _dest,
                         unsigned int _width, unsigned int _height,
                         unsigned int _pitchs, unsigned int _pitchd, unsigned int _threads ) :
        src(_src), dest(_dest), width(_width), height(_height), pitchs(_pitchs), pitchd(_pitchd), threads(_threads) {
            
        }
    };
    
    
    
    
    template <BAYER_LAYOUT LAYOUT, bool HAS_ALPHA>
    void BAYER2GRAY_8U_TBB<LAYOUT,HAS_ALPHA>::_bayer2gray_8u(unsigned int _start,
                                                           unsigned int _stop ) const {
        
        unsigned long long buff[2];
        buff[0] = 0x0908060504020100ull; buff[1] = 0x808080800e0d0c0aull;
        __m128i shuffle = _mm_load_si128((const __m128i*)buff);
        
        const int destoffset = (HAS_ALPHA ? 16 : 12);
        const int srcshift = (LAYOUT == GBRG || LAYOUT == GRBG ? -pitchs : 0);
        const int destshift = (LAYOUT == GBRG || LAYOUT == GRBG ? 0 : pitchd);
        
        for( int y=_start; y<_stop; y+=2 ) {
            // Reads 16 pixels (1 + 14 + 1)
            const unsigned char* tsrc = src + y*pitchs;
            // Skipps first pixel
            unsigned char* tdest = dest + y*pitchd;
            tdest[0] = tdest[1] = tdest[2] = 0; tdest += 3;
            int x=0;
            for( ; x<pitchs-16; x+=14 ) {
                
                if( LAYOUT == RGGB || LAYOUT == BGGR ) {
                    
                    __m128i v1 = _mm_loadu_si128((const __m128i*)(tsrc - pitchs));
                    __m128i v2 = _mm_loadu_si128((const __m128i*)tsrc);
                    __m128i v3 = _mm_loadu_si128((const __m128i*)(tsrc + pitchs));
                    
                    // Row - 1
                    __m128i v1r = _mm_and_si128(v1, _mm_set1_epi16(0x00FF)); // LSB
                    __m128i v1g = _mm_and_si128(v1, _mm_set1_epi16(0xFF00)); // MSB
                    // Current row
                    __m128i v2g = _mm_and_si128(v2, _mm_set1_epi16(0x00FF));
                    __m128i v2b = _mm_and_si128(v2, _mm_set1_epi16(0xFF00));
                    // Row + 1
                    __m128i v3r = _mm_and_si128(v3, _mm_set1_epi16(0x00FF));
                    __m128i v3g = _mm_and_si128(v3, _mm_set1_epi16(0xFF00));
                    
                    // Red even
                    __m128i vr = _mm_add_epi16(v1r, v3r);
                    // Red odd
                    __m128i vr_shift2r = _mm_srli_si128(vr, 2);
                    __m128i add_t = _mm_add_epi16(vr,vr_shift2r);
                    __m128i vrt = _mm_slli_epi16(add_t, 6);
                    // Red combined
                    vr = _mm_srli_si128(_mm_add_epi8(_mm_srli_epi16(vr,1),_mm_and_si128(_mm_set1_epi16(0xFF00), vrt)),1);
                    
                    
                    // Green even
                    __m128i vg = v2g;
                    // Green odd
                    __m128i green_shift2r = _mm_srli_si128(v2g, 2);
                    add_t = _mm_add_epi16(v2g,green_shift2r);
                    add_t = _mm_add_epi16(_mm_add_epi16(_mm_srli_si128(v1g,1),_mm_srli_si128(v3g,1)), add_t);
                    add_t = _mm_add_epi16(_mm_set1_epi16(2), add_t);
                    vrt = _mm_slli_epi16(add_t, 6);
                    // Green combined
                    vg = _mm_srli_si128(_mm_add_epi8(vg, _mm_and_si128(_mm_set1_epi16(0xFF00), vrt)),1);
                    
                    // Blue even
                    __m128i blue_shift3r = _mm_srli_si128(v2b, 3);
                    add_t = _mm_add_epi16(_mm_set1_epi16(1),_mm_add_epi16(_mm_srli_si128(v2b, 1),blue_shift3r));
                    vrt = _mm_srli_epi16(add_t, 1);
                    // Blue Combined
                    __m128i vb = _mm_srli_si128(_mm_add_epi8(v2b, _mm_slli_si128(vrt,2)),1);
                    
                    
                    
                    
                    
                    
                    __m128i rgl = (LAYOUT == RGGB ? _mm_unpacklo_epi8(vr,vg) : _mm_unpacklo_epi8(vb,vg));
                    __m128i rgh = (LAYOUT == RGGB ? _mm_unpackhi_epi8(vr,vg) : _mm_unpackhi_epi8(vb,vg));
                    
                    __m128i bxl = (LAYOUT == RGGB ? _mm_unpacklo_epi8(vb, _mm_set1_epi16(0)) : _mm_unpacklo_epi8(vr, _mm_set1_epi16(0)));
                    __m128i bxh = (LAYOUT == RGGB ? _mm_unpackhi_epi8(vb, _mm_set1_epi16(0)) : _mm_unpackhi_epi8(vr, _mm_set1_epi16(0)));
                    
                    __m128i tdestxll = _mm_shuffle_epi8(_mm_unpacklo_epi16(rgl, bxl), shuffle);
                    __m128i tdestxlh = _mm_shuffle_epi8(_mm_unpackhi_epi16(rgl, bxl), shuffle);
                    __m128i tdestxhl = _mm_shuffle_epi8(_mm_unpacklo_epi16(rgh, bxh), shuffle);
                    __m128i tdestxhh = _mm_shuffle_epi8(_mm_unpackhi_epi16(rgh, bxh), shuffle);
                    
                    _mm_storeu_si128((__m128i*)tdest, tdestxll);
                    _mm_storeu_si128((__m128i*)(tdest+destoffset), tdestxlh);
                    _mm_storeu_si128((__m128i*)(tdest+destoffset*2), tdestxhl);
                    _mm_storeu_si128((__m128i*)(tdest+destoffset*3), tdestxhh);
                }
                
                { // THIS BLOCK ALWAYS BEGINS WITH RED OR BLUE
                    __m128i v1 = _mm_loadu_si128((const __m128i*)(tsrc + srcshift));
                    __m128i v2 = _mm_loadu_si128((const __m128i*)(tsrc + srcshift + pitchs));
                    __m128i v3 = _mm_loadu_si128((const __m128i*)(tsrc + srcshift + 2*pitchs));
                    
                    // Row - 1
                    __m128i v1g = _mm_and_si128(v1, _mm_set1_epi16(0x00FF));
                    __m128i v1b = _mm_and_si128(v1, _mm_set1_epi16(0xFF00));
                    // Current row
                    __m128i v2r = _mm_and_si128(v2, _mm_set1_epi16(0x00FF));
                    __m128i v2g = _mm_and_si128(v2, _mm_set1_epi16(0xFF00));
                    // Row + 1
                    __m128i v3g = _mm_and_si128(v3, _mm_set1_epi16(0x00FF));
                    __m128i v3b = _mm_and_si128(v3, _mm_set1_epi16(0xFF00));
                    
                    // Red even
                    __m128i vr = v2r;
                    // Red odd
                    __m128i shuffle_t = _mm_srli_si128(vr, 2);
                    __m128i add_t = _mm_add_epi16(vr,shuffle_t);
                    __m128i vrt = _mm_slli_epi16(add_t, 7);
                    // Red combined
                    vr = _mm_srli_si128(_mm_add_epi8(vr,_mm_and_si128(_mm_set1_epi16(0xFF00), vrt)),1);
                    
                    
                    // Green odd
                    __m128i vg = v2g;
                    // Green even
                    add_t = _mm_add_epi16(_mm_slli_si128(v2g,1),_mm_srli_si128(vg, 1));
                    add_t = _mm_add_epi16(_mm_add_epi16(v1g, v3g), add_t);
                    vrt = _mm_srli_epi16(add_t, 2);
                    // Green combined
                    vg = _mm_srli_si128(_mm_add_epi8(vg, _mm_and_si128(_mm_set1_epi16(0x00FF), _mm_slli_si128(vrt,0))),1);
                    
                    // Blue odd
                    v1b = _mm_srli_si128(v1b,1);
                    v3b = _mm_srli_si128(v3b,1);
                    __m128i vb = _mm_add_epi16(v1b, v3b);
                    // Blue even
                    __m128i blue_shift2left = _mm_srli_si128(vb, 2);
                    add_t = _mm_add_epi16(vb,blue_shift2left);
                    vrt = _mm_srli_epi16(_mm_add_epi16(_mm_set1_epi16(2), add_t), 2);
                    // Blue combined
                    vb = _mm_srli_si128(_mm_add_epi8(_mm_and_si128(_mm_set1_epi16(0xFF00),_mm_slli_epi16(vb, 7)), _mm_and_si128(_mm_set1_epi16(0x00FF), _mm_slli_si128(vrt,2))),1);
                    
                    __m128i rgl = (LAYOUT == RGGB || LAYOUT == GBRG? _mm_unpacklo_epi8(vr,vg) : _mm_unpacklo_epi8(vb,vg));
                    __m128i rgh = (LAYOUT == RGGB || LAYOUT == GBRG ? _mm_unpackhi_epi8(vr,vg) : _mm_unpackhi_epi8(vb,vg));
                    
                    __m128i bxl = (LAYOUT == RGGB || LAYOUT == GBRG ? _mm_unpacklo_epi8(vb, _mm_set1_epi16(0)) : _mm_unpacklo_epi8(vr, _mm_set1_epi16(0)));
                    __m128i bxh = (LAYOUT == RGGB || LAYOUT == GBRG ? _mm_unpackhi_epi8(vb, _mm_set1_epi16(0)) : _mm_unpackhi_epi8(vr, _mm_set1_epi16(0)));
                    
                    __m128i tdestxll = _mm_shuffle_epi8(_mm_unpacklo_epi16(rgl, bxl), shuffle);
                    __m128i tdestxlh = _mm_shuffle_epi8(_mm_unpackhi_epi16(rgl, bxl), shuffle);
                    __m128i tdestxhl = _mm_shuffle_epi8(_mm_unpacklo_epi16(rgh, bxh), shuffle);
                    __m128i tdestxhh = _mm_shuffle_epi8(_mm_unpackhi_epi16(rgh, bxh), shuffle);
                    
                    _mm_storeu_si128((__m128i*)(tdest+destshift), tdestxll); // pix 0-3
                    _mm_storeu_si128((__m128i*)(tdest+destshift+destoffset), tdestxlh); // pix 4-7
                    _mm_storeu_si128((__m128i*)(tdest+destshift+destoffset*2), tdestxhl); // pix 8-11
                    _mm_storeu_si128((__m128i*)(tdest+destshift+destoffset*3), tdestxhh); // pix 12-13
                    
                    
                }
                
                if( LAYOUT == GBRG || LAYOUT == GRBG ) {
                    __m128i v1 = _mm_loadu_si128((const __m128i*)(tsrc));
                    __m128i v2 = _mm_loadu_si128((const __m128i*)(tsrc + pitchs));
                    __m128i v3 = _mm_loadu_si128((const __m128i*)(tsrc + 2*pitchs));
                    
                    // Row - 1
                    __m128i v1r = _mm_and_si128(v1, _mm_set1_epi16(0x00FF)); // LSB
                    __m128i v1g = _mm_and_si128(v1, _mm_set1_epi16(0xFF00)); // MSB
                    // Current row
                    __m128i v2g = _mm_and_si128(v2, _mm_set1_epi16(0x00FF));
                    __m128i v2b = _mm_and_si128(v2, _mm_set1_epi16(0xFF00));
                    // Row + 1
                    __m128i v3r = _mm_and_si128(v3, _mm_set1_epi16(0x00FF));
                    __m128i v3g = _mm_and_si128(v3, _mm_set1_epi16(0xFF00));
                    
                    
                    // Red even
                    __m128i vr = _mm_add_epi16(v1r, v3r);
                    // Red odd
                    __m128i vr_shift2r = _mm_srli_si128(vr, 2);
                    __m128i add_t = _mm_add_epi16(vr,vr_shift2r);
                    __m128i vrt = _mm_slli_epi16(add_t, 6);
                    // Red combined
                    vr = _mm_srli_si128(_mm_add_epi8(_mm_srli_epi16(vr,1),_mm_and_si128(_mm_set1_epi16(0xFF00), vrt)),1);
                    
                    
                    // Green even
                    __m128i vg = v2g;
                    // Green odd
                    __m128i green_shift2r = _mm_srli_si128(v2g, 2);
                    add_t = _mm_add_epi16(v2g,green_shift2r);
                    add_t = _mm_add_epi16(_mm_add_epi16(_mm_srli_si128(v1g,1),_mm_srli_si128(v3g,1)), add_t);
                    add_t = _mm_add_epi16(_mm_set1_epi16(2), add_t);
                    vrt = _mm_slli_epi16(add_t, 6);
                    // Green combined
                    vg = _mm_srli_si128(_mm_add_epi8(vg, _mm_and_si128(_mm_set1_epi16(0xFF00), vrt)),1);
                    
                    // Blue even
                    __m128i blue_shift3r = _mm_srli_si128(v2b, 3);
                    add_t = _mm_add_epi16(_mm_set1_epi16(1),_mm_add_epi16(_mm_srli_si128(v2b, 1),blue_shift3r));
                    vrt = _mm_srli_epi16(add_t, 1);
                    
                    __m128i vb = _mm_srli_si128(_mm_add_epi8(v2b, _mm_slli_si128(vrt,2)),1);
                    
                    
                    __m128i rgl = (LAYOUT == GBRG ? _mm_unpacklo_epi8(vr,vg) : _mm_unpacklo_epi8(vb,vg));
                    __m128i rgh = (LAYOUT == GBRG ? _mm_unpackhi_epi8(vr,vg) : _mm_unpackhi_epi8(vb,vg));
                    
                    __m128i bxl = (LAYOUT == GBRG ? _mm_unpacklo_epi8(vb, _mm_set1_epi16(0)) : _mm_unpacklo_epi8(vr, _mm_set1_epi16(0)));
                    __m128i bxh = (LAYOUT == GBRG ? _mm_unpackhi_epi8(vb, _mm_set1_epi16(0)) : _mm_unpackhi_epi8(vr, _mm_set1_epi16(0)));
                    
                    
                    __m128i tdestxll = _mm_shuffle_epi8(_mm_unpacklo_epi16(rgl, bxl), shuffle);
                    __m128i tdestxlh = _mm_shuffle_epi8(_mm_unpackhi_epi16(rgl, bxl), shuffle);
                    __m128i tdestxhl = _mm_shuffle_epi8(_mm_unpacklo_epi16(rgh, bxh), shuffle);
                    __m128i tdestxhh = _mm_shuffle_epi8(_mm_unpackhi_epi16(rgh, bxh), shuffle);
                    
                    _mm_storeu_si128((__m128i*)(tdest+pitchd), tdestxll);
                    _mm_storeu_si128((__m128i*)(tdest+pitchd+destoffset), tdestxlh);
                    _mm_storeu_si128((__m128i*)(tdest+pitchd+destoffset*2), tdestxhl);
                    _mm_storeu_si128((__m128i*)(tdest+pitchd+destoffset*3), tdestxhh);
                    
                }
                tsrc+=14;
                tdest += (HAS_ALPHA ? 4*14 : 3*14);
                
            } // loop x
            
            if( x<pitchs ) {
                
                tsrc -= pitchs;
                
                bool start_with_green = (LAYOUT == GRBG || LAYOUT == GBRG);
                int blue = (LAYOUT == BGGR || LAYOUT == GBRG ? -1 : 1 );
                tdest += 1;
                int width_offset = width-x-2;
                
                for( int p=0; p<2; ++p, tsrc += pitchs, tdest += pitchd ) {
                    
                    int t0, t1;
                    const unsigned char *bayerEnd = tsrc + width_offset;
                    
                    if( start_with_green ) {
                        t0 = (tsrc[1] + tsrc[pitchs * 2 + 1] + 1) >> 1;
                        t1 = (tsrc[pitchs] + tsrc[pitchs + 2] + 1) >> 1;
                        tdest[-blue] = (unsigned char)t0;
                        tdest[0] = tsrc[pitchs + 1];
                        tdest[blue] = (unsigned char)t1;
                        tsrc++;
                        tdest += 3;
                    }
                    
                    if (blue > 0) {
                        for (; tsrc <= bayerEnd - 2; tsrc += 2, tdest += 6) {
                            t0 = (tsrc[0] + tsrc[2] + tsrc[pitchs * 2] +
                                  tsrc[pitchs * 2 + 2] + 2) >> 2;
                            t1 = (tsrc[1] + tsrc[pitchs] +
                                  tsrc[pitchs + 2] + tsrc[pitchs * 2 + 1] +
                                  2) >> 2;
                            tdest[-1] = (uint8_t) t0;
                            tdest[0] = (uint8_t) t1;
                            tdest[1] = tsrc[pitchs + 1];
                            
                            t0 = (tsrc[2] + tsrc[pitchs * 2 + 2] + 1) >> 1;
                            t1 = (tsrc[pitchs + 1] + tsrc[pitchs + 3] +
                                  1) >> 1;
                            tdest[2] = (uint8_t) t0;
                            tdest[3] = tsrc[pitchs + 2];
                            tdest[4] = (uint8_t) t1;
                        }
                    } else {
                        for (; tsrc <= bayerEnd - 2; tsrc += 2, tdest += 6) {
                            t0 = (tsrc[0] + tsrc[2] + tsrc[pitchs * 2] +
                                  tsrc[pitchs * 2 + 2] + 2) >> 2;
                            t1 = (tsrc[1] + tsrc[pitchs] +
                                  tsrc[pitchs + 2] + tsrc[pitchs * 2 + 1] +
                                  2) >> 2;
                            tdest[1] = (uint8_t) t0;
                            tdest[0] = (uint8_t) t1;
                            tdest[-1] = tsrc[pitchs + 1];
                            
                            t0 = (tsrc[2] + tsrc[pitchs * 2 + 2] + 1) >> 1;
                            t1 = (tsrc[pitchs + 1] + tsrc[pitchs + 3] +
                                  1) >> 1;
                            tdest[4] = (uint8_t) t0;
                            tdest[3] = tsrc[pitchs + 2];
                            tdest[2] = (uint8_t) t1;
                        }
                    }
                    
                    if (tsrc < bayerEnd) {
                        t0 = (tsrc[0] + tsrc[2] + tsrc[pitchs * 2] +
                              tsrc[pitchs * 2 + 2] + 2) >> 2;
                        t1 = (tsrc[1] + tsrc[pitchs] +
                              tsrc[pitchs + 2] + tsrc[pitchs * 2 + 1] +
                              2) >> 2;
                        tdest[-blue] = (uint8_t) t0;
                        tdest[0] = (uint8_t) t1;
                        tdest[blue] = tsrc[pitchs + 1];
                        tsrc++;
                        tdest += 3;
                    }
                    
                    tsrc -= width_offset;
                    tdest -= width_offset * 3;
                    
                    blue = -blue;
                    start_with_green = !start_with_green;
                }
            }
            
        }
    }
    
} // namespace pvcore