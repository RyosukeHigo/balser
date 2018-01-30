
// Generates a probability image given a histogram
class GENERATE_PROBIM_TBB {
    
    const unsigned char* src;               // Image
    const float* hist;                      // Histogram
    float* dest;                            // Resulting image
    unsigned int width;                     // Image width
    unsigned int height;                    // Image height
    unsigned int pitch;                     // Image pitch
    unsigned int srcchannels;               // Image number of channels
    unsigned int histx;                     // Histogram x-dim
    unsigned int histy;                     // Histogram y-dim
    unsigned int histz;                     // Histogram z-dim
    unsigned int histchannels;              // Desired number of histogram channels
    const short* maxr;                      // Max width per row in image
    unsigned int threads;                   // Desired number of threads
    
public:
    void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        
        // Sort out indices - threads process whole rows
        unsigned int height_p_thread = (height/threads);
        
        unsigned int start = (unsigned int)r.begin()*height_p_thread;
        unsigned int stop  = (unsigned int)( (threads == r.end()) ? (size_t)height : r.end()*height_p_thread );
        
        _generate_probim( start, stop );
    }
    
    
    GENERATE_PROBIM_TBB(const unsigned char* _src,
                              const float* _hist,
                              float* _dest,
                              unsigned int _width,
                              unsigned int _height,
                              unsigned int _pitch,
                              unsigned int _srcchannels,
                              unsigned int _histx,
                              unsigned int _histy,
                              unsigned int _histz,
                              unsigned int _histchannels,
                              const short* _maxr,
                              unsigned int _threads ) :
    src(_src), hist(_hist), dest(_dest),
    width(_width), height(_height), pitch(_pitch), srcchannels(_srcchannels),
    histx(_histx), histy(_histy), histz(_histz),
    histchannels(_histchannels),maxr(_maxr), threads(_threads) {
        
    }
	
	
	
private:
    inline void _generate_probim(unsigned int _start,
                                 unsigned int _stop ) const {
        
        float idimx = (float)histx/256.0f;
        float idimy = (float)histy/256.0f;
        
        unsigned int dpitch = pitch/srcchannels;

        switch( histchannels ) {
            case 1:
                for( unsigned int i=_start; i<_stop; ++i ) {
                    unsigned int idx = pitch*i;
                    int j=0;
                    int w = (maxr == NULL ? width : maxr[i]);
                    for( ; j<w; j+=1 ) {
                        unsigned char c1 = src[idx]; idx+=srcchannels;
                        unsigned int idx1 = (unsigned int)((float)(c1)*idimx);
                        
                        dest[i*dpitch+j] = hist[idx1];
                    }
                    //memset( &(dest[i*dpitch+j]), 0, (width-j)*sizeof(float) );
                    
                }
                break;

            case 2:
                for( unsigned int i=_start; i<_stop; ++i ) {
                    unsigned int idx = pitch*i;
                    int j=0;
                    int w = (maxr == NULL ? width : maxr[i]);
                    for( ; j<w; j+=1 ) {
                        unsigned char c1 = src[idx]; idx+=1;
                        unsigned char c2 = src[idx]; idx+=2;
                        unsigned int idx1 = (unsigned int)((float)(c1)*idimx);
                        unsigned int idx2 = (unsigned int)((float)(c2)*idimy);
                        
                        float p = hist[idx2*histx+idx1];
                        
                        dest[i*dpitch+j] = p;
                    }
                    //memset( &(dest[i*dpitch+j]), 0, (width-j)*sizeof(float) );

                }
                break;
                
            case 3:
            {
#ifdef USE_SSE
                unsigned char __buff[8*48+15];
                size_t align = ((size_t)__buff) % 16;
                unsigned long long *buff = (unsigned long long*)(__buff+align);
                
                buff[0]  = 0x8009800680038000ull; buff[1]  = 0x80808080800f800cull;    // Puts red colors from v0 in low
                buff[4]  = 0x800a800780048001ull; buff[5]  = 0x808080808080800dull;    // Puts green colors from v0 in low
                buff[8]  = 0x800b800880058002ull; buff[9]  = 0x808080808080800eull;    // Puts blue colors from v0 in low
                
                buff[2]  = 0x8080808080808080ull; buff[3]  = 0x8005800280808080ull;    // Puts red colors from v1 in low
                buff[6]  = 0x8080808080808080ull; buff[7]  = 0x8006800380008080ull;    // Puts green colors from v1 in low
                buff[10] = 0x8080808080808080ull; buff[11] = 0x8007800480018080ull;    // Puts blue colors from v1 in low
                
                buff[12] = 0x8080800e800b8008ull; buff[13] = 0x8080808080808080ull;    // Puts red colors from v1 in high
                buff[16] = 0x8080800f800c8009ull; buff[17] = 0x8080808080808080ull;    // Puts green colors from v1 in high
                buff[20] = 0x80808080800d800aull; buff[21] = 0x8080808080808080ull;    // Puts blue colors from v1 in high
                
                buff[14] = 0x8001808080808080ull; buff[15] = 0x800d800a80078004ull;    // Puts red colors from v2 in high
                buff[18] = 0x8002808080808080ull; buff[19] = 0x800e800b80088005ull;    // Puts green colors from v2 in high
                buff[22] = 0x8003800080808080ull; buff[23] = 0x800f800c80098006ull;    // Puts blue colors from v2 in high
                
                // Read masks
                const __m128i mLowRv0 = _mm_load_si128((const __m128i*)buff+0);
                const __m128i mLowRv1 = _mm_load_si128((const __m128i*)buff+1);
                const __m128i mLowGv0 = _mm_load_si128((const __m128i*)buff+2);
                const __m128i mLowGv1 = _mm_load_si128((const __m128i*)buff+3);
                const __m128i mLowBv0 = _mm_load_si128((const __m128i*)buff+4);
                const __m128i mLowBv1 = _mm_load_si128((const __m128i*)buff+5);
                
                const __m128i mHighRv0 = _mm_load_si128( (const __m128i*)buff+6 );
                const __m128i mHighRv1 = _mm_load_si128( (const __m128i*)buff+7 );
                const __m128i mHighGv0 = _mm_load_si128( (const __m128i*)buff+8 );
                const __m128i mHighGv1 = _mm_load_si128( (const __m128i*)buff+9 );
                const __m128i mHighBv0 = _mm_load_si128( (const __m128i*)buff+10);
                const __m128i mHighBv1 = _mm_load_si128( (const __m128i*)buff+11);

                
                const __m128i mHistx = _mm_setr_epi16((short)histx,(short)histx,(short)histx,(short)histx,
                                                      (short)histx,(short)histx,(short)histx,(short)histx);
                const __m128i mHisty = _mm_setr_epi16((short)histy,(short)histy,(short)histy,(short)histy,
                                                      (short)histy,(short)histy,(short)histy,(short)histy);
                const __m128i mHistz = _mm_setr_epi16((short)histz,(short)histz,(short)histz,(short)histz,
                                                      (short)histz,(short)histz,(short)histz,(short)histz);
                
                const unsigned char* tsrc = src;
				short idxslo[8];
				short idxshi[8];
                short idxs1[8];
                short idxs2[8];
                short idxs3[8];
                for( unsigned int i=_start; i<_stop; ++i ) {
                    tsrc = src + pitch*i;
                    unsigned int didx = dpitch*i;
                    int j=0;
                    int w = (maxr == NULL ? width : maxr[i]+15);
                    for( ; j<w; j+=16 ) {
                        // Read image
                        const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
                        const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
                        const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;

                        // Low vector
                        const __m128i reds_low   = _mm_or_si128( _mm_shuffle_epi8(v0,mLowRv0) , _mm_shuffle_epi8(v1,mLowRv1) );
                        const __m128i greens_low = _mm_or_si128( _mm_shuffle_epi8(v0,mLowGv0) , _mm_shuffle_epi8(v1,mLowGv1) );
                        const __m128i blues_low  = _mm_or_si128( _mm_shuffle_epi8(v0,mLowBv0) , _mm_shuffle_epi8(v1,mLowBv1) );
                        
                        __m128i midx1 = _mm_srli_epi16(_mm_mullo_epi16(mHistx,reds_low), 8);
                        __m128i midx2 = _mm_srli_epi16(_mm_mullo_epi16(mHisty,greens_low), 8);
                        __m128i midx3 = _mm_srli_epi16(_mm_mullo_epi16(mHistz,blues_low), 8);
						
						__m128i midxlo = _mm_add_epi16(_mm_mullo_epi16(midx3, _mm_set1_epi16(histy*histx)),
													 _mm_add_epi16(_mm_mullo_epi16(midx2, _mm_set1_epi16(histx)), midx1));
						
//						__m128 tostore = _mm_set_ps(hist[_mm_extract_epi16(midxlo, 3)],
//													hist[_mm_extract_epi16(midxlo, 2)],
//													hist[_mm_extract_epi16(midxlo, 1)],
//													hist[_mm_extract_epi16(midxlo, 0)]);
//						
//						_mm_storeu_si128((__m128i*)&dest[didx+j+0], tostore);
//						
//						tostore = _mm_set_ps(hist[_mm_extract_epi16(midxlo, 7)],
//											 hist[_mm_extract_epi16(midxlo, 6)],
//											 hist[_mm_extract_epi16(midxlo, 5)],
//											 hist[_mm_extract_epi16(midxlo, 4)]);
//						
//						_mm_storeu_si128((__m128i*)&dest[didx+j+4], tostore);
						

                        _mm_storeu_si128((__m128i*)idxs1, midx1);
                        _mm_storeu_si128((__m128i*)idxs2, midx2);
                        _mm_storeu_si128((__m128i*)idxs3, midx3);
						
						_mm_storeu_si128((__m128i*)idxslo, midxlo);
						
						// HERE WE CAN USE GATHER INSTRUCTIONS
						
                        for( unsigned int ii=0; ii<8; ++ii ) {
							dest[didx+j+ii] = hist[idxslo[ii]];
//                            dest[didx+j+ii] = hist[(idxs3[ii]*histy+idxs2[ii])*histx+idxs1[ii]];
                        }

                    
                        // High vector
                        const __m128i reds_high   = _mm_or_si128( _mm_shuffle_epi8(v1,mHighRv0) , _mm_shuffle_epi8(v2,mHighRv1) );
                        const __m128i greens_high = _mm_or_si128( _mm_shuffle_epi8(v1,mHighGv0) , _mm_shuffle_epi8(v2,mHighGv1) );
                        const __m128i blues_high  = _mm_or_si128( _mm_shuffle_epi8(v1,mHighBv0) , _mm_shuffle_epi8(v2,mHighBv1) );

                        midx1 = _mm_srli_epi16(_mm_mullo_epi16(mHistx,reds_high), 8);
                        midx2 = _mm_srli_epi16(_mm_mullo_epi16(mHisty,greens_high), 8);
                        midx3 = _mm_srli_epi16(_mm_mullo_epi16(mHistz,blues_high), 8);
						
						__m128i midxhi = _mm_add_epi16(_mm_mullo_epi16(midx3, _mm_set1_epi16(histy*histx)),
											 _mm_add_epi16(_mm_mullo_epi16(midx2, _mm_set1_epi16(histx)), midx1));
						
//                        _mm_storeu_si128((__m128i*)idxs1, midx1);
//                        _mm_storeu_si128((__m128i*)idxs2, midx2);
//                        _mm_storeu_si128((__m128i*)idxs3, midx3);
						
						_mm_storeu_si128((__m128i*)idxshi, midxhi);
						

						for( unsigned int ii=0; ii<8; ++ii ) {
							dest[didx+j+8+ii] = hist[idxshi[ii]];
//                            dest[didx+j+8+ii] = hist[(idxs3[ii]*histy+idxs2[ii])*histx+idxs1[ii]];
                        }
						
						
//						 tostore = _mm_set_ps(hist[_mm_extract_epi16(midxhi, 3)],
//													hist[_mm_extract_epi16(midxhi, 2)],
//													hist[_mm_extract_epi16(midxhi, 1)],
//													hist[_mm_extract_epi16(midxhi, 0)]);
//						
//						_mm_storeu_si128((__m128i*)&dest[didx+j+8], tostore);
//						
//						 tostore = _mm_set_ps(hist[_mm_extract_epi16(midxhi, 7)],
//													hist[_mm_extract_epi16(midxhi, 6)],
//													hist[_mm_extract_epi16(midxhi, 5)],
//													hist[_mm_extract_epi16(midxhi, 4)]);
//						
//						_mm_storeu_si128((__m128i*)&dest[didx+j+12], tostore);
						

                    
                    }
                    //memset( &(dest[i*dpitch+j]), 0, (width-j)*sizeof(float) );
                    
                }
            }
#else
                // WORKING
                
                for( unsigned int i=_start; i<_stop; ++i ) {
                    unsigned int idx = pitch*i;
                    unsigned int didx = dpitch*i;
                    int j=0;
                    for( ; j<maxr[i]; ++j ) {
                        // using short
                        unsigned int idx1 = (((short)src[idx])*histx)>>8;idx++;
                        unsigned int idx2 = (((short)src[idx])*histy)>>8;idx++;
                        unsigned int idx3 = (((short)src[idx])*histz)>>8;idx++;
                        
                        // Using float
                        //unsigned int idx1 = (unsigned int)((float)(src[idx])*idimx);idx++;
                        //unsigned int idx2 = (unsigned int)((float)(src[idx])*idimy);idx++;
                        //unsigned int idx3 = (unsigned int)((float)(src[idx])*idimz);idx++;
                        
                        float p = hist[idx3*histy*histx+idx2*histx+idx1];
                        
                        dest[didx+j] = p;
                    }
                    
                }
                break;
#endif
       }
        
    }
    
};
