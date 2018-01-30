
void bgr2lab(const unsigned char* _src,
             unsigned char* _dest,
             unsigned int _width,
             unsigned int _height,
             unsigned int _pitch,
             unsigned int _threads) {
    
    unsigned int globalX = get_global_id(0);
    unsigned int globalY = get_global_id(1);
    
    unsigned int global_memid = 3*mad24(globalY,width,globalX);
    
    float r = (float)src[global_memid+2];
    float g = (float)src[global_memid+1];
    float b = (float)src[global_memid];
    float x,y,z;
    
    float t = 0.008856;
    
    x = 0.433953f*r + 0.376219f*g + 0.189828f*b;
    y = 0.212671f*r + 0.715160f*g + 0.072169f*b;
    z = 0.017758f*r + 0.109477f*g + 0.872766f*b;
    
    x /= 0.950456f;
    y /= 1.0f;
    z /= 1.088754f;
    
    float y3 = rootn(y,3);
    
    bool xt = x>t;
    bool yt = y>t;
    bool zt = z>t;
    
    
    if(xt) {
        x = rootn(x,3);
    } else {
        x = 7.787f*x + 0.13793103448275862f;
    }
    
    if(yt) {
        y = y3;
    } else {
        y = 7.787f*y + 0.13793103448275862f;
    }
    
    if(zt) {
        z = rootn(z,3);
    } else {
        z = 7.787f*z + 0.13793103448275862f;
    }
    
    if(yt) {
        dest[global_memid] = (116.0f*y3 - 16.0f) / 100.0f;
    } else {
        dest[global_memid] = (903.3*y) / 100.0f;
    }
    dest[global_memid+1] = 500.0f * (x-y) / 256.0f + 0.5f;
    dest[global_memid+2] = 200.0f * (y-z) / 256.0f + 0.5f;
    
}





class RGB2LAB_8U_TBB {
    
    const unsigned char *src;
    unsigned char *dest;
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    unsigned int threads;
    
private:
    void _rgb2lab_8u_sse(unsigned int _start,
                         unsigned int _stop ) const;
    
public:
    void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        
        // Let's make the last thread do the least work
        unsigned int blockSize = (height+threads-1)/threads;
        
        size_t start = r.begin()*blockSize;
        size_t stop  = r.end()*blockSize;
        if( r.end() == threads ) {
            stop = height;
        }
        
        _rgb2luv_8u_sse( start, stop );
    }
    
    RGB2LAB_8U_TBB( const unsigned char* _src, unsigned char* _dest, unsigned int _width, unsigned int _height, unsigned int _pitch, unsigned int _threads ) :
    src(_src), dest(_dest), width(_width), height(_height), pitch(_pitch), threads(_threads) {
        
    }
    
};



class RGB2LUV_32F_TBB {
    
    const float *src;
    float *dest;
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    unsigned int threads;
    
private:
    void _rgb2luv_32f_sse(unsigned int _start,
                          unsigned int _stop ) const;
    
public:
    void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        
        // Let's make the last thread do the least work
        unsigned int blockSize = (height+threads-1)/threads;
        
        size_t start = r.begin()*blockSize;
        size_t stop  = r.end()*blockSize;
        if( r.end() == threads ) {
            stop = height;
        }
        
        _rgb2luv_32f_sse( start, stop );
    }
    
    RGB2LUV_32F_TBB( const float* _src, float* _dest, unsigned int _width, unsigned int _height, unsigned int _pitch, unsigned int _threads ) :
    src(_src), dest(_dest), width(_width), height(_height), pitch(_pitch), threads(_threads) {
        
    }
    
};

// takes vectors of size 2 (__m128[2])
inline void rgb2lab_sse(const __m128* r, const __m128* g, const __m128* b,
                        __m128* L, __m128* u, __m128* v ) {
    
    __m128 x[2];
    __m128 y[2];
    __m128 z[2];
    
    x[0] = _mm_add_ps(_mm_mul_ps(r[0],_mm_set_ps1(0.412453f)),
                    _mm_add_ps(_mm_mul_ps(g[0],_mm_set_ps1(0.357580f)),
                               _mm_mul_ps(b[0],_mm_set_ps1(0.180423f))));
    y[0] = _mm_add_ps(_mm_mul_ps(r[0],_mm_set_ps1(0.212671f)),
                      _mm_add_ps(_mm_mul_ps(g[0],_mm_set_ps1(0.715160f)),
                                 _mm_mul_ps(b[0],_mm_set_ps1(0.072169f))));
    z[0] = _mm_add_ps(_mm_mul_ps(r[0],_mm_set_ps1(0.019334f)),
                    _mm_add_ps(_mm_mul_ps(g[0],_mm_set_ps1(0.119193f)),
                               _mm_mul_ps(b[0],_mm_set_ps1(0.950227f))));
    
    x[1] = _mm_add_ps(_mm_mul_ps(r[1],_mm_set_ps1(0.412453f)),
                           _mm_add_ps(_mm_mul_ps(g[1],_mm_set_ps1(0.357580f)),
                                      _mm_mul_ps(b[1],_mm_set_ps1(0.180423f))));
    y[1] = _mm_add_ps(_mm_mul_ps(r[1],_mm_set_ps1(0.212671f)),
                      _mm_add_ps(_mm_mul_ps(g[1],_mm_set_ps1(0.715160f)),
                                 _mm_mul_ps(b[1],_mm_set_ps1(0.072169f))));
    z[1] = _mm_add_ps(_mm_mul_ps(r[1],_mm_set_ps1(0.019334f)),
                           _mm_add_ps(_mm_mul_ps(g[1],_mm_set_ps1(0.119193f)),
                                      _mm_mul_ps(b[1],_mm_set_ps1(0.950227f))));
    
    x[0] = _mm_div_ps( x[0], _mm_set_ps1(0.950456f));
    z[0] = _mm_div_ps( z[0], _mm_set_ps1(1.088754f));
    x[1] = _mm_div_ps( x[1], _mm_set_ps1(0.950456f));
    z[1] = _mm_div_ps( z[1], _mm_set_ps1(1.088754f));
    
    
    __m128 xcube[2];
    __m128 ycube[2];
    __m128 zcube[2];
    cubeRoot_p(x, xcube);
    cubeRoot_p(y, ycube);
    cubeRoot_p(z, zcube);
    
    
    __m128 Lbig0 = _mm_sub_ps(_mm_mul_ps(_mm_set_ps1(116.0f),cube[0]),_mm_set_ps1(16.0f));
    __m128 Lsmall0 = _mm_mul_ps(y[0], _mm_set_ps1(903.3f));
    
    __m128 Lbig1 = _mm_sub_ps(_mm_mul_ps(_mm_set_ps1(116.0f),cube[1]),_mm_set_ps1(16.0f));
    __m128 Lsmall1 = _mm_mul_ps(y[1], _mm_set_ps1(903.3f));
    
    
    // Select
    __m128 ygt0 = _mm_cmpgt_ps(y[0], _mm_set_ps1(0.008856f));
    __m128 ygt1 = _mm_cmpgt_ps(y[1], _mm_set_ps1(0.008856f));
    
    L[0] = _mm_or_ps(_mm_and_ps(ygt0, Lbig0), _mm_andnot_ps(ygt0, Lsmall0));
    L[1] = _mm_or_ps(_mm_and_ps(ygt1, Lbig1), _mm_andnot_ps(ygt1, Lsmall1));
    
    //
    __m128 t0 = _mm_add_ps(x0, _mm_add_ps(_mm_mul_ps(_mm_set_ps1(15.0f), y[0]),
                                          _mm_mul_ps(_mm_set_ps1(3.0f), z0)));
    __m128 ti0 = _mm_div_ps(_mm_set_ps1(1.0f),t0);
    
    u[0] = _mm_mul_ps(_mm_set_ps1(4.0f), _mm_mul_ps(x0, ti0));
    v[0] = _mm_mul_ps(_mm_set_ps1(9.0f), _mm_mul_ps(y[0], ti0));
    
    u[0] = _mm_mul_ps(_mm_set_ps1(13.0f), _mm_mul_ps(L[0], _mm_sub_ps(u[0], _mm_set_ps1(0.19793943f))));
    v[0] = _mm_mul_ps(_mm_set_ps1(13.0f), _mm_mul_ps(L[0], _mm_sub_ps(v[0], _mm_set_ps1(0.46831096f))));
    
    __m128 t1 = _mm_add_ps(x1, _mm_add_ps(_mm_mul_ps(_mm_set_ps1(15.0f), y[1]),
                                          _mm_mul_ps(_mm_set_ps1(3.0f), z1)));
    __m128 ti1 = _mm_div_ps(_mm_set_ps1(1.0f),t1);
    
    u[1] = _mm_mul_ps(_mm_set_ps1(4.0f), _mm_mul_ps(x1, ti1));
    v[1] = _mm_mul_ps(_mm_set_ps1(9.0f), _mm_mul_ps(y[1], ti1));
    
    u[1] = _mm_mul_ps(_mm_set_ps1(13.0f), _mm_mul_ps(L[1], _mm_sub_ps(u[1], _mm_set_ps1(0.19793943f))));
    v[1] = _mm_mul_ps(_mm_set_ps1(13.0f), _mm_mul_ps(L[1], _mm_sub_ps(v[1], _mm_set_ps1(0.46831096f))));
    
    
}


void RGB2LUV_32F_TBB::_rgb2luv_32f_sse(unsigned int _start,
                                       unsigned int _stop ) const {
    
    static bool first = true;
    // Change this to struct for better readability?
    static unsigned char __buff[8*72+31];
    static size_t align = ((size_t)__buff) % 32;
    static unsigned long long *buff = (unsigned long long*)(__buff+align);
    
    if( first ) {
        // Shuffle bytes for processing
        buff[0]  = 0x8080800380808000ull; buff[1]  = 0x8080800980808006ull;    // Puts red colors from v0 in r1
        buff[2]  = 0x8080800480808001ull; buff[3]  = 0x8080800a80808007ull;    // Puts green colors from v0 in g1
        buff[4]  = 0x8080800580808002ull; buff[5]  = 0x8080800b80808008ull;    // Puts blue colors from v0 in b1
        
        buff[6]  = 0x8080800f8080800cull; buff[7]  = 0x8080808080808080ull;    // Puts red colors from v0 in r2
        first = false;
    }
    
    const unsigned int widthz = (pitch/3) >> 3;
    
    float Ls[8], us[8], vs[8];
    
    for( int h=_start; h<_stop; ++h ) {
        const float *tsrc = src + pitch*h;
        float *tdest = dest + pitch*h;
        for( int w=0; w<widthz; ++w ) {
            
            __m128 reds[2],greens[2],blues[2];
            __m128 L[2],u[2],v[2];
            
            // Load float values
            reds[0]   = _mm_set_ps(tsrc[0],tsrc[3],tsrc[6],tsrc[9]);
            greens[0] = _mm_set_ps(tsrc[1],tsrc[4],tsrc[7],tsrc[10]);
            blues[0]  = _mm_set_ps(tsrc[2],tsrc[5],tsrc[8],tsrc[11]); tsrc += 12;
            
            reds[1]   = _mm_set_ps(tsrc[0],tsrc[3],tsrc[6],tsrc[9]);
            greens[1] = _mm_set_ps(tsrc[1],tsrc[4],tsrc[7],tsrc[10]);
            blues[1]  = _mm_set_ps(tsrc[2],tsrc[5],tsrc[8],tsrc[11]); tsrc += 12;
            
            rgb2luv_sse(reds, greens, blues, L, u, v);
            
            L[0] = _mm_add_ps(_mm_mul_ps(L[0], _mm_set_ps1(k_post_coeffs_32f[0])), _mm_set_ps1(k_post_coeffs_32f[1]));
            L[1] = _mm_add_ps(_mm_mul_ps(L[1], _mm_set_ps1(k_post_coeffs_32f[0])), _mm_set_ps1(k_post_coeffs_32f[1]));
            u[0] = _mm_add_ps(_mm_mul_ps(u[0], _mm_set_ps1(k_post_coeffs_32f[2])), _mm_set_ps1(k_post_coeffs_32f[3]));
            u[1] = _mm_add_ps(_mm_mul_ps(u[1], _mm_set_ps1(k_post_coeffs_32f[2])), _mm_set_ps1(k_post_coeffs_32f[3]));
            v[0] = _mm_add_ps(_mm_mul_ps(v[0], _mm_set_ps1(k_post_coeffs_32f[4])), _mm_set_ps1(k_post_coeffs_32f[5]));
            v[1] = _mm_add_ps(_mm_mul_ps(v[1], _mm_set_ps1(k_post_coeffs_32f[4])), _mm_set_ps1(k_post_coeffs_32f[5]));
            
            
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
            
            //memcpy( tdest, tsrc-24, 24*sizeof(float)); tdest += 24;
        }
    }
    
}

void RGB2LUV_8U_TBB::_rgb2luv_8u_sse(unsigned int _start,
                                     unsigned int _stop ) const {
    
    static bool first = true;
    // Change this to struct for better readability?
    static unsigned char __buff[8*72+31];
    static size_t align = ((size_t)__buff) % 32;
    static unsigned long long *buff = (unsigned long long*)(__buff+align);
    
    if( first ) {
        // Shuffle bytes for processing
        buff[0]  = 0x8080800380808000ull; buff[1]  = 0x8080800980808006ull;    // Puts red colors from v0 in r1
        buff[2]  = 0x8080800480808001ull; buff[3]  = 0x8080800a80808007ull;    // Puts green colors from v0 in g1
        buff[4]  = 0x8080800580808002ull; buff[5]  = 0x8080800b80808008ull;    // Puts blue colors from v0 in b1
        
        buff[6]  = 0x8080800f8080800cull; buff[7]  = 0x8080808080808080ull;    // Puts red colors from v0 in r2
        buff[8] =  0x8080808080808080ull; buff[9] =  0x8080800580808002ull;    // Puts red colors from v1 in r2
        buff[10] = 0x808080808080800dull; buff[11] = 0x8080808080808080ull;    // Puts green colors from v0 in g2
        buff[12] = 0x8080800080808080ull; buff[13] = 0x8080800680808003ull;    // Puts green colors from v1 in g2
        buff[14] = 0x808080808080800eull; buff[15] = 0x8080808080808080ull;    // Puts blue colors from v0 in b2
        buff[16] = 0x8080800180808080ull; buff[17] = 0x8080800780808004ull;    // Puts blue colors from v1 in b2
        
        buff[18] = 0x8080800b80808008ull; buff[19] = 0x808080808080800eull;    // Puts red colors from v1 in r3
        buff[20] = 0x8080808080808080ull; buff[21] = 0x8080800180808080ull;    // Puts red colors from v2 in r3
        buff[22] = 0x8080800c80808009ull; buff[23] = 0x808080808080800full;    // Puts green colors from v1 in g3
        buff[24] = 0x8080808080808080ull; buff[25] = 0x8080800280808080ull;    // Puts green colors from v2 in g3
        buff[26] = 0x8080800d8080800aull; buff[27] = 0x8080808080808080ull;    // Puts blue colors from v1 in b3
        buff[28] = 0x8080808080808080ull; buff[29] = 0x8080800380808000ull;    // Puts blue colors from v2 in b3
        
        buff[30] = 0x8080800780808004ull; buff[31] = 0x8080800d8080800aull;    // Puts red colors from v2 in r4
        buff[32] = 0x8080800880808005ull; buff[33] = 0x8080800e8080800bull;    // Puts green colors from v2 in g4
        buff[34] = 0x8080800980808006ull; buff[35] = 0x8080800f8080800cull;    // Puts blue colors from v2 in b4
        
        // Shuffle bytes for storing
        buff[36] = 0x8008808004808000ull; buff[37] = 0x8080808080800c80ull;    // Puts L0 in dest0
        buff[38] = 0x0880800480800080ull; buff[39] = 0x80808080800c8080ull;    // Puts u0 in dest0
        buff[40] = 0x8080048080008080ull; buff[41] = 0x808080800c808008ull;    // Puts v0 in dest0
        
        buff[42] = 0x8080808080808080ull; buff[43] = 0x0480800080808080ull;    // Puts L1 in dest0
        buff[44] = 0x8080808080808080ull; buff[45] = 0x8080008080808080ull;    // Puts u1 in dest0
        buff[46] = 0x8080808080808080ull; buff[47] = 0x8000808080808080ull;    // Puts v1 in dest0
        
        buff[48] = 0x80800c8080088080ull; buff[49] = 0x8080808080808080ull;    // Puts L1 in dest1
        buff[50] = 0x800c808008808004ull; buff[51] = 0x8080808080808080ull;    // Puts u1 in dest1
        buff[52] = 0x0c80800880800480ull; buff[53] = 0x8080808080808080ull;    // Puts v1 in dest1
        
        buff[54] = 0x8080808080808080ull; buff[55] = 0x8008808004808000ull;    // Puts L2 in dest1
        buff[56] = 0x8080808080808080ull; buff[57] = 0x0880800480800080ull;    // Puts u2 in dest1
        buff[58] = 0x8080808080808080ull; buff[59] = 0x8080048080008080ull;    // Puts v2 in dest1
        
        buff[60] = 0x8080808080800c80ull; buff[61] = 0x8080808080808080ull;    // Puts L2 in dest2
        buff[62] = 0x80808080800c8080ull; buff[63] = 0x8080808080808080ull;    // Puts u2 in dest2
        buff[64] = 0x808080800c808008ull; buff[65] = 0x8080808080808080ull;    // Puts v2 in dest2
        
        buff[66] = 0x0480800080808080ull; buff[67] = 0x80800c8080088080ull;    // Puts L3 in dest2
        buff[68] = 0x8080008080808080ull; buff[69] = 0x800c808008808004ull;    // Puts u3 in dest2
        buff[70] = 0x8000808080808080ull; buff[71] = 0x0c80800880800480ull;    // Puts v3 in dest2
        
        first = false;
    }
    
    const unsigned int widthz = (pitch/3) >> 4;
    
    for( int h=_start; h<_stop; ++h ) {
        const unsigned char *tsrc = src + pitch*h;
        unsigned char *tdest = dest + pitch*h;
        for( int w=0; w<widthz; ++w ) {
            // Load into vectors
            const __m128i v0 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
            const __m128i v1 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
            const __m128i v2 = _mm_load_si128((const __m128i*)tsrc); tsrc += 16;
            
            // ============================ FIRST 8 COLORS
            const __m128i shuffle_r0 = _mm_load_si128((const __m128i*)buff+0);
            const __m128i shuffle_g0 = _mm_load_si128((const __m128i*)buff+1);
            const __m128i shuffle_b0 = _mm_load_si128((const __m128i*)buff+2);
            
            const __m128i shuffle_r1a = _mm_load_si128((const __m128i*)buff+3);
            const __m128i shuffle_r1b = _mm_load_si128((const __m128i*)buff+4);
            const __m128i shuffle_g1a = _mm_load_si128((const __m128i*)buff+5);
            const __m128i shuffle_g1b = _mm_load_si128((const __m128i*)buff+6);
            const __m128i shuffle_b1a = _mm_load_si128((const __m128i*)buff+7);
            const __m128i shuffle_b1b = _mm_load_si128((const __m128i*)buff+8);
            
            __m128i redsi[2], greensi[2], bluesi[2];
            __m128 reds[2], greens[2], blues[2];
            __m128 L[2], u[2], v[2];
            
            redsi[0] = _mm_shuffle_epi8(v0,shuffle_r0);
            greensi[0] = _mm_shuffle_epi8(v0,shuffle_g0);
            bluesi[0] = _mm_shuffle_epi8(v0,shuffle_b0);
            
            redsi[1] = _mm_or_si128( _mm_shuffle_epi8(v0,shuffle_r1a) , _mm_shuffle_epi8(v1,shuffle_r1b) );
            greensi[1] = _mm_or_si128( _mm_shuffle_epi8(v0,shuffle_g1a) , _mm_shuffle_epi8(v1,shuffle_g1b) );
            bluesi[1] = _mm_or_si128( _mm_shuffle_epi8(v0,shuffle_b1a) , _mm_shuffle_epi8(v1,shuffle_b1b) );
            
            // Normalized float values
            register __m128 factor = _mm_set_ps1(0.0039215686274509803f);
            reds[0]   = _mm_mul_ps(_mm_cvtepi32_ps(redsi[0]),  _mm_set_ps1(0.0039215686274509803f));
            reds[1]   = _mm_mul_ps(_mm_cvtepi32_ps(redsi[1]),  _mm_set_ps1(0.0039215686274509803f));
            greens[0] = _mm_mul_ps(_mm_cvtepi32_ps(greensi[0]),_mm_set_ps1(0.0039215686274509803f));
            greens[1] = _mm_mul_ps(_mm_cvtepi32_ps(greensi[1]),_mm_set_ps1(0.0039215686274509803f));
            blues[0]  = _mm_mul_ps(_mm_cvtepi32_ps(bluesi[0]), _mm_set_ps1(0.0039215686274509803f));
            blues[1]  = _mm_mul_ps(_mm_cvtepi32_ps(bluesi[1]), _mm_set_ps1(0.0039215686274509803f));
            
            rgb2lab_sse(reds, greens, blues, L, u, v);
            
            
            L[0] = _mm_add_ps(_mm_mul_ps(L[0], _mm_set_ps1(k_post_coeffs_8u[0])), _mm_set_ps1(k_post_coeffs_8u[1]));
            L[1] = _mm_add_ps(_mm_mul_ps(L[1], _mm_set_ps1(k_post_coeffs_8u[0])), _mm_set_ps1(k_post_coeffs_8u[1]));
            u[0] = _mm_add_ps(_mm_mul_ps(u[0], _mm_set_ps1(k_post_coeffs_8u[2])), _mm_set_ps1(k_post_coeffs_8u[3]));
            u[1] = _mm_add_ps(_mm_mul_ps(u[1], _mm_set_ps1(k_post_coeffs_8u[2])), _mm_set_ps1(k_post_coeffs_8u[3]));
            v[0] = _mm_add_ps(_mm_mul_ps(v[0], _mm_set_ps1(k_post_coeffs_8u[4])), _mm_set_ps1(k_post_coeffs_8u[5]));
            v[1] = _mm_add_ps(_mm_mul_ps(v[1], _mm_set_ps1(k_post_coeffs_8u[4])), _mm_set_ps1(k_post_coeffs_8u[5]));
            
            
            __m128i Li[4], ui[4], vi[4];
            Li[0] = _mm_cvtps_epi32( L[0] );
            Li[1] = _mm_cvtps_epi32( L[1] );
            ui[0] = _mm_cvtps_epi32( u[0] );
            ui[1] = _mm_cvtps_epi32( u[1] );
            vi[0] = _mm_cvtps_epi32( v[0] );
            vi[1] = _mm_cvtps_epi32( v[1] );
            
            const __m128i shuffle_L0 = _mm_load_si128((const __m128i*)buff+18);
            const __m128i shuffle_u0 = _mm_load_si128((const __m128i*)buff+19);
            const __m128i shuffle_v0 = _mm_load_si128((const __m128i*)buff+20);
            
            const __m128i shuffle_L1a = _mm_load_si128((const __m128i*)buff+21);
            const __m128i shuffle_L1b = _mm_load_si128((const __m128i*)buff+24);
            const __m128i shuffle_u1a = _mm_load_si128((const __m128i*)buff+22);
            const __m128i shuffle_u1b = _mm_load_si128((const __m128i*)buff+25);
            const __m128i shuffle_v1a = _mm_load_si128((const __m128i*)buff+23);
            const __m128i shuffle_v1b = _mm_load_si128((const __m128i*)buff+26);
            
            
            __m128i dest0 = _mm_shuffle_epi8(Li[0],shuffle_L0);
            dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(ui[0],shuffle_u0) );
            dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(vi[0],shuffle_v0) );
            
            dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(Li[1],shuffle_L1a) );
            dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(ui[1],shuffle_u1a) );
            dest0 = _mm_or_si128( dest0, _mm_shuffle_epi8(vi[1],shuffle_v1a) );
            
            __m128i dest1 = _mm_shuffle_epi8(Li[1],shuffle_L1b);
            dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(ui[1],shuffle_u1b) );
            dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(vi[1],shuffle_v1b) );
            
            
            // ============================ SECOND 8 NUMBERS
            const __m128i shuffle_r2a = _mm_load_si128((const __m128i*)buff+9);
            const __m128i shuffle_r2b = _mm_load_si128((const __m128i*)buff+10);
            const __m128i shuffle_g2a = _mm_load_si128((const __m128i*)buff+11);
            const __m128i shuffle_g2b = _mm_load_si128((const __m128i*)buff+12);
            const __m128i shuffle_b2a = _mm_load_si128((const __m128i*)buff+13);
            const __m128i shuffle_b2b = _mm_load_si128((const __m128i*)buff+14);
            
            const __m128i shuffle_r3 = _mm_load_si128((const __m128i*)buff+15);
            const __m128i shuffle_g3 = _mm_load_si128((const __m128i*)buff+16);
            const __m128i shuffle_b3 = _mm_load_si128((const __m128i*)buff+17);
            
            redsi[0]   = _mm_or_si128( _mm_shuffle_epi8(v1,shuffle_r2a) , _mm_shuffle_epi8(v2,shuffle_r2b) );
            greensi[0] = _mm_or_si128( _mm_shuffle_epi8(v1,shuffle_g2a) , _mm_shuffle_epi8(v2,shuffle_g2b) );
            bluesi[0]  = _mm_or_si128( _mm_shuffle_epi8(v1,shuffle_b2a) , _mm_shuffle_epi8(v2,shuffle_b2b) );
            
            redsi[1]   = _mm_shuffle_epi8(v2,shuffle_r3);
            greensi[1] = _mm_shuffle_epi8(v2,shuffle_g3);
            bluesi[1]  = _mm_shuffle_epi8(v2,shuffle_b3);
            
            
            // Normalized float values
            reds[0]   = _mm_mul_ps(_mm_cvtepi32_ps(redsi[0]),_mm_set_ps1(0.0039215686274509803f));
            reds[1]   = _mm_mul_ps(_mm_cvtepi32_ps(redsi[1]),_mm_set_ps1(0.0039215686274509803f));
            greens[0] = _mm_mul_ps(_mm_cvtepi32_ps(greensi[0]),_mm_set_ps1(0.0039215686274509803f));
            greens[1] = _mm_mul_ps(_mm_cvtepi32_ps(greensi[1]),_mm_set_ps1(0.0039215686274509803f));
            blues[0]  = _mm_mul_ps(_mm_cvtepi32_ps(bluesi[0]),_mm_set_ps1(0.0039215686274509803f));
            blues[1]  = _mm_mul_ps(_mm_cvtepi32_ps(bluesi[1]),_mm_set_ps1(0.0039215686274509803f));
            
            rgb2luv_sse(reds, greens, blues, L, u, v);
            
            L[0] = _mm_add_ps(_mm_mul_ps(L[0], _mm_set_ps1(k_post_coeffs_8u[0])), _mm_set_ps1(k_post_coeffs_8u[1]));
            L[1] = _mm_add_ps(_mm_mul_ps(L[1], _mm_set_ps1(k_post_coeffs_8u[0])), _mm_set_ps1(k_post_coeffs_8u[1]));
            u[0] = _mm_add_ps(_mm_mul_ps(u[0], _mm_set_ps1(k_post_coeffs_8u[2])), _mm_set_ps1(k_post_coeffs_8u[3]));
            u[1] = _mm_add_ps(_mm_mul_ps(u[1], _mm_set_ps1(k_post_coeffs_8u[2])), _mm_set_ps1(k_post_coeffs_8u[3]));
            v[0] = _mm_add_ps(_mm_mul_ps(v[0], _mm_set_ps1(k_post_coeffs_8u[4])), _mm_set_ps1(k_post_coeffs_8u[5]));
            v[1] = _mm_add_ps(_mm_mul_ps(v[1], _mm_set_ps1(k_post_coeffs_8u[4])), _mm_set_ps1(k_post_coeffs_8u[5]));
            
            Li[2] = _mm_cvtps_epi32( L[0] );
            Li[3] = _mm_cvtps_epi32( L[1] );
            ui[2] = _mm_cvtps_epi32( u[0] );
            ui[3] = _mm_cvtps_epi32( u[1] );
            vi[2] = _mm_cvtps_epi32( v[0] );
            vi[3] = _mm_cvtps_epi32( v[1] );
            
            const __m128i shuffle_L2a = _mm_load_si128((const __m128i*)buff+27);
            const __m128i shuffle_L2b = _mm_load_si128((const __m128i*)buff+30);
            const __m128i shuffle_u2a = _mm_load_si128((const __m128i*)buff+28);
            const __m128i shuffle_u2b = _mm_load_si128((const __m128i*)buff+31);
            const __m128i shuffle_v2a = _mm_load_si128((const __m128i*)buff+29);
            const __m128i shuffle_v2b = _mm_load_si128((const __m128i*)buff+32);
            
            const __m128i shuffle_L3 = _mm_load_si128((const __m128i*)buff+33);
            const __m128i shuffle_u3 = _mm_load_si128((const __m128i*)buff+34);
            const __m128i shuffle_v3 = _mm_load_si128((const __m128i*)buff+35);
            
            
            dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(Li[2],shuffle_L2a) );
            dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(ui[2],shuffle_u2a) );
            dest1 = _mm_or_si128( dest1, _mm_shuffle_epi8(vi[2],shuffle_v2a) );
            
            __m128i dest2 = _mm_shuffle_epi8(ui[2],shuffle_u2b);
            dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(ui[3],shuffle_u3) );
            
            dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(Li[2],shuffle_L2b) );
            dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(Li[3],shuffle_L3) );
            dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(vi[2],shuffle_v2b) );
            dest2 = _mm_or_si128( dest2, _mm_shuffle_epi8(vi[3],shuffle_v3) );
            
            _mm_store_si128((__m128i*)tdest, dest0); tdest+=16;
            _mm_store_si128((__m128i*)tdest, dest1); tdest+=16;
            _mm_store_si128((__m128i*)tdest, dest2); tdest+=16;
            
        }
    }
    
}


inline void luv2rgb(const float* L, const float* u, const float* v,
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
    v1 = (*v)*t + 0.46831096;
    x = 2.25f * u1 * y / v1 ;
    z = (12.0f - 3.0f*u1 - 20.0f*v1) * y / (4.0f*v1);
    
    *r =  3.240479f*x - 1.53715f*y  - 0.498535f*z;
    *g = -0.969256f*x + 1.875991f*y + 0.041556f*z;
    *b =  0.055648f*x - 0.204043f*y + 1.057311f*z;
    
}

void _luv2rgb_8u(const unsigned char* _src, unsigned char* _dest,
                 unsigned int _width, unsigned int _height,
                 unsigned int _pitch, unsigned int _threads) {
    
    
    for( int h=0; h<_height; ++h ) {
        const unsigned char* tsrc = _src+h*_pitch;
        unsigned char* tdest = _dest+h*_pitch;
        for( int w=0; w<_width; ++w ) {
            unsigned char L = tsrc[0];
            unsigned char u = tsrc[1];
            unsigned char v = tsrc[2]; tsrc+=3;
            
            float Lf = k_8u_32f[(int)L]*k_pre_coeffs[0] + k_pre_coeffs[1];
            float uf = k_8u_32f[(int)u]*k_pre_coeffs[2] + k_pre_coeffs[3];
            float vf = k_8u_32f[(int)v]*k_pre_coeffs[4] + k_pre_coeffs[5];
            
            float r,g,b;
            
            luv2rgb(&Lf, &uf, &vf, &r, &g, &b);
            
            int ri = (int)(255.0f*r + (r >= 0.0f ? 0.5f : -0.5f));
            ri = (ri < 0.0f ? 0.0f : ri);
            int gi = (int)(255.0f*g + (g >= 0.0f ? 0.5f : -0.5f));
            gi = (gi < 0.0f ? 0.0f : gi);
            int bi = (int)(255.0f*b + (b >= 0.0f ? 0.5f : -0.5f));
            bi = (bi < 0.0f ? 0.0f : bi);
            
            tdest[0] = static_cast<unsigned char>(ri>255?255:ri);
            tdest[1] = static_cast<unsigned char>(gi>255?255:gi);
            tdest[2] = static_cast<unsigned char>(bi>255?255:bi); tdest+=3;
        }
    }
    
}

