#include <cmath>

namespace pvcore {
    
    class CART2POL_8U_TBB {
    private:
        const unsigned char* src;
        unsigned char* dest;
        float xc, yc;
        unsigned int width, height;
        unsigned int nChannels;
        unsigned int pitchs;
        unsigned int maxrad;
        unsigned int theta;
        unsigned int pitchd;
        short* maxr;
        unsigned int threads;
        static float *mycos, *mysin;
        static int mytheta;
        
    public:
        CART2POL_8U_TBB(const unsigned char* _src, // Source image
                        unsigned char* _dest, // Dest image
                        float _xc, float _yc, // Center of transform
                        unsigned int _width, // Image dimension
                        unsigned int _height,
                        unsigned int _nChannels,
                        unsigned int _pitchs, // Stride source
                        unsigned int _maxrad, // Destination image dimension
                        unsigned int _theta,
                        unsigned int _pitchd, // Stride dest
                        short* _maxr, // (returned) Radius for each row
                        unsigned int _threads);
        
        void operator()( const tbb::blocked_range<size_t>& _r ) const;
        
    };
    
    void CART2POL_8U_TBB::operator()( const tbb::blocked_range<size_t>& _r ) const {
        
        unsigned int start = (unsigned int)_r.begin()*theta/threads;
        unsigned int stop = (unsigned int)_r.end()*theta/threads;
        
        for( unsigned int t=start; t<stop; ++t ) {
            maxr[t] = maxrad;
            for( unsigned int r=0; r<maxrad; r+=1 ) {
                
                int x = (int)xc + (int)((r)*mycos[t]);
                int y = (int)yc + (int)((r)*mysin[t]);
                
                int didx = (t*pitchd+nChannels*r);
                int sidx = (y*pitchs+nChannels*x);
                
                if( x<0 || x>=(int)width || y<0 || y>=(int)height ) {
                    maxr[t] = std::min(maxr[t],(short)r);
                    //unsigned int* dtmp = (unsigned int*)&(dest[didx]);
                    
                    //*dtmp = *dtmp & 0xFF000000;
                    
                    memset(&(dest[didx]), 0, nChannels*(maxrad-r)); // FILL ALL AND BREAK AFTER HERE!!!
                    break;
                    
                } else {
                    memcpy(&(dest[didx]), &(src[sidx]), nChannels);
                }
            }
        }
    }

    
    CART2POL_8U_TBB::CART2POL_8U_TBB(const unsigned char* _src, // Source image
                                     unsigned char* _dest, // Dest image
                                     float _xc, float _yc, // Center of transform
                                     unsigned int _width, // Image dimension
                                     unsigned int _height,
                                     unsigned int _nChannels,
                                     unsigned int _pitchs, // Stride source
                                     unsigned int _maxrad, // Destination image dimension
                                     unsigned int _theta,
                                     unsigned int _pitchd, // Stride dest
                                     short* _maxr, // (returned) Radius for each row
                                     unsigned int _threads) {
        
        src=_src, dest=_dest; xc=_xc; yc=_yc;
        width=_width; height=_height; nChannels=_nChannels; pitchs=_pitchs;
        maxrad=_maxrad; theta=_theta; pitchd=_pitchd; maxr=_maxr; threads=_threads;
        
        if( _theta != mytheta ) {
            if( mycos != NULL ) {
                delete [] mycos;
            }
            if( mysin != NULL ) {
                delete [] mysin;
            }
            mycos = new float[theta];
            mysin = new float[theta];
            
            for( unsigned int i=0; i<theta; ++i ) {
                mycos[i] = std::cos((float)i*2.0f*3.141562f/(float)(theta+1));
                mysin[i] = std::sin((float)i*2.0f*3.141562f/(float)(theta+1));
            }
            mytheta = _theta;
        }
    }
    
    float* CART2POL_8U_TBB::mycos = NULL;
    float* CART2POL_8U_TBB::mysin = NULL;
    int CART2POL_8U_TBB::mytheta = -1;
    
} // namespace pvcore