#include "pvcore/imagetransform.h"

class GRADMAG_8U_TBB {
    
private:
    const unsigned char* src;
    unsigned char* dest;
    unsigned int width, height;
    unsigned int threads;
    
public:
    void operator()( const tbb::blocked_range<size_t>& _r ) const {
        
        unsigned int start = (unsigned int)_r.begin()*height/threads;
        unsigned int stop = (unsigned int)_r.end()*height/threads;
        
        unsigned int adjust = 0;
        if( start == 0 ) {
            
            for( unsigned int x=1; x<width-2; x+=1 ) {
                
            }
        }
        
        for( unsigned int y=start+adjust; y<stop; ++y ) {
            for( unsigned int x=0; x<width; x+=1 ) {
                
                int didx = (y*width+x);
                int sidx = (y*width+x);
                
                dest[didx] = src[sidx];
                
            }
        }
        
    }
    
    
    GRADMAG_8U_TBB(const unsigned char* _src, // Source image
                    unsigned char* _dest, // Dest image
                    unsigned int _width,
                    unsigned int _height, //
                    unsigned int _threads)
    :
    src(_src), dest(_dest), width(_width), height(_height), threads(_threads) {}
        
};