#include "pvcore/imagetransform.h"

#include <cstdlib>

namespace pvcore {
    
    
    class FLIPH_8U_TBB {
    private:
        const unsigned char* src;
        unsigned char* dest;
        unsigned int width, height, channels;
        unsigned int pitch;
        unsigned int threads;
        void _fliph_c( unsigned int _start,
                      unsigned int _stop ) const ;
        void _fliph_m( unsigned int _start,
                      unsigned int _stop ) const ;
        
    public:
        void operator()( const tbb::blocked_range<size_t>& _r ) const;
        FLIPH_8U_TBB(const unsigned char* _src, // Source image
                     unsigned char* _dest, // Dest image
                     unsigned int _width,
                     unsigned int _height,
                     unsigned int _nChannels,
                     unsigned int _pitch,
                     unsigned int _threads);
        
    };
    
    
    class FLIPV_8U_TBB {
    private:
        const unsigned char* src;
        unsigned char* dest;
        unsigned int width, height, channels;
        unsigned int pitch;
        unsigned int threads;
        void _flipv_c( unsigned int _start,
                      unsigned int _stop ) const ;
        void _flipv_m( unsigned int _start,
                      unsigned int _stop ) const ;
    public:
        void operator()( const tbb::blocked_range<size_t>& _r ) const;
        
        FLIPV_8U_TBB(const unsigned char* _src, // Source image
                     unsigned char* _dest, // Dest image
                     unsigned int _width,
                     unsigned int _height,
                     unsigned int _nChannels,
                     unsigned int _pitch,
                     unsigned int _threads);
    };
    

    
    // ============================= FLIP HORIZONTAL =============================
    void FLIPH_8U_TBB::_fliph_c( unsigned int _start,
                                unsigned int _stop ) const {
        // This is the number of 3*16 blocks assigned to the thread
        for( unsigned int y=_start; y<_stop; ++y ) {
            unsigned int idx = y*pitch;
            for( unsigned int x=0; x<width; ++x ) {
                memcpy( &dest[idx+3*(width-x-1)], &src[idx+3*x], 3 );
            }
        }
    }
    
    
    void FLIPH_8U_TBB::_fliph_m( unsigned int _start,
                                unsigned int _stop ) const {
        
        for( unsigned int y=_start; y<_stop; ++y ) {
            unsigned int idx = y*pitch;
            for( size_t x=0; x<width; ++x ) {
                dest[idx+width-x-1] = src[idx+x];
            }
        }
        
    }
    
    
    void FLIPH_8U_TBB::operator()( const tbb::blocked_range<size_t>& _r ) const {
        
        // Sort out indices - threads process whole rows
        unsigned int height_p_thread = (height/threads);
        
        unsigned int start = (unsigned int)_r.begin()*height_p_thread;
        unsigned int stop  = (unsigned int)( (threads == _r.end()) ? (size_t)height : _r.end()*height_p_thread );
        
        if( channels == 3 ) {
            _fliph_c( start, stop );
        } else {
            _fliph_m( start, stop );
        }
        
        
    }
    
    
    FLIPH_8U_TBB::FLIPH_8U_TBB(const unsigned char* _src, // Source image
                               unsigned char* _dest, // Dest image
                               unsigned int _width,
                               unsigned int _height,
                               unsigned int _nChannels,
                               unsigned int _pitch,
                               unsigned int _threads) {
        src=_src; dest=_dest; width=_width; height=_height;
        channels=_nChannels; pitch=_pitch; threads=_threads;
    }

    
    
    // ============================= FLIP VERTICAL =============================
    void FLIPV_8U_TBB::_flipv_c( unsigned int _start,
                               unsigned int _stop ) const {
        
        for( unsigned int y=_start; y<_stop; ++y ) {
            memcpy(dest+(pitch*(height-y-1)),
                   src+pitch*y, 3*width );
            memcpy(dest+pitch*y,
                   src+(pitch*(height-y-1)), 3*width );
        }
        
    }
    
    
    void FLIPV_8U_TBB::_flipv_m( unsigned int _start,
                               unsigned int _stop ) const {
        
        for( unsigned int i=_start; i < _stop; ++ i ) {
            memcpy(dest+(width*(height-i-1)),
                   src+width*i, width );
            memcpy(dest+width*i,
                   src+(width*(height-i-1)), width );
        }
        
    }
    
    void FLIPV_8U_TBB::operator()( const tbb::blocked_range<size_t>& _r ) const {
        
        // Sort out indices - threads process whole rows
        unsigned int height_p_thread = (height/(2*threads));
        
        unsigned int start = (unsigned int)_r.begin()*height_p_thread;
        unsigned int stop  = (unsigned int)( (threads == _r.end()) ? (size_t)height/2 + (height%2) : _r.end()*height_p_thread );
        
        if( channels == 3 ) {
            _flipv_c( start, stop );
        } else {
            _flipv_m( start, stop );
        }
        
        
    }
    
    
    FLIPV_8U_TBB::FLIPV_8U_TBB(const unsigned char* _src, // Source image
                               unsigned char* _dest, // Dest image
                               unsigned int _width,
                               unsigned int _height,
                               unsigned int _nChannels,
                               unsigned int _pitch,
                               unsigned int _threads) {
        src=_src; dest=_dest; width=_width; height=_height;
        channels=_nChannels; pitch=_pitch; threads=_threads;
    }
    
    
    
} // namespace pvcore

