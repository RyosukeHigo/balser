template <typename T>
class CREATE_HISTOGRAM_TBB {
    
    const unsigned char* src;
    const T* mask;
    T maskid;
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    unsigned int srcchannels;
    unsigned int dimx;
    unsigned int dimy;
    unsigned int dimz;
    float* hist;
    unsigned int threads;
    unsigned int histchannels;
    float invtheta;
    
public:
    void operator()( const tbb::blocked_range<size_t>& r ) const {
        
        // Sort out indices
        unsigned int height_p_thread = (height/threads);
        
        unsigned int start = (unsigned int)r.begin()*height_p_thread;
        unsigned int stop  = (unsigned int)( (threads == r.end()) ? height : r.end()*height_p_thread );
        
        _create_histogram( start, stop, (unsigned int)r.begin() );
        
    }
    
    CREATE_HISTOGRAM_TBB( const unsigned char* _src, const T* _mask,
                               T _maskid,
                               unsigned int _width, unsigned int _height,
                               unsigned int _pitch, unsigned int _srcchannels,
                               unsigned int _histdimx,
                               unsigned int _histdimy,
                               unsigned int _histdimz,
                               float* _hist, unsigned int _threads,
                               unsigned int _histchannels, float _invtheta) :
    
    src(_src), mask(_mask), maskid(_maskid), width(_width), height(_height),
    pitch(_pitch), srcchannels(_srcchannels),
    dimx(_histdimx), dimy(_histdimy), dimz(_histdimz), hist(_hist), threads(_threads), histchannels(_histchannels), invtheta(_invtheta) {
    }
    
    
private:
    inline void _create_histogram(unsigned int _h1, unsigned int _h2,
                                  unsigned int _threadId) const {
        
        // Select histogram for current thread
        float* _hist = hist + dimx*dimy*dimz*_threadId;
        
        float idimx = (float)dimx/256.0f;
        float idimy = (float)dimy/256.0f;
        float idimz = (float)dimz/256.0f;
        switch( histchannels ) {
            case 1:
                for( unsigned int i=_h1; i<_h2; ++i ) {
                    const T* __mask = mask + i*pitch/srcchannels;
                    for( unsigned int j=0; j<width; ++j ) {
                        if( __mask[j] != maskid ) continue;
                        float c1 = src[(i*pitch + srcchannels*j)];
                        unsigned int idx1 = (unsigned int)std::floor(c1*idimx);
                        
                        _hist[idx1] += j*invtheta;
                    }
                    /*if( _mask->uncertainty() == 0 ) {
                     continue;
                     }
                     float factor = 1.0f/_mask->uncertainty();
                     for( ; j<mdata[i]+_mask->uncertainty(); ++j ) {
                     unsigned char c1 = data[nChannels*(i*width+j)+0];
                     unsigned int idx1 = std::floor((float)(c1)*idimx);
                     
                     hist[idx1] += (1.0f-(j-mdata[i]))*factor*i*invtheta;
                     }*/
                }
                break;
                
            case 2:
                for( unsigned int i=_h1; i<_h2; ++i ) {
                    const T* __mask = mask + i*pitch/srcchannels;
                    unsigned int j=0;
                    for( ; j<width; ++j ) {
                        if( __mask[j] != maskid ) continue;
                        unsigned int idx = (i*pitch + srcchannels*j);
                        float c1 = (float)src[idx+0];
                        float c2 = (float)src[idx+1];
                        unsigned int idx1 = (unsigned int)std::floor(c1*idimx);
                        unsigned int idx2 = (unsigned int)std::floor(c2*idimy);
                        _hist[idx2*dimx + idx1] += j*invtheta;
                    }
                    
                    
                    /*if( _mask->uncertainty() == 0 ) {
                     continue;
                     }
                     float factor = 1.0f/_mask->uncertainty();
                     for( ; j<mdata[i]+_mask->uncertainty(); ++j ) {
                     unsigned char c1 = data[3*(i*width+j)+0];
                     unsigned char c2 = data[3*(i*width+j)+1];
                     unsigned int idx1 = std::floor((float)(c1)*idimx);
                     unsigned int idx2 = std::floor((float)(c2)*idimy);
                     
                     hist[idx2*dimx + idx1] += (1.0f-(j-mdata[i]))*factor*i*invtheta;
                     }*/
                }
                
                break;
                
            case 3:
                for( unsigned int i=_h1; i<_h2; ++i ) {
                    const T* __mask = mask + i*pitch/srcchannels;
                    unsigned int j=0;
                    for( ; j<width; ++j ) {
                        if( __mask[j] != maskid ) continue;
                        unsigned int idx = (i*pitch + srcchannels*j);
                        float c1 = (float)src[idx+0];
                        float c2 = (float)src[idx+1];
                        float c3 = (float)src[idx+2];
                        unsigned int idx1 = (unsigned int)std::floor(c1*idimx);
                        unsigned int idx2 = (unsigned int)std::floor(c2*idimy);
                        unsigned int idx3 = (unsigned int)std::floor(c3*idimz);
                        
                        _hist[idx3*dimy*dimx + idx2*dimx + idx1] += j*invtheta;
                    }
                }
                break;
                
        }
        
    }
    
};
