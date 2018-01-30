//
//  core_cl.h
//  
//
//  Created by Niklas Bergström on 2013-06-26.
//
//

#ifndef _CORECL_H_
#define _CORECL_H_

#ifdef __linux__
#include <CL/cl.hpp>
#elif defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#elif defined(_WIN64)
#include <OpenCL/cl.hpp>
#endif // __linux__

#include <map>

namespace pvcore {
    
#define PV_GLOBAL_SIZE(x,y) ( ((x) % (y) == 0) ? (x) : ((x) + ((y) - (x) % (y)) ) )
    
    /** \brief Different device types
     *  The different available device types
     *	Can be set through the PVEnvironment constructor or,
     *	if not set there, by the defaults or a command line argument.
     */
    enum eOpenCLTypes {
//        CL_CPU_TYPE = 0, // Currently CPU is not accepted as device
        CL_NVD_TYPE = 0,
        CL_AMD_TYPE,
        CL_INTEL_TYPE,
        CL_NO_TYPE
    };
    
    
    
    // ============================= VARIABLES =============================
    // Global variables holding information about the OpenCL devices etc.
    // =====================================================================
    // Indicates whether the OpenCL environment is loaded
    extern bool g_cl_inited;
    // Vector of available platforms (assumes one)
    extern std::vector<cl::Platform> g_cl_platforms;
    // Vector of available devices
    extern std::vector<cl::Device> g_cl_devices;
    // Map that holds the device type of each device
    extern std::map<cl::Device*,int> g_cl_device_types;
    // Context (one for all devices => not threadsafe!!!)
    extern cl::Context g_cl_context;
    // Map with command queues (one per device)
    extern std::map<cl::Device*,cl::CommandQueue*> g_cl_command_queue;
    // Vector indicating which of the device types are available
    extern std::vector<bool> g_cl_available_device_types;
    // Indicator to the initialization whether kernels should be recompiled
    extern bool g_force_recompile;
    
    
    // INIT FUNCTION
    void CLInit( bool _printInfo, bool _forceCompile, cl_context_properties properties[] = NULL );
    void CLFree();
    cl_int CLLoadProgram(std::vector<cl::Device>& _devices,
                         std::map< cl::Device*, cl::Program>& _programs,
                         const char* _filename,
                         std::string& _flags );
    
}

#endif // _CORECL_H_
