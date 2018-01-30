//
//  core_cl.cpp
//
//
//  Created by Niklas Bergström on 2013-06-26.
//
//

#include "pvcore/cl.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iostream>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

// Defines for kernel source directory.
// "X" should be defined at compile time
#define KDIR(X) #X
#define KERNELDIR_Q(X) KDIR(X)

// Define names for settings folders
#define SETTINGS_FOLDER "/.pvcore/"
#define BINARIES_FOLDER "binaries/"

// For
#ifdef _WINDOWS
#  include <windows.h>
#  include <ShlObj.h>
#  include <direct.h>
#else
#  include <unistd.h>
#  include <pwd.h>
#endif


namespace pvcore {

	// Private methods for cl operations
	namespace utils {

		/**@cond fileutils*/
		bool cachedCLBinariesExists( const char* _filename, int _device );
		int cacheCLBinaries( const char* _binaries, const size_t _size, const char* _filename, int _device );

		// Loads kernel sources or, if they exist, binaries
		void loadKernelBinaries( char** _binaries, size_t& _size, const char* _filename, int _device );
		void loadKernelSources( const char* _filename, int _device, std::string& _source );

		// Folder manipulation
		void getBinariesFolder( char* path );
		void getSourcesFolder( char* path );

		// Create folders
		void getSettingsFolder( std::string& path );
		int createFolder( const char* path );
		void createSettingsFolder( void );

		// Getting filename for kernels
		char* getFilenameWithExtension( const char* _filename, int _device, const char* ext);

	}


	// VARIABLES
	bool g_cl_inited;
	std::vector<cl::Platform> g_cl_platforms; // Assume one platform
	std::vector<cl::Device> g_cl_devices; // Assume one or more devices
	std::map<cl::Device*,int> g_cl_device_types;
	cl::Context  g_cl_context;
	std::map<cl::Device*,cl::CommandQueue*> g_cl_command_queue;
	std::vector<bool> g_cl_available_device_types;
	bool g_force_recompile;


	std::vector<cl::Device> cl_nvd_devices; // Assume many devices
	std::vector<cl::Device> cl_amd_devices; // Assume many devices
	std::vector<cl::Device> cl_intel_devices; // Assume many devices





	void CLFree() {

		for( int i=(int)g_cl_devices.size()-1; i>=0; --i ) {
			cl::Device& dev = g_cl_devices[i];
			delete g_cl_command_queue.at(&dev);
		}

		cl_nvd_devices.clear();
		cl_amd_devices.clear();
		cl_intel_devices.clear();

		g_cl_available_device_types.clear();

		g_cl_devices.clear();

		g_cl_platforms.clear();
	}



	void CLInit( bool _printInfo, bool _forceCompile, cl_context_properties properties[] ) {

		g_force_recompile = _forceCompile;

		//        if( g_cl_inited ) {
		//            CLFree();
		//        }

		cl_int err = CL_SUCCESS;
		std::string vendor;
		cl_device_type type;

		g_cl_available_device_types.push_back(false);
		g_cl_available_device_types.push_back(false);
		g_cl_available_device_types.push_back(false);

		// Get platform
		err = cl::Platform::get( &g_cl_platforms );
		if( err != CL_SUCCESS ) {
			std::cout << "Error getting platform\n";
			exit(1);
		}


		// Get all GPU-devices
		err = g_cl_platforms[0].getDevices( CL_DEVICE_TYPE_GPU, &g_cl_devices );
		if( err != CL_SUCCESS ) {
			std::cout << "Error getting devices\n";
			exit(1);
		}


		// Create one context for all devices
		g_cl_context = cl::Context( g_cl_devices[0], properties, NULL, NULL, &err );
		if( err != CL_SUCCESS ) {
			std::cout << "Error creating OpenCL context: " << err;
			exit(1);
		}

		// Create one command queue for each device
		for( int i=0; i<g_cl_devices.size(); ++i ) {
			g_cl_devices[i].getInfo( CL_DEVICE_TYPE, &type );
			g_cl_devices[i].getInfo( CL_DEVICE_VENDOR, &vendor );
			if( vendor.find("NVIDIA") == 0 ) {
				g_cl_available_device_types[CL_NVD_TYPE] = true;
				cl_nvd_devices.push_back(g_cl_devices[i]);
				g_cl_command_queue[&g_cl_devices[i]] = new cl::CommandQueue(g_cl_context, g_cl_devices[i]);
				g_cl_device_types[&g_cl_devices[i]] = CL_NVD_TYPE;
			} else if( vendor.find("AMD") == 0 ) {
				g_cl_available_device_types[CL_AMD_TYPE] = true;
				cl_amd_devices.push_back(g_cl_devices[i]);
				g_cl_command_queue[&g_cl_devices[i]] = new cl::CommandQueue(g_cl_context, g_cl_devices[i]);
				g_cl_device_types[&g_cl_devices[i]] = CL_AMD_TYPE;
			} /*else if( vendor.find("Intel") == 0 ) {
			  g_cl_available_device_types[CL_INTEL_TYPE] = true;
			  cl_intel_devices.push_back(g_cl_devices[i]);
			  g_cl_command_queue[&g_cl_devices[i]] = new cl::CommandQueue(g_cl_context, g_cl_devices[i]);
			  g_cl_device_types[&g_cl_devices[i]] = CL_INTEL_TYPE;
			  }*/
		}


		// Print info about all connected devices
		if( _printInfo ) {
			for( int i=0; i<g_cl_devices.size(); ++i ) {
				std::string info;

				g_cl_devices[i].getInfo( CL_DEVICE_NAME, &info );
				std::cout << "Device name: " << info << std::endl;

				g_cl_devices[i].getInfo( CL_DEVICE_EXTENSIONS, &info );
				std::cout << "Device supports extensions: " << info << std::endl;

				cl_device_local_mem_type memtype;
				g_cl_devices[i].getInfo( CL_DEVICE_LOCAL_MEM_TYPE, &memtype );
				if( memtype == CL_LOCAL) {
					std::cout << "Device local memory type is CL_LOCAL"  << std::endl << std::endl;
				} else if( memtype == CL_GLOBAL) {
					std::cout << "Device local memory type is CL_GLOBAL" << std::endl << std::endl;
				}
			}
		}


		// Indicate that OpenCL is inited
		g_cl_inited = true;
	}


	std::mutex loadProgramMutex;

	cl_int CLLoadProgram(std::vector<cl::Device>& _devices,
		std::map< cl::Device*, cl::Program>& _programs,
		const char* _filename,
		std::string& _flags ) {

			cl_int err = CL_SUCCESS;

			std::lock_guard<std::mutex> lock(loadProgramMutex);

			// Load program for each device
			for( int i=0; i<_devices.size(); ++i ) {
				cl::Device& device = _devices[i];
				int device_type = g_cl_device_types[&device]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;

				cl::Program program;
				// =============================================================
				// Cached binaries exist. Load and build them!
				// =============================================================
				if( utils::cachedCLBinariesExists( _filename, device_type ) && !g_force_recompile ) {
					size_t size[1];

					char** rawbinaries = new char*[1];
					utils::loadKernelBinaries(&(rawbinaries[0]), size[0], _filename, device_type );

					cl::Program::Binaries binaries( 1, std::make_pair( rawbinaries[0],size[0]) );

					std::vector<cl::Device> devices;
					devices.push_back(device);

					switch( device_type ) {
					case CL_NVD_TYPE:
						program = cl::Program( g_cl_context, devices, binaries );
						err = program.build( devices );
						break;
					case CL_AMD_TYPE:
						program = cl::Program( g_cl_context, devices, binaries );
						err = program.build( devices );
						break;
					case CL_INTEL_TYPE:
						program = cl::Program( g_cl_context, devices, binaries );
						err = program.build( devices );
						break;
					default:
						break;
					}

					delete [] *rawbinaries;
					delete [] rawbinaries;


					// =============================================================
					// No cached binaries exist. Load from source, build and cache
					// =============================================================
				} else {

					std::string rawsource;
					utils::loadKernelSources( _filename, device_type, rawsource );

					std::pair<const char*, ::size_t> src;
					src.first = rawsource.c_str();
					src.second = rawsource.length();
					cl::Program::Sources sources;
					sources.push_back( src );
					program = cl::Program( g_cl_context, sources );


					// Set build options
					std::string buildOptions = "-Ibin";
					buildOptions += _flags;

					// Output build info
					switch( device_type ) {
					case CL_NVD_TYPE:
						// Seems like building for all devices is needed
						err = program.build( g_cl_devices, buildOptions.c_str() );
						if( err != CL_SUCCESS ) {
							std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl_nvd_devices[0]) << std::endl;
							std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl_nvd_devices[0]) << std::endl;
							std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_nvd_devices[0]) << std::endl;
							return err;
						}
						break;
					case CL_AMD_TYPE:
						err = program.build( g_cl_devices, buildOptions.c_str() );
						if( err != CL_SUCCESS ) {
							std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl_amd_devices[0]) << std::endl;
							std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl_amd_devices[0]) << std::endl;
							std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_amd_devices[0]) << std::endl;
							return err;
						}
						break;
					case CL_INTEL_TYPE:
						err = program.build( g_cl_devices, buildOptions.c_str() );
						if( err != CL_SUCCESS ) {
							std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl_intel_devices[0]) << std::endl;
							std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl_intel_devices[0]) << std::endl;
							std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_intel_devices[0]) << std::endl;
							return err;
						}
						break;
					default:
						// NOT AVAILABLE
						break;
					}

					// A list of pointers to the binary data
					std::vector<size_t> sizes;

					cl_int ndev = program.getInfo<CL_PROGRAM_NUM_DEVICES>();
					std::cout << "ndev: " << ndev << std::endl;

					//A list of pointers to the binary data
					sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();

					std::vector<unsigned char*> binaries;
					std::vector<char*> bnz = program.getInfo<CL_PROGRAM_BINARIES>(&err);

					// Cache appropriate binaries
					for( int j=0; j<sizes.size(); ++j ) {
						if( device_type == g_cl_device_types[&(_devices[j])] ) {
							utils::cacheCLBinaries((const char*)bnz[j], sizes[j], _filename, device_type );
						}
					}

					std::vector<cl::Kernel> kernels;
					err = program.createKernels(&kernels);

				}

				_programs[&device] = program;

			}

			return err;
	}



	namespace utils {


		// Function for checking whether cached binaries exist
		bool cachedCLBinariesExists(const char* _filename, int _device) {

			// Get filename file extension
			char* filename = getFilenameWithExtension( _filename, _device, (const char*)"ptx");

			// Get binaries folder
			char path[256];
			getBinariesFolder( path );

			// Try to open
#ifdef _WINDOWS
			strcat_s(path,filename);
#else
			strcat(path,filename);
#endif

			delete [] filename;
			filename = NULL;
#ifdef _WINDOWS
			WIN32_FIND_DATA FindFileData;
			HANDLE handle = FindFirstFile(path, &FindFileData) ;
			int found = handle != INVALID_HANDLE_VALUE;
			if(found) 
			{
				FindClose(handle);
				return true;
			}
#else
			if( access( path, F_OK ) != -1 ) {
				return true;
			}
#endif
			else {
				return false;
			}

		}


		// Function for caching binaries
		int cacheCLBinaries(const char* _binaries, const size_t _size, const char* _filename, int _device ) {
			// Make sure the folder exists
			createSettingsFolder();

			// Change file extension
			char* filename = getFilenameWithExtension( _filename, _device, (const char*)"ptx");


			// Get binaries folder
			char path[256];
			getBinariesFolder(path);

			// Save the binary
			strcat(path,filename);
			delete [] filename;
			filename = NULL;

			FILE* fp = fopen(path, "w");
			// This shouldn't happen since we check for the existence first
			if(fp == NULL) {
				return(EXIT_FAILURE);
			}
			fwrite(_binaries,sizeof(unsigned char),_size,fp);
			fclose(fp);

			std::cout << "Cached\n";

			return 0;
		}

		// Loads kernel from cached binaries
		void loadKernelBinaries(char** _binaries, size_t& _size, const char* _filename, int _device) {

			// Get filename file extension
			char* filename = getFilenameWithExtension( _filename, _device, (const char*)"ptx");

			// Get binaries folder
			char path[256];
			getBinariesFolder(path);

			// Try to open
			strcat(path,filename);

			struct stat statbuf;

			stat(path, &statbuf);
			_size = statbuf.st_size;

			std::ifstream binaryfile(path,  std::ifstream::in );
			if(binaryfile.is_open()) {
				(*_binaries) = new char[_size];
				binaryfile.read((*_binaries), _size);
				binaryfile.close();
			}

			delete [] filename;

		}

		// Loads kernel from source and returns the source as a char array
		void loadKernelSources( const char* _filename, int _device, std::string& _source ) {

			char* filename = getFilenameWithExtension( _filename, _device, (const char*)"cl");

			// Concatenate filename with kernel directory
			char path[256];
			path[0] = '\0';
			getSourcesFolder(path);

			strcat(path,filename);

			std::cout << "kernel sources: " << path << std::endl;

			// Open file
			std::ifstream sourcefile(path);
			std::string tmp( std::istreambuf_iterator<char>(sourcefile),
				(std::istreambuf_iterator<char>()) );

			_source.swap(tmp);

			//delete [] filename;

		}


		char* getFilenameWithExtension( const char* _filename, int _device, const char* ext) {
			// Get filename for specific device
			size_t endidx = strcspn(_filename,(const char*)"\0");
			char* filename = new char[endidx+8];
			strncpy(filename,_filename,endidx);
			filename[endidx] = '\0';
			switch( _device ) {
			case CL_NVD_TYPE: // NVIDIA
				strcat(filename,"_nvd.\0");
				break;
			case CL_AMD_TYPE: // AMD
				strcat(filename,"_amd.\0");
				break;
				/*                case CL_CPU_TYPE: // CPU
				strcat(filename,"_cpu.\0");
				break;*/
			case CL_INTEL_TYPE: // INTEL
				strcat(filename,"_intel.\0");
				break;
			}
			strcat(filename,ext);

			return filename;
		}



		void getBinariesFolder( char* path ) {
			// Get home folder
#ifdef _WINDOWS
			char homefolder[256];
			if (!SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, 0, homefolder))) {
				return;
			}
#else
			struct passwd *pw = getpwuid(getuid());
			const char* homefolder = (pw->pw_dir);
#endif
			strcpy(path,homefolder);
			strcat(path,SETTINGS_FOLDER);
			strcat(path,BINARIES_FOLDER);
		}




		void getSourcesFolder( char* path ) {
#ifdef _WINDOWS

#else
			getwd(path);
#endif

			// Set at compiletime using compiler flag
			strcat(path,"/share/pvcore/kernels/");
			//            strcat(path,"/../../share/pvcore/kernels/");
		}



		// ========================== Folder management
		void getSettingsFolder( std::string& path ) {
			// Get home folder
#ifdef _WINDOWS
			char homefolder[256];
			if (!SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, 0, homefolder))) {
				return;
			}
#else
			struct passwd *pw = getpwuid(getuid());
			const char* homefolder = (pw->pw_dir);
#endif           
			path += homefolder;
			path += SETTINGS_FOLDER;
		}



		int createFolder(const char* path) {
			FILE *fp;
			fp = fopen(path, "r");

			if(fp == NULL) {
				int err;
#ifdef _WINDOWS
				if ((err = _mkdir(path)) == -1)
#else
				if ((err = mkdir(path, 0755)) == -1)
#endif
				{
					printf("Couldn't create directory: %s\n",strerror(err));
					return(EXIT_FAILURE);
				}
			} else {
				fclose(fp);
			}

			return 0;
		}



		void createSettingsFolder(void) {
#ifdef _WINDOWS
			char homefolder[256];
			if (!SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, 0, homefolder))) {
				return;
			}
#else
			struct passwd *pw = getpwuid(getuid());
			const char* homefolder = (pw->pw_dir);
#endif            

			char path[256];
			strcpy(path,homefolder);

			// Create settings folder
			strcat(path,SETTINGS_FOLDER);
			createFolder(path);

			// Create binaries folder
			strcat(path,BINARIES_FOLDER);
			createFolder(path);
		}



	} // namespace utils

} // namespace pvcore
