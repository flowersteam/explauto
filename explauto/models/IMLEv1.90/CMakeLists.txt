cmake_minimum_required(VERSION 2.6)

project(IMLE)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)



# ------------------- SETUP ---------------------------

# Binaries and libraries location
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# Paths to my code
#set(EIGEN_SERIALIZED "eigenSerialized")
set(EigenSerialized "eigenSerializedLib")
set(imleLib "imleLib")
set(imleDemo "imleDemo")
set(imleModule "imleModule")
set(imleModuleDemo "imleModuleDemo")

option(BUILD_IMLE_DEMO "Build IMLE simple demo?" ON)
option(BUILD_IMLE_MODULE "Build YARP module for IMLE (requires YARP)?" OFF)


# ------------------- BUILDING -------------------------

add_subdirectory( ${EigenSerialized} )

add_subdirectory ( ${imleLib} )
if ( BUILD_IMLE_DEMO )
    add_subdirectory ( ${imleDemo} )
endif ( BUILD_IMLE_DEMO )

if ( BUILD_IMLE_MODULE )
    add_subdirectory ( ${imleModule} )

    option(BUILD_IMLE_MODULE_DEMO "Build examples for the YARP module?" ON)
    if ( BUILD_IMLE_MODULE_DEMO )
        add_subdirectory ( ${imleModuleDemo} )
    endif ( BUILD_IMLE_MODULE_DEMO )
endif ( BUILD_IMLE_MODULE )

add_subdirectory( "python" )
