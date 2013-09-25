
This library provides a C++ implementation of IMLE: for details on this algorithm see

    "Online Learning of Multi-valued Functions with an Infinite Mixture of Linear Experts",
    Bruno Damas and José-Santos Victor, Neural Computation (submitted)

Copyright (C) 2012, Bruno Damas
email: bdamas@isr.ist.utl.pt

************************************

Software Requirements:
 - Boost 1.44 or newer
 
The library is a header only library: to use it, just include "imle.hpp" in your code. It relies heavily on Eigen, a C++ template library for linear algebra (http://eigen.tuxfamily.org). IMLE library is in a beta stage, and much of the documentation is still missing. However, a very simple demo is included with the code; this, together with the inspection of the imle class interface, should provide some guidelines on how to easily integrate the library in your code.

To compile the demo you will need Cmake build system (version 2.6 or newer). In Linux, starting from the root directory of the source code and issuing the following commands should generate the executable demo under build/bin (this was sucessfully tested using gcc 4.5.2):
    mkdir build
    cd build
    cmake ..
    make
    
To compile the demo code in Windows, under Visual Studio (or other IDE compatible with CMake), the easiest way to use CMake is via its GUI. The code was sucessfully compiled and tested using Visual Studio C++ 2010 Express, under Windows XP.

************************************



