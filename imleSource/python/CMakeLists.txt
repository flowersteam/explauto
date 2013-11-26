CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# imle
set( INCLUDE ${INCLUDE} ${imle_INCLUDE_DIRS})
set( LIBS ${LIBS} ${imle_LIBRARIES})

include_directories(${INCLUDE})

FIND_PACKAGE(PythonLibs REQUIRED)

FIND_PACKAGE(Boost)
IF(Boost_FOUND)

set( INCLUDE ${INCLUDE} ${Boost_INCLUDE_DIRS})
set( LIBS ${LIBS} ${Boost_LIBRARIES})

  INCLUDE_DIRECTORIES("${PYTHON_INCLUDE_DIRS}")
  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)
  FIND_PACKAGE(Boost COMPONENTS python)

  ADD_LIBRARY(imle SHARED myimle.cpp main-python.cpp)
  TARGET_LINK_LIBRARIES(imle ${Boost_LIBRARIES})
ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()

TARGET_LINK_LIBRARIES(imle ${PYTHON_LIBRARIES})

INCLUDE_DIRECTORIES("/usr/include/python2.7/" "/usr/lib/python2.7/dist-packages/numpy/" "/usr/lib/python2.7/dist-packages/numpy/core/include/")

SET_TARGET_PROPERTIES(imle PROPERTIES OUTPUT_NAME "_imle" PREFIX "" SUFFIX ".so")


include_directories(${INCLUDE})
target_link_libraries(imle ${LIBS})
