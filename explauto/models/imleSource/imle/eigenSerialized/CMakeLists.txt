# Eigen Serialized

cmake_minimum_required(VERSION 2.6)

# ---------------- REQUIREMENTS -----------------------

# BOOST
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS serialization)
set( INCLUDE ${INCLUDE} ${Boost_INCLUDE_DIRS})
set( LIBS ${LIBS} ${Boost_LIBRARIES})


# EIGEN2
set( INCLUDE ${INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/Eigen)


# ----------------- BUILDING --------------------------

set( SRC EigenSerialized.h EigenMatrixBaseSerialize.h )
#set( TARGET eigenSerialized)

# Template only library: build an empty target, so that source files appear in IDE's
# (Not working with CodeBlocks and cmake 2.8.3... )
#add_custom_target( ${TARGET} SOURCES ${SRC} )

SOURCE_GROUP(eigenSerializedHeaders FILES ${SRC})

## Temp solution while the above solution does not work...
#include_directories(${INCLUDE} )
#include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ) # This is for including EigenMatrixBaseSerialize.h
#set(SRC EigenSerialized.cpp ${SRC})
#add_library(${TARGET} ${SRC})
#target_link_libraries(${TARGET} ${LIBS})


# ----------------- FORWARD ---------------------------

set(eigenSerialized_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE} CACHE PATH "eigenSerialized include location")
set(eigenSerialized_LIBRARIES ${LIBS} CACHE PATH "eigenSerialized libraries")
set(eigenSerialized_FOUND TRUE CACHE BOOL "eigenSerialized Found!")

