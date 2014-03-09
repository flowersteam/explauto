# IMLE ALGORITHM

cmake_minimum_required(VERSION 2.6)

project(imle)

# ---------------- REQUIREMENTS -----------------------

# eigenSerialized
add_subdirectory( "eigenSerialized" )
set( INCLUDE ${INCLUDE} ${eigenSerialized_INCLUDE_DIRS})
set( LIBS ${LIBS} ${eigenSerialized_LIBRARIES})

# -------------      BUILDING     --------------------

set(SRC imle.hpp _imle.hpp expert.hpp _expert.hpp ILearner.hpp)
SOURCE_GROUP(imleHeaders FILES ${SRC})

# ----------------- FORWARD ---------------------------

set(imle_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE} CACHE PATH "imle include location")
set(imle_LIBRARIES ${LIBS} CACHE PATH "imle libraries")
set(imle_FOUND TRUE CACHE BOOL "imle Found!")

