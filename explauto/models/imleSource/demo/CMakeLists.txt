# IMLE DEMO

cmake_minimum_required(VERSION 2.6)

project(demo)


# ---------------- REQUIREMENTS -----------------------

# imle
set( INCLUDE ${INCLUDE} ${imle_INCLUDE_DIRS})
set( LIBS ${LIBS} ${imle_LIBRARIES})

# BOOST
set(Boost_USE_STATIC_LIBS   OFF)
set(Boost_USE_MULTITHREADED ON)
set(BOOST_ALL_DYN_LINK ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS random)
set( INCLUDE ${INCLUDE} ${Boost_INCLUDE_DIRS})
set( LIBS ${LIBS} ${Boost_LIBRARIES})

# -------------      BUILDING     --------------------

include_directories(${INCLUDE})

# IMLE Demo
set(SRC imleDemo.cpp)
set(TARGET imleDemo)
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${LIBS})
