################################################################################
#
# Project settings
#
################################################################################

#
# Minimum required version of cmake 
#
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

#
# Project name and programming languages used
#
PROJECT( HPMPC )
set(CMAKE_C_STANDARD 99)

#
# Folder path for generated executables
#
SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )

#
# Folder path for generated libraries
#
SET( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )

################################################################################
#
# User configuration
#
################################################################################

IF( NOT CMAKE_VERBOSE_MAKEFILE )
	SET( CMAKE_VERBOSE_MAKEFILE OFF )
ENDIF( NOT CMAKE_VERBOSE_MAKEFILE )

#
# Build type
#
IF( NOT CMAKE_BUILD_TYPE )
	SET(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
		FORCE
	)
ENDIF( NOT CMAKE_BUILD_TYPE )

#
# Target type
#
IF ( NOT HPMPC_TARGET )
    SET( HPMPC_TARGET X64_AVX CACHE STRING
   	    "Choose build target: X64_AVX2, X64_AVX, X64_SSE3, CORTEX_A57, CORTEX_A15, CORTEX_A9, CORTEX_A7, C99_4x4"
    )
ENDIF( )

#
# Build options
#
option(HPMPC_BUILD_TESTS "Build tests" ON)

################################################################################
#
# Compiler settings
#
################################################################################

SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC" )

IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX2" )
    ADD_DEFINITIONS( -DTARGET_X64_AVX2 )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -mavx2 -mfma")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    ADD_DEFINITIONS( -DTARGET_X64_AVX )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -mavx")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    ADD_DEFINITIONS( -DTARGET_X64_SSE3 )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -msse3")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A57" )
    ADD_DEFINITIONS( -DTARGET_CORTEX_A57 )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+fp+simd -mcpu=cortex-a57")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A15" )
    ADD_DEFINITIONS( -DTARGET_CORTEX_A15 )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -marm -mfloat-abi=hard -mfpu=neon-vfpv4 -mcpu=cortex-a15")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A9" )
    ADD_DEFINITIONS( -DTARGET_CORTEX_A9 )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -marm -mfloat-abi=hard -mfpu=neon -mcpu=cortex-a9")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A7" )
    ADD_DEFINITIONS( -DTARGET_CORTEX_A7 )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -marm -mfloat-abi=hard -mfpu=neon-vfpv4 -mcpu=cortex-a7")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4X4" )
    ADD_DEFINITIONS( -DTARGET_C99_4X4 )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4X4_PREFETCH" )
    ADD_DEFINITIONS( -DTARGET_C99_4X4_PREFETCH )
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")

ELSE( )
    MESSAGE( ERROR "You chose unsupported target ${HPMPC_TARGET}")	  
ENDIF( )

# Link math library against all targets
LINK_LIBRARIES( m )

# This will add the "make test" target
ENABLE_TESTING()

################################################################################
#
# Configure the target header file
#
################################################################################

STRING(TOUPPER ${HPMPC_TARGET} TARGET_DEF)
SET( TARGET_DEF TARGET_${TARGET_DEF})

CONFIGURE_FILE(
	"${PROJECT_SOURCE_DIR}/include/target.h.in"
	"${CMAKE_CURRENT_BINARY_DIR}/include/target.h"    # Spit target.h in the build dir, don't pollute the source folder.
	@ONLY
)

################################################################################
#
# Include directories
#
################################################################################

INCLUDE_DIRECTORIES( include )
INCLUDE_DIRECTORIES( "${CMAKE_CURRENT_BINARY_DIR}/include" )    # Make target.h accessible

################################################################################
#
# Build the library
#
################################################################################

MESSAGE( STATUS "Compiling the library for ${HPMPC_TARGET} target")

#
# Macro for adding source files
#
MACRO( HPMPC_ADD_SOURCE_FILES )
#	MESSAGE( STATUS "${ARGV}" )
	UNSET( _SRC )
    FOREACH( SRC ${ARGV} )
        GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)		
		SET( _SRC ${_SRC} ${ABS_SRC} )
	ENDFOREACH()
	SET ( SOURCES ${SOURCES} ${_SRC} PARENT_SCOPE )
ENDMACRO( HPMPC_ADD_SOURCE_FILES )

#
# Add source folders
#
UNSET( SOURCES )
ADD_SUBDIRECTORY( kernel )
ADD_SUBDIRECTORY( blas )
ADD_SUBDIRECTORY( auxiliary )
ADD_SUBDIRECTORY( lqcp_solvers )
ADD_SUBDIRECTORY( mpc_solvers )
ADD_SUBDIRECTORY( interfaces )
ADD_SUBDIRECTORY( reference_code )
#ADD_SUBDIRECTORY( codegen )

#
# Add library
#
ADD_LIBRARY( hpmpc STATIC ${SOURCES} )

################################################################################
#
# Installation rules
#
################################################################################

INSTALL(
    DIRECTORY
	    include/
    DESTINATION
        include/hpmpc
	PATTERN "*.h.in*" EXCLUDE
)

INSTALL(
	TARGETS
	    hpmpc
    DESTINATION
        lib
)

INSTALL(
	FILES
	    "${CMAKE_CURRENT_BINARY_DIR}/include/target.h"
    DESTINATION
        "include/hpmpc"
)

################################################################################
#
# Build examples
#
################################################################################

ADD_SUBDIRECTORY( test_problems )

################################################################################
#
# Build tests
#
################################################################################
if(HPMPC_BUILD_TESTS)
    add_subdirectory(test)
endif()
