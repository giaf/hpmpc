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
PROJECT( HPMPC C )

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
    SET( HPMPC_TARGET X64_SSE3 CACHE STRING
   	    "Choose build target: C99_4x4, C99_2x2, X64_SSE3, X64_AVX, X86_ATOM, POWERPC_G2"
    )
ENDIF( )

################################################################################
#
# Compiler settings
#
################################################################################

SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic -Wfloat-equal -fPIC -std=c99" )

IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    ADD_DEFINITIONS( -DTARGET_X64_AVX )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mavx")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    ADD_DEFINITIONS( -DTARGET_X64_SSE3 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -msse3")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "NEON" )
    ADD_DEFINITIONS( -DTARGET_NEON )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "POWERPC_G2" )
    ADD_DEFINITIONS( -DTARGET_POWERPC_G2 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mcpu=603e")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4x4" )
    ADD_DEFINITIONS( -DTARGET_C99_4X4 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_2x2" )
    ADD_DEFINITIONS( -DTARGET_C99_2X2 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X86_ATOM" )
    ADD_DEFINITIONS( -DTARGET_X86_ATOM )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -msse3 -mfpmath=sse -march=atom")

ELSE( )
    MESSAGE( ERROR "You chose unsupported target ${HPMPC_TARGET}")	  
ENDIF( )

################################################################################
#
# Configure the target header file
#
################################################################################

STRING(TOUPPER ${HPMPC_TARGET} TARGET_DEF)
SET( TARGET_DEF TARGET_${TARGET_DEF})

CONFIGURE_FILE(
	"${PROJECT_SOURCE_DIR}/include/target.h.in"
	"${PROJECT_BINARY_DIR}/include/target.h"
	@ONLY
)

################################################################################
#
# Include directories
#
################################################################################

INCLUDE_DIRECTORIES( include )

################################################################################
#
# Build the library
#
################################################################################

MESSAGE( STATUS "Compiling the library for ${HPMPC_TARGET} target")

UNSET( SOURCES )
ADD_SUBDIRECTORY( kernel )
ADD_SUBDIRECTORY( blas )
ADD_SUBDIRECTORY( auxiliary )
ADD_SUBDIRECTORY( lqcp_solvers )
ADD_SUBDIRECTORY( mpc_solvers )
ADD_SUBDIRECTORY( codegen )

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
    DIRECTORY
		${PROJECT_BINARY_DIR}/include/
    DESTINATION
        include/hpmpc
)

INSTALL(
	TARGETS
	    hpmpc
    DESTINATION
        lib
)

################################################################################
#
# Build examples
#
################################################################################

ADD_SUBDIRECTORY( test_problems )
