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
    SET( HPMPC_TARGET X64_AVX CACHE STRING
   	    "Choose build target: X64_AVX2, X64_AVX, X64_SSE3, CORTEX_A15, CORTEX_A9, C99_4x4"
    )
ENDIF( )

################################################################################
#
# Compiler settings
#
################################################################################

SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -fPIC" )

IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX2" )
    ADD_DEFINITIONS( -DTARGET_X64_AVX2 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mavx2 -mfma")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    ADD_DEFINITIONS( -DTARGET_X64_AVX )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mavx")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    ADD_DEFINITIONS( -DTARGET_X64_SSE3 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -msse3")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A15" )
    ADD_DEFINITIONS( -DTARGET_CORTEX_A15 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -marm -mfloat-abi=hard -mfpu=neon-vfpv4 -mcpu=cortex-a15")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A9" )
    ADD_DEFINITIONS( -DTARGET_CORTEX_A9 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -marm -mfloat-abi=hard -mfpu=neon -mcpu=cortex-a9")

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4x4" )
    ADD_DEFINITIONS( -DTARGET_C99_4X4 )
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")

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
