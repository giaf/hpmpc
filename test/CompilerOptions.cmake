################################################################################
#
# Compiler settings - General
#
################################################################################

INCLUDE( CheckCXXCompilerFlag )

################################################################################
#
# Compiler settings - GCC/G++; Linux, Apple
#
################################################################################
MESSAGE(CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID})
IF (    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
	
	SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC" )

################################################################################
#
# Compiler settings - MS Visual Studio; Windows
#
################################################################################
ELSEIF( MSVC )
    # On MSVC, use statically linked C runtime for release builds (to avoid problems with missing runtime).
	SET(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /D NDEBUG")
	SET(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1")

	SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nologo -EHsc " )
	
	#
	# Some common definitions
	#
	ADD_DEFINITIONS( -DWIN32 )
	#ADD_DEFINITIONS( -D__NO_COPYRIGHT__ )
	#ADD_DEFINITIONS( -Dsnprintf=_snprintf )
	#ADD_DEFINITIONS( -Dusleep=Sleep )
	#ADD_DEFINITIONS( -Dsleep=Sleep )
	#ADD_DEFINITIONS( -D_CRT_SECURE_NO_WARNINGS )
	#ADD_DEFINITIONS( -D_SCL_SECURE_NO_WARNINGS )
	#ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
	#ADD_DEFINITIONS(-D_SCL_SECURE_NO_DEPRECATE)
	#ADD_DEFINITIONS(-D_SECURE_SCL=0)
	#ADD_DEFINITIONS(-D_ITERATOR_DEBUG_LEVEL=0)
	#ADD_DEFINITIONS( -D__NO_PIPES__ )
	#ADD_DEFINITIONS( "/wd4068")
	
	#
	# Enable project grouping when making Visual Studio solution
	# NOTE: This feature is NOT supported in Express editions
	#
	SET_PROPERTY( GLOBAL PROPERTY USE_FOLDERS ON )

ENDIF( )


