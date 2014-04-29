
IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    SET( SRCS blas_d_lib4.c blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    SET( SRCS blas_d_lib4.c blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "NEON" )
    SET( SRCS blas_d_lib4.c blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "POWERPC_G2" )
    SET( SRCS blas_d_lib4.c blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4x4" )
	SET( SRCS blas_d_lib4.c blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_2x2" )
    SET( SRCS blas_d_lib2.c blas_s_lib2.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X86_ATOM" )
    SET( SRCS blas_d_lib2.c blas_s_lib4.c  )

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )

FOREACH( SRC ${SRCS} )
    GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)
	SET( SRC_BLAS ${SRC_BLAS} ${ABS_SRC} )
ENDFOREACH()
SET ( SOURCES ${SOURCES} ${SRC_BLAS} PARENT_SCOPE )
