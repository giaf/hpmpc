
IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    SET( SRC_DOUBLE code_generator_blas_d_lib4.c )
    SET( SRC_SINGLE code_generator_blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    SET( SRC_DOUBLE code_generator_blas_d_lib4.c )
    SET( SRC_SINGLE code_generator_blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "NEON" )
    SET( SRC_DOUBLE code_generator_blas_d_lib4.c )
    SET( SRC_SINGLE code_generator_blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "POWERPC_G2" )
    SET( SRC_DOUBLE code_generator_blas_d_lib4.c )
    SET( SRC_SINGLE code_generator_blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4x4" )
    SET( SRC_DOUBLE code_generator_blas_d_lib4.c )
    SET( SRC_SINGLE code_generator_blas_s_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_2x2" )
    SET( SRC_DOUBLE code_generator_blas_d_lib2.c )
    SET( SRC_SINGLE code_generator_blas_s_lib2.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X86_ATOM" )
    SET( SRC_DOUBLE code_generator_blas_d_lib2.c )
    SET( SRC_SINGLE code_generator_blas_s_lib4.c )

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )

FOREACH( SRC ${SRC_DOUBLE} ${SRC_SINGLE} )
    GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)
	SET( SRC_CODEGEN ${SRC_CODEGEN} ${ABS_SRC} )
ENDFOREACH()
SET ( SOURCES ${SOURCES} ${SRC_CODEGEN} PARENT_SCOPE )

ADD_EXECUTABLE( code_generator_dricposv code_generator_dricposv.c )
TARGET_LINK_LIBRARIES( code_generator_dricposv hpmpc )

ADD_EXECUTABLE( code_generator_sricposv code_generator_sricposv.c )
TARGET_LINK_LIBRARIES( code_generator_sricposv hpmpc )
