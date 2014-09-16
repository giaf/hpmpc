
IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX2" )
    SET( SRCS aux_d_c99_lib4.c aux_s_c99_lib8.c block_size_x64_avx.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    SET( SRCS aux_d_c99_lib4.c aux_s_c99_lib8.c block_size_x64_avx.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    SET( SRCS aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_x64_sse.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A15" )
    SET( SRCS aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_cortex_a15.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A9" )
    SET( SRCS aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_cortex_a15.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4x4" )
	SET( SRCS aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_c99_4x4.c )

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )

FOREACH( SRC ${SRCS} )
    GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)
	SET( SRC_AUX ${SRC_AUX} ${ABS_SRC} )
ENDFOREACH()
SET ( SOURCES ${SOURCES} ${SRC_AUX} PARENT_SCOPE )
