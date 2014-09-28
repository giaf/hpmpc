
IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX2" )
    HPMPC_ADD_SOURCE_FILES( aux_d_c99_lib4.c aux_s_c99_lib8.c block_size_x64_avx.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    HPMPC_ADD_SOURCE_FILES( aux_d_c99_lib4.c aux_s_c99_lib8.c block_size_x64_avx.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    HPMPC_ADD_SOURCE_FILES( aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_x64_sse.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A15" )
    HPMPC_ADD_SOURCE_FILES( aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_cortex_a15.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A9" )
    HPMPC_ADD_SOURCE_FILES( aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_cortex_a15.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A7" )
    HPMPC_ADD_SOURCE_FILES( aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_cortex_a15.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4x4" )
	HPMPC_ADD_SOURCE_FILES( aux_d_c99_lib4.c aux_s_c99_lib4.c block_size_c99_4x4.c )

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )
