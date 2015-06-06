
IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX2" )
    HPMPC_ADD_SOURCE_FILES( d_aux_lib4.c s_aux_lib8.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    HPMPC_ADD_SOURCE_FILES( d_aux_lib4.c s_aux_lib8.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    HPMPC_ADD_SOURCE_FILES( d_aux_lib4.c s_aux_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A15" )
    HPMPC_ADD_SOURCE_FILES( d_aux_lib4.c s_aux_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A9" )
    HPMPC_ADD_SOURCE_FILES( d_aux_lib4.c s_aux_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A7" )
    HPMPC_ADD_SOURCE_FILES( d_aux_lib4.c s_aux_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4X4" )
	HPMPC_ADD_SOURCE_FILES( d_aux_lib4.c s_aux_lib4.c )

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )
