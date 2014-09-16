
IF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    SET( SRC_DOUBLE kernel_dgemm_sse3_lib4.c kernel_dtrmm_sse3_lib4.c kernel_dtrsm_sse3_lib4.c kernel_dpotrf_sse3_lib4.c kernel_dgemv_c99_lib4.c kernel_dtrmv_c99_lib4.c kernel_dtrsv_c99_lib4.c kernel_dsymv_c99_lib4.c kernel_dtran_c99_lib4.c )
    SET( SRC_SINGLE kernel_sgemm_sse_lib4.c kernel_strmm_sse_lib4.c kernel_strsm_sse_lib4.c kernel_spotrf_sse_lib4.c kernel_sgemv_c99_lib4.c kernel_strmv_c99_lib4.c kernel_strsv_c99_lib4.c kernel_ssymv_c99_lib4.c kernel_stran_c99_lib4.c )

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )

FOREACH( SRC ${SRC_DOUBLE} ${SRC_SINGLE} )
    GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)
	SET( SRC_KERNEL ${SRC_KERNEL} ${ABS_SRC} )
ENDFOREACH()
SET ( SOURCES ${SOURCES} ${SRC_KERNEL} PARENT_SCOPE )
