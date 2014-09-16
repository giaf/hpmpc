
IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    SET( SRC_DOUBLE kernel_dgemm_avx_lib4.c kernel_dtrmm_avx_lib4.c kernel_dtrsm_avx_lib4.c kernel_dpotrf_avx_lib4.c kernel_dgemv_avx_lib4.c kernel_dtrmv_avx_lib4.c kernel_dtrsv_avx_lib4.c kernel_dsymv_avx_lib4.c kernel_dtran_avx_lib4.c )
    SET( SRC_SINGLE kernel_sgemm_avx_lib8.c kernel_strmm_avx_lib8.c kernel_strsm_avx_lib8.c kernel_spotrf_avx_lib8.c kernel_sgemv_avx_lib8.c kernel_strmv_avx_lib8.c kernel_strsv_avx_lib8.c kernel_ssymv_avx_lib8.c kernel_stran_avx_lib8.c )

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )

FOREACH( SRC ${SRC_DOUBLE} ${SRC_SINGLE} )
    GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)
	SET( SRC_KERNEL ${SRC_KERNEL} ${ABS_SRC} )
ENDFOREACH()
SET ( SOURCES ${SOURCES} ${SRC_KERNEL} PARENT_SCOPE )
