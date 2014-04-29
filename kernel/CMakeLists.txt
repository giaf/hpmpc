
IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
    SET( SRC_DOUBLE kernel_dgemm_avx_lib4.c kernel_dpotrf_sse3_lib4.c kernel_dgemv_avx_lib4.c corner_dtrmm_avx_lib4.c corner_dpotrf_sse2_lib4.c kernel_dsymv_avx_lib4.c )
    SET( SRC_SINGLE kernel_sgemm_avx_lib4.c kernel_spotrf_c99_lib4.c kernel_sgemv_c99_lib4.c corner_strmm_sse_lib4.c corner_spotrf_c99_lib4.c kernel_ssymv_c99_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_SSE3" )
    SET( SRC_DOUBLE kernel_dgemm_sse3_lib4.c kernel_dpotrf_sse3_lib4.c kernel_dgemv_sse3_lib4.c corner_dtrmm_sse3_lib4.c corner_dpotrf_sse2_lib4.c kernel_dsymv_c99_lib4.c )
    SET( SRC_SINGLE kernel_sgemm_sse_lib4.c kernel_spotrf_c99_lib4.c kernel_sgemv_c99_lib4.c corner_strmm_sse_lib4.c corner_spotrf_c99_lib4.c kernel_ssymv_c99_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "NEON" )
    SET( SRC_DOUBLE kernel_dgemm_neon_lib4.c kernel_dpotrf_c99_lib4.c kernel_dgemv_c99_lib4.c corner_dtrmm_c99_lib4.c corner_dpotrf_c99_lib4.c kernel_dsymv_c99_lib4.c )
    SET( SRC_SINGLE kernel_sgemm_neon_lib4.c kernel_spotrf_c99_lib4.c kernel_sgemv_c99_lib4.c corner_strmm_c99_lib4.c corner_spotrf_c99_lib4.c kernel_ssymv_c99_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "POWERPC_G2" )
    SET( SRC_DOUBLE kernel_dgemm_ppc_g2_lib4.c kernel_dpotrf_c99_lib4.c kernel_dgemv_c99_lib4.c corner_dtrmm_c99_lib4.c corner_dpotrf_c99_lib4.c kernel_dsymv_c99_lib4.c )
    SET( SRC_SINGLE kernel_sgemm_ppc_g2_lib4.c kernel_spotrf_c99_lib4.c kernel_sgemv_c99_lib4.c corner_strmm_c99_lib4.c corner_spotrf_c99_lib4.c kernel_ssymv_c99_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_4x4" )
    SET( SRC_DOUBLE kernel_dgemm_c99_lib4.c kernel_dpotrf_c99_lib4.c kernel_dgemv_c99_lib4.c corner_dtrmm_c99_lib4.c corner_dpotrf_c99_lib4.c kernel_dsymv_c99_lib4.c )
    SET( SRC_SINGLE kernel_sgemm_c99_lib4.c kernel_spotrf_c99_lib4.c kernel_sgemv_c99_lib4.c corner_strmm_c99_lib4.c corner_spotrf_c99_lib4.c kernel_ssymv_c99_lib4.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "C99_2x2" )
    SET( SRC_DOUBLE kernel_dgemm_c99_lib2.c kernel_dpotrf_c99_lib2.c kernel_dgemv_c99_lib2.c corner_dtrmm_c99_lib2.c corner_dpotrf_c99_lib2.c kernel_dsymv_c99_lib2.c )
    SET( SRC_SINGLE kernel_sgemm_c99_lib2.c kernel_spotrf_c99_lib2.c kernel_sgemv_c99_lib2.c corner_strmm_c99_lib2.c corner_spotrf_c99_lib2.c kernel_ssymv_c99_lib2.c )

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X86_ATOM" )
    SET( SRC_DOUBLE kernel_dgemm_x86_atom_lib2.c kernel_dpotrf_c99_lib2.c kernel_dgemv_c99_lib2.c corner_dtrmm_c99_lib2.c corner_dpotrf_c99_lib2.c kernel_dsymv_c99_lib2.c )
    SET( SRC_SINGLE kernel_sgemm_x86_atom_lib4.c kernel_spotrf_c99_lib4.c kernel_sgemv_x86_atom_lib4.c corner_strmm_c99_lib4.c corner_spotrf_c99_lib4.c kernel_ssymv_c99_lib4.c )

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )

FOREACH( SRC ${SRC_DOUBLE} ${SRC_SINGLE} )
    GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)
	SET( SRC_KERNEL ${SRC_KERNEL} ${ABS_SRC} )
ENDFOREACH()
SET ( SOURCES ${SOURCES} ${SRC_KERNEL} PARENT_SCOPE )
