
HPMPC_ADD_SOURCE_FILES(
    # Double
	kernel_dgemm_sse3_lib4.c kernel_dtrmm_sse3_lib4.c kernel_dtrsm_sse3_lib4.c kernel_dpotrf_sse3_lib4.c kernel_dgemv_c99_lib4.c kernel_dtrmv_c99_lib4.c kernel_dtrsv_c99_lib4.c kernel_dsymv_c99_lib4.c kernel_dtran_c99_lib4.c
	# Single
    kernel_sgemm_sse_lib4.c kernel_strmm_sse_lib4.c kernel_strsm_sse_lib4.c kernel_spotrf_sse_lib4.c kernel_sgemv_c99_lib4.c kernel_strmv_c99_lib4.c kernel_strsv_c99_lib4.c kernel_ssymv_c99_lib4.c kernel_stran_c99_lib4.c
)
