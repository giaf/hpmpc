HPMPC_ADD_SOURCE_FILES(
	# Double precision
    kernel_dgemm_c99_lib4.c kernel_dtrmm_c99_lib4.c kernel_dgemm_dtrsm_c99_lib4.c kernel_dsyrk_dpotrf_c99_lib4.c kernel_dgemv_c99_lib4.c kernel_dtrmv_c99_lib4.c kernel_dtrsv_c99_lib4.c kernel_dsymv_c99_lib4.c kernel_dtran_c99_lib4.c
	# Single precision
    kernel_sgemm_c99_lib4.c kernel_strmm_c99_lib4.c kernel_strsm_c99_lib4.c kernel_spotrf_c99_lib4.c kernel_sgemv_c99_lib4.c kernel_strmv_c99_lib4.c kernel_strsv_c99_lib4.c kernel_ssymv_c99_lib4.c kernel_stran_c99_lib4.c
)
