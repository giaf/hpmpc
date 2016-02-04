
HPMPC_ADD_SOURCE_FILES(
	# Double
    kernel_dgemm_avx2_lib4.c kernel_dtrmm_avx2_lib4.c kernel_dtrsm_avx2_lib4.c kernel_dsyrk_avx2_lib4.c kernel_dpotrf_avx2_lib4.c kernel_dgemv_avx2_lib4.c kernel_dtrmv_avx2_lib4.c kernel_dtrsv_avx2_lib4.c kernel_dsymv_avx2_lib4.c kernel_dtran_avx_lib4.c kernel_dttmm_avx2_lib4.c kernel_dtrinv_c99_lib4.c kernel_dgetrf_avx2_lib4.c
	# Single
    kernel_sgemm_avx2_lib8.c kernel_strmm_avx2_lib8.c kernel_sgemm_strsm_avx2_lib8.c kernel_ssryk_spotrf_avx2_lib8.c kernel_sgemv_avx_lib8.c kernel_strmv_avx_lib8.c kernel_strsv_avx_lib8.c kernel_ssymv_avx_lib8.c kernel_stran_avx2_lib8.c
)
