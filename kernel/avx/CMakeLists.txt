IF ( ${HPMPC_TARGET} STREQUAL "X64_AVX2" )
	HPMPC_ADD_SOURCE_FILES
	(
	# Double
	kernel_dcopy_avx_lib4.c
	# Single
	)

ELSEIF ( ${HPMPC_TARGET} STREQUAL "X64_AVX" )
	HPMPC_ADD_SOURCE_FILES
	(
	# Double
	kernel_dgemm_avx_lib4.c kernel_dtrmm_avx_lib4.c kernel_dtrsm_avx_lib4.c kernel_dsyrk_avx_lib4.c kernel_dpotrf_avx_lib4.c kernel_dgemv_avx_lib4.c kernel_dtrmv_avx_lib4.c kernel_dtrsv_avx_lib4.c kernel_dsymv_avx_lib4.c kernel_dtran_avx_lib4.c kernel_dttmm_avx_lib4.c kernel_dtrinv_avx_lib4.c kernel_dcopy_avx_lib4.c kernel_dgetrf_avx_lib4.c
	# Single
    kernel_sgemm_avx_lib8.c kernel_strmm_avx_lib8.c kernel_sgemm_strsm_avx_lib8.c kernel_ssyrk_spotrf_avx_lib8.c kernel_sgemv_avx_lib8.c kernel_strmv_avx_lib8.c kernel_strsv_avx_lib8.c kernel_ssymv_avx_lib8.c kernel_stran_avx_lib8.c
	)
