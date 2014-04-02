void kernel_dgemm_pp_nt_8x4_avx_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg);
void kernel_dgemm_pp_nt_8x2_avx_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg);
void kernel_dgemm_pp_nt_8x1_avx_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg);
void kernel_dgemm_pp_nt_4x4_avx_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dgemm_pp_nt_4x2_avx_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dgemm_pp_nt_4x1_avx_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(int kmax, int kna_dummy, double *A, int sda, int shf, double *L, int sdl);
void kernel_dpotrf_dtrsv_4x4_sse_lib4(int kmax, int kna_dummy, double *A, int sda);


