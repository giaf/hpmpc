// kernel
void kernel_sgemm_pp_nt_2x2_c99_lib2(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_sgemm_pp_nt_2x1_c99_lib2(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_spotrf_strsv_scopy_2x2_c99_lib2(int kmax, float *A, int sda, int shf, float *L, int sdl);
void kernel_spotrf_strsv_2x2_c99_lib2(int kmax, float *A, int sda);
void kernel_spotrf_strsv_1x1_c99_lib2(int kmax, float *A, int sda);
void kernel_sgemv_t_2_c99_lib2(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_t_1_c99_lib2(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_n_2_c99_lib2(int kmax, float *A, float *x, float *y, int alg);
void kernel_sgemv_n_1_c99_lib2(int kmax, float *A, float *x, float *y, int alg);
//// corner
void corner_strmm_pp_nt_2x1_c99_lib2(float *A, float *B, float *C, int ldc);
void corner_spotrf_dtrsv_dcopy_1x1_c99_lib2(float *A, int sda, int shf, float *L, int sdl);

