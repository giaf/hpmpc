// kernel
void kernel_sgemm_pp_nt_4x4_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_sgemm_pp_nt_4x3_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_sgemm_pp_nt_4x2_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_sgemm_pp_nt_4x1_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_spotrf_strsv_scopy_4x4_c99_lib4(int kmax, float *A, int sda, int shf, float *L, int sdl);
void kernel_spotrf_strsv_4x4_c99_lib4(int kmax, float *A, int sda);
void kernel_spotrf_strsv_3x3_c99_lib4(int kmax, float *A, int sda);
void kernel_spotrf_strsv_2x2_c99_lib4(int kmax, float *A, int sda);
void kernel_spotrf_strsv_1x1_c99_lib4(int kmax, float *A, int sda);
void kernel_sgemv_t_8_c99_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_t_4_c99_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_t_2_c99_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_t_1_c99_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_n_8_c99_lib4(int kmax, float *A0, float *A1, float *x, float *y, int alg);
void kernel_sgemv_n_4_c99_lib4(int kmax, float *A, float *x, float *y, int alg);
void kernel_sgemv_n_2_c99_lib4(int kmax, float *A, float *x, float *y, int alg);
void kernel_sgemv_n_1_c99_lib4(int kmax, float *A, float *x, float *y, int alg);
//// corner
void corner_strmm_pp_nt_4x3_c99_lib4(float *A, float *B, float *C, int ldc);
void corner_strmm_pp_nt_4x2_c99_lib4(float *A, float *B, float *C, int ldc);
void corner_strmm_pp_nt_4x1_c99_lib4(float *A, float *B, float *C, int ldc);
void corner_spotrf_strsv_scopy_3x3_c99_lib4(float *A, int sda, int shf, float *L, int sdl);
void corner_spotrf_strsv_scopy_2x2_c99_lib4(float *A, int sda, int shf, float *L, int sdl);
void corner_spotrf_strsv_scopy_1x1_c99_lib4(float *A, int sda, int shf, float *L, int sdl);

