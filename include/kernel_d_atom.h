// kernel
void kernel_dgemm_pp_nt_2x2_atom_lib2(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dgemm_pp_nt_2x1_c99_lib2(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dpotrf_dtrsv_dcopy_2x2_c99_lib2(int kmax, double *A, int sda, int shf, double *L, int sdl);
void kernel_dpotrf_dtrsv_2x2_c99_lib2(int kmax, double *A, int sda);
void kernel_dpotrf_dtrsv_1x1_c99_lib2(int kmax, double *A, int sda);
void kernel_dgemv_t_2_c99_lib2(int kmax, int kna, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_t_1_c99_lib2(int kmax, int kna, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_n_2_c99_lib2(int kmax, double *A, double *x, double *y, int alg);
void kernel_dgemv_n_1_c99_lib2(int kmax, double *A, double *x, double *y, int alg);
//// corner
void corner_dtrmm_pp_nt_2x1_c99_lib2(double *A, double *B, double *C, int ldc);
void corner_dpotrf_dtrsv_dcopy_1x1_c99_lib2(double *A, int sda, int shf, double *L, int sdl);

