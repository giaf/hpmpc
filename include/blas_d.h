void dgemm_ppp_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, int alg);
void dtrmm_ppp_lib(int m, int n, int offset, double *pA, int sda, double *pB, int sdb, double *pC, int sdc);
//void dsyrk_ppp_lib(int n, int m, double *pA, int sda, double *pC, int sdc);
void dsyrk_ppp_lib(int n, int m, int k, double *pA, int sda, double *pC, int sdc);
void dpotrf_p_dcopy_p_t_lib(int n, int nna, double *pC, int sdc, double *pL, int sdl);
void dgemv_p_n_lib(int n, int m, int offset, double *pA, int sda, double *x, double *y, int alg);
void dgemv_p_t_lib(int n, int m, int offset, double *pA, int sda, double *x, double *y, int alg);
void dtrmv_p_n_lib(int m, int offset, double *pA, int sda, double *x, double *y);
void dtrmv_p_t_lib(int m, int offset, double *pA, int sda, double *x, double *y);
void dsymv_p_lib(int m, double *pA, int sda, double *x, double *y);
void dtrsv_p_n_lib(int n, double *pA, int sda, double *x);
void dtrsv_p_t_lib(int n, double *pA, int sda, double *x);

