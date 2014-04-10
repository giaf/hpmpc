void sgemm_ppp_nt_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, float *pC, int sdc, int alg);
void strmm_ppp_lib(int m, int n, int offset, float *pA, int sda, float *pB, int sdb, float *pC, int sdc);
//void dsyrk_ppp_lib(int n, int m, float *pA, int sda, float *pC, int sdc);
void ssyrk_ppp_lib(int n, int m, int k, float *pA, int sda, float *pC, int sdc);
void spotrf_p_lib(int m, int n, float *pC, int sdc);
void spotrf_p_scopy_p_t_lib(int n, int nna, float *pC, int sdc, float *pL, int sdl);
void sgemv_p_n_lib(int n, int m, int offset, float *pA, int sda, float *x, float *y, int alg);
void sgemv_p_t_lib(int n, int m, int offset, float *pA, int sda, float *x, float *y, int alg);
void strmv_p_n_lib(int m, int offset, float *pA, int sda, float *x, float *y);
void strmv_p_t_lib(int m, int offset, float *pA, int sda, float *x, float *y);
void ssymv_p_lib(int m, float *pA, int sda, float *x, float *y);
void strsv_p_n_lib(int n, float *pA, int sda, float *x);
void strsv_p_t_lib(int n, float *pA, int sda, float *x);

