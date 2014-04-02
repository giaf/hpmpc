void dgemm_ppp_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, int alg);
void dtrmm_ppp_lib(int m, int n, int offset, double *pA, int sda, double *pB, int sdb, double *pC, int sdc);
void dsyrk_ppp_lib(int n, int m, double *pA, int sda, double *pC, int sdc);
void dpotrf_dcopy_lib(int n, int nna, double *pC, int sdc, double *pL, int sdl);



