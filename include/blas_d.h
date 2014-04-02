void dgemm_ppp_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, int alg);
void dtrmm_ppp_lib(int m, int n, int offset, double *pA, int sda, double *pB, int sdb, double *pC, int sdc);
void dsyrk_ppp_lib(int n, int m, double *pA, int sda, double *pC, int sdc);



