void dgemm_ppp_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, int alg);
void dtrmm_pup_nn_lib(int n, int m, double *pA, int sda, double *B, int ldb, double *pC, int sdc);
void dsyrk_ppp_lib(int n, int m, double *pA, int sda, double *pC, int sdc);



