void d_zeros(double **pA, int row, int col);
void d_zeros_align(double **pA, int row, int col);
void d_eye(double **pA, int row);
void d_copy_mat(int row, int col, double *A, int lda, double *B, int ldb);
void d_copy_pmat(int row, int col, int bs, double *A, int sda, double *B, int sdb);
void d_copy_pmat_lo(int row, int bs, double *A, int sda, double *B, int sdb);
void d_align_pmat(int row, int col, int offset, int bs, double *A, int sda, double *B, int sdb);
void d_cvt_mat2pmat(int row, int col, int offset, int bs, double *A, int lda, double *pA, int sda);
void d_cvt_pmat2mat(int row, int col, int offset, int bs, double *pA, int sda, double *A, int lda);
void d_print_mat(int row, int col, double *A, int lda);
void d_print_pmat(int row, int col, int bs, double *A, int sda);

