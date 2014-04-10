void s_zeros(float **pA, int row, int col);
void s_zeros_align(float **pA, int row, int col);
void s_eye(float **pA, int row);
void s_copy_mat(int row, int col, float *A, int lda, float *B, int ldb);
void s_copy_pmat(int row, int col, int bs, float *A, int sda, float *B, int sdb);
void s_copy_pmat_lo(int row, int bs, float *A, int sda, float *B, int sdb);
void s_align_pmat(int row, int col, int offset, int bs, float *A, int sda, float *B, int sdb);
void s_cvt_mat2pmat(int row, int col, int offset, int bs, float *A, int lda, float *B, int sdb);
void s_cvt_pmat2mat(int row, int col, int offset, int bs, float *pA, int sda, float *A, int lda);
void s_print_mat(int row, int col, float *A, int lda);
void s_print_pmat(int row, int col, int bs, float *A, int sda);

