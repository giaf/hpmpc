/* ----------------------------------------------------------------------------------------
**
** Utilities functions. 
**
** Temporary version, do not diffuse.
**
** Author: Gianluca Frison, giaf@imm.dtu.dk
**
-----------------------------------------------------------------------------------------*/
	
/* Copies the entries of the matrix A into the matrix B (of the same size).
** Arguments:
** trans - character: if 'n', the matrix A is copied into B; if 't' the transposed of
**         the matrix A in copied into B
** row - integer that specifies the number of rows of the matrices A and B
** col - integer that specifies the number of columns of the matrices A and B
** ptrA - pointer to double, pointing to the first element of the A matrix
** lda - integer that specifies the leading dimension of the array A
** ptrB - pointer to double, pointing to the first element fo the B matrix
** ldb - integer that specifies the leading dimension of the array B
*/

void dgemm_nn_3l(int m, int n, int k, double *A, int lda , double *B, int ldb, double *C, int ldc);
void daxpy_3l(int n, double da, double *dx, double *dy);
void dscal_3l(int n, double da, double *dx);

/* copies a matrix into another matrix */
void dmcopy(int row, int col, double *ptrA, int lda, double *ptrB, int ldb);

/* solution of a system of linear equations */
void dgesv_3l(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb, int *info);

/* matrix exponential */
void expm(int row, double *A);
