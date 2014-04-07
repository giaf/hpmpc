#include <math.h>



void corner_dpotrf_dtrsv_dcopy_3x3_c99_lib4(double *A, int sda, int shf, double *L, int sdl)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	const int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);
	const int shfi2 = ((shfi+2)/lda)*lda*(sdl-1);

	double
		a_00, a_10, a_20, a_11, a_21, a_22;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;
	L[0+0*lda+shfi0] = a_00;
	a_00 = 1.0/a_00;
	a_10 = A[1+lda*0] * a_00;
	a_20 = A[2+lda*0] * a_00;
	A[1+lda*0] = a_10;
	A[2+lda*0] = a_20;
	L[0+1*lda+shfi0] = a_10;
	L[0+2*lda+shfi0] = a_20;

	a_11 = sqrt(A[1+lda*1] - a_10*a_10);
	A[1+lda*1] = a_11;
	L[1+1*lda+shfi1] = a_11;
	a_11 = 1.0/a_11;
	a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
	A[2+lda*1] = a_21;
	L[1+2*lda+shfi1] = a_21;
	
	a_22 = sqrt(A[2+lda*2] - a_20*a_20 - a_21*a_21);
	A[2+lda*2] = a_22;
	L[2+2*lda+shfi2] = a_22;

	}



void corner_dpotrf_dtrsv_dcopy_2x2_c99_lib4(double *A, int sda, int shf, double *L, int sdl)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	const int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);

	double
		a_00, a_10, a_11;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;
	L[0+0*lda+shfi0] = a_00;
	a_00 = 1.0/a_00;
	a_10 = A[1+lda*0] * a_00;
	A[1+lda*0] = a_10;
	L[0+1*lda+shfi0] = a_10;

	a_11 = sqrt(A[1+lda*1] - a_10*a_10);
	A[1+lda*1] = a_11;
	L[1+1*lda+shfi1] = a_11;

	}


void corner_dpotrf_dtrsv_dcopy_1x1_c99_lib4(double *A, int sda, int shf, double *L, int sdl)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);

	double
		a_00;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;
	L[0+0*lda+shfi0] = a_00;

	}

