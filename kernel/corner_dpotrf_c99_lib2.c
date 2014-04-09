#include <math.h>



void corner_dpotrf_dtrsv_dcopy_1x1_c99_lib2(double *A, int sda, int shf, double *L, int sdl)
	{
	
	const int lda = 2;
	
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

