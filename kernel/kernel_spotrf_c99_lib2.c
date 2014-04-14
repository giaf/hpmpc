#include <math.h>



void kernel_spotrf_strsv_scopy_2x2_c99_lib2(int kmax, float *A, int sda, int shf, float *L, int sdl)
	{
	
	const int lda = 2;
	
	L += shf*(lda+1);
	int shfi = shf + lda - 2;
	int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);

	float
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
	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_11 = 1.0/a_11;
	
	int k, kk, kend;
	
	float
		a_0, a_1,
		a_0a, a_1a,
		a_0b, a_1b,
		*AA, *LL;
	
	AA = A + 2;
	LL = L + 2*lda;
	k = 0;
	for(; k<kmax-1; k+=2)
		{

		AA += lda*(sda-1);
		
		a_0a = AA[0+lda*0] * a_00;
		a_0b = AA[1+lda*0] * a_00;
		AA[0+lda*0] = a_0a;
		AA[1+lda*0] = a_0b;
		LL[0+shfi0+0*lda] = a_0a;
		LL[0+shfi0+1*lda] = a_0b;
	
		a_1a = (AA[0+lda*1] - a_0a * a_10) * a_11;
		a_1b = (AA[1+lda*1] - a_0b * a_10) * a_11;
		AA[0+lda*1] = a_1a;
		AA[1+lda*1] = a_1b;
		LL[1+shfi1+0*lda] = a_1a;
		LL[1+shfi1+1*lda] = a_1b;
	
		AA += 2;
		LL += 2*lda;

		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		a_0 = AA[lda*0] * a_00;
		AA[lda*0] = a_0;
		LL[0+shfi0] = a_0;
	
		a_1 = (AA[lda*1] - a_0 * a_10) * a_11;
		AA[lda*1] = a_1;
		LL[1+shfi1] = a_1;
	
		AA += 1;
		LL += lda;
		}
	
	}



void kernel_spotrf_strsv_2x2_c99_lib2(int kmax, float *A, int sda)
	{
	
	const int lda = 2;
	
	float
		a_00, a_10, a_11;
	
	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;
	a_00 = 1.0/a_00;
	a_10 = A[1+lda*0] * a_00;
	A[1+lda*0] = a_10;
	
	a_11 = sqrt(A[1+lda*1] - a_10*a_10);
	A[1+lda*1] = a_11;
	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_11 = 1.0/a_11;
	
	int k, kk, kend;
	
	float
		a_0, a_1,
		a_0a, a_1a,
		a_0b, a_1b,
		*AA;
	
	AA = A + 2;
	k = 0;
	for(; k<kmax-1; k+=2)
		{

		AA += lda*(sda-1);
		
		a_0a = AA[0+lda*0] * a_00;
		a_0b = AA[1+lda*0] * a_00;
		AA[0+lda*0] = a_0a;
		AA[1+lda*0] = a_0b;
	
		a_1a = (AA[0+lda*1] - a_0a * a_10) * a_11;
		a_1b = (AA[1+lda*1] - a_0b * a_10) * a_11;
		AA[0+lda*1] = a_1a;
		AA[1+lda*1] = a_1b;
	
		AA += 2;

		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		a_0 = AA[lda*0] * a_00;
		AA[lda*0] = a_0;
	
		a_1 = (AA[lda*1] - a_0 * a_10) * a_11;
		AA[lda*1] = a_1;
	
		AA += 1;
		}
	
	}


void kernel_spotrf_strsv_1x1_c99_lib2(int kmax, float *A, int sda)
	{
	
	const int lda = 2;
	
	float
		a_00;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;


	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_00 = 1.0/a_00;

	int k; //, kna;
	
	float
		b_00,
		b_10,
		*AA;
	
	AA = A + 1;
	k = 0;
	
/*	kna = 1;*/
	b_00 = AA[0+lda*0];

	b_00 *= a_00;
	AA[0+lda*0] = b_00;

	AA += 1;
	k++;

	for(; k<kmax-1; k+=2)
		{

		AA += lda*(sda-1);

		b_00 = AA[0+lda*0];
		b_10 = AA[1+lda*0];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[0+lda*0] = b_00;
		AA[1+lda*0] = b_10;

		AA += 2;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00 = AA[0+lda*0];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;

		AA += 1;
		}
	
	}

