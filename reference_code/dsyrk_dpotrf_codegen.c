#include <math.h>

#define N 4+0*40


int dpotrf_codegen_0(double *A)
	{

	const int n = N+0;
	const int lda = N+0;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_1(double *A)
	{

	const int n = N+1*4;
	const int lda = N+1*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_2(double *A)
	{

	const int n = N+2*4;
	const int lda = N+2*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_3(double *A)
	{

	const int n = N+3*4;
	const int lda = N+3*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_4(double *A)
	{

	const int n = N+4*4;
	const int lda = N+4*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_5(double *A)
	{

	const int n = N+5*4;
	const int lda = N+5*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_6(double *A)
	{

	const int n = N+6*4;
	const int lda = N+6*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_7(double *A)
	{

	const int n = N+7*4;
	const int lda = N+7*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_8(double *A)
	{

	const int n = N+8*4;
	const int lda = N+8*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}



int dpotrf_codegen_9(double *A)
	{

	const int n = N+9*4;
	const int lda = N+9*4;

	double a_jj, temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{

		a_jj = A[jj+lda*jj];

		for(kk=0; kk<jj; kk++)
			{
			a_jj -= A[jj+lda*kk] * A[jj+lda*kk];
			}

		if(a_jj<=0)
			{
			A[jj+lda*jj] = a_jj;
			return jj;
			}
		
		a_jj = sqrt(a_jj);
		A[jj+lda*jj] = a_jj;

		for(kk=0; kk<jj; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj+1; ii<n; ii++)
				{
				A[ii+lda*jj] -= temp * A[ii+lda*kk];
				}
			}

		a_jj = 1.0 / a_jj;

		for(ii=jj+1; ii<n; ii++)
			{
			A[ii+lda*jj] = A[ii+lda*jj] * a_jj;
			}

		}
	
	return 0;
	
	}

