/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                                                                                                 *
**************************************************************************************************/

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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
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
			A[ii+lda*jj] *= a_jj;
			}

		}
	
	return 0;
	
	}


void dsyrk_codegen_0(double *A, double *C)
	{

	const int n = N+0*4;
	const int lda = N+0*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_1(double *A, double *C)
	{

	const int n = N+1*4;
	const int lda = N+1*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_2(double *A, double *C)
	{

	const int n = N+2*4;
	const int lda = N+2*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_3(double *A, double *C)
	{

	const int n = N+3*4;
	const int lda = N+3*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_4(double *A, double *C)
	{

	const int n = N+4*4;
	const int lda = N+4*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_5(double *A, double *C)
	{

	const int n = N+5*4;
	const int lda = N+5*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_6(double *A, double *C)
	{

	const int n = N+6*4;
	const int lda = N+6*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_7(double *A, double *C)
	{

	const int n = N+7*4;
	const int lda = N+7*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_8(double *A, double *C)
	{

	const int n = N+8*4;
	const int lda = N+8*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dsyrk_codegen_9(double *A, double *C)
	{

	const int n = N+9*4;
	const int lda = N+9*4;

	double temp;

	int ii, jj, kk;

	for(jj=0; jj<n; jj++)
		{
		for(kk=0; kk<n; kk++)
			{
			temp = A[jj+lda*kk];
			for(ii=jj; ii<n; ii++)
				{
				C[ii+lda*jj] += temp * A[ii+lda*kk];
				}
			}
		}
	
	}



void dcopy_codegen_0(double *A, double *C)
	{

	const int n = N+0*4;
	const int lda = N+0*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_1(double *A, double *C)
	{

	const int n = N+1*4;
	const int lda = N+1*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_2(double *A, double *C)
	{

	const int n = N+2*4;
	const int lda = N+2*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_3(double *A, double *C)
	{

	const int n = N+3*4;
	const int lda = N+3*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_4(double *A, double *C)
	{

	const int n = N+4*4;
	const int lda = N+4*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_5(double *A, double *C)
	{

	const int n = N+5*4;
	const int lda = N+5*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_6(double *A, double *C)
	{

	const int n = N+6*4;
	const int lda = N+6*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_7(double *A, double *C)
	{

	const int n = N+7*4;
	const int lda = N+7*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_8(double *A, double *C)
	{

	const int n = N+8*4;
	const int lda = N+8*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}



void dcopy_codegen_9(double *A, double *C)
	{

	const int n = N+9*4;
	const int lda = N+9*4;

	int ii, jj;

	for(jj=0; jj<n; jj++)
		{
		for(ii=jj; ii<n; ii++)
			{
			C[ii+lda*jj] = A[ii+lda*jj];
			}
		}
	
	}

