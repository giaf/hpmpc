/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     *
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

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
/*#include <emmintrin.h>  // SSE2*/
#include <pmmintrin.h>  // SSE3
//#include <smmintrin.h>  // SSE4
//#include <immintrin.h>  // AVX



// it moves vertically across blocks
void kernel_sgemv_t_8_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{

	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;
	int ka = kmax-kna; // number from aligned positon
	
	__m128
		a_03_0, a_03_1, a_03_2, a_03_3,
		x_03,
		y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7;
	
	y_0 = _mm_setzero_ps();	
	y_1 = _mm_setzero_ps();	
	y_2 = _mm_setzero_ps();	
	y_3 = _mm_setzero_ps();	
	y_4 = _mm_setzero_ps();	
	y_5 = _mm_setzero_ps();	
	y_6 = _mm_setzero_ps();	
	y_7 = _mm_setzero_ps();	

	k = 0;
	if(kna>0)
		{
		for(; k<kna; k++)
			{
		
			x_03 = _mm_load_ss( &x[0] );

			a_03_0 = _mm_load_ss( &A[0+lda*0] );
			a_03_1 = _mm_load_ss( &A[0+lda*1] );
			a_03_2 = _mm_load_ss( &A[0+lda*2] );
			a_03_3 = _mm_load_ss( &A[0+lda*3] );
		
			a_03_0 = _mm_mul_ss( a_03_0, x_03 );
			y_0    = _mm_add_ss( y_0, a_03_0 );
			a_03_1 = _mm_mul_ss( a_03_1, x_03 );
			y_1    = _mm_add_ss( y_1, a_03_1 );
			a_03_2 = _mm_mul_ss( a_03_2, x_03 );
			y_2    = _mm_add_ss( y_2, a_03_2 );
			a_03_3 = _mm_mul_ss( a_03_3, x_03 );
			y_3    = _mm_add_ss( y_3, a_03_3 );
		
			a_03_0 = _mm_load_ss( &A[0+lda*4] );
			a_03_1 = _mm_load_ss( &A[0+lda*5] );
			a_03_2 = _mm_load_ss( &A[0+lda*6] );
			a_03_3 = _mm_load_ss( &A[0+lda*7] );
		
			a_03_0 = _mm_mul_ss( a_03_0, x_03 );
			y_4    = _mm_add_ss( y_4, a_03_0 );
			a_03_1 = _mm_mul_ss( a_03_1, x_03 );
			y_5    = _mm_add_ss( y_5, a_03_1 );
			a_03_2 = _mm_mul_ss( a_03_2, x_03 );
			y_6    = _mm_add_ss( y_6, a_03_2 );
			a_03_3 = _mm_mul_ss( a_03_3, x_03 );
			y_7    = _mm_add_ss( y_7, a_03_3 );
		
			x += 1;
			A += 1;

			}

		A += (sda-1)*lda;
		}

	k = 0;
	for(; k<ka-7; k+=8)
		{
		
		x_03 = _mm_loadu_ps( &x[0] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );
		a_03_2 = _mm_load_ps( &A[0+lda*2] );
		a_03_3 = _mm_load_ps( &A[0+lda*3] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_0    = _mm_add_ps( y_0, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_1    = _mm_add_ps( y_1, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_2    = _mm_add_ps( y_2, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_3    = _mm_add_ps( y_3, a_03_3 );
		
		a_03_0 = _mm_load_ps( &A[0+lda*4] );
		a_03_1 = _mm_load_ps( &A[0+lda*5] );
		a_03_2 = _mm_load_ps( &A[0+lda*6] );
		a_03_3 = _mm_load_ps( &A[0+lda*7] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_4    = _mm_add_ps( y_4, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_5    = _mm_add_ps( y_5, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_6    = _mm_add_ps( y_6, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_7    = _mm_add_ps( y_7, a_03_3 );
		
		x += 4;
		A += 4;

		A += (sda-1)*lda;

		x_03 = _mm_loadu_ps( &x[0] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );
		a_03_2 = _mm_load_ps( &A[0+lda*2] );
		a_03_3 = _mm_load_ps( &A[0+lda*3] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_0    = _mm_add_ps( y_0, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_1    = _mm_add_ps( y_1, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_2    = _mm_add_ps( y_2, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_3    = _mm_add_ps( y_3, a_03_3 );
		
		a_03_0 = _mm_load_ps( &A[0+lda*4] );
		a_03_1 = _mm_load_ps( &A[0+lda*5] );
		a_03_2 = _mm_load_ps( &A[0+lda*6] );
		a_03_3 = _mm_load_ps( &A[0+lda*7] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_4    = _mm_add_ps( y_4, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_5    = _mm_add_ps( y_5, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_6    = _mm_add_ps( y_6, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_7    = _mm_add_ps( y_7, a_03_3 );
		
		x += 4;
		A += 4;

		A += (sda-1)*lda;

		}
	for(; k<ka-3; k+=4)
		{
		
		x_03 = _mm_loadu_ps( &x[0] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );
		a_03_2 = _mm_load_ps( &A[0+lda*2] );
		a_03_3 = _mm_load_ps( &A[0+lda*3] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_0    = _mm_add_ps( y_0, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_1    = _mm_add_ps( y_1, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_2    = _mm_add_ps( y_2, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_3    = _mm_add_ps( y_3, a_03_3 );
		
		a_03_0 = _mm_load_ps( &A[0+lda*4] );
		a_03_1 = _mm_load_ps( &A[0+lda*5] );
		a_03_2 = _mm_load_ps( &A[0+lda*6] );
		a_03_3 = _mm_load_ps( &A[0+lda*7] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_4    = _mm_add_ps( y_4, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_5    = _mm_add_ps( y_5, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_6    = _mm_add_ps( y_6, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_7    = _mm_add_ps( y_7, a_03_3 );
		
		x += 4;
		A += 4;

		A += (sda-1)*lda;

		}
	for(; k<ka; k++)
		{
		
		x_03 = _mm_load_ss( &x[0] );

		a_03_0 = _mm_load_ss( &A[0+lda*0] );
		a_03_1 = _mm_load_ss( &A[0+lda*1] );
		a_03_2 = _mm_load_ss( &A[0+lda*2] );
		a_03_3 = _mm_load_ss( &A[0+lda*3] );
	
		a_03_0 = _mm_mul_ss( a_03_0, x_03 );
		y_0    = _mm_add_ss( y_0, a_03_0 );
		a_03_1 = _mm_mul_ss( a_03_1, x_03 );
		y_1    = _mm_add_ss( y_1, a_03_1 );
		a_03_2 = _mm_mul_ss( a_03_2, x_03 );
		y_2    = _mm_add_ss( y_2, a_03_2 );
		a_03_3 = _mm_mul_ss( a_03_3, x_03 );
		y_3    = _mm_add_ss( y_3, a_03_3 );

		a_03_0 = _mm_load_ss( &A[0+lda*4] );
		a_03_1 = _mm_load_ss( &A[0+lda*5] );
		a_03_2 = _mm_load_ss( &A[0+lda*6] );
		a_03_3 = _mm_load_ss( &A[0+lda*7] );
	
		a_03_0 = _mm_mul_ss( a_03_0, x_03 );
		y_4    = _mm_add_ss( y_4, a_03_0 );
		a_03_1 = _mm_mul_ss( a_03_1, x_03 );
		y_5    = _mm_add_ss( y_5, a_03_1 );
		a_03_2 = _mm_mul_ss( a_03_2, x_03 );
		y_6    = _mm_add_ss( y_6, a_03_2 );
		a_03_3 = _mm_mul_ss( a_03_3, x_03 );
		y_7    = _mm_add_ss( y_7, a_03_3 );

		x += 1;
		A += 1;
		
		}

	__m128
		y_03, y_47;

	y_0 = _mm_hadd_ps(y_0, y_1);
	y_2 = _mm_hadd_ps(y_2, y_3);
	y_0 = _mm_hadd_ps(y_0, y_2);

	y_4 = _mm_hadd_ps(y_4, y_5);
	y_6 = _mm_hadd_ps(y_6, y_7);
	y_4 = _mm_hadd_ps(y_4, y_6);

	if(alg==0)
		{
		_mm_storeu_ps(&y[0], y_0);
		_mm_storeu_ps(&y[4], y_4);
		}
	else if(alg==1)
		{
		y_03 = _mm_loadu_ps( &y[0] );
		y_47 = _mm_loadu_ps( &y[4] );

		y_03 = _mm_add_ps(y_03, y_0);
		y_47 = _mm_add_ps(y_47, y_4);
	
		_mm_storeu_ps(&y[0], y_03);
		_mm_storeu_ps(&y[4], y_47);
		}
	else // alg==-1
		{
		y_03 = _mm_loadu_ps( &y[0] );
		y_47 = _mm_loadu_ps( &y[4] );

		y_03 = _mm_sub_ps(y_03, y_0);
		y_47 = _mm_sub_ps(y_47, y_4);
	
		_mm_storeu_ps(&y[0], y_03);
		_mm_storeu_ps(&y[4], y_47);
		}

	}



// it moves vertically across blocks
void kernel_sgemv_t_4_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{

	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;
	int ka = kmax-kna; // number from aligned positon
	
	__m128
		a_03_0, a_03_1, a_03_2, a_03_3,
		x_03,
		y_0, y_1, y_2, y_3;
	
	y_0 = _mm_setzero_ps();	
	y_1 = _mm_setzero_ps();	
	y_2 = _mm_setzero_ps();	
	y_3 = _mm_setzero_ps();	

	k = 0;
	if(kna>0)
		{
		for(; k<kna; k++)
			{
		
			x_03 = _mm_load_ss( &x[0] );

			a_03_0 = _mm_load_ss( &A[0+lda*0] );
			a_03_1 = _mm_load_ss( &A[0+lda*1] );
			a_03_2 = _mm_load_ss( &A[0+lda*2] );
			a_03_3 = _mm_load_ss( &A[0+lda*3] );
		
			a_03_0 = _mm_mul_ss( a_03_0, x_03 );
			y_0    = _mm_add_ss( y_0, a_03_0 );
			a_03_1 = _mm_mul_ss( a_03_1, x_03 );
			y_1    = _mm_add_ss( y_1, a_03_1 );
			a_03_2 = _mm_mul_ss( a_03_2, x_03 );
			y_2    = _mm_add_ss( y_2, a_03_2 );
			a_03_3 = _mm_mul_ss( a_03_3, x_03 );
			y_3    = _mm_add_ss( y_3, a_03_3 );
		
			x += 1;
			A += 1;

			}

		A += (sda-1)*lda;
		}

	k = 0;
	for(; k<ka-7; k+=8)
		{
		
		x_03 = _mm_loadu_ps( &x[0] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );
		a_03_2 = _mm_load_ps( &A[0+lda*2] );
		a_03_3 = _mm_load_ps( &A[0+lda*3] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_0    = _mm_add_ps( y_0, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_1    = _mm_add_ps( y_1, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_2    = _mm_add_ps( y_2, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_3    = _mm_add_ps( y_3, a_03_3 );
		
		x += 4;
		A += 4;

		A += (sda-1)*lda;

		x_03 = _mm_loadu_ps( &x[0] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );
		a_03_2 = _mm_load_ps( &A[0+lda*2] );
		a_03_3 = _mm_load_ps( &A[0+lda*3] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_0    = _mm_add_ps( y_0, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_1    = _mm_add_ps( y_1, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_2    = _mm_add_ps( y_2, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_3    = _mm_add_ps( y_3, a_03_3 );
		
		x += 4;
		A += 4;

		A += (sda-1)*lda;

		}
	for(; k<ka-3; k+=4)
		{
		
		x_03 = _mm_loadu_ps( &x[0] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );
		a_03_2 = _mm_load_ps( &A[0+lda*2] );
		a_03_3 = _mm_load_ps( &A[0+lda*3] );
		
		a_03_0 = _mm_mul_ps( a_03_0, x_03 );
		y_0    = _mm_add_ps( y_0, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_03 );
		y_1    = _mm_add_ps( y_1, a_03_1 );
		a_03_2 = _mm_mul_ps( a_03_2, x_03 );
		y_2    = _mm_add_ps( y_2, a_03_2 );
		a_03_3 = _mm_mul_ps( a_03_3, x_03 );
		y_3    = _mm_add_ps( y_3, a_03_3 );
		
		x += 4;
		A += 4;

		A += (sda-1)*lda;

		}
	for(; k<ka; k++)
		{
		
		x_03 = _mm_load_ss( &x[0] );

		a_03_0 = _mm_load_ss( &A[0+lda*0] );
		a_03_1 = _mm_load_ss( &A[0+lda*1] );
		a_03_2 = _mm_load_ss( &A[0+lda*2] );
		a_03_3 = _mm_load_ss( &A[0+lda*3] );
	
		a_03_0 = _mm_mul_ss( a_03_0, x_03 );
		y_0    = _mm_add_ss( y_0, a_03_0 );
		a_03_1 = _mm_mul_ss( a_03_1, x_03 );
		y_1    = _mm_add_ss( y_1, a_03_1 );
		a_03_2 = _mm_mul_ss( a_03_2, x_03 );
		y_2    = _mm_add_ss( y_2, a_03_2 );
		a_03_3 = _mm_mul_ss( a_03_3, x_03 );
		y_3    = _mm_add_ss( y_3, a_03_3 );

		x += 1;
		A += 1;
		
		}

	__m128
		y_03;

	y_0 = _mm_hadd_ps(y_0, y_1);
	y_2 = _mm_hadd_ps(y_2, y_3);
	y_0 = _mm_hadd_ps(y_0, y_2);

	if(alg==0)
		{
		_mm_storeu_ps(&y[0], y_0);
		}
	else if(alg==1)
		{
		y_03 = _mm_loadu_ps( &y[0] );

		y_03 = _mm_add_ps(y_03, y_0);
	
		_mm_storeu_ps(&y[0], y_03);
		}
	else // alg==-1
		{
		y_03 = _mm_loadu_ps( &y[0] );

		y_03 = _mm_sub_ps(y_03, y_0);
	
		_mm_storeu_ps(&y[0], y_03);
		}

	}



// it moves vertically across blocks
void kernel_sgemv_t_2_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	const int bs  = 4;
	
	int
		k, ka=kmax-kna;
	
	float
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0;
	
	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0 = x[0];
		
			y_0 += A[0+lda*0] * x_0;
			y_1 += A[0+lda*1] * x_0;
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k=0;
	for(; k<ka-bs+1; k+=bs)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];
		
		y_0 += A[0+lda*0] * x_0;
		y_1 += A[0+lda*1] * x_0;

		y_0 += A[1+lda*0] * x_1;
		y_1 += A[1+lda*1] * x_1;
		
		y_0 += A[2+lda*0] * x_2;
		y_1 += A[2+lda*1] * x_2;

		y_0 += A[3+lda*0] * x_3;
		y_1 += A[3+lda*1] * x_3;
		
		A += sda*bs;
		x += 4;

		}
	
	for(; k<ka; k++)
		{
		
		x_0 = x[0];
	
		y_0 += A[0+lda*0] * x_0;
		y_1 += A[0+lda*1] * x_0;
	
		A += 1;
		x += 1;
		
		}

	if(alg==0)
		{
		y[0] = y_0;
		y[1] = y_1;
		}
	else if(alg==1)
		{
		y[0] += y_0;
		y[1] += y_1;
		}
	else // alg==-1
		{
		y[0] -= y_0;
		y[1] -= y_1;
		}
	
	}



// it moves vertically across blocks
void kernel_sgemv_t_1_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	const int bs  = 4;
	
	int
		k, ka=kmax-kna;
	
	float
		x_0, x_1, x_2, x_3,
		y_0=0;
	
	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0 = x[0];
		
			y_0 += A[0+lda*0] * x_0;
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k=0;
	for(; k<ka-bs+1; k+=bs)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];
		
		y_0 += A[0+lda*0] * x_0;
		y_0 += A[1+lda*0] * x_1;
		y_0 += A[2+lda*0] * x_2;
		y_0 += A[3+lda*0] * x_3;
		
		A += sda*bs;
		x += 4;

		}
	
	for(; k<ka; k++)
		{
		
		x_0 = x[0];
	
		y_0 += A[0+lda*0] * x_0;
	
		A += 1;
		x += 1;
		
		}

	if(alg==0)
		{
		y[0] = y_0;
		}
	else if(alg==1)
		{
		y[0] += y_0;
		}
	else // alg==-1
		{
		y[0] -= y_0;
		}
	
	}



// it moves horizontally inside a block
// it moves horizontally inside a block
void kernel_sgemv_n_8_lib4(int kmax, float *A0, float *A1, float *x, float *y, int alg)
	{

/*printf("\nciaoc\n");*/
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128
		a_03_0, a_03_1, a_47_0, a_47_1,
		x_0, x_1,
		y_03, y_47, Y_03, Y_47, z_03, z_47;
	
	y_03 = _mm_setzero_ps();	
	y_47 = _mm_setzero_ps();	
	Y_03 = _mm_setzero_ps();	
	Y_47 = _mm_setzero_ps();	

	k=0;
	for(; k<kmax-7; k+=8)
		{

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_03_0 = _mm_load_ps( &A0[0+lda*0] );
		a_47_0 = _mm_load_ps( &A1[0+lda*0] );
		a_03_1 = _mm_load_ps( &A0[0+lda*1] );
		a_47_1 = _mm_load_ps( &A1[0+lda*1] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_47_0 = _mm_mul_ps( a_47_0, x_0 );
		y_47   = _mm_add_ps( y_47, a_47_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		a_47_1 = _mm_mul_ps( a_47_1, x_1 );
		Y_47   = _mm_add_ps( Y_47, a_47_1 );


		x_0 = _mm_load1_ps( &x[2] );
		x_1 = _mm_load1_ps( &x[3] );

		a_03_0 = _mm_load_ps( &A0[0+lda*2] );
		a_47_0 = _mm_load_ps( &A1[0+lda*2] );
		a_03_1 = _mm_load_ps( &A0[0+lda*3] );
		a_47_1 = _mm_load_ps( &A1[0+lda*3] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_47_0 = _mm_mul_ps( a_47_0, x_0 );
		y_47   = _mm_add_ps( y_47, a_47_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		a_47_1 = _mm_mul_ps( a_47_1, x_1 );
		Y_47   = _mm_add_ps( Y_47, a_47_1 );
		
		A0 += 4*lda;
		A1 += 4*lda;
		x  += 4;

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_03_0 = _mm_load_ps( &A0[0+lda*0] );
		a_47_0 = _mm_load_ps( &A1[0+lda*0] );
		a_03_1 = _mm_load_ps( &A0[0+lda*1] );
		a_47_1 = _mm_load_ps( &A1[0+lda*1] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_47_0 = _mm_mul_ps( a_47_0, x_0 );
		y_47   = _mm_add_ps( y_47, a_47_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		a_47_1 = _mm_mul_ps( a_47_1, x_1 );
		Y_47   = _mm_add_ps( Y_47, a_47_1 );


		x_0 = _mm_load1_ps( &x[2] );
		x_1 = _mm_load1_ps( &x[3] );

		a_03_0 = _mm_load_ps( &A0[0+lda*2] );
		a_47_0 = _mm_load_ps( &A1[0+lda*2] );
		a_03_1 = _mm_load_ps( &A0[0+lda*3] );
		a_47_1 = _mm_load_ps( &A1[0+lda*3] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_47_0 = _mm_mul_ps( a_47_0, x_0 );
		y_47   = _mm_add_ps( y_47, a_47_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		a_47_1 = _mm_mul_ps( a_47_1, x_1 );
		Y_47   = _mm_add_ps( Y_47, a_47_1 );
		
		A0 += 4*lda;
		A1 += 4*lda;
		x  += 4;

		}
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_03_0 = _mm_load_ps( &A0[0+lda*0] );
		a_47_0 = _mm_load_ps( &A1[0+lda*0] );
		a_03_1 = _mm_load_ps( &A0[0+lda*1] );
		a_47_1 = _mm_load_ps( &A1[0+lda*1] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_47_0 = _mm_mul_ps( a_47_0, x_0 );
		y_47   = _mm_add_ps( y_47, a_47_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		a_47_1 = _mm_mul_ps( a_47_1, x_1 );
		Y_47   = _mm_add_ps( Y_47, a_47_1 );


		x_0 = _mm_load1_ps( &x[2] );
		x_1 = _mm_load1_ps( &x[3] );

		a_03_0 = _mm_load_ps( &A0[0+lda*2] );
		a_47_0 = _mm_load_ps( &A1[0+lda*2] );
		a_03_1 = _mm_load_ps( &A0[0+lda*3] );
		a_47_1 = _mm_load_ps( &A1[0+lda*3] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_47_0 = _mm_mul_ps( a_47_0, x_0 );
		y_47   = _mm_add_ps( y_47, a_47_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		a_47_1 = _mm_mul_ps( a_47_1, x_1 );
		Y_47   = _mm_add_ps( Y_47, a_47_1 );
		
		A0 += 4*lda;
		A1 += 4*lda;
		x  += 4;

		}
	if(kmax%4>=2)
		{

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_03_0 = _mm_load_ps( &A0[0+lda*0] );
		a_47_0 = _mm_load_ps( &A1[0+lda*0] );
		a_03_1 = _mm_load_ps( &A0[0+lda*1] );
		a_47_1 = _mm_load_ps( &A1[0+lda*1] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_47_0 = _mm_mul_ps( a_47_0, x_0 );
		y_47   = _mm_add_ps( y_47, a_47_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		a_47_1 = _mm_mul_ps( a_47_1, x_1 );
		Y_47   = _mm_add_ps( Y_47, a_47_1 );
		
		A0 += 2*lda;
		A1 += 2*lda;
		x  += 2;

		}
	
	y_03 = _mm_add_ps( y_03, Y_03 );
	y_47 = _mm_add_ps( y_47, Y_47 );

	if(kmax%2==1)
		{

		x_0 = _mm_load1_ps( &x[0] );

		a_03_0 = _mm_load_ps( &A0[0+lda*0] );
		a_47_0 = _mm_load_ps( &A1[0+lda*0] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_47_0 = _mm_mul_ps( a_47_0, x_0 );
		y_47   = _mm_add_ps( y_47, a_47_0 );
		
/*		A += 1*lda;*/
/*		x += 1;*/

		}

	if(alg==0)
		{
		_mm_storeu_ps(&y[0], y_03);
		_mm_storeu_ps(&y[4], y_47);
		}
	else if(alg==1)
		{
		z_03 = _mm_loadu_ps( &y[0] );
		z_47 = _mm_loadu_ps( &y[4] );

		z_03 = _mm_add_ps( z_03, y_03 );
		z_47 = _mm_add_ps( z_47, y_47 );

		_mm_storeu_ps(&y[0], z_03);
		_mm_storeu_ps(&y[4], z_47);
		}
	else // alg==-1
		{
		z_03 = _mm_loadu_ps( &y[0] );
		z_47 = _mm_loadu_ps( &y[4] );

		z_03 = _mm_sub_ps( z_03, y_03 );
		z_47 = _mm_sub_ps( z_47, y_47 );

		_mm_storeu_ps(&y[0], z_03);
		_mm_storeu_ps(&y[4], z_47);
		}

	}



// it moves horizontally inside a block
void kernel_sgemv_n_4_lib4(int kmax, float *A, float *x, float *y, int alg)
	{

/*printf("\nciaoc\n");*/
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128
		a_03_0, a_03_1,
		x_0, x_1,
		y_03, Y_03, z_03;
	
	y_03 = _mm_setzero_ps();	
	Y_03 = _mm_setzero_ps();	

	k=0;
	for(; k<kmax-7; k+=8)
		{

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );


		x_0 = _mm_load1_ps( &x[2] );
		x_1 = _mm_load1_ps( &x[3] );

		a_03_0 = _mm_load_ps( &A[0+lda*2] );
		a_03_1 = _mm_load_ps( &A[0+lda*3] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		
		A += 4*lda;
		x += 4;

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );


		x_0 = _mm_load1_ps( &x[2] );
		x_1 = _mm_load1_ps( &x[3] );

		a_03_0 = _mm_load_ps( &A[0+lda*2] );
		a_03_1 = _mm_load_ps( &A[0+lda*3] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		
		A += 4*lda;
		x += 4;

		}
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );


		x_0 = _mm_load1_ps( &x[2] );
		x_1 = _mm_load1_ps( &x[3] );

		a_03_0 = _mm_load_ps( &A[0+lda*2] );
		a_03_1 = _mm_load_ps( &A[0+lda*3] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );
		
		A += 4*lda;
		x += 4;

		}
	if(kmax%4>=2)
		{

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );
		a_03_1 = _mm_load_ps( &A[0+lda*1] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		a_03_1 = _mm_mul_ps( a_03_1, x_1 );
		Y_03   = _mm_add_ps( Y_03, a_03_1 );

		A += 2*lda;
		x += 2;

		}

	y_03 = _mm_add_ps( y_03, Y_03 );

	if(kmax%2==1)
		{

		x_0 = _mm_load1_ps( &x[0] );

		a_03_0 = _mm_load_ps( &A[0+lda*0] );

		a_03_0 = _mm_mul_ps( a_03_0, x_0 );
		y_03   = _mm_add_ps( y_03, a_03_0 );
		
/*		A += 1*lda;*/
/*		x += 1;*/

		}

	if(alg==0)
		{
		_mm_storeu_ps(&y[0], y_03);
		}
	else if(alg==1)
		{
		z_03 = _mm_loadu_ps( &y[0] );

		z_03 = _mm_add_ps( z_03, y_03 );

		_mm_storeu_ps(&y[0], z_03);
		}
	else // alg==-1
		{
		z_03 = _mm_loadu_ps( &y[0] );

		z_03 = _mm_sub_ps( z_03, y_03 );

		_mm_storeu_ps(&y[0], z_03);
		}

	}



// it moves horizontally inside a block
void kernel_sgemv_n_2_lib4(int kmax, float *A, float *x, float *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	float
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0;
	
	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];

		y_0 += A[0+lda*0] * x_0;
		y_1 += A[1+lda*0] * x_0;

		y_0 += A[0+lda*1] * x_1;
		y_1 += A[1+lda*1] * x_1;

		y_0 += A[0+lda*2] * x_2;
		y_1 += A[1+lda*2] * x_2;

		y_0 += A[0+lda*3] * x_3;
		y_1 += A[1+lda*3] * x_3;
		
		A += 4*lda;
		x += 4;

		}

	for(; k<kmax; k++)
		{

		x_0 = x[0];

		y_0 += A[0+lda*0] * x_0;
		y_1 += A[1+lda*0] * x_0;
		
		A += 1*lda;
		x += 1;

		}

	if(alg==0)
		{
		y[0] = y_0;
		y[1] = y_1;
		}
	else if(alg==1)
		{
		y[0] += y_0;
		y[1] += y_1;
		}
	else // alg==-1
		{
		y[0] -= y_0;
		y[1] -= y_1;
		}

	}



// it moves horizontally inside a block
void kernel_sgemv_n_1_lib4(int kmax, float *A, float *x, float *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	float
		x_0, x_1, x_2, x_3,
		y_0=0;
	
	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];

		y_0 += A[0+lda*0] * x_0;
		y_0 += A[0+lda*1] * x_1;
		y_0 += A[0+lda*2] * x_2;
		y_0 += A[0+lda*3] * x_3;
		
		A += 4*lda;
		x += 4;

		}

	for(; k<kmax; k++)
		{

		x_0 = x[0];

		y_0 += A[0+lda*0] * x_0;
		
		A += 1*lda;
		x += 1;

		}

	if(alg==0)
		{
		y[0] = y_0;
		}
	else if(alg==1)
		{
		y[0] += y_0;
		}
	else // alg==-1
		{
		y[0] -= y_0;
		}

	}

