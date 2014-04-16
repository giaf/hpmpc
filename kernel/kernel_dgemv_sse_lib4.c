/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical Univeristy of Denmark. All rights reserved.                     *
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
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
//#include <immintrin.h>  // AVX



void kernel_dgemv_t_8_sse_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int
		k, ka=kmax-kna;
	
	__m128d
		a_00_10, a_01_11, a_02_12, a_03_13,
		x_0_1,
		y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7;
	
	y_0 = _mm_setzero_pd();	
	y_1 = _mm_setzero_pd();	
	y_2 = _mm_setzero_pd();	
	y_3 = _mm_setzero_pd();	
	y_4 = _mm_setzero_pd();	
	y_5 = _mm_setzero_pd();	
	y_6 = _mm_setzero_pd();	
	y_7 = _mm_setzero_pd();	
	
	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0_1 = _mm_load_sd( &x[0] );

			a_00_10 = _mm_load_sd( &A[0+lda*0] );
			a_01_11 = _mm_load_sd( &A[0+lda*1] );
			a_02_12 = _mm_load_sd( &A[0+lda*2] );
			a_03_13 = _mm_load_sd( &A[0+lda*3] );
			
			a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
			y_0 = _mm_add_sd( y_0, a_00_10 );
			a_01_11 = _mm_mul_sd( a_01_11, x_0_1 );
			y_1 = _mm_add_sd( y_1, a_01_11 );
			a_02_12 = _mm_mul_sd( a_02_12, x_0_1 );
			y_2 = _mm_add_sd( y_2, a_02_12 );
			a_03_13 = _mm_mul_sd( a_03_13, x_0_1 );
			y_3 = _mm_add_sd( y_3, a_03_13 );

			a_00_10 = _mm_load_sd( &A[0+lda*4] );
			a_01_11 = _mm_load_sd( &A[0+lda*5] );
			a_02_12 = _mm_load_sd( &A[0+lda*6] );
			a_03_13 = _mm_load_sd( &A[0+lda*7] );
			
			a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
			y_4 = _mm_add_sd( y_4, a_00_10 );
			a_01_11 = _mm_mul_sd( a_01_11, x_0_1 );
			y_5 = _mm_add_sd( y_5, a_01_11 );
			a_02_12 = _mm_mul_sd( a_02_12, x_0_1 );
			y_6 = _mm_add_sd( y_6, a_02_12 );
			a_03_13 = _mm_mul_sd( a_03_13, x_0_1 );
			y_7 = _mm_add_sd( y_7, a_03_13 );

			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k=0;
	for(; k<ka-3; k+=4)
		{
		
		x_0_1 = _mm_loadu_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		a_01_11 = _mm_load_pd( &A[0+lda*1] );
		a_02_12 = _mm_load_pd( &A[0+lda*2] );
		a_03_13 = _mm_load_pd( &A[0+lda*3] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_0 = _mm_add_pd( y_0, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_0_1 );
		y_1 = _mm_add_pd( y_1, a_01_11 );
		a_02_12 = _mm_mul_pd( a_02_12, x_0_1 );
		y_2 = _mm_add_pd( y_2, a_02_12 );
		a_03_13 = _mm_mul_pd( a_03_13, x_0_1 );
		y_3 = _mm_add_pd( y_3, a_03_13 );

		a_00_10 = _mm_load_pd( &A[0+lda*4] );
		a_01_11 = _mm_load_pd( &A[0+lda*5] );
		a_02_12 = _mm_load_pd( &A[0+lda*6] );
		a_03_13 = _mm_load_pd( &A[0+lda*7] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_4 = _mm_add_pd( y_4, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_0_1 );
		y_5 = _mm_add_pd( y_5, a_01_11 );
		a_02_12 = _mm_mul_pd( a_02_12, x_0_1 );
		y_6 = _mm_add_pd( y_6, a_02_12 );
		a_03_13 = _mm_mul_pd( a_03_13, x_0_1 );
		y_7 = _mm_add_pd( y_7, a_03_13 );


		x_0_1 = _mm_loadu_pd( &x[2] );

		a_00_10 = _mm_load_pd( &A[2+lda*0] );
		a_01_11 = _mm_load_pd( &A[2+lda*1] );
		a_02_12 = _mm_load_pd( &A[2+lda*2] );
		a_03_13 = _mm_load_pd( &A[2+lda*3] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_0 = _mm_add_pd( y_0, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_0_1 );
		y_1 = _mm_add_pd( y_1, a_01_11 );
		a_02_12 = _mm_mul_pd( a_02_12, x_0_1 );
		y_2 = _mm_add_pd( y_2, a_02_12 );
		a_03_13 = _mm_mul_pd( a_03_13, x_0_1 );
		y_3 = _mm_add_pd( y_3, a_03_13 );
		
		a_00_10 = _mm_load_pd( &A[2+lda*4] );
		a_01_11 = _mm_load_pd( &A[2+lda*5] );
		a_02_12 = _mm_load_pd( &A[2+lda*6] );
		a_03_13 = _mm_load_pd( &A[2+lda*7] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_4 = _mm_add_pd( y_4, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_0_1 );
		y_5 = _mm_add_pd( y_5, a_01_11 );
		a_02_12 = _mm_mul_pd( a_02_12, x_0_1 );
		y_6 = _mm_add_pd( y_6, a_02_12 );
		a_03_13 = _mm_mul_pd( a_03_13, x_0_1 );
		y_7 = _mm_add_pd( y_7, a_03_13 );
		

		A += 4 + (sda-1)*lda;
		x += 4;

		}
	
	for(; k<ka; k++)
		{
		
		x_0_1 = _mm_load_sd( &x[0] );

		a_00_10 = _mm_load_sd( &A[0+lda*0] );
		a_01_11 = _mm_load_sd( &A[0+lda*1] );
		a_02_12 = _mm_load_sd( &A[0+lda*2] );
		a_03_13 = _mm_load_sd( &A[0+lda*3] );
	
		a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
		y_0 = _mm_add_sd( y_0, a_00_10 );
		a_01_11 = _mm_mul_sd( a_01_11, x_0_1 );
		y_1 = _mm_add_sd( y_1, a_01_11 );
		a_02_12 = _mm_mul_sd( a_02_12, x_0_1 );
		y_2 = _mm_add_sd( y_2, a_02_12 );
		a_03_13 = _mm_mul_sd( a_03_13, x_0_1 );
		y_3 = _mm_add_sd( y_3, a_03_13 );
	
		a_00_10 = _mm_load_sd( &A[0+lda*4] );
		a_01_11 = _mm_load_sd( &A[0+lda*5] );
		a_02_12 = _mm_load_sd( &A[0+lda*6] );
		a_03_13 = _mm_load_sd( &A[0+lda*7] );
	
		a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
		y_4 = _mm_add_sd( y_4, a_00_10 );
		a_01_11 = _mm_mul_sd( a_01_11, x_0_1 );
		y_5 = _mm_add_sd( y_5, a_01_11 );
		a_02_12 = _mm_mul_sd( a_02_12, x_0_1 );
		y_6 = _mm_add_sd( y_6, a_02_12 );
		a_03_13 = _mm_mul_sd( a_03_13, x_0_1 );
		y_7 = _mm_add_sd( y_7, a_03_13 );
	
		A += 1;
		x += 1;
		
		}

	__m128d
		y_0_1, y_2_3, y_4_5, y_6_7;

	y_0 = _mm_hadd_pd(y_0, y_1);
	y_2 = _mm_hadd_pd(y_2, y_3);
	y_4 = _mm_hadd_pd(y_4, y_5);
	y_6 = _mm_hadd_pd(y_6, y_7);

	if(alg==0)
		{
		_mm_storeu_pd(&y[0], y_0);
		_mm_storeu_pd(&y[2], y_2);
		_mm_storeu_pd(&y[4], y_4);
		_mm_storeu_pd(&y[6], y_6);
		}
	else if(alg==1)
		{
		y_0_1 = _mm_loadu_pd( &y[0] );
		y_2_3 = _mm_loadu_pd( &y[2] );
		y_4_5 = _mm_loadu_pd( &y[4] );
		y_6_7 = _mm_loadu_pd( &y[6] );

		y_0_1 = _mm_add_pd(y_0_1, y_0);
		y_2_3 = _mm_add_pd(y_2_3, y_2);
		y_4_5 = _mm_add_pd(y_4_5, y_4);
		y_6_7 = _mm_add_pd(y_6_7, y_6);
	
		_mm_storeu_pd(&y[0], y_0_1);
		_mm_storeu_pd(&y[2], y_2_3);
		_mm_storeu_pd(&y[4], y_4_5);
		_mm_storeu_pd(&y[6], y_6_7);
		}
	else // alg==-1
		{
		y_0_1 = _mm_loadu_pd( &y[0] );
		y_2_3 = _mm_loadu_pd( &y[2] );
		y_4_5 = _mm_loadu_pd( &y[4] );
		y_6_7 = _mm_loadu_pd( &y[6] );
	
		y_0_1 = _mm_sub_pd(y_0_1, y_0);
		y_2_3 = _mm_sub_pd(y_2_3, y_2);
		y_4_5 = _mm_sub_pd(y_4_5, y_4);
		y_6_7 = _mm_sub_pd(y_6_7, y_6);
	
		_mm_storeu_pd(&y[0], y_0_1);
		_mm_storeu_pd(&y[2], y_2_3);
		_mm_storeu_pd(&y[4], y_4_5);
		_mm_storeu_pd(&y[6], y_6_7);
		}

	}



void kernel_dgemv_t_4_sse_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int
		k, ka=kmax-kna;
	
	__m128d
		a_00_10, a_01_11, a_02_12, a_03_13,
		x_0_1,
		y_0, y_1, y_2, y_3;
	
	y_0 = _mm_setzero_pd();	
	y_1 = _mm_setzero_pd();	
	y_2 = _mm_setzero_pd();	
	y_3 = _mm_setzero_pd();	
	
	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0_1 = _mm_load_sd( &x[0] );

			a_00_10 = _mm_load_sd( &A[0+lda*0] );
			a_01_11 = _mm_load_sd( &A[0+lda*1] );
			a_02_12 = _mm_load_sd( &A[0+lda*2] );
			a_03_13 = _mm_load_sd( &A[0+lda*3] );
			
			a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
			y_0 = _mm_add_sd( y_0, a_00_10 );
			a_01_11 = _mm_mul_sd( a_01_11, x_0_1 );
			y_1 = _mm_add_sd( y_1, a_01_11 );
			a_02_12 = _mm_mul_sd( a_02_12, x_0_1 );
			y_2 = _mm_add_sd( y_2, a_02_12 );
			a_03_13 = _mm_mul_sd( a_03_13, x_0_1 );
			y_3 = _mm_add_sd( y_3, a_03_13 );

			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k=0;
	for(; k<ka-3; k+=4)
		{
		
		x_0_1 = _mm_loadu_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		a_01_11 = _mm_load_pd( &A[0+lda*1] );
		a_02_12 = _mm_load_pd( &A[0+lda*2] );
		a_03_13 = _mm_load_pd( &A[0+lda*3] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_0 = _mm_add_pd( y_0, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_0_1 );
		y_1 = _mm_add_pd( y_1, a_01_11 );
		a_02_12 = _mm_mul_pd( a_02_12, x_0_1 );
		y_2 = _mm_add_pd( y_2, a_02_12 );
		a_03_13 = _mm_mul_pd( a_03_13, x_0_1 );
		y_3 = _mm_add_pd( y_3, a_03_13 );


		x_0_1 = _mm_loadu_pd( &x[2] );

		a_00_10 = _mm_load_pd( &A[2+lda*0] );
		a_01_11 = _mm_load_pd( &A[2+lda*1] );
		a_02_12 = _mm_load_pd( &A[2+lda*2] );
		a_03_13 = _mm_load_pd( &A[2+lda*3] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_0 = _mm_add_pd( y_0, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_0_1 );
		y_1 = _mm_add_pd( y_1, a_01_11 );
		a_02_12 = _mm_mul_pd( a_02_12, x_0_1 );
		y_2 = _mm_add_pd( y_2, a_02_12 );
		a_03_13 = _mm_mul_pd( a_03_13, x_0_1 );
		y_3 = _mm_add_pd( y_3, a_03_13 );
		

		A += 4 + (sda-1)*lda;
		x += 4;

		}
	
	for(; k<ka; k++)
		{
		
		x_0_1 = _mm_load_sd( &x[0] );

		a_00_10 = _mm_load_sd( &A[0+lda*0] );
		a_01_11 = _mm_load_sd( &A[0+lda*1] );
		a_02_12 = _mm_load_sd( &A[0+lda*2] );
		a_03_13 = _mm_load_sd( &A[0+lda*3] );
	
		a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
		y_0 = _mm_add_sd( y_0, a_00_10 );
		a_01_11 = _mm_mul_sd( a_01_11, x_0_1 );
		y_1 = _mm_add_sd( y_1, a_01_11 );
		a_02_12 = _mm_mul_sd( a_02_12, x_0_1 );
		y_2 = _mm_add_sd( y_2, a_02_12 );
		a_03_13 = _mm_mul_sd( a_03_13, x_0_1 );
		y_3 = _mm_add_sd( y_3, a_03_13 );
	
		A += 1;
		x += 1;
		
		}

	__m128d
		y_0_1, y_2_3;

	y_0 = _mm_hadd_pd(y_0, y_1);
	y_2 = _mm_hadd_pd(y_2, y_3);

	if(alg==0)
		{
		_mm_storeu_pd(&y[0], y_0);
		_mm_storeu_pd(&y[2], y_2);
		}
	else if(alg==1)
		{
		y_0_1 = _mm_loadu_pd( &y[0] );
		y_2_3 = _mm_loadu_pd( &y[2] );

		y_0_1 = _mm_add_pd(y_0_1, y_0);
		y_2_3 = _mm_add_pd(y_2_3, y_2);
	
		_mm_storeu_pd(&y[0], y_0_1);
		_mm_storeu_pd(&y[2], y_2_3);
		}
	else // alg==-1
		{
		y_0_1 = _mm_loadu_pd( &y[0] );
		y_2_3 = _mm_loadu_pd( &y[2] );
	
		y_0_1 = _mm_sub_pd(y_0_1, y_0);
		y_2_3 = _mm_sub_pd(y_2_3, y_2);
	
		_mm_storeu_pd(&y[0], y_0_1);
		_mm_storeu_pd(&y[2], y_2_3);
		}

	}



void kernel_dgemv_t_2_sse_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int
		k, ka=kmax-kna;
	
	__m128d
		a_00_10, a_01_11,
		x_0_1,
		y_0, y_1;
	
	y_0 = _mm_setzero_pd();	
	y_1 = _mm_setzero_pd();	
	
	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0_1 = _mm_load_sd( &x[0] );

			a_00_10 = _mm_load_sd( &A[0+lda*0] );
			a_01_11 = _mm_load_sd( &A[0+lda*1] );
		
			a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
			y_0 = _mm_add_sd( y_0, a_00_10 );
			a_01_11 = _mm_mul_sd( a_01_11, x_0_1 );
			y_1 = _mm_add_sd( y_1, a_01_11 );
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k=0;
	for(; k<ka-3; k+=4)
		{
		
		x_0_1 = _mm_loadu_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		a_01_11 = _mm_load_pd( &A[0+lda*1] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_0 = _mm_add_pd( y_0, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_0_1 );
		y_1 = _mm_add_pd( y_1, a_01_11 );

		x_0_1 = _mm_loadu_pd( &x[2] );

		a_00_10 = _mm_load_pd( &A[2+lda*0] );
		a_01_11 = _mm_load_pd( &A[2+lda*1] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_0 = _mm_add_pd( y_0, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_0_1 );
		y_1 = _mm_add_pd( y_1, a_01_11 );
		
		A += 4 + (sda-1)*lda;
		x += 4;

		}
	
	for(; k<ka; k++)
		{
		
		x_0_1 = _mm_load_sd( &x[0] );

		a_00_10 = _mm_load_sd( &A[0+lda*0] );
		a_01_11 = _mm_load_sd( &A[0+lda*1] );
	
		a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
		y_0 = _mm_add_sd( y_0, a_00_10 );
		a_01_11 = _mm_mul_sd( a_01_11, x_0_1 );
		y_1 = _mm_add_sd( y_1, a_01_11 );
	
		A += 1;
		x += 1;
		
		}

	__m128d
		y_0_1;

	y_0 = _mm_hadd_pd(y_0, y_1);

	if(alg==0)
		{
		_mm_storeu_pd(&y[0], y_0);
		}
	else if(alg==1)
		{
		y_0_1 = _mm_loadu_pd( &y[0] );

		y_0_1 = _mm_add_pd(y_0_1, y_0);
	
		_mm_storeu_pd(&y[0], y_0_1);
		}
	else // alg==-1
		{
		y_0_1 = _mm_loadu_pd( &y[0] );
	
		y_0_1 = _mm_sub_pd(y_0_1, y_0);
	
		_mm_storeu_pd(&y[0], y_0_1);
		}
	
	}



void kernel_dgemv_t_1_sse_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int
		k, ka=kmax-kna;
	
	__m128d
		a_00_10,
		x_0_1,
		y_0;
	
	y_0 = _mm_setzero_pd();	
	
	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0_1 = _mm_load_sd( &x[0] );

			a_00_10 = _mm_load_sd( &A[0+lda*0] );
		
			a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
			y_0 = _mm_add_sd( y_0, a_00_10 );
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k=0;
	for(; k<ka-3; k+=4)
		{
		
		x_0_1 = _mm_loadu_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_0 = _mm_add_pd( y_0, a_00_10 );

		x_0_1 = _mm_loadu_pd( &x[2] );

		a_00_10 = _mm_load_pd( &A[2+lda*0] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0_1 );
		y_0 = _mm_add_pd( y_0, a_00_10 );
		
		A += 4 + (sda-1)*lda;
		x += 4;

		}
	
	for(; k<ka; k++)
		{
		
		x_0_1 = _mm_load_sd( &x[0] );

		a_00_10 = _mm_load_sd( &A[0+lda*0] );
	
		a_00_10 = _mm_mul_sd( a_00_10, x_0_1 );
		y_0 = _mm_add_sd( y_0, a_00_10 );
	
		A += 1;
		x += 1;
		
		}

	__m128d
		y_0_1;

	y_0 = _mm_hadd_pd(y_0, y_0);

	if(alg==0)
		{
		_mm_store_sd(&y[0], y_0);
		}
	else if(alg==1)
		{
		y_0_1 = _mm_load_sd( &y[0] );

		y_0_1 = _mm_add_sd(y_0_1, y_0);
	
		_mm_store_sd(&y[0], y_0_1);
		}
	else // alg==-1
		{
		y_0_1 = _mm_load_sd( &y[0] );
	
		y_0_1 = _mm_sub_sd(y_0_1, y_0);
	
		_mm_store_sd(&y[0], y_0_1);
		}

	}



// it moves horizontally inside a block
void kernel_dgemv_n_8_sse_lib4(int kmax, double *A0, double *A1, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128d
		a_00_10, a_20_30, a_40_50, a_60_70,
		x_0,
		y_0_1, y_2_3, y_4_5, y_6_7, z_0_1, z_2_3, z_4_5, z_6_7;
	
	y_0_1 = _mm_setzero_pd();	
	y_2_3 = _mm_setzero_pd();	
	y_4_5 = _mm_setzero_pd();	
	y_6_7 = _mm_setzero_pd();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_loaddup_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A0[0+lda*0] );
		a_20_30 = _mm_load_pd( &A0[2+lda*0] );
		a_40_50 = _mm_load_pd( &A1[0+lda*0] );
		a_60_70 = _mm_load_pd( &A1[2+lda*0] );

		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_20_30 = _mm_mul_pd( a_20_30, x_0 );
		y_2_3   = _mm_add_pd( y_2_3, a_20_30 );
		a_40_50 = _mm_mul_pd( a_40_50, x_0 );
		y_4_5   = _mm_add_pd( y_4_5, a_40_50 );
		a_60_70 = _mm_mul_pd( a_60_70, x_0 );
		y_6_7   = _mm_add_pd( y_6_7, a_60_70 );


		x_0 = _mm_loaddup_pd( &x[1] );

		a_00_10 = _mm_load_pd( &A0[0+lda*1] );
		a_20_30 = _mm_load_pd( &A0[2+lda*1] );
		a_40_50 = _mm_load_pd( &A1[0+lda*1] );
		a_60_70 = _mm_load_pd( &A1[2+lda*1] );

		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_20_30 = _mm_mul_pd( a_20_30, x_0 );
		y_2_3   = _mm_add_pd( y_2_3, a_20_30 );
		a_40_50 = _mm_mul_pd( a_40_50, x_0 );
		y_4_5   = _mm_add_pd( y_4_5, a_40_50 );
		a_60_70 = _mm_mul_pd( a_60_70, x_0 );
		y_6_7   = _mm_add_pd( y_6_7, a_60_70 );


		x_0 = _mm_loaddup_pd( &x[2] );

		a_00_10 = _mm_load_pd( &A0[0+lda*2] );
		a_20_30 = _mm_load_pd( &A0[2+lda*2] );
		a_40_50 = _mm_load_pd( &A1[0+lda*2] );
		a_60_70 = _mm_load_pd( &A1[2+lda*2] );

		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_20_30 = _mm_mul_pd( a_20_30, x_0 );
		y_2_3   = _mm_add_pd( y_2_3, a_20_30 );
		a_40_50 = _mm_mul_pd( a_40_50, x_0 );
		y_4_5   = _mm_add_pd( y_4_5, a_40_50 );
		a_60_70 = _mm_mul_pd( a_60_70, x_0 );
		y_6_7   = _mm_add_pd( y_6_7, a_60_70 );

		
		x_0 = _mm_loaddup_pd( &x[3] );

		a_00_10 = _mm_load_pd( &A0[0+lda*3] );
		a_20_30 = _mm_load_pd( &A0[2+lda*3] );
		a_40_50 = _mm_load_pd( &A1[0+lda*3] );
		a_60_70 = _mm_load_pd( &A1[2+lda*3] );

		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_20_30 = _mm_mul_pd( a_20_30, x_0 );
		y_2_3   = _mm_add_pd( y_2_3, a_20_30 );
		a_40_50 = _mm_mul_pd( a_40_50, x_0 );
		y_4_5   = _mm_add_pd( y_4_5, a_40_50 );
		a_60_70 = _mm_mul_pd( a_60_70, x_0 );
		y_6_7   = _mm_add_pd( y_6_7, a_60_70 );


		A0 += 4*lda;
		A1 += 4*lda;
		x  += 4;

		}
	
	for(; k<kmax; k++)
		{

		x_0 = _mm_loaddup_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A0[0+lda*0] );
		a_20_30 = _mm_load_pd( &A0[2+lda*0] );
		a_40_50 = _mm_load_pd( &A1[0+lda*0] );
		a_60_70 = _mm_load_pd( &A1[2+lda*0] );

		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_20_30 = _mm_mul_pd( a_20_30, x_0 );
		y_2_3   = _mm_add_pd( y_2_3, a_20_30 );
		a_40_50 = _mm_mul_pd( a_40_50, x_0 );
		y_4_5   = _mm_add_pd( y_4_5, a_40_50 );
		a_60_70 = _mm_mul_pd( a_60_70, x_0 );
		y_6_7   = _mm_add_pd( y_6_7, a_60_70 );
		
		A0 += 1*lda;
		A1 += 1*lda;
		x  += 1;

		}

	if(alg==0)
		{
		_mm_storeu_pd(&y[0], y_0_1);
		_mm_storeu_pd(&y[2], y_2_3);
		_mm_storeu_pd(&y[4], y_4_5);
		_mm_storeu_pd(&y[6], y_6_7);
		}
	else if(alg==1)
		{
		z_0_1 = _mm_loadu_pd( &y[0] );
		z_2_3 = _mm_loadu_pd( &y[2] );
		z_4_5 = _mm_loadu_pd( &y[4] );
		z_6_7 = _mm_loadu_pd( &y[6] );

		z_0_1 = _mm_add_pd( z_0_1, y_0_1 );
		z_2_3 = _mm_add_pd( z_2_3, y_2_3 );
		z_4_5 = _mm_add_pd( z_4_5, y_4_5 );
		z_6_7 = _mm_add_pd( z_6_7, y_6_7 );

		_mm_storeu_pd(&y[0], z_0_1);
		_mm_storeu_pd(&y[2], z_2_3);
		_mm_storeu_pd(&y[4], z_4_5);
		_mm_storeu_pd(&y[6], z_6_7);
		}
	else // alg==-1
		{
		z_0_1 = _mm_loadu_pd( &y[0] );
		z_2_3 = _mm_loadu_pd( &y[2] );
		z_4_5 = _mm_loadu_pd( &y[4] );
		z_6_7 = _mm_loadu_pd( &y[6] );

		z_0_1 = _mm_sub_pd( z_0_1, y_0_1 );
		z_2_3 = _mm_sub_pd( z_2_3, y_2_3 );
		z_4_5 = _mm_sub_pd( z_4_5, y_4_5 );
		z_6_7 = _mm_sub_pd( z_6_7, y_6_7 );

		_mm_storeu_pd(&y[0], z_0_1);
		_mm_storeu_pd(&y[2], z_2_3);
		_mm_storeu_pd(&y[4], z_4_5);
		_mm_storeu_pd(&y[6], z_6_7);
		}

	}



// it moves horizontally inside a block
void kernel_dgemv_n_4_sse_lib4(int kmax, double *A, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128d
		a_00_10, a_01_11, a_20_30, a_21_31,
		x_0, x_1,
		y_0_1, y_2_3, y_0_1_b, y_2_3_b, z_0_1, z_2_3;
	
	y_0_1   = _mm_setzero_pd();	
	y_2_3   = _mm_setzero_pd();	
	y_0_1_b = _mm_setzero_pd();	
	y_2_3_b = _mm_setzero_pd();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_loaddup_pd( &x[0] );
		x_1 = _mm_loaddup_pd( &x[1] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		a_20_30 = _mm_load_pd( &A[2+lda*0] );
		a_01_11 = _mm_load_pd( &A[0+lda*1] );
		a_21_31 = _mm_load_pd( &A[2+lda*1] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_20_30 = _mm_mul_pd( a_20_30, x_0 );
		y_2_3   = _mm_add_pd( y_2_3, a_20_30 );
		a_01_11 = _mm_mul_pd( a_01_11, x_1 );
		y_0_1_b = _mm_add_pd( y_0_1_b, a_01_11 );
		a_21_31 = _mm_mul_pd( a_21_31, x_1 );
		y_2_3_b = _mm_add_pd( y_2_3_b, a_21_31 );


		x_0 = _mm_loaddup_pd( &x[2] );
		x_1 = _mm_loaddup_pd( &x[3] );

		a_00_10 = _mm_load_pd( &A[0+lda*2] );
		a_20_30 = _mm_load_pd( &A[2+lda*2] );
		a_01_11 = _mm_load_pd( &A[0+lda*3] );
		a_21_31 = _mm_load_pd( &A[2+lda*3] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_20_30 = _mm_mul_pd( a_20_30, x_0 );
		y_2_3   = _mm_add_pd( y_2_3, a_20_30 );
		a_01_11 = _mm_mul_pd( a_01_11, x_1 );
		y_0_1_b = _mm_add_pd( y_0_1_b, a_01_11 );
		a_21_31 = _mm_mul_pd( a_21_31, x_1 );
		y_2_3_b = _mm_add_pd( y_2_3_b, a_21_31 );

		
		A += 4*lda;
		x += 4;

		}
	
	y_0_1 = _mm_add_pd( y_0_1, y_0_1_b );
	y_2_3 = _mm_add_pd( y_2_3, y_2_3_b );

	for(; k<kmax; k++)
		{

		x_0 = _mm_loaddup_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		a_20_30 = _mm_load_pd( &A[2+lda*0] );

		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_20_30 = _mm_mul_pd( a_20_30, x_0 );
		y_2_3   = _mm_add_pd( y_2_3, a_20_30 );
		
		A += 1*lda;
		x += 1;

		}

	if(alg==0)
		{
		_mm_storeu_pd(&y[0], y_0_1);
		_mm_storeu_pd(&y[2], y_2_3);
		}
	else if(alg==1)
		{
		z_0_1 = _mm_loadu_pd( &y[0] );
		z_2_3 = _mm_loadu_pd( &y[2] );

		z_0_1 = _mm_add_pd( z_0_1, y_0_1 );
		z_2_3 = _mm_add_pd( z_2_3, y_2_3 );

		_mm_storeu_pd(&y[0], z_0_1);
		_mm_storeu_pd(&y[2], z_2_3);
		}
	else // alg==-1
		{
		z_0_1 = _mm_loadu_pd( &y[0] );
		z_2_3 = _mm_loadu_pd( &y[2] );

		z_0_1 = _mm_sub_pd( z_0_1, y_0_1 );
		z_2_3 = _mm_sub_pd( z_2_3, y_2_3 );

		_mm_storeu_pd(&y[0], z_0_1);
		_mm_storeu_pd(&y[2], z_2_3);
		}

	}



// it moves horizontally inside a block
void kernel_dgemv_n_2_sse_lib4(int kmax, double *A, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128d
		a_00_10, a_01_11,
		x_0, x_1,
		y_0_1, y_0_1_b, z_0_1;
	
	y_0_1   = _mm_setzero_pd();	
	y_0_1_b = _mm_setzero_pd();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_loaddup_pd( &x[0] );
		x_1 = _mm_loaddup_pd( &x[1] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		a_01_11 = _mm_load_pd( &A[0+lda*1] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_1 );
		y_0_1_b = _mm_add_pd( y_0_1_b, a_01_11 );


		x_0 = _mm_loaddup_pd( &x[2] );
		x_1 = _mm_loaddup_pd( &x[3] );

		a_00_10 = _mm_load_pd( &A[0+lda*2] );
		a_01_11 = _mm_load_pd( &A[0+lda*3] );
		
		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		a_01_11 = _mm_mul_pd( a_01_11, x_1 );
		y_0_1_b = _mm_add_pd( y_0_1_b, a_01_11 );

		
		A += 4*lda;
		x += 4;

		}
	
	y_0_1 = _mm_add_pd( y_0_1, y_0_1_b );

	for(; k<kmax; k++)
		{

		x_0 = _mm_loaddup_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );

		a_00_10 = _mm_mul_pd( a_00_10, x_0 );
		y_0_1   = _mm_add_pd( y_0_1, a_00_10 );
		
		A += 1*lda;
		x += 1;

		}

	if(alg==0)
		{
		_mm_storeu_pd(&y[0], y_0_1);
		}
	else if(alg==1)
		{
		z_0_1 = _mm_loadu_pd( &y[0] );

		z_0_1 = _mm_add_pd( z_0_1, y_0_1 );

		_mm_storeu_pd(&y[0], z_0_1);
		}
	else // alg==-1
		{
		z_0_1 = _mm_loadu_pd( &y[0] );

		z_0_1 = _mm_sub_pd( z_0_1, y_0_1 );

		_mm_storeu_pd(&y[0], z_0_1);
		}

	}



// it moves horizontally inside a block
void kernel_dgemv_n_1_sse_lib4(int kmax, double *A, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128d
		a_00_10, a_01_11,
		x_0, x_1,
		y_0_1, y_0_1_b, z_0_1;
	
	y_0_1   = _mm_setzero_pd();	
	y_0_1_b = _mm_setzero_pd();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_load_sd( &x[0] );
		x_1 = _mm_load_sd( &x[1] );

		a_00_10 = _mm_load_sd( &A[0+lda*0] );
		a_01_11 = _mm_load_sd( &A[0+lda*1] );

		a_00_10 = _mm_mul_sd( a_00_10, x_0 );
		y_0_1   = _mm_add_sd( y_0_1, a_00_10 );
		a_01_11 = _mm_mul_sd( a_01_11, x_1 );
		y_0_1_b = _mm_add_sd( y_0_1_b, a_01_11 );


		x_0 = _mm_load_sd( &x[2] );
		x_1 = _mm_load_sd( &x[3] );

		a_00_10 = _mm_load_sd( &A[0+lda*2] );
		a_01_11 = _mm_load_sd( &A[0+lda*3] );

		a_00_10 = _mm_mul_sd( a_00_10, x_0 );
		y_0_1   = _mm_add_sd( y_0_1, a_00_10 );
		a_01_11 = _mm_mul_sd( a_01_11, x_1 );
		y_0_1_b = _mm_add_sd( y_0_1_b, a_01_11 );

		
		A += 4*lda;
		x += 4;

		}

	for(; k<kmax; k++)
		{

		x_0 = _mm_load_sd( &x[0] );

		a_00_10 = _mm_load_sd( &A[0+lda*0] );

		a_00_10 = _mm_mul_sd( a_00_10, x_0 );
		y_0_1   = _mm_add_sd( y_0_1, a_00_10 );
		
		A += 1*lda;
		x += 1;

		}

	if(alg==0)
		{
		_mm_store_sd(&y[0], y_0_1);
		}
	else if(alg==1)
		{
		z_0_1 = _mm_load_sd( &y[0] );

		z_0_1 = _mm_add_sd( z_0_1, y_0_1 );

		_mm_store_sd(&y[0], z_0_1);
		}
	else // alg==-1
		{
		z_0_1 = _mm_load_sd( &y[0] );

		z_0_1 = _mm_sub_sd( z_0_1, y_0_1 );

		_mm_store_sd(&y[0], z_0_1);
		}

	}

