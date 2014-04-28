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
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX



void kernel_dgemv_t_8_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	__builtin_prefetch( A + 0*lda );
	__builtin_prefetch( A + 2*lda );
	__builtin_prefetch( A + 4*lda );
	__builtin_prefetch( A + 6*lda );

	double *tA, *tx;

	int k;
	int ka = kmax-kna; // number from aligned positon
	
	__m256d
		aaxx_temp, 
		a_00_10_20_30, a_01_11_21_31, a_02_12_22_32, a_03_13_23_33,
		x_0_1_2_3,
		y_00, y_11, y_22, y_33, y_44, y_55, y_66, y_77;
	
	__m128d
		ax_temp,
		a_00_10, a_01_11, a_02_12, a_03_13,
		x_0_1,
		y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7;
	
	y_00 = _mm256_setzero_pd();
	y_11 = _mm256_setzero_pd();
	y_22 = _mm256_setzero_pd();
	y_33 = _mm256_setzero_pd();
	y_44 = _mm256_setzero_pd();
	y_55 = _mm256_setzero_pd();
	y_66 = _mm256_setzero_pd();
	y_77 = _mm256_setzero_pd();
	
	y_0 = _mm256_castpd256_pd128(y_00);
	y_1 = _mm256_castpd256_pd128(y_11);
	y_2 = _mm256_castpd256_pd128(y_22);
	y_3 = _mm256_castpd256_pd128(y_33);
	y_4 = _mm256_castpd256_pd128(y_44);
	y_5 = _mm256_castpd256_pd128(y_55);
	y_6 = _mm256_castpd256_pd128(y_66);
	y_7 = _mm256_castpd256_pd128(y_77);

	if(kna>0) // it can be only kna = {1, 2, 3}
		{
		k=0;
		if(kna%2==1)
			{
		
			x_0_1 = _mm_load_sd( &x[0] );

			a_00_10 = _mm_load_sd( &A[0+lda*0] );
			a_01_11 = _mm_load_sd( &A[0+lda*1] );
			a_02_12 = _mm_load_sd( &A[0+lda*2] );
			a_03_13 = _mm_load_sd( &A[0+lda*3] );

			ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
			y_0 = _mm_add_sd (y_0, ax_temp );
			ax_temp = _mm_mul_sd( a_01_11, x_0_1 );	
			y_1 = _mm_add_sd (y_1, ax_temp );
			ax_temp = _mm_mul_sd( a_02_12, x_0_1 );	
			y_2 = _mm_add_sd (y_2, ax_temp );
			ax_temp = _mm_mul_sd( a_03_13, x_0_1 );	
			y_3 = _mm_add_sd (y_3, ax_temp );
		
			a_00_10 = _mm_load_sd( &A[0+lda*4] );
			a_01_11 = _mm_load_sd( &A[0+lda*5] );
			a_02_12 = _mm_load_sd( &A[0+lda*6] );
			a_03_13 = _mm_load_sd( &A[0+lda*7] );

			ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
			y_4 = _mm_add_sd (y_4, ax_temp );
			ax_temp = _mm_mul_sd( a_01_11, x_0_1 );	
			y_5 = _mm_add_sd (y_5, ax_temp );
			ax_temp = _mm_mul_sd( a_02_12, x_0_1 );	
			y_6 = _mm_add_sd (y_6, ax_temp );
			ax_temp = _mm_mul_sd( a_03_13, x_0_1 );	
			y_7 = _mm_add_sd (y_7, ax_temp );
		
			A += 1;
			x += 1;
			k++;
		
			}
	
		if(kna%4>=2)
			{
		
			x_0_1 = _mm_load_pd( &x[0] );

			a_00_10 = _mm_load_pd( &A[0+lda*0] );
			a_01_11 = _mm_load_pd( &A[0+lda*1] );
			a_02_12 = _mm_load_pd( &A[0+lda*2] );
			a_03_13 = _mm_load_pd( &A[0+lda*3] );

			ax_temp = _mm_mul_pd( a_00_10, x_0_1 );	
			y_0 = _mm_add_pd (y_0, ax_temp );
			ax_temp = _mm_mul_pd( a_01_11, x_0_1 );	
			y_1 = _mm_add_pd (y_1, ax_temp );
			ax_temp = _mm_mul_pd( a_02_12, x_0_1 );	
			y_2 = _mm_add_pd (y_2, ax_temp );
			ax_temp = _mm_mul_pd( a_03_13, x_0_1 );	
			y_3 = _mm_add_pd (y_3, ax_temp );
		
			a_00_10 = _mm_load_pd( &A[0+lda*4] );
			a_01_11 = _mm_load_pd( &A[0+lda*5] );
			a_02_12 = _mm_load_pd( &A[0+lda*6] );
			a_03_13 = _mm_load_pd( &A[0+lda*7] );

			ax_temp = _mm_mul_pd( a_00_10, x_0_1 );	
			y_4 = _mm_add_pd (y_4, ax_temp );
			ax_temp = _mm_mul_pd( a_01_11, x_0_1 );	
			y_5 = _mm_add_pd (y_5, ax_temp );
			ax_temp = _mm_mul_pd( a_02_12, x_0_1 );	
			y_6 = _mm_add_pd (y_6, ax_temp );
			ax_temp = _mm_mul_pd( a_03_13, x_0_1 );	
			y_7 = _mm_add_pd (y_7, ax_temp );
		
			A += 2;
			x += 2;
			k += 2;
		
			}

		A += (sda-1)*lda;
		}

	k = lda*(ka/lda);
	tA = A + (ka/lda)*sda*lda;
	tx = x + (ka/lda)*lda;

	if(ka-k>0) // it can be only ka-k = {1, 2, 3}
		{
		if((ka-k)>=2)
			{
		
			x_0_1 = _mm_load_pd( &tx[0] );

			a_00_10 = _mm_load_pd( &tA[0+lda*0] );
			a_01_11 = _mm_load_pd( &tA[0+lda*1] );
			a_02_12 = _mm_load_pd( &tA[0+lda*2] );
			a_03_13 = _mm_load_pd( &tA[0+lda*3] );

			ax_temp = _mm_mul_pd( a_00_10, x_0_1 );	
			y_0 = _mm_add_pd (y_0, ax_temp );
			ax_temp = _mm_mul_pd( a_01_11, x_0_1 );	
			y_1 = _mm_add_pd (y_1, ax_temp );
			ax_temp = _mm_mul_pd( a_02_12, x_0_1 );	
			y_2 = _mm_add_pd (y_2, ax_temp );
			ax_temp = _mm_mul_pd( a_03_13, x_0_1 );	
			y_3 = _mm_add_pd (y_3, ax_temp );
		
			a_00_10 = _mm_load_pd( &tA[0+lda*4] );
			a_01_11 = _mm_load_pd( &tA[0+lda*5] );
			a_02_12 = _mm_load_pd( &tA[0+lda*6] );
			a_03_13 = _mm_load_pd( &tA[0+lda*7] );

			ax_temp = _mm_mul_pd( a_00_10, x_0_1 );	
			y_4 = _mm_add_pd (y_4, ax_temp );
			ax_temp = _mm_mul_pd( a_01_11, x_0_1 );	
			y_5 = _mm_add_pd (y_5, ax_temp );
			ax_temp = _mm_mul_pd( a_02_12, x_0_1 );	
			y_6 = _mm_add_pd (y_6, ax_temp );
			ax_temp = _mm_mul_pd( a_03_13, x_0_1 );	
			y_7 = _mm_add_pd (y_7, ax_temp );
		
			tA += 2;
			tx += 2;
			k+=2;
		
			}

		if((ka-k)==1)
			{
		
			x_0_1 = _mm_load_sd( &tx[0] );

			a_00_10 = _mm_load_sd( &tA[0+lda*0] );
			a_01_11 = _mm_load_sd( &tA[0+lda*1] );
			a_02_12 = _mm_load_sd( &tA[0+lda*2] );
			a_03_13 = _mm_load_sd( &tA[0+lda*3] );

			ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
			y_0 = _mm_add_sd (y_0, ax_temp );
			ax_temp = _mm_mul_sd( a_01_11, x_0_1 );	
			y_1 = _mm_add_sd (y_1, ax_temp );
			ax_temp = _mm_mul_sd( a_02_12, x_0_1 );	
			y_2 = _mm_add_sd (y_2, ax_temp );
			ax_temp = _mm_mul_sd( a_03_13, x_0_1 );	
			y_3 = _mm_add_sd (y_3, ax_temp );
		
			a_00_10 = _mm_load_sd( &tA[0+lda*4] );
			a_01_11 = _mm_load_sd( &tA[0+lda*5] );
			a_02_12 = _mm_load_sd( &tA[0+lda*6] );
			a_03_13 = _mm_load_sd( &tA[0+lda*7] );

			ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
			y_4 = _mm_add_sd (y_4, ax_temp );
			ax_temp = _mm_mul_sd( a_01_11, x_0_1 );	
			y_5 = _mm_add_sd (y_5, ax_temp );
			ax_temp = _mm_mul_sd( a_02_12, x_0_1 );	
			y_6 = _mm_add_sd (y_6, ax_temp );
			ax_temp = _mm_mul_sd( a_03_13, x_0_1 );	
			y_7 = _mm_add_sd (y_7, ax_temp );
		
			tA += 1;
			tx += 1;
			k++;
		
			}

		}

	y_00 = _mm256_castpd128_pd256(y_0);
	y_11 = _mm256_castpd128_pd256(y_1);
	y_22 = _mm256_castpd128_pd256(y_2);
	y_33 = _mm256_castpd128_pd256(y_3);
	y_44 = _mm256_castpd128_pd256(y_4);
	y_55 = _mm256_castpd128_pd256(y_5);
	y_66 = _mm256_castpd128_pd256(y_6);
	y_77 = _mm256_castpd128_pd256(y_7);
		
	k=0;
	for(; k<ka-7; k+=8)
		{

		__builtin_prefetch( A + sda*lda + 0*lda );
		__builtin_prefetch( A + sda*lda + 2*lda );

		x_0_1_2_3 = _mm256_loadu_pd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*2] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*3] );
	
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_00 = _mm256_add_pd( y_00, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_11 = _mm256_add_pd( y_11, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_22 = _mm256_add_pd( y_22, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_33 = _mm256_add_pd( y_33, aaxx_temp );
	
		__builtin_prefetch( A + sda*lda + 4*lda );
		__builtin_prefetch( A + sda*lda + 6*lda );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*4] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*5] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*6] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*7] );
	
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_44 = _mm256_add_pd( y_44, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_55 = _mm256_add_pd( y_55, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_66 = _mm256_add_pd( y_66, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_77 = _mm256_add_pd( y_77, aaxx_temp );

		A += 4 + (sda-1)*lda;
		x += 4;


		__builtin_prefetch( A + sda*lda + 0*lda );
		__builtin_prefetch( A + sda*lda + 2*lda );

		x_0_1_2_3 = _mm256_loadu_pd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*2] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*3] );
	
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_00 = _mm256_add_pd( y_00, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_11 = _mm256_add_pd( y_11, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_22 = _mm256_add_pd( y_22, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_33 = _mm256_add_pd( y_33, aaxx_temp );
	
		__builtin_prefetch( A + sda*lda + 4*lda );
		__builtin_prefetch( A + sda*lda + 6*lda );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*4] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*5] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*6] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*7] );
	
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_44 = _mm256_add_pd( y_44, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_55 = _mm256_add_pd( y_55, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_66 = _mm256_add_pd( y_66, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_77 = _mm256_add_pd( y_77, aaxx_temp );

		A += 4 + (sda-1)*lda;
		x += 4;

		}
	for(; k<ka-3; k+=4)
		{

		__builtin_prefetch( A + sda*lda + 0*lda );
		__builtin_prefetch( A + sda*lda + 2*lda );

		x_0_1_2_3 = _mm256_loadu_pd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*2] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*3] );
	
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_00 = _mm256_add_pd( y_00, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_11 = _mm256_add_pd( y_11, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_22 = _mm256_add_pd( y_22, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_33 = _mm256_add_pd( y_33, aaxx_temp );
	
		__builtin_prefetch( A + sda*lda + 4*lda );
		__builtin_prefetch( A + sda*lda + 6*lda );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*4] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*5] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*6] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*7] );
	
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_44 = _mm256_add_pd( y_44, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_55 = _mm256_add_pd( y_55, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_66 = _mm256_add_pd( y_66, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_77 = _mm256_add_pd( y_77, aaxx_temp );

		A += 4 + (sda-1)*lda;
		x += 4;

		}
		
	__m256d
		y_0_1_2_3, y_4_5_6_7;

	y_00 = _mm256_hadd_pd(y_00, y_11);
	y_22 = _mm256_hadd_pd(y_22, y_33);
	y_44 = _mm256_hadd_pd(y_44, y_55);
	y_66 = _mm256_hadd_pd(y_66, y_77);

	y_11 = _mm256_permute2f128_pd(y_22, y_00, 2 );	
	y_00 = _mm256_permute2f128_pd(y_22, y_00, 19);	
	y_55 = _mm256_permute2f128_pd(y_66, y_44, 2 );	
	y_44 = _mm256_permute2f128_pd(y_66, y_44, 19);	

	y_00 = _mm256_add_pd( y_00, y_11 );
	y_44 = _mm256_add_pd( y_44, y_55 );

	if(alg==0)
		{
		_mm256_storeu_pd(&y[0], y_00);
		_mm256_storeu_pd(&y[4], y_44);
		}
	else if(alg==1)
		{
		y_0_1_2_3 = _mm256_loadu_pd( &y[0] );
		y_4_5_6_7 = _mm256_loadu_pd( &y[4] );

		y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, y_00 );
		y_4_5_6_7 = _mm256_add_pd( y_4_5_6_7, y_44 );

		_mm256_storeu_pd(&y[0], y_0_1_2_3);
		_mm256_storeu_pd(&y[4], y_4_5_6_7);
		}
	else // alg==-1
		{
		y_0_1_2_3 = _mm256_loadu_pd( &y[0] );
		y_4_5_6_7 = _mm256_loadu_pd( &y[4] );
	
		y_0_1_2_3 = _mm256_sub_pd( y_0_1_2_3, y_00 );
		y_4_5_6_7 = _mm256_sub_pd( y_4_5_6_7, y_44 );
	
		_mm256_storeu_pd(&y[0], y_0_1_2_3);
		_mm256_storeu_pd(&y[4], y_4_5_6_7);
		}

	}



void kernel_dgemv_t_4_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	__builtin_prefetch( A + 0*lda );
	__builtin_prefetch( A + 2*lda );

	double *tA, *tx;

	int k;
	int ka = kmax-kna; // number from aligned positon
	
	__m256d
		aaxx_temp,
		a_00_10_20_30, a_01_11_21_31, a_02_12_22_32, a_03_13_23_33,
		x_0_1_2_3,
		y_00, y_11, y_22, y_33;
	
	__m128d
		ax_temp,
		a_00_10, a_01_11, a_02_12, a_03_13,
		x_0_1,
		y_0, y_1, y_2, y_3;
	
	y_00 = _mm256_setzero_pd();
	y_11 = _mm256_setzero_pd();
	y_22 = _mm256_setzero_pd();
	y_33 = _mm256_setzero_pd();
	
	y_0 = _mm256_castpd256_pd128(y_00);
	y_1 = _mm256_castpd256_pd128(y_11);
	y_2 = _mm256_castpd256_pd128(y_22);
	y_3 = _mm256_castpd256_pd128(y_33);

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

			ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
			y_0 = _mm_add_sd (y_0, ax_temp );
			ax_temp = _mm_mul_sd( a_01_11, x_0_1 );	
			y_1 = _mm_add_sd (y_1, ax_temp );
			ax_temp = _mm_mul_sd( a_02_12, x_0_1 );	
			y_2 = _mm_add_sd (y_2, ax_temp );
			ax_temp = _mm_mul_sd( a_03_13, x_0_1 );	
			y_3 = _mm_add_sd (y_3, ax_temp );
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k = lda*(ka/lda);
	tA = A + (ka/lda)*sda*lda;
	tx = x + (ka/lda)*lda;

	for(; k<ka; k++)
		{
		x_0_1 = _mm_load_sd( &tx[0] );

		a_00_10 = _mm_load_sd( &tA[0+lda*0] );
		a_01_11 = _mm_load_sd( &tA[0+lda*1] );
		a_02_12 = _mm_load_sd( &tA[0+lda*2] );
		a_03_13 = _mm_load_sd( &tA[0+lda*3] );
		
			ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
			y_0 = _mm_add_sd (y_0, ax_temp );
			ax_temp = _mm_mul_sd( a_01_11, x_0_1 );	
			y_1 = _mm_add_sd (y_1, ax_temp );
			ax_temp = _mm_mul_sd( a_02_12, x_0_1 );	
			y_2 = _mm_add_sd (y_2, ax_temp );
			ax_temp = _mm_mul_sd( a_03_13, x_0_1 );	
			y_3 = _mm_add_sd (y_3, ax_temp );
		
		tA += 1;
		tx += 1;

		}

	y_00 = _mm256_castpd128_pd256(y_0);
	y_11 = _mm256_castpd128_pd256(y_1);
	y_22 = _mm256_castpd128_pd256(y_2);
	y_33 = _mm256_castpd128_pd256(y_3);

	k=0;
	for(; k<ka-7; k+=8)
		{
		
		__builtin_prefetch( A + sda*lda + 0*lda );
		__builtin_prefetch( A + sda*lda + 2*lda );

		x_0_1_2_3 = _mm256_loadu_pd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*2] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*3] );
		
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_00 = _mm256_add_pd( y_00, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_11 = _mm256_add_pd( y_11, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_22 = _mm256_add_pd( y_22, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_33 = _mm256_add_pd( y_33, aaxx_temp );
		
		A += 4 + (sda-1)*lda;
		x += 4;


		__builtin_prefetch( A + sda*lda + 0*lda );
		__builtin_prefetch( A + sda*lda + 2*lda );

		x_0_1_2_3 = _mm256_loadu_pd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*2] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*3] );
		
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_00 = _mm256_add_pd( y_00, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_11 = _mm256_add_pd( y_11, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_22 = _mm256_add_pd( y_22, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_33 = _mm256_add_pd( y_33, aaxx_temp );
		
		A += 4 + (sda-1)*lda;
		x += 4;

		}
	for(; k<ka-3; k+=4)
		{
		
		__builtin_prefetch( A + sda*lda + 0*lda );
		__builtin_prefetch( A + sda*lda + 2*lda );

		x_0_1_2_3 = _mm256_loadu_pd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );
		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*2] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*3] );
		
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_00 = _mm256_add_pd( y_00, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_11 = _mm256_add_pd( y_11, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_02_12_22_32, x_0_1_2_3 );
		y_22 = _mm256_add_pd( y_22, aaxx_temp );
		aaxx_temp = _mm256_mul_pd( a_03_13_23_33, x_0_1_2_3 );
		y_33 = _mm256_add_pd( y_33, aaxx_temp );
		
		A += 4 + (sda-1)*lda;
		x += 4;

		}

	__m256d
		y_0_1_2_3;

	y_00 = _mm256_hadd_pd(y_00, y_11);
	y_22 = _mm256_hadd_pd(y_22, y_33);

	y_11 = _mm256_permute2f128_pd(y_22, y_00, 2 );	
	y_00 = _mm256_permute2f128_pd(y_22, y_00, 19);	

	y_00 = _mm256_add_pd( y_00, y_11 );

	if(alg==0)
		{
		_mm256_storeu_pd(&y[0], y_00);
		}
	else if(alg==1)
		{
		y_0_1_2_3 = _mm256_loadu_pd( &y[0] );

		y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, y_00 );

		_mm256_storeu_pd(&y[0], y_0_1_2_3);
		}
	else // alg==-1
		{
		y_0_1_2_3 = _mm256_loadu_pd( &y[0] );
	
		y_0_1_2_3 = _mm256_sub_pd( y_0_1_2_3, y_00 );
	
		_mm256_storeu_pd(&y[0], y_0_1_2_3);
		}

	}



void kernel_dgemv_t_2_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	double *tA, *tx;

	int k;
	int ka = kmax-kna; // number from aligned positon
	
	__m256d
		aaxx_temp,
		a_00_10_20_30, a_01_11_21_31,
		x_0_1_2_3,
		y_00, y_11;
	
	__m128d
		ax_temp,
		a_00_10, a_01_11,
		x_0_1,
		y_0, y_1, y_0_1;
	
	y_00 = _mm256_setzero_pd();
	y_11 = _mm256_setzero_pd();
	
	y_0 = _mm256_castpd256_pd128(y_00);
	y_1 = _mm256_castpd256_pd128(y_11);

	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0_1 = _mm_load_sd( &x[0] );

			a_00_10 = _mm_load_sd( &A[0+lda*0] );
			a_01_11 = _mm_load_sd( &A[0+lda*1] );
		
/*			y_0 += a_00_10 * x_0_1;*/
			ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
			y_0 = _mm_add_sd (y_0, ax_temp );
/*			y_1 += a_01_11 * x_0_1;*/
			ax_temp = _mm_mul_sd( a_01_11, x_0_1 );	
			y_1 = _mm_add_sd (y_1, ax_temp );
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k = lda*(ka/lda);
	tA = A + (ka/lda)*sda*lda;
	tx = x + (ka/lda)*lda;

	for(; k<ka; k++)
		{
		x_0_1 = _mm_load_sd( &tx[0] );

		a_00_10 = _mm_load_sd( &tA[0+lda*0] );
		a_01_11 = _mm_load_sd( &tA[0+lda*1] );
		
/*		y_0 += a_00_10 * x_0_1;*/
		ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
		y_0 = _mm_add_sd (y_0, ax_temp );
/*		y_1 += a_01_11 * x_0_1;*/
		ax_temp = _mm_mul_sd( a_01_11, x_0_1 );	
		y_1 = _mm_add_sd (y_1, ax_temp );
		
		tA += 1;
		tx += 1;

		}

	y_00 = _mm256_castpd128_pd256(y_0);
	y_11 = _mm256_castpd128_pd256(y_1);

	k=0;
	for(; k<ka-3; k+=4)
		{
		
		x_0_1_2_3 = _mm256_loadu_pd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );
		
/*		y_00 += a_00_10_20_30 * x_0_1_2_3;*/
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_00 = _mm256_add_pd( y_00, aaxx_temp );
/*		y_11 += a_01_11_21_31 * x_0_1_2_3;*/
		aaxx_temp = _mm256_mul_pd( a_01_11_21_31, x_0_1_2_3 );
		y_11 = _mm256_add_pd( y_11, aaxx_temp );

		A += 4 + (sda-1)*lda;
		x += 4;

		}

	y_00 = _mm256_hadd_pd(y_00, y_11);

	y_1 = _mm256_extractf128_pd(y_00, 1);
	y_0 = _mm256_castpd256_pd128(y_00);

/*	y_0 += y_1;*/
	y_0 = _mm_add_pd( y_0, y_1 );

	if(alg==0)
		{
		_mm_storeu_pd(&y[0], y_0);
		}
	else if(alg==1)
		{
		y_0_1 = _mm_loadu_pd( &y[0] );

/*		y_0_1 += y_0;*/
		y_0_1 = _mm_add_pd( y_0_1, y_0 );

		_mm_storeu_pd(&y[0], y_0_1);
		}
	else // alg==-1
		{
		y_0_1 = _mm_loadu_pd( &y[0] );
	
/*		y_0_1 -= y_0;*/
		y_0_1 = _mm_sub_pd( y_0_1, y_0 );
	
		_mm_storeu_pd(&y[0], y_0_1);
		}

	}



void kernel_dgemv_t_1_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	double *tA, *tx;

	int k;
	int ka = kmax-kna; // number from aligned positon
	
	__m256d
		aaxx_temp,
		a_00_10_20_30,
		x_0_1_2_3,
		y_00;
	
	__m128d
		ax_temp,
		a_00_10,
		x_0_1,
		y_0, y_1, y_0_1;
	
	y_00 = _mm256_setzero_pd();
	
	y_0 = _mm256_castpd256_pd128(y_00);

	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0_1 = _mm_load_sd( &x[0] );

			a_00_10 = _mm_load_sd( &A[0+lda*0] );
		
/*			y_0 += a_00_10 * x_0_1;*/
			ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
			y_0 = _mm_add_sd (y_0, ax_temp );
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k = lda*(ka/lda);
	tA = A + (ka/lda)*sda*lda;
	tx = x + (ka/lda)*lda;

	for(; k<ka; k++)
		{
		x_0_1 = _mm_load_sd( &tx[0] );

		a_00_10 = _mm_load_sd( &tA[0+lda*0] );
		
/*		y_0 += a_00_10 * x_0_1;*/
		ax_temp = _mm_mul_sd( a_00_10, x_0_1 );	
		y_0 = _mm_add_sd (y_0, ax_temp );
		
		tA += 1;
		tx += 1;

		}

	y_00 = _mm256_castpd128_pd256(y_0);

	k=0;
	for(; k<ka-3; k+=4)
		{
		
		x_0_1_2_3 = _mm256_loadu_pd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		
/*		y_00 += a_00_10_20_30 * x_0_1_2_3;*/
		aaxx_temp = _mm256_mul_pd( a_00_10_20_30, x_0_1_2_3 );
		y_00 = _mm256_add_pd( y_00, aaxx_temp );
		
		A += 4 + (sda-1)*lda;
		x += 4;

		}

	y_00 = _mm256_hadd_pd(y_00, y_00);

	y_1 = _mm256_extractf128_pd(y_00, 1);
	y_0 = _mm256_castpd256_pd128(y_00);

/*	y_0 += y_1;*/
	y_0 = _mm_add_sd( y_0, y_1 );

	if(alg==0)
		{
		_mm_store_sd(&y[0], y_0);
		}
	else if(alg==1)
		{
		y_0_1 = _mm_load_sd( &y[0] );

/*		y_0_1 += y_0;*/
		y_0_1 = _mm_add_sd( y_0_1, y_0 );

		_mm_store_sd(&y[0], y_0_1);
		}
	else // alg==-1
		{
		y_0_1 = _mm_load_sd( &y[0] );
	
/*		y_0_1 -= y_0;*/
		y_0_1 = _mm_sub_sd( y_0_1, y_0 );
	
		_mm_store_sd(&y[0], y_0_1);
		}

	}



// it moves horizontally inside a block
void kernel_dgemv_n_8_lib4(int kmax, double *A0, double *A1, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m256d
		ax_temp,
		a_00_10_20_30, a_01_11_21_31,
		a_40_50_60_70, a_41_51_61_71,
		x_0, x_1,
		y_0_1_2_3, y_0_1_2_3_b, z_0_1_2_3,
		y_4_5_6_7, y_4_5_6_7_b, z_4_5_6_7;
	
	y_0_1_2_3   = _mm256_setzero_pd();	
	y_4_5_6_7   = _mm256_setzero_pd();	
	y_0_1_2_3_b = _mm256_setzero_pd();	
	y_4_5_6_7_b = _mm256_setzero_pd();	

	if(kmax<=64)
		{

		k=0;
		for(; k<kmax-3; k+=4)
			{

/*			__builtin_prefetch( A0 + 4*lda );*/
/*			__builtin_prefetch( A1 + 4*lda );*/

			x_0 = _mm256_broadcast_sd( &x[0] );
			x_1 = _mm256_broadcast_sd( &x[1] );

			a_00_10_20_30 = _mm256_load_pd( &A0[0+lda*0] );
			a_40_50_60_70 = _mm256_load_pd( &A1[0+lda*0] );
			a_01_11_21_31 = _mm256_load_pd( &A0[0+lda*1] );
			a_41_51_61_71 = _mm256_load_pd( &A1[0+lda*1] );

	/*		y_0_1_2_3   += a_00_10_20_30 * x_0;*/
			ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
			y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
	/*		y_4_5_6_7   += a_40_50_60_70 * x_0;*/
			ax_temp = _mm256_mul_pd( a_40_50_60_70, x_0 );
			y_4_5_6_7 = _mm256_add_pd( y_4_5_6_7, ax_temp );
	/*		y_0_1_2_3_b += a_01_11_21_31 * x_1;*/
			ax_temp = _mm256_mul_pd( a_01_11_21_31, x_1 );
			y_0_1_2_3_b = _mm256_add_pd( y_0_1_2_3_b, ax_temp );
	/*		y_4_5_6_7_b += a_41_51_61_71 * x_1;*/
			ax_temp = _mm256_mul_pd( a_41_51_61_71, x_1 );
			y_4_5_6_7_b = _mm256_add_pd( y_4_5_6_7_b, ax_temp );

/*			__builtin_prefetch( A0 + 5*lda );*/
/*			__builtin_prefetch( A1 + 5*lda );*/

			x_0 = _mm256_broadcast_sd( &x[2] );
			x_1 = _mm256_broadcast_sd( &x[3] );

			a_00_10_20_30 = _mm256_load_pd( &A0[0+lda*2] );
			a_40_50_60_70 = _mm256_load_pd( &A1[0+lda*2] );
			a_01_11_21_31 = _mm256_load_pd( &A0[0+lda*3] );
			a_41_51_61_71 = _mm256_load_pd( &A1[0+lda*3] );

	/*		y_0_1_2_3   += a_00_10_20_30 * x_0;*/
			ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
			y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
	/*		y_4_5_6_7   += a_40_50_60_70 * x_0;*/
			ax_temp = _mm256_mul_pd( a_40_50_60_70, x_0 );
			y_4_5_6_7 = _mm256_add_pd( y_4_5_6_7, ax_temp );
	/*		y_0_1_2_3_b += a_01_11_21_31 * x_1;*/
			ax_temp = _mm256_mul_pd( a_01_11_21_31, x_1 );
			y_0_1_2_3_b = _mm256_add_pd( y_0_1_2_3_b, ax_temp );
	/*		y_4_5_6_7_b += a_41_51_61_71 * x_1;*/
			ax_temp = _mm256_mul_pd( a_41_51_61_71, x_1 );
			y_4_5_6_7_b = _mm256_add_pd( y_4_5_6_7_b, ax_temp );
		
			A0 += 4*lda;
			A1 += 4*lda;
			x  += 4;

			}
		
		}
	else
		{

		k=0;
		for(; k<kmax-3; k+=4)
			{

			__builtin_prefetch( A0 + 4*lda );
			__builtin_prefetch( A1 + 4*lda );

			x_0 = _mm256_broadcast_sd( &x[0] );
			x_1 = _mm256_broadcast_sd( &x[1] );

			a_00_10_20_30 = _mm256_load_pd( &A0[0+lda*0] );
			a_40_50_60_70 = _mm256_load_pd( &A1[0+lda*0] );
			a_01_11_21_31 = _mm256_load_pd( &A0[0+lda*1] );
			a_41_51_61_71 = _mm256_load_pd( &A1[0+lda*1] );

	/*		y_0_1_2_3   += a_00_10_20_30 * x_0;*/
			ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
			y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
	/*		y_4_5_6_7   += a_40_50_60_70 * x_0;*/
			ax_temp = _mm256_mul_pd( a_40_50_60_70, x_0 );
			y_4_5_6_7 = _mm256_add_pd( y_4_5_6_7, ax_temp );
	/*		y_0_1_2_3_b += a_01_11_21_31 * x_1;*/
			ax_temp = _mm256_mul_pd( a_01_11_21_31, x_1 );
			y_0_1_2_3_b = _mm256_add_pd( y_0_1_2_3_b, ax_temp );
	/*		y_4_5_6_7_b += a_41_51_61_71 * x_1;*/
			ax_temp = _mm256_mul_pd( a_41_51_61_71, x_1 );
			y_4_5_6_7_b = _mm256_add_pd( y_4_5_6_7_b, ax_temp );

			__builtin_prefetch( A0 + 5*lda );
			__builtin_prefetch( A1 + 5*lda );

			x_0 = _mm256_broadcast_sd( &x[2] );
			x_1 = _mm256_broadcast_sd( &x[3] );

			a_00_10_20_30 = _mm256_load_pd( &A0[0+lda*2] );
			a_40_50_60_70 = _mm256_load_pd( &A1[0+lda*2] );
			a_01_11_21_31 = _mm256_load_pd( &A0[0+lda*3] );
			a_41_51_61_71 = _mm256_load_pd( &A1[0+lda*3] );

	/*		y_0_1_2_3   += a_00_10_20_30 * x_0;*/
			ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
			y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
	/*		y_4_5_6_7   += a_40_50_60_70 * x_0;*/
			ax_temp = _mm256_mul_pd( a_40_50_60_70, x_0 );
			y_4_5_6_7 = _mm256_add_pd( y_4_5_6_7, ax_temp );
	/*		y_0_1_2_3_b += a_01_11_21_31 * x_1;*/
			ax_temp = _mm256_mul_pd( a_01_11_21_31, x_1 );
			y_0_1_2_3_b = _mm256_add_pd( y_0_1_2_3_b, ax_temp );
	/*		y_4_5_6_7_b += a_41_51_61_71 * x_1;*/
			ax_temp = _mm256_mul_pd( a_41_51_61_71, x_1 );
			y_4_5_6_7_b = _mm256_add_pd( y_4_5_6_7_b, ax_temp );
		
			A0 += 4*lda;
			A1 += 4*lda;
			x  += 4;

			}
	
		}


	if(kmax%4>=2)
		{

		x_0 = _mm256_broadcast_sd( &x[0] );
		x_1 = _mm256_broadcast_sd( &x[1] );

		a_00_10_20_30 = _mm256_load_pd( &A0[0+lda*0] );
		a_40_50_60_70 = _mm256_load_pd( &A1[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A0[0+lda*1] );
		a_41_51_61_71 = _mm256_load_pd( &A1[0+lda*1] );

/*		y_0_1_2_3   += a_00_10_20_30 * x_0;*/
		ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
		y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
/*		y_4_5_6_7   += a_40_50_60_70 * x_0;*/
		ax_temp = _mm256_mul_pd( a_40_50_60_70, x_0 );
		y_4_5_6_7 = _mm256_add_pd( y_4_5_6_7, ax_temp );
/*		y_0_1_2_3_b += a_01_11_21_31 * x_1;*/
		ax_temp = _mm256_mul_pd( a_01_11_21_31, x_1 );
		y_0_1_2_3_b = _mm256_add_pd( y_0_1_2_3_b, ax_temp );
/*		y_4_5_6_7_b += a_41_51_61_71 * x_1;*/
		ax_temp = _mm256_mul_pd( a_41_51_61_71, x_1 );
		y_4_5_6_7_b = _mm256_add_pd( y_4_5_6_7_b, ax_temp );
		
		A0 += 2*lda;
		A1 += 2*lda;
		x  += 2;

		}
	
	y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, y_0_1_2_3_b );
	y_4_5_6_7 = _mm256_add_pd( y_4_5_6_7, y_4_5_6_7_b );

	if(kmax%2==1)
		{

		x_0 = _mm256_broadcast_sd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A0[0+lda*0] );
		a_40_50_60_70 = _mm256_load_pd( &A1[0+lda*0] );

/*		y_0_1_2_3   += a_00_10_20_30 * x_0;*/
		ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
		y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
/*		y_4_5_6_7   += a_40_50_60_70 * x_0;*/
		ax_temp = _mm256_mul_pd( a_40_50_60_70, x_0 );
		y_4_5_6_7 = _mm256_add_pd( y_4_5_6_7, ax_temp );
		
/*		A0 += 1*lda;*/
/*		A1 += 1*lda;*/
/*		x  += 1;*/

		}

	if(alg==0)
		{
		_mm256_storeu_pd(&y[0], y_0_1_2_3);
		_mm256_storeu_pd(&y[4], y_4_5_6_7);
		}
	else if(alg==1)
		{
		z_0_1_2_3 = _mm256_loadu_pd( &y[0] );
		z_4_5_6_7 = _mm256_loadu_pd( &y[4] );

/*		z_0_1_2_3 += y_0_1_2_3;*/
		z_0_1_2_3 = _mm256_add_pd( z_0_1_2_3, y_0_1_2_3 );
/*		z_4_5_6_7 += y_4_5_6_7;*/
		z_4_5_6_7 = _mm256_add_pd( z_4_5_6_7, y_4_5_6_7 );

		_mm256_storeu_pd(&y[0], z_0_1_2_3);
		_mm256_storeu_pd(&y[4], z_4_5_6_7);
		}
	else // alg==-1
		{
		z_0_1_2_3 = _mm256_loadu_pd( &y[0] );
		z_4_5_6_7 = _mm256_loadu_pd( &y[4] );

/*		z_0_1_2_3 -= y_0_1_2_3;*/
		z_0_1_2_3 = _mm256_sub_pd( z_0_1_2_3, y_0_1_2_3 );
/*		z_4_5_6_7 -= y_4_5_6_7;*/
		z_4_5_6_7 = _mm256_sub_pd( z_4_5_6_7, y_4_5_6_7 );

		_mm256_storeu_pd(&y[0], z_0_1_2_3);
		_mm256_storeu_pd(&y[4], z_4_5_6_7);
		}

	}



// it moves horizontally inside a block
void kernel_dgemv_n_4_lib4(int kmax, double *A, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m256d
		ax_temp,
		a_00_10_20_30, a_01_11_21_31, a_02_12_22_32, a_03_13_23_33,
		x_0, x_1, x_2, x_3,
		y_0_1_2_3, y_0_1_2_3_b, y_0_1_2_3_c, y_0_1_2_3_d, z_0_1_2_3;
	
	y_0_1_2_3   = _mm256_setzero_pd();	
	y_0_1_2_3_b = _mm256_setzero_pd();	
	y_0_1_2_3_c = _mm256_setzero_pd();	
	y_0_1_2_3_d = _mm256_setzero_pd();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm256_broadcast_sd( &x[0] );
		x_1 = _mm256_broadcast_sd( &x[1] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );

		x_2 = _mm256_broadcast_sd( &x[2] );
		x_3 = _mm256_broadcast_sd( &x[3] );

		a_02_12_22_32 = _mm256_load_pd( &A[0+lda*2] );
		a_03_13_23_33 = _mm256_load_pd( &A[0+lda*3] );

		ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
		y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
		ax_temp = _mm256_mul_pd( a_01_11_21_31, x_1 );
		y_0_1_2_3_b = _mm256_add_pd( y_0_1_2_3_b, ax_temp );

		ax_temp = _mm256_mul_pd( a_02_12_22_32, x_2 );
		y_0_1_2_3_c = _mm256_add_pd( y_0_1_2_3_c, ax_temp );
		ax_temp = _mm256_mul_pd( a_03_13_23_33, x_3 );
		y_0_1_2_3_d = _mm256_add_pd( y_0_1_2_3_d, ax_temp );
		
		A += 4*lda;
		x += 4;

		}
	
	y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, y_0_1_2_3_c );
	y_0_1_2_3_b = _mm256_add_pd( y_0_1_2_3_b, y_0_1_2_3_d );

	if(kmax%4>=2)
		{

		x_0 = _mm256_broadcast_sd( &x[0] );
		x_1 = _mm256_broadcast_sd( &x[1] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );
		a_01_11_21_31 = _mm256_load_pd( &A[0+lda*1] );

		ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
		y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
		ax_temp = _mm256_mul_pd( a_01_11_21_31, x_1 );
		y_0_1_2_3_b = _mm256_add_pd( y_0_1_2_3_b, ax_temp );

		A += 2*lda;
		x += 2;

		}

	y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, y_0_1_2_3_b );

	if(kmax%2==1)
		{

		x_0 = _mm256_broadcast_sd( &x[0] );

		a_00_10_20_30 = _mm256_load_pd( &A[0+lda*0] );

		ax_temp = _mm256_mul_pd( a_00_10_20_30, x_0 );
		y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, ax_temp );
		
/*		A += 1*lda;*/
/*		x += 1;*/

		}

	if(alg==0)
		{
		_mm256_storeu_pd(&y[0], y_0_1_2_3);
		}
	else if(alg==1)
		{
		z_0_1_2_3 = _mm256_loadu_pd( &y[0] );

		z_0_1_2_3 = _mm256_add_pd ( z_0_1_2_3, y_0_1_2_3 );

		_mm256_storeu_pd(&y[0], z_0_1_2_3);
		}
	else // alg==-1
		{
		z_0_1_2_3 = _mm256_loadu_pd( &y[0] );

		z_0_1_2_3 = _mm256_sub_pd ( z_0_1_2_3, y_0_1_2_3 );

		_mm256_storeu_pd(&y[0], z_0_1_2_3);
		}

	}



// it moves horizontally inside a block
void kernel_dgemv_n_2_lib4(int kmax, double *A, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128d
		ax_temp,
		a_00_10, a_01_11, a_02_12, a_03_13,
		x_0, x_1, x_2, x_3,
		y_0_1, y_0_1_b, y_0_1_c, y_0_1_d, z_0_1;
	
	y_0_1 = _mm_setzero_pd();	
	y_0_1_b = _mm_setzero_pd();	
	y_0_1_c = _mm_setzero_pd();	
	y_0_1_d = _mm_setzero_pd();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_loaddup_pd( &x[0] );
		x_1 = _mm_loaddup_pd( &x[1] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		a_01_11 = _mm_load_pd( &A[0+lda*1] );

		x_2 = _mm_loaddup_pd( &x[2] );
		x_3 = _mm_loaddup_pd( &x[3] );

		a_02_12 = _mm_load_pd( &A[0+lda*2] );
		a_03_13 = _mm_load_pd( &A[0+lda*3] );

/*		y_0_1 += a_00_10 * x_0;*/
		ax_temp = _mm_mul_pd( a_00_10, x_0 );
		y_0_1 = _mm_add_pd( y_0_1, ax_temp );
/*		y_0_1 += a_01_11 * x_1;*/
		ax_temp = _mm_mul_pd( a_01_11, x_1 );
		y_0_1_b = _mm_add_pd( y_0_1_b, ax_temp );

/*		y_0_1 += a_02_12 * x_2;*/
		ax_temp = _mm_mul_pd( a_02_12, x_2 );
		y_0_1_c = _mm_add_pd( y_0_1_c, ax_temp );
/*		y_0_1 += a_03_13 * x_3;*/
		ax_temp = _mm_mul_pd( a_03_13, x_3 );
		y_0_1_d = _mm_add_pd( y_0_1_d, ax_temp );
		
		A += 4*lda;
		x += 4;

		}

	y_0_1 = _mm_add_pd( y_0_1, y_0_1_c );
	y_0_1_b = _mm_add_pd( y_0_1_b, y_0_1_d );

	if(kmax%4>=2)
		{

		x_0 = _mm_loaddup_pd( &x[0] );
		x_1 = _mm_loaddup_pd( &x[1] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );
		a_01_11 = _mm_load_pd( &A[0+lda*1] );

/*		y_0_1 += a_00_10 * x_0;*/
		ax_temp = _mm_mul_pd( a_00_10, x_0 );
		y_0_1 = _mm_add_pd( y_0_1, ax_temp );
/*		y_0_1 += a_01_11 * x_1;*/
		ax_temp = _mm_mul_pd( a_01_11, x_1 );
		y_0_1_b = _mm_add_pd( y_0_1_b, ax_temp );

		A += 2*lda;
		x += 2;

		}

	y_0_1 = _mm_add_pd( y_0_1, y_0_1_b );

	if(kmax%2==1)
		{

		x_0 = _mm_loaddup_pd( &x[0] );

		a_00_10 = _mm_load_pd( &A[0+lda*0] );

/*		y_0_1 += a_00_10 * x_0;*/
		ax_temp = _mm_mul_pd( a_00_10, x_0 );
		y_0_1 = _mm_add_pd( y_0_1, ax_temp );

/*		A += 1*lda;*/
/*		x += 1;*/

		}

	if(alg==0)
		{
		_mm_storeu_pd(&y[0], y_0_1);
		}
	else if(alg==1)
		{
		z_0_1 = _mm_loadu_pd( &y[0] );

/*		z_0_1 += y_0_1;*/
		z_0_1 = _mm_add_pd( z_0_1, y_0_1 );

		_mm_storeu_pd(&y[0], z_0_1);
		}
	else // alg==-1
		{
		z_0_1 = _mm_loadu_pd( &y[0] );

/*		z_0_1 -= y_0_1;*/
		z_0_1 = _mm_sub_pd( z_0_1, y_0_1 );

		_mm_storeu_pd(&y[0], z_0_1);
		}

	}



// it moves horizontally inside a block
void kernel_dgemv_n_1_lib4(int kmax, double *A, double *x, double *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128d
		ax_temp,
		a_00, a_01, a_02, a_03,
		x_0, x_1, x_2, x_3,
		y_0, y_0_b, y_0_c, y_0_d, z_0;
	
	y_0 = _mm_setzero_pd();	
	y_0_b = _mm_setzero_pd();	
	y_0_c = _mm_setzero_pd();	
	y_0_d = _mm_setzero_pd();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_load_sd( &x[0] );
		x_1 = _mm_load_sd( &x[1] );

		a_00 = _mm_load_sd( &A[0+lda*0] );
		a_01 = _mm_load_sd( &A[0+lda*1] );

		x_2 = _mm_load_sd( &x[2] );
		x_3 = _mm_load_sd( &x[3] );

		a_02 = _mm_load_sd( &A[0+lda*2] );
		a_03 = _mm_load_sd( &A[0+lda*3] );

/*		y_0 += a_00 * x_0;*/
		ax_temp = _mm_mul_sd( a_00, x_0 );
		y_0 = _mm_add_sd( y_0, ax_temp );
/*		y_0 += a_01 * x_1;*/
		ax_temp = _mm_mul_sd( a_01, x_1 );
		y_0_b = _mm_add_sd( y_0_b, ax_temp );

/*		y_0 += a_02 * x_2;*/
		ax_temp = _mm_mul_sd( a_02, x_2 );
		y_0_c = _mm_add_sd( y_0_c, ax_temp );
/*		y_0 += a_03 * x_3;*/
		ax_temp = _mm_mul_sd( a_03, x_3 );
		y_0_d = _mm_add_sd( y_0_d, ax_temp );
		
		A += 4*lda;
		x += 4;

		}

	y_0 = _mm_add_pd( y_0, y_0_c );
	y_0_b = _mm_add_pd( y_0_b, y_0_d );

	if(kmax%4>=2)
		{

		x_0 = _mm_load_sd( &x[0] );
		x_1 = _mm_load_sd( &x[1] );

		a_00 = _mm_load_sd( &A[0+lda*0] );
		a_01 = _mm_load_sd( &A[0+lda*1] );

/*		y_0 += a_00 * x_0;*/
		ax_temp = _mm_mul_sd( a_00, x_0 );
		y_0 = _mm_add_sd( y_0, ax_temp );
/*		y_0 += a_01 * x_1;*/
		ax_temp = _mm_mul_sd( a_01, x_1 );
		y_0_b = _mm_add_sd( y_0_b, ax_temp );

		A += 2*lda;
		x += 2;

		}

	y_0 = _mm_add_pd( y_0, y_0_b );

	if(kmax%2==1)
		{

		x_0 = _mm_load_sd( &x[0] );

		a_00 = _mm_load_sd( &A[0+lda*0] );

/*		y_0 += a_00 * x_0;*/
		ax_temp = _mm_mul_sd( a_00, x_0 );
		y_0 = _mm_add_sd( y_0, ax_temp );
		
/*		A += 1*lda;*/
/*		x += 1;*/

		}

	if(alg==0)
		{
		_mm_store_sd(&y[0], y_0);
		}
	else if(alg==1)
		{
		z_0 = _mm_load_sd( &y[0] );

/*		z_0 += y_0;*/
		z_0 = _mm_add_sd( z_0, y_0 );

		_mm_store_sd(&y[0], z_0);
		}
	else // alg==-1
		{
		z_0 = _mm_load_sd( &y[0] );

/*		z_0 -= y_0;*/
		z_0 = _mm_sub_sd( z_0, y_0 );

		_mm_store_sd(&y[0], z_0);
		}

	}

