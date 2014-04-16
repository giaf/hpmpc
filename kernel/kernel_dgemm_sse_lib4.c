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



// 4x4 with data packed in 4
void kernel_dgemm_pp_nt_4x4_sse_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int ldc = 4;*/

	int k;
	
	__m128d
		c_00_11, c_01_10, c_02_13, c_03_12, c_20_31, c_21_30, c_22_33, c_23_32,
		a_01, a_23,
		b_01, b_10, b_23, b_32, b_temp_0, b_temp_1;
	
	c_00_11 = _mm_setzero_pd();
	c_01_10 = _mm_setzero_pd();
	c_02_13 = _mm_setzero_pd();
	c_03_12 = _mm_setzero_pd();
	c_20_31 = _mm_setzero_pd();
	c_21_30 = _mm_setzero_pd();
	c_22_33 = _mm_setzero_pd();
	c_23_32 = _mm_setzero_pd();
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{
		a_01 = _mm_load_pd(&A[0]);
		a_23 = _mm_load_pd(&A[2]);
		
		b_01 = _mm_load_pd(&B[0]);
		b_23 = _mm_load_pd(&B[2]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
		b_32 = _mm_shuffle_pd(b_23, b_23, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		
		b_temp_0 = b_23;
		b_23 = _mm_mul_pd( a_01, b_23 );
		c_02_13 = _mm_add_pd( c_02_13, b_23 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_22_33 = _mm_add_pd( c_22_33, b_temp_0 );

		b_temp_1 = b_32;
		b_32 = _mm_mul_pd( a_01, b_32 );
		c_03_12 = _mm_add_pd( c_03_12, b_32 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_23_32 = _mm_add_pd( c_23_32, b_temp_1 );
		
		
		a_01 = _mm_load_pd(&A[4]);
		a_23 = _mm_load_pd(&A[6]);
		
		b_01 = _mm_load_pd(&B[4]);
		b_23 = _mm_load_pd(&B[6]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
		b_32 = _mm_shuffle_pd(b_23, b_23, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		
		b_temp_0 = b_23;
		b_23 = _mm_mul_pd( a_01, b_23 );
		c_02_13 = _mm_add_pd( c_02_13, b_23 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_22_33 = _mm_add_pd( c_22_33, b_temp_0 );

		b_temp_1 = b_32;
		b_32 = _mm_mul_pd( a_01, b_32 );
		c_03_12 = _mm_add_pd( c_03_12, b_32 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_23_32 = _mm_add_pd( c_23_32, b_temp_1 );
		
		
		a_01 = _mm_load_pd(&A[8]);
		a_23 = _mm_load_pd(&A[10]);
		
		b_01 = _mm_load_pd(&B[8]);
		b_23 = _mm_load_pd(&B[10]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
		b_32 = _mm_shuffle_pd(b_23, b_23, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		
		b_temp_0 = b_23;
		b_23 = _mm_mul_pd( a_01, b_23 );
		c_02_13 = _mm_add_pd( c_02_13, b_23 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_22_33 = _mm_add_pd( c_22_33, b_temp_0 );

		b_temp_1 = b_32;
		b_32 = _mm_mul_pd( a_01, b_32 );
		c_03_12 = _mm_add_pd( c_03_12, b_32 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_23_32 = _mm_add_pd( c_23_32, b_temp_1 );
		
		
		a_01 = _mm_load_pd(&A[12]);
		a_23 = _mm_load_pd(&A[14]);
		
		b_01 = _mm_load_pd(&B[12]);
		b_23 = _mm_load_pd(&B[14]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
		b_32 = _mm_shuffle_pd(b_23, b_23, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		
		b_temp_0 = b_23;
		b_23 = _mm_mul_pd( a_01, b_23 );
		c_02_13 = _mm_add_pd( c_02_13, b_23 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_22_33 = _mm_add_pd( c_22_33, b_temp_0 );

		b_temp_1 = b_32;
		b_32 = _mm_mul_pd( a_01, b_32 );
		c_03_12 = _mm_add_pd( c_03_12, b_32 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_23_32 = _mm_add_pd( c_23_32, b_temp_1 );

		
		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		a_01 = _mm_load_pd(&A[0]);
		a_23 = _mm_load_pd(&A[2]);
		
		b_01 = _mm_load_pd(&B[0]);
		b_23 = _mm_load_pd(&B[2]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
		b_32 = _mm_shuffle_pd(b_23, b_23, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		
		b_temp_0 = b_23;
		b_23 = _mm_mul_pd( a_01, b_23 );
		c_02_13 = _mm_add_pd( c_02_13, b_23 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_22_33 = _mm_add_pd( c_22_33, b_temp_0 );

		b_temp_1 = b_32;
		b_32 = _mm_mul_pd( a_01, b_32 );
		c_03_12 = _mm_add_pd( c_03_12, b_32 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_23_32 = _mm_add_pd( c_23_32, b_temp_1 );
		

		A += 4;
		B += 4;

		}

	__m128d
		c_00_10, c_20_30, c_01_11, c_21_31, c_02_12, c_22_32, c_03_13, c_23_33,
		d_00_10, d_20_30, d_01_11, d_21_31, d_02_12, d_22_32, d_03_13, d_23_33;

	c_00_10 = _mm_blend_pd(c_00_11, c_01_10, 2);
	c_01_11 = _mm_blend_pd(c_01_10, c_00_11, 2);
	c_02_12 = _mm_blend_pd(c_02_13, c_03_12, 2);
	c_03_13 = _mm_blend_pd(c_03_12, c_02_13, 2);
	c_20_30 = _mm_blend_pd(c_20_31, c_21_30, 2);
	c_21_31 = _mm_blend_pd(c_21_30, c_20_31, 2);
	c_22_32 = _mm_blend_pd(c_22_33, c_23_32, 2);
	c_23_33 = _mm_blend_pd(c_23_32, c_22_33, 2);

	if(alg==0)
		{
		_mm_store_pd(&C[0+ldc*0], c_00_10);
		_mm_store_pd(&C[2+ldc*0], c_20_30);
		_mm_store_pd(&C[0+ldc*1], c_01_11);
		_mm_store_pd(&C[2+ldc*1], c_21_31);
		_mm_store_pd(&C[0+ldc*2], c_02_12);
		_mm_store_pd(&C[2+ldc*2], c_22_32);
		_mm_store_pd(&C[0+ldc*3], c_03_13);
		_mm_store_pd(&C[2+ldc*3], c_23_33);
		}
	else if(alg==1)
		{
		d_00_10 = _mm_load_pd(&C[0+ldc*0]);
		d_20_30 = _mm_load_pd(&C[2+ldc*0]);
		d_01_11 = _mm_load_pd(&C[0+ldc*1]);
		d_21_31 = _mm_load_pd(&C[2+ldc*1]);
		d_02_12 = _mm_load_pd(&C[0+ldc*2]);
		d_22_32 = _mm_load_pd(&C[2+ldc*2]);
		d_03_13 = _mm_load_pd(&C[0+ldc*3]);
		d_23_33 = _mm_load_pd(&C[2+ldc*3]);
		
		d_00_10 = _mm_add_pd( d_00_10, c_00_10 ); 
		d_01_11 = _mm_add_pd( d_01_11, c_01_11 ); 
		d_02_12 = _mm_add_pd( d_02_12, c_02_12 ); 
		d_03_13 = _mm_add_pd( d_03_13, c_03_13 );
		d_20_30 = _mm_add_pd( d_20_30, c_20_30 ); 
		d_21_31 = _mm_add_pd( d_21_31, c_21_31 ); 
		d_22_32 = _mm_add_pd( d_22_32, c_22_32 ); 
		d_23_33 = _mm_add_pd( d_23_33, c_23_33 );

		_mm_store_pd(&C[0+ldc*0], d_00_10);
		_mm_store_pd(&C[2+ldc*0], d_20_30);
		_mm_store_pd(&C[0+ldc*1], d_01_11);
		_mm_store_pd(&C[2+ldc*1], d_21_31);
		_mm_store_pd(&C[0+ldc*2], d_02_12);
		_mm_store_pd(&C[2+ldc*2], d_22_32);
		_mm_store_pd(&C[0+ldc*3], d_03_13);
		_mm_store_pd(&C[2+ldc*3], d_23_33);
		}
	else
		{
		d_00_10 = _mm_load_pd(&C[0+ldc*0]);
		d_20_30 = _mm_load_pd(&C[2+ldc*0]);
		d_01_11 = _mm_load_pd(&C[0+ldc*1]);
		d_21_31 = _mm_load_pd(&C[2+ldc*1]);
		d_02_12 = _mm_load_pd(&C[0+ldc*2]);
		d_22_32 = _mm_load_pd(&C[2+ldc*2]);
		d_03_13 = _mm_load_pd(&C[0+ldc*3]);
		d_23_33 = _mm_load_pd(&C[2+ldc*3]);
		
		d_00_10 = _mm_sub_pd( d_00_10, c_00_10 ); 
		d_01_11 = _mm_sub_pd( d_01_11, c_01_11 ); 
		d_02_12 = _mm_sub_pd( d_02_12, c_02_12 ); 
		d_03_13 = _mm_sub_pd( d_03_13, c_03_13 );
		d_20_30 = _mm_sub_pd( d_20_30, c_20_30 ); 
		d_21_31 = _mm_sub_pd( d_21_31, c_21_31 ); 
		d_22_32 = _mm_sub_pd( d_22_32, c_22_32 ); 
		d_23_33 = _mm_sub_pd( d_23_33, c_23_33 );

		_mm_store_pd(&C[0+ldc*0], d_00_10);
		_mm_store_pd(&C[2+ldc*0], d_20_30);
		_mm_store_pd(&C[0+ldc*1], d_01_11);
		_mm_store_pd(&C[2+ldc*1], d_21_31);
		_mm_store_pd(&C[0+ldc*2], d_02_12);
		_mm_store_pd(&C[2+ldc*2], d_22_32);
		_mm_store_pd(&C[0+ldc*3], d_03_13);
		_mm_store_pd(&C[2+ldc*3], d_23_33);
		}

	}



// 4x3 with data packed in 4
void kernel_dgemm_pp_nt_4x3_sse_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int ldc = 4;*/

	int k;
	
	__m128d
		c_00_10, c_01_11, c_02_12, c_20_30, c_21_31, c_22_32,
		a_01, a_23,
		b_0, b_1, b_2, b_temp;
	
	c_00_10 = _mm_setzero_pd();
	c_01_11 = _mm_setzero_pd();
	c_02_12 = _mm_setzero_pd();
	c_20_30 = _mm_setzero_pd();
	c_21_31 = _mm_setzero_pd();
	c_22_32 = _mm_setzero_pd();
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{
		a_01 = _mm_load_pd(&A[0]);
		a_23 = _mm_load_pd(&A[2]);
		
		b_0 = _mm_loaddup_pd(&B[0]);
		b_1 = _mm_loaddup_pd(&B[1]);
		b_2 = _mm_loaddup_pd(&B[2]);

		b_temp = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10 = _mm_add_pd( c_00_10, b_0 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_20_30 = _mm_add_pd( c_20_30, b_temp );

		b_temp = b_1;
		b_1 = _mm_mul_pd( a_01, b_1 );
		c_01_11 = _mm_add_pd( c_01_11, b_1 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_21_31 = _mm_add_pd( c_21_31, b_temp );

		b_temp = b_2;
		b_2 = _mm_mul_pd( a_01, b_2 );
		c_02_12 = _mm_add_pd( c_02_12, b_2 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_22_32 = _mm_add_pd( c_22_32, b_temp );
		
		
		a_01 = _mm_load_pd(&A[4]);
		a_23 = _mm_load_pd(&A[6]);
		
		b_0 = _mm_loaddup_pd(&B[4]);
		b_1 = _mm_loaddup_pd(&B[5]);
		b_2 = _mm_loaddup_pd(&B[6]);

		b_temp = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10 = _mm_add_pd( c_00_10, b_0 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_20_30 = _mm_add_pd( c_20_30, b_temp );

		b_temp = b_1;
		b_1 = _mm_mul_pd( a_01, b_1 );
		c_01_11 = _mm_add_pd( c_01_11, b_1 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_21_31 = _mm_add_pd( c_21_31, b_temp );

		b_temp = b_2;
		b_2 = _mm_mul_pd( a_01, b_2 );
		c_02_12 = _mm_add_pd( c_02_12, b_2 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_22_32 = _mm_add_pd( c_22_32, b_temp );
		
		
		a_01 = _mm_load_pd(&A[8]);
		a_23 = _mm_load_pd(&A[10]);
		
		b_0 = _mm_loaddup_pd(&B[8]);
		b_1 = _mm_loaddup_pd(&B[9]);
		b_2 = _mm_loaddup_pd(&B[10]);

		b_temp = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10 = _mm_add_pd( c_00_10, b_0 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_20_30 = _mm_add_pd( c_20_30, b_temp );

		b_temp = b_1;
		b_1 = _mm_mul_pd( a_01, b_1 );
		c_01_11 = _mm_add_pd( c_01_11, b_1 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_21_31 = _mm_add_pd( c_21_31, b_temp );

		b_temp = b_2;
		b_2 = _mm_mul_pd( a_01, b_2 );
		c_02_12 = _mm_add_pd( c_02_12, b_2 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_22_32 = _mm_add_pd( c_22_32, b_temp );
		
		
		a_01 = _mm_load_pd(&A[12]);
		a_23 = _mm_load_pd(&A[14]);
		
		b_0 = _mm_loaddup_pd(&B[12]);
		b_1 = _mm_loaddup_pd(&B[13]);
		b_2 = _mm_loaddup_pd(&B[14]);

		b_temp = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10 = _mm_add_pd( c_00_10, b_0 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_20_30 = _mm_add_pd( c_20_30, b_temp );

		b_temp = b_1;
		b_1 = _mm_mul_pd( a_01, b_1 );
		c_01_11 = _mm_add_pd( c_01_11, b_1 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_21_31 = _mm_add_pd( c_21_31, b_temp );

		b_temp = b_2;
		b_2 = _mm_mul_pd( a_01, b_2 );
		c_02_12 = _mm_add_pd( c_02_12, b_2 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_22_32 = _mm_add_pd( c_22_32, b_temp );

		
		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		a_01 = _mm_load_pd(&A[0]);
		a_23 = _mm_load_pd(&A[2]);
		
		b_0 = _mm_loaddup_pd(&B[0]);
		b_1 = _mm_loaddup_pd(&B[1]);
		b_2 = _mm_loaddup_pd(&B[2]);

		b_temp = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10 = _mm_add_pd( c_00_10, b_0 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_20_30 = _mm_add_pd( c_20_30, b_temp );

		b_temp = b_1;
		b_1 = _mm_mul_pd( a_01, b_1 );
		c_01_11 = _mm_add_pd( c_01_11, b_1 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_21_31 = _mm_add_pd( c_21_31, b_temp );

		b_temp = b_2;
		b_2 = _mm_mul_pd( a_01, b_2 );
		c_02_12 = _mm_add_pd( c_02_12, b_2 );
		b_temp = _mm_mul_pd( a_23, b_temp );
		c_22_32 = _mm_add_pd( c_22_32, b_temp );
		

		A += 4;
		B += 4;

		}

	__m128d
		d_00_10, d_20_30, d_01_11, d_21_31, d_02_12, d_22_32;

	if(alg==0)
		{
		_mm_store_pd(&C[0+ldc*0], c_00_10);
		_mm_store_pd(&C[2+ldc*0], c_20_30);
		_mm_store_pd(&C[0+ldc*1], c_01_11);
		_mm_store_pd(&C[2+ldc*1], c_21_31);
		_mm_store_pd(&C[0+ldc*2], c_02_12);
		_mm_store_pd(&C[2+ldc*2], c_22_32);
		}
	else if(alg==1)
		{
		d_00_10 = _mm_load_pd(&C[0+ldc*0]);
		d_20_30 = _mm_load_pd(&C[2+ldc*0]);
		d_01_11 = _mm_load_pd(&C[0+ldc*1]);
		d_21_31 = _mm_load_pd(&C[2+ldc*1]);
		d_02_12 = _mm_load_pd(&C[0+ldc*2]);
		d_22_32 = _mm_load_pd(&C[2+ldc*2]);
		
		d_00_10 = _mm_add_pd( d_00_10, c_00_10 ); 
		d_01_11 = _mm_add_pd( d_01_11, c_01_11 ); 
		d_02_12 = _mm_add_pd( d_02_12, c_02_12 ); 
		d_20_30 = _mm_add_pd( d_20_30, c_20_30 ); 
		d_21_31 = _mm_add_pd( d_21_31, c_21_31 ); 
		d_22_32 = _mm_add_pd( d_22_32, c_22_32 ); 

		_mm_store_pd(&C[0+ldc*0], d_00_10);
		_mm_store_pd(&C[2+ldc*0], d_20_30);
		_mm_store_pd(&C[0+ldc*1], d_01_11);
		_mm_store_pd(&C[2+ldc*1], d_21_31);
		_mm_store_pd(&C[0+ldc*2], d_02_12);
		_mm_store_pd(&C[2+ldc*2], d_22_32);
		}
	else
		{
		d_00_10 = _mm_load_pd(&C[0+ldc*0]);
		d_20_30 = _mm_load_pd(&C[2+ldc*0]);
		d_01_11 = _mm_load_pd(&C[0+ldc*1]);
		d_21_31 = _mm_load_pd(&C[2+ldc*1]);
		d_02_12 = _mm_load_pd(&C[0+ldc*2]);
		d_22_32 = _mm_load_pd(&C[2+ldc*2]);
		
		d_00_10 = _mm_sub_pd( d_00_10, c_00_10 ); 
		d_01_11 = _mm_sub_pd( d_01_11, c_01_11 ); 
		d_02_12 = _mm_sub_pd( d_02_12, c_02_12 ); 
		d_20_30 = _mm_sub_pd( d_20_30, c_20_30 ); 
		d_21_31 = _mm_sub_pd( d_21_31, c_21_31 ); 
		d_22_32 = _mm_sub_pd( d_22_32, c_22_32 ); 

		_mm_store_pd(&C[0+ldc*0], d_00_10);
		_mm_store_pd(&C[2+ldc*0], d_20_30);
		_mm_store_pd(&C[0+ldc*1], d_01_11);
		_mm_store_pd(&C[2+ldc*1], d_21_31);
		_mm_store_pd(&C[0+ldc*2], d_02_12);
		_mm_store_pd(&C[2+ldc*2], d_22_32);
		}

	}



// 4x2 with data packed in 4
void kernel_dgemm_pp_nt_4x2_sse_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int ldc = 4;*/

	int k;
	
	__m128d
		c_00_11, c_01_10, c_20_31, c_21_30,
		a_01, a_23,
		b_01, b_10, b_temp_0, b_temp_1;
	
	c_00_11 = _mm_setzero_pd();
	c_01_10 = _mm_setzero_pd();
	c_20_31 = _mm_setzero_pd();
	c_21_30 = _mm_setzero_pd();
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{
		a_01 = _mm_load_pd(&A[0]);
		a_23 = _mm_load_pd(&A[2]);
		
		b_01 = _mm_load_pd(&B[0]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		
		
		a_01 = _mm_load_pd(&A[4]);
		a_23 = _mm_load_pd(&A[6]);
		
		b_01 = _mm_load_pd(&B[4]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		
		
		a_01 = _mm_load_pd(&A[8]);
		a_23 = _mm_load_pd(&A[10]);
		
		b_01 = _mm_load_pd(&B[8]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		
		
		a_01 = _mm_load_pd(&A[12]);
		a_23 = _mm_load_pd(&A[14]);
		
		b_01 = _mm_load_pd(&B[12]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );

		
		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		a_01 = _mm_load_pd(&A[0]);
		a_23 = _mm_load_pd(&A[2]);
		
		b_01 = _mm_load_pd(&B[0]);
		b_10 = _mm_shuffle_pd(b_01, b_01, 1);
	
		b_temp_0 = b_01;
		b_01 = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, b_01 );
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_31 = _mm_add_pd( c_20_31, b_temp_0 );

		b_temp_1 = b_10;
		b_10 = _mm_mul_pd( a_01, b_10 );
		c_01_10 = _mm_add_pd( c_01_10, b_10 );
		b_temp_1 = _mm_mul_pd( a_23, b_temp_1 );
		c_21_30 = _mm_add_pd( c_21_30, b_temp_1 );
		

		A += 4;
		B += 4;

		}

	__m128d
		c_00_10, c_20_30, c_01_11, c_21_31,
		d_00_10, d_20_30, d_01_11, d_21_31;

	c_00_10 = _mm_blend_pd(c_00_11, c_01_10, 2);
	c_01_11 = _mm_blend_pd(c_01_10, c_00_11, 2);
	c_20_30 = _mm_blend_pd(c_20_31, c_21_30, 2);
	c_21_31 = _mm_blend_pd(c_21_30, c_20_31, 2);

	if(alg==0)
		{
		_mm_store_pd(&C[0+ldc*0], c_00_10);
		_mm_store_pd(&C[2+ldc*0], c_20_30);
		_mm_store_pd(&C[0+ldc*1], c_01_11);
		_mm_store_pd(&C[2+ldc*1], c_21_31);
		}
	else if(alg==1)
		{
		d_00_10 = _mm_load_pd(&C[0+ldc*0]);
		d_20_30 = _mm_load_pd(&C[2+ldc*0]);
		d_01_11 = _mm_load_pd(&C[0+ldc*1]);
		d_21_31 = _mm_load_pd(&C[2+ldc*1]);
		
		d_00_10 = _mm_add_pd( d_00_10, c_00_10 ); 
		d_01_11 = _mm_add_pd( d_01_11, c_01_11 ); 
		d_20_30 = _mm_add_pd( d_20_30, c_20_30 ); 
		d_21_31 = _mm_add_pd( d_21_31, c_21_31 ); 

		_mm_store_pd(&C[0+ldc*0], d_00_10);
		_mm_store_pd(&C[2+ldc*0], d_20_30);
		_mm_store_pd(&C[0+ldc*1], d_01_11);
		_mm_store_pd(&C[2+ldc*1], d_21_31);
		}
	else
		{
		d_00_10 = _mm_load_pd(&C[0+ldc*0]);
		d_20_30 = _mm_load_pd(&C[2+ldc*0]);
		d_01_11 = _mm_load_pd(&C[0+ldc*1]);
		d_21_31 = _mm_load_pd(&C[2+ldc*1]);
		
		d_00_10 = _mm_sub_pd( d_00_10, c_00_10 ); 
		d_01_11 = _mm_sub_pd( d_01_11, c_01_11 ); 
		d_20_30 = _mm_sub_pd( d_20_30, c_20_30 ); 
		d_21_31 = _mm_sub_pd( d_21_31, c_21_31 ); 

		_mm_store_pd(&C[0+ldc*0], d_00_10);
		_mm_store_pd(&C[2+ldc*0], d_20_30);
		_mm_store_pd(&C[0+ldc*1], d_01_11);
		_mm_store_pd(&C[2+ldc*1], d_21_31);
		}

	}



// 4x1 with data packed in 4
void kernel_dgemm_pp_nt_4x1_sse_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int ldc = 4;*/

	int k;
	
	__m128d
		c_00_10, c_20_30, c_00_10_b, c_20_30_b,
		a_01, a_23,
		b_0, b_temp_0;
	
	c_00_10   = _mm_setzero_pd();
	c_20_30   = _mm_setzero_pd();
	c_00_10_b = _mm_setzero_pd();
	c_20_30_b = _mm_setzero_pd();
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{
		a_01 = _mm_load_pd(&A[0]);
		a_23 = _mm_load_pd(&A[2]);
		
		b_0 = _mm_loaddup_pd(&B[0]);
	
		b_temp_0 = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10 = _mm_add_pd( c_00_10, b_0 );	
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_30 = _mm_add_pd( c_20_30, b_temp_0 );	
		
		
		a_01 = _mm_load_pd(&A[4]);
		a_23 = _mm_load_pd(&A[6]);
		
		b_0 = _mm_loaddup_pd(&B[4]);
	
		b_temp_0 = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10_b = _mm_add_pd( c_00_10_b, b_0 );	
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_30_b = _mm_add_pd( c_20_30_b, b_temp_0 );	
		
		
		a_01 = _mm_load_pd(&A[8]);
		a_23 = _mm_load_pd(&A[10]);
		
		b_0 = _mm_loaddup_pd(&B[8]);
	
		b_temp_0 = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10 = _mm_add_pd( c_00_10, b_0 );	
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_30 = _mm_add_pd( c_20_30, b_temp_0 );	
		
		
		a_01 = _mm_load_pd(&A[12]);
		a_23 = _mm_load_pd(&A[14]);
		
		b_0 = _mm_loaddup_pd(&B[12]);
	
		b_temp_0 = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10_b = _mm_add_pd( c_00_10_b, b_0 );	
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_30_b = _mm_add_pd( c_20_30_b, b_temp_0 );	
		
		
		A += 16;
		B += 16;

		}
	
	c_00_10 = _mm_add_pd( c_00_10, c_00_10_b );
	c_20_30 = _mm_add_pd( c_20_30, c_20_30_b );
	
	for(; k<kmax; k++)
		{

		a_01 = _mm_load_pd(&A[0]);
		a_23 = _mm_load_pd(&A[2]);
		
		b_0 = _mm_loaddup_pd(&B[0]);
	
		b_temp_0 = b_0;
		b_0 = _mm_mul_pd( a_01, b_0 );
		c_00_10 = _mm_add_pd( c_00_10, b_0 );	
		b_temp_0 = _mm_mul_pd( a_23, b_temp_0 );
		c_20_30 = _mm_add_pd( c_20_30, b_temp_0 );	
		
		A += 4;
		B += 4;

		}

	__m128d
		d_00_10, d_20_30;

	if(alg==0)
		{
		_mm_store_pd(&C[0+ldc*0], c_00_10);
		_mm_store_pd(&C[2+ldc*0], c_20_30);
		}
	else if(alg==1)
		{
		d_00_10 = _mm_load_pd(&C[0+ldc*0]);
		d_20_30 = _mm_load_pd(&C[2+ldc*0]);
		
		d_00_10 = _mm_add_pd( d_00_10, c_00_10 ); 
		d_20_30 = _mm_add_pd( d_20_30, c_20_30 ); 

		_mm_store_pd(&C[0+ldc*0], d_00_10);
		_mm_store_pd(&C[2+ldc*0], d_20_30);
		}
	else
		{
		d_00_10 = _mm_load_pd(&C[0+ldc*0]);
		d_20_30 = _mm_load_pd(&C[2+ldc*0]);
		
		d_00_10 = _mm_sub_pd( d_00_10, c_00_10 ); 
		d_20_30 = _mm_sub_pd( d_20_30, c_20_30 ); 

		_mm_store_pd(&C[0+ldc*0], d_00_10);
		_mm_store_pd(&C[2+ldc*0], d_20_30);
		}

	}

