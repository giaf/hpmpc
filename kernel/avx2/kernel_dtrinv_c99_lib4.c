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
* Foundation, Inc., 51 Franklin Street, Fifth Floor, A_00_invoston, MA  02110-1301  USA                  *
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




// computes the inverse of a 4x4 lower trinagular matrix stored as *fact output from cholesky, and stores it at an upper triangular matrix
void corner_dtrinv_4x4_lib4(double *fact, double *C)
	{

	const int bs = 4;

	double
		c_00=0.0,
		c_10=0.0, c_11=0.0,
		c_20=0.0, c_21=0.0, c_22=0.0,
		c_30=0.0, c_31=0.0, c_32=0.0, c_33=0.0,
		d_20=0.0, d_21=0.0,
		d_30=0.0, d_31=0.0;

//	c_00 = 1.0/A[0+bs*0];
//	c_11 = 1.0/A[1+bs*1];
//	c_22 = 1.0/A[2+bs*2];
//	c_33 = 1.0/A[3+bs*3];
	c_00 = fact[0];
	c_11 = fact[2];
	c_22 = fact[5];
	c_33 = fact[9];

	C[0+bs*0] = c_00;
	C[1+bs*1] = c_11;
	C[2+bs*2] = c_22;
	C[3+bs*3] = c_33;

//	c_10 = A[1+bs*0];
//	c_32 = A[3+bs*2];
	c_10 = fact[1];
	c_32 = fact[8];

	c_10 = - c_11*c_10*c_00;
	c_32 = - c_33*c_32*c_22;

	C[0+bs*1] = c_10;
	C[2+bs*3] = c_32;

//	c_20 = A[2+bs*1]*c_10;
//	c_30 = A[3+bs*1]*c_10;
//	c_21 = A[2+bs*1]*c_11;
//	c_31 = A[3+bs*1]*c_11;
//	c_20 = fact[3];
//	c_30 = fact[6];
	c_21 = fact[4];
	c_31 = fact[7];

	c_20 = c_21*c_10;
	c_30 = c_31*c_10;
	c_21 = c_21*c_11;
	c_31 = c_31*c_11;

	c_20 += fact[3]*c_00;
	c_30 += fact[6]*c_00;
//	c_21 += A[2+bs*0]*c_01;
//	c_31 += A[3+bs*0]*c_01;

	d_20 = c_22*c_20;
	d_30 = c_32*c_20;
	d_21 = c_22*c_21;
	d_31 = c_32*c_21;

//	d_20 += c_23*c_30;
	d_30 += c_33*c_30;
//	d_21 += c_23*c_31;
	d_31 += c_33*c_31;

	C[0+bs*2] = - d_20;
	C[0+bs*3] = - d_30;
	C[1+bs*2] = - d_21;
	C[1+bs*3] = - d_31;

	return;

	}



// computes the inverse of a 2x2 lower trinagular matrix stored as *fact output from cholesky, and stores it at an upper triangular matrix
void corner_dtrinv_2x2_lib4(double *fact, double *C)
	{

	const int bs = 4;

	double
		c_00=0.0,
		c_10=0.0, c_11=0.0;

	c_00 = fact[0];
	c_11 = fact[2];

	C[0+bs*0] = c_00;
	C[1+bs*1] = c_11;

	c_10 = fact[1];

	c_10 = - c_11*c_10*c_00;

	C[0+bs*1] = c_10;

	return;

	}



void kernel_dtrinv_8x4_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *fact)
	{

	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	
	const int bs = 4;
	
	int k;
	
	__m256d
		zeros,
		a_0, a_4, A_0, A_4,
		b_0, b_1, b_2,
		c_00, c_01, c_03, c_02,
		c_40, c_41, c_43, c_42;
	
	// prefetch
	a_0 = _mm256_load_pd( &A0[0] );
	a_4 = _mm256_load_pd( &A1[0] );
	b_0 = _mm256_broadcast_pd( (__m128d *) &B[0] );
	b_2 = _mm256_broadcast_pd( (__m128d *) &B[2] );

	zeros = _mm256_setzero_pd();

	c_00 = _mm256_setzero_pd();
	c_01 = _mm256_setzero_pd();
	c_03 = _mm256_setzero_pd();
	c_02 = _mm256_setzero_pd();
	c_40 = _mm256_setzero_pd();
	c_41 = _mm256_setzero_pd();
	c_43 = _mm256_setzero_pd();
	c_42 = _mm256_setzero_pd();

	k = 0;

	// k = 0
	a_0  = _mm256_blend_pd( zeros, a_0, 0x1 );
	A_0  = _mm256_load_pd( &A0[4] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
	c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
	b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
	c_03 = _mm256_fnmadd_pd( a_0, b_1, c_03 );
			
	// k = 1
	A_0  = _mm256_blend_pd( zeros, A_0, 0x3 );
	a_0  = _mm256_load_pd( &A0[8] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( A_0, b_0, c_00 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
	c_01 = _mm256_fnmadd_pd( A_0, b_1, c_01 );
	b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( A_0, b_2, c_02 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch
	c_03 = _mm256_fnmadd_pd( A_0, b_1, c_03 );

	// k = 2
	a_0  = _mm256_blend_pd( zeros, a_0, 0x7 );
	A_0  = _mm256_load_pd( &A0[12] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
	c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
	b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[14] ); // prefetch
	c_03 = _mm256_fnmadd_pd( a_0, b_1, c_03 );

	// k = 3
	a_0  = _mm256_load_pd( &A0[16] ); // prefetch
	a_4  = _mm256_load_pd( &A1[16] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( A_0, b_0, c_00 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
	c_01 = _mm256_fnmadd_pd( A_0, b_1, c_01 );
	b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( A_0, b_2, c_02 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[18] ); // prefetch
	c_03 = _mm256_fnmadd_pd( A_0, b_1, c_03 );
		
	A0 += 16;
	A1 += 16;
	B  += 16;

	// k = 4
	a_4  = _mm256_blend_pd( zeros, a_4, 0x1 );
	A_0  = _mm256_load_pd( &A0[4] ); // prefetch
	A_4  = _mm256_load_pd( &A1[4] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
	c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
	c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
	c_41 = _mm256_fnmadd_pd( a_4, b_1, c_41 );
	b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
	c_42 = _mm256_fnmadd_pd( a_4, b_2, c_42 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
	c_03 = _mm256_fnmadd_pd( a_0, b_1, c_03 );
	c_43 = _mm256_fnmadd_pd( a_4, b_1, c_43 );
				
	// k = 5
	A_4  = _mm256_blend_pd( zeros, A_4, 0x3 );
	a_0  = _mm256_load_pd( &A0[8] ); // prefetch
	a_4  = _mm256_load_pd( &A1[8] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( A_0, b_0, c_00 );
	c_40 = _mm256_fnmadd_pd( A_4, b_0, c_40 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
	c_01 = _mm256_fnmadd_pd( A_0, b_1, c_01 );
	c_41 = _mm256_fnmadd_pd( A_4, b_1, c_41 );
	b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( A_0, b_2, c_02 );
	c_42 = _mm256_fnmadd_pd( A_4, b_2, c_42 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch
	c_03 = _mm256_fnmadd_pd( A_0, b_1, c_03 );
	c_43 = _mm256_fnmadd_pd( A_4, b_1, c_43 );

	// k = 6
	a_4  = _mm256_blend_pd( zeros, a_4, 0x7 );
	A_0  = _mm256_load_pd( &A0[12] ); // prefetch
	A_4  = _mm256_load_pd( &A1[12] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
	c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
	c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
	c_41 = _mm256_fnmadd_pd( a_4, b_1, c_41 );
	b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
	c_42 = _mm256_fnmadd_pd( a_4, b_2, c_42 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[14] ); // prefetch
	c_03 = _mm256_fnmadd_pd( a_0, b_1, c_03 );
	c_43 = _mm256_fnmadd_pd( a_4, b_1, c_43 );
		
	// k = 7
	a_0  = _mm256_load_pd( &A0[16] ); // prefetch
	a_4  = _mm256_load_pd( &A1[16] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( A_0, b_0, c_00 );
	c_40 = _mm256_fnmadd_pd( A_4, b_0, c_40 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
	c_01 = _mm256_fnmadd_pd( A_0, b_1, c_01 );
	c_41 = _mm256_fnmadd_pd( A_4, b_1, c_41 );
	b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( A_0, b_2, c_02 );
	c_42 = _mm256_fnmadd_pd( A_4, b_2, c_42 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[18] ); // prefetch
	c_03 = _mm256_fnmadd_pd( A_0, b_1, c_03 );
	c_43 = _mm256_fnmadd_pd( A_4, b_1, c_43 );

	A0 += 16;
	A1 += 16;
	B  += 16;

	k = 8;

	for(; k<kmax-3; k+=4) // correction in cholesky is multiple of block size 4
		{
		
/*	__builtin_prefetch( A+32 );*/
		A_0  = _mm256_load_pd( &A0[4] ); // prefetch
		A_4  = _mm256_load_pd( &A1[4] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
		c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
		c_41 = _mm256_fnmadd_pd( a_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
		c_42 = _mm256_fnmadd_pd( a_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		c_03 = _mm256_fnmadd_pd( a_0, b_1, c_03 );
		c_43 = _mm256_fnmadd_pd( a_4, b_1, c_43 );
		
/*	__builtin_prefetch( A+40 );*/
		a_0  = _mm256_load_pd( &A0[8] ); // prefetch
		a_4  = _mm256_load_pd( &A1[8] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fnmadd_pd( A_0, b_0, c_00 );
		c_40 = _mm256_fnmadd_pd( A_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_01 = _mm256_fnmadd_pd( A_0, b_1, c_01 );
		c_41 = _mm256_fnmadd_pd( A_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fnmadd_pd( A_0, b_2, c_02 );
		c_42 = _mm256_fnmadd_pd( A_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch
		c_03 = _mm256_fnmadd_pd( A_0, b_1, c_03 );
		c_43 = _mm256_fnmadd_pd( A_4, b_1, c_43 );
	
/*	__builtin_prefetch( A+48 );*/
		A_0  = _mm256_load_pd( &A0[12] ); // prefetch
		A_4  = _mm256_load_pd( &A1[12] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
		c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
		c_41 = _mm256_fnmadd_pd( a_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
		c_42 = _mm256_fnmadd_pd( a_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[14] ); // prefetch
		c_03 = _mm256_fnmadd_pd( a_0, b_1, c_03 );
		c_43 = _mm256_fnmadd_pd( a_4, b_1, c_43 );
	
/*	__builtin_prefetch( A+56 );*/
		a_0  = _mm256_load_pd( &A0[16] ); // prefetch
		a_4  = _mm256_load_pd( &A1[16] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fnmadd_pd( A_0, b_0, c_00 );
		c_40 = _mm256_fnmadd_pd( A_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		c_01 = _mm256_fnmadd_pd( A_0, b_1, c_01 );
		c_41 = _mm256_fnmadd_pd( A_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fnmadd_pd( A_0, b_2, c_02 );
		c_42 = _mm256_fnmadd_pd( A_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[18] ); // prefetch
		c_03 = _mm256_fnmadd_pd( A_0, b_1, c_03 );
		c_43 = _mm256_fnmadd_pd( A_4, b_1, c_43 );
	
		A0 += 16;
		A1 += 16;
		B  += 16;

		}

	__m256d
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33,
		d_40_50_60_70, d_41_51_61_71, d_42_52_62_72, d_43_53_63_73;

	d_00_10_20_30 = _mm256_blend_pd( c_00, c_01, 0xa );
	d_01_11_21_31 = _mm256_blend_pd( c_00, c_01, 0x5 );
	d_02_12_22_32 = _mm256_blend_pd( c_02, c_03, 0xa );
	d_03_13_23_33 = _mm256_blend_pd( c_02, c_03, 0x5 );
	d_40_50_60_70 = _mm256_blend_pd( c_40, c_41, 0xa );
	d_41_51_61_71 = _mm256_blend_pd( c_40, c_41, 0x5 );
	d_42_52_62_72 = _mm256_blend_pd( c_42, c_43, 0xa );
	d_43_53_63_73 = _mm256_blend_pd( c_42, c_43, 0x5 );

	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	d_40_50_60_70 = _mm256_mul_pd( d_40_50_60_70, a_00 );
	_mm256_store_pd( &C0[0+bs*0], d_00_10_20_30 );
	_mm256_store_pd( &C1[0+bs*0], d_40_50_60_70 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	d_01_11_21_31 = _mm256_fnmadd_pd( d_00_10_20_30, a_10, d_01_11_21_31 );
	d_41_51_61_71 = _mm256_fnmadd_pd( d_40_50_60_70, a_10, d_41_51_61_71 );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
	_mm256_store_pd( &C0[0+bs*1], d_01_11_21_31 );
	_mm256_store_pd( &C1[0+bs*1], d_41_51_61_71 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	d_02_12_22_32 = _mm256_fnmadd_pd( d_00_10_20_30, a_20, d_02_12_22_32 );
	d_42_52_62_72 = _mm256_fnmadd_pd( d_40_50_60_70, a_20, d_42_52_62_72 );
	d_02_12_22_32 = _mm256_fnmadd_pd( d_01_11_21_31, a_21, d_02_12_22_32 );
	d_42_52_62_72 = _mm256_fnmadd_pd( d_41_51_61_71, a_21, d_42_52_62_72 );
	d_02_12_22_32 = _mm256_mul_pd( d_02_12_22_32, a_22 );
	d_42_52_62_72 = _mm256_mul_pd( d_42_52_62_72, a_22 );
	_mm256_store_pd( &C0[0+bs*2], d_02_12_22_32 );
	_mm256_store_pd( &C1[0+bs*2], d_42_52_62_72 );

	a_30 = _mm256_broadcast_sd( &fact[6] );
	a_31 = _mm256_broadcast_sd( &fact[7] );
	a_32 = _mm256_broadcast_sd( &fact[8] );
	a_33 = _mm256_broadcast_sd( &fact[9] );
	d_03_13_23_33 = _mm256_fnmadd_pd( d_00_10_20_30, a_30, d_03_13_23_33 );
	d_43_53_63_73 = _mm256_fnmadd_pd( d_40_50_60_70, a_30, d_43_53_63_73 );
	d_03_13_23_33 = _mm256_fnmadd_pd( d_01_11_21_31, a_31, d_03_13_23_33 );
	d_43_53_63_73 = _mm256_fnmadd_pd( d_41_51_61_71, a_31, d_43_53_63_73 );
	d_03_13_23_33 = _mm256_fnmadd_pd( d_02_12_22_32, a_32, d_03_13_23_33 );
	d_43_53_63_73 = _mm256_fnmadd_pd( d_42_52_62_72, a_32, d_43_53_63_73 );
	d_03_13_23_33 = _mm256_mul_pd( d_03_13_23_33, a_33 );
	d_43_53_63_73 = _mm256_mul_pd( d_43_53_63_73, a_33 );
	_mm256_store_pd( &C0[0+bs*3], d_03_13_23_33 );
	_mm256_store_pd( &C1[0+bs*3], d_43_53_63_73 );


	}



void kernel_dtrinv_4x4_lib4(int kmax, double *A, double *B, double *C, double *fact)
	{

	const int bs = 4;

	int k;

	__m256d
		zeros, 
		a_0, A_0,
		b_0, B_0, b_1, b_2, B_2, b_3,
		c_00, c_01, c_03, c_02,
		C_00, C_01, C_03, C_02;
	
	// prefetch
	a_0 = _mm256_load_pd( &A[0] );
	b_0 = _mm256_broadcast_pd( (__m128d *) &B[0] );
	b_2 = _mm256_broadcast_pd( (__m128d *) &B[2] );

	// zero registers
	zeros = _mm256_setzero_pd();

	c_00 = _mm256_setzero_pd();
	c_01 = _mm256_setzero_pd();
	c_03 = _mm256_setzero_pd();
	c_02 = _mm256_setzero_pd();
	C_00 = _mm256_setzero_pd();
	C_01 = _mm256_setzero_pd();
	C_03 = _mm256_setzero_pd();
	C_02 = _mm256_setzero_pd();

	k = 0;

	// k = 0
	a_0  = _mm256_blend_pd( zeros, a_0, 0x1 );
	B_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
	A_0  = _mm256_load_pd( &A[4] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
	c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
	b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
	c_03 = _mm256_fnmadd_pd( a_0, b_3, c_03 );
	B_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		
	// k = 1
	A_0  = _mm256_blend_pd( zeros, A_0, 0x3 );
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
	a_0  = _mm256_load_pd( &A[8] ); // prefetch
	b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
	C_00 = _mm256_fnmadd_pd( A_0, B_0, C_00 );
	C_01 = _mm256_fnmadd_pd( A_0, b_1, C_01 );
	b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
	C_02 = _mm256_fnmadd_pd( A_0, B_2, C_02 );
	C_03 = _mm256_fnmadd_pd( A_0, b_3, C_03 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch

	// k = 2
	a_0  = _mm256_blend_pd( zeros, a_0, 0x7 );
	B_0  = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
	A_0  = _mm256_load_pd( &A[12] ); // prefetch
	b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
	c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
	c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
	b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
	c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
	c_03 = _mm256_fnmadd_pd( a_0, b_3, c_03 );
	B_2  = _mm256_broadcast_pd( (__m128d *) &B[14] ); // prefetch

	// k = 3
	b_0  = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
	a_0  = _mm256_load_pd( &A[16] ); // prefetch
	b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
	C_00 = _mm256_fnmadd_pd( A_0, B_0, C_00 );
	C_01 = _mm256_fnmadd_pd( A_0, b_1, C_01 );
	b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
	C_02 = _mm256_fnmadd_pd( A_0, B_2, C_02 );
	C_03 = _mm256_fnmadd_pd( A_0, b_3, C_03 );
	b_2  = _mm256_broadcast_pd( (__m128d *) &B[18] ); // prefetch

	A += 16;
	B += 16;
	k = 4;

	for(; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		B_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		A_0  = _mm256_load_pd( &A[4] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
		c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
		b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
		c_03 = _mm256_fnmadd_pd( a_0, b_3, c_03 );
		B_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		
		
/*	__builtin_prefetch( A+40 );*/
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		a_0  = _mm256_load_pd( &A[8] ); // prefetch
		b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
		C_00 = _mm256_fnmadd_pd( A_0, B_0, C_00 );
		C_01 = _mm256_fnmadd_pd( A_0, b_1, C_01 );
		b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
		C_02 = _mm256_fnmadd_pd( A_0, B_2, C_02 );
		C_03 = _mm256_fnmadd_pd( A_0, b_3, C_03 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch


/*	__builtin_prefetch( A+48 );*/
		B_0  = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		A_0  = _mm256_load_pd( &A[12] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
		c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
		b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
		c_03 = _mm256_fnmadd_pd( a_0, b_3, c_03 );
		B_2  = _mm256_broadcast_pd( (__m128d *) &B[14] ); // prefetch


/*	__builtin_prefetch( A+56 );*/
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		a_0  = _mm256_load_pd( &A[16] ); // prefetch
		b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
		C_00 = _mm256_fnmadd_pd( A_0, B_0, C_00 );
		C_01 = _mm256_fnmadd_pd( A_0, b_1, C_01 );
		b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
		C_02 = _mm256_fnmadd_pd( A_0, B_2, C_02 );
		C_03 = _mm256_fnmadd_pd( A_0, b_3, C_03 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[18] ); // prefetch

		A += 16;
		B += 16;

		}
	
	__m256d
		d_00, d_01, d_02, d_03;

	d_00 = _mm256_blend_pd( c_00, c_01, 0xa );
	d_01 = _mm256_blend_pd( c_00, c_01, 0x5 );
	d_02 = _mm256_blend_pd( c_02, c_03, 0xa );
	d_03 = _mm256_blend_pd( c_02, c_03, 0x5 );

	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00 = _mm256_mul_pd( d_00, a_00 );
	_mm256_store_pd( &C[0+bs*0], d_00 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	d_01 = _mm256_fnmadd_pd( d_00, a_10, d_01 );
	d_01 = _mm256_mul_pd( d_01, a_11 );
	_mm256_store_pd( &C[0+bs*1], d_01 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	d_02 = _mm256_fnmadd_pd( d_00, a_20, d_02 );
	d_02 = _mm256_fnmadd_pd( d_01, a_21, d_02 );
	d_02 = _mm256_mul_pd( d_02, a_22 );
	_mm256_store_pd( &C[0+bs*2], d_02 );

	a_30 = _mm256_broadcast_sd( &fact[6] );
	a_31 = _mm256_broadcast_sd( &fact[7] );
	a_32 = _mm256_broadcast_sd( &fact[8] );
	a_33 = _mm256_broadcast_sd( &fact[9] );
	d_03 = _mm256_fnmadd_pd( d_00, a_30, d_03 );
	d_03 = _mm256_fnmadd_pd( d_01, a_31, d_03 );
	d_03 = _mm256_fnmadd_pd( d_02, a_32, d_03 );
	d_03 = _mm256_mul_pd( d_03, a_33 );
	_mm256_store_pd( &C[0+bs*3], d_03 );

	}
	
	
	
void kernel_dtrinv_4x2_lib4(int kmax, double *A, double *B, double *C, double *fact)
	{

	const int bs = 4;

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1,
		c_00=0, c_01=0,
		c_10=0, c_11=0,
		c_20=0, c_21=0,
		c_30=0, c_31=0;
	
	// triangle at the beginning

	// k=0
	a_0 = A[0+bs*0];
		
	b_0 = B[0+bs*0];
	b_1 = B[1+bs*0];
		
	c_00 -= a_0 * b_0;

	c_01 -= a_0 * b_1;


	// k=1
	a_0 = A[0+bs*1];
	a_1 = A[1+bs*1];
		
	b_0 = B[0+bs*1];
	b_1 = B[1+bs*1];
		
	c_00 -= a_0 * b_0;
	c_10 -= a_1 * b_0;

	c_01 -= a_0 * b_1;
	c_11 -= a_1 * b_1;


	// k=2
	a_0 = A[0+bs*2];
	a_1 = A[1+bs*2];
	a_2 = A[2+bs*2];
		
	b_0 = B[0+bs*2];
	b_1 = B[1+bs*2];
		
	c_00 -= a_0 * b_0;
	c_10 -= a_1 * b_0;
	c_20 -= a_2 * b_0;

	c_01 -= a_0 * b_1;
	c_11 -= a_1 * b_1;
	c_21 -= a_2 * b_1;


	// k=3
	a_0 = A[0+bs*3];
	a_1 = A[1+bs*3];
	a_2 = A[2+bs*3];
	a_3 = A[3+bs*3];
		
	b_0 = B[0+bs*3];
	b_1 = B[1+bs*3];
		
	c_00 -= a_0 * b_0;
	c_10 -= a_1 * b_0;
	c_20 -= a_2 * b_0;
	c_30 -= a_3 * b_0;

	c_01 -= a_0 * b_1;
	c_11 -= a_1 * b_1;
	c_21 -= a_2 * b_1;
	c_31 -= a_3 * b_1;


	A += 16;
	B += 16;
	k = 4;
		
	for(; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;
		
		
		A += 16;
		B += 16;

		}

//	c_00 += D[0+bs*0];
//	c_10 += D[1+bs*0];
//	c_20 += D[2+bs*0];
//	c_30 += D[3+bs*0];

//	c_01 += D[0+bs*1];
//	c_11 += D[1+bs*1];
//	c_21 += D[2+bs*1];
//	c_31 += D[3+bs*1];
	
	// dtrsm
	double
		a_00, a_10, a_11;
	
	a_00 = fact[0];
	c_00 *= a_00;
	c_10 *= a_00;
	c_20 *= a_00;
	c_30 *= a_00;
	C[0+bs*0] = c_00;
	C[1+bs*0] = c_10;
	C[2+bs*0] = c_20;
	C[3+bs*0] = c_30;

	a_10 = fact[1];
	a_11 = fact[2];
	c_01 -= c_00*a_10;
	c_11 -= c_10*a_10;
	c_21 -= c_20*a_10;
	c_31 -= c_30*a_10;
	c_01 *= a_11;
	c_11 *= a_11;
	c_21 *= a_11;
	c_31 *= a_11;
	C[0+bs*1] = c_01;
	C[1+bs*1] = c_11;
	C[2+bs*1] = c_21;
	C[3+bs*1] = c_31;

	}
	
	
	
#if 0
// computes the inverse of a 4x4 lower trinagular matrix, and stores it at an upper triangular matrix
void corner_dtrinv_4x4_lib4_old(double *A, double *C)
	{

	const int bs = 4;

	double
		c_00=0.0,
		c_10=0.0, c_11=0.0,
		c_20=0.0, c_21=0.0, c_22=0.0,
		c_30=0.0, c_31=0.0, c_32=0.0, c_33=0.0,
		d_20=0.0, d_21=0.0,
		d_30=0.0, d_31=0.0;

	c_00 = 1.0/A[0+bs*0];
	c_11 = 1.0/A[1+bs*1];
	c_22 = 1.0/A[2+bs*2];
	c_33 = 1.0/A[3+bs*3];

	C[0+bs*0] = c_00;
	C[1+bs*1] = c_11;
	C[2+bs*2] = c_22;
	C[3+bs*3] = c_33;

	c_10 = - c_11*A[1+bs*0]*c_00;
	c_32 = - c_33*A[3+bs*2]*c_22;

	C[0+bs*1] = c_10;
	C[2+bs*3] = c_32;

	c_20 = A[2+bs*1]*c_10;
	c_30 = A[3+bs*1]*c_10;
	c_21 = A[2+bs*1]*c_11;
	c_31 = A[3+bs*1]*c_11;

	c_20 += A[2+bs*0]*c_00;
	c_30 += A[3+bs*0]*c_00;
//	c_21 += A[2+bs*0]*c_01;
//	c_31 += A[3+bs*0]*c_01;

	d_20 = c_22*c_20;
	d_30 = c_32*c_20;
	d_21 = c_22*c_21;
	d_31 = c_32*c_21;

//	d_20 += c_23*c_30;
	d_30 += c_33*c_30;
//	d_21 += c_23*c_31;
	d_31 += c_33*c_31;

	C[0+bs*2] = - d_20;
	C[0+bs*3] = - d_30;
	C[1+bs*2] = - d_21;
	C[1+bs*3] = - d_31;

	return;

	}
#endif



#if 0
// A_00_inv is the mxm top-left matrix already inverted; A_11_inv is the 4x4 bottom-right matrix already inverted, A_10 it the 4xm matrix to invert, C is the mx4 matrix to write the result
void kernel_dtrinv_4x4_lib4_old(int kmax, double *A_00_inv, double *A_10, double *A_11_inv, double *C, int sdc)
	{

	// assume kmax multile of bs !!!

	const int bs = 4;

	int k;

	double
		temp,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;

	// initial triangle

	a_0 = A_10[0+bs*0];
	a_1 = A_10[1+bs*0];
	a_2 = A_10[2+bs*0];
	a_3 = A_10[3+bs*0];
	
	b_0 = A_00_inv[0+bs*0];
	
	c_00 += a_0 * b_0;
	c_10 += a_1 * b_0;
	c_20 += a_2 * b_0;
	c_30 += a_3 * b_0;


	a_0 = A_10[0+bs*1];
	a_1 = A_10[1+bs*1];
	a_2 = A_10[2+bs*1];
	a_3 = A_10[3+bs*1];
	
	b_0 = A_00_inv[0+bs*1];
	b_1 = A_00_inv[1+bs*1];
	
	c_00 += a_0 * b_0;
	c_10 += a_1 * b_0;
	c_20 += a_2 * b_0;
	c_30 += a_3 * b_0;

	c_01 += a_0 * b_1;
	c_11 += a_1 * b_1;
	c_21 += a_2 * b_1;
	c_31 += a_3 * b_1;


	a_0 = A_10[0+bs*2];
	a_1 = A_10[1+bs*2];
	a_2 = A_10[2+bs*2];
	a_3 = A_10[3+bs*2];
	
	b_0 = A_00_inv[0+bs*2];
	b_1 = A_00_inv[1+bs*2];
	b_2 = A_00_inv[2+bs*2];
	
	c_00 += a_0 * b_0;
	c_10 += a_1 * b_0;
	c_20 += a_2 * b_0;
	c_30 += a_3 * b_0;

	c_01 += a_0 * b_1;
	c_11 += a_1 * b_1;
	c_21 += a_2 * b_1;
	c_31 += a_3 * b_1;

	c_02 += a_0 * b_2;
	c_12 += a_1 * b_2;
	c_22 += a_2 * b_2;
	c_32 += a_3 * b_2;


	a_0 = A_10[0+bs*3];
	a_1 = A_10[1+bs*3];
	a_2 = A_10[2+bs*3];
	a_3 = A_10[3+bs*3];
	
	b_0 = A_00_inv[0+bs*3];
	b_1 = A_00_inv[1+bs*3];
	b_2 = A_00_inv[2+bs*3];
	b_3 = A_00_inv[3+bs*3];
	
	c_00 += a_0 * b_0;
	c_10 += a_1 * b_0;
	c_20 += a_2 * b_0;
	c_30 += a_3 * b_0;

	c_01 += a_0 * b_1;
	c_11 += a_1 * b_1;
	c_21 += a_2 * b_1;
	c_31 += a_3 * b_1;

	c_02 += a_0 * b_2;
	c_12 += a_1 * b_2;
	c_22 += a_2 * b_2;
	c_32 += a_3 * b_2;

	c_03 += a_0 * b_3;
	c_13 += a_1 * b_3;
	c_23 += a_2 * b_3;
	c_33 += a_3 * b_3;


	A_10 += 16;
	A_00_inv += 16;

	for(k=4; k<kmax-3; k+=4)
		{
		
		a_0 = A_10[0+bs*0];
		a_1 = A_10[1+bs*0];
		a_2 = A_10[2+bs*0];
		a_3 = A_10[3+bs*0];
		
		b_0 = A_00_inv[0+bs*0];
		b_1 = A_00_inv[1+bs*0];
		b_2 = A_00_inv[2+bs*0];
		b_3 = A_00_inv[3+bs*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		a_0 = A_10[0+bs*1];
		a_1 = A_10[1+bs*1];
		a_2 = A_10[2+bs*1];
		a_3 = A_10[3+bs*1];
		
		b_0 = A_00_inv[0+bs*1];
		b_1 = A_00_inv[1+bs*1];
		b_2 = A_00_inv[2+bs*1];
		b_3 = A_00_inv[3+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		a_0 = A_10[0+bs*2];
		a_1 = A_10[1+bs*2];
		a_2 = A_10[2+bs*2];
		a_3 = A_10[3+bs*2];
		
		b_0 = A_00_inv[0+bs*2];
		b_1 = A_00_inv[1+bs*2];
		b_2 = A_00_inv[2+bs*2];
		b_3 = A_00_inv[3+bs*2];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		a_0 = A_10[0+bs*3];
		a_1 = A_10[1+bs*3];
		a_2 = A_10[2+bs*3];
		a_3 = A_10[3+bs*3];
		
		b_0 = A_00_inv[0+bs*3];
		b_1 = A_00_inv[1+bs*3];
		b_2 = A_00_inv[2+bs*3];
		b_3 = A_00_inv[3+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;
		
		
		A_10 += 16;
		A_00_inv += 16;

		}



	// transpose
	temp = c_10;
	c_10 = c_01;
	c_01 = temp;

	temp = c_20;
	c_20 = c_02;
	c_02 = temp;

	temp = c_30;
	c_30 = c_03;
	c_03 = temp;

	temp = c_21;
	c_21 = c_12;
	c_12 = temp;

	temp = c_31;
	c_31 = c_13;
	c_13 = temp;

	temp = c_32;
	c_32 = c_23;
	c_23 = temp;



	// final triangle

	b_3 = A_11_inv[3+bs*3];

	c_03 = c_03 * b_3;
	c_13 = c_13 * b_3;
	c_23 = c_23 * b_3;
	c_33 = c_33 * b_3;


	b_3 = A_11_inv[2+bs*3];
	b_2 = A_11_inv[2+bs*2];

	c_03 += c_02 * b_3;
	c_13 += c_12 * b_3;
	c_23 += c_22 * b_3;
	c_33 += c_32 * b_3;

	c_02 = c_02 * b_2;
	c_12 = c_12 * b_2;
	c_22 = c_22 * b_2;
	c_32 = c_32 * b_2;


	b_3 = A_11_inv[1+bs*3];
	b_2 = A_11_inv[1+bs*2];
	b_1 = A_11_inv[1+bs*1];

	c_03 += c_01 * b_3;
	c_13 += c_11 * b_3;
	c_23 += c_21 * b_3;
	c_33 += c_31 * b_3;

	c_02 += c_01 * b_2;
	c_12 += c_11 * b_2;
	c_22 += c_21 * b_2;
	c_32 += c_31 * b_2;

	c_01 = c_01 * b_1;
	c_11 = c_11 * b_1;
	c_21 = c_21 * b_1;
	c_31 = c_31 * b_1;
	

	b_3 = A_11_inv[0+bs*3];
	b_2 = A_11_inv[0+bs*2];
	b_1 = A_11_inv[0+bs*1];
	b_0 = A_11_inv[0+bs*0];

	c_03 += c_00 * b_3;
	c_13 += c_10 * b_3;
	c_23 += c_20 * b_3;
	c_33 += c_30 * b_3;

	c_02 += c_00 * b_2;
	c_12 += c_10 * b_2;
	c_22 += c_20 * b_2;
	c_32 += c_30 * b_2;

	c_01 += c_00 * b_1;
	c_11 += c_10 * b_1;
	c_21 += c_20 * b_1;
	c_31 += c_30 * b_1;

	c_00 = c_00 * b_0;
	c_10 = c_10 * b_0;
	c_20 = c_20 * b_0;
	c_30 = c_30 * b_0;


	// change sign & store result
	C[0+bs*0] = - c_00;
	C[1+bs*0] = - c_10;
	C[2+bs*0] = - c_20;
	C[3+bs*0] = - c_30;

	C[0+bs*1] = - c_01;
	C[1+bs*1] = - c_11;
	C[2+bs*1] = - c_21;
	C[3+bs*1] = - c_31;

	C[0+bs*2] = - c_02;
	C[1+bs*2] = - c_12;
	C[2+bs*2] = - c_22;
	C[3+bs*2] = - c_32;

	C[0+bs*3] = - c_03;
	C[1+bs*3] = - c_13;
	C[2+bs*3] = - c_23;
	C[3+bs*3] = - c_33;

	}
#endif







