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



// normal-transposed, 8x4 with data packed in 4
void kernel_dgemm_pp_nt_8x4_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31,
		c_40_51_62_73, c_41_50_63_72, c_43_52_61_70, c_42_53_60_71;
	
	// prefetch
	a_0123        = _mm256_load_pd( &A0[0] );
	a_4567        = _mm256_load_pd( &A1[0] );
	b_0123        = _mm256_load_pd( &B[0] );

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();
	c_40_51_62_73 = _mm256_setzero_pd();
	c_41_50_63_72 = _mm256_setzero_pd();
	c_43_52_61_70 = _mm256_setzero_pd();
	c_42_53_60_71 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A0+16 );*/
/*	__builtin_prefetch( A1+16 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_temp );


/*	__builtin_prefetch( A0+24 );*/
/*	__builtin_prefetch( A1+24 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_temp );


		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch 
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_temp );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch 
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_temp );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch */
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch */
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
/*		a_4567        = _mm256_load_pd( &A[4] ); // prefetch */
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_temp );
		
/*		A0 += 4;*/
/*		A1 += 4;*/
/*		B  += 4;*/
		
		}

	__m256d
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31,
		c_40_50_62_72, c_41_51_63_73, c_42_52_60_70, c_43_53_61_71,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33,
		d_40_50_60_70, d_41_51_61_71, d_42_52_62_72, d_43_53_63_73;

	c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
	c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
	c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
	c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
	c_40_50_62_72 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0xa );
	c_41_51_63_73 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0x5 );
	c_42_52_60_70 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0xa );
	c_43_53_61_71 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0x5 );
	
	c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
	c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
	c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
	c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
	c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
	c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
	c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
	c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );

	if(alg==0) // C = A * B'
		{
		_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
		_mm256_store_pd( &C0[0+ldc*1], c_01_11_21_31 );
		_mm256_store_pd( &C0[0+ldc*2], c_02_12_22_32 );
		_mm256_store_pd( &C0[0+ldc*3], c_03_13_23_33 );
		_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );
		_mm256_store_pd( &C1[0+ldc*1], c_41_51_61_71 );
		_mm256_store_pd( &C1[0+ldc*2], c_42_52_62_72 );
		_mm256_store_pd( &C1[0+ldc*3], c_43_53_63_73 );
		}
	else if(alg==1) // C += A * B'
		{
		d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
		d_02_12_22_32 = _mm256_load_pd( &C0[0+ldc*2] );
		d_03_13_23_33 = _mm256_load_pd( &C0[0+ldc*3] );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		d_42_52_62_72 = _mm256_load_pd( &C1[0+ldc*2] );
		d_43_53_63_73 = _mm256_load_pd( &C1[0+ldc*3] );
		
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
		d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
		d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );

		_mm256_store_pd( &C0[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &C0[0+ldc*1], d_01_11_21_31 );
		_mm256_store_pd( &C0[0+ldc*2], d_02_12_22_32 );
		_mm256_store_pd( &C0[0+ldc*3], d_03_13_23_33 );
		_mm256_store_pd( &C1[0+ldc*0], d_40_50_60_70 );
		_mm256_store_pd( &C1[0+ldc*1], d_41_51_61_71 );
		_mm256_store_pd( &C1[0+ldc*2], d_42_52_62_72 );
		_mm256_store_pd( &C1[0+ldc*3], d_43_53_63_73 );
		}
	else // C -= A * B'
		{
		d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
		d_02_12_22_32 = _mm256_load_pd( &C0[0+ldc*2] );
		d_03_13_23_33 = _mm256_load_pd( &C0[0+ldc*3] );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		d_42_52_62_72 = _mm256_load_pd( &C1[0+ldc*2] );
		d_43_53_63_73 = _mm256_load_pd( &C1[0+ldc*3] );
		
		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
		d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );
		d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
		d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
		d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, c_42_52_62_72 );
		d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, c_43_53_63_73 );

		_mm256_store_pd( &C0[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &C0[0+ldc*1], d_01_11_21_31 );
		_mm256_store_pd( &C0[0+ldc*2], d_02_12_22_32 );
		_mm256_store_pd( &C0[0+ldc*3], d_03_13_23_33 );
		_mm256_store_pd( &C1[0+ldc*0], d_40_50_60_70 );
		_mm256_store_pd( &C1[0+ldc*1], d_41_51_61_71 );
		_mm256_store_pd( &C1[0+ldc*2], d_42_52_62_72 );
		_mm256_store_pd( &C1[0+ldc*3], d_43_53_63_73 );
		}

	}



// normal-transposed, 8x3 with data packed in 8 TODO prefetch
void kernel_dgemm_pp_nt_8x3_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256d
		a_0123, a_4567,
		b_0, b_1, b_2,
		ab_temp,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72;
	
	c_00_10_20_30 = _mm256_setzero_pd();
	c_01_11_21_31 = _mm256_setzero_pd();
	c_02_12_22_32 = _mm256_setzero_pd();
	c_40_50_60_70 = _mm256_setzero_pd();
	c_41_51_61_71 = _mm256_setzero_pd();
	c_42_52_62_72 = _mm256_setzero_pd();

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[1] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[2] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );

		
		a_0123        = _mm256_load_pd( &A0[4] );
		a_4567        = _mm256_load_pd( &A1[4] );
		b_0           = _mm256_broadcast_sd( &B[4] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[5] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[6] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );

		
		a_0123        = _mm256_load_pd( &A0[8] );
		a_4567        = _mm256_load_pd( &A1[8] );
		b_0           = _mm256_broadcast_sd( &B[8] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[9] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[10] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );

		
		a_0123        = _mm256_load_pd( &A0[12] );
		a_4567        = _mm256_load_pd( &A1[12] );
		b_0           = _mm256_broadcast_sd( &B[12] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[13] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[14] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );
		

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[1] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[2] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );

		
		a_0123        = _mm256_load_pd( &A0[4] );
		a_4567        = _mm256_load_pd( &A1[4] );
		b_0           = _mm256_broadcast_sd( &B[4] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[5] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[6] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[1] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[2] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );
		
/*		A += 4;*/
/*		B += 4;*/
		
		}

	__m256d
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32,
		d_40_50_60_70, d_41_51_61_71, d_42_52_62_72;

	if(alg==0)
		{
		_mm256_store_pd(&C0[0+ldc*0], c_00_10_20_30);
		_mm256_store_pd(&C0[0+ldc*1], c_01_11_21_31);
		_mm256_store_pd(&C0[0+ldc*2], c_02_12_22_32);
		_mm256_store_pd(&C1[0+ldc*0], c_40_50_60_70);
		_mm256_store_pd(&C1[0+ldc*1], c_41_51_61_71);
		_mm256_store_pd(&C1[0+ldc*2], c_42_52_62_72);
		}
	else if(alg==1)
		{
		d_00_10_20_30 = _mm256_load_pd(&C0[0+ldc*0]);
		d_01_11_21_31 = _mm256_load_pd(&C0[0+ldc*1]);
		d_02_12_22_32 = _mm256_load_pd(&C0[0+ldc*2]);
		d_40_50_60_70 = _mm256_load_pd(&C1[0+ldc*0]);
		d_41_51_61_71 = _mm256_load_pd(&C1[0+ldc*1]);
		d_42_52_62_72 = _mm256_load_pd(&C1[0+ldc*2]);

		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
		d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );

		_mm256_store_pd(&C0[0+ldc*0], d_00_10_20_30);
		_mm256_store_pd(&C0[0+ldc*1], d_01_11_21_31);
		_mm256_store_pd(&C0[0+ldc*2], d_02_12_22_32);
		_mm256_store_pd(&C1[0+ldc*0], d_40_50_60_70);
		_mm256_store_pd(&C1[0+ldc*1], d_41_51_61_71);
		_mm256_store_pd(&C1[0+ldc*2], d_42_52_62_72);
		}
	else
		{
		d_00_10_20_30 = _mm256_load_pd(&C0[0+ldc*0]);
		d_01_11_21_31 = _mm256_load_pd(&C0[0+ldc*1]);
		d_02_12_22_32 = _mm256_load_pd(&C0[0+ldc*2]);
		d_40_50_60_70 = _mm256_load_pd(&C1[0+ldc*0]);
		d_41_51_61_71 = _mm256_load_pd(&C1[0+ldc*1]);
		d_42_52_62_72 = _mm256_load_pd(&C1[0+ldc*2]);

		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
		d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
		d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
		d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, c_42_52_62_72 );

		_mm256_store_pd(&C0[0+ldc*0], d_00_10_20_30);
		_mm256_store_pd(&C0[0+ldc*1], d_01_11_21_31);
		_mm256_store_pd(&C0[0+ldc*2], d_02_12_22_32);
		_mm256_store_pd(&C1[0+ldc*0], d_40_50_60_70);
		_mm256_store_pd(&C1[0+ldc*1], d_41_51_61_71);
		_mm256_store_pd(&C1[0+ldc*2], d_42_52_62_72);
		}

	
	}



// normal-transposed, 8x2 with data packed in 4 TODO prefetch
void kernel_dgemm_pp_nt_8x2_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
		
	int k;
	
	__m128d
		b_01;
		
	__m256d
		a_0123, a_4567,
		b_0101, b_1010,
		ab_temp, // temporary results
		c_00_11_20_31, c_01_10_21_30,
		c_40_51_60_71, c_41_50_61_70;
	
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	c_40_51_60_71 = _mm256_setzero_pd();
	c_41_50_61_70 = _mm256_setzero_pd();

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_01          = _mm_load_pd( &B[0] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp          = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp          = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_0101 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_1010 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_temp );

		
		
		a_0123        = _mm256_load_pd( &A0[4] );
		a_4567        = _mm256_load_pd( &A1[4] );
		b_01          = _mm_load_pd( &B[4] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp          = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp          = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_0101 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_1010 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_temp );

		
		
		a_0123        = _mm256_load_pd( &A0[8] );
		a_4567        = _mm256_load_pd( &A1[8] );
		b_01          = _mm_load_pd( &B[8] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp          = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp          = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_0101 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_1010 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_temp );

		
		
		a_0123        = _mm256_load_pd( &A0[12] );
		a_4567        = _mm256_load_pd( &A1[12] );
		b_01          = _mm_load_pd( &B[12] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp          = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp          = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_0101 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_1010 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_temp );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_01          = _mm_load_pd( &B[0] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp          = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp          = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_0101 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_1010 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_temp );

		
		
		a_0123        = _mm256_load_pd( &A0[4] );
		a_4567        = _mm256_load_pd( &A1[4] );
		b_01          = _mm_load_pd( &B[4] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp          = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp          = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_0101 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_1010 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_temp );

		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_01          = _mm_load_pd( &B[0] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp          = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp          = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_0101 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_temp );
		ab_temp          = _mm256_mul_pd( a_4567, b_1010 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_temp );
		
/*		A += 4;*/
/*		B += 4;*/
		
		}

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		c_40_50_60_70, c_41_51_61_71,
		d_00_10_20_30, d_01_11_21_31,
		d_40_50_60_70, d_41_51_61_71;

		c_00_10_20_30 = _mm256_blend_pd(c_00_11_20_31, c_01_10_21_30, 10);
		c_01_11_21_31 = _mm256_blend_pd(c_00_11_20_31, c_01_10_21_30, 5);
		c_40_50_60_70 = _mm256_blend_pd(c_40_51_60_71, c_41_50_61_70, 10);
		c_41_51_61_71 = _mm256_blend_pd(c_40_51_60_71, c_41_50_61_70, 5);

	if(alg==0)
		{
		_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
		_mm256_store_pd( &C0[0+ldc*1], c_01_11_21_31 );
		_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );
		_mm256_store_pd( &C1[0+ldc*1], c_41_51_61_71 );
		}
	else if(alg==1)
		{
		d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );

		_mm256_store_pd( &C0[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &C0[0+ldc*1], d_01_11_21_31 );
		_mm256_store_pd( &C1[0+ldc*0], d_40_50_60_70 );
		_mm256_store_pd( &C1[0+ldc*1], d_41_51_61_71 );
		}
	else
		{
		d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		
		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
		d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
		d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );

		_mm256_store_pd( &C0[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &C0[0+ldc*1], d_01_11_21_31 );
		_mm256_store_pd( &C1[0+ldc*0], d_40_50_60_70 );
		_mm256_store_pd( &C1[0+ldc*1], d_41_51_61_71 );
		}

	}



// normal-transposed, 8x1 with data packed in 4 TODO prefetch
void kernel_dgemm_pp_nt_8x1_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	

	int k;
	
	__m256d
		a_0123, a_4567,
		b_0,
		ab_temp, // temporary results
		c_00_10_20_30, c_00_10_20_30_b, c_40_50_60_70, c_40_50_60_70_b;
	
	c_00_10_20_30   = _mm256_setzero_pd();
	c_00_10_20_30_b = _mm256_setzero_pd();
	c_40_50_60_70   = _mm256_setzero_pd();
	c_40_50_60_70_b = _mm256_setzero_pd();

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A0[4] );
		a_4567          = _mm256_load_pd( &A1[4] );
		b_0             = _mm256_broadcast_sd( &B[4] );
		ab_temp         = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30_b = _mm256_add_pd( c_00_10_20_30_b, ab_temp );
		ab_temp         = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70_b = _mm256_add_pd( c_40_50_60_70_b, ab_temp );

		
		
		a_0123        = _mm256_load_pd( &A0[8] );
		a_4567        = _mm256_load_pd( &A1[8] );
		b_0           = _mm256_broadcast_sd( &B[8] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A0[12] );
		a_4567          = _mm256_load_pd( &A1[12] );
		b_0             = _mm256_broadcast_sd( &B[12] );
		ab_temp         = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30_b = _mm256_add_pd( c_00_10_20_30_b, ab_temp );
		ab_temp         = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70_b = _mm256_add_pd( c_40_50_60_70_b, ab_temp );

		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A0[4] );
		a_4567          = _mm256_load_pd( &A1[4] );
		b_0             = _mm256_broadcast_sd( &B[4] );
		ab_temp         = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30_b = _mm256_add_pd( c_00_10_20_30_b, ab_temp );
		ab_temp         = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70_b = _mm256_add_pd( c_40_50_60_70_b, ab_temp );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30  , c_00_10_20_30_b );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70  , c_40_50_60_70_b );

	if(kmax%2==1)
		{
		
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
		
/*		A += 4;*/
/*		B += 4;*/
		
		}

	__m256d
		d_00_10_20_30, d_40_50_60_70;

	if(alg==0)
		{
		_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
		_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );
		}
	else if(alg==1)
		{
		d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );

		_mm256_store_pd( &C0[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &C1[0+ldc*0], d_40_50_60_70 );
		}
	else
		{
		d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
		d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );

		_mm256_store_pd( &C0[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &C1[0+ldc*0], d_40_50_60_70 );
		}

	
	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dgemm_pp_nt_4x4_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256d
		a_0123, A_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
	// prefetch
	a_0123        = _mm256_load_pd( &A[0] );
	b_0123        = _mm256_load_pd( &B[0] );

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		A_0123        = _mm256_load_pd( &A[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( A_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( A_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( A_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( A_0123, b_2301 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		A_0123        = _mm256_load_pd( &A[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( A_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( A_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( A_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( A_0123, b_2301 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		A_0123        = _mm256_load_pd( &A[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( A_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( A_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( A_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( A_0123, b_2301 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kmax%2==1)
		{
		
/*		A_0123        = _mm256_load_pd( &A[4] ); // prefetch */
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
/*		b_0123        = _mm256_load_pd( &A[4] ); // prefetch */
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
/*		A += 4;*/
/*		B += 4;*/
		
		}

	__m256d
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33;

	c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
	c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
	c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
	c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
	
	c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
	c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
	c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
	c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );

	if(alg==0)
		{
		_mm256_store_pd(&C[0+ldc*0], c_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], c_01_11_21_31);
		_mm256_store_pd(&C[0+ldc*2], c_02_12_22_32);
		_mm256_store_pd(&C[0+ldc*3], c_03_13_23_33);
		}
	else if(alg==1)
		{
		d_00_10_20_30 = _mm256_load_pd(&C[0+ldc*0]);
		d_01_11_21_31 = _mm256_load_pd(&C[0+ldc*1]);
		d_02_12_22_32 = _mm256_load_pd(&C[0+ldc*2]);
		d_03_13_23_33 = _mm256_load_pd(&C[0+ldc*3]);
		
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );

		_mm256_store_pd(&C[0+ldc*0], d_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], d_01_11_21_31);
		_mm256_store_pd(&C[0+ldc*2], d_02_12_22_32);
		_mm256_store_pd(&C[0+ldc*3], d_03_13_23_33);
		}
	else
		{
		d_00_10_20_30 = _mm256_load_pd(&C[0+ldc*0]);
		d_01_11_21_31 = _mm256_load_pd(&C[0+ldc*1]);
		d_02_12_22_32 = _mm256_load_pd(&C[0+ldc*2]);
		d_03_13_23_33 = _mm256_load_pd(&C[0+ldc*3]);
		
		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
		d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );

		_mm256_store_pd(&C[0+ldc*0], d_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], d_01_11_21_31);
		_mm256_store_pd(&C[0+ldc*2], d_02_12_22_32);
		_mm256_store_pd(&C[0+ldc*3], d_03_13_23_33);
		}

	}



// normal-transposed, 4x3 with data packed in 8 TODO prefetch
void kernel_dgemm_pp_nt_4x3_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256d
		a_0123,
		b_0, b_1, b_2,
		ab_temp,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32;
	
	c_00_10_20_30 = _mm256_setzero_pd();
	c_01_11_21_31 = _mm256_setzero_pd();
	c_02_12_22_32 = _mm256_setzero_pd();

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[1] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[2] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );

		
		a_0123        = _mm256_load_pd( &A[4] );
		b_0           = _mm256_broadcast_sd( &B[4] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[5] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[6] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );

		
		a_0123        = _mm256_load_pd( &A[8] );
		b_0           = _mm256_broadcast_sd( &B[8] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[9] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[10] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );

		
		a_0123        = _mm256_load_pd( &A[12] );
		b_0           = _mm256_broadcast_sd( &B[12] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[13] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[14] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		

		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[1] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[2] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );

		
		a_0123        = _mm256_load_pd( &A[4] );
		b_0           = _mm256_broadcast_sd( &B[4] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[5] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[6] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kmax%2==1)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[1] );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[2] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		
/*		A += 4;*/
/*		B += 4;*/
		
		}

	__m256d
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32;

	if(alg==0)
		{
		_mm256_store_pd(&C[0+ldc*0], c_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], c_01_11_21_31);
		_mm256_store_pd(&C[0+ldc*2], c_02_12_22_32);
		}
	else if(alg==1)
		{
		d_00_10_20_30 = _mm256_load_pd(&C[0+ldc*0]);
		d_01_11_21_31 = _mm256_load_pd(&C[0+ldc*1]);
		d_02_12_22_32 = _mm256_load_pd(&C[0+ldc*2]);

		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );

		_mm256_store_pd(&C[0+ldc*0], d_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], d_01_11_21_31);
		_mm256_store_pd(&C[0+ldc*2], d_02_12_22_32);
		}
	else
		{
		d_00_10_20_30 = _mm256_load_pd(&C[0+ldc*0]);
		d_01_11_21_31 = _mm256_load_pd(&C[0+ldc*1]);
		d_02_12_22_32 = _mm256_load_pd(&C[0+ldc*2]);

		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );

		_mm256_store_pd(&C[0+ldc*0], d_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], d_01_11_21_31);
		_mm256_store_pd(&C[0+ldc*2], d_02_12_22_32);
		}

	
	}



// normal-transposed, 4x2 with data packed in 4 TODO prefetch
void kernel_dgemm_pp_nt_4x2_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m128d
		b_01;
		
	__m256d
		a_0123,
		b_0101, b_1010,
		ab_temp, // temporary results
		c_00_11_20_31, c_01_10_21_30, c_00_11_20_31_b, c_01_10_21_30_b;
	
	c_00_11_20_31   = _mm256_setzero_pd();
	c_01_10_21_30   = _mm256_setzero_pd();
	c_00_11_20_31_b = _mm256_setzero_pd();
	c_01_10_21_30_b = _mm256_setzero_pd();

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_01          = _mm_load_pd( &B[0] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A[4] );
		b_01            = _mm_load_pd( &B[4] );
		b_0101          = _mm256_castpd128_pd256( b_01 );
		b_0101          = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010          = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp         = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31_b = _mm256_add_pd( c_00_11_20_31_b, ab_temp );
		ab_temp         = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30_b = _mm256_add_pd( c_01_10_21_30_b, ab_temp );

		
		
		a_0123        = _mm256_load_pd( &A[8] );
		b_01          = _mm_load_pd( &B[8] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A[12] );
		b_01            = _mm_load_pd( &B[12] );
		b_0101          = _mm256_castpd128_pd256( b_01 );
		b_0101          = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010          = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp         = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31_b = _mm256_add_pd( c_00_11_20_31_b, ab_temp );
		ab_temp         = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30_b = _mm256_add_pd( c_01_10_21_30_b, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_01          = _mm_load_pd( &B[0] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A[4] );
		b_01            = _mm_load_pd( &B[4] );
		b_0101          = _mm256_castpd128_pd256( b_01 );
		b_0101          = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010          = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp         = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31_b = _mm256_add_pd( c_00_11_20_31_b, ab_temp );
		ab_temp         = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30_b = _mm256_add_pd( c_01_10_21_30_b, ab_temp );

		A += 8;
		B += 8;

		}

	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, c_00_11_20_31_b );
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, c_01_10_21_30_b );
	
	if(kmax%2==1)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_01          = _mm_load_pd( &B[0] );
		b_0101        = _mm256_castpd128_pd256( b_01 );
		b_0101        = _mm256_permute2f128_pd(b_0101, b_0101, 0);
		b_1010        = _mm256_shuffle_pd(b_0101, b_0101, 5);
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
/*		A += 4;*/
/*		B += 4;*/
		
		}

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		d_00_10_20_30, d_01_11_21_31;

	c_00_10_20_30 = _mm256_blend_pd(c_00_11_20_31, c_01_10_21_30, 10);
	c_01_11_21_31 = _mm256_blend_pd(c_00_11_20_31, c_01_10_21_30, 5);

	if(alg==0)
		{
		_mm256_store_pd(&C[0+ldc*0], c_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], c_01_11_21_31);
		}
	else if(alg==1)
		{
		d_00_10_20_30 = _mm256_load_pd(&C[0+ldc*0]);
		d_01_11_21_31 = _mm256_load_pd(&C[0+ldc*1]);
		
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );

		_mm256_store_pd(&C[0+ldc*0], d_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], d_01_11_21_31);
		}
	else
		{
		d_00_10_20_30 = _mm256_load_pd(&C[0+ldc*0]);
		d_01_11_21_31 = _mm256_load_pd(&C[0+ldc*1]);
		
		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );

		_mm256_store_pd(&C[0+ldc*0], d_00_10_20_30);
		_mm256_store_pd(&C[0+ldc*1], d_01_11_21_31);
		}

	}



// normal-transposed, 4x1 with data packed in 8 TODO prefetch
void kernel_dgemm_pp_nt_4x1_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256d
		a_0123,
		b_0,
		ab_temp,
		c_00_10_20_30, c_00_10_20_30_b, c_00_10_20_30_c, c_00_10_20_30_d;
	
	c_00_10_20_30   = _mm256_setzero_pd();
	c_00_10_20_30_b = _mm256_setzero_pd();
	c_00_10_20_30_c = _mm256_setzero_pd();
	c_00_10_20_30_d = _mm256_setzero_pd();

	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A[4] );
		b_0             = _mm256_broadcast_sd( &B[4] );
		ab_temp         = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30_b = _mm256_add_pd( c_00_10_20_30_b, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A[8] );
		b_0             = _mm256_broadcast_sd( &B[8] );
		ab_temp         = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30_c = _mm256_add_pd( c_00_10_20_30_c, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A[12] );
		b_0             = _mm256_broadcast_sd( &B[12] );
		ab_temp         = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30_d = _mm256_add_pd( c_00_10_20_30_d, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	c_00_10_20_30   = _mm256_add_pd( c_00_10_20_30  , c_00_10_20_30_c );
	c_00_10_20_30_b = _mm256_add_pd( c_00_10_20_30_b, c_00_10_20_30_d );
	
	if(kmax%4>=2)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );

		
		
		a_0123          = _mm256_load_pd( &A[4] );
		b_0             = _mm256_broadcast_sd( &B[4] );
		ab_temp         = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30_b = _mm256_add_pd( c_00_10_20_30_b, ab_temp );

		
		
		A += 8;
		B += 8;

		}

	c_00_10_20_30   = _mm256_add_pd( c_00_10_20_30  , c_00_10_20_30_b );

	if(kmax%2==1)
		{
		
		a_0123        = _mm256_load_pd( &A[0] );
		b_0           = _mm256_broadcast_sd( &B[0] );
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		
/*		A += 4;*/
/*		B += 4;*/
		
		}

	__m256d
		d_00_10_20_30;

	if(alg==0)
		{
		_mm256_store_pd(&C[0+ldc*0], c_00_10_20_30);
		}
	else if(alg==1)
		{
		d_00_10_20_30 = _mm256_load_pd(&C[0+ldc*0]);

		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );

		_mm256_store_pd(&C[0+ldc*0], d_00_10_20_30);
		}
	else
		{
		d_00_10_20_30 = _mm256_load_pd(&C[0+ldc*0]);

		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );

		_mm256_store_pd(&C[0+ldc*0], d_00_10_20_30);
		}

	
	}



/*// packed-normal, 8x4 with data packed in 4*/
/*void kernel_dgemm_pu_nn_8x4_lib4(int kmax, double *A0, double *A1, double *B, int ldb, double *C0, double *C1, int ldc, int alg, int triang)*/
/*	{*/

/*	if(kmax<=0)*/
/*		return;*/
/*	*/
/*	int k;*/
/*	*/
/*	__m256d*/
/*		a_0123, a_4567,*/
/*		b_0, b_1, b_2, b_3,*/
/*		ab_temp,*/
/*		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,*/
/*		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73;*/
/*	*/
/*	c_00_10_20_30 = _mm256_setzero_pd();*/
/*	c_01_11_21_31 = _mm256_setzero_pd();*/
/*	c_02_12_22_32 = _mm256_setzero_pd();*/
/*	c_03_13_23_33 = _mm256_setzero_pd();*/
/*	c_40_50_60_70 = _mm256_setzero_pd();*/
/*	c_41_51_61_71 = _mm256_setzero_pd();*/
/*	c_42_52_62_72 = _mm256_setzero_pd();*/
/*	c_43_53_63_73 = _mm256_setzero_pd();*/

/*	k = 0;*/
/*	if(triang==1)*/
/*		{*/

/*		a_0123 = _mm256_load_pd( &A0[0] );*/
/*		a_4567 = _mm256_load_pd( &A1[0] );*/
/*	*/
/*		b_0 = _mm256_broadcast_sd( &B[0+ldb*0] );*/
/*	*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*	*/

/*		a_0123 = _mm256_load_pd( &A0[4] );*/
/*		a_4567 = _mm256_load_pd( &A1[4] );*/
/*	*/
/*		b_0 = _mm256_broadcast_sd( &B[1+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[1+ldb*1] );*/
/*	*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/

/*	*/
/*		a_0123 = _mm256_load_pd( &A0[8] );*/
/*		a_4567 = _mm256_load_pd( &A1[8] );*/
/*	*/
/*		b_0 = _mm256_broadcast_sd( &B[2+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[2+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[2+ldb*2] );*/
/*	*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/

/*	*/
/*		a_0123 = _mm256_load_pd( &A0[12] );*/
/*		a_4567 = _mm256_load_pd( &A1[12] );*/
/*	*/
/*		b_0 = _mm256_broadcast_sd( &B[3+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[3+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[3+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[3+ldb*3] );*/
/*	*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_3 );*/
/*		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_temp );*/

/*	*/
/*		A0 += 16;*/
/*		A1 += 16;*/
/*		B  += 4;*/
/*		k  += 4;*/
/*		*/
/*		}*/



/*	for(; k<kmax-3; k+=4)*/
/*		{*/
/*		*/
/*		a_0123 = _mm256_load_pd( &A0[0] );*/
/*		a_4567 = _mm256_load_pd( &A1[0] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[0+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[0+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[0+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[0+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_3 );*/
/*		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_temp );*/
/*		*/

/*		a_0123 = _mm256_load_pd( &A0[4] );*/
/*		a_4567 = _mm256_load_pd( &A1[4] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[1+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[1+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[1+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[1+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_3 );*/
/*		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_temp );*/

/*		*/
/*		a_0123 = _mm256_load_pd( &A0[8] );*/
/*		a_4567 = _mm256_load_pd( &A1[8] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[2+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[2+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[2+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[2+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_3 );*/
/*		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_temp );*/

/*		*/
/*		a_0123 = _mm256_load_pd( &A0[12] );*/
/*		a_4567 = _mm256_load_pd( &A1[12] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[3+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[3+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[3+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[3+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_3 );*/
/*		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_temp );*/

/*		*/
/*		A0 += 16;*/
/*		A1 += 16;*/
/*		B  += 4;*/

/*		}*/
/*	*/
/*	if(kmax%4>=2)*/
/*		{*/
/*		*/
/*		a_0123 = _mm256_load_pd( &A0[0] );*/
/*		a_4567 = _mm256_load_pd( &A1[0] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[0+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[0+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[0+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[0+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_3 );*/
/*		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_temp );*/
/*		*/

/*		a_0123 = _mm256_load_pd( &A0[4] );*/
/*		a_4567 = _mm256_load_pd( &A1[4] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[1+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[1+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[1+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[1+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_3 );*/
/*		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_temp );*/

/*		*/
/*		A0 += 8;*/
/*		A1 += 8;*/
/*		B  += 2;*/

/*		}*/

/*	if(kmax%2==1)*/
/*		{*/
/*		*/
/*		a_0123 = _mm256_load_pd( &A0[0] );*/
/*		a_4567 = _mm256_load_pd( &A1[0] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[0+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[0+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[0+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[0+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_0 );*/
/*		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_1 );*/
/*		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_2 );*/
/*		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_4567, b_3 );*/
/*		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_temp );*/
/*		*/

/*		}*/

/*	__m256d*/
/*		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33,*/
/*		d_40_50_60_70, d_41_51_61_71, d_42_52_62_72, d_43_53_63_73;*/

/*	if(alg==0)*/
/*		{*/
/*		_mm256_storeu_pd( &C0[0+ldc*0], c_00_10_20_30 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*1], c_01_11_21_31 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*2], c_02_12_22_32 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*3], c_03_13_23_33 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*0], c_40_50_60_70 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*1], c_41_51_61_71 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*2], c_42_52_62_72 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*3], c_43_53_63_73 );*/
/*		}*/
/*	else if(alg==1)*/
/*		{*/
/*		d_00_10_20_30 = _mm256_loadu_pd(&C0[0+ldc*0]);*/
/*		d_01_11_21_31 = _mm256_loadu_pd(&C0[0+ldc*1]);*/
/*		d_02_12_22_32 = _mm256_loadu_pd(&C0[0+ldc*2]);*/
/*		d_03_13_23_33 = _mm256_loadu_pd(&C0[0+ldc*3]);*/
/*		d_40_50_60_70 = _mm256_loadu_pd(&C1[0+ldc*0]);*/
/*		d_41_51_61_71 = _mm256_loadu_pd(&C1[0+ldc*1]);*/
/*		d_42_52_62_72 = _mm256_loadu_pd(&C1[0+ldc*2]);*/
/*		d_43_53_63_73 = _mm256_loadu_pd(&C1[0+ldc*3]);*/
/*		*/
/*		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );*/
/*		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );*/
/*		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );*/
/*		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );*/
/*		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );*/
/*		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );*/
/*		d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );*/
/*		d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );*/

/*		_mm256_storeu_pd( &C0[0+ldc*0], d_00_10_20_30 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*1], d_01_11_21_31 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*2], d_02_12_22_32 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*3], d_03_13_23_33 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*0], d_40_50_60_70 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*1], d_41_51_61_71 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*2], d_42_52_62_72 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*3], d_43_53_63_73 );*/
/*		}*/
/*	else*/
/*		{*/
/*		d_00_10_20_30 = _mm256_loadu_pd(&C0[0+ldc*0]);*/
/*		d_01_11_21_31 = _mm256_loadu_pd(&C0[0+ldc*1]);*/
/*		d_02_12_22_32 = _mm256_loadu_pd(&C0[0+ldc*2]);*/
/*		d_03_13_23_33 = _mm256_loadu_pd(&C0[0+ldc*3]);*/
/*		d_40_50_60_70 = _mm256_loadu_pd(&C1[0+ldc*0]);*/
/*		d_41_51_61_71 = _mm256_loadu_pd(&C1[0+ldc*1]);*/
/*		d_42_52_62_72 = _mm256_loadu_pd(&C1[0+ldc*2]);*/
/*		d_43_53_63_73 = _mm256_loadu_pd(&C1[0+ldc*3]);*/
/*		*/
/*		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );*/
/*		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );*/
/*		d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );*/
/*		d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );*/
/*		d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );*/
/*		d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );*/
/*		d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, c_42_52_62_72 );*/
/*		d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, c_43_53_63_73 );*/

/*		_mm256_storeu_pd( &C0[0+ldc*0], d_00_10_20_30 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*1], d_01_11_21_31 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*2], d_02_12_22_32 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*3], d_03_13_23_33 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*0], d_40_50_60_70 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*1], d_41_51_61_71 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*2], d_42_52_62_72 );*/
/*		_mm256_storeu_pd( &C1[0+ldc*3], d_43_53_63_73 );*/
/*		}*/

/*	}*/



/*// packed-normal, 4x4 with data packed in 4*/
/*void kernel_dgemm_pu_nn_4x4_lib4(int kmax, double *A0, double *B, int ldb, double *C0, int ldc, int alg, int triang)*/
/*	{*/

/*	if(kmax<=0)*/
/*		return;*/
/*	*/
/*	int k;*/
/*	*/
/*	__m256d*/
/*		a_0123, a_4567,*/
/*		b_0, b_1, b_2, b_3,*/
/*		ab_temp,*/
/*		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,*/
/*		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73;*/
/*	*/
/*	c_00_10_20_30 = _mm256_setzero_pd();*/
/*	c_01_11_21_31 = _mm256_setzero_pd();*/
/*	c_02_12_22_32 = _mm256_setzero_pd();*/
/*	c_03_13_23_33 = _mm256_setzero_pd();*/
/*	c_40_50_60_70 = _mm256_setzero_pd();*/
/*	c_41_51_61_71 = _mm256_setzero_pd();*/
/*	c_42_52_62_72 = _mm256_setzero_pd();*/
/*	c_43_53_63_73 = _mm256_setzero_pd();*/

/*	k = 0;*/
/*	if(triang==1)*/
/*		{*/

/*		a_0123 = _mm256_load_pd( &A0[0] );*/
/*	*/
/*		b_0 = _mm256_broadcast_sd( &B[0+ldb*0] );*/
/*	*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*	*/

/*		a_0123 = _mm256_load_pd( &A0[4] );*/
/*	*/
/*		b_0 = _mm256_broadcast_sd( &B[1+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[1+ldb*1] );*/
/*	*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/

/*	*/
/*		a_0123 = _mm256_load_pd( &A0[8] );*/
/*	*/
/*		b_0 = _mm256_broadcast_sd( &B[2+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[2+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[2+ldb*2] );*/
/*	*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/

/*	*/
/*		a_0123 = _mm256_load_pd( &A0[12] );*/
/*	*/
/*		b_0 = _mm256_broadcast_sd( &B[3+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[3+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[3+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[3+ldb*3] );*/
/*	*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/

/*	*/
/*		A0 += 16;*/
/*		B  += 4;*/
/*		k  += 4;*/
/*		*/
/*		}*/



/*	for(; k<kmax-3; k+=4)*/
/*		{*/
/*		*/
/*		a_0123 = _mm256_load_pd( &A0[0] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[0+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[0+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[0+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[0+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		*/

/*		a_0123 = _mm256_load_pd( &A0[4] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[1+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[1+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[1+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[1+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/

/*		*/
/*		a_0123 = _mm256_load_pd( &A0[8] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[2+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[2+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[2+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[2+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/

/*		*/
/*		a_0123 = _mm256_load_pd( &A0[12] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[3+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[3+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[3+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[3+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/

/*		*/
/*		A0 += 16;*/
/*		B  += 4;*/

/*		}*/
/*	*/
/*	if(kmax%4>=2)*/
/*		{*/
/*		*/
/*		a_0123 = _mm256_load_pd( &A0[0] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[0+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[0+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[0+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[0+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		*/

/*		a_0123 = _mm256_load_pd( &A0[4] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[1+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[1+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[1+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[1+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/

/*		*/
/*		A0 += 8;*/
/*		B  += 2;*/

/*		}*/

/*	if(kmax%2==1)*/
/*		{*/
/*		*/
/*		a_0123 = _mm256_load_pd( &A0[0] );*/
/*		*/
/*		b_0 = _mm256_broadcast_sd( &B[0+ldb*0] );*/
/*		b_1 = _mm256_broadcast_sd( &B[0+ldb*1] );*/
/*		b_2 = _mm256_broadcast_sd( &B[0+ldb*2] );*/
/*		b_3 = _mm256_broadcast_sd( &B[0+ldb*3] );*/
/*		*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_0 );*/
/*		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_1 );*/
/*		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_2 );*/
/*		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );*/
/*		ab_temp = _mm256_mul_pd( a_0123, b_3 );*/
/*		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );*/
/*		*/
/*		}*/

/*	__m256d*/
/*		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33;*/

/*	if(alg==0)*/
/*		{*/
/*		_mm256_storeu_pd( &C0[0+ldc*0], c_00_10_20_30 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*1], c_01_11_21_31 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*2], c_02_12_22_32 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*3], c_03_13_23_33 );*/
/*		}*/
/*	else if(alg==1)*/
/*		{*/
/*		d_00_10_20_30 = _mm256_loadu_pd(&C0[0+ldc*0]);*/
/*		d_01_11_21_31 = _mm256_loadu_pd(&C0[0+ldc*1]);*/
/*		d_02_12_22_32 = _mm256_loadu_pd(&C0[0+ldc*2]);*/
/*		d_03_13_23_33 = _mm256_loadu_pd(&C0[0+ldc*3]);*/
/*		*/
/*		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );*/
/*		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );*/
/*		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );*/
/*		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );*/

/*		_mm256_storeu_pd( &C0[0+ldc*0], d_00_10_20_30 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*1], d_01_11_21_31 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*2], d_02_12_22_32 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*3], d_03_13_23_33 );*/
/*		}*/
/*	else*/
/*		{*/
/*		d_00_10_20_30 = _mm256_loadu_pd(&C0[0+ldc*0]);*/
/*		d_01_11_21_31 = _mm256_loadu_pd(&C0[0+ldc*1]);*/
/*		d_02_12_22_32 = _mm256_loadu_pd(&C0[0+ldc*2]);*/
/*		d_03_13_23_33 = _mm256_loadu_pd(&C0[0+ldc*3]);*/
/*		*/
/*		d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );*/
/*		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );*/
/*		d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );*/
/*		d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );*/

/*		_mm256_storeu_pd( &C0[0+ldc*0], d_00_10_20_30 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*1], d_01_11_21_31 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*2], d_02_12_22_32 );*/
/*		_mm256_storeu_pd( &C0[0+ldc*3], d_03_13_23_33 );*/
/*		}*/

/*	}*/

