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
void kernel_dtrmm_nt_8x4_lib4(int kadd, double *A0, double *A1, double *B, double *D0, double *D1)
	{
	
	const int ldc = 4;

	int k;
	
	__m256d
		zeros,
		a_0123, a_4567, //A_0123,
		b_0, b_1, b_2, b_3,
		b_0101, b_1010, b_2323, b_3232,
		b_0123, b_1032, b_3210, b_2301,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73,
		c_00_11_20_31, c_01_10_21_30, c_03_12_23_32, c_02_13_22_33,
		c_40_51_60_71, c_41_50_61_70, c_43_52_63_72, c_42_53_62_73,
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31,
		c_40_51_62_73, c_41_50_63_72, c_43_52_61_70, c_42_53_60_71;
	
	zeros = _mm256_setzero_pd();
	
	// prefetch
	a_0123        = _mm256_load_pd( &A0[0] );
	a_4567        = _mm256_load_pd( &A1[0] );
	b_0           = _mm256_broadcast_sd( &B[0] );


/*	__builtin_prefetch( A+32 );*/
	c_00_10_20_30 = _mm256_mul_pd( a_0123, b_0 );
	a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
	c_00_11_20_31 = _mm256_blend_pd( c_00_10_20_30, zeros, 0xa );
	c_01_10_21_30 = _mm256_blend_pd( c_00_10_20_30, zeros, 0x5 );
	c_40_50_60_70 = _mm256_mul_pd( a_4567, b_0 );
	a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
	b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
	c_40_51_60_71 = _mm256_blend_pd( c_40_50_60_70, zeros, 0xa );
	c_41_50_61_70 = _mm256_blend_pd( c_40_50_60_70, zeros, 0x5 );
	
	
/*	__builtin_prefetch( A+40 );*/
	ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
	b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
	b_2           = _mm256_broadcast_sd( &B[10] );
	c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
	ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
	a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
	ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
	b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch*/
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
	a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
	c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


/*	__builtin_prefetch( A+48 );*/
	ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
	c_02_12_22_32 = _mm256_mul_pd( a_0123, b_2 );
	c_02_13_22_33 = _mm256_blend_pd( c_02_12_22_32, zeros, 0xa );
	c_03_12_23_32 = _mm256_blend_pd( c_02_12_22_32, zeros, 0x5 );
	c_42_52_62_72 = _mm256_mul_pd( a_4567, b_2 );
	c_42_53_62_73 = _mm256_blend_pd( c_42_52_62_72, zeros, 0xa );
	c_43_52_63_72 = _mm256_blend_pd( c_42_52_62_72, zeros, 0x5 );
	b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
	b_0123        = _mm256_load_pd( &B[12] ); // prefetch*/
	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
	c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
	ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
	a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
	ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
	a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
	c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
	c_00_11_22_33 = _mm256_blend_pd( c_00_11_20_31, c_02_13_22_33, 0xc );
	c_02_13_20_31 = _mm256_blend_pd( c_00_11_20_31, c_02_13_22_33, 0x3 );
	c_01_10_23_32 = _mm256_blend_pd( c_01_10_21_30, c_03_12_23_32, 0xc );
	c_03_12_21_30 = _mm256_blend_pd( c_01_10_21_30, c_03_12_23_32, 0x3 );
	c_40_51_62_73 = _mm256_blend_pd( c_40_51_60_71, c_42_53_62_73, 0xc );
	c_42_53_60_71 = _mm256_blend_pd( c_40_51_60_71, c_42_53_62_73, 0x3 );
	c_41_50_63_72 = _mm256_blend_pd( c_41_50_61_70, c_43_52_63_72, 0xc );
	c_43_52_61_70 = _mm256_blend_pd( c_41_50_61_70, c_43_52_63_72, 0x3 );

/*	__builtin_prefetch( A+56 );*/
	ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
	b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
	b_0123        = _mm256_load_pd( &B[16] ); // prefetch
	c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
	c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
	ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
	b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
	c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
	c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
	ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
	b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
	c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
	c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
	ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
	a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
	ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
	a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
	c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
	c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
	
	A0 += 16;
	A1 += 16;
	B  += 16;

	for(k=4; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );


/*	__builtin_prefetch( A+48 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );


/*	__builtin_prefetch( A+56 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kadd%4>=2)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kadd%2==1)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
/*		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch*/
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
/*		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch*/
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		}

	__m256d
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31,
		c_40_50_62_72, c_41_51_63_73, c_42_52_60_70, c_43_53_61_71;
	
	c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
	c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
	c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
	_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
	c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
	_mm256_store_pd( &D0[0+ldc*2], c_02_12_22_32 );

	c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
	c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
	c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
	_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
	c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
	_mm256_store_pd( &D0[0+ldc*3], c_03_13_23_33 );

	c_40_50_62_72 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0xa );
	c_42_52_60_70 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0xa );
	c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
	_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
	c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
	_mm256_store_pd( &D1[0+ldc*2], c_42_52_62_72 );

	c_41_51_63_73 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0x5 );
	c_43_53_61_71 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0x5 );
	c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
	_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
	c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
	_mm256_store_pd( &D1[0+ldc*3], c_43_53_63_73 );

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dtrmm_nt_4x4_lib4(int kadd, double *A, double *B, double *D)
	{
	
	const int ldc = 4;

	int k;
	
	__m256d
		zeros,
		a_0123,
		b_0, b_1, b_2, b_3,
		b_0101, b_1010, b_2323, b_3232,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_00_11_20_31, c_01_10_21_30, c_03_12_23_32, c_02_13_22_33,
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
	zeros = _mm256_setzero_pd();
	
	// prefetch
	a_0123        = _mm256_load_pd( &A[0] );
	b_0           = _mm256_broadcast_sd( &B[0] );


/*	__builtin_prefetch( A+32 );*/
	c_00_10_20_30 = _mm256_mul_pd( a_0123, b_0 );
	a_0123        = _mm256_load_pd( &A[4] ); // prefetch
	c_00_11_20_31 = _mm256_blend_pd( c_00_10_20_30, zeros, 0xa );
	c_01_10_21_30 = _mm256_blend_pd( c_00_10_20_30, zeros, 0x5 );
	b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
	
	
/*	__builtin_prefetch( A+40 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
	b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
	b_2           = _mm256_broadcast_sd( &B[10] );
	ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
	a_0123        = _mm256_load_pd( &A[8] ); // prefetch
	b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch*/
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );


/*	__builtin_prefetch( A+48 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
	c_02_12_22_32 = _mm256_mul_pd( a_0123, b_2 );
	c_02_13_22_33 = _mm256_blend_pd( c_02_12_22_32, zeros, 0xa );
	c_03_12_23_32 = _mm256_blend_pd( c_02_12_22_32, zeros, 0x5 );
	b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
	b_0123        = _mm256_load_pd( &B[12] ); // prefetch*/
	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
	ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
	a_0123        = _mm256_load_pd( &A[12] ); // prefetch
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
	c_00_11_22_33 = _mm256_blend_pd( c_00_11_20_31, c_02_13_22_33, 0xc );
	c_02_13_20_31 = _mm256_blend_pd( c_00_11_20_31, c_02_13_22_33, 0x3 );
	c_01_10_23_32 = _mm256_blend_pd( c_01_10_21_30, c_03_12_23_32, 0xc );
	c_03_12_21_30 = _mm256_blend_pd( c_01_10_21_30, c_03_12_23_32, 0x3 );

/*	__builtin_prefetch( A+56 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
	b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
	b_0123        = _mm256_load_pd( &B[16] ); // prefetch
	c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
	ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
	b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
	c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
	ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
	b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
	c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
	ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
	a_0123        = _mm256_load_pd( &A[16] ); // prefetch
	c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
	
	A += 16;
	B += 16;

	for(k=4; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kadd%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kadd%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch*/
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		}

	__m256d
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31;
	
	c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
	c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
	c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
	_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
	c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
	_mm256_store_pd( &D[0+ldc*2], c_02_12_22_32 );

	c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
	c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
	c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
	_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
	c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
	_mm256_store_pd( &D[0+ldc*3], c_03_13_23_33 );

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dtrmm_nt_4x4_lib4_old(int kadd, double *A, double *B, double *D)
	{
	
/*	if(kmax<=0)*/
/*		return;*/
	
	const int ldc = 4;

	int k;
	
	__m256d
		a_0123, //A_0123,
		b_0, b_1, b_2, b_3,
		ab_temp, // temporary results
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33;
	
	// prefetch
	a_0123        = _mm256_load_pd( &A[0] );
	b_0           = _mm256_broadcast_sd( &B[0] );
	b_1           = _mm256_broadcast_sd( &B[5] );
	b_2           = _mm256_broadcast_sd( &B[10] );
	b_3           = _mm256_broadcast_sd( &B[15] );

	// zero registers
	c_00_10_20_30 = _mm256_mul_pd( a_0123, b_0 );
	a_0123        = _mm256_load_pd( &A[4] ); // prefetch
	b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
	
	
/*	__builtin_prefetch( A+40 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
	c_01_11_21_31 = _mm256_mul_pd( a_0123, b_1 );
	a_0123        = _mm256_load_pd( &A[8] ); // prefetch
	b_1           = _mm256_broadcast_sd( &B[9] );


/*	__builtin_prefetch( A+48 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	b_0           = _mm256_broadcast_sd( &B[12] ); // prefetch
	ab_temp       = _mm256_mul_pd( a_0123, b_1 );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
	b_1           = _mm256_broadcast_sd( &B[13] );
	c_02_12_22_32 = _mm256_mul_pd( a_0123, b_2 );
	a_0123        = _mm256_load_pd( &A[12] ); // prefetch
	b_2           = _mm256_broadcast_sd( &B[14] );


/*	__builtin_prefetch( A+56 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	b_0           = _mm256_broadcast_sd( &B[16] ); // prefetch
	ab_temp       = _mm256_mul_pd( a_0123, b_1 );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
	b_1           = _mm256_broadcast_sd( &B[17] );
	ab_temp       = _mm256_mul_pd( a_0123, b_2 );
	c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
	b_2           = _mm256_broadcast_sd( &B[18] );
	c_03_13_23_33 = _mm256_mul_pd( a_0123, b_3 );
	a_0123        = _mm256_load_pd( &A[16] ); // prefetch
	b_3           = _mm256_broadcast_sd( &B[19] );
	
	A += 16;
	B += 16;

	for(k=4; k<kadd-3; k+=4)
		{
		
	/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[5] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[6] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[7] );
	
	
	/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[9] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[10] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[11] );


	/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[13] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[14] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[15] );


	/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[17] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[18] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[19] );
		
		A += 16;
		B += 16;

		}
	
	if(kadd%4>=2)
		{
		
	/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[5] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[6] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[7] );
	
	
	/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[9] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[10] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[11] );
		
		A += 8;
		B += 8;

		}

	if(kadd%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		
		}

	_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &D[0+ldc*2], c_02_12_22_32 );
	_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
	_mm256_store_pd( &D[0+ldc*3], c_03_13_23_33 );

	}



/*inline void corner_dtrmm_pp_nt_8x3_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_nt_8x3_lib4(double *A0, double *A1, double *B, double *C0, double *C1)
	{
	
	const int ldc = 4;

	__m256d
		ab_temp,
		a_00_10_20_30, a_01_11_21_31, a_02_12_22_32, a_40_50_60_70, a_41_51_61_71, a_42_52_62_72,
		b_00, b_10, b_20, b_11, b_21, b_22,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_40_50_60_70, c_41_51_61_71, c_42_52_62_72;
	
	a_00_10_20_30 = _mm256_load_pd( &A0[0+4*0] );
	a_40_50_60_70 = _mm256_load_pd( &A1[0+4*0] );
	a_01_11_21_31 = _mm256_load_pd( &A0[0+4*1] );
	a_41_51_61_71 = _mm256_load_pd( &A1[0+4*1] );
	a_02_12_22_32 = _mm256_load_pd( &A0[0+4*2] );
	a_42_52_62_72 = _mm256_load_pd( &A1[0+4*2] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	b_10 = _mm256_broadcast_sd( &B[0+4*1] );
	b_20 = _mm256_broadcast_sd( &B[0+4*2] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );
	c_40_50_60_70 = _mm256_mul_pd( a_40_50_60_70, b_00 );

	ab_temp = _mm256_mul_pd( a_01_11_21_31, b_10 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	ab_temp = _mm256_mul_pd( a_41_51_61_71, b_10 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );

	ab_temp = _mm256_mul_pd( a_02_12_22_32, b_20 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	ab_temp = _mm256_mul_pd( a_42_52_62_72, b_20 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );

	_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );
	
	// second column 
	b_11 = _mm256_broadcast_sd( &B[1+4*1] );
	b_21 = _mm256_broadcast_sd( &B[1+4*2] );

	c_01_11_21_31 = _mm256_mul_pd( a_01_11_21_31, b_11 );
	c_41_51_61_71 = _mm256_mul_pd( a_41_51_61_71, b_11 );

	ab_temp = _mm256_mul_pd( a_02_12_22_32, b_21 );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
	ab_temp = _mm256_mul_pd( a_42_52_62_72, b_21 );
	c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
	
	_mm256_store_pd( &C0[0+ldc*1], c_01_11_21_31 );
	_mm256_store_pd( &C1[0+ldc*1], c_41_51_61_71 );
	
	// third column 
	b_22 = _mm256_broadcast_sd( &B[2+4*2] );

	c_02_12_22_32 = _mm256_mul_pd( a_02_12_22_32, b_22 );
	c_42_52_62_72 = _mm256_mul_pd( a_42_52_62_72, b_22 );

	_mm256_store_pd( &C0[0+ldc*2], c_02_12_22_32 );
	_mm256_store_pd( &C1[0+ldc*2], c_42_52_62_72 );

	}
	


/*inline void corner_dtrmm_pp_nt_8x2_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_nt_8x2_lib4(double *A0, double *A1, double *B, double *C0, double *C1)
	{
	
	const int ldc = 4;

	__m256d
		ab_temp,
		a_00_10_20_30, a_01_11_21_31, a_40_50_60_70, a_41_51_61_71,
		b_00, b_10, b_11,
		c_00_10_20_30, c_01_11_21_31, c_40_50_60_70, c_41_51_61_71;
	
	a_00_10_20_30 = _mm256_load_pd( &A0[0+4*0] );
	a_40_50_60_70 = _mm256_load_pd( &A1[0+4*0] );
	a_01_11_21_31 = _mm256_load_pd( &A0[0+4*1] );
	a_41_51_61_71 = _mm256_load_pd( &A1[0+4*1] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	b_10 = _mm256_broadcast_sd( &B[0+4*1] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );
	c_40_50_60_70 = _mm256_mul_pd( a_40_50_60_70, b_00 );

	ab_temp = _mm256_mul_pd( a_01_11_21_31, b_10 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	ab_temp = _mm256_mul_pd( a_41_51_61_71, b_10 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
	
	_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );

	// second column 
	b_11 = _mm256_broadcast_sd( &B[1+4*1] );

	c_01_11_21_31 = _mm256_mul_pd( a_01_11_21_31, b_11 );
	c_41_51_61_71 = _mm256_mul_pd( a_41_51_61_71, b_11 );
	
	_mm256_store_pd( &C0[0+ldc*1], c_01_11_21_31 );
	_mm256_store_pd( &C1[0+ldc*1], c_41_51_61_71 );
	
	}



/*inline void corner_dtrmm_pp_nt_8x1_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_nt_8x1_lib4(double *A0, double *A1, double *B, double *C0, double *C1)
	{
	
	const int ldc = 4;

	__m256d
		a_00_10_20_30, a_40_50_60_70,
		b_00,
		c_00_10_20_30, c_40_50_60_70;
	
	a_00_10_20_30 = _mm256_load_pd( &A0[0+4*0] );
	a_40_50_60_70 = _mm256_load_pd( &A1[0+4*0] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );
	c_40_50_60_70 = _mm256_mul_pd( a_40_50_60_70, b_00 );

	_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );
	
	}


/*inline void corner_dtrmm_pp_nt_4x3_lib4(double *A, double *B, double *C, int ldc)*/
void corner_dtrmm_nt_4x3_lib4(double *A, double *B, double *C)
	{
	
	const int ldc = 4;

	__m256d
		ab_temp,
		a_00_10_20_30, a_01_11_21_31, a_02_12_22_32,
		b_00, b_10, b_20, b_11, b_21, b_22,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32;
	
	a_00_10_20_30 = _mm256_load_pd( &A[0+4*0] );
	a_01_11_21_31 = _mm256_load_pd( &A[0+4*1] );
	a_02_12_22_32 = _mm256_load_pd( &A[0+4*2] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	b_10 = _mm256_broadcast_sd( &B[0+4*1] );
	b_20 = _mm256_broadcast_sd( &B[0+4*2] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );

	ab_temp = _mm256_mul_pd( a_01_11_21_31, b_10 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );

	ab_temp = _mm256_mul_pd( a_02_12_22_32, b_20 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );

	_mm256_store_pd( &C[0+ldc*0], c_00_10_20_30 );
	
	// second column 
	b_11 = _mm256_broadcast_sd( &B[1+4*1] );
	b_21 = _mm256_broadcast_sd( &B[1+4*2] );

	c_01_11_21_31 = _mm256_mul_pd( a_01_11_21_31, b_11 );

	ab_temp = _mm256_mul_pd( a_02_12_22_32, b_21 );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
	
	_mm256_store_pd( &C[0+ldc*1], c_01_11_21_31 );
	
	// third column 
	b_22 = _mm256_broadcast_sd( &B[2+4*2] );

	c_02_12_22_32 = _mm256_mul_pd( a_02_12_22_32, b_22 );

	_mm256_store_pd( &C[0+ldc*2], c_02_12_22_32 );

	}
	


/*inline void corner_dtrmm_pp_nt_4x2_lib4(double *A, double *B, double *C, int ldc)*/
void corner_dtrmm_nt_4x2_lib4(double *A, double *B, double *C)
	{
	
	const int ldc = 4;

	__m256d
		ab_temp,
		a_00_10_20_30, a_01_11_21_31,
		b_00, b_10, b_11,
		c_00_10_20_30, c_01_11_21_31;
	
	a_00_10_20_30 = _mm256_load_pd( &A[0+4*0] );
	a_01_11_21_31 = _mm256_load_pd( &A[0+4*1] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	b_10 = _mm256_broadcast_sd( &B[0+4*1] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );

	ab_temp = _mm256_mul_pd( a_01_11_21_31, b_10 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	
	_mm256_store_pd( &C[0+ldc*0], c_00_10_20_30 );

	// second column 
	b_11 = _mm256_broadcast_sd( &B[1+4*1] );

	c_01_11_21_31 = _mm256_mul_pd( a_01_11_21_31, b_11 );
	
	_mm256_store_pd( &C[0+ldc*1], c_01_11_21_31 );
	
	}



/*inline void corner_dtrmm_pp_nt_4x1_lib4(double *A, double *B, double *C, int ldc)*/
void corner_dtrmm_nt_4x1_lib4(double *A, double *B, double *C)
	{
	
	const int ldc = 4;

	__m256d
		a_00_10_20_30,
		b_00,
		c_00_10_20_30;
	
	a_00_10_20_30 = _mm256_load_pd( &A[0+4*0] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );

	_mm256_store_pd( &C[0+ldc*0], c_00_10_20_30 );
	
	}
