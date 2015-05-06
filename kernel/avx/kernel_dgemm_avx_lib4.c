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

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX



// normal-transposed, 8x4 with data packed in 4
//void kernel_dgemm_nt_8x4_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg)
void kernel_dgemm_nt_8x4_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int ldc = 4;
	
	int k;
	
	__m256d
		a_0123, a_4567, A_0123, A_4567,
		b_0123, b_1032, b_3210, b_2301,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31,
		c_40_51_62_73, c_41_50_63_72, c_43_52_61_70, c_42_53_60_71;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A0[0] );
	a_4567 = _mm256_load_pd( &A1[0] );
	b_0123 = _mm256_load_pd( &B[0] );

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();
	c_40_51_62_73 = _mm256_setzero_pd();
	c_41_50_63_72 = _mm256_setzero_pd();
	c_43_52_61_70 = _mm256_setzero_pd();
	c_42_53_60_71 = _mm256_setzero_pd();


#if 0
	for(k=0; k<kmax-3; k+=4) // TODO prefetch A0 and A1 using 2 extra registers ????????????????
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
	
	if(kmax%4>=2)
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

	if(kmax%2==1)
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
#else

	for(k=0; k<kmax-3; k+=4) // TODO prefetch A0 and A1 using 2 extra registers ????????????????
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		A_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		A_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_1032 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp0 );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( A_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_3210 );
		b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_1032 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp0 );


/*	__builtin_prefetch( A+48 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		A_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		A_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_1032 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp0 );


/*	__builtin_prefetch( A+56 );*/
		ab_tmp0       = _mm256_mul_pd( A_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_3210 );
		b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_1032 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp0 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		A_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		A_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_1032 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp0 );
		
		
		ab_tmp0       = _mm256_mul_pd( A_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_3210 );
		b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( A_4567, b_1032 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp0 );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_0123 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
/*		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch*/
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		ab_tmp0       = _mm256_mul_pd( a_4567, b_1032 );
/*		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch*/
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp0 );
		
		}
#endif

	__m256d
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31,
		c_40_50_62_72, c_41_51_63_73, c_42_52_60_70, c_43_53_61_71,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73,
		c_00_01_22_23, c_10_11_32_33, c_02_03_20_21, c_12_13_30_31,
		c_40_41_62_63, c_50_51_72_73, c_42_43_60_61, c_52_53_70_71,
		c_00_01_02_03, c_10_11_12_13, c_20_21_22_23, c_30_31_32_33,
		c_40_41_42_43, c_50_51_52_53, c_60_61_62_63, c_70_71_72_73,
		c_00_01_20_21, c_10_11_30_31, c_02_03_22_23, c_12_13_32_33,
		c_40_41_60_61, c_50_51_70_71, c_42_43_62_63, c_52_53_72_73,
		d_00_01_02_03, d_10_11_12_13, d_20_21_22_23, d_30_31_32_33,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33,
		d_40_50_60_70, d_41_51_61_71, d_42_52_62_72, d_43_53_63_73,
		d_00_10_02_12, d_01_11_03_13, d_20_30_22_32, d_21_31_23_33; 

	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0)
			{
			c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
			c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
			c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
			c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );

			c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
			c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
			c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
			c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );

			_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
			_mm256_store_pd( &D0[0+ldc*2], c_02_12_22_32 );
			_mm256_store_pd( &D0[0+ldc*3], c_03_13_23_33 );

			c_40_50_62_72 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0xa );
			c_41_51_63_73 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0x5 );
			c_42_52_60_70 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0xa );
			c_43_53_61_71 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0x5 );
			
			c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
			c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
			c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
			c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );

			_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
			_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
			_mm256_store_pd( &D1[0+ldc*2], c_42_52_62_72 );
			_mm256_store_pd( &D1[0+ldc*3], c_43_53_63_73 );
			}
		else // transposed
			{
			c_00_01_22_23 = _mm256_unpacklo_pd( c_00_11_22_33, c_01_10_23_32 );
			c_10_11_32_33 = _mm256_unpackhi_pd( c_01_10_23_32, c_00_11_22_33 );
			c_02_03_20_21 = _mm256_unpacklo_pd( c_02_13_20_31, c_03_12_21_30 );
			c_12_13_30_31 = _mm256_unpackhi_pd( c_03_12_21_30, c_02_13_20_31 );

			c_00_01_02_03 = _mm256_permute2f128_pd( c_00_01_22_23, c_02_03_20_21, 0x20 );
			c_10_11_12_13 = _mm256_permute2f128_pd( c_10_11_32_33, c_12_13_30_31, 0x20 );
			c_20_21_22_23 = _mm256_permute2f128_pd( c_02_03_20_21, c_00_01_22_23, 0x31 );
			c_30_31_32_33 = _mm256_permute2f128_pd( c_12_13_30_31, c_10_11_32_33, 0x31 );

			_mm256_store_pd( &D0[0+ldc*0], c_00_01_02_03 );
			_mm256_store_pd( &D0[0+ldc*1], c_10_11_12_13 );
			_mm256_store_pd( &D0[0+ldc*2], c_20_21_22_23 );
			_mm256_store_pd( &D0[0+ldc*3], c_30_31_32_33 );

			c_40_41_62_63 = _mm256_shuffle_pd( c_40_51_62_73, c_41_50_63_72, 0x0 );
			c_50_51_72_73 = _mm256_shuffle_pd( c_41_50_63_72, c_40_51_62_73, 0xf );
			c_42_43_60_61 = _mm256_shuffle_pd( c_42_53_60_71, c_43_52_61_70, 0x0 );
			c_52_53_70_71 = _mm256_shuffle_pd( c_43_52_61_70, c_42_53_60_71, 0xf );

			c_40_41_42_43 = _mm256_permute2f128_pd( c_40_41_62_63, c_42_43_60_61, 0x20 );
			c_50_51_52_53 = _mm256_permute2f128_pd( c_50_51_72_73, c_52_53_70_71, 0x20 );
			c_60_61_62_63 = _mm256_permute2f128_pd( c_42_43_60_61, c_40_41_62_63, 0x31 );
			c_70_71_72_73 = _mm256_permute2f128_pd( c_52_53_70_71, c_50_51_72_73, 0x31 );

			_mm256_store_pd( &D0[0+ldc*4], c_40_41_42_43 );
			_mm256_store_pd( &D0[0+ldc*5], c_50_51_52_53 );
			_mm256_store_pd( &D0[0+ldc*6], c_60_61_62_63 );
			_mm256_store_pd( &D0[0+ldc*7], c_70_71_72_73 );
			}
		}
	else 
		{
		if(tc==0) // C
			{

			// AB + C
			c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
			c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
			c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
			c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
			
			c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
			c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
			c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
			c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );

			c_40_50_62_72 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0xa );
			c_41_51_63_73 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0x5 );
			c_42_52_60_70 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0xa );
			c_43_53_61_71 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0x5 );
			
			c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
			c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
			c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
			c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );

			d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
			d_02_12_22_32 = _mm256_load_pd( &C0[0+ldc*2] );
			d_03_13_23_33 = _mm256_load_pd( &C0[0+ldc*3] );
			
			d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
			d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
			d_42_52_62_72 = _mm256_load_pd( &C1[0+ldc*2] );
			d_43_53_63_73 = _mm256_load_pd( &C1[0+ldc*3] );
			
			if(alg==1) // AB = A*B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );

				d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
				d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
				d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );
				}
			else // AB = - A*B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );

				d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
				d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, c_42_52_62_72 );
				d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, c_43_53_63_73 );
				}

			if(td==0) // AB + C 
				{
				_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
				_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
				_mm256_store_pd( &D0[0+ldc*2], d_02_12_22_32 );
				_mm256_store_pd( &D0[0+ldc*3], d_03_13_23_33 );

				_mm256_store_pd( &D1[0+ldc*0], d_40_50_60_70 );
				_mm256_store_pd( &D1[0+ldc*1], d_41_51_61_71 );
				_mm256_store_pd( &D1[0+ldc*2], d_42_52_62_72 );
				_mm256_store_pd( &D1[0+ldc*3], d_43_53_63_73 );
				}
			else // t(AB + C)
				{
				c_00_01_20_21 = _mm256_unpacklo_pd( d_00_10_20_30, d_01_11_21_31 );
				c_10_11_30_31 = _mm256_unpackhi_pd( d_00_10_20_30, d_01_11_21_31 );
				c_02_03_22_23 = _mm256_unpacklo_pd( d_02_12_22_32, d_03_13_23_33 );
				c_12_13_32_33 = _mm256_unpackhi_pd( d_02_12_22_32, d_03_13_23_33 );

				c_00_01_02_03 = _mm256_permute2f128_pd( c_00_01_20_21, c_02_03_22_23, 0x20 );
				c_20_21_22_23 = _mm256_permute2f128_pd( c_00_01_20_21, c_02_03_22_23, 0x31 );
				c_10_11_12_13 = _mm256_permute2f128_pd( c_10_11_30_31, c_12_13_32_33, 0x20 );
				c_30_31_32_33 = _mm256_permute2f128_pd( c_10_11_30_31, c_12_13_32_33, 0x31 );

				_mm256_store_pd( &D0[0+ldc*0], c_00_01_02_03 );
				_mm256_store_pd( &D0[0+ldc*1], c_10_11_12_13 );
				_mm256_store_pd( &D0[0+ldc*2], c_20_21_22_23 );
				_mm256_store_pd( &D0[0+ldc*3], c_30_31_32_33 );

				c_40_41_60_61 = _mm256_unpacklo_pd( d_40_50_60_70, d_41_51_61_71 );
				c_50_51_70_71 = _mm256_unpackhi_pd( d_40_50_60_70, d_41_51_61_71 );
				c_42_43_62_63 = _mm256_unpacklo_pd( d_42_52_62_72, d_43_53_63_73 );
				c_52_53_72_73 = _mm256_unpackhi_pd( d_42_52_62_72, d_43_53_63_73 );

				c_40_41_42_43 = _mm256_permute2f128_pd( c_40_41_60_61, c_42_43_62_63, 0x20 );
				c_60_61_62_63 = _mm256_permute2f128_pd( c_40_41_60_61, c_42_43_62_63, 0x31 );
				c_50_51_52_53 = _mm256_permute2f128_pd( c_50_51_70_71, c_52_53_72_73, 0x20 );
				c_70_71_72_73 = _mm256_permute2f128_pd( c_50_51_70_71, c_52_53_72_73, 0x31 );

				_mm256_store_pd( &D0[0+ldc*4], c_40_41_42_43 );
				_mm256_store_pd( &D0[0+ldc*5], c_50_51_52_53 );
				_mm256_store_pd( &D0[0+ldc*6], c_60_61_62_63 );
				_mm256_store_pd( &D0[0+ldc*7], c_70_71_72_73 );
				}

			}
		else // t(C)
			{

			c_00_01_22_23 = _mm256_unpacklo_pd( c_00_11_22_33, c_01_10_23_32 );
			c_10_11_32_33 = _mm256_unpackhi_pd( c_01_10_23_32, c_00_11_22_33 );
			c_02_03_20_21 = _mm256_unpacklo_pd( c_02_13_20_31, c_03_12_21_30 );
			c_12_13_30_31 = _mm256_unpackhi_pd( c_03_12_21_30, c_02_13_20_31 );

			c_00_01_02_03 = _mm256_permute2f128_pd( c_00_01_22_23, c_02_03_20_21, 0x20 );
			c_10_11_12_13 = _mm256_permute2f128_pd( c_10_11_32_33, c_12_13_30_31, 0x20 );
			c_20_21_22_23 = _mm256_permute2f128_pd( c_02_03_20_21, c_00_01_22_23, 0x31 );
			c_30_31_32_33 = _mm256_permute2f128_pd( c_12_13_30_31, c_10_11_32_33, 0x31 );

			d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
			d_02_12_22_32 = _mm256_load_pd( &C0[0+ldc*2] );
			d_03_13_23_33 = _mm256_load_pd( &C0[0+ldc*3] );

			c_40_41_62_63 = _mm256_unpacklo_pd( c_40_51_62_73, c_41_50_63_72 );
			c_50_51_72_73 = _mm256_unpackhi_pd( c_41_50_63_72, c_40_51_62_73 );
			c_42_43_60_61 = _mm256_unpacklo_pd( c_42_53_60_71, c_43_52_61_70 );
			c_52_53_70_71 = _mm256_unpackhi_pd( c_43_52_61_70, c_42_53_60_71 );

			c_40_41_42_43 = _mm256_permute2f128_pd( c_40_41_62_63, c_42_43_60_61, 0x20 );
			c_50_51_52_53 = _mm256_permute2f128_pd( c_50_51_72_73, c_52_53_70_71, 0x20 );
			c_60_61_62_63 = _mm256_permute2f128_pd( c_42_43_60_61, c_40_41_62_63, 0x31 );
			c_70_71_72_73 = _mm256_permute2f128_pd( c_52_53_70_71, c_50_51_72_73, 0x31 );

			d_40_50_60_70 = _mm256_load_pd( &C0[0+ldc*4] );
			d_41_51_61_71 = _mm256_load_pd( &C0[0+ldc*5] );
			d_42_52_62_72 = _mm256_load_pd( &C0[0+ldc*6] );
			d_43_53_63_73 = _mm256_load_pd( &C0[0+ldc*7] );

			if(alg==1) // AB = A*B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_01_02_03 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_10_11_12_13 );
				d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_20_21_22_23 );
				d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_30_31_32_33 );

				d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_41_42_43 );
				d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_50_51_52_53 );
				d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_60_61_62_63 );
				d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_70_71_72_73 );
				}
			else // AB = - A*B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_01_02_03 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_10_11_12_13 );
				d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_20_21_22_23 );
				d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_30_31_32_33 );

				d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_41_42_43 );
				d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_50_51_52_53 );
				d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, c_60_61_62_63 );
				d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, c_70_71_72_73 );
				}

			if(td==0) // t( t(AB) + C )
				{
				c_00_01_20_21 = _mm256_unpacklo_pd( d_00_10_20_30, d_01_11_21_31 );
				c_10_11_30_31 = _mm256_unpackhi_pd( d_00_10_20_30, d_01_11_21_31 );
				c_02_03_22_23 = _mm256_unpacklo_pd( d_02_12_22_32, d_03_13_23_33 );
				c_12_13_32_33 = _mm256_unpackhi_pd( d_02_12_22_32, d_03_13_23_33 );

				c_00_01_02_03 = _mm256_permute2f128_pd( c_00_01_20_21, c_02_03_22_23, 0x20 );
				c_20_21_22_23 = _mm256_permute2f128_pd( c_00_01_20_21, c_02_03_22_23, 0x31 );
				c_10_11_12_13 = _mm256_permute2f128_pd( c_10_11_30_31, c_12_13_32_33, 0x20 );
				c_30_31_32_33 = _mm256_permute2f128_pd( c_10_11_30_31, c_12_13_32_33, 0x31 );

				_mm256_store_pd( &D0[0+ldc*0], c_00_01_02_03 );
				_mm256_store_pd( &D0[0+ldc*1], c_10_11_12_13 );
				_mm256_store_pd( &D0[0+ldc*2], c_20_21_22_23 );
				_mm256_store_pd( &D0[0+ldc*3], c_30_31_32_33 );

				c_40_41_60_61 = _mm256_unpacklo_pd( d_40_50_60_70, d_41_51_61_71 );
				c_50_51_70_71 = _mm256_unpackhi_pd( d_40_50_60_70, d_41_51_61_71 );
				c_42_43_62_63 = _mm256_unpacklo_pd( d_42_52_62_72, d_43_53_63_73 );
				c_52_53_72_73 = _mm256_unpackhi_pd( d_42_52_62_72, d_43_53_63_73 );

				c_40_41_42_43 = _mm256_permute2f128_pd( c_40_41_60_61, c_42_43_62_63, 0x20 );
				c_60_61_62_63 = _mm256_permute2f128_pd( c_40_41_60_61, c_42_43_62_63, 0x31 );
				c_50_51_52_53 = _mm256_permute2f128_pd( c_50_51_70_71, c_52_53_72_73, 0x20 );
				c_70_71_72_73 = _mm256_permute2f128_pd( c_50_51_70_71, c_52_53_72_73, 0x31 );

				_mm256_store_pd( &D1[0+ldc*0], c_40_41_42_43 );
				_mm256_store_pd( &D1[0+ldc*1], c_50_51_52_53 );
				_mm256_store_pd( &D1[0+ldc*2], c_60_61_62_63 );
				_mm256_store_pd( &D1[0+ldc*3], c_70_71_72_73 );
				}
			else // t(AB) + C
				{
				_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
				_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
				_mm256_store_pd( &D0[0+ldc*2], d_02_12_22_32 );
				_mm256_store_pd( &D0[0+ldc*3], d_03_13_23_33 );

				_mm256_store_pd( &D0[0+ldc*4], d_40_50_60_70 );
				_mm256_store_pd( &D0[0+ldc*5], d_41_51_61_71 );
				_mm256_store_pd( &D0[0+ldc*6], d_42_52_62_72 );
				_mm256_store_pd( &D0[0+ldc*7], d_43_53_63_73 );
				}

			}

		}

	}



// normal-transposed, 8x4 with data packed in 4
void kernel_dgemm_nt_m8x4_lib4(int m, int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int ldc = 4;
	
	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31,
		c_40_51_62_73, c_41_50_63_72, c_43_52_61_70, c_42_53_60_71;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A0[0] );
	a_4567 = _mm256_load_pd( &A1[0] );
	b_0123 = _mm256_load_pd( &B[0] );

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
	
	if(kmax%4>=2)
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

	if(kmax%2==1)
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

	__m256i
		mask_i;

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

	if(m>=8)
		{

		if(alg==0) // C = A * B'
			{
			_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
			_mm256_store_pd( &D0[0+ldc*2], c_02_12_22_32 );
			_mm256_store_pd( &D0[0+ldc*3], c_03_13_23_33 );
			_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
			_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
			_mm256_store_pd( &D1[0+ldc*2], c_42_52_62_72 );
			_mm256_store_pd( &D1[0+ldc*3], c_43_53_63_73 );
			}
		else 
			{
			d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
			d_02_12_22_32 = _mm256_load_pd( &C0[0+ldc*2] );
			d_03_13_23_33 = _mm256_load_pd( &C0[0+ldc*3] );
			d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
			d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
			d_42_52_62_72 = _mm256_load_pd( &C1[0+ldc*2] );
			d_43_53_63_73 = _mm256_load_pd( &C1[0+ldc*3] );
		
			if(alg==1) // C += A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
				d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
				d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
				d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );
				}
			else // C -= A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );
				d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
				d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, c_42_52_62_72 );
				d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, c_43_53_63_73 );
				}

			_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
			_mm256_store_pd( &D0[0+ldc*2], d_02_12_22_32 );
			_mm256_store_pd( &D0[0+ldc*3], d_03_13_23_33 );
			_mm256_store_pd( &D1[0+ldc*0], d_40_50_60_70 );
			_mm256_store_pd( &D1[0+ldc*1], d_41_51_61_71 );
			_mm256_store_pd( &D1[0+ldc*2], d_42_52_62_72 );
			_mm256_store_pd( &D1[0+ldc*3], d_43_53_63_73 );
			}

		}
	else
		{

		const double mask_f[] = {4.5, 5.5, 6.5, 7.5};
		double m_f = m;

		mask_i = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( mask_f ), _mm256_broadcast_sd( &m_f ) ) );

		if(alg==0) // C = A * B'
			{
			_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
			_mm256_store_pd( &D0[0+ldc*2], c_02_12_22_32 );
			_mm256_store_pd( &D0[0+ldc*3], c_03_13_23_33 );
			_mm256_maskstore_pd( &D1[0+ldc*0], mask_i, c_40_50_60_70 );
			_mm256_maskstore_pd( &D1[0+ldc*1], mask_i, c_41_51_61_71 );
			_mm256_maskstore_pd( &D1[0+ldc*2], mask_i, c_42_52_62_72 );
			_mm256_maskstore_pd( &D1[0+ldc*3], mask_i, c_43_53_63_73 );
			}
		else 
			{
			d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
			d_02_12_22_32 = _mm256_load_pd( &C0[0+ldc*2] );
			d_03_13_23_33 = _mm256_load_pd( &C0[0+ldc*3] );
			d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
			d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
			d_42_52_62_72 = _mm256_load_pd( &C1[0+ldc*2] );
			d_43_53_63_73 = _mm256_load_pd( &C1[0+ldc*3] );
		
			if(alg==1) // C += A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
				d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
				d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
				d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );
				}
			else // C -= A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );
				d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
				d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, c_42_52_62_72 );
				d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, c_43_53_63_73 );
				}

			_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
			_mm256_store_pd( &D0[0+ldc*2], d_02_12_22_32 );
			_mm256_store_pd( &D0[0+ldc*3], d_03_13_23_33 );
			_mm256_maskstore_pd( &D1[0+ldc*0], mask_i, d_40_50_60_70 );
			_mm256_maskstore_pd( &D1[0+ldc*1], mask_i, d_41_51_61_71 );
			_mm256_maskstore_pd( &D1[0+ldc*2], mask_i, d_42_52_62_72 );
			_mm256_maskstore_pd( &D1[0+ldc*3], mask_i, d_43_53_63_73 );
			}
		
		}

	}



// normal-transposed, 8x2 with data packed in 4
void kernel_dgemm_nt_8x2_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int ldc = 4;

	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0101, b_1010,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_20_31, c_01_10_21_30,
		c_40_51_60_71, c_41_50_61_70;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A0[0] );
	a_4567 = _mm256_load_pd( &A1[0] );
	b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

	// zero registers
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	c_40_51_60_71 = _mm256_setzero_pd();
	c_41_50_61_70 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


/*	__builtin_prefetch( A+48 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


/*	__builtin_prefetch( A+56 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
/*		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch*/
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
/*		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch*/
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		}

	__m128d
		d_00_10, d_01_11, d_02_12, d_03_13,
		d_04_14, d_05_15, d_06_16, d_07_17,
		c_00_01, c_10_11, c_20_21, c_30_31,
		c_40_41, c_50_51, c_60_61, c_70_71;

	__m256d
		c_00_01_20_21, c_10_11_30_31,
		c_40_41_60_61, c_50_51_70_71,
		c_00_10_20_30, c_01_11_21_31,
		c_40_50_60_70, c_41_51_61_71,
		d_00_10_20_30, d_01_11_21_31,
		d_40_50_60_70, d_41_51_61_71;


	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0) // AB = A * B'
			{
			c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
			c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
			c_40_50_60_70 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0xa );
			c_41_51_61_71 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0x5 );

			_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
			_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
			_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
			}
		else // AB = t( A * B' )
			{
			c_00_01_20_21 = _mm256_unpacklo_pd( c_00_11_20_31, c_01_10_21_30 );
			c_10_11_30_31 = _mm256_unpackhi_pd( c_01_10_21_30, c_00_11_20_31 );

			c_20_21 = _mm256_extractf128_pd( c_00_01_20_21, 0x1 );
			c_00_01 = _mm256_castpd256_pd128( c_00_01_20_21 );
			c_30_31 = _mm256_extractf128_pd( c_10_11_30_31, 0x1 );
			c_10_11 = _mm256_castpd256_pd128( c_10_11_30_31 );

			_mm_store_pd( &D0[0+ldc*0], c_00_01 );
			_mm_store_pd( &D0[0+ldc*1], c_10_11 );
			_mm_store_pd( &D0[0+ldc*2], c_20_21 );
			_mm_store_pd( &D0[0+ldc*3], c_30_31 );

			c_40_41_60_61 = _mm256_unpacklo_pd( c_40_51_60_71, c_41_50_61_70 );
			c_50_51_70_71 = _mm256_unpackhi_pd( c_41_50_61_70, c_40_51_60_71 );

			c_60_61 = _mm256_extractf128_pd( c_40_41_60_61, 0x1 );
			c_40_41 = _mm256_castpd256_pd128( c_40_41_60_61 );
			c_70_71 = _mm256_extractf128_pd( c_50_51_70_71, 0x1 );
			c_50_51 = _mm256_castpd256_pd128( c_50_51_70_71 );

			_mm_store_pd( &D0[0+ldc*4], c_40_41 );
			_mm_store_pd( &D0[0+ldc*5], c_50_51 );
			_mm_store_pd( &D0[0+ldc*6], c_60_61 );
			_mm_store_pd( &D0[0+ldc*7], c_70_71 );
			}
		}
	else 
		{
		if(tc==0) // C
			{
			c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
			c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
			c_40_50_60_70 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0xa );
			c_41_51_61_71 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0x5 );

			d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
			d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
			d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		
			if(alg==1) // AB = A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
				}
			else // AB = - A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
				}

			if(td==0) // AB + C
				{
				_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
				_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
				_mm256_store_pd( &D1[0+ldc*0], d_40_50_60_70 );
				_mm256_store_pd( &D1[0+ldc*1], d_41_51_61_71 );
				}
			else // t(AB + C)
				{
				c_00_01_20_21 = _mm256_unpacklo_pd( d_00_10_20_30, d_01_11_21_31 );
				c_10_11_30_31 = _mm256_unpackhi_pd( d_00_10_20_30, d_01_11_21_31 );
				
				c_20_21 = _mm256_extractf128_pd( c_00_01_20_21, 0x1 );
				c_00_01 = _mm256_castpd256_pd128( c_00_01_20_21 );
				c_30_31 = _mm256_extractf128_pd( c_10_11_30_31, 0x1 );
				c_10_11 = _mm256_castpd256_pd128( c_10_11_30_31 );

				_mm_store_pd( &D0[0+ldc*0], c_00_01 );
				_mm_store_pd( &D0[0+ldc*1], c_10_11 );
				_mm_store_pd( &D0[0+ldc*2], c_20_21 );
				_mm_store_pd( &D0[0+ldc*3], c_30_31 );

				c_40_41_60_61 = _mm256_unpacklo_pd( d_40_50_60_70, d_41_51_61_71 );
				c_50_51_70_71 = _mm256_unpackhi_pd( d_40_50_60_70, d_41_51_61_71 );
				
				c_60_61 = _mm256_extractf128_pd( c_40_41_60_61, 0x1 );
				c_40_41 = _mm256_castpd256_pd128( c_40_41_60_61 );
				c_70_71 = _mm256_extractf128_pd( c_50_51_70_71, 0x1 );
				c_50_51 = _mm256_castpd256_pd128( c_50_51_70_71 );

				_mm_store_pd( &D0[0+ldc*4], c_40_41 );
				_mm_store_pd( &D0[0+ldc*5], c_50_51 );
				_mm_store_pd( &D0[0+ldc*6], c_60_61 );
				_mm_store_pd( &D0[0+ldc*7], c_70_71 );
				}
			}
		else // t(C)
			{
			c_00_01_20_21 = _mm256_unpacklo_pd( c_00_11_20_31, c_01_10_21_30 );
			c_10_11_30_31 = _mm256_unpackhi_pd( c_01_10_21_30, c_00_11_20_31 );
			c_40_41_60_61 = _mm256_unpacklo_pd( c_40_51_60_71, c_41_50_61_70 );
			c_50_51_70_71 = _mm256_unpackhi_pd( c_41_50_61_70, c_40_51_60_71 );

			c_20_21 = _mm256_extractf128_pd( c_00_01_20_21, 0x1 );
			c_00_01 = _mm256_castpd256_pd128( c_00_01_20_21 );
			c_30_31 = _mm256_extractf128_pd( c_10_11_30_31, 0x1 );
			c_10_11 = _mm256_castpd256_pd128( c_10_11_30_31 );
			c_60_61 = _mm256_extractf128_pd( c_40_41_60_61, 0x1 );
			c_40_41 = _mm256_castpd256_pd128( c_40_41_60_61 );
			c_70_71 = _mm256_extractf128_pd( c_50_51_70_71, 0x1 );
			c_50_51 = _mm256_castpd256_pd128( c_50_51_70_71 );

			d_00_10 = _mm_load_pd( &C0[0+ldc*0] );
			d_01_11 = _mm_load_pd( &C0[0+ldc*1] );
			d_02_12 = _mm_load_pd( &C0[0+ldc*2] );
			d_03_13 = _mm_load_pd( &C0[0+ldc*3] );
			d_04_14 = _mm_load_pd( &C0[0+ldc*4] );
			d_05_15 = _mm_load_pd( &C0[0+ldc*5] );
			d_06_16 = _mm_load_pd( &C0[0+ldc*6] );
			d_07_17 = _mm_load_pd( &C0[0+ldc*7] );

			if(alg==1) // AB = A * B'
				{
				d_00_10 = _mm_add_pd( d_00_10, c_00_01 );
				d_01_11 = _mm_add_pd( d_01_11, c_10_11 );
				d_02_12 = _mm_add_pd( d_02_12, c_20_21 );
				d_03_13 = _mm_add_pd( d_03_13, c_30_31 );
				d_04_14 = _mm_add_pd( d_04_14, c_40_41 );
				d_05_15 = _mm_add_pd( d_05_15, c_50_51 );
				d_06_16 = _mm_add_pd( d_06_16, c_60_61 );
				d_07_17 = _mm_add_pd( d_07_17, c_70_71 );
				}
			else // AB = - A * B'
				{
				d_00_10 = _mm_sub_pd( d_00_10, c_00_01 );
				d_01_11 = _mm_sub_pd( d_01_11, c_10_11 );
				d_02_12 = _mm_sub_pd( d_02_12, c_20_21 );
				d_03_13 = _mm_sub_pd( d_03_13, c_30_31 );
				d_04_14 = _mm_sub_pd( d_04_14, c_40_41 );
				d_05_15 = _mm_sub_pd( d_05_15, c_50_51 );
				d_06_16 = _mm_sub_pd( d_06_16, c_60_61 );
				d_07_17 = _mm_sub_pd( d_07_17, c_70_71 );
				}

			if(td==0) // t( t(AB) + C )
				{
				c_00_01_20_21 = _mm256_insertf128_pd( _mm256_castpd128_pd256( d_00_10 ), d_02_12, 0x1 );
				c_10_11_30_31 = _mm256_insertf128_pd( _mm256_castpd128_pd256( d_01_11 ), d_03_13, 0x1 );
				c_40_41_60_61 = _mm256_insertf128_pd( _mm256_castpd128_pd256( d_04_14 ), d_06_16, 0x1 );
				c_50_51_70_71 = _mm256_insertf128_pd( _mm256_castpd128_pd256( d_05_15 ), d_07_17, 0x1 );

				c_00_10_20_30 = _mm256_unpacklo_pd( c_00_01_20_21, c_10_11_30_31 );
				c_01_11_21_31 = _mm256_unpackhi_pd( c_00_01_20_21, c_10_11_30_31 );
				c_40_50_60_70 = _mm256_unpacklo_pd( c_40_41_60_61, c_50_51_70_71 );
				c_41_51_61_71 = _mm256_unpackhi_pd( c_40_41_60_61, c_50_51_70_71 );

				_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
				_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
				_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
				_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
				}
			else // t(AB) + C
				{
				_mm_store_pd( &D0[0+ldc*0], d_00_10 );
				_mm_store_pd( &D0[0+ldc*1], d_01_11 );
				_mm_store_pd( &D0[0+ldc*2], d_02_12 );
				_mm_store_pd( &D0[0+ldc*3], d_03_13 );
				_mm_store_pd( &D0[0+ldc*4], d_04_14 );
				_mm_store_pd( &D0[0+ldc*5], d_05_15 );
				_mm_store_pd( &D0[0+ldc*6], d_06_16 );
				_mm_store_pd( &D0[0+ldc*7], d_07_17 );
				}

			}
		}

	}



// normal-transposed, 8x2 with data packed in 4
void kernel_dgemm_nt_m8x2_lib4(int m, int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int ldc = 4;

	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0101, b_1010,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_20_31, c_01_10_21_30,
		c_40_51_60_71, c_41_50_61_70;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A0[0] );
	a_4567 = _mm256_load_pd( &A1[0] );
	b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

	// zero registers
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	c_40_51_60_71 = _mm256_setzero_pd();
	c_41_50_61_70 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


/*	__builtin_prefetch( A+48 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


/*	__builtin_prefetch( A+56 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
/*		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch*/
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
/*		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch*/
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		}

	__m256i
		mask_i;

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		c_40_50_60_70, c_41_51_61_71,
		d_00_10_20_30, d_01_11_21_31,
		d_40_50_60_70, d_41_51_61_71;

	c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
	c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
	c_40_50_60_70 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0xa );
	c_41_51_61_71 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0x5 );

	if(m>=8)
		{

		if(alg==0) // C = A * B'
			{
			_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
			_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
			_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
			}
		else 
			{
			d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
			d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
			d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		
			if(alg==1) // C += A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
				}
			else // C -= A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
				}

			_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
			_mm256_store_pd( &D1[0+ldc*0], d_40_50_60_70 );
			_mm256_store_pd( &D1[0+ldc*1], d_41_51_61_71 );
			}

		}
	else
		{

		const double mask_f[] = {4.5, 5.5, 6.5, 7.5};
		double m_f = m;

		mask_i = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( mask_f ), _mm256_broadcast_sd( &m_f ) ) );

		if(alg==0) // C = A * B'
			{
			_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
			_mm256_maskstore_pd( &D1[0+ldc*0], mask_i, c_40_50_60_70 );
			_mm256_maskstore_pd( &D1[0+ldc*1], mask_i, c_41_51_61_71 );
			}
		else 
			{
			d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
			d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
			d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		
			if(alg==1) // C += A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
				}
			else // C -= A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
				d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
				}

			_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
			_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
			_mm256_maskstore_pd( &D1[0+ldc*0], mask_i, d_40_50_60_70 );
			_mm256_maskstore_pd( &D1[0+ldc*1], mask_i, d_41_51_61_71 );
			}
		
		}

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dgemm_nt_4x4_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int ldc = 4;

	int k;
	
	__m256d
		v0, v1, v2, v3,
		u0, u1, u2, u3,
		a_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A[0] );
	b_0123 = _mm256_load_pd( &B[0] );

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
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
	
	if(kmax%4>=2)
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

	if(kmax%2==1)
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
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_00_01_22_23, c_10_11_32_33, c_02_03_20_21, c_12_13_30_31,
		c_00_01_02_03, c_10_11_12_13, c_20_21_22_23, c_30_31_32_33,
		c_00_01_20_21, c_10_11_30_31, c_02_03_22_23, c_12_13_32_33,
		d_00_01_02_03, d_10_11_12_13, d_20_21_22_23, d_30_31_32_33, 
		d_00_10_02_12, d_01_11_03_13, d_20_30_22_32, d_21_31_23_33, 
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33;

	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0)
			{
			c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
			c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
			c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
			c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
			
			c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
			c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
			c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
			c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );

			_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
			_mm256_store_pd( &D[0+ldc*2], c_02_12_22_32 );
			_mm256_store_pd( &D[0+ldc*3], c_03_13_23_33 );
			}
		else // transposed
			{
			c_00_01_22_23 = _mm256_shuffle_pd( c_00_11_22_33, c_01_10_23_32, 0x0 );
			c_10_11_32_33 = _mm256_shuffle_pd( c_01_10_23_32, c_00_11_22_33, 0xf );
			c_02_03_20_21 = _mm256_shuffle_pd( c_02_13_20_31, c_03_12_21_30, 0x0 );
			c_12_13_30_31 = _mm256_shuffle_pd( c_03_12_21_30, c_02_13_20_31, 0xf );

			c_00_01_02_03 = _mm256_permute2f128_pd( c_00_01_22_23, c_02_03_20_21, 0x20 );
			c_10_11_12_13 = _mm256_permute2f128_pd( c_10_11_32_33, c_12_13_30_31, 0x20 );
			c_20_21_22_23 = _mm256_permute2f128_pd( c_02_03_20_21, c_00_01_22_23, 0x31 );
			c_30_31_32_33 = _mm256_permute2f128_pd( c_12_13_30_31, c_10_11_32_33, 0x31 );

			_mm256_store_pd( &D[0+ldc*0], c_00_01_02_03 );
			_mm256_store_pd( &D[0+ldc*1], c_10_11_12_13 );
			_mm256_store_pd( &D[0+ldc*2], c_20_21_22_23 );
			_mm256_store_pd( &D[0+ldc*3], c_30_31_32_33 );
			}
		}
	else 
		{
		if(tc==0) // C
			{

			// AB + C
			c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
			c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
			c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
			c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
			
			c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
			c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
			c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
			c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );

			d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
			d_02_12_22_32 = _mm256_load_pd( &C[0+ldc*2] );
			d_03_13_23_33 = _mm256_load_pd( &C[0+ldc*3] );
			
			if(alg==1) // AB = A*B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
				}
			else // AB = - A*B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );
				}

			if(td==0) // AB + C 
				{
				_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
				_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
				_mm256_store_pd( &D[0+ldc*2], d_02_12_22_32 );
				_mm256_store_pd( &D[0+ldc*3], d_03_13_23_33 );
				}
			else // t(AB + C)
				{
				c_00_01_20_21 = _mm256_unpacklo_pd( d_00_10_20_30, d_01_11_21_31 );
				c_10_11_30_31 = _mm256_unpackhi_pd( d_00_10_20_30, d_01_11_21_31 );
				c_02_03_22_23 = _mm256_unpacklo_pd( d_02_12_22_32, d_03_13_23_33 );
				c_12_13_32_33 = _mm256_unpackhi_pd( d_02_12_22_32, d_03_13_23_33 );

				c_00_01_02_03 = _mm256_permute2f128_pd( c_00_01_20_21, c_02_03_22_23, 0x20 );
				c_20_21_22_23 = _mm256_permute2f128_pd( c_00_01_20_21, c_02_03_22_23, 0x31 );
				c_10_11_12_13 = _mm256_permute2f128_pd( c_10_11_30_31, c_12_13_32_33, 0x20 );
				c_30_31_32_33 = _mm256_permute2f128_pd( c_10_11_30_31, c_12_13_32_33, 0x31 );

				_mm256_store_pd( &D[0+ldc*0], c_00_01_02_03 );
				_mm256_store_pd( &D[0+ldc*1], c_10_11_12_13 );
				_mm256_store_pd( &D[0+ldc*2], c_20_21_22_23 );
				_mm256_store_pd( &D[0+ldc*3], c_30_31_32_33 );
				}

			}
		else // t(C)
			{

			c_00_01_22_23 = _mm256_shuffle_pd( c_00_11_22_33, c_01_10_23_32, 0x0 );
			c_10_11_32_33 = _mm256_shuffle_pd( c_01_10_23_32, c_00_11_22_33, 0xf );
			c_02_03_20_21 = _mm256_shuffle_pd( c_02_13_20_31, c_03_12_21_30, 0x0 );
			c_12_13_30_31 = _mm256_shuffle_pd( c_03_12_21_30, c_02_13_20_31, 0xf );

			c_00_01_02_03 = _mm256_permute2f128_pd( c_00_01_22_23, c_02_03_20_21, 0x20 );
			c_10_11_12_13 = _mm256_permute2f128_pd( c_10_11_32_33, c_12_13_30_31, 0x20 );
			c_20_21_22_23 = _mm256_permute2f128_pd( c_02_03_20_21, c_00_01_22_23, 0x31 );
			c_30_31_32_33 = _mm256_permute2f128_pd( c_12_13_30_31, c_10_11_32_33, 0x31 );

			d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
			d_02_12_22_32 = _mm256_load_pd( &C[0+ldc*2] );
			d_03_13_23_33 = _mm256_load_pd( &C[0+ldc*3] );

			if(alg==1) // AB = A*B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_01_02_03 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_10_11_12_13 );
				d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_20_21_22_23 );
				d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_30_31_32_33 );
				}
			else // AB = - A*B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_01_02_03 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_10_11_12_13 );
				d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_20_21_22_23 );
				d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_30_31_32_33 );
				}

			if(td==0) // t( t(AB) + C )
				{
				c_00_01_20_21 = _mm256_unpacklo_pd( d_00_10_20_30, d_01_11_21_31 );
				c_10_11_30_31 = _mm256_unpackhi_pd( d_00_10_20_30, d_01_11_21_31 );
				c_02_03_22_23 = _mm256_unpacklo_pd( d_02_12_22_32, d_03_13_23_33 );
				c_12_13_32_33 = _mm256_unpackhi_pd( d_02_12_22_32, d_03_13_23_33 );

				c_00_01_02_03 = _mm256_permute2f128_pd( c_00_01_20_21, c_02_03_22_23, 0x20 );
				c_20_21_22_23 = _mm256_permute2f128_pd( c_00_01_20_21, c_02_03_22_23, 0x31 );
				c_10_11_12_13 = _mm256_permute2f128_pd( c_10_11_30_31, c_12_13_32_33, 0x20 );
				c_30_31_32_33 = _mm256_permute2f128_pd( c_10_11_30_31, c_12_13_32_33, 0x31 );

				_mm256_store_pd( &D[0+ldc*0], c_00_01_02_03 );
				_mm256_store_pd( &D[0+ldc*1], c_10_11_12_13 );
				_mm256_store_pd( &D[0+ldc*2], c_20_21_22_23 );
				_mm256_store_pd( &D[0+ldc*3], c_30_31_32_33 );
				}
			else // t(AB) + C
				{
				_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
				_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
				_mm256_store_pd( &D[0+ldc*2], d_02_12_22_32 );
				_mm256_store_pd( &D[0+ldc*3], d_03_13_23_33 );
				}

			}
		}

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dgemm_nt_m4x4_lib4(int m, int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int ldc = 4;

	int k;
	
	__m256d
		a_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A[0] );
	b_0123 = _mm256_load_pd( &B[0] );

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
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
	
	if(kmax%4>=2)
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

	if(kmax%2==1)
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

	__m256i
		mask_i;

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

	if(m>=4)
		{

		if(alg==0) // C = A * B'
			{
			_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
			_mm256_store_pd( &D[0+ldc*2], c_02_12_22_32 );
			_mm256_store_pd( &D[0+ldc*3], c_03_13_23_33 );
			}
		else 
			{
			d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
			d_02_12_22_32 = _mm256_load_pd( &C[0+ldc*2] );
			d_03_13_23_33 = _mm256_load_pd( &C[0+ldc*3] );
		
			if(alg==1) // C += A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
				}
			else // C -= A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );
				}

			_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
			_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
			_mm256_store_pd( &D[0+ldc*2], d_02_12_22_32 );
			_mm256_store_pd( &D[0+ldc*3], d_03_13_23_33 );
			}

		}
	else
		{

		const double mask_f[] = {0.5, 1.5, 2.5, 3.5};
		double m_f = m;

		mask_i = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( mask_f ), _mm256_broadcast_sd( &m_f ) ) );

		if(alg==0) // C = A * B'
			{
			_mm256_maskstore_pd( &D[0+ldc*0], mask_i, c_00_10_20_30 );
			_mm256_maskstore_pd( &D[0+ldc*1], mask_i, c_01_11_21_31 );
			_mm256_maskstore_pd( &D[0+ldc*2], mask_i, c_02_12_22_32 );
			_mm256_maskstore_pd( &D[0+ldc*3], mask_i, c_03_13_23_33 );
			}
		else 
			{
			d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
			d_02_12_22_32 = _mm256_load_pd( &C[0+ldc*2] );
			d_03_13_23_33 = _mm256_load_pd( &C[0+ldc*3] );
		
			if(alg==1) // C += A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
				}
			else // C -= A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
				d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );
				}

			_mm256_maskstore_pd( &D[0+ldc*0], mask_i, d_00_10_20_30 );
			_mm256_maskstore_pd( &D[0+ldc*1], mask_i, d_01_11_21_31 );
			_mm256_maskstore_pd( &D[0+ldc*2], mask_i, d_02_12_22_32 );
			_mm256_maskstore_pd( &D[0+ldc*3], mask_i, d_03_13_23_33 );
			}

		}

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_dgemm_nt_4x2_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int ldc = 4;

	int k;
	
	__m256d
		a_0123,
		b_0101, b_1010,
		ab_temp, // temporary results
		c_00_11_20_31, c_01_10_21_30, C_00_11_20_31, C_01_10_21_30;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A[0] );
	b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

	// zero registers
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	C_00_11_20_31 = _mm256_setzero_pd();
	C_01_10_21_30 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
		
		
		A += 8;
		B += 8;

		}
	
	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, C_00_11_20_31 );
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, C_01_10_21_30 );

	if(kmax%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch*/
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		}

	__m128d
		d_00_10, d_01_11, d_02_12, d_03_13,
		c_00_01, c_10_11, c_20_21, c_30_31;

	__m256d
		c_00_01_20_21, c_10_11_30_31,
		c_00_10_20_30, c_01_11_21_31,
		d_00_10_20_30, d_01_11_21_31;

	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0) // AB = A * B'
			{
			c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
			c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );

			_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
			}
		else // AB = t( A * B' )
			{
			c_00_01_20_21 = _mm256_unpacklo_pd( c_00_10_20_30, c_01_11_21_31 );
			c_10_11_30_31 = _mm256_unpackhi_pd( c_00_10_20_30, c_01_11_21_31 );

			c_20_21 = _mm256_extractf128_pd( c_00_01_20_21, 0x1 );
			c_00_01 = _mm256_castpd256_pd128( c_00_01_20_21 );
			c_30_31 = _mm256_extractf128_pd( c_10_11_30_31, 0x1 );
			c_10_11 = _mm256_castpd256_pd128( c_10_11_30_31 );

			_mm_store_pd( &D[0+ldc*0], c_00_01 );
			_mm_store_pd( &D[0+ldc*1], c_10_11 );
			_mm_store_pd( &D[0+ldc*2], c_20_21 );
			_mm_store_pd( &D[0+ldc*3], c_30_31 );
			}
		}
	else 
		{
		if(tc==0) // C
			{
			c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
			c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );

			d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
		
			if(alg==1) // AB = A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				}
			else // AB = - A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				}

			if(td==0) // AB + C
				{
				_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
				_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
				}
			else // t(AB + C)
				{
				c_00_01_20_21 = _mm256_unpacklo_pd( d_00_10_20_30, d_01_11_21_31 );
				c_10_11_30_31 = _mm256_unpackhi_pd( d_00_10_20_30, d_01_11_21_31 );
				
				c_20_21 = _mm256_extractf128_pd( c_00_01_20_21, 0x1 );
				c_00_01 = _mm256_castpd256_pd128( c_00_01_20_21 );
				c_30_31 = _mm256_extractf128_pd( c_10_11_30_31, 0x1 );
				c_10_11 = _mm256_castpd256_pd128( c_10_11_30_31 );

				_mm_store_pd( &D[0+ldc*0], c_00_01 );
				_mm_store_pd( &D[0+ldc*1], c_10_11 );
				_mm_store_pd( &D[0+ldc*2], c_20_21 );
				_mm_store_pd( &D[0+ldc*3], c_30_31 );
				}
			}
		else // t(C)
			{
			c_00_01_20_21 = _mm256_unpacklo_pd( c_00_11_20_31, c_01_10_21_30 );
			c_10_11_30_31 = _mm256_unpackhi_pd( c_01_10_21_30, c_00_11_20_31 );
				
			c_20_21 = _mm256_extractf128_pd( c_00_01_20_21, 0x1 );
			c_00_01 = _mm256_castpd256_pd128( c_00_01_20_21 );
			c_30_31 = _mm256_extractf128_pd( c_10_11_30_31, 0x1 );
			c_10_11 = _mm256_castpd256_pd128( c_10_11_30_31 );

			d_00_10 = _mm_load_pd( &C[0+ldc*0] );
			d_01_11 = _mm_load_pd( &C[0+ldc*1] );
			d_02_12 = _mm_load_pd( &C[0+ldc*2] );
			d_03_13 = _mm_load_pd( &C[0+ldc*3] );
		
			if(alg==1) // AB = A * B'
				{
				d_00_10 = _mm_add_pd( d_00_10, c_00_01 );
				d_01_11 = _mm_add_pd( d_01_11, c_10_11 );
				d_02_12 = _mm_add_pd( d_02_12, c_20_21 );
				d_03_13 = _mm_add_pd( d_03_13, c_30_31 );
				}
			else // AB = - A * B'
				{
				d_00_10 = _mm_sub_pd( d_00_10, c_00_01 );
				d_01_11 = _mm_sub_pd( d_01_11, c_10_11 );
				d_02_12 = _mm_sub_pd( d_02_12, c_20_21 );
				d_03_13 = _mm_sub_pd( d_03_13, c_30_31 );
				}

			if(td==0) // t( t(AB) + C )
				{
				c_00_01_20_21 = _mm256_insertf128_pd( _mm256_castpd128_pd256( d_00_10 ), d_02_12, 0x1 );
				c_10_11_30_31 = _mm256_insertf128_pd( _mm256_castpd128_pd256( d_01_11 ), d_03_13, 0x1 );

				c_00_10_20_30 = _mm256_unpacklo_pd( c_00_01_20_21, c_10_11_30_31 );
				c_01_11_21_31 = _mm256_unpackhi_pd( c_00_01_20_21, c_10_11_30_31 );

				_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
				_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
				}
			else // t(AB) + C
				{
				_mm_store_pd( &D[0+ldc*0], d_00_10 );
				_mm_store_pd( &D[0+ldc*1], d_01_11 );
				_mm_store_pd( &D[0+ldc*2], d_02_12 );
				_mm_store_pd( &D[0+ldc*3], d_03_13 );
				}

			}
		}

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_dgemm_nt_m4x2_lib4(int m, int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int ldc = 4;

	int k;
	
	__m256d
		a_0123,
		b_0101, b_1010,
		ab_temp, // temporary results
		c_00_11_20_31, c_01_10_21_30, C_00_11_20_31, C_01_10_21_30;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A[0] );
	b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

	// zero registers
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	C_00_11_20_31 = _mm256_setzero_pd();
	C_01_10_21_30 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
		
		
		A += 8;
		B += 8;

		}
	
	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, C_00_11_20_31 );
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, C_01_10_21_30 );

	if(kmax%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch*/
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		}

	__m256i
		mask_i;

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		d_00_10_20_30, d_01_11_21_31;

	c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
	c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );

	if(m>=4)
		{

		if(alg==0) // C = A * B'
			{
			_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
			_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
			}
		else 
			{
			d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
		
			if(alg==1) // C += A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				}
			else // C -= A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				}

			_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
			_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
			}

		}
	else
		{

		const double mask_f[] = {0.5, 1.5, 2.5, 3.5};
		double m_f = m;

		mask_i = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( mask_f ), _mm256_broadcast_sd( &m_f ) ) );

		if(alg==0) // C = A * B'
			{
			_mm256_maskstore_pd( &D[0+ldc*0], mask_i, c_00_10_20_30 );
			_mm256_maskstore_pd( &D[0+ldc*1], mask_i, c_01_11_21_31 );
			}
		else 
			{
			d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
		
			if(alg==1) // C += A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
				}
			else // C -= A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
				}

			_mm256_maskstore_pd( &D[0+ldc*0], mask_i, d_00_10_20_30 );
			_mm256_maskstore_pd( &D[0+ldc*1], mask_i, d_01_11_21_31 );
			}

		}

	}



// normal-transposed, 2x4 with data packed in 4
void kernel_dgemm_nt_2x4_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int ldc = 4;

	int k;
	
	__m256d
		a_0101,
		b_0123, b_1032,
		ab_temp, // temporary results
		c_00_11_02_13, c_01_10_03_12, C_00_11_02_13, C_01_10_03_12;
	
	// prefetch
	a_0101 = _mm256_broadcast_pd( (__m128d *) &A[0] );
	b_0123 = _mm256_load_pd( &B[0] );

	// zero registers
	c_00_11_02_13 = _mm256_setzero_pd();
	c_01_10_03_12 = _mm256_setzero_pd();
	C_00_11_02_13 = _mm256_setzero_pd();
	C_01_10_03_12 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
		c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
		a_0101        = _mm256_broadcast_pd( (__m128d *) &A[4] ); // prefetch
		c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
		C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
		a_0101        = _mm256_broadcast_pd( (__m128d *) &A[8] ); // prefetch
		C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
		c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
		a_0101        = _mm256_broadcast_pd( (__m128d *) &A[12] ); // prefetch
		c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
		C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
		a_0101        = _mm256_broadcast_pd( (__m128d *) &A[16] ); // prefetch
		C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
		c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
		a_0101        = _mm256_broadcast_pd( (__m128d *) &A[4] ); // prefetch
		c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
		C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
		a_0101        = _mm256_broadcast_pd( (__m128d *) &A[8] ); // prefetch
		C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, C_00_11_02_13 );
	c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, C_01_10_03_12 );

	if(kmax%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
		c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
		ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
/*		a_0101        = _mm256_broadcast_pd( (__m128d *) &A[4] ); // prefetch*/
		c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
		
		}

	__m256d
		d_00_10_02_12, d_01_11_03_13,
		c_00_01_02_03, c_10_11_12_13,
		d_00_01_02_03, d_10_11_12_13,
		d_00_10_20_30, d_01_11_21_31,
		d_00_01_20_21, d_10_11_30_31,
		c_00_10_02_12, c_01_11_03_13;
	
	__m128d	
		c_00_10, c_01_11, c_02_12, c_03_13,
		d_00_10, d_01_11, d_02_12, d_03_13;

	c_00_10_02_12 = _mm256_blend_pd( c_00_11_02_13, c_01_10_03_12, 0xa );
	c_01_11_03_13 = _mm256_blend_pd( c_00_11_02_13, c_01_10_03_12, 0x5 );
	
	c_02_12 = _mm256_extractf128_pd( c_00_10_02_12, 0x1 );
	c_00_10 = _mm256_castpd256_pd128( c_00_10_02_12 );
	c_03_13 = _mm256_extractf128_pd( c_01_11_03_13, 0x1 );
	c_01_11 = _mm256_castpd256_pd128( c_01_11_03_13 );

	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0) // AB = A * B'
			{
			c_00_10_02_12 = _mm256_blend_pd( c_00_11_02_13, c_01_10_03_12, 0xa );
			c_01_11_03_13 = _mm256_blend_pd( c_00_11_02_13, c_01_10_03_12, 0x5 );
			
			c_02_12 = _mm256_extractf128_pd( c_00_10_02_12, 0x1 );
			c_00_10 = _mm256_castpd256_pd128( c_00_10_02_12 );
			c_03_13 = _mm256_extractf128_pd( c_01_11_03_13, 0x1 );
			c_01_11 = _mm256_castpd256_pd128( c_01_11_03_13 );

			_mm_store_pd( &D[0+ldc*0], c_00_10 );
			_mm_store_pd( &D[0+ldc*1], c_01_11 );
			_mm_store_pd( &D[0+ldc*2], c_02_12 );
			_mm_store_pd( &D[0+ldc*3], c_03_13 );
			}
		else // AB = t( A * B' )
			{
			//c_00_01_02_03 = _mm256_shuffle_pd( c_00_11_02_13, c_01_10_03_12, 0x0 );
			//c_10_11_12_13 = _mm256_shuffle_pd( c_01_10_03_12, c_00_11_02_13, 0xf );
			c_00_01_02_03 = _mm256_unpacklo_pd( c_00_11_02_13, c_01_10_03_12 );
			c_10_11_12_13 = _mm256_unpackhi_pd( c_01_10_03_12, c_00_11_02_13 );

			_mm256_store_pd( &D[0+ldc*0], c_00_01_02_03 );
			_mm256_store_pd( &D[0+ldc*1], c_10_11_12_13 );

			}
		}
	else
		{
		if(tc==0) // C
			{
			c_00_10_02_12 = _mm256_blend_pd( c_00_11_02_13, c_01_10_03_12, 0xa );
			c_01_11_03_13 = _mm256_blend_pd( c_00_11_02_13, c_01_10_03_12, 0x5 );
			
			c_02_12 = _mm256_extractf128_pd( c_00_10_02_12, 0x1 );
			c_00_10 = _mm256_castpd256_pd128( c_00_10_02_12 );
			c_03_13 = _mm256_extractf128_pd( c_01_11_03_13, 0x1 );
			c_01_11 = _mm256_castpd256_pd128( c_01_11_03_13 );

			d_00_10 = _mm_load_pd( &C[0+ldc*0] );
			d_01_11 = _mm_load_pd( &C[0+ldc*1] );
			d_02_12 = _mm_load_pd( &C[0+ldc*2] );
			d_03_13 = _mm_load_pd( &C[0+ldc*3] );
		
			if(alg==1) // AB = A * B'
				{
				d_00_10 = _mm_add_pd( d_00_10, c_00_10 );
				d_01_11 = _mm_add_pd( d_01_11, c_01_11 );
				d_02_12 = _mm_add_pd( d_02_12, c_02_12 );
				d_03_13 = _mm_add_pd( d_03_13, c_03_13 );
				}
			else // AB = - A * B'
				{
				d_00_10 = _mm_sub_pd( d_00_10, c_00_10 );
				d_01_11 = _mm_sub_pd( d_01_11, c_01_11 );
				d_02_12 = _mm_sub_pd( d_02_12, c_02_12 );
				d_03_13 = _mm_sub_pd( d_03_13, c_03_13 );
				}

			if(td==0) // AB + C
				{
				_mm_store_pd( &D[0+ldc*0], d_00_10 );
				_mm_store_pd( &D[0+ldc*1], d_01_11 );
				_mm_store_pd( &D[0+ldc*2], d_02_12 );
				_mm_store_pd( &D[0+ldc*3], d_03_13 );
				}
			else // t(AB + C)
				{
				d_00_10_02_12 = _mm256_insertf128_pd( _mm256_castpd128_pd256( d_00_10 ), d_02_12, 0x1 );
				d_01_11_03_13 = _mm256_insertf128_pd( _mm256_castpd128_pd256( d_01_11 ), d_03_13, 0x1 );

				//d_00_01_02_03 = _mm256_shuffle_pd( d_00_10_02_12, d_01_11_03_13, 0x0 );
				//d_10_11_12_13 = _mm256_shuffle_pd( d_00_10_02_12, d_01_11_03_13, 0xf );
				d_00_01_02_03 = _mm256_unpacklo_pd( d_00_10_02_12, d_01_11_03_13 );
				d_10_11_12_13 = _mm256_unpackhi_pd( d_00_10_02_12, d_01_11_03_13 );

				_mm256_store_pd( &D[0+ldc*0], d_00_01_02_03 );
				_mm256_store_pd( &D[0+ldc*1], d_10_11_12_13 );
				}
			}
		else // t(C)
			{
			//c_00_01_02_03 = _mm256_shuffle_pd( c_00_11_02_13, c_01_10_03_12, 0x0 );
			//c_10_11_12_13 = _mm256_shuffle_pd( c_01_10_03_12, c_00_11_02_13, 0xf );
			c_00_01_02_03 = _mm256_unpacklo_pd( c_00_11_02_13, c_01_10_03_12 );
			c_10_11_12_13 = _mm256_unpackhi_pd( c_01_10_03_12, c_00_11_02_13 );

			d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
			d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );

			if(alg==1) // AB = A * B'
				{
				d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_01_02_03 );
				d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_10_11_12_13 );
				}
			else // AB = - A * B'
				{
				d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_01_02_03 );
				d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_10_11_12_13 );
				}
			
			if(td==0) // t( t(AB) + C )
				{
				//d_00_01_20_21 = _mm256_shuffle_pd( d_00_10_20_30, d_01_11_21_31, 0x0 );
				//d_10_11_30_31 = _mm256_shuffle_pd( d_00_10_20_30, d_01_11_21_31, 0xf );
				d_00_01_20_21 = _mm256_unpacklo_pd( d_00_10_20_30, d_01_11_21_31 );
				d_10_11_30_31 = _mm256_unpackhi_pd( d_00_10_20_30, d_01_11_21_31 );

				c_02_12 = _mm256_extractf128_pd( d_00_01_20_21, 0x1 );
				c_00_10 = _mm256_castpd256_pd128( d_00_01_20_21 );
				c_03_13 = _mm256_extractf128_pd( d_10_11_30_31, 0x1 );
				c_01_11 = _mm256_castpd256_pd128( d_10_11_30_31 );

				_mm_store_pd( &D[0+ldc*0], c_00_10 );
				_mm_store_pd( &D[0+ldc*1], c_01_11 );
				_mm_store_pd( &D[0+ldc*2], c_02_12 );
				_mm_store_pd( &D[0+ldc*3], c_03_13 );
				}
			else // t(AB) + C
				{
				_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
				_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
				}

			}
		}

	}



// normal-transposed, 2x2 with data packed in 4
void kernel_dgemm_nt_2x2_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int ldc = 4;

	int k;
	
	__m128d
		a_01,
		b_01, b_10,
		ab_temp, // temporary results
		c_00_11, c_01_10, C_00_11, C_01_10;
	
	// prefetch
	a_01 = _mm_load_pd( &A[0] );
	b_01 = _mm_load_pd( &B[0] );

	// zero registers
	c_00_11 = _mm_setzero_pd();
	c_01_10 = _mm_setzero_pd();
	C_00_11 = _mm_setzero_pd();
	C_01_10 = _mm_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[4] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[4] ); // prefetch
		c_01_10 = _mm_add_pd( c_01_10, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp = _mm_mul_pd( a_01, b_01 );
		C_00_11 = _mm_add_pd( C_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[8] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[8] ); // prefetch
		C_01_10 = _mm_add_pd( C_01_10, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[12] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[12] ); // prefetch
		c_01_10 = _mm_add_pd( c_01_10, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp = _mm_mul_pd( a_01, b_01 );
		C_00_11 = _mm_add_pd( C_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[16] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[16] ); // prefetch
		C_01_10 = _mm_add_pd( C_01_10, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_temp = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[4] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[4] ); // prefetch
		c_01_10 = _mm_add_pd( c_01_10, ab_temp );
		
		
		ab_temp = _mm_mul_pd( a_01, b_01 );
		C_00_11 = _mm_add_pd( C_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[8] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[8] ); // prefetch
		C_01_10 = _mm_add_pd( C_01_10, ab_temp );
		
		
		A += 8;
		B += 8;

		}
	
	c_00_11 = _mm_add_pd( c_00_11, C_00_11 );
	c_01_10 = _mm_add_pd( c_01_10, C_01_10 );

	if(kmax%2==1)
		{
		
		ab_temp = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
/*		b_01    = _mm_load_pd( &B[4] ); // prefetch*/
		ab_temp = _mm_mul_pd( a_01, b_10 );
/*		a_01    = _mm_load_pd( &A[4] ); // prefetch*/
		c_01_10 = _mm_add_pd( c_01_10, ab_temp );
		
		}

	__m128d
		c_00_10, c_01_11,
		c_00_01, c_10_11,
		d_00_11, d_01_10,
		d_00_10, d_01_11,
		d_00_01, d_10_11;

	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0)
			{
			c_00_10 = _mm_blend_pd( c_00_11, c_01_10, 0x2 );
			c_01_11 = _mm_blend_pd( c_00_11, c_01_10, 0x1 );

			_mm_store_pd( &D[0+ldc*0], c_00_10 );
			_mm_store_pd( &D[0+ldc*1], c_01_11 );
			}
		else
			{
			//c_00_01 = _mm_shuffle_pd( c_00_11, c_01_10, 0x0 );
			//c_10_11 = _mm_shuffle_pd( c_01_10, c_00_11, 0x3 );
			c_00_01 = _mm_unpacklo_pd( c_00_11, c_01_10 );
			c_10_11 = _mm_unpackhi_pd( c_01_10, c_00_11 );

			_mm_store_pd( &D[0+ldc*0], c_00_01 );
			_mm_store_pd( &D[0+ldc*1], c_10_11 );
			}
		}
	else 
		{
		if(tc==0) // C
			{
			c_00_10 = _mm_blend_pd( c_00_11, c_01_10, 0x2 );
			c_01_11 = _mm_blend_pd( c_00_11, c_01_10, 0x1 );

			d_00_10 = _mm_load_pd( &C[0+ldc*0] );
			d_01_11 = _mm_load_pd( &C[0+ldc*1] );
		
			if(alg==1) // AB = A * B'
				{
				d_00_10 = _mm_add_pd( d_00_10, c_00_10 );
				d_01_11 = _mm_add_pd( d_01_11, c_01_11 );
				}
			else // AB = - A * B'
				{
				d_00_10 = _mm_sub_pd( d_00_10, c_00_10 );
				d_01_11 = _mm_sub_pd( d_01_11, c_01_11 );
				}

			if(td==0) // AB + C
				{
				_mm_store_pd( &D[0+ldc*0], d_00_10 );
				_mm_store_pd( &D[0+ldc*1], d_01_11 );
				}
			else // t(AB + C)
				{
				//d_00_01 = _mm_shuffle_pd( d_00_11, d_01_10, 0x0 );
				//d_10_11 = _mm_shuffle_pd( d_00_11, d_01_10, 0x3 );
				d_00_01 = _mm_unpacklo_pd( d_00_10, d_01_11 );
				d_10_11 = _mm_unpackhi_pd( d_00_10, d_01_11 );

				_mm_store_pd( &D[0+ldc*0], d_00_01 );
				_mm_store_pd( &D[0+ldc*1], d_10_11 );
				}
			}
		else // t(C)
			{
			//c_00_01 = _mm_shuffle_pd( c_00_11, c_01_10, 0x0 );
			//c_10_11 = _mm_shuffle_pd( c_01_10, c_00_11, 0x3 );
			c_00_01 = _mm_unpacklo_pd( c_00_11, c_01_10 );
			c_10_11 = _mm_unpackhi_pd( c_01_10, c_00_11 );

			d_00_10 = _mm_load_pd( &C[0+ldc*0] );
			d_01_11 = _mm_load_pd( &C[0+ldc*1] );

			if(alg==1) // AB = A * B'
				{
				d_00_10 = _mm_add_pd( d_00_10, c_00_01 );
				d_01_11 = _mm_add_pd( d_01_11, c_10_11 );
				}
			else // AB = - A * B'
				{
				d_00_10 = _mm_sub_pd( d_00_10, c_00_01 );
				d_01_11 = _mm_sub_pd( d_01_11, c_10_11 );
				}

			if(td==0) // t( t(AB) + C )
				{
				//d_00_01 = _mm_shuffle_pd( d_00_10, d_01_11, 0x0 );
				//d_10_11 = _mm_shuffle_pd( d_00_10, d_01_11, 0x3 );
				d_00_01 = _mm_unpacklo_pd( d_00_10, d_01_11 );
				d_10_11 = _mm_unpackhi_pd( d_00_10, d_01_11 );

				_mm_store_pd( &D[0+ldc*0], d_00_01 );
				_mm_store_pd( &D[0+ldc*1], d_10_11 );
				}
			else // t(AB) + C
				{
				_mm_store_pd( &D[0+ldc*0], d_00_10 );
				_mm_store_pd( &D[0+ldc*1], d_01_11 );
				}
			}
		}

	}



// normal-normal, 4x4 with data packed in 4
void kernel_dgemm_nn_4x4_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
		
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
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


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		b_2 = B[1+bs*2];
		b_3 = B[1+bs*3];
		
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


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		b_2 = B[2+bs*2];
		b_3 = B[2+bs*3];
		
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


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		b_2 = B[3+bs*2];
		b_3 = B[3+bs*3];
		
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
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
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


		A += 4;
		B += 1;

		}
		
	double
		d_00, d_01, d_02, d_03,
		d_10, d_11, d_12, d_13,
		d_20, d_21, d_22, d_23,
		d_30, d_31, d_32, d_33;
	
	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0) // not transpose D
			{
			D[0+bs*0] = c_00;
			D[1+bs*0] = c_10;
			D[2+bs*0] = c_20;
			D[3+bs*0] = c_30;

			D[0+bs*1] = c_01;
			D[1+bs*1] = c_11;
			D[2+bs*1] = c_21;
			D[3+bs*1] = c_31;

			D[0+bs*2] = c_02;
			D[1+bs*2] = c_12;
			D[2+bs*2] = c_22;
			D[3+bs*2] = c_32;

			D[0+bs*3] = c_03;
			D[1+bs*3] = c_13;
			D[2+bs*3] = c_23;
			D[3+bs*3] = c_33;
			}
		else // transpose D
			{
			D[0+bs*0] = c_00;
			D[1+bs*0] = c_01;
			D[2+bs*0] = c_02;
			D[3+bs*0] = c_03;
			
			D[0+bs*1] = c_10;
			D[1+bs*1] = c_11;
			D[2+bs*1] = c_12;
			D[3+bs*1] = c_13;

			D[0+bs*2] = c_20;
			D[1+bs*2] = c_21;
			D[2+bs*2] = c_22;
			D[3+bs*2] = c_23;

			D[0+bs*3] = c_30;
			D[1+bs*3] = c_31;
			D[2+bs*3] = c_32;
			D[3+bs*3] = c_33;
			}
		}
	else // D = C +/- A * B'
		{
		if(tc==td) // not transpose C
			{
			d_00 = C[0+bs*0];
			d_10 = C[1+bs*0];
			d_20 = C[2+bs*0];
			d_30 = C[3+bs*0];
			
			d_01 = C[0+bs*1];
			d_11 = C[1+bs*1];
			d_21 = C[2+bs*1];
			d_31 = C[3+bs*1];
			
			d_02 = C[0+bs*2];
			d_12 = C[1+bs*2];
			d_22 = C[2+bs*2];
			d_32 = C[3+bs*2];
			
			d_03 = C[0+bs*3];
			d_13 = C[1+bs*3];
			d_23 = C[2+bs*3];
			d_33 = C[3+bs*3];
			}
		else // transpose C
			{
			d_00 = C[0+bs*0];
			d_01 = C[1+bs*0];
			d_02 = C[2+bs*0];
			d_03 = C[3+bs*0];
			
			d_10 = C[0+bs*1];
			d_11 = C[1+bs*1];
			d_12 = C[2+bs*1];
			d_13 = C[3+bs*1];
			
			d_20 = C[0+bs*2];
			d_21 = C[1+bs*2];
			d_22 = C[2+bs*2];
			d_23 = C[3+bs*2];
			
			d_30 = C[0+bs*3];
			d_31 = C[1+bs*3];
			d_32 = C[2+bs*3];
			d_33 = C[3+bs*3];
			}
		
		if(alg==1) // D = C + A * B'
			{
			d_00 += c_00;
			d_10 += c_10;
			d_20 += c_20;
			d_30 += c_30;

			d_01 += c_01;
			d_11 += c_11;
			d_21 += c_21;
			d_31 += c_31;

			d_02 += c_02;
			d_12 += c_12;
			d_22 += c_22;
			d_32 += c_32;

			d_03 += c_03;
			d_13 += c_13;
			d_23 += c_23;
			d_33 += c_33;
			}
		else // D = C - A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;
			d_20 -= c_20;
			d_30 -= c_30;

			d_01 -= c_01;
			d_11 -= c_11;
			d_21 -= c_21;
			d_31 -= c_31;

			d_02 -= c_02;
			d_12 -= c_12;
			d_22 -= c_22;
			d_32 -= c_32;

			d_03 -= c_03;
			d_13 -= c_13;
			d_23 -= c_23;
			d_33 -= c_33;
			}

		if(td==0) // not transpose D
			{
			D[0+bs*0] = d_00;
			D[1+bs*0] = d_10;
			D[2+bs*0] = d_20;
			D[3+bs*0] = d_30;

			D[0+bs*1] = d_01;
			D[1+bs*1] = d_11;
			D[2+bs*1] = d_21;
			D[3+bs*1] = d_31;

			D[0+bs*2] = d_02;
			D[1+bs*2] = d_12;
			D[2+bs*2] = d_22;
			D[3+bs*2] = d_32;

			D[0+bs*3] = d_03;
			D[1+bs*3] = d_13;
			D[2+bs*3] = d_23;
			D[3+bs*3] = d_33;
			}
		else // transpose D
			{
			D[0+bs*0] = d_00;
			D[1+bs*0] = d_01;
			D[2+bs*0] = d_02;
			D[3+bs*0] = d_03;

			D[0+bs*1] = d_10;
			D[1+bs*1] = d_11;
			D[2+bs*1] = d_12;
			D[3+bs*1] = d_13;

			D[0+bs*2] = d_20;
			D[1+bs*2] = d_21;
			D[2+bs*2] = d_22;
			D[3+bs*2] = d_23;

			D[0+bs*3] = d_30;
			D[1+bs*3] = d_31;
			D[2+bs*3] = d_32;
			D[3+bs*3] = d_33;
			}
		}
	
	}



// normal-normal, 4x2 with data packed in 4
void kernel_dgemm_nn_4x2_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1,
		c_00=0, c_01=0,
		c_10=0, c_11=0,
		c_20=0, c_21=0,
		c_30=0, c_31=0;
		
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		A += 4;
		B += 1;

		}
		
	double
		d_00, d_01,
		d_10, d_11,
		d_20, d_21,
		d_30, d_31;
	
	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0) // not transpose D
			{
			D[0+bs*0] = c_00;
			D[1+bs*0] = c_10;
			D[2+bs*0] = c_20;
			D[3+bs*0] = c_30;

			D[0+bs*1] = c_01;
			D[1+bs*1] = c_11;
			D[2+bs*1] = c_21;
			D[3+bs*1] = c_31;
			}
		else // transpose D
			{
			D[0+bs*0] = c_00;
			D[1+bs*0] = c_01;

			D[0+bs*1] = c_10;
			D[1+bs*1] = c_11;

			D[0+bs*2] = c_20;
			D[1+bs*2] = c_21;

			D[0+bs*3] = c_30;
			D[1+bs*3] = c_31;
			}
		}
	else 
		{
		if(tc==td) // not transpose C
			{
			d_00 = C[0+bs*0];
			d_10 = C[1+bs*0];
			d_20 = C[2+bs*0];
			d_30 = C[3+bs*0];
			
			d_01 = C[0+bs*1];
			d_11 = C[1+bs*1];
			d_21 = C[2+bs*1];
			d_31 = C[3+bs*1];
			}
		else // transpose C
			{
			d_00 = C[0+bs*0];
			d_01 = C[1+bs*0];

			d_10 = C[0+bs*1];
			d_11 = C[1+bs*1];

			d_20 = C[0+bs*2];
			d_21 = C[1+bs*2];

			d_30 = C[0+bs*3];
			d_31 = C[1+bs*3];
			}
		
		if(alg==1) // D = C + A * B'
			{
			d_00 += c_00;
			d_10 += c_10;
			d_20 += c_20;
			d_30 += c_30;

			d_01 += c_01;
			d_11 += c_11;
			d_21 += c_21;
			d_31 += c_31;
			}
		else // D = C - A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;
			d_20 -= c_20;
			d_30 -= c_30;

			d_01 -= c_01;
			d_11 -= c_11;
			d_21 -= c_21;
			d_31 -= c_31;
			}

		if(td==0) // not transpose D
			{
			D[0+bs*0] = d_00;
			D[1+bs*0] = d_10;
			D[2+bs*0] = d_20;
			D[3+bs*0] = d_30;

			D[0+bs*1] = d_01;
			D[1+bs*1] = d_11;
			D[2+bs*1] = d_21;
			D[3+bs*1] = d_31;
			}
		else // transpose D
			{
			D[0+bs*0] = d_00;
			D[1+bs*0] = d_01;

			D[0+bs*1] = d_10;
			D[1+bs*1] = d_11;

			D[0+bs*2] = d_20;
			D[1+bs*2] = d_21;

			D[0+bs*3] = d_30;
			D[1+bs*3] = d_31;
			}
		}
	
	}



// normal-normal, 2x4 with data packed in 4
void kernel_dgemm_nn_2x4_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		a_0, a_1,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0;
		
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		b_2 = B[1+bs*2];
		b_3 = B[1+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		b_2 = B[2+bs*2];
		b_3 = B[2+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		b_2 = B[3+bs*2];
		b_3 = B[3+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		A += 4;
		B += 1;

		}
		
	double
		d_00, d_01, d_02, d_03,
		d_10, d_11, d_12, d_13;
	
	if(alg==0) // D = A * B' , there is no tc
		{
		if(td==0) // not transpose D
			{
			D[0+bs*0] = c_00;
			D[1+bs*0] = c_10;

			D[0+bs*1] = c_01;
			D[1+bs*1] = c_11;

			D[0+bs*2] = c_02;
			D[1+bs*2] = c_12;

			D[0+bs*3] = c_03;
			D[1+bs*3] = c_13;
			}
		else // transpose D
			{
			D[0+bs*0] = c_00;
			D[1+bs*0] = c_01;
			D[2+bs*0] = c_02;
			D[3+bs*0] = c_03;

			D[0+bs*1] = c_10;
			D[1+bs*1] = c_11;
			D[2+bs*1] = c_12;
			D[3+bs*1] = c_13;
			}
		}
	else 
		{
		if(tc==td) // not transpose C
			{
			d_00 = C[0+bs*0];
			d_10 = C[1+bs*0];
			
			d_01 = C[0+bs*1];
			d_11 = C[1+bs*1];
			
			d_02 = C[0+bs*2];
			d_12 = C[1+bs*2];
			
			d_03 = C[0+bs*3];
			d_13 = C[1+bs*3];
			}
		else // transpose C
			{
			d_00 = C[0+bs*0];
			d_01 = C[1+bs*0];
			d_02 = C[2+bs*0];
			d_03 = C[3+bs*0];

			d_10 = C[0+bs*1];
			d_11 = C[1+bs*1];
			d_12 = C[2+bs*1];
			d_13 = C[3+bs*1];
			}
		
		if(alg==1) // C += A * B'
			{
			d_00 += c_00;
			d_10 += c_10;

			d_01 += c_01;
			d_11 += c_11;

			d_02 += c_02;
			d_12 += c_12;

			d_03 += c_03;
			d_13 += c_13;
			}
		else // C -= A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;

			d_01 -= c_01;
			d_11 -= c_11;

			d_02 -= c_02;
			d_12 -= c_12;

			d_03 -= c_03;
			d_13 -= c_13;
			}

		if(td==0) // not transpose D
			{
			D[0+bs*0] = d_00;
			D[1+bs*0] = d_10;

			D[0+bs*1] = d_01;
			D[1+bs*1] = d_11;

			D[0+bs*2] = d_02;
			D[1+bs*2] = d_12;

			D[0+bs*3] = d_03;
			D[1+bs*3] = d_13;
			}
		else // transpose D
			{
			D[0+bs*0] = d_00;
			D[1+bs*0] = d_01;
			D[2+bs*0] = d_02;
			D[3+bs*0] = d_03;

			D[0+bs*1] = d_10;
			D[1+bs*1] = d_11;
			D[2+bs*1] = d_12;
			D[3+bs*1] = d_13;
			}
		}
	
	}



// normal-normal, 2x2 with data packed in 4
void kernel_dgemm_nn_2x2_lib4(int kmax, double *A, double *B, int sdb, double *C, double *D, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		a_0, a_1,
		b_0, b_1,
		c_00=0, c_01=0,
		c_10=0, c_11=0;
		
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		A += 4;
		B += 1;

		}
		
	double
		d_00, d_01,
		d_10, d_11;
	
	if(alg==0) // D = A * B'
		{
		if(td==0) // not transpose D
			{
			D[0+bs*0] = c_00;
			D[1+bs*0] = c_10;

			D[0+bs*1] = c_01;
			D[1+bs*1] = c_11;
			}
		else // transpose D
			{
			D[0+bs*0] = c_00;
			D[1+bs*0] = c_01;

			D[0+bs*1] = c_10;
			D[1+bs*1] = c_11;
			}
		}
	else 
		{
		if(tc==td) // not transpose C
			{
			d_00 = C[0+bs*0];
			d_10 = C[1+bs*0];
			
			d_01 = C[0+bs*1];
			d_11 = C[1+bs*1];
			}
		else // transpose C
			{
			d_00 = C[0+bs*0];
			d_01 = C[1+bs*0];
			
			d_10 = C[0+bs*1];
			d_11 = C[1+bs*1];
			}
		
		if(alg==1) // D = C + A * B'
			{
			d_00 += c_00;
			d_10 += c_10;

			d_01 += c_01;
			d_11 += c_11;
			}
		else // D = C - A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;

			d_01 -= c_01;
			d_11 -= c_11;
			}

		if(td==0) // not transpose D
			{
			D[0+bs*0] = d_00;
			D[1+bs*0] = d_10;

			D[0+bs*1] = d_01;
			D[1+bs*1] = d_11;
			}
		else // transpose D
			{
			D[0+bs*0] = d_00;
			D[1+bs*0] = d_01;

			D[0+bs*1] = d_10;
			D[1+bs*1] = d_11;
			}
		}
	
	}



// A is the diagonal of a matrix
void kernel_dgemm_diag_left_4_lib4(int kmax, double *A, double *B, double *C, double *D, int alg)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

#if 1

	__m256d
		sign,
		a_00,
		b_00,
		c_00, c_01, c_02, c_03,
		d_00, d_01, d_02, d_03;
		
	if(alg==-1)
		{
		a_00 = _mm256_load_pd( &A[0] );
		long long long_sign = 0x8000000000000000;
		sign = _mm256_broadcast_sd( (double *) &long_sign );
		a_00 = _mm256_xor_pd( sign, a_00 );
		}
	else
		{
		a_00 = _mm256_load_pd( &A[0] );
		}
	
	if(alg==0)
		{
		
		for(k=0; k<kmax-3; k+=4)
			{

			b_00 = _mm256_load_pd( &B[0] );
			d_00 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[4] );
			d_01 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[8] );
			d_02 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[12] );
			d_03 = _mm256_mul_pd( a_00, b_00 );

			_mm256_store_pd( &D[0], d_00 );
			_mm256_store_pd( &D[4], d_01 );
			_mm256_store_pd( &D[8], d_02 );
			_mm256_store_pd( &D[12], d_03 );
			
			B += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_00 = _mm256_load_pd( &B[0] );
			c_00 = _mm256_mul_pd( a_00, b_00 );

			_mm256_store_pd( &D[0], c_00 );
		
			B += 4;
			D += 4;
			
			}

		}
	else
		{

		for(k=0; k<kmax-3; k+=4)
			{
			
			b_00 = _mm256_load_pd( &B[0] );
			d_00 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[4] );
			d_01 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[8] );
			d_02 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[12] );
			d_03 = _mm256_mul_pd( a_00, b_00 );

			c_00 = _mm256_load_pd( &C[0] );
			d_00 = _mm256_add_pd( c_00, d_00 );
			c_01 = _mm256_load_pd( &C[4] );
			d_01 = _mm256_add_pd( c_01, d_01 );
			c_02 = _mm256_load_pd( &C[8] );
			d_02 = _mm256_add_pd( c_02, d_02 );
			c_03 = _mm256_load_pd( &C[12] );
			d_03 = _mm256_add_pd( c_03, d_03 );

			_mm256_store_pd( &D[0], d_00 );
			_mm256_store_pd( &D[4], d_01 );
			_mm256_store_pd( &D[8], d_02 );
			_mm256_store_pd( &D[12], d_03 );
	
			B += 16;
			C += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_00 = _mm256_load_pd( &B[0] );
			d_00 = _mm256_mul_pd( a_00, b_00 );

			c_00 = _mm256_load_pd( &C[0] );
			d_00 = _mm256_add_pd( c_00, d_00 );

			_mm256_store_pd( &D[0], d_00 );
	
			B += 4;
			C += 4;
			D += 4;
			
			}

		}


#else

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_0, c_1, c_2, c_3;
		
	if(alg==-1)
		{
		a_0 = - A[0];
		a_1 = - A[1];
		a_2 = - A[2];
		a_3 = - A[3];
		}
	else
		{
		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		}
	
	if(alg==0)
		{
		
		for(k=0; k<kmax-3; k+=4)
			{
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
			c_0 = a_0 * b_0;
			c_1 = a_1 * b_1;
			c_2 = a_2 * b_2;
			c_3 = a_3 * b_3;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
			D[2+bs*0] = c_2;
			D[3+bs*0] = c_3;
			

			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			b_2 = B[2+bs*1];
			b_3 = B[3+bs*1];
			
			c_0 = a_0 * b_0;
			c_1 = a_1 * b_1;
			c_2 = a_2 * b_2;
			c_3 = a_3 * b_3;

			D[0+bs*1] = c_0;
			D[1+bs*1] = c_1;
			D[2+bs*1] = c_2;
			D[3+bs*1] = c_3;
			

			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			b_2 = B[2+bs*2];
			b_3 = B[3+bs*2];
			
			c_0 = a_0 * b_0;
			c_1 = a_1 * b_1;
			c_2 = a_2 * b_2;
			c_3 = a_3 * b_3;

			D[0+bs*2] = c_0;
			D[1+bs*2] = c_1;
			D[2+bs*2] = c_2;
			D[3+bs*2] = c_3;
			

			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			b_2 = B[2+bs*3];
			b_3 = B[3+bs*3];
			
			c_0 = a_0 * b_0;
			c_1 = a_1 * b_1;
			c_2 = a_2 * b_2;
			c_3 = a_3 * b_3;

			D[0+bs*3] = c_0;
			D[1+bs*3] = c_1;
			D[2+bs*3] = c_2;
			D[3+bs*3] = c_3;

			B += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
			c_0 = a_0 * b_0;
			c_1 = a_1 * b_1;
			c_2 = a_2 * b_2;
			c_3 = a_3 * b_3;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
			D[2+bs*0] = c_2;
			D[3+bs*0] = c_3;
		
			B += 4;
			D += 4;
			
			}

		}
	else
		{

		for(k=0; k<kmax-3; k+=4)
			{
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
			c_0 = C[0+bs*0] + a_0 * b_0;
			c_1 = C[1+bs*0] + a_1 * b_1;
			c_2 = C[2+bs*0] + a_2 * b_2;
			c_3 = C[3+bs*0] + a_3 * b_3;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
			D[2+bs*0] = c_2;
			D[3+bs*0] = c_3;
			

			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			b_2 = B[2+bs*1];
			b_3 = B[3+bs*1];
			
			c_0 = C[0+bs*1] + a_0 * b_0;
			c_1 = C[1+bs*1] + a_1 * b_1;
			c_2 = C[2+bs*1] + a_2 * b_2;
			c_3 = C[3+bs*1] + a_3 * b_3;

			D[0+bs*1] = c_0;
			D[1+bs*1] = c_1;
			D[2+bs*1] = c_2;
			D[3+bs*1] = c_3;
			

			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			b_2 = B[2+bs*2];
			b_3 = B[3+bs*2];
			
			c_0 = C[0+bs*2] + a_0 * b_0;
			c_1 = C[1+bs*2] + a_1 * b_1;
			c_2 = C[2+bs*2] + a_2 * b_2;
			c_3 = C[3+bs*2] + a_3 * b_3;

			D[0+bs*2] = c_0;
			D[1+bs*2] = c_1;
			D[2+bs*2] = c_2;
			D[3+bs*2] = c_3;
			

			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			b_2 = B[2+bs*3];
			b_3 = B[3+bs*3];
			
			c_0 = C[0+bs*3] + a_0 * b_0;
			c_1 = C[1+bs*3] + a_1 * b_1;
			c_2 = C[2+bs*3] + a_2 * b_2;
			c_3 = C[3+bs*3] + a_3 * b_3;

			D[0+bs*3] = c_0;
			D[1+bs*3] = c_1;
			D[2+bs*3] = c_2;
			D[3+bs*3] = c_3;

			B += 16;
			C += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
			c_0 = C[0+bs*0] + a_0 * b_0;
			c_1 = C[1+bs*0] + a_1 * b_1;
			c_2 = C[2+bs*0] + a_2 * b_2;
			c_3 = C[3+bs*0] + a_3 * b_3;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
			D[2+bs*0] = c_2;
			D[3+bs*0] = c_3;
		
			B += 4;
			C += 4;
			D += 4;
			
			}

		}

#endif
	
	}



// A is the diagonal of a matrix
void kernel_dgemm_diag_left_3_lib4(int kmax, double *A, double *B, double *C, double *D, int alg)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	__m256i
		mask;

	__m256d
		sign,
		a_00,
		b_00,
		c_00, c_01, c_02, c_03,
		d_00, d_01, d_02, d_03;
	
	mask = _mm256_set_epi64x( 1, -1, -1, -1 );
		
	if(alg==-1)
		{
		a_00 = _mm256_load_pd( &A[0] );
		long long long_sign = 0x8000000000000000;
		sign = _mm256_broadcast_sd( (double *) &long_sign );
		a_00 = _mm256_xor_pd( sign, a_00 );
		}
	else
		{
		a_00 = _mm256_load_pd( &A[0] );
		}
	
	if(alg==0)
		{
		
		for(k=0; k<kmax-3; k+=4)
			{

			b_00 = _mm256_load_pd( &B[0] );
			d_00 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[4] );
			d_01 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[8] );
			d_02 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[12] );
			d_03 = _mm256_mul_pd( a_00, b_00 );

			_mm256_maskstore_pd( &D[0], mask, d_00 );
			_mm256_maskstore_pd( &D[4], mask, d_01 );
			_mm256_maskstore_pd( &D[8], mask, d_02 );
			_mm256_maskstore_pd( &D[12], mask, d_03 );
			
			B += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_00 = _mm256_load_pd( &B[0] );
			c_00 = _mm256_mul_pd( a_00, b_00 );

			_mm256_maskstore_pd( &D[0], mask, c_00 );
		
			B += 4;
			D += 4;
			
			}

		}
	else
		{

		for(k=0; k<kmax-3; k+=4)
			{
			
			b_00 = _mm256_load_pd( &B[0] );
			d_00 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[4] );
			d_01 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[8] );
			d_02 = _mm256_mul_pd( a_00, b_00 );
			b_00 = _mm256_load_pd( &B[12] );
			d_03 = _mm256_mul_pd( a_00, b_00 );

			c_00 = _mm256_load_pd( &C[0] );
			d_00 = _mm256_add_pd( c_00, d_00 );
			c_01 = _mm256_load_pd( &C[4] );
			d_01 = _mm256_add_pd( c_01, d_01 );
			c_02 = _mm256_load_pd( &C[8] );
			d_02 = _mm256_add_pd( c_02, d_02 );
			c_03 = _mm256_load_pd( &C[12] );
			d_03 = _mm256_add_pd( c_03, d_03 );

			_mm256_maskstore_pd( &D[0], mask, d_00 );
			_mm256_maskstore_pd( &D[4], mask, d_01 );
			_mm256_maskstore_pd( &D[8], mask, d_02 );
			_mm256_maskstore_pd( &D[12], mask, d_03 );
	
			B += 16;
			C += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_00 = _mm256_load_pd( &B[0] );
			d_00 = _mm256_mul_pd( a_00, b_00 );

			c_00 = _mm256_load_pd( &C[0] );
			d_00 = _mm256_add_pd( c_00, d_00 );

			_mm256_maskstore_pd( &D[0], mask, d_00 );
	
			B += 4;
			C += 4;
			D += 4;
			
			}

		}

	}



// A is the diagonal of a matrix
void kernel_dgemm_diag_left_2_lib4(int kmax, double *A, double *B, double *C, double *D, int alg)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	__m128d
		sign,
		a_00,
		b_00,
		c_00, c_01, c_02, c_03,
		d_00, d_01, d_02, d_03;
		
	if(alg==-1)
		{
		a_00 = _mm_load_pd( &A[0] );
		long long long_sign = 0x8000000000000000;
		sign = _mm_loaddup_pd( (double *) &long_sign );
		a_00 = _mm_xor_pd( sign, a_00 );
		}
	else
		{
		a_00 = _mm_load_pd( &A[0] );
		}
	
	if(alg==0)
		{
		
		for(k=0; k<kmax-3; k+=4)
			{

			b_00 = _mm_load_pd( &B[0] );
			d_00 = _mm_mul_pd( a_00, b_00 );
			b_00 = _mm_load_pd( &B[4] );
			d_01 = _mm_mul_pd( a_00, b_00 );
			b_00 = _mm_load_pd( &B[8] );
			d_02 = _mm_mul_pd( a_00, b_00 );
			b_00 = _mm_load_pd( &B[12] );
			d_03 = _mm_mul_pd( a_00, b_00 );

			_mm_store_pd( &D[0], d_00 );
			_mm_store_pd( &D[4], d_01 );
			_mm_store_pd( &D[8], d_02 );
			_mm_store_pd( &D[12], d_03 );
			
			B += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_00 = _mm_load_pd( &B[0] );
			c_00 = _mm_mul_pd( a_00, b_00 );

			_mm_store_pd( &D[0], c_00 );
		
			B += 4;
			D += 4;
			
			}

		}
	else
		{

		for(k=0; k<kmax-3; k+=4)
			{
			
			b_00 = _mm_load_pd( &B[0] );
			d_00 = _mm_mul_pd( a_00, b_00 );
			b_00 = _mm_load_pd( &B[4] );
			d_01 = _mm_mul_pd( a_00, b_00 );
			b_00 = _mm_load_pd( &B[8] );
			d_02 = _mm_mul_pd( a_00, b_00 );
			b_00 = _mm_load_pd( &B[12] );
			d_03 = _mm_mul_pd( a_00, b_00 );

			c_00 = _mm_load_pd( &C[0] );
			d_00 = _mm_add_pd( c_00, d_00 );
			c_01 = _mm_load_pd( &C[4] );
			d_01 = _mm_add_pd( c_01, d_01 );
			c_02 = _mm_load_pd( &C[8] );
			d_02 = _mm_add_pd( c_02, d_02 );
			c_03 = _mm_load_pd( &C[12] );
			d_03 = _mm_add_pd( c_03, d_03 );

			_mm_store_pd( &D[0], d_00 );
			_mm_store_pd( &D[4], d_01 );
			_mm_store_pd( &D[8], d_02 );
			_mm_store_pd( &D[12], d_03 );
	
			B += 16;
			C += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_00 = _mm_load_pd( &B[0] );
			d_00 = _mm_mul_pd( a_00, b_00 );

			c_00 = _mm_load_pd( &C[0] );
			d_00 = _mm_add_pd( c_00, d_00 );

			_mm_store_pd( &D[0], d_00 );
	
			B += 4;
			C += 4;
			D += 4;
			
			}

		}

	
	}


// A is the diagonal of a matrix
void kernel_dgemm_diag_left_1_lib4(int kmax, double *A, double *B, double *C, double *D, int alg)
	{
	
	if(kmax<=0)
		return;
	
	const int bs = 4;

	int k;

	double
		a_0,
		b_0,
		c_0;
		
	if(alg==-1)
		{
		a_0 = A[0];
		}
	else
		{
		a_0 = A[0];
		}
		
	if(alg==0)
		{
		
		for(k=0; k<kmax-3; k+=4)
			{
			
			b_0 = B[0+bs*0];
			
			c_0 = a_0 * b_0;

			D[0+bs*0] = c_0;
			

			b_0 = B[0+bs*1];
			
			c_0 = a_0 * b_0;

			D[0+bs*1] = c_0;
			

			b_0 = B[0+bs*2];
			
			c_0 = a_0 * b_0;

			D[0+bs*2] = c_0;
			

			b_0 = B[0+bs*3];
			
			c_0 = a_0 * b_0;

			D[0+bs*3] = c_0;

			B += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_0 = B[0+bs*0];
			
			c_0 = a_0 * b_0;

			D[0+bs*0] = c_0;
		
			B += 4;
			D += 4;
			
			}
		
		}
	else
		{
		
		for(k=0; k<kmax-3; k+=4)
			{
			
			b_0 = B[0+bs*0];
			
			c_0 = C[0+bs*0] + a_0 * b_0;

			D[0+bs*0] = c_0;
			

			b_0 = B[0+bs*1];
			
			c_0 = C[0+bs*1] + a_0 * b_0;

			D[0+bs*1] = c_0;
			

			b_0 = B[0+bs*2];
			
			c_0 = C[0+bs*2] + a_0 * b_0;

			D[0+bs*2] = c_0;
			

			b_0 = B[0+bs*3];
			
			c_0 = C[0+bs*3] + a_0 * b_0;

			D[0+bs*3] = c_0;

			B += 16;
			C += 16;
			D += 16;
			
			}
		for(; k<kmax; k++)
			{
			
			b_0 = B[0+bs*0];
			
			c_0 = C[0+bs*0] + a_0 * b_0;

			D[0+bs*0] = c_0;
		
			B += 4;
			C += 4;
			D += 4;
			
			}

		}
		
	}



