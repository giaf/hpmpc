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
void kernel_dtrsm_pp_nt_8x4_lib4(int kadd, int ksub, double *A0, double *A1, double *B, double *C0, double *C1, double *D0, double *D1, int ldc, double *fact)
	{
	
/*	if(kmax<=0)*/
/*		return;*/
	
	const int d_ncl = 2;
	
	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_tmp0, ab_tmp1, // temporary results
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


	for(k=0; k<kadd-3; k+=4)
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
		
	if(ksub>0)
		{
		if(kadd>0)
			{
			A0 += 4*((d_ncl-kadd%d_ncl)%d_ncl);
			A1 += 4*((d_ncl-kadd%d_ncl)%d_ncl);
			B  += 4*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		// prefetch
		a_0123        = _mm256_load_pd( &A0[0] );
		a_4567        = _mm256_load_pd( &A1[0] );
		b_0123        = _mm256_load_pd( &B[0] );
		}

	for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_sub_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_sub_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_sub_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_sub_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_sub_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_sub_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );


/*	__builtin_prefetch( A+48 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_sub_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_sub_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_sub_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );


/*	__builtin_prefetch( A+56 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_sub_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_sub_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_sub_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

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
	d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
	d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
	d_02_12_22_32 = _mm256_load_pd( &C0[0+ldc*2] );
	d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
	c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
	c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
	d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
	d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
	d_03_13_23_33 = _mm256_load_pd( &C0[0+ldc*3] );
	d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
	c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
	c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
	d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
	d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
	d_42_52_62_72 = _mm256_load_pd( &C1[0+ldc*2] );
	d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
	c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
	c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
	d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
	d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
	d_43_53_63_73 = _mm256_load_pd( &C1[0+ldc*3] );
	d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );
		
	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	d_40_50_60_70 = _mm256_mul_pd( d_40_50_60_70, a_00 );
	_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
	_mm256_store_pd( &D1[0+ldc*0], d_40_50_60_70 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	ab_tmp0 = _mm256_mul_pd( d_00_10_20_30, a_10 );
	ab_tmp1 = _mm256_mul_pd( d_40_50_60_70, a_10 );
	d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_tmp0 );
	d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, ab_tmp1 );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
	_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
	_mm256_store_pd( &D1[0+ldc*1], d_41_51_61_71 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	ab_tmp0 = _mm256_mul_pd( d_00_10_20_30, a_20 );
	ab_tmp1 = _mm256_mul_pd( d_40_50_60_70, a_20 );
	d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, ab_tmp0 );
	d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, ab_tmp1 );
	ab_tmp0 = _mm256_mul_pd( d_01_11_21_31, a_21 );
	ab_tmp1 = _mm256_mul_pd( d_41_51_61_71, a_21 );
	d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, ab_tmp0 );
	d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, ab_tmp1 );
	d_02_12_22_32 = _mm256_mul_pd( d_02_12_22_32, a_22 );
	d_42_52_62_72 = _mm256_mul_pd( d_42_52_62_72, a_22 );
	_mm256_store_pd( &D0[0+ldc*2], d_02_12_22_32 );
	_mm256_store_pd( &D1[0+ldc*2], d_42_52_62_72 );

	a_30 = _mm256_broadcast_sd( &fact[6] );
	a_31 = _mm256_broadcast_sd( &fact[7] );
	a_32 = _mm256_broadcast_sd( &fact[8] );
	a_33 = _mm256_broadcast_sd( &fact[9] );
	ab_tmp0 = _mm256_mul_pd( d_00_10_20_30, a_30 );
	ab_tmp1 = _mm256_mul_pd( d_40_50_60_70, a_30 );
	d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, ab_tmp0 );
	d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, ab_tmp1 );
	ab_tmp0 = _mm256_mul_pd( d_01_11_21_31, a_31 );
	ab_tmp1 = _mm256_mul_pd( d_41_51_61_71, a_31 );
	d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, ab_tmp0 );
	d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, ab_tmp1 );
	ab_tmp0 = _mm256_mul_pd( d_02_12_22_32, a_32 );
	ab_tmp1 = _mm256_mul_pd( d_42_52_62_72, a_32 );
	d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, ab_tmp0 );
	d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, ab_tmp1 );
	d_03_13_23_33 = _mm256_mul_pd( d_03_13_23_33, a_33 );
	d_43_53_63_73 = _mm256_mul_pd( d_43_53_63_73, a_33 );
	_mm256_store_pd( &D0[0+ldc*3], d_03_13_23_33 );
	_mm256_store_pd( &D1[0+ldc*3], d_43_53_63_73 );

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dtrsm_pp_nt_4x4_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, int ldc, double *fact)
	{
	
/*	if(kmax<=0)*/
/*		return;*/
	
	const int d_ncl = 2;

	int k;
	
	__m256d
		a_0123,
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


	for(k=0; k<kadd-3; k+=4)
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

	if(ksub>0)
		{
		if(kadd>0)
			{
			A += 4*((d_ncl-kadd%d_ncl)%d_ncl);
			B += 4*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		// prefetch
		a_0123        = _mm256_load_pd( &A[0] );
		b_0123        = _mm256_load_pd( &B[0] );
		}

	for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
		
		A += 16;
		B += 16;

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
	d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
	d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
	d_02_12_22_32 = _mm256_load_pd( &C[0+ldc*2] );
	d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
	c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
	c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
	d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
	d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
	d_03_13_23_33 = _mm256_load_pd( &C[0+ldc*3] );
	d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );

	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	ab_temp = _mm256_mul_pd( d_00_10_20_30, a_10 );
	d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_temp );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	ab_temp = _mm256_mul_pd( d_00_10_20_30, a_20 );
	d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, ab_temp );
	ab_temp = _mm256_mul_pd( d_01_11_21_31, a_21 );
	d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, ab_temp );
	d_02_12_22_32 = _mm256_mul_pd( d_02_12_22_32, a_22 );
	_mm256_store_pd( &D[0+ldc*2], d_02_12_22_32 );

	a_30 = _mm256_broadcast_sd( &fact[6] );
	a_31 = _mm256_broadcast_sd( &fact[7] );
	a_32 = _mm256_broadcast_sd( &fact[8] );
	a_33 = _mm256_broadcast_sd( &fact[9] );
	ab_temp = _mm256_mul_pd( d_00_10_20_30, a_30 );
	d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, ab_temp );
	ab_temp = _mm256_mul_pd( d_01_11_21_31, a_31 );
	d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, ab_temp );
	ab_temp = _mm256_mul_pd( d_02_12_22_32, a_32 );
	d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, ab_temp );
	d_03_13_23_33 = _mm256_mul_pd( d_03_13_23_33, a_33 );
	_mm256_store_pd( &D[0+ldc*3], d_03_13_23_33 );

	}

