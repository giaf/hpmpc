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
void kernel_dpotrf_pp_nt_8x4_lib4(int kadd, int ksub, double *A0, double *A1, double *B, double *C0, double *C1, double *D0, double *D1, int ldc, double *fact, int *info)
	{
	
/*	if(kmax<=0)*/
/*		return;*/
	
	const int d_ncl = 2;
	
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


	for(k=0; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
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
		
		
/*	__builtin_prefetch( A+40 );*/
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


/*	__builtin_prefetch( A+48 );*/
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


/*	__builtin_prefetch( A+56 );*/
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
	
	if(kadd%4>=2)
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

	if(kadd%2==1)
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
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_sub_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_sub_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_sub_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_sub_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_sub_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_sub_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_sub_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_sub_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_sub_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_0123 );
		c_40_51_62_73 = _mm256_sub_pd( c_40_51_62_73, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch 
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_1032 );
		c_41_50_63_72 = _mm256_sub_pd( c_41_50_63_72, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_3210 );
		c_43_52_61_70 = _mm256_sub_pd( c_43_52_61_70, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_temp );
		
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
		
	// factorize the upper 4x4 matrix
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_20, sa_30, sa_11, sa_21, sa_31, sa_22, sa_32, sa_33;

	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	


	if(1) // TODO avoid correction if a_jj==0
	{	

	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00_10_20_30) );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		sa_10 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_00_10_20_30), zeros_ones, 0x1 );
		sa_20 = _mm256_extractf128_pd( d_00_10_20_30, 0x1 ); // a_20 & a_30
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D0[0+ldc*0], sa_00 ); // a_00
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		sa_00 = _mm_movedup_pd( sa_00 );
		sa_10 = _mm_mul_sd( sa_10, sa_00 );
		sa_10 = _mm_movedup_pd( sa_10 );
		sa_20 = _mm_mul_pd( sa_20, sa_00 ); // a_20 & a_30
		}
	else // comile
		{
/*		*info = 1; return;*/
		sa_00 = _mm_setzero_pd();
		_mm_store_sd( &D0[0+ldc*0], sa_00 ); // a_00
		sa_10 = sa_00;
		sa_20 = sa_00; // a_20 & a_30
		}
	_mm_store_sd( &D0[1+ldc*0], sa_10 ); // a_10
	_mm_store_pd( &D0[2+ldc*0], sa_20 ); // a_20 & a_30

	// second row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_01_11_21_31), zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_10, sa_10 );
	sa_11 = _mm_sub_sd( sa_11, sab_temp );
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		sa_21 = _mm256_extractf128_pd( d_01_11_21_31, 0x1 ); // a_21 & a_31
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D0[1+ldc*1], sa_11 ); // a_11
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		sa_11 = _mm_movedup_pd( sa_11 );
		sab_temp = _mm_mul_pd( sa_20, sa_10 ); // a_21 & a_31
		sa_21 = _mm_sub_pd( sa_21, sab_temp ); // a_21 & a_31
		sa_21 = _mm_mul_pd( sa_21, sa_11 ); // a_21 & a_31
		}
	else // comile
		{
/*		*info = 1; return;*/
		sa_11 = _mm_setzero_pd();
		_mm_store_sd( &D0[1+ldc*1], sa_11 ); // a_11
		sa_21 = sa_11; // a_21 & a_31
		}
	_mm_store_pd( &D0[2+ldc*1], sa_21 ); // a_21 & a_31
	
	// third row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_22 = _mm256_extractf128_pd( d_02_12_22_32, 0x1 ); // a_22 & a_32
	sab_temp = _mm_movedup_pd( sa_20 );
	sab_temp = _mm_mul_pd( sa_20, sab_temp );
	sa_22 = _mm_sub_pd( sa_22, sab_temp );
	sab_temp = _mm_movedup_pd( sa_21 );
	sab_temp = _mm_mul_pd( sa_21, sab_temp );
	sa_22 = _mm_sub_pd( sa_22, sab_temp );
	if( _mm_comigt_sd ( sa_22, zeros_ones ) )
		{
		sa_32 = _mm_shuffle_pd( sa_22, zeros_ones, 0x1 ); // a_31
		sa_22 = _mm_sqrt_sd( sa_22, sa_22 );
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D0[2+ldc*2], sa_22 ); // a_22
		sa_22 = _mm_div_sd( zeros_ones, sa_22 );
		sa_32 = _mm_mul_sd( sa_32, sa_22 );
		}
	else // comile
		{
/*		*info = 1; return;*/
		sa_22 = _mm_setzero_pd();
		_mm_store_sd( &D0[2+ldc*2], sa_22 ); // a_22
		sa_32 = sa_22; // a_21 & a_31
		}
	_mm_store_sd( &D0[3+ldc*2], sa_32 ); // a_32

	sa_30 = _mm_shuffle_pd( sa_20, zeros_ones, 0x1 ); // a_30
	sa_31 = _mm_shuffle_pd( sa_21, zeros_ones, 0x1 ); // a_31

	// fourth row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_33 = _mm_shuffle_pd( _mm256_extractf128_pd( d_03_13_23_33, 0x1 ), zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_30, sa_30 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	sab_temp = _mm_mul_sd( sa_31, sa_31 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	sab_temp = _mm_mul_sd( sa_32, sa_32 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	if( _mm_comigt_sd ( sa_33, zeros_ones ) )
		{
		sa_33 = _mm_sqrt_sd( sa_33, sa_33 );
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D0[3+ldc*3], sa_33 ); // a_33
		sa_33 = _mm_div_sd( zeros_ones, sa_33 );
		}
	else // comile
		{
/*		*info = 1; return;*/
		sa_33 = _mm_setzero_pd();
		_mm_store_sd( &D0[3+ldc*3], sa_33 ); // a_33
		}

	// duplicate & store

	_mm_store_sd( &fact[0], sa_00 );
	_mm_store_sd( &fact[1], sa_10 );
	_mm_store_sd( &fact[3], sa_20 );
	_mm_store_sd( &fact[6], sa_30 );
	_mm_store_sd( &fact[2], sa_11 );
	_mm_store_sd( &fact[4], sa_21 );
	_mm_store_sd( &fact[7], sa_31 );
	_mm_store_sd( &fact[5], sa_22 );
	_mm_store_sd( &fact[8], sa_32 );
	_mm_store_sd( &fact[9], sa_33 );

/*	sa_00 = _mm_movedup_pd( sa_00 );*/
/*	sa_10 = _mm_movedup_pd( sa_10 );*/
	sa_20 = _mm_movedup_pd( sa_20 );
	sa_30 = _mm_movedup_pd( sa_30 );
/*	sa_11 = _mm_movedup_pd( sa_11 );*/
	sa_21 = _mm_movedup_pd( sa_21 );
	sa_31 = _mm_movedup_pd( sa_31 );
	sa_22 = _mm_movedup_pd( sa_22 );
	sa_32 = _mm_movedup_pd( sa_32 );
	sa_33 = _mm_movedup_pd( sa_33 );

	a_00 = _mm256_castpd128_pd256( sa_00 );
	a_10 = _mm256_castpd128_pd256( sa_10 );
	a_20 = _mm256_castpd128_pd256( sa_20 );
	a_30 = _mm256_castpd128_pd256( sa_30 );
	a_11 = _mm256_castpd128_pd256( sa_11 );
	a_21 = _mm256_castpd128_pd256( sa_21 );
	a_31 = _mm256_castpd128_pd256( sa_31 );
	a_22 = _mm256_castpd128_pd256( sa_22 );
	a_32 = _mm256_castpd128_pd256( sa_32 );
	a_33 = _mm256_castpd128_pd256( sa_33 );

	a_00 = _mm256_permute2f128_pd( a_00, a_00, 0x0 );
	a_10 = _mm256_permute2f128_pd( a_10, a_10, 0x0 );
	a_20 = _mm256_permute2f128_pd( a_20, a_20, 0x0 );
	a_30 = _mm256_permute2f128_pd( a_30, a_30, 0x0 );
	a_11 = _mm256_permute2f128_pd( a_11, a_11, 0x0 );
	a_21 = _mm256_permute2f128_pd( a_21, a_21, 0x0 );
	a_31 = _mm256_permute2f128_pd( a_31, a_31, 0x0 );
	a_22 = _mm256_permute2f128_pd( a_22, a_22, 0x0 );
	a_32 = _mm256_permute2f128_pd( a_32, a_32, 0x0 );
	a_33 = _mm256_permute2f128_pd( a_33, a_33, 0x0 );

	} // end of comment out
	else
	{

	zeros_ones = _mm_setzero_pd();
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00_10_20_30) );
	if( _mm_comile_sd ( sa_00, zeros_ones ) ) { *info = 1; return; }
sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
	sa_10 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_00_10_20_30), zeros_ones, 0x1 );
	sa_20 = _mm256_extractf128_pd( d_00_10_20_30, 0x1 );
	sa_30 = _mm_shuffle_pd( sa_20, zeros_ones, 0x1 );
	zeros_ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &D0[0+ldc*0], sa_00 );
sa_00 = _mm_div_sd( zeros_ones, sa_00 );
	sa_11 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_01_11_21_31), zeros_ones, 0x1 );
	sa_21 = _mm256_extractf128_pd( d_01_11_21_31, 0x1 );
	sa_31 = _mm_shuffle_pd( sa_21, zeros_ones, 0x1 );
	_mm_store_sd( &fact[0], sa_00 );
	sa_10 = _mm_mul_sd( sa_10, sa_00 );
	sa_20 = _mm_mul_sd( sa_20, sa_00 );
	sa_30 = _mm_mul_sd( sa_30, sa_00 );
	sab_temp = _mm_mul_sd( sa_10, sa_10 );
	zeros_ones = _mm_setzero_pd();
	sa_11 = _mm_sub_sd( sa_11, sab_temp );
	if( _mm_comile_sd ( sa_11, zeros_ones ) ) { *info = 1; return; }
sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
	sa_00 = _mm_movedup_pd( sa_00 );
	a_00 = _mm256_castpd128_pd256( sa_00 );
	a_00 = _mm256_permute2f128_pd( a_00, a_00, 0x0 );
	_mm_store_sd( &D0[1+ldc*0], sa_10 );
	_mm_store_sd( &fact[1], sa_10 );
	sa_10 = _mm_movedup_pd( sa_10 );
	_mm_store_sd( &D0[2+ldc*0], sa_20 );
	_mm_store_sd( &fact[3], sa_20 );
	sa_20 = _mm_movedup_pd( sa_20 );
	_mm_store_sd( &D0[3+ldc*0], sa_30 );
	_mm_store_sd( &fact[6], sa_30 );
	sa_30 = _mm_movedup_pd( sa_30 );
	zeros_ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &D0[1+ldc*1], sa_11 );
sa_11 = _mm_div_sd( zeros_ones, sa_11 );
	sa_22 = _mm256_extractf128_pd( d_02_12_22_32, 0x1 );
	sa_32 = _mm_shuffle_pd( sa_22, zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_20, sa_20 );
	sa_22 = _mm_sub_sd( sa_22, sab_temp );
	_mm_store_sd( &fact[2], sa_11 );
	sab_temp = _mm_mul_sd( sa_20, sa_10 );
	sa_21 = _mm_sub_sd( sa_21, sab_temp );
	sab_temp = _mm_mul_sd( sa_30, sa_10 );
	sa_31 = _mm_sub_sd( sa_31, sab_temp );
	sa_21 = _mm_mul_sd( sa_21, sa_11 );
	sa_31 = _mm_mul_sd( sa_31, sa_11 );
	sab_temp = _mm_mul_sd( sa_21, sa_21 );
	sa_22 = _mm_sub_sd( sa_22, sab_temp );
	zeros_ones = _mm_setzero_pd();
	if( _mm_comile_sd ( sa_22, zeros_ones ) ) { *info = 1; return; }
sa_22 = _mm_sqrt_sd( sa_22, sa_22 );
	sa_11 = _mm_movedup_pd( sa_11 );
	a_10 = _mm256_castpd128_pd256( sa_10 );
	a_10 = _mm256_permute2f128_pd( a_10, a_10, 0x0 );
	sa_33 = _mm_shuffle_pd( _mm256_extractf128_pd( d_03_13_23_33, 0x1 ), zeros_ones, 0x1 );
	_mm_store_sd( &D0[2+ldc*1], sa_21 );
	_mm_store_sd( &fact[4], sa_21 );
	_mm_store_sd( &D0[3+ldc*1], sa_31 );
	_mm_store_sd( &fact[7], sa_31 );
	sab_temp = _mm_mul_sd( sa_30, sa_30 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	sab_temp = _mm_mul_sd( sa_31, sa_31 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	_mm_store_sd( &D0[2+ldc*2], sa_22 );
	zeros_ones = _mm_set_sd( 1.0 );
sa_22 = _mm_div_sd( zeros_ones, sa_22 );
	sa_21 = _mm_movedup_pd( sa_21 );
	zeros_ones = _mm_setzero_pd();
	a_11 = _mm256_castpd128_pd256( sa_11 );
	a_11 = _mm256_permute2f128_pd( a_11, a_11, 0x0 );
	sa_31 = _mm_movedup_pd( sa_31 );
	sab_temp = _mm_mul_sd( sa_30, sa_20 );
	sa_32 = _mm_sub_sd( sa_32, sab_temp );
	sab_temp = _mm_mul_sd( sa_31, sa_21 );
	sa_32 = _mm_sub_sd( sa_32, sab_temp );
	_mm_store_sd( &fact[5], sa_22 );
	sa_32 = _mm_mul_sd( sa_32, sa_22 );
	sab_temp = _mm_mul_sd( sa_32, sa_32 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	if( _mm_comile_sd ( sa_33, zeros_ones ) ) { *info = 1; return; }
sa_33 = _mm_sqrt_sd( sa_33, sa_33 );
	sa_22 = _mm_movedup_pd( sa_22 );
	a_20 = _mm256_castpd128_pd256( sa_20 );
	a_20 = _mm256_permute2f128_pd( a_20, a_20, 0x0 );
	a_30 = _mm256_castpd128_pd256( sa_30 );
	a_30 = _mm256_permute2f128_pd( a_30, a_30, 0x0 );
	_mm_store_sd( &D0[3+ldc*2], sa_32 );
	_mm_store_sd( &fact[8], sa_32 );
	sa_32 = _mm_movedup_pd( sa_32 );
	a_32 = _mm256_castpd128_pd256( sa_32 );
	a_32 = _mm256_permute2f128_pd( a_32, a_32, 0x0 );
	zeros_ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &D0[3+ldc*3], sa_33 );
sa_33 = _mm_div_sd( zeros_ones, sa_33 );
	a_31 = _mm256_castpd128_pd256( sa_31 );
	a_31 = _mm256_permute2f128_pd( a_31, a_31, 0x0 );
	a_22 = _mm256_castpd128_pd256( sa_22 );
	a_22 = _mm256_permute2f128_pd( a_22, a_22, 0x0 );
	a_21 = _mm256_castpd128_pd256( sa_21 );
	a_21 = _mm256_permute2f128_pd( a_21, a_21, 0x0 );
	_mm_store_sd( &fact[9], sa_33 );
	sa_33 = _mm_movedup_pd( sa_33 );
	a_33 = _mm256_castpd128_pd256( sa_33 );
	a_33 = _mm256_permute2f128_pd( a_33, a_33, 0x0 );
	}

/*printf("\nciao %f %f %f %f %f %f %f %f %f %f\n", fact[0], fact[1], fact[2], fact[3], fact[4], fact[5], fact[6], fact[7], fact[8], fact[9]);*/

	// solve the lower 4x4 matrix
	d_40_50_60_70 = _mm256_mul_pd( d_40_50_60_70, a_00 );
	_mm256_store_pd( &D1[0+ldc*0], d_40_50_60_70 );

	ab_temp = _mm256_mul_pd( d_40_50_60_70, a_10 );
	d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, ab_temp );
	d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
	_mm256_store_pd( &D1[0+ldc*1], d_41_51_61_71 );

	ab_temp = _mm256_mul_pd( d_40_50_60_70, a_20 );
	d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, ab_temp );
	ab_temp = _mm256_mul_pd( d_41_51_61_71, a_21 );
	d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, ab_temp );
	d_42_52_62_72 = _mm256_mul_pd( d_42_52_62_72, a_22 );
	_mm256_store_pd( &D1[0+ldc*2], d_42_52_62_72 );

	ab_temp = _mm256_mul_pd( d_40_50_60_70, a_30 );
	d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, ab_temp );
	ab_temp = _mm256_mul_pd( d_41_51_61_71, a_31 );
	d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, ab_temp );
	ab_temp = _mm256_mul_pd( d_42_52_62_72, a_32 );
	d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, ab_temp );
	d_43_53_63_73 = _mm256_mul_pd( d_43_53_63_73, a_33 );
	_mm256_store_pd( &D1[0+ldc*3], d_43_53_63_73 );

/*d_print_mat(4, 4, D0, 4);*/
/*d_print_mat(4, 4, D1, 4);*/

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dpotrf_pp_nt_4x4_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, int ldc, double *fact, int *info)
	{
	
/*	if(kmax<=0)*/
/*		return;*/
	
	const int d_ncl = 2;
	
	int k;
	
	__m256d
		a_0123, //A_0123,
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
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+48 );*/
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
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch 
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kadd%4>=2)
		{
		
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
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch 
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kadd%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch */
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch */
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
/*		A0 += 4;*/
/*		A1 += 4;*/
/*		B  += 4;*/
		
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
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch 
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
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
		
	// factorize the upper 4x4 matrix
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_20, sa_30, sa_11, sa_21, sa_31, sa_22, sa_32, sa_33;

	if(1) // TODO avoid correction if a_jj==0
	{	

	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00_10_20_30) );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		sa_10 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_00_10_20_30), zeros_ones, 0x1 );
		sa_20 = _mm256_extractf128_pd( d_00_10_20_30, 0x1 ); // a_20 & a_30
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[0+ldc*0], sa_00 ); // a_00
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		sa_00 = _mm_movedup_pd( sa_00 );
		sa_10 = _mm_mul_sd( sa_10, sa_00 );
		sa_10 = _mm_movedup_pd( sa_10 );
		sa_20 = _mm_mul_pd( sa_20, sa_00 ); // a_20 & a_30
		}
	else // comile
		{
/*		*info = 1; return;*/
		sa_00 = _mm_setzero_pd();
		_mm_store_sd( &D[0+ldc*0], sa_00 ); // a_00
		sa_10 = sa_00;
		sa_20 = sa_00; // a_20 & a_30
		}
	_mm_store_sd( &D[1+ldc*0], sa_10 ); // a_10
	_mm_store_pd( &D[2+ldc*0], sa_20 ); // a_20 & a_30

	// second row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_01_11_21_31), zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_10, sa_10 );
	sa_11 = _mm_sub_sd( sa_11, sab_temp );
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		sa_21 = _mm256_extractf128_pd( d_01_11_21_31, 0x1 ); // a_21 & a_31
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[1+ldc*1], sa_11 ); // a_11
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		sa_11 = _mm_movedup_pd( sa_11 );
		sab_temp = _mm_mul_pd( sa_20, sa_10 ); // a_21 & a_31
		sa_21 = _mm_sub_pd( sa_21, sab_temp ); // a_21 & a_31
		sa_21 = _mm_mul_pd( sa_21, sa_11 ); // a_21 & a_31
		}
	else // comile
		{
/*		*info = 1; return;*/
		sa_11 = _mm_setzero_pd();
		_mm_store_sd( &D[1+ldc*1], sa_11 ); // a_11
		sa_21 = sa_11; // a_21 & a_31
		}
	_mm_store_pd( &D[2+ldc*1], sa_21 ); // a_21 & a_31
	
	// third row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_22 = _mm256_extractf128_pd( d_02_12_22_32, 0x1 ); // a_22 & a_32
	sab_temp = _mm_movedup_pd( sa_20 );
	sab_temp = _mm_mul_pd( sa_20, sab_temp );
	sa_22 = _mm_sub_pd( sa_22, sab_temp );
	sab_temp = _mm_movedup_pd( sa_21 );
	sab_temp = _mm_mul_pd( sa_21, sab_temp );
	sa_22 = _mm_sub_pd( sa_22, sab_temp );
	if( _mm_comigt_sd ( sa_22, zeros_ones ) )
		{
		sa_32 = _mm_shuffle_pd( sa_22, zeros_ones, 0x1 ); // a_31
		sa_22 = _mm_sqrt_sd( sa_22, sa_22 );
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[2+ldc*2], sa_22 ); // a_22
		sa_22 = _mm_div_sd( zeros_ones, sa_22 );
		sa_32 = _mm_mul_sd( sa_32, sa_22 );
		}
	else // comile
		{
/*		*info = 1; return;*/
		sa_22 = _mm_setzero_pd();
		_mm_store_sd( &D[2+ldc*2], sa_22 ); // a_22
		sa_32 = sa_22; // a_21 & a_31
		}
	_mm_store_sd( &D[3+ldc*2], sa_32 ); // a_32

	sa_30 = _mm_shuffle_pd( sa_20, zeros_ones, 0x1 ); // a_30
	sa_31 = _mm_shuffle_pd( sa_21, zeros_ones, 0x1 ); // a_31

	// fourth row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_33 = _mm_shuffle_pd( _mm256_extractf128_pd( d_03_13_23_33, 0x1 ), zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_30, sa_30 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	sab_temp = _mm_mul_sd( sa_31, sa_31 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	sab_temp = _mm_mul_sd( sa_32, sa_32 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	if( _mm_comigt_sd ( sa_33, zeros_ones ) )
		{
		sa_33 = _mm_sqrt_sd( sa_33, sa_33 );
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[3+ldc*3], sa_33 ); // a_33
		sa_33 = _mm_div_sd( zeros_ones, sa_33 );
		}
	else // comile
		{
/*		*info = 1; return;*/
		sa_33 = _mm_setzero_pd();
		_mm_store_sd( &D[3+ldc*3], sa_33 ); // a_33
		}

	// duplicate & store

	_mm_store_sd( &fact[0], sa_00 );
	_mm_store_sd( &fact[1], sa_10 );
	_mm_store_sd( &fact[3], sa_20 );
	_mm_store_sd( &fact[6], sa_30 );
	_mm_store_sd( &fact[2], sa_11 );
	_mm_store_sd( &fact[4], sa_21 );
	_mm_store_sd( &fact[7], sa_31 );
	_mm_store_sd( &fact[5], sa_22 );
	_mm_store_sd( &fact[8], sa_32 );
	_mm_store_sd( &fact[9], sa_33 );

	} // end of comment out
	else
	{
	zeros_ones = _mm_setzero_pd( );
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00_10_20_30) );
	if( _mm_comile_sd ( sa_00, zeros_ones ) ) { *info = 1; return; }
sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
	sa_10 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_00_10_20_30), zeros_ones, 0x1 );
	sa_20 = _mm256_extractf128_pd( d_00_10_20_30, 0x1 );
	sa_30 = _mm_shuffle_pd( sa_20, zeros_ones, 0x1 );
	zeros_ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &D[0+ldc*0], sa_00 );
sa_00 = _mm_div_sd( zeros_ones, sa_00 );
	sa_11 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_01_11_21_31), zeros_ones, 0x1 );
	sa_21 = _mm256_extractf128_pd( d_01_11_21_31, 0x1 );
	sa_31 = _mm_shuffle_pd( sa_21, zeros_ones, 0x1 );
	_mm_store_sd( &fact[0], sa_00 );
	sa_10 = _mm_mul_sd( sa_10, sa_00 );
	sa_20 = _mm_mul_sd( sa_20, sa_00 );
	sa_30 = _mm_mul_sd( sa_30, sa_00 );
	sab_temp = _mm_mul_sd( sa_10, sa_10 );
	zeros_ones = _mm_setzero_pd( );
	sa_11 = _mm_sub_sd( sa_11, sab_temp );
	if( _mm_comile_sd ( sa_11, zeros_ones ) ) { *info = 1; return; }
sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
	_mm_store_sd( &D[1+ldc*0], sa_10 );
	_mm_store_sd( &fact[1], sa_10 );
	_mm_store_sd( &D[2+ldc*0], sa_20 );
	_mm_store_sd( &fact[3], sa_20 );
	_mm_store_sd( &D[3+ldc*0], sa_30 );
	_mm_store_sd( &fact[6], sa_30 );
	zeros_ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &D[1+ldc*1], sa_11 );
sa_11 = _mm_div_sd( zeros_ones, sa_11 );
	sa_22 = _mm256_extractf128_pd( d_02_12_22_32, 0x1 );
	sa_32 = _mm_shuffle_pd( sa_22, zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_20, sa_20 );
	sa_22 = _mm_sub_sd( sa_22, sab_temp );
	_mm_store_sd( &fact[2], sa_11 );
	sab_temp = _mm_mul_sd( sa_20, sa_10 );
	sa_21 = _mm_sub_sd( sa_21, sab_temp );
	sab_temp = _mm_mul_sd( sa_30, sa_10 );
	sa_31 = _mm_sub_sd( sa_31, sab_temp );
	sa_21 = _mm_mul_sd( sa_21, sa_11 );
	sa_31 = _mm_mul_sd( sa_31, sa_11 );
	sab_temp = _mm_mul_sd( sa_21, sa_21 );
	sa_22 = _mm_sub_sd( sa_22, sab_temp );
	zeros_ones = _mm_setzero_pd( );
	if( _mm_comile_sd ( sa_22, zeros_ones ) ) { *info = 1; return; }
sa_22 = _mm_sqrt_sd( sa_22, sa_22 );
	sa_33 = _mm_shuffle_pd( _mm256_extractf128_pd( d_03_13_23_33, 0x1 ), zeros_ones, 0x1 );
	_mm_store_sd( &D[2+ldc*1], sa_21 );
	_mm_store_sd( &fact[4], sa_21 );
	_mm_store_sd( &D[3+ldc*1], sa_31 );
	_mm_store_sd( &fact[7], sa_31 );
	sab_temp = _mm_mul_sd( sa_30, sa_30 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	sab_temp = _mm_mul_sd( sa_31, sa_31 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	_mm_store_sd( &D[2+ldc*2], sa_22 );
	zeros_ones = _mm_set_sd( 1.0 );
sa_22 = _mm_div_sd( zeros_ones, sa_22 );
	zeros_ones = _mm_setzero_pd( );
	sab_temp = _mm_mul_sd( sa_30, sa_20 );
	sa_32 = _mm_sub_sd( sa_32, sab_temp );
	sab_temp = _mm_mul_sd( sa_31, sa_21 );
	sa_32 = _mm_sub_sd( sa_32, sab_temp );
	_mm_store_sd( &fact[5], sa_22 );
	sa_32 = _mm_mul_sd( sa_32, sa_22 );
	sab_temp = _mm_mul_sd( sa_32, sa_32 );
	sa_33 = _mm_sub_sd( sa_33, sab_temp );
	if( _mm_comile_sd ( sa_33, zeros_ones ) ) { *info = 1; return; }
sa_33 = _mm_sqrt_sd( sa_33, sa_33 );
	_mm_store_sd( &D[3+ldc*2], sa_32 );
	_mm_store_sd( &fact[8], sa_32 );
	zeros_ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &D[3+ldc*3], sa_33 );
sa_33 = _mm_div_sd( zeros_ones, sa_33 );
	_mm_store_sd( &fact[9], sa_33 );
	}

	}

