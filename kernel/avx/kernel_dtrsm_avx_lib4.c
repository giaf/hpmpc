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



void kernel_dgemm_dtrsm_nt_8x4_lib4(int kadd, int ksub, double *Ap0, int sdap, double *Bp, double *Am0, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg)
	{

	double *Ap1 = Ap0 + 4*sdap;
	double *Am1 = Am0 + 4*sdam;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	
	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31,
		c_40_51_62_73, c_41_50_63_72, c_43_52_61_70, c_42_53_60_71;
	
	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();
	c_40_51_62_73 = _mm256_setzero_pd();
	c_41_50_63_72 = _mm256_setzero_pd();
	c_43_52_61_70 = _mm256_setzero_pd();
	c_42_53_60_71 = _mm256_setzero_pd();

	k = 0;

	//printf("\n%d\n", kadd);

	if(kadd>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &Ap0[0] );
		a_4567 = _mm256_load_pd( &Ap1[0] );
		b_0123 = _mm256_load_pd( &Bp[0] );

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
			Ap0 += 16;
			Ap1 += 16;
			Bp  += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
			
			Ap0 += 8;
			Ap1 += 8;
			Bp  += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
	/*		b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch*/
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
	/*		a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch*/
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
	/*		a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch*/
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
//			Ap0 += 4; // keep it !!!
//			Ap1 += 4; // keep it !!!
//			Bp  += 4; // keep it !!!

			}
		}

	if(ksub>0)
		{

		//d_print_mat(4, 4, A0, 4);
		//d_print_mat(4, 4, A1, 4);

		// prefetch
		a_0123 = _mm256_load_pd( &Am0[0] );
		a_4567 = _mm256_load_pd( &Am1[0] );
		b_0123 = _mm256_load_pd( &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[4] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Am0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Am1[4] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[8] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Am0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Am1[8] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[12] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Am0[12] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Am1[12] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[16] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Am0[16] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Am1[16] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );
			
			Am0 += 16;
			Am1 += 16;
			Bm  += 16;

			}

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
	
	if(alg==0)
		{
		d_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		d_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		d_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
		d_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
		d_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
		d_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_00_10_20_30 = _mm256_load_pd( &C0[0+bs*0] );
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_02_12_22_32 = _mm256_load_pd( &C0[0+bs*2] );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
		c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+bs*1] );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_03_13_23_33 = _mm256_load_pd( &C0[0+bs*3] );
		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
		c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
		c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+bs*0] );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
		d_42_52_62_72 = _mm256_load_pd( &C1[0+bs*2] );
		d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
		c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
		c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+bs*1] );
		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
		d_43_53_63_73 = _mm256_load_pd( &C1[0+bs*3] );
		d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );
		}
		
	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	d_40_50_60_70 = _mm256_mul_pd( d_40_50_60_70, a_00 );
	_mm256_store_pd( &D0[0+bs*0], d_00_10_20_30 );
	_mm256_store_pd( &D1[0+bs*0], d_40_50_60_70 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	ab_tmp0 = _mm256_mul_pd( d_00_10_20_30, a_10 );
	ab_tmp1 = _mm256_mul_pd( d_40_50_60_70, a_10 );
	d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_tmp0 );
	d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, ab_tmp1 );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
	_mm256_store_pd( &D0[0+bs*1], d_01_11_21_31 );
	_mm256_store_pd( &D1[0+bs*1], d_41_51_61_71 );

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
	_mm256_store_pd( &D0[0+bs*2], d_02_12_22_32 );
	_mm256_store_pd( &D1[0+bs*2], d_42_52_62_72 );

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
	_mm256_store_pd( &D0[0+bs*3], d_03_13_23_33 );
	_mm256_store_pd( &D1[0+bs*3], d_43_53_63_73 );

	}



void kernel_dgemm_dtrsm_nt_8x4_vs_lib4(int km, int kn, int tri, int kadd, int ksub, double *Ap0, int sdap, double *Bp, double *Am0, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg)
	{

	double *Ap1 = Ap0 + 4*sdap;
	double *Am1 = Am0 + 4*sdam;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	
	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	double d_temp;
	
	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31,
		c_40_51_62_73, c_41_50_63_72, c_43_52_61_70, c_42_53_60_71;
	
	__m256i 
		mask_m;

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();
	c_40_51_62_73 = _mm256_setzero_pd();
	c_41_50_63_72 = _mm256_setzero_pd();
	c_43_52_61_70 = _mm256_setzero_pd();
	c_42_53_60_71 = _mm256_setzero_pd();

	k = 0;

	//printf("\n%d\n", kadd);

	if(kadd>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &Ap0[0] );
		a_4567 = _mm256_load_pd( &Ap1[0] );
		b_0123 = _mm256_load_pd( &Bp[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				ab_tmp1       = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
				
				// k = 1
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x3 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				// k = 2
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x7 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				// k = 3
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
				a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
				
				Ap0 += 16;
				Ap1 += 16;
				Bp  += 16;
				k  += 4;

				if(kadd>=8)
					{

					// k = 4
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x1 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
					b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
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
					a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
					a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
					c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
					
					// k = 5
					ab_tmp1       = _mm256_setzero_pd();
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
					b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
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
					a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
					a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
					c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

					// k = 6
					ab_tmp1       = _mm256_setzero_pd();
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x7 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
					b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
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
					a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
					a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
					c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

					// k = 7
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
					b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
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
					a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
					a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
					c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
					
					Ap0 += 16;
					Ap1 += 16;
					Bp  += 16;
					k  += 4;

					}
				else
					{

					if(kadd>4)
						{

						// k = 4
						a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x1 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
						b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
						ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
						b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
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
						a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
						ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
						a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
						c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
						c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

						k  += 1;

						if(kadd>5)
							{
							
							// k = 5
							ab_tmp1       = _mm256_setzero_pd();
							a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x3 );
							ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
							b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
							ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
							b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
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
							a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
							ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
							a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
							c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
							c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

							k  += 1;

							if(kadd>6)
								{	

								// k = 6
								ab_tmp1       = _mm256_setzero_pd();
								a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x7 );
								ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
								b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
								ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
								b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
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
								a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
								ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
								a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
								c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
								c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

								k  += 1;

								}

							}

						}

					}

				}
			else // kadd = {1 2 3}
				{

				ab_tmp1       = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				k  += 1;

				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
					c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
					b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
					c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
					b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
					c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
					a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

					k  += 1;

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x7 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
						b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
						b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
						c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
						b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
						c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
						b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
						c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
						a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
						c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

						k  += 1;

						}

					}

				}

			}

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
			Ap0 += 16;
			Ap1 += 16;
			Bp  += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
			
			Ap0 += 8;
			Ap1 += 8;
			Bp  += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
	/*		b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch*/
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
	/*		a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch*/
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
	/*		a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch*/
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
			
//			Ap0 += 4; // keep it !!!
//			Ap1 += 4; // keep it !!!
//			Bp  += 4; // keep it !!!

			}
		}

	if(ksub>0)
		{

		//d_print_mat(4, 4, A0, 4);
		//d_print_mat(4, 4, A1, 4);

		// prefetch
		a_0123 = _mm256_load_pd( &Am0[0] );
		a_4567 = _mm256_load_pd( &Am1[0] );
		b_0123 = _mm256_load_pd( &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[4] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Am0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Am1[4] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[8] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Am0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Am1[8] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[12] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Am0[12] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Am1[12] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[16] ); // prefetch
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
			a_0123        = _mm256_load_pd( &Am0[16] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
			a_4567        = _mm256_load_pd( &Am1[16] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			c_42_53_60_71 = _mm256_sub_pd( c_42_53_60_71, ab_tmp1 );
			
			Am0 += 16;
			Am1 += 16;
			Bm  += 16;

			}

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
	
	if(alg==0)
		{
		d_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		d_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		d_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
		d_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
		d_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
		d_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_00_10_20_30 = _mm256_load_pd( &C0[0+bs*0] );
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_02_12_22_32 = _mm256_load_pd( &C0[0+bs*2] );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
		c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+bs*1] );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_03_13_23_33 = _mm256_load_pd( &C0[0+bs*3] );
		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
		c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
		c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+bs*0] );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
		d_42_52_62_72 = _mm256_load_pd( &C1[0+bs*2] );
		d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
		c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
		c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+bs*1] );
		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
		d_43_53_63_73 = _mm256_load_pd( &C1[0+bs*3] );
		d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );
		}
		
	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	d_temp = km - 4.0;
	mask_m = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &d_temp ) ) );

	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	d_40_50_60_70 = _mm256_mul_pd( d_40_50_60_70, a_00 );
	_mm256_store_pd( &D0[0+bs*0], d_00_10_20_30 );
	_mm256_maskstore_pd( &D1[0+bs*0], mask_m, d_40_50_60_70 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	ab_tmp0 = _mm256_mul_pd( d_00_10_20_30, a_10 );
	ab_tmp1 = _mm256_mul_pd( d_40_50_60_70, a_10 );
	d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_tmp0 );
	d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, ab_tmp1 );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
	_mm256_store_pd( &D0[0+bs*1], d_01_11_21_31 );
	_mm256_maskstore_pd( &D1[0+bs*1], mask_m, d_41_51_61_71 );

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
	_mm256_store_pd( &D0[0+bs*2], d_02_12_22_32 );
	_mm256_maskstore_pd( &D1[0+bs*2], mask_m, d_42_52_62_72 );

	if(kn>=4)
		{
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
		_mm256_store_pd( &D0[0+bs*3], d_03_13_23_33 );
		_mm256_maskstore_pd( &D1[0+bs*3], mask_m, d_43_53_63_73 );
		}

	}



void kernel_dgemm_dtrsm_nt_6x4_vs_lib4(int km, int kn, int tri, int kadd, int ksub, double *Ap0, int sdap, double *Bp, double *Am0, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg)
	{

	double *Ap1 = Ap0 + 4*sdap;
	double *Am1 = Am0 + 4*sdam;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	
	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	double d_temp;
	
	int k;
	
	__m256d
		a_0123, a_4545, A_0123, A_4545,
		b_0123, b_1032, b_3210, b_2301,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31,
		c_40_51_42_53, c_41_50_43_52;
	
	__m256i 
		mask_m;

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();
	c_40_51_42_53 = _mm256_setzero_pd();
	c_41_50_43_52 = _mm256_setzero_pd();

	k = 0;

	//printf("\n%d\n", kadd);

	if(kadd>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &Ap0[0] );
		a_4545 = _mm256_broadcast_pd( (__m128d *) &Ap1[0] );
		b_0123 = _mm256_load_pd( &Bp[0] );

#if 0 // TODO
		if(tri==1)
			{

			if(kadd>=4)
				{

				ab_tmp1       = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
				
				// k = 1
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x3 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				// k = 2
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x7 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				// k = 3
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
				a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
				
				Ap0 += 16;
				Ap1 += 16;
				Bp  += 16;
				k  += 4;

				if(kadd>=8)
					{

					// k = 4
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x1 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
					b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
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
					a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
					a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
					c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
					
					// k = 5
					ab_tmp1       = _mm256_setzero_pd();
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
					b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
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
					a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
					a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
					c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

					// k = 6
					ab_tmp1       = _mm256_setzero_pd();
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x7 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
					b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
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
					a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
					a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
					c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

					// k = 7
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
					b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
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
					a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
					a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
					c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
					
					Ap0 += 16;
					Ap1 += 16;
					Bp  += 16;
					k  += 4;

					}
				else
					{

					if(kadd>4)
						{

						// k = 4
						a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x1 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
						b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
						ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
						b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
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
						a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
						ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
						a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
						c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
						c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

						k  += 1;

						if(kadd>5)
							{
							
							// k = 5
							ab_tmp1       = _mm256_setzero_pd();
							a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x3 );
							ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
							b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
							ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
							b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
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
							a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
							ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
							a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
							c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
							c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

							k  += 1;

							if(kadd>6)
								{	

								// k = 6
								ab_tmp1       = _mm256_setzero_pd();
								a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x7 );
								ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
								b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
								ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
								b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
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
								a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
								ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
								a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
								c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
								c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );

								k  += 1;

								}

							}

						}

					}

				}
			else // kadd = {1 2 3}
				{

				ab_tmp1       = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				k  += 1;

				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
					c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
					b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
					c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
					b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
					c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
					a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

					k  += 1;

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x7 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
						b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
						b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
						c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
						b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
						c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
						b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
						c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
						a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
						c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

						k  += 1;

						}

					}

				}

			}
#endif

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			A_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			c_40_51_42_53 = _mm256_add_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			A_4545        = _mm256_broadcast_pd( (__m128d *) &Ap1[4] ); // prefetch
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_1032 );
			c_41_50_43_52 = _mm256_add_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( A_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			c_40_51_42_53 = _mm256_add_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			a_4545        = _mm256_broadcast_pd( (__m128d *) &Ap1[8] ); // prefetch
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_1032 );
			c_41_50_43_52 = _mm256_add_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			A_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
			c_40_51_42_53 = _mm256_add_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			A_4545        = _mm256_broadcast_pd( (__m128d *) &Ap1[12] ); // prefetch
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_1032 );
			c_41_50_43_52 = _mm256_add_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( A_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
			c_40_51_42_53 = _mm256_add_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			a_4545        = _mm256_broadcast_pd( (__m128d *) &Ap1[16] ); // prefetch
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_1032 );
			c_41_50_43_52 = _mm256_add_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		
			
			Ap0 += 16;
			Ap1 += 16;
			Bp  += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			A_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			c_40_51_42_53 = _mm256_add_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			A_4545        = _mm256_broadcast_pd( (__m128d *) &Ap1[4] ); // prefetch
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_1032 );
			c_41_50_43_52 = _mm256_add_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			
			
			ab_tmp0       = _mm256_mul_pd( A_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			c_40_51_42_53 = _mm256_add_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			a_4545        = _mm256_broadcast_pd( (__m128d *) &Ap1[8] ); // prefetch
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_1032 );
			c_41_50_43_52 = _mm256_add_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			
			
			Ap0 += 8;
			Ap1 += 8;
			Bp  += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			//A_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_0123 );
			//b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			c_40_51_42_53 = _mm256_add_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			//A_4545        = _mm256_broadcast_pd( (__m128d *) &Ap1[4] ); // prefetch
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_1032 );
			c_41_50_43_52 = _mm256_add_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
			
//			Ap0 += 4; // keep it !!!
//			Ap1 += 4; // keep it !!!
//			Bp  += 4; // keep it !!!

			}
		}

	if(ksub>0)
		{

		//d_print_mat(4, 4, A0, 4);
		//d_print_mat(4, 4, A1, 4);

		// prefetch
		a_0123 = _mm256_load_pd( &Am0[0] );
		a_4545 = _mm256_broadcast_pd( (__m128d *) &Am1[0] );
		b_0123 = _mm256_load_pd( &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			A_0123        = _mm256_load_pd( &Am0[4] ); // prefetch
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[4] ); // prefetch
			c_40_51_42_53 = _mm256_sub_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			A_4545        = _mm256_broadcast_pd( (__m128d *) &Am1[4] ); // prefetch
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_1032 );
			c_41_50_43_52 = _mm256_sub_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( A_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			a_0123        = _mm256_load_pd( &Am0[8] ); // prefetch
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[8] ); // prefetch
			c_40_51_42_53 = _mm256_sub_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			a_4545        = _mm256_broadcast_pd( (__m128d *) &Am1[8] ); // prefetch
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_1032 );
			c_41_50_43_52 = _mm256_sub_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			A_0123        = _mm256_load_pd( &Am0[12] ); // prefetch
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[12] ); // prefetch
			c_40_51_42_53 = _mm256_sub_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			A_4545        = _mm256_broadcast_pd( (__m128d *) &Am1[12] ); // prefetch
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_4545, b_1032 );
			c_41_50_43_52 = _mm256_sub_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( A_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			a_0123        = _mm256_load_pd( &Am0[16] ); // prefetch
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_0123 );
			b_0123        = _mm256_load_pd( &Bm[16] ); // prefetch
			c_40_51_42_53 = _mm256_sub_pd( c_40_51_42_53, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			a_4545        = _mm256_broadcast_pd( (__m128d *) &Am1[16] ); // prefetch
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_4545, b_1032 );
			c_41_50_43_52 = _mm256_sub_pd( c_41_50_43_52, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_3210 );
			b_1032        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_tmp0 );
			ab_tmp0       = _mm256_mul_pd( A_0123, b_1032 );
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_tmp0 );
			

			Am0 += 16;
			Am1 += 16;
			Bm  += 16;

			}

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
	c_40_50_60_70 = _mm256_blend_pd( c_40_51_42_53, c_41_50_43_52, 0xa );
	c_41_51_61_71 = _mm256_blend_pd( c_40_51_42_53, c_41_50_43_52, 0x5 );
	
	if(alg==0)
		{
		d_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		d_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		d_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_42_52_62_72 = _mm256_castpd128_pd256( _mm256_extractf128_pd( c_40_50_60_70, 0x1 ) );
		d_40_50_60_70 = c_40_50_60_70;
		d_43_53_63_73 = _mm256_castpd128_pd256( _mm256_extractf128_pd( c_41_51_61_71, 0x1 ) );
		d_42_52_62_72 = c_42_52_62_72;
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_00_10_20_30 = _mm256_load_pd( &C0[0+bs*0] );
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_02_12_22_32 = _mm256_load_pd( &C0[0+bs*2] );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
		c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+bs*1] );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_03_13_23_33 = _mm256_load_pd( &C0[0+bs*3] );
		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
		c_42_52_62_72 = _mm256_castpd128_pd256( _mm256_extractf128_pd( c_40_50_60_70, 0x1 ) );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+bs*0] );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
		d_42_52_62_72 = _mm256_load_pd( &C1[0+bs*2] );
		d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
		c_43_53_63_73 = _mm256_castpd128_pd256( _mm256_extractf128_pd( c_41_51_61_71, 0x1 ) );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+bs*1] );
		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
		d_43_53_63_73 = _mm256_load_pd( &C1[0+bs*3] );
		d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );
		}
		
	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	d_temp = km - 4.0;
	mask_m = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &d_temp ) ) );

	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	d_40_50_60_70 = _mm256_mul_pd( d_40_50_60_70, a_00 );
	_mm256_store_pd( &D0[0+bs*0], d_00_10_20_30 );
	_mm256_maskstore_pd( &D1[0+bs*0], mask_m, d_40_50_60_70 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	ab_tmp0 = _mm256_mul_pd( d_00_10_20_30, a_10 );
	ab_tmp1 = _mm256_mul_pd( d_40_50_60_70, a_10 );
	d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_tmp0 );
	d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, ab_tmp1 );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
	_mm256_store_pd( &D0[0+bs*1], d_01_11_21_31 );
	_mm256_maskstore_pd( &D1[0+bs*1], mask_m, d_41_51_61_71 );

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
	_mm256_store_pd( &D0[0+bs*2], d_02_12_22_32 );
	_mm256_maskstore_pd( &D1[0+bs*2], mask_m, d_42_52_62_72 );

	if(kn>=4)
		{
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
		_mm256_store_pd( &D0[0+bs*3], d_03_13_23_33 );
		_mm256_maskstore_pd( &D1[0+bs*3], mask_m, d_43_53_63_73 );
		}

	}



void kernel_dgemm_dtrsm_nt_8x2_vs_lib4(int km, int kn, int tri, int kadd, int ksub, double *Ap0, int sdap, double *Bp, double *Am0, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg)
	{
	
	double *Ap1 = Ap0 + 4*sdap;
	double *Am1 = Am0 + 4*sdam;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	
	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	double d_temp;
	
	int k;
	
	__m256d
		zeros,
		a_0123, a_4567, //A_0123,
		b_0101, b_1010,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_20_31, c_01_10_21_30,
		c_40_51_60_71, c_41_50_61_70;
	
	__m256i
		mask_m;
	
	// zero registers
	zeros = _mm256_setzero_pd();
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	c_40_51_60_71 = _mm256_setzero_pd();
	c_41_50_61_70 = _mm256_setzero_pd();

	k = 0;

	if(kadd>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &Ap0[0] );
		a_4567 = _mm256_load_pd( &Ap1[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &Bp[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
				
				// k = 1
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

				// k = 2
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

				// k = 2
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
				a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
				
				Ap0 += 16;
				Ap1 += 16;
				Bp  += 16;
				k  += 4;

				if(kadd>=8)
					{

					// k = 4
					a_4567        = _mm256_blend_pd( zeros, a_4567, 0x1 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
					b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
					b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
					c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
					c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
					a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
					a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
					c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
					c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
					
					// k = 5
					a_4567        = _mm256_blend_pd( zeros, a_4567, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
					b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
					b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
					c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
					c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
					a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
					a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
					c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
					c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );

					// k = 6
					a_4567        = _mm256_blend_pd( zeros, a_4567, 0x7 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
					b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
					b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
					c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
					c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
					a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
					a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
					c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
					c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );

					// k = 7
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
					b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
					ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
					b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
					c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
					c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
					a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
					ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
					a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
					c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
					c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
					
					Ap0 += 16;
					Ap1 += 16;
					Bp  += 16;
					k  += 4;

					}
				else
					{

					if(kadd>4)
						{

						// k = 4
						a_4567        = _mm256_blend_pd( zeros, a_4567, 0x1 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
						b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
						ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
						b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
						c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
						c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
						a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
						ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
						a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
						c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
						c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );

						k  += 1;

						if(kadd>5)
							{
						
							// k = 5
							a_4567        = _mm256_blend_pd( zeros, a_4567, 0x3 );
							ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
							b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
							ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
							b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
							c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
							c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
							ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
							a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
							ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
							a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
							c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
							c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );

							k  += 1;

							if(kadd>6)
								{

								// k = 6
								a_4567        = _mm256_blend_pd( zeros, a_4567, 0x7 );
								ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
								b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
								ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
								b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
								c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
								c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
								ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
								a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
								ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
								a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
								c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
								c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );

								k  += 1;

								}

							}

						}

					}

				}
			else
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

				k  += 1;

				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
					b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
					b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
					c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
					a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
					c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

					k  += 1;

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
						b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
						b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
						c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
						a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
						c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

						k  += 1;

						}

					}

				}

			}

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
			
			Ap0 += 16;
			Ap1 += 16;
			Bp  += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
			
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
			
			
			Ap0 += 8;
			Ap1 += 8;
			Bp  += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
	/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch*/
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
	/*		a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch*/
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
	/*		a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch*/
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
			
//			Ap0 += 4; // keep it !!!
//			Ap1 += 4; // keep it !!!
//			Bp  += 4; // keep it !!!

			}

		}
		
	if(ksub>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &Am0[0] );
		a_4567 = _mm256_load_pd( &Am1[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[4] ); // prefetch
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_sub_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Am0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Am1[4] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_sub_pd( c_41_50_61_70, ab_tmp1 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[8] ); // prefetch
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_sub_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Am0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Am1[8] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_sub_pd( c_41_50_61_70, ab_tmp1 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[12] ); // prefetch
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_sub_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Am0[12] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Am1[12] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_sub_pd( c_41_50_61_70, ab_tmp1 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[16] ); // prefetch
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_sub_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Am0[16] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &Am1[16] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_sub_pd( c_41_50_61_70, ab_tmp1 );
			
			Am0 += 16;
			Am1 += 16;
			Bm  += 16;

			}

		}

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		c_40_50_60_70, c_41_51_61_71,
		d_00_10_20_30, d_01_11_21_31,
		d_40_50_60_70, d_41_51_61_71;

	if(alg==0)
		{
		d_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
		d_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
		d_40_50_60_70 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0xa );
		d_41_51_61_71 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0x5 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
		d_00_10_20_30 = _mm256_load_pd( &C0[0+bs*0] );
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+bs*1] );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		c_40_50_60_70 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0xa );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+bs*0] );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
		c_41_51_61_71 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0x5 );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+bs*1] );
		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
		}
		
	__m256d
		a_00, a_10, a_11;
	
	d_temp = km - 4.0;
	mask_m = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &d_temp ) ) );

	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	d_40_50_60_70 = _mm256_mul_pd( d_40_50_60_70, a_00 );
	_mm256_store_pd( &D0[0+bs*0], d_00_10_20_30 );
	_mm256_maskstore_pd( &D1[0+bs*0], mask_m, d_40_50_60_70 );

	if(kn>=2)
		{
		a_10 = _mm256_broadcast_sd( &fact[1] );
		a_11 = _mm256_broadcast_sd( &fact[2] );
		ab_tmp0 = _mm256_mul_pd( d_00_10_20_30, a_10 );
		ab_tmp1 = _mm256_mul_pd( d_40_50_60_70, a_10 );
		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_tmp0 );
		d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, ab_tmp1 );
		d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
		d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
		_mm256_store_pd( &D0[0+bs*1], d_01_11_21_31 );
		_mm256_maskstore_pd( &D1[0+bs*1], mask_m, d_41_51_61_71 );
		}

	}



void kernel_dgemm_dtrsm_nt_4x4_vs_lib4(int km, int kn, int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;

	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	double d_temp;
	
	int k;
	
	__m256d
		zeros,
		a_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
	__m256i
		mask_m;
	
	// zero registers
	zeros = _mm256_setzero_pd();
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();

	k = 0;

	if(kadd>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &Ap[0] );
		b_0123 = _mm256_load_pd( &Bp[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
				
				// k = 1
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );

				// k = 2
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );

				// k = 3
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap[16] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
				
				Ap += 16;
				Bp += 16;
				k += 4;


				}
			else
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
				ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );

				k += 1;

				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
					ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
					c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
					ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
					b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
					c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
					ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
					b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
					c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
					ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
					a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );

					k += 1;

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
						ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
						b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
						b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
						c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
						ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
						b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
						c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
						ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
						b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
						c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
						ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
						a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
						c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );

						k += 1;

						}

					}

				}

			}

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[16] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
			Ap += 16;
			Bp += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
			
			Ap += 8;
			Bp += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
	/*		b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch*/
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
	/*		a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch*/
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
//			Ap += 4; // keep it !!!
//			Bp += 4; // keep it !!!

			}

		}

	if(ksub>0)
		{
		
		// prefetch
		a_0123 = _mm256_load_pd( &Am[0] );
		b_0123 = _mm256_load_pd( &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[4] ); // prefetch
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Am[4] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[8] ); // prefetch
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Am[8] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[12] ); // prefetch
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Am[12] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[16] ); // prefetch
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Am[16] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
			
			Am += 16;
			Bm += 16;

			}

		}

	__m256d
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33;

	c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
	c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
	c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
	c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
	
	if(alg==0)
		{
		d_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		d_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		d_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_00_10_20_30 = _mm256_load_pd( &C[0+bs*0] );
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_02_12_22_32 = _mm256_load_pd( &C[0+bs*2] );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
		c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_01_11_21_31 = _mm256_load_pd( &C[0+bs*1] );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_03_13_23_33 = _mm256_load_pd( &C[0+bs*3] );
		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
		}

	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	d_temp = km - 0.0;
	mask_m = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &d_temp ) ) );

	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	_mm256_maskstore_pd( &D[0+bs*0], mask_m, d_00_10_20_30 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	ab_temp = _mm256_mul_pd( d_00_10_20_30, a_10 );
	d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_temp );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	_mm256_maskstore_pd( &D[0+bs*1], mask_m, d_01_11_21_31 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	ab_temp = _mm256_mul_pd( d_00_10_20_30, a_20 );
	d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, ab_temp );
	ab_temp = _mm256_mul_pd( d_01_11_21_31, a_21 );
	d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, ab_temp );
	d_02_12_22_32 = _mm256_mul_pd( d_02_12_22_32, a_22 );
	_mm256_maskstore_pd( &D[0+bs*2], mask_m, d_02_12_22_32 );

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
	_mm256_maskstore_pd( &D[0+bs*3], mask_m, d_03_13_23_33 );

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_dgemm_dtrsm_nt_4x2_vs_lib4(int km, int kn, int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;

	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	double d_temp;
	
	int k;
	
	__m256d
		zeros,
		a_0123,
		b_0101, b_1010,
		ab_temp, // temporary results
		c_00_11_20_31, c_01_10_21_30, C_00_11_20_31, C_01_10_21_30;
	
	__m256i
		mask_m;

	// zero registers
	zeros = _mm256_setzero_pd();
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	C_00_11_20_31 = _mm256_setzero_pd();
	C_01_10_21_30 = _mm256_setzero_pd();

	k = 0;

	if(kadd>0)
		{
	
		// prefetch
		a_0123 = _mm256_load_pd( &Ap[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &Bp[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
				
				// k = 1
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
				C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );

				// k = 2
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );

				// k = 3
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap[16] ); // prefetch
				C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
				
				Ap += 16;
				Bp += 16;
				k += 4;

				}
			else
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );

				k += 1;
				
				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
					ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
					C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
					b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
					b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
					ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
					a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
					C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );

					k += 1;

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
						ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
						c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
						b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
						b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
						ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
						a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
						c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );

						k += 1;

						}

					}

				}

			}

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
			C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap[16] ); // prefetch
			C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
			
			Ap += 16;
			Bp += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
			
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
			C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
			
			
			Ap += 8;
			Bp += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
	/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch*/
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
	/*		a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch*/
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
			
//			Ap += 4; // keep it !!!
//			Bp += 4; // keep it !!!

			}

		}

	if(ksub>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &Am[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Am[4] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			C_00_11_20_31 = _mm256_sub_pd( C_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Am[8] ); // prefetch
			C_01_10_21_30 = _mm256_sub_pd( C_01_10_21_30, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[12] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Am[12] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			C_00_11_20_31 = _mm256_sub_pd( C_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[16] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &Am[16] ); // prefetch
			C_01_10_21_30 = _mm256_sub_pd( C_01_10_21_30, ab_temp );
			
			Am += 16;
			Bm += 16;

			}

		}

	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, C_00_11_20_31 );
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, C_01_10_21_30 );

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		d_00_10_20_30, d_01_11_21_31;

	if(alg==0)
		{
		d_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
		d_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
		d_00_10_20_30 = _mm256_load_pd( &C[0+bs*0] );
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
		d_01_11_21_31 = _mm256_load_pd( &C[0+bs*1] );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		}

	__m256d
		a_00, a_10, a_11;
	
	d_temp = km - 4.0;
	mask_m = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &d_temp ) ) );

	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	_mm256_maskstore_pd( &D[0+bs*0], mask_m, d_00_10_20_30 );

	if(kn>=2)
		{
		a_10 = _mm256_broadcast_sd( &fact[1] );
		a_11 = _mm256_broadcast_sd( &fact[2] );
		ab_temp = _mm256_mul_pd( d_00_10_20_30, a_10 );
		d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_temp );
		d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
		_mm256_maskstore_pd( &D[0+bs*1], mask_m, d_01_11_21_31 );
		}

	}



// normal-transposed, 2x4 with data packed in 4
void kernel_dgemm_dtrsm_nt_2x4_vs_lib4(int km, int kn, int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;

	int k;
	
	__m256d
		zeros,
		a_0101,
		b_0123, b_1032,
		ab_temp, // temporary results
		c_00_11_02_13, c_01_10_03_12, C_00_11_02_13, C_01_10_03_12;

	// zero registers
	zeros = _mm256_setzero_pd();
	c_00_11_02_13 = _mm256_setzero_pd();
	c_01_10_03_12 = _mm256_setzero_pd();
	C_00_11_02_13 = _mm256_setzero_pd();
	C_01_10_03_12 = _mm256_setzero_pd();

	k = 0;

	if(kadd>0)
		{
	
		// prefetch
		a_0101 = _mm256_broadcast_pd( (__m128d *) &Ap[0] );
		b_0123 = _mm256_load_pd( &Bp[0] );

		if(tri==1)
			{

			if(kadd>=2)
				{

				// k = 0
				a_0101        = _mm256_blend_pd( zeros, a_0101, 0x5 );
				ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
				c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
				a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[4] ); // prefetch
				c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
				
				// k = 1
				ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
				C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
				a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[8] ); // prefetch
				C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, ab_temp );

				Ap += 8;
				Bp += 8;
				k += 2;

				}
			else
				{

				// k = 0
				a_0101        = _mm256_blend_pd( zeros, a_0101, 0x5 );
				ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
				c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
				a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[4] ); // prefetch
				c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
	
				k += 1;

				}
			
			}

		for(k=0; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[4] ); // prefetch
			c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[8] ); // prefetch
			C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[12] ); // prefetch
			c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[16] ); // prefetch
			C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, ab_temp );
			
			Ap += 16;
			Bp += 16;

			}
		
		if(kadd%4>=2)
			{
			
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[4] ); // prefetch
			c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
			
			
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[8] ); // prefetch
			C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, ab_temp );
			
			Ap += 8;
			Bp += 8;

			}

		if(kadd%2==1)
			{
			
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
	/*		b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch*/
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
	/*		a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[4] ); // prefetch*/
			c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
			
//			Ap += 4; // keep it !!!
//			Bp += 4; // keep it !!!

			}

		}

	if(ksub>0)
		{

		// prefetch
		a_0101 = _mm256_broadcast_pd( (__m128d *) &Am[0] );
		b_0123 = _mm256_load_pd( &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_sub_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Am[4] ); // prefetch
			c_01_10_03_12 = _mm256_sub_pd( c_01_10_03_12, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			C_00_11_02_13 = _mm256_sub_pd( C_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Am[8] ); // prefetch
			C_01_10_03_12 = _mm256_sub_pd( C_01_10_03_12, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_sub_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[12] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Am[12] ); // prefetch
			c_01_10_03_12 = _mm256_sub_pd( c_01_10_03_12, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			C_00_11_02_13 = _mm256_sub_pd( C_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[16] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Am[16] ); // prefetch
			C_01_10_03_12 = _mm256_sub_pd( C_01_10_03_12, ab_temp );
			
			Am += 16;
			Bm += 16;

			}

		}

	c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, C_00_11_02_13 );
	c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, C_01_10_03_12 );

	__m256d
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
	
	if(alg==0)
		{
		d_00_10 = c_00_10;
		d_02_12 = c_02_12;
		d_01_11 = c_01_11;
		d_03_13 = c_03_13;
		}
	else
		{
		d_00_10 = _mm_load_pd( &C[0+bs*0] );
		d_00_10 = _mm_add_pd( d_00_10, c_00_10 );
		d_02_12 = _mm_load_pd( &C[0+bs*2] );
		d_02_12 = _mm_add_pd( d_02_12, c_02_12 );
		d_01_11 = _mm_load_pd( &C[0+bs*1] );
		d_01_11 = _mm_add_pd( d_01_11, c_01_11 );
		d_03_13 = _mm_load_pd( &C[0+bs*3] );
		d_03_13 = _mm_add_pd( d_03_13, c_03_13 );
		}

	__m128d
		ab_tmp0,
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	if(km>=2)
		{
		a_00 = _mm_loaddup_pd( &fact[0] );
		d_00_10 = _mm_mul_pd( d_00_10, a_00 );
		_mm_store_pd( &D[0+bs*0], d_00_10 );

		a_10 = _mm_loaddup_pd( &fact[1] );
		a_11 = _mm_loaddup_pd( &fact[2] );
		ab_tmp0 = _mm_mul_pd( d_00_10, a_10 );
		d_01_11 = _mm_sub_pd( d_01_11, ab_tmp0 );
		d_01_11 = _mm_mul_pd( d_01_11, a_11 );
		_mm_store_pd( &D[0+bs*1], d_01_11 );

		a_20 = _mm_loaddup_pd( &fact[3] );
		a_21 = _mm_loaddup_pd( &fact[4] );
		a_22 = _mm_loaddup_pd( &fact[5] );
		ab_tmp0 = _mm_mul_pd( d_00_10, a_20 );
		d_02_12 = _mm_sub_pd( d_02_12, ab_tmp0 );
		ab_tmp0 = _mm_mul_pd( d_01_11, a_21 );
		d_02_12 = _mm_sub_pd( d_02_12, ab_tmp0 );
		d_02_12 = _mm_mul_pd( d_02_12, a_22 );
		_mm_store_pd( &D[0+bs*2], d_02_12 );

		if(kn>=4)
			{
			a_30 = _mm_loaddup_pd( &fact[6] );
			a_31 = _mm_loaddup_pd( &fact[7] );
			a_32 = _mm_loaddup_pd( &fact[8] );
			a_33 = _mm_loaddup_pd( &fact[9] );
			ab_tmp0 = _mm_mul_pd( d_00_10, a_30 );
			d_03_13 = _mm_sub_pd( d_03_13, ab_tmp0 );
			ab_tmp0 = _mm_mul_pd( d_01_11, a_31 );
			d_03_13 = _mm_sub_pd( d_03_13, ab_tmp0 );
			ab_tmp0 = _mm_mul_pd( d_02_12, a_32 );
			d_03_13 = _mm_sub_pd( d_03_13, ab_tmp0 );
			d_03_13 = _mm_mul_pd( d_03_13, a_33 );
			_mm_store_pd( &D[0+bs*3], d_03_13 );
			}
		}
	else
		{
		a_00 = _mm_load_pd( &fact[0] );
		d_00_10 = _mm_mul_sd( d_00_10, a_00 );
		_mm_store_sd( &D[0+bs*0], d_00_10 );

		a_10 = _mm_load_pd( &fact[1] );
		a_11 = _mm_load_pd( &fact[2] );
		ab_tmp0 = _mm_mul_sd( d_00_10, a_10 );
		d_01_11 = _mm_sub_sd( d_01_11, ab_tmp0 );
		d_01_11 = _mm_mul_sd( d_01_11, a_11 );
		_mm_store_sd( &D[0+bs*1], d_01_11 );

		a_20 = _mm_load_pd( &fact[3] );
		a_21 = _mm_load_pd( &fact[4] );
		a_22 = _mm_load_pd( &fact[5] );
		ab_tmp0 = _mm_mul_sd( d_00_10, a_20 );
		d_02_12 = _mm_sub_sd( d_02_12, ab_tmp0 );
		ab_tmp0 = _mm_mul_sd( d_01_11, a_21 );
		d_02_12 = _mm_sub_sd( d_02_12, ab_tmp0 );
		d_02_12 = _mm_mul_sd( d_02_12, a_22 );
		_mm_store_sd( &D[0+bs*2], d_02_12 );

		if(kn>=4)
			{
			a_30 = _mm_load_pd( &fact[6] );
			a_31 = _mm_load_pd( &fact[7] );
			a_32 = _mm_load_pd( &fact[8] );
			a_33 = _mm_load_pd( &fact[9] );
			ab_tmp0 = _mm_mul_sd( d_00_10, a_30 );
			d_03_13 = _mm_sub_sd( d_03_13, ab_tmp0 );
			ab_tmp0 = _mm_mul_sd( d_01_11, a_31 );
			d_03_13 = _mm_sub_sd( d_03_13, ab_tmp0 );
			ab_tmp0 = _mm_mul_sd( d_02_12, a_32 );
			d_03_13 = _mm_sub_sd( d_03_13, ab_tmp0 );
			d_03_13 = _mm_mul_sd( d_03_13, a_33 );
			_mm_store_sd( &D[0+bs*3], d_03_13 );
			}
		}

	}



// normal-transposed, 2x2 with data packed in 4
void kernel_dgemm_dtrsm_nt_2x2_vs_lib4(int km, int kn, int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;

	int k;
	
	__m128d
		a_01,
		b_01, b_10,
		ab_temp, // temporary results
		c_00_11, c_01_10, C_00_11, C_01_10;
	
	// zero registers
	c_00_11 = _mm_setzero_pd();
	c_01_10 = _mm_setzero_pd();
	C_00_11 = _mm_setzero_pd();
	C_01_10 = _mm_setzero_pd();

	if(kadd>0)
		{

		// prefetch
		a_01 = _mm_load_pd( &Ap[0] );
		b_01 = _mm_load_pd( &Bp[0] );

		if(tri==1)
			{

			if(kadd>=2)
				{

				// k = 0
				ab_temp = _mm_mul_sd( a_01, b_01 );
				c_00_11 = _mm_add_sd( c_00_11, ab_temp );
				b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
				b_01    = _mm_load_pd( &Bp[4] ); // prefetch
				ab_temp = _mm_mul_pd( a_01, b_10 );
				a_01    = _mm_load_pd( &Ap[4] ); // prefetch
				c_01_10 = _mm_add_pd( c_01_10, ab_temp );
				
				// k = 1
				ab_temp = _mm_mul_pd( a_01, b_01 );
				C_00_11 = _mm_add_pd( C_00_11, ab_temp );
				b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
				b_01    = _mm_load_pd( &Bp[8] ); // prefetch
				ab_temp = _mm_mul_pd( a_01, b_10 );
				a_01    = _mm_load_pd( &Ap[8] ); // prefetch
				C_01_10 = _mm_add_pd( C_01_10, ab_temp );

				Ap += 8;
				Bp += 8;
				k += 2;

				}
			else
				{

				// k = 0
				ab_temp = _mm_mul_sd( a_01, b_01 );
				c_00_11 = _mm_add_sd( c_00_11, ab_temp );
				b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
				b_01    = _mm_load_pd( &Bp[4] ); // prefetch
				ab_temp = _mm_mul_pd( a_01, b_10 );
				a_01    = _mm_load_pd( &Ap[4] ); // prefetch
				c_01_10 = _mm_add_pd( c_01_10, ab_temp );
	
				k += 1;

				}

			}

		for(k=0; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_add_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[4] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Ap[4] ); // prefetch
			c_01_10 = _mm_add_pd( c_01_10, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			C_00_11 = _mm_add_pd( C_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[8] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Ap[8] ); // prefetch
			C_01_10 = _mm_add_pd( C_01_10, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_add_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[12] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Ap[12] ); // prefetch
			c_01_10 = _mm_add_pd( c_01_10, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			C_00_11 = _mm_add_pd( C_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[16] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Ap[16] ); // prefetch
			C_01_10 = _mm_add_pd( C_01_10, ab_temp );
			
			Ap += 16;
			Bp += 16;

			}
		
		if(kadd%4>=2)
			{
			
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_add_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[4] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Ap[4] ); // prefetch
			c_01_10 = _mm_add_pd( c_01_10, ab_temp );
			
			
			ab_temp = _mm_mul_pd( a_01, b_01 );
			C_00_11 = _mm_add_pd( C_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[8] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Ap[8] ); // prefetch
			C_01_10 = _mm_add_pd( C_01_10, ab_temp );
			
			Ap += 8;
			Bp += 8;

			}

		if(kadd%2==1)
			{
			
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_add_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
	/*		b_01    = _mm_load_pd( &Bp[4] ); // prefetch*/
			ab_temp = _mm_mul_pd( a_01, b_10 );
	/*		a_01    = _mm_load_pd( &Ap[4] ); // prefetch*/
			c_01_10 = _mm_add_pd( c_01_10, ab_temp );
			
//			Ap += 4; // keep it !!!
//			Bp += 4; // keep it !!!

			}

		}

	if(ksub>0)
		{

		// prefetch
		a_01 = _mm_load_pd( &Am[0] );
		b_01 = _mm_load_pd( &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_sub_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bm[4] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Am[4] ); // prefetch
			c_01_10 = _mm_sub_pd( c_01_10, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			C_00_11 = _mm_sub_pd( C_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bm[8] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Am[8] ); // prefetch
			C_01_10 = _mm_sub_pd( C_01_10, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_sub_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bm[12] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Am[12] ); // prefetch
			c_01_10 = _mm_sub_pd( c_01_10, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			C_00_11 = _mm_sub_pd( C_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bm[16] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &Am[16] ); // prefetch
			C_01_10 = _mm_sub_pd( C_01_10, ab_temp );
			
			Am += 16;
			Bm += 16;

			}

		}

	c_00_11 = _mm_add_pd( c_00_11, C_00_11 );
	c_01_10 = _mm_add_pd( c_01_10, C_01_10 );

	__m128d
		c_00_10, c_01_11,
		d_00_10, d_01_11;

	if(alg==0)
		{
		d_00_10 = _mm_blend_pd( c_00_11, c_01_10, 0x2 );
		d_01_11 = _mm_blend_pd( c_00_11, c_01_10, 0x1 );
		}
	else
		{
		c_00_10 = _mm_blend_pd( c_00_11, c_01_10, 0x2 );
		d_00_10 = _mm_load_pd( &C[0+bs*0] );
		d_00_10 = _mm_add_pd( d_00_10, c_00_10 );
		c_01_11 = _mm_blend_pd( c_00_11, c_01_10, 0x1 );
		d_01_11 = _mm_load_pd( &C[0+bs*1] );
		d_01_11 = _mm_add_pd( d_01_11, c_01_11 );
		}

	__m128d
		a_00, a_10, a_11;
	
	if(km>=2)
		{
		a_00 = _mm_loaddup_pd( &fact[0] );
		d_00_10 = _mm_mul_pd( d_00_10, a_00 );
		_mm_store_pd( &D[0+bs*0], d_00_10 );

		if(kn>=2)
			{
			a_10 = _mm_loaddup_pd( &fact[1] );
			a_11 = _mm_loaddup_pd( &fact[2] );
			ab_temp = _mm_mul_pd( d_00_10, a_10 );
			d_01_11 = _mm_sub_pd( d_01_11, ab_temp );
			d_01_11 = _mm_mul_pd( d_01_11, a_11 );
			_mm_store_pd( &D[0+bs*1], d_01_11 );
			}
		}
	else
		{
		a_00 = _mm_load_pd( &fact[0] );
		d_00_10 = _mm_mul_sd( d_00_10, a_00 );
		_mm_store_sd( &D[0+bs*0], d_00_10 );

		if(kn>=2)
			{
			a_10 = _mm_load_pd( &fact[1] );
			a_11 = _mm_load_pd( &fact[2] );
			ab_temp = _mm_mul_sd( d_00_10, a_10 );
			d_01_11 = _mm_sub_sd( d_01_11, ab_temp );
			d_01_11 = _mm_mul_sd( d_01_11, a_11 );
			_mm_store_sd( &D[0+bs*1], d_01_11 );
			}
		}

	}



