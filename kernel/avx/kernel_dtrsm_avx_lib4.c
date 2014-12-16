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

#include "../../include/block_size.h"



// normal-transposed, 8x4 with data packed in 4
void kernel_dgemm_dtrsm_nt_8x4_lib4(int tri, int kadd, int ksub, double *A0, double *A1, double *B, double *C0, double *C1, double *D0, double *D1, double *fact, int alg)
	{

	const int bs = 4;
	const int d_ncl = D_NCL;
	
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
		a_0123 = _mm256_load_pd( &A0[0] );
		a_4567 = _mm256_load_pd( &A1[0] );
		b_0123 = _mm256_load_pd( &B[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				ab_tmp1       = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &B[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
				
				// k = 1
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x3 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &B[8] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				// k = 2
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x7 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &B[12] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				// k = 3
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &B[16] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
				a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
				
				A0 += 16;
				A1 += 16;
				B  += 16;
				k  += 4;

				if(kadd>=8)
					{

					// k = 4
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x1 );
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
					
					// k = 5
					ab_tmp1       = _mm256_setzero_pd();
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x3 );
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

					// k = 6
					ab_tmp1       = _mm256_setzero_pd();
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x7 );
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

					// k = 7
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
					k  += 4;

					}
				else
					{

					// k = 4
					a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x1 );
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

					if(kadd>5)
						{
						
						// k = 5
						ab_tmp1       = _mm256_setzero_pd();
						a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x3 );
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

						if(kadd>6)
							{	

							// k = 6
							ab_tmp1       = _mm256_setzero_pd();
							a_4567        = _mm256_blend_pd( ab_tmp1, a_4567, 0x7 );
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

							A0 += 4;
							A1 += 4;
							B  += 4;
							k  += 1;

							}

						A0 += 4;
						A1 += 4;
						B  += 4;
						k  += 1;

						}

					A0 += 4;
					A1 += 4;
					B  += 4;
					k  += 1;

					}

				}
			else // kadd = {1 2 3}
				{

				ab_tmp1       = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &B[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					b_0123        = _mm256_load_pd( &B[8] ); // prefetch
					c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
					b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
					c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
					b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
					c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
					a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
					c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x7 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
						b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
						b_0123        = _mm256_load_pd( &B[12] ); // prefetch
						c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
						b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
						c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
						b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
						c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
						a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
						c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );

						A0 += 4;
						A1 += 4;
						B  += 4;
						k  += 1;

						}

					A0 += 4;
					A1 += 4;
					B  += 4;
					k  += 1;

					}

				A0 += 4;
				A1 += 4;
				B  += 4;
				k  += 1;

				}

			}

		for(; k<kadd-3; k+=4)
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
		
		for(; k<kadd-1; k+=2)
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

		for(; k<kadd; k+=1)
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
			
			A0 += 4; // keep it !!!
			A1 += 4; // keep it !!!
			B  += 4; // keep it !!!

			}

		if(ksub>0)
			{
			A0 += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			A1 += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B  += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}

		}

	if(ksub>0)
		{

		d_print_mat(4, 4, A0, 4);
		d_print_mat(4, 4, A1, 4);

		// prefetch
		a_0123 = _mm256_load_pd( &A0[0] );
		a_4567 = _mm256_load_pd( &A1[0] );
		b_0123 = _mm256_load_pd( &B[0] );

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



// normal-transposed, 8x2 with data packed in 4
void kernel_dgemm_dtrsm_nt_8x2_lib4(int tri, int kadd, int ksub, double *A0, double *A1, double *B, double *C0, double *C1, double *D0, double *D1, double *fact, int alg)
	{
	
	const int bs = 4;
	const int d_ncl = D_NCL;
	
	int k;
	
	__m256d
		zeros,
		a_0123, a_4567, //A_0123,
		b_0101, b_1010,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_20_31, c_01_10_21_30,
		c_40_51_60_71, c_41_50_61_70;
	
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
		a_0123 = _mm256_load_pd( &A0[0] );
		a_4567 = _mm256_load_pd( &A1[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
				
				// k = 1
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

				// k = 2
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

				// k = 2
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
				a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
				
				A0 += 16;
				A1 += 16;
				B  += 16;
				k  += 4;

				if(kadd>=8)
					{

					// k = 4
					a_4567        = _mm256_blend_pd( zeros, a_4567, 0x1 );
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
					
					// k = 5
					a_4567        = _mm256_blend_pd( zeros, a_4567, 0x3 );
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

					// k = 6
					a_4567        = _mm256_blend_pd( zeros, a_4567, 0x7 );
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

					// k = 7
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
					k  += 4;

					}
				else
					{

					// k = 4
					a_4567        = _mm256_blend_pd( zeros, a_4567, 0x1 );
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

					if(kadd>5)
						{
					
						// k = 5
						a_4567        = _mm256_blend_pd( zeros, a_4567, 0x3 );
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

						if(kadd>6)
							{

							// k = 6
							a_4567        = _mm256_blend_pd( zeros, a_4567, 0x7 );
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

							A0 += 4;
							A1 += 4;
							B  += 4;
							k  += 1;

							}

						A0 += 4;
						A1 += 4;
						B  += 4;
						k  += 1;

						}

					A0 += 4;
					A1 += 4;
					B  += 4;
					k  += 1;

					}

				}
			else
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
					b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
					b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
					c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
					a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
					c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
						b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
						b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
						c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
						ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
						a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
						c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );

						A0 += 4;
						A1 += 4;
						B  += 4;
						k  += 1;

						}

					A0 += 4;
					A1 += 4;
					B  += 4;
					k  += 1;

					}

				A0 += 4;
				A1 += 4;
				B  += 4;
				k  += 1;

				}

			}

		for(; k<kadd-3; k+=4)
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
		
		for(; k<kadd-1; k+=2)
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

		for(; k<kadd; k+=1)
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
			
			A0 += 4; // keep it !!!
			A1 += 4; // keep it !!!
			B  += 4; // keep it !!!

			}

		if(ksub>0)
			{
			A0 += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			A1 += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B  += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}

		}
		
	if(ksub>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &A0[0] );
		a_4567 = _mm256_load_pd( &A1[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_sub_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_sub_pd( c_41_50_61_70, ab_tmp1 );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_sub_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_sub_pd( c_41_50_61_70, ab_tmp1 );


	/*	__builtin_prefetch( A+48 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_sub_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_sub_pd( c_41_50_61_70, ab_tmp1 );


	/*	__builtin_prefetch( A+56 );*/
			ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_tmp0 );
			c_40_51_60_71 = _mm256_sub_pd( c_40_51_60_71, ab_tmp1 );
			ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
			ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
			a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_tmp0 );
			c_41_50_61_70 = _mm256_sub_pd( c_41_50_61_70, ab_tmp1 );
			
			A0 += 16;
			A1 += 16;
			B  += 16;

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

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dgemm_dtrsm_nt_4x4_lib4(int tri, int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;
	const int d_ncl = D_NCL;

	int k;
	
	__m256d
		zeros,
		a_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
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
		a_0123 = _mm256_load_pd( &A[0] );
		b_0123 = _mm256_load_pd( &B[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
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
				
				// k = 1
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
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

				// k = 2
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
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

				// k = 3
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
				k += 4;


				}
			else
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
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

				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
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

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
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

						A += 4;
						B += 4;
						k += 1;

						}

					A += 4;
					B += 4;
					k += 1;

					}

				A += 4;
				B += 4;
				k += 1;

				}

			}

		for(; k<kadd-3; k+=4)
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
		
		for(; k<kadd-1; k+=2)
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

		for(; k<kadd; k+=1)
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
			
			A += 4; // keep it !!!
			B += 4; // keep it !!!

			}

		if(ksub>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}

		}

	if(ksub>0)
		{
		
		// prefetch
		a_0123 = _mm256_load_pd( &A[0] );
		b_0123 = _mm256_load_pd( &B[0] );

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
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	_mm256_store_pd( &D[0+bs*0], d_00_10_20_30 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	ab_temp = _mm256_mul_pd( d_00_10_20_30, a_10 );
	d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_temp );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	_mm256_store_pd( &D[0+bs*1], d_01_11_21_31 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	ab_temp = _mm256_mul_pd( d_00_10_20_30, a_20 );
	d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, ab_temp );
	ab_temp = _mm256_mul_pd( d_01_11_21_31, a_21 );
	d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, ab_temp );
	d_02_12_22_32 = _mm256_mul_pd( d_02_12_22_32, a_22 );
	_mm256_store_pd( &D[0+bs*2], d_02_12_22_32 );

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
	_mm256_store_pd( &D[0+bs*3], d_03_13_23_33 );

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_dgemm_dtrsm_nt_4x2_lib4(int tri, int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;
	const int d_ncl = D_NCL;

	int k;
	
	__m256d
		zeros,
		a_0123,
		b_0101, b_1010,
		ab_temp, // temporary results
		c_00_11_20_31, c_01_10_21_30, C_00_11_20_31, C_01_10_21_30;

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
		a_0123 = _mm256_load_pd( &A[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A[4] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
				
				// k = 1
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A[8] ); // prefetch
				C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );

				// k = 2
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A[12] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );

				// k = 3
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A[16] ); // prefetch
				C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
				
				A += 16;
				B += 16;
				k += 4;

				}
			else
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
				b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
				a_0123        = _mm256_load_pd( &A[4] ); // prefetch
				c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
				
				if(kadd>1)
					{
					
					// k = 1
					a_0123        = _mm256_blend_pd( zeros, a_0123, 0x3 );
					ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
					C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
					b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
					b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
					ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
					a_0123        = _mm256_load_pd( &A[8] ); // prefetch
					C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd( zeros, a_0123, 0x7 );
						ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
						c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
						b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
						b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
						ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
						a_0123        = _mm256_load_pd( &A[12] ); // prefetch
						c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );

						A += 4;
						B += 4;
						k += 1;

						}

					A += 4;
					B += 4;
					k += 1;

					}


				A += 4;
				B += 4;
				k += 1;

				}

			}

		for(; k<kadd-3; k+=4)
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
		
		for(; k<kadd-1; k+=2)
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

		for(; k<kadd; k+=1)
			{
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
	/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
	/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch*/
			c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
			
			A += 4; // keep it !!!
			B += 4; // keep it !!!

			}

		if(ksub>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}

		}

	if(ksub>0)
		{

		// prefetch
		a_0123 = _mm256_load_pd( &A[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &A[4] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			C_00_11_20_31 = _mm256_sub_pd( C_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &A[8] ); // prefetch
			C_01_10_21_30 = _mm256_sub_pd( C_01_10_21_30, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			c_00_11_20_31 = _mm256_sub_pd( c_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &A[12] ); // prefetch
			c_01_10_21_30 = _mm256_sub_pd( c_01_10_21_30, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
			C_00_11_20_31 = _mm256_sub_pd( C_00_11_20_31, ab_temp );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
			a_0123        = _mm256_load_pd( &A[16] ); // prefetch
			C_01_10_21_30 = _mm256_sub_pd( C_01_10_21_30, ab_temp );
			
			A += 16;
			B += 16;

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
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00_10_20_30 = _mm256_mul_pd( d_00_10_20_30, a_00 );
	_mm256_store_pd( &D[0+bs*0], d_00_10_20_30 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	ab_temp = _mm256_mul_pd( d_00_10_20_30, a_10 );
	d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, ab_temp );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	_mm256_store_pd( &D[0+bs*1], d_01_11_21_31 );

	}



// normal-transposed, 2x4 with data packed in 4
void kernel_dgemm_dtrsm_nt_2x4_lib4(int tri, int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;
	const int d_ncl = D_NCL;

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
		a_0101 = _mm256_broadcast_pd( (__m128d *) &A[0] );
		b_0123 = _mm256_load_pd( &B[0] );

		if(tri==1)
			{

			if(kadd>=2)
				{

				// k = 0
				a_0101        = _mm256_blend_pd( zeros, a_0101, 0x5 );
				ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
				c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &B[4] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
				a_0101        = _mm256_broadcast_pd( (__m128d *) &A[4] ); // prefetch
				c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
				
				// k = 1
				ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
				C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &B[8] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
				a_0101        = _mm256_broadcast_pd( (__m128d *) &A[8] ); // prefetch
				C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, ab_temp );

				A += 8;
				B += 8;
				k += 2;

				}
			else
				{

				// k = 0
				a_0101        = _mm256_blend_pd( zeros, a_0101, 0x5 );
				ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
				c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &B[4] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
				a_0101        = _mm256_broadcast_pd( (__m128d *) &A[4] ); // prefetch
				c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
	
				A += 4;
				B += 4;
				k += 1;

				}
			
			}

		for(k=0; k<kadd-3; k+=4)
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
		
		if(kadd%4>=2)
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

		if(kadd%2==1)
			{
			
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
	/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
	/*		a_0101        = _mm256_broadcast_pd( (__m128d *) &A[4] ); // prefetch*/
			c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, ab_temp );
			
			A += 4; // keep it !!!
			B += 4; // keep it !!!

			}

		if(ksub>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}

		}

	if(ksub>0)
		{

		// prefetch
		a_0101 = _mm256_broadcast_pd( (__m128d *) &A[0] );
		b_0123 = _mm256_load_pd( &B[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_sub_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &B[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &A[4] ); // prefetch
			c_01_10_03_12 = _mm256_sub_pd( c_01_10_03_12, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			C_00_11_02_13 = _mm256_sub_pd( C_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &B[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &A[8] ); // prefetch
			C_01_10_03_12 = _mm256_sub_pd( C_01_10_03_12, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			c_00_11_02_13 = _mm256_sub_pd( c_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &B[12] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &A[12] ); // prefetch
			c_01_10_03_12 = _mm256_sub_pd( c_01_10_03_12, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0101, b_0123 );
			C_00_11_02_13 = _mm256_sub_pd( C_00_11_02_13, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &B[16] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0101, b_1032 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &A[16] ); // prefetch
			C_01_10_03_12 = _mm256_sub_pd( C_01_10_03_12, ab_temp );
			
			A += 16;
			B += 16;

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



// normal-transposed, 2x2 with data packed in 4
void kernel_dgemm_dtrsm_nt_2x2_lib4(int tri, int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;
	const int d_ncl = D_NCL;

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
		a_01 = _mm_load_pd( &A[0] );
		b_01 = _mm_load_pd( &B[0] );

		if(tri==1)
			{

			if(kadd>=2)
				{

				// k = 0
				ab_temp = _mm_mul_sd( a_01, b_01 );
				c_00_11 = _mm_add_sd( c_00_11, ab_temp );
				b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
				b_01    = _mm_load_pd( &B[4] ); // prefetch
				ab_temp = _mm_mul_pd( a_01, b_10 );
				a_01    = _mm_load_pd( &A[4] ); // prefetch
				c_01_10 = _mm_add_pd( c_01_10, ab_temp );
				
				// k = 1
				ab_temp = _mm_mul_pd( a_01, b_01 );
				C_00_11 = _mm_add_pd( C_00_11, ab_temp );
				b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
				b_01    = _mm_load_pd( &B[8] ); // prefetch
				ab_temp = _mm_mul_pd( a_01, b_10 );
				a_01    = _mm_load_pd( &A[8] ); // prefetch
				C_01_10 = _mm_add_pd( C_01_10, ab_temp );

				A += 8;
				B += 8;
				k += 2;

				}
			else
				{

				// k = 0
				ab_temp = _mm_mul_sd( a_01, b_01 );
				c_00_11 = _mm_add_sd( c_00_11, ab_temp );
				b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
				b_01    = _mm_load_pd( &B[4] ); // prefetch
				ab_temp = _mm_mul_pd( a_01, b_10 );
				a_01    = _mm_load_pd( &A[4] ); // prefetch
				c_01_10 = _mm_add_pd( c_01_10, ab_temp );
	
				A += 4;
				B += 4;
				k += 1;

				}

			}

		for(k=0; k<kadd-3; k+=4)
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
		
		if(kadd%4>=2)
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

		if(kadd%2==1)
			{
			
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_add_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
	/*		b_01    = _mm_load_pd( &B[4] ); // prefetch*/
			ab_temp = _mm_mul_pd( a_01, b_10 );
	/*		a_01    = _mm_load_pd( &A[4] ); // prefetch*/
			c_01_10 = _mm_add_pd( c_01_10, ab_temp );
			
			A += 4; // keep it !!!
			B += 4; // keep it !!!

			}

		if(ksub>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}

		}

	if(ksub>0)
		{

		// prefetch
		a_01 = _mm_load_pd( &A[0] );
		b_01 = _mm_load_pd( &B[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_sub_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &B[4] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &A[4] ); // prefetch
			c_01_10 = _mm_sub_pd( c_01_10, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			C_00_11 = _mm_sub_pd( C_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &B[8] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &A[8] ); // prefetch
			C_01_10 = _mm_sub_pd( C_01_10, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			c_00_11 = _mm_sub_pd( c_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &B[12] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &A[12] ); // prefetch
			c_01_10 = _mm_sub_pd( c_01_10, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp = _mm_mul_pd( a_01, b_01 );
			C_00_11 = _mm_sub_pd( C_00_11, ab_temp );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &B[16] ); // prefetch
			ab_temp = _mm_mul_pd( a_01, b_10 );
			a_01    = _mm_load_pd( &A[16] ); // prefetch
			C_01_10 = _mm_sub_pd( C_01_10, ab_temp );
			
			A += 16;
			B += 16;

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
	
	a_00 = _mm_loaddup_pd( &fact[0] );
	d_00_10 = _mm_mul_pd( d_00_10, a_00 );
	_mm_store_pd( &D[0+bs*0], d_00_10 );

	a_10 = _mm_loaddup_pd( &fact[1] );
	a_11 = _mm_loaddup_pd( &fact[2] );
	ab_temp = _mm_mul_pd( d_00_10, a_10 );
	d_01_11 = _mm_sub_pd( d_01_11, ab_temp );
	d_01_11 = _mm_mul_pd( d_01_11, a_11 );
	_mm_store_pd( &D[0+bs*1], d_01_11 );

	}



#if 0
// A is upper triangular
void kernel_dtrmm_dtrsm_nt_4x4_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg)
	{

	const int bs = 4;
	const int d_ncl = D_NCL;

//	d_print_mat(4, kadd, A, 4);
//	d_print_mat(4, kadd, B, 4);

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
	
	// initialize loop counter
	k = 0;

	if(kadd>=4)
		{

		// initial triangle

		// k=0
		a_0 = A[0+bs*0];
			
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
			
		c_00 += a_0 * b_0;

		c_01 += a_0 * b_1;

		c_02 += a_0 * b_2;

		c_03 += a_0 * b_3;


		// k=1
		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
			
		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		b_3 = B[3+bs*1];
			
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		// k=2
		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
			
		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		b_2 = B[2+bs*2];
		b_3 = B[3+bs*2];
			
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;


		// k=3
		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
			
		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		b_2 = B[2+bs*3];
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
		B += 16;
		k = 4;
			

		for(; k<kadd-3; k+=4)
			{
			
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
			a_2 = A[2+bs*0];
			a_3 = A[3+bs*0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
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
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			b_2 = B[2+bs*1];
			b_3 = B[3+bs*1];
			
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
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			b_2 = B[2+bs*2];
			b_3 = B[3+bs*2];
			
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
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			b_2 = B[2+bs*3];
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
			B += 16;

			}
		for(; k<kadd; k++)
			{
			
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
			a_2 = A[2+bs*0];
			a_3 = A[3+bs*0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
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
			B += 4;

			}
		}
	else if(kadd>0)
		{

		// k = 0
		a_0 = A[0+bs*0];
			
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
			
		c_00 += a_0 * b_0;

		c_01 += a_0 * b_1;

		c_02 += a_0 * b_2;

		c_03 += a_0 * b_3;

		A += 4;
		B += 4;
		k += 1;

		if(kadd>1)
			{

			// k = 1
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
				
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
				
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;

			c_02 += a_0 * b_2;
			c_12 += a_1 * b_2;

			c_03 += a_0 * b_3;
			c_13 += a_1 * b_3;

			A += 4;
			B += 4;
			k += 1;

			if(kadd>2)
				{

				// k = 2
				a_0 = A[0+bs*0];
				a_1 = A[1+bs*0];
				a_2 = A[2+bs*0];
					
				b_0 = B[0+bs*0];
				b_1 = B[1+bs*0];
				b_2 = B[2+bs*0];
				b_3 = B[3+bs*0];
					
				c_00 += a_0 * b_0;
				c_10 += a_1 * b_0;
				c_20 += a_2 * b_0;

				c_01 += a_0 * b_1;
				c_11 += a_1 * b_1;
				c_21 += a_2 * b_1;

				c_02 += a_0 * b_2;
				c_12 += a_1 * b_2;
				c_22 += a_2 * b_2;

				c_03 += a_0 * b_3;
				c_13 += a_1 * b_3;
				c_23 += a_2 * b_3;

				A += 4;
				B += 4;
				k += 1;

				}

			}

		}

	if(ksub>0)
		{
		if(kadd>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		}

	for(k=0; k<ksub-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		b_3 = B[3+bs*1];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		b_2 = B[2+bs*2];
		b_3 = B[3+bs*2];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		b_2 = B[2+bs*3];
		b_3 = B[3+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;
		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		c_23 -= a_2 * b_3;
		c_33 -= a_3 * b_3;
		
		
		A += 16;
		B += 16;

		}

	if(alg!=0)
		{
		c_00 += C[0+bs*0];
		c_10 += C[1+bs*0];
		c_20 += C[2+bs*0];
		c_30 += C[3+bs*0];

		c_01 += C[0+bs*1];
		c_11 += C[1+bs*1];
		c_21 += C[2+bs*1];
		c_31 += C[3+bs*1];

		c_02 += C[0+bs*2];
		c_12 += C[1+bs*2];
		c_22 += C[2+bs*2];
		c_32 += C[3+bs*2];

		c_03 += C[0+bs*3];
		c_13 += C[1+bs*3];
		c_23 += C[2+bs*3];
		c_33 += C[3+bs*3];
		}
	
	// dtrsm
	double
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = fact[0];
	c_00 *= a_00;
	c_10 *= a_00;
	c_20 *= a_00;
	c_30 *= a_00;
	D[0+bs*0] = c_00;
	D[1+bs*0] = c_10;
	D[2+bs*0] = c_20;
	D[3+bs*0] = c_30;

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
	D[0+bs*1] = c_01;
	D[1+bs*1] = c_11;
	D[2+bs*1] = c_21;
	D[3+bs*1] = c_31;

	a_20 = fact[3];
	a_21 = fact[4];
	a_22 = fact[5];
	c_02 -= c_00*a_20;
	c_12 -= c_10*a_20;
	c_22 -= c_20*a_20;
	c_32 -= c_30*a_20;
	c_02 -= c_01*a_21;
	c_12 -= c_11*a_21;
	c_22 -= c_21*a_21;
	c_32 -= c_31*a_21;
	c_02 *= a_22;
	c_12 *= a_22;
	c_22 *= a_22;
	c_32 *= a_22;
	D[0+bs*2] = c_02;
	D[1+bs*2] = c_12;
	D[2+bs*2] = c_22;
	D[3+bs*2] = c_32;

	a_30 = fact[6];
	a_31 = fact[7];
	a_32 = fact[8];
	a_33 = fact[9];
	c_03 -= c_00*a_30;
	c_13 -= c_10*a_30;
	c_23 -= c_20*a_30;
	c_33 -= c_30*a_30;
	c_03 -= c_01*a_31;
	c_13 -= c_11*a_31;
	c_23 -= c_21*a_31;
	c_33 -= c_31*a_31;
	c_03 -= c_02*a_32;
	c_13 -= c_12*a_32;
	c_23 -= c_22*a_32;
	c_33 -= c_32*a_32;
	c_03 *= a_33;
	c_13 *= a_33;
	c_23 *= a_33;
	c_33 *= a_33;
	D[0+bs*3] = c_03;
	D[1+bs*3] = c_13;
	D[2+bs*3] = c_23;
	D[3+bs*3] = c_33;

	}
	
	
	
// A is upper triangular
void kernel_dtrmm_dtrsm_nt_4x2_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg)
	{

	const int bs = 4;
	const int d_ncl = D_NCL;

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1,
		c_00=0, c_01=0,
		c_10=0, c_11=0,
		c_20=0, c_21=0,
		c_30=0, c_31=0;

	// initialize loop counter
	k = 0;

	if(kadd>=4)
		{

		// initial triangle

		// k=0
		a_0 = A[0+bs*0];
			
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
			
		c_00 += a_0 * b_0;

		c_01 += a_0 * b_1;


		// k=1
		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
			
		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
			
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		// k=2
		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
			
		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
			
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;


		// k=3
		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
			
		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
			
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		A += 16;
		B += 16;
		k = 4;

			
		for(; k<kadd-3; k+=4)
			{
			
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
			a_2 = A[2+bs*0];
			a_3 = A[3+bs*0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			
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
			
			b_0 = B[0+bs*1];
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
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			
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
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;
			c_20 += a_2 * b_0;
			c_30 += a_3 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;
			c_21 += a_2 * b_1;
			c_31 += a_3 * b_1;
			
			
			A += 16;
			B += 16;

			}
		for(; k<kadd; k++)
			{
			
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
			a_2 = A[2+bs*0];
			a_3 = A[3+bs*0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;
			c_20 += a_2 * b_0;
			c_30 += a_3 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;
			c_21 += a_2 * b_1;
			c_31 += a_3 * b_1;


			A += 4;
			B += 4;

			}
		}
	else if(kadd>0)
		{

		// k = 0
		a_0 = A[0+bs*0];
			
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
			
		c_00 += a_0 * b_0;

		c_01 += a_0 * b_1;

		A += 4;
		B += 4;
		k += 1;

		if(kadd>1)
			{

			// k = 1
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
				
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
				
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;

			A += 4;
			B += 4;
			k += 1;

			if(kadd>2)
				{

				// k = 2
				a_0 = A[0+bs*0];
				a_1 = A[1+bs*0];
				a_2 = A[2+bs*0];
					
				b_0 = B[0+bs*0];
				b_1 = B[1+bs*0];
					
				c_00 += a_0 * b_0;
				c_10 += a_1 * b_0;
				c_20 += a_2 * b_0;

				c_01 += a_0 * b_1;
				c_11 += a_1 * b_1;
				c_21 += a_2 * b_1;

				A += 4;
				B += 4;
				k += 1;

				}

			}

		}

	if(ksub>0)
		{
		if(kadd>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		}

	for(k=0; k<ksub-3; k+=4)
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

	if(alg!=0)
		{
		c_00 += C[0+bs*0];
		c_10 += C[1+bs*0];
		c_20 += C[2+bs*0];
		c_30 += C[3+bs*0];

		c_01 += C[0+bs*1];
		c_11 += C[1+bs*1];
		c_21 += C[2+bs*1];
		c_31 += C[3+bs*1];
		}
	
	// dtrsm
	double
		a_00, a_10, a_11;
	
	a_00 = fact[0];
	c_00 *= a_00;
	c_10 *= a_00;
	c_20 *= a_00;
	c_30 *= a_00;
	D[0+bs*0] = c_00;
	D[1+bs*0] = c_10;
	D[2+bs*0] = c_20;
	D[3+bs*0] = c_30;

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
	D[0+bs*1] = c_01;
	D[1+bs*1] = c_11;
	D[2+bs*1] = c_21;
	D[3+bs*1] = c_31;

	}
	
	
	
// A is upper triangular
void kernel_dtrmm_dtrsm_nt_2x4_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg)
	{

	const int bs = 4;
	const int d_ncl = D_NCL;

	int k;

	double
		a_0, a_1,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0;
		
	// initialize loop counter
	k = 0;

	if(kadd>=2)
		{

		// initial triangle

		// k=0
		a_0 = A[0+bs*0];
			
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
			
		c_00 += a_0 * b_0;

		c_01 += a_0 * b_1;

		c_02 += a_0 * b_2;

		c_03 += a_0 * b_3;


		// k=1
		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
			
		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		b_3 = B[3+bs*1];
			
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		A += 8;
		B += 8;
		k = 2;
			

		
		for(; k<kadd-3; k+=4)
			{
			
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
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
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			b_2 = B[2+bs*1];
			b_3 = B[3+bs*1];
			
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
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			b_2 = B[2+bs*2];
			b_3 = B[3+bs*2];
			
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
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			b_2 = B[2+bs*3];
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
			B += 16;

			}
		for(; k<kadd; k++)
			{
			
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;

			c_02 += a_0 * b_2;
			c_12 += a_1 * b_2;

			c_03 += a_0 * b_3;
			c_13 += a_1 * b_3;


			A += 4;
			B += 4;

			}
		}
	else if(kadd>0)
		{

		// k = 0
		a_0 = A[0+bs*0];
			
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
			
		c_00 += a_0 * b_0;

		c_01 += a_0 * b_1;

		c_02 += a_0 * b_2;

		c_03 += a_0 * b_3;

		A += 4;
		B += 4;
		k += 1;

		}

	if(ksub>0)
		{
		if(kadd>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		}

	for(k=0; k<ksub-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		
		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		b_3 = B[3+bs*1];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		
		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		b_2 = B[2+bs*2];
		b_3 = B[3+bs*2];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		
		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		b_2 = B[2+bs*3];
		b_3 = B[3+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;

		c_02 -= a_0 * b_2;
		c_12 -= a_1 * b_2;

		c_03 -= a_0 * b_3;
		c_13 -= a_1 * b_3;
		
		
		A += 16;
		B += 16;

		}

	if(alg!=0)
		{
		c_00 += C[0+bs*0];
		c_10 += C[1+bs*0];

		c_01 += C[0+bs*1];
		c_11 += C[1+bs*1];

		c_02 += C[0+bs*2];
		c_12 += C[1+bs*2];

		c_03 += C[0+bs*3];
		c_13 += C[1+bs*3];
		}
	
	// dtrsm
	double
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = fact[0];
	c_00 *= a_00;
	c_10 *= a_00;
	D[0+bs*0] = c_00;
	D[1+bs*0] = c_10;

	a_10 = fact[1];
	a_11 = fact[2];
	c_01 -= c_00*a_10;
	c_11 -= c_10*a_10;
	c_01 *= a_11;
	c_11 *= a_11;
	D[0+bs*1] = c_01;
	D[1+bs*1] = c_11;

	a_20 = fact[3];
	a_21 = fact[4];
	a_22 = fact[5];
	c_02 -= c_00*a_20;
	c_12 -= c_10*a_20;
	c_02 -= c_01*a_21;
	c_12 -= c_11*a_21;
	c_02 *= a_22;
	c_12 *= a_22;
	D[0+bs*2] = c_02;
	D[1+bs*2] = c_12;

	a_30 = fact[6];
	a_31 = fact[7];
	a_32 = fact[8];
	a_33 = fact[9];
	c_03 -= c_00*a_30;
	c_13 -= c_10*a_30;
	c_03 -= c_01*a_31;
	c_13 -= c_11*a_31;
	c_03 -= c_02*a_32;
	c_13 -= c_12*a_32;
	c_03 *= a_33;
	c_13 *= a_33;
	D[0+bs*3] = c_03;
	D[1+bs*3] = c_13;

	}
	
	
	
// A is upper triangular
void kernel_dtrmm_dtrsm_nt_2x2_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg)
	{

	const int bs = 4;
	const int d_ncl = D_NCL;

	int k;

	double
		a_0, a_1,
		b_0, b_1,
		c_00=0, c_01=0,
		c_10=0, c_11=0;

	// initialize loop counter
	k = 0;

	if(kadd>=2)
		{

		// initial triangle

		// k=0
		a_0 = A[0+bs*0];
			
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
			
		c_00 += a_0 * b_0;

		c_01 += a_0 * b_1;


		// k=1
		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
			
		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
			
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		A += 8;
		B += 8;
		k = 2;
			

			
		for(; k<kadd-3; k+=4)
			{
			
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;


			a_0 = A[0+bs*1];
			a_1 = A[1+bs*1];
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;


			a_0 = A[0+bs*2];
			a_1 = A[1+bs*2];
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;


			a_0 = A[0+bs*3];
			a_1 = A[1+bs*3];
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;
			
			
			A += 16;
			B += 16;

			}
		for(; k<kadd; k++)
			{
			
			a_0 = A[0+bs*0];
			a_1 = A[1+bs*0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			
			c_00 += a_0 * b_0;
			c_10 += a_1 * b_0;

			c_01 += a_0 * b_1;
			c_11 += a_1 * b_1;


			A += 4;
			B += 4;

			}
		}
	else if(kadd>0)
		{

		// k = 0
		a_0 = A[0+bs*0];
			
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
			
		c_00 += a_0 * b_0;

		c_01 += a_0 * b_1;

		A += 4;
		B += 4;
		k += 1;

		}

	if(ksub>0)
		{
		if(kadd>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		}

	for(k=0; k<ksub-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		
		b_0 = B[0+bs*1];
		b_1 = B[1+bs*1];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		
		b_0 = B[0+bs*2];
		b_1 = B[1+bs*2];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		
		b_0 = B[0+bs*3];
		b_1 = B[1+bs*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_01 -= a_0 * b_1;
		c_11 -= a_1 * b_1;
		
		
		A += 16;
		B += 16;

		}

	if(alg!=0)
		{
		c_00 += C[0+bs*0];
		c_10 += C[1+bs*0];

		c_01 += C[0+bs*1];
		c_11 += C[1+bs*1];
		}
	
	// dtrsm
	double
		a_00, a_10, a_11;
	
	a_00 = fact[0];
	c_00 *= a_00;
	c_10 *= a_00;
	D[0+bs*0] = c_00;
	D[1+bs*0] = c_10;

	a_10 = fact[1];
	a_11 = fact[2];
	c_01 -= c_00*a_10;
	c_11 -= c_10*a_10;
	c_01 *= a_11;
	c_11 *= a_11;
	D[0+bs*1] = c_01;
	D[1+bs*1] = c_11;

	}
#endif	
	
	

