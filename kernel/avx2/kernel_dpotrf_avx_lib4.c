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
#include <math.h>  // TODO remove !!!

#include "../../include/block_size.h"



// normal-transposed, 8x4 with data packed in 4
void kernel_dsyrk_dpotrf_nt_8x4_lib4(int tri, int kadd, int ksub, double *Ap0, int sdap, double *Bp, double *Am0, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg)
	{
	
	double *Ap1 = Ap0 + 4*sdap;
	double *Am1 = Am0 + 4*sdam;
	double *C1  = C0  + 4*sdc;
	double *D1  = D0  + 4*sdd;
	
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

	if(kadd>0)
		{

		// prefetch
		a_0123        = _mm256_load_pd( &Ap0[0] );
		a_4567        = _mm256_load_pd( &Ap1[0] );
		b_0123        = _mm256_load_pd( &Bp[0] );

		if(tri==1)
			{
			
			if(kadd>=4)
				{

				ab_tmp1       = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch

				// k = 1
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x3 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
				a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch

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
				k   += 4;

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
					k   += 4;

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

								Ap0 += 4;
								Ap1 += 4;
								Bp  += 4;
								k   += 1;

								}

							Ap0 += 4;
							Ap1 += 4;
							Bp  += 4;
							k   += 1;

							}

						Ap0 += 4;
						Ap1 += 4;
						Bp  += 4;
						k   += 1;

						}

					}

				}
			else
				{

				ab_tmp1       = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x1 );
				ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
				a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch

				if(kadd>1)
					{

					// k = 1
					a_0123        = _mm256_blend_pd( ab_tmp1, a_0123, 0x3 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
					c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
					ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
					c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
					a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch

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

						Ap0 += 4;
						Ap1 += 4;
						Bp  += 4;
						k   += 1;

						}

					Ap0 += 4;
					Ap1 += 4;
					Bp  += 4;
					k   += 1;

					}

				Ap0 += 4;
				Ap1 += 4;
				Bp  += 4;
				k   += 1;

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
			
	/*printf("\nk0 = ciao %d\n", (d_ncl-kadd%d_ncl)%d_ncl);*/

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
			
//			A0 += 4; // keep it !!!
//			A1 += 4; // keep it !!!
//			B  += 4; // keep it !!!

			}

		}

	if(ksub>0)
		{

		// prefetch
		a_0123        = _mm256_load_pd( &Am0[0] );
		a_4567        = _mm256_load_pd( &Am1[0] );
		b_0123        = _mm256_load_pd( &Bm[0] );

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
		d_00, d_01, d_02, d_03,
		d_40, d_41, d_42, d_43;

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
		d_00 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		d_02 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_01 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		d_03 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_40 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
		d_42 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
		d_41 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
		d_43 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_00 = _mm256_load_pd( &C0[0+bs*0] );
		d_00 = _mm256_add_pd( d_00, c_00_10_20_30 );
		d_02 = _mm256_load_pd( &C0[0+bs*2] );
		d_02 = _mm256_add_pd( d_02, c_02_12_22_32 );
		c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_01 = _mm256_load_pd( &C0[0+bs*1] );
		d_01 = _mm256_add_pd( d_01, c_01_11_21_31 );
		d_03 = _mm256_load_pd( &C0[0+bs*3] );
		d_03 = _mm256_add_pd( d_03, c_03_13_23_33 );
		c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
		c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
		d_40 = _mm256_load_pd( &C1[0+bs*0] );
		d_40 = _mm256_add_pd( d_40, c_40_50_60_70 );
		d_42 = _mm256_load_pd( &C1[0+bs*2] );
		d_42 = _mm256_add_pd( d_42, c_42_52_62_72 );
		c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
		c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
		d_41 = _mm256_load_pd( &C1[0+bs*1] );
		d_41 = _mm256_add_pd( d_41, c_41_51_61_71 );
		d_43 = _mm256_load_pd( &C1[0+bs*3] );
		d_43 = _mm256_add_pd( d_43, c_43_53_63_73 );
		}
		

#if 1
	// factorize
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_20, sa_30, sa_11, sa_21, sa_31, sa_22, sa_32, sa_33;

	__m256d
		temp,
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;

	__m256i
		mask;

	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00) );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		sa_00 = _mm_movedup_pd( sa_00 );
		_mm_store_sd( &fact[0], sa_00 );
		a_00 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_00 ), _mm256_castpd128_pd256( sa_00 ), 0x0 );
		d_00  = _mm256_mul_pd( d_00, a_00 );
		_mm256_store_pd( &D0[0+bs*0], d_00 ); // a_00
		d_40 = _mm256_mul_pd( d_40, a_00 );
		_mm256_store_pd( &D1[0+bs*0], d_40 );
		}
	else // comile
		{
		a_00 = _mm256_setzero_pd( ); // zero a_00 in d, continue factorizing with 1
		_mm_store_sd( &fact[0], _mm256_castpd256_pd128( a_00 ) ); // store 0 in fact
		d_00  = _mm256_blend_pd( d_00, a_00, 0x1 );
		_mm256_store_pd( &D0[0+bs*0], d_00 );
		_mm256_store_pd( &D1[0+bs*0], d_40 );
		}

	// second row
	sa_10 = _mm_permute_pd( _mm256_castpd256_pd128(d_00), 0x3 );
	_mm_store_sd( &fact[1], sa_10 );
	a_10 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_10 ), _mm256_castpd128_pd256( sa_10 ), 0x0 );
	temp  = _mm256_mul_pd( d_00, a_10 );
	d_01  = _mm256_sub_pd( d_01, temp );
	temp  = _mm256_mul_pd( d_40, a_10 );
	d_41  = _mm256_sub_pd( d_41, temp );
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_permute_pd( _mm256_castpd256_pd128(d_01), 0x3 );
	mask = _mm256_set_epi64x( -1, -1, -1, 1 ); // static memory and load instead ???
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		sa_11 = _mm_movedup_pd( sa_11 );
		_mm_store_sd( &fact[2], sa_11 );
		a_11  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_11 ), _mm256_castpd128_pd256( sa_11 ), 0x0 );
		d_01  = _mm256_mul_pd( d_01, a_11 );
		_mm256_maskstore_pd( &D0[0+bs*1], mask, d_01 ); // a_00
		d_41 = _mm256_mul_pd( d_41, a_11 );
		_mm256_store_pd( &D1[0+bs*1], d_41 );
		}
	else // comile
		{
		a_11 = _mm256_setzero_pd( ); // zero a_00 in d, continue factorizing with 1
		_mm_store_sd( &fact[2], _mm256_castpd256_pd128( a_11 ) ); // store 0 in fact
		d_01  = _mm256_blend_pd( d_01, a_11, 0x3 );
		_mm256_maskstore_pd( &D0[0+bs*1], mask, d_01 );
		_mm256_store_pd( &D1[0+bs*1], d_41 );
		}

	// third row
	sa_20 = _mm256_extractf128_pd( d_00, 0x1 ); // a_20 & a_30
	sa_20 = _mm_permute_pd( sa_20, 0x0 );
	_mm_store_sd( &fact[3], sa_20 );
	a_20  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_20 ), _mm256_castpd128_pd256( sa_20 ), 0x0 );
	temp  = _mm256_mul_pd( d_00, a_20 );
	d_02  = _mm256_sub_pd( d_02, temp );
	temp  = _mm256_mul_pd( d_40, a_20 );
	d_42  = _mm256_sub_pd( d_42, temp );
	sa_21 = _mm256_extractf128_pd( d_01, 0x1 ); // a_20 & a_30
	sa_21 = _mm_permute_pd( sa_21, 0x0 );
	_mm_store_sd( &fact[4], sa_21 );
	a_21  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_21 ), _mm256_castpd128_pd256( sa_21 ), 0x0 );
	temp  = _mm256_mul_pd( d_01, a_21 );
	d_02  = _mm256_sub_pd( d_02, temp );
	temp  = _mm256_mul_pd( d_41, a_21 );
	d_42  = _mm256_sub_pd( d_42, temp );
	sa_22 = _mm256_extractf128_pd( d_02, 0x1 ); // a_22 & a_32
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	mask = _mm256_set_epi64x( -1, -1, 1, 1 );
	if( _mm_comigt_sd ( sa_22, zeros_ones ) )
		{
		sa_22 = _mm_sqrt_sd( sa_22, sa_22 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_22 = _mm_div_sd( zeros_ones, sa_22 );
		sa_22 = _mm_movedup_pd( sa_22 );
		_mm_store_sd( &fact[5], sa_22 );
		a_22  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_22 ), _mm256_castpd128_pd256( sa_22 ), 0x0 );
		d_02  = _mm256_mul_pd( d_02, a_22 );
		_mm256_maskstore_pd( &D0[0+bs*2], mask, d_02 ); // a_00
		d_42 = _mm256_mul_pd( d_42, a_22 );
		_mm256_store_pd( &D1[0+bs*2], d_42 );
		}
	else // comile
		{
		a_22 = _mm256_setzero_pd( ); // zero a_00 in d, continue factorizing with 1
		_mm_store_sd( &fact[5], _mm256_castpd256_pd128( a_22 ) ); // store 0 in fact
		d_02  = _mm256_blend_pd( d_02, a_22, 0x7 );
		_mm256_maskstore_pd( &D0[0+bs*2], mask, d_02 );
		_mm256_store_pd( &D1[0+bs*2], d_42 );
		}

	// fourth row
	sa_30 = _mm256_extractf128_pd( d_00, 0x1 ); // a_20 & a_30
	sa_30 = _mm_permute_pd( sa_30, 0x3 );
	_mm_store_sd( &fact[6], sa_30 );
	a_30  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_30 ), _mm256_castpd128_pd256( sa_30 ), 0x0 );
	temp  = _mm256_mul_pd( d_00, a_30 );
	d_03  = _mm256_sub_pd( d_03, temp );
	temp  = _mm256_mul_pd( d_40, a_30 );
	d_43  = _mm256_sub_pd( d_43, temp );
	sa_31 = _mm256_extractf128_pd( d_01, 0x1 ); // a_21 & a_31
	sa_31 = _mm_permute_pd( sa_31, 0x3 );
	_mm_store_sd( &fact[7], sa_31 );
	a_31  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_31 ), _mm256_castpd128_pd256( sa_31 ), 0x0 );
	temp  = _mm256_mul_pd( d_01, a_31 );
	d_03  = _mm256_sub_pd( d_03, temp );
	temp  = _mm256_mul_pd( d_41, a_31 );
	d_43  = _mm256_sub_pd( d_43, temp );
	sa_32 = _mm256_extractf128_pd( d_02, 0x1 ); // a_22 & a_32
	sa_32 = _mm_permute_pd( sa_32, 0x3 );
	_mm_store_sd( &fact[8], sa_32 );
	a_32  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_32 ), _mm256_castpd128_pd256( sa_32 ), 0x00 );
	temp  = _mm256_mul_pd( d_02, a_32 );
	d_03  = _mm256_sub_pd( d_03, temp );
	temp  = _mm256_mul_pd( d_42, a_32 );
	d_43  = _mm256_sub_pd( d_43, temp );
	sa_33 = _mm256_extractf128_pd( d_03, 0x1 ); // a_33
	sa_33 = _mm_permute_pd( sa_33, 0x3 );
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	mask = _mm256_set_epi64x( -1, 1, 1, 1 );
	if( _mm_comigt_sd ( sa_33, zeros_ones ) )
		{
		sa_33 = _mm_sqrt_sd( sa_33, sa_33 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_33 = _mm_div_sd( zeros_ones, sa_33 );
		sa_33 = _mm_movedup_pd( sa_33 );
		_mm_store_sd( &fact[9], sa_33 );
		a_33  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_33 ), _mm256_castpd128_pd256( sa_33 ), 0x00 );
		d_03  = _mm256_mul_pd( d_03, a_33 );
		_mm256_maskstore_pd( &D0[0+bs*3], mask, d_03 ); // a_00
		d_43 = _mm256_mul_pd( d_43, a_33 );
		_mm256_store_pd( &D1[0+bs*3], d_43 );
		}
	else // comile
		{
		a_33 = _mm256_setzero_pd( ); // zero a_00 in d, continue factorizing with 1
		_mm_store_sd( &fact[9], _mm256_castpd256_pd128( a_33 ) ); // store 0 in fact
		//d_03  = _mm256_blend_pd( d_03, a_33, 0xf );
		//_mm256_maskstore_pd( &D0[0+bs*3], mask, d_03 );
		_mm256_maskstore_pd( &D0[0+bs*3], mask, a_33 );
		_mm256_store_pd( &D1[0+bs*3], d_43 ); // a_00
		}

#else

	// factorize the upper 4x4 matrix
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_20, sa_30, sa_11, sa_21, sa_31, sa_22, sa_32, sa_33;

	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	


	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00) );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		sa_10 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_00), zeros_ones, 0x1 );
		sa_20 = _mm256_extractf128_pd( d_00, 0x1 ); // a_20 & a_30
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D0[0+bs*0], sa_00 ); // a_00
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		sa_00 = _mm_movedup_pd( sa_00 );
		sa_10 = _mm_mul_sd( sa_10, sa_00 );
		sa_10 = _mm_movedup_pd( sa_10 );
		sa_20 = _mm_mul_pd( sa_20, sa_00 ); // a_20 & a_30
		}
	else // comile
		{
		sa_00 = _mm_setzero_pd();
		_mm_store_sd( &D0[0+bs*0], sa_00 ); // a_00
		sa_10 = sa_00;
		sa_20 = sa_00; // a_20 & a_30
		}
	_mm_store_sd( &D0[1+bs*0], sa_10 ); // a_10
	_mm_store_pd( &D0[2+bs*0], sa_20 ); // a_20 & a_30

	// second row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_01), zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_10, sa_10 );
	sa_11 = _mm_sub_sd( sa_11, sab_temp );
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		sa_21 = _mm256_extractf128_pd( d_01, 0x1 ); // a_21 & a_31
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D0[1+bs*1], sa_11 ); // a_11
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		sa_11 = _mm_movedup_pd( sa_11 );
		sab_temp = _mm_mul_pd( sa_20, sa_10 ); // a_21 & a_31
		sa_21 = _mm_sub_pd( sa_21, sab_temp ); // a_21 & a_31
		sa_21 = _mm_mul_pd( sa_21, sa_11 ); // a_21 & a_31
		}
	else // comile
		{
		sa_11 = _mm_setzero_pd();
		_mm_store_sd( &D0[1+bs*1], sa_11 ); // a_11
		sa_21 = sa_11; // a_21 & a_31
		}
	_mm_store_pd( &D0[2+bs*1], sa_21 ); // a_21 & a_31
	
	// third row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_22 = _mm256_extractf128_pd( d_02, 0x1 ); // a_22 & a_32
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
		_mm_store_sd( &D0[2+bs*2], sa_22 ); // a_22
		sa_22 = _mm_div_sd( zeros_ones, sa_22 );
		sa_32 = _mm_mul_sd( sa_32, sa_22 );
		}
	else // comile
		{
		sa_22 = _mm_setzero_pd();
		_mm_store_sd( &D0[2+bs*2], sa_22 ); // a_22
		sa_32 = sa_22; // a_21 & a_31
		}
	_mm_store_sd( &D0[3+bs*2], sa_32 ); // a_32

	sa_30 = _mm_shuffle_pd( sa_20, zeros_ones, 0x1 ); // a_30
	sa_31 = _mm_shuffle_pd( sa_21, zeros_ones, 0x1 ); // a_31

	// fourth row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_33 = _mm_shuffle_pd( _mm256_extractf128_pd( d_03, 0x1 ), zeros_ones, 0x1 );
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
		_mm_store_sd( &D0[3+bs*3], sa_33 ); // a_33
		sa_33 = _mm_div_sd( zeros_ones, sa_33 );
		}
	else // comile
		{
		sa_33 = _mm_setzero_pd();
		_mm_store_sd( &D0[3+bs*3], sa_33 ); // a_33
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


	// solve the lower 4x4 matrix
	d_40= _mm256_mul_pd( d_40, a_00 );
	_mm256_store_pd( &D1[0+bs*0], d_40);

	ab_tmp0 = _mm256_mul_pd( d_40, a_10 );
	d_41= _mm256_sub_pd( d_41, ab_tmp0 );
	d_41= _mm256_mul_pd( d_41, a_11 );
	_mm256_store_pd( &D1[0+bs*1], d_41);

	ab_tmp0 = _mm256_mul_pd( d_40, a_20 );
	d_42= _mm256_sub_pd( d_42, ab_tmp0 );
	ab_tmp0 = _mm256_mul_pd( d_41, a_21 );
	d_42= _mm256_sub_pd( d_42, ab_tmp0 );
	d_42= _mm256_mul_pd( d_42, a_22 );
	_mm256_store_pd( &D1[0+bs*2], d_42);

	ab_tmp0 = _mm256_mul_pd( d_40, a_30 );
	d_43= _mm256_sub_pd( d_43, ab_tmp0 );
	ab_tmp0 = _mm256_mul_pd( d_41, a_31 );
	d_43= _mm256_sub_pd( d_43, ab_tmp0 );
	ab_tmp0 = _mm256_mul_pd( d_42, a_32 );
	d_43= _mm256_sub_pd( d_43, ab_tmp0 );
	d_43= _mm256_mul_pd( d_43, a_33 );
	_mm256_store_pd( &D1[0+bs*3], d_43);
#endif

	//d_print_mat(4, 4, D0, 4);
	//d_print_mat(4, 4, D1, 4);
	//d_print_mat(1, 10, fact, 1);
	//exit(1);

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dsyrk_dpotrf_nt_4x4_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;
	const int d_ncl = D_NCL;
	
	int k;
	
	__m256d
		zeros,
		a_0123, //A_0123,
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
		a_0123        = _mm256_load_pd( &Ap[0] );
		b_0123        = _mm256_load_pd( &Bp[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0123        = _mm256_blend_pd(zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch

				// k = 1
				a_0123        = _mm256_blend_pd(zeros, a_0123, 0x3 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
				a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch

				// k = 2
				a_0123        = _mm256_blend_pd(zeros, a_0123, 0x7 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
				ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );

				// k = 3
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
				b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch 
				ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
				c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
				b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
				c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
				b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
				ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
				a_0123        = _mm256_load_pd( &Ap[16] ); // prefetch
				c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
				
				Ap += 16;
				Bp += 16;
				k  += 4;

				}
			else
				{

				// k = 0
				a_0123        = _mm256_blend_pd(zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
				a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
				c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
				b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch

				if(kadd>1)
					{

					// k = 1
					a_0123        = _mm256_blend_pd(zeros, a_0123, 0x3 );
					ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
					c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
					b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
					b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
					ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
					c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
					a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch

					if(kadd>2)
						{

						// k = 2
						a_0123        = _mm256_blend_pd(zeros, a_0123, 0x7 );
						ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
						c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
						b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
						b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
						ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
						c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
						b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
						ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
						c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
						b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
						ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
						a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
						c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );

						Ap += 4;
						Bp += 4;
						k  += 1;

						}

					Ap += 4;
					Bp += 4;
					k  += 1;

					}

				Ap += 4;
				Bp += 4;
				k  += 1;

				}

			}

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch 
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[16] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
			Ap += 16;
			Bp += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch 
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
			
			Ap += 8;
			Bp += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
	/*		b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch */
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
	/*		a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch */
			c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
			
//			Ap += 4; // keep it !!!
//			Bp += 4; // keep it !!!
			
			}

		}
		
	if(ksub>0)
		{

		// prefetch
		a_0123        = _mm256_load_pd( &Am[0] );
		b_0123        = _mm256_load_pd( &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[4] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Am[4] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );
			
			
	/*	__builtin_prefetch( A+40 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[8] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Am[8] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );


	/*	__builtin_prefetch( A+48 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[12] ); // prefetch
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
			ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
			a_0123        = _mm256_load_pd( &Am[12] ); // prefetch
			c_02_13_20_31 = _mm256_sub_pd( c_02_13_20_31, ab_temp );


	/*	__builtin_prefetch( A+56 );*/
			ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
			c_00_11_22_33 = _mm256_sub_pd( c_00_11_22_33, ab_temp );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[16] ); // prefetch 
			ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
			c_01_10_23_32 = _mm256_sub_pd( c_01_10_23_32, ab_temp );
			b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
			ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
			c_03_12_21_30 = _mm256_sub_pd( c_03_12_21_30, ab_temp );
			b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
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
		d_00, d_01, d_02, d_03;

	c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
	c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
	c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
	c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
	
	if(alg==0)
		{
		d_00 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		d_02 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_01 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		d_03 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
		c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
		d_00 = _mm256_load_pd( &C[0+bs*0] );
		d_00 = _mm256_add_pd( d_00, c_00_10_20_30 );
		d_02 = _mm256_load_pd( &C[0+bs*2] );
		d_02 = _mm256_add_pd( d_02, c_02_12_22_32 );
		c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
		c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		d_01 = _mm256_load_pd( &C[0+bs*1] );
		d_01 = _mm256_add_pd( d_01, c_01_11_21_31 );
		d_03 = _mm256_load_pd( &C[0+bs*3] );
		d_03 = _mm256_add_pd( d_03, c_03_13_23_33 );
		}
		
#if 1
	// factorize
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_20, sa_30, sa_11, sa_21, sa_31, sa_22, sa_32, sa_33;

	__m256d
		temp,
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;

	__m256i
		mask;

	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00) );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		sa_00 = _mm_movedup_pd( sa_00 );
		_mm_store_sd( &fact[0], sa_00 );
		a_00 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_00 ), _mm256_castpd128_pd256( sa_00 ), 0x0 );
		d_00  = _mm256_mul_pd( d_00, a_00 );
		_mm256_store_pd( &D[0+bs*0], d_00 ); // a_00
		}
	else // comile
		{
		a_00  = _mm256_setzero_pd();
		_mm_store_sd( &fact[0], _mm256_castpd256_pd128(a_00) );
		d_00  = _mm256_blend_pd( d_00, a_00, 0x1 );
		_mm256_store_pd( &D[0+bs*0], d_00 );
		}

	// second row
	sa_10 = _mm_permute_pd( _mm256_castpd256_pd128(d_00), 0x3 );
	_mm_store_sd( &fact[1], sa_10 );
	a_10 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_10 ), _mm256_castpd128_pd256( sa_10 ), 0x0 );
	temp  = _mm256_mul_pd( d_00, a_10 );
	d_01  = _mm256_sub_pd( d_01, temp );
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_permute_pd( _mm256_castpd256_pd128(d_01), 0x3 );
	mask = _mm256_set_epi64x( -1, -1, -1, 1 ); // static memory and load instead ???
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		sa_11 = _mm_movedup_pd( sa_11 );
		_mm_store_sd( &fact[2], sa_11 );
		a_11  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_11 ), _mm256_castpd128_pd256( sa_11 ), 0x0 );
		d_01  = _mm256_mul_pd( d_01, a_11 );
		_mm256_maskstore_pd( &D[0+bs*1], mask, d_01 ); // a_00
		}
	else // comile
		{
		a_11  = _mm256_setzero_pd();
		_mm_store_sd( &fact[2], _mm256_castpd256_pd128(a_11) );
		d_01  = _mm256_blend_pd( d_01, a_11, 0x3 );
		_mm256_maskstore_pd( &D[0+bs*1], mask, d_01 );
		}

	// third row
	sa_20 = _mm256_extractf128_pd( d_00, 0x1 ); // a_20 & a_30
	sa_20 = _mm_permute_pd( sa_20, 0x0 );
	_mm_store_sd( &fact[3], sa_20 );
	a_20  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_20 ), _mm256_castpd128_pd256( sa_20 ), 0x0 );
	temp  = _mm256_mul_pd( d_00, a_20 );
	d_02  = _mm256_sub_pd( d_02, temp );
	sa_21 = _mm256_extractf128_pd( d_01, 0x1 ); // a_20 & a_30
	sa_21 = _mm_permute_pd( sa_21, 0x0 );
	_mm_store_sd( &fact[4], sa_21 );
	a_21  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_21 ), _mm256_castpd128_pd256( sa_21 ), 0x0 );
	temp  = _mm256_mul_pd( d_01, a_21 );
	d_02  = _mm256_sub_pd( d_02, temp );
	sa_22 = _mm256_extractf128_pd( d_02, 0x1 ); // a_22 & a_32
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	mask = _mm256_set_epi64x( -1, -1, 1, 1 );
	if( _mm_comigt_sd ( sa_22, zeros_ones ) )
		{
		sa_22 = _mm_sqrt_sd( sa_22, sa_22 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_22 = _mm_div_sd( zeros_ones, sa_22 );
		sa_22 = _mm_movedup_pd( sa_22 );
		_mm_store_sd( &fact[5], sa_22 );
		a_22  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_22 ), _mm256_castpd128_pd256( sa_22 ), 0x0 );
		d_02  = _mm256_mul_pd( d_02, a_22 );
		_mm256_maskstore_pd( &D[0+bs*2], mask, d_02 ); // a_00
		}
	else // comile
		{
		a_22  = _mm256_setzero_pd();
		_mm_store_sd( &fact[5], _mm256_castpd256_pd128(a_22) );
		d_02  = _mm256_blend_pd( d_02, a_22, 0x7 );
		_mm256_maskstore_pd( &D[0+bs*2], mask, d_02 );
		}

	// fourth row
	sa_30 = _mm256_extractf128_pd( d_00, 0x1 ); // a_20 & a_30
	sa_30 = _mm_permute_pd( sa_30, 0x3 );
	_mm_store_sd( &fact[6], sa_30 );
	a_30  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_30 ), _mm256_castpd128_pd256( sa_30 ), 0x0 );
	temp  = _mm256_mul_pd( d_00, a_30 );
	d_03  = _mm256_sub_pd( d_03, temp );
	sa_31 = _mm256_extractf128_pd( d_01, 0x1 ); // a_21 & a_31
	sa_31 = _mm_permute_pd( sa_31, 0x3 );
	_mm_store_sd( &fact[7], sa_31 );
	a_31  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_31 ), _mm256_castpd128_pd256( sa_31 ), 0x0 );
	temp  = _mm256_mul_pd( d_01, a_31 );
	d_03  = _mm256_sub_pd( d_03, temp );
	sa_32 = _mm256_extractf128_pd( d_02, 0x1 ); // a_22 & a_32
	sa_32 = _mm_permute_pd( sa_32, 0x3 );
	_mm_store_sd( &fact[8], sa_32 );
	a_32  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_32 ), _mm256_castpd128_pd256( sa_32 ), 0x00 );
	temp  = _mm256_mul_pd( d_02, a_32 );
	d_03  = _mm256_sub_pd( d_03, temp );
	sa_33 = _mm256_extractf128_pd( d_03, 0x1 ); // a_33
	sa_33 = _mm_permute_pd( sa_33, 0x3 );
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	mask = _mm256_set_epi64x( -1, 1, 1, 1 );
	if( _mm_comigt_sd ( sa_33, zeros_ones ) )
		{
		sa_33 = _mm_sqrt_sd( sa_33, sa_33 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_33 = _mm_div_sd( zeros_ones, sa_33 );
		sa_33 = _mm_movedup_pd( sa_33 );
		_mm_store_sd( &fact[9], sa_33 );
		a_33  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_33 ), _mm256_castpd128_pd256( sa_33 ), 0x00 );
		d_03  = _mm256_mul_pd( d_03, a_33 );
		_mm256_maskstore_pd( &D[0+bs*3], mask, d_03 ); // a_00
		}
	else // comile
		{
		a_33  = _mm256_setzero_pd();
		_mm_store_sd( &fact[9], _mm256_castpd256_pd128(a_33) );
		_mm256_maskstore_pd( &D[0+bs*3], mask, a_33 );
		}

#else



	// factorize the upper 4x4 matrix
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_20, sa_30, sa_11, sa_21, sa_31, sa_22, sa_32, sa_33;


	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00) );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		sa_10 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_00), zeros_ones, 0x1 );
		sa_20 = _mm256_extractf128_pd( d_00, 0x1 ); // a_20 & a_30
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[0+bs*0], sa_00 ); // a_00
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		sa_00 = _mm_movedup_pd( sa_00 );
		sa_10 = _mm_mul_sd( sa_10, sa_00 );
		sa_10 = _mm_movedup_pd( sa_10 );
		sa_20 = _mm_mul_pd( sa_20, sa_00 ); // a_20 & a_30
		}
	else // comile
		{
		sa_00 = _mm_setzero_pd();
		_mm_store_sd( &D[0+bs*0], sa_00 ); // a_00
		sa_10 = sa_00;
		sa_20 = sa_00; // a_20 & a_30
		}
	_mm_store_sd( &D[1+bs*0], sa_10 ); // a_10
	_mm_store_pd( &D[2+bs*0], sa_20 ); // a_20 & a_30

	// second row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_01), zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_10, sa_10 );
	sa_11 = _mm_sub_sd( sa_11, sab_temp );
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		sa_21 = _mm256_extractf128_pd( d_01, 0x1 ); // a_21 & a_31
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[1+bs*1], sa_11 ); // a_11
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		sa_11 = _mm_movedup_pd( sa_11 );
		sab_temp = _mm_mul_pd( sa_20, sa_10 ); // a_21 & a_31
		sa_21 = _mm_sub_pd( sa_21, sab_temp ); // a_21 & a_31
		sa_21 = _mm_mul_pd( sa_21, sa_11 ); // a_21 & a_31
		}
	else // comile
		{
		sa_11 = _mm_setzero_pd();
		_mm_store_sd( &D[1+bs*1], sa_11 ); // a_11
		sa_21 = sa_11; // a_21 & a_31
		}
	_mm_store_pd( &D[2+bs*1], sa_21 ); // a_21 & a_31
	
	// third row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_22 = _mm256_extractf128_pd( d_02, 0x1 ); // a_22 & a_32
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
		_mm_store_sd( &D[2+bs*2], sa_22 ); // a_22
		sa_22 = _mm_div_sd( zeros_ones, sa_22 );
		sa_32 = _mm_mul_sd( sa_32, sa_22 );
		}
	else // comile
		{
		sa_22 = _mm_setzero_pd();
		_mm_store_sd( &D[2+bs*2], sa_22 ); // a_22
		sa_32 = sa_22; // a_21 & a_31
		}
	_mm_store_sd( &D[3+bs*2], sa_32 ); // a_32

	sa_30 = _mm_shuffle_pd( sa_20, zeros_ones, 0x1 ); // a_30
	sa_31 = _mm_shuffle_pd( sa_21, zeros_ones, 0x1 ); // a_31

	// fourth row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_33 = _mm_shuffle_pd( _mm256_extractf128_pd( d_03, 0x1 ), zeros_ones, 0x1 );
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
		_mm_store_sd( &D[3+bs*3], sa_33 ); // a_33
		sa_33 = _mm_div_sd( zeros_ones, sa_33 );
		}
	else // comile
		{
		sa_33 = _mm_setzero_pd();
		_mm_store_sd( &D[3+bs*3], sa_33 ); // a_33
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
#endif

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_dsyrk_dpotrf_nt_4x2_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp,double *Am, double *Bm, double *C, double *D, double *fact, int alg)
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
		a_0123 = _mm256_load_pd( &Ap[0] );
		b_0101 = _mm256_broadcast_pd( (__m128d *) &Bp[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
		
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
				k  += 4;

				}
			else
				{


				// k = 0
				a_0123        = _mm256_blend_pd( zeros, a_0123, 0x1 );
				ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
				a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
				c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
				b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch

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

						Ap += 4;
						Bp += 4;
						k  += 1;

						}

					Ap += 4;
					Bp += 4;
					k  += 1;

					}

				Ap += 4;
				Bp += 4;
				k  += 1;

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
		d_00, d_01;

	if(alg==0)
		{
		d_00 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
		d_01 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
		d_00 = _mm256_load_pd( &C[0+bs*0] );
		d_00 = _mm256_add_pd( d_00, c_00_10_20_30 );
		c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
		d_01 = _mm256_load_pd( &C[0+bs*1] );
		d_01 = _mm256_add_pd( d_01, c_01_11_21_31 );
		}
		
#if 1
	// factorize
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_20, sa_30, sa_11, sa_21, sa_31, sa_22, sa_32, sa_33;

	__m256d
		temp,
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;

	__m256i
		mask;

	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00) );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		sa_00 = _mm_movedup_pd( sa_00 );
		_mm_store_sd( &fact[0], sa_00 );
		a_00 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_00 ), _mm256_castpd128_pd256( sa_00 ), 0x0 );
		d_00  = _mm256_mul_pd( d_00, a_00 );
		_mm256_store_pd( &D[0+bs*0], d_00 ); // a_00
		}
	else // comile
		{
		a_00  = _mm256_setzero_pd();
		_mm_store_sd( &fact[0], _mm256_castpd256_pd128(a_00) );
		d_00  = _mm256_blend_pd( d_00, a_00, 0x1 );
		_mm256_store_pd( &D[0+bs*0], d_00 );
		}

	// second row
	sa_10 = _mm_permute_pd( _mm256_castpd256_pd128(d_00), 0x3 );
	_mm_store_sd( &fact[1], sa_10 );
	a_10 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_10 ), _mm256_castpd128_pd256( sa_10 ), 0x0 );
	temp  = _mm256_mul_pd( d_00, a_10 );
	d_01  = _mm256_sub_pd( d_01, temp );
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_permute_pd( _mm256_castpd256_pd128(d_01), 0x3 );
	mask = _mm256_set_epi64x( -1, -1, -1, 1 ); // static memory and load instead ???
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		zeros_ones = _mm_set_sd( 1.0 );
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		sa_11 = _mm_movedup_pd( sa_11 );
		_mm_store_sd( &fact[2], sa_11 );
		a_11  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( sa_11 ), _mm256_castpd128_pd256( sa_11 ), 0x0 );
		d_01  = _mm256_mul_pd( d_01, a_11 );
		_mm256_maskstore_pd( &D[0+bs*1], mask, d_01 ); // a_00
		}
	else // comile
		{
		a_11  = _mm256_setzero_pd();
		_mm_store_sd( &fact[2], _mm256_castpd256_pd128(a_11) );
		d_01  = _mm256_blend_pd( d_01, a_11, 0x3 );
		_mm256_maskstore_pd( &D[0+bs*1], mask, d_01 );
		}


#else




	// factorize the upper 4x4 matrix
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_20, sa_30, sa_11, sa_21, sa_31;


	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, _mm256_castpd256_pd128(d_00) );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		sa_10 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_00), zeros_ones, 0x1 );
		sa_20 = _mm256_extractf128_pd( d_00, 0x1 ); // a_20 & a_30
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[0+bs*0], sa_00 ); // a_00
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		sa_00 = _mm_movedup_pd( sa_00 );
		sa_10 = _mm_mul_sd( sa_10, sa_00 );
		sa_10 = _mm_movedup_pd( sa_10 );
		sa_20 = _mm_mul_pd( sa_20, sa_00 ); // a_20 & a_30
		}
	else // comile
		{
		sa_00 = _mm_setzero_pd();
		_mm_store_sd( &D[0+bs*0], sa_00 ); // a_00
		sa_10 = sa_00;
		sa_20 = sa_00; // a_20 & a_30
		}
	_mm_store_sd( &D[1+bs*0], sa_10 ); // a_10
	_mm_store_pd( &D[2+bs*0], sa_20 ); // a_20 & a_30

	// second row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_shuffle_pd( _mm256_castpd256_pd128(d_01), zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_10, sa_10 );
	sa_11 = _mm_sub_sd( sa_11, sab_temp );
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		sa_21 = _mm256_extractf128_pd( d_01, 0x1 ); // a_21 & a_31
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[1+bs*1], sa_11 ); // a_11
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		sa_11 = _mm_movedup_pd( sa_11 );
		sab_temp = _mm_mul_pd( sa_20, sa_10 ); // a_21 & a_31
		sa_21 = _mm_sub_pd( sa_21, sab_temp ); // a_21 & a_31
		sa_21 = _mm_mul_pd( sa_21, sa_11 ); // a_21 & a_31
		}
	else // comile
		{
		sa_11 = _mm_setzero_pd();
		_mm_store_sd( &D[1+bs*1], sa_11 ); // a_11
		sa_21 = sa_11; // a_21 & a_31
		}
	_mm_store_pd( &D[2+bs*1], sa_21 ); // a_21 & a_31
	
	// duplicate & store

	_mm_store_sd( &fact[0], sa_00 );
	_mm_store_sd( &fact[1], sa_10 );
	_mm_store_sd( &fact[2], sa_11 );
#endif


	}



// normal-transposed, 2x2 with data packed in 4
void kernel_dsyrk_dpotrf_nt_2x2_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
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

	k = 0;

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
				b_01    = _mm_load_pd( &Bp[4] ); // prefetch
				a_01    = _mm_load_pd( &Ap[4] ); // prefetch

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
				k  += 2;

				}
			else
				{

				// k = 0
				ab_temp = _mm_mul_sd( a_01, b_01 );
				c_00_11 = _mm_add_sd( c_00_11, ab_temp );
				b_01    = _mm_load_pd( &Bp[4] ); // prefetch
				a_01    = _mm_load_pd( &Ap[4] ); // prefetch

				Ap += 4;
				Bp += 4;
				k  += 1;

				}

			}

		for(; k<kadd-3; k+=4)
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
		
		for(; k<kadd-1; k+=2)
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

		for(; k<kadd; k+=1)
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
		
	// factorize the upper 4x4 matrix
	__m128d
		zeros_ones, sab_temp,
		sa_00, sa_10, sa_11;


	// first row
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_00 = _mm_move_sd( sa_00, d_00_10 );
	if( _mm_comigt_sd ( sa_00, zeros_ones ) )
		{
		sa_00 = _mm_sqrt_sd( sa_00, sa_00 );
		sa_10 = _mm_shuffle_pd( d_00_10, zeros_ones, 0x1 );
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[0+bs*0], sa_00 ); // a_00
		sa_00 = _mm_div_sd( zeros_ones, sa_00 );
		_mm_store_sd( &fact[0], sa_00 );
		sa_10 = _mm_mul_sd( sa_10, sa_00 );
		}
	else // comile
		{
		sa_00 = _mm_setzero_pd();
		_mm_store_sd( &D[0+bs*0], sa_00 ); // a_00
		//sa_10 = sa_00;
		_mm_store_sd( &fact[0], sa_00 );
		}
	_mm_store_sd( &D[1+bs*0], sa_10 ); // a_10

	// second row
	_mm_store_sd( &fact[1], sa_10 );
	zeros_ones = _mm_set_sd( 1e-15 ); // 0.0 ???
	sa_11 = _mm_shuffle_pd( d_01_11, zeros_ones, 0x1 );
	sab_temp = _mm_mul_sd( sa_10, sa_10 );
	sa_11 = _mm_sub_sd( sa_11, sab_temp );
	if( _mm_comigt_sd ( sa_11, zeros_ones ) )
		{
		sa_11 = _mm_sqrt_sd( sa_11, sa_11 );
		zeros_ones = _mm_set_sd( 1.0 );
		_mm_store_sd( &D[1+bs*1], sa_11 ); // a_11
		sa_11 = _mm_div_sd( zeros_ones, sa_11 );
		_mm_store_sd( &fact[2], sa_11 );
		}
	else // comile
		{
		sa_11 = _mm_setzero_pd();
		_mm_store_sd( &D[1+bs*1], sa_11 ); // a_11
		_mm_store_sd( &fact[2], sa_11 );
		}
	

	}

