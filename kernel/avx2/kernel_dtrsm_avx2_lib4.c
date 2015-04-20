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

#include "../../include/block_size.h"



// normal-transposed, 12x4 with data packed in 4
void kernel_dgemm_dtrsm_nt_12x4_lib4(int tri, int kadd, int ksub, double *Ap0, int sdap, double *Bp, double *Am0, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg)
	{

	double *Ap1 = Ap0 + 4*sdap;
	double *Ap2 = Ap0 + 8*sdap;
	double *Am1 = Am0 + 4*sdam;
	double *Am2 = Am0 + 8*sdam;
	double *C1 = C0 + 4*sdc;
	double *C2 = C0 + 8*sdc;
	double *D1 = D0 + 4*sdd;
	double *D2 = D0 + 8*sdd;
	
	const int bs = 4;
	
	int k;
	
	__m256d
		zeros,
		a_0, a_4, a_8,
		b_0,
		c_00, c_01, c_03, c_02,
		c_40, c_41, c_43, c_42,
		c_80, c_81, c_83, c_82;
	
	// zero registers
	c_00 = _mm256_setzero_pd();
	c_01 = _mm256_setzero_pd();
	c_03 = _mm256_setzero_pd();
	c_02 = _mm256_setzero_pd();
	c_40 = _mm256_setzero_pd();
	c_41 = _mm256_setzero_pd();
	c_43 = _mm256_setzero_pd();
	c_42 = _mm256_setzero_pd();
	c_80 = _mm256_setzero_pd();
	c_81 = _mm256_setzero_pd();
	c_83 = _mm256_setzero_pd();
	c_82 = _mm256_setzero_pd();

	k = 0;

	//printf("\n%d\n", kadd);

	if(kadd>0)
		{

		// prefetch
		a_0 = _mm256_load_pd( &Ap0[0] );
		a_4 = _mm256_load_pd( &Ap1[0] );
		a_8 = _mm256_load_pd( &Ap2[0] );
		b_0 = _mm256_load_pd( &Bp[0] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				zeros = _mm256_setzero_pd(); // TODO use mask load instead !!!!!!!!!!

				// k = 0
				a_0  = _mm256_blend_pd( zeros, a_0, 0x1 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
				b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
				c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
				a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
				b_0  = _mm256_load_pd( &Bp[4] ); // prefetch
				
				// k = 1
				a_0  = _mm256_blend_pd( zeros, a_0, 0x3 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
				b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
				c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
				a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
				b_0  = _mm256_load_pd( &Bp[8] ); // prefetch

				// k = 2
				a_0  = _mm256_blend_pd( zeros, a_0, 0x7 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
				b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
				c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
				a_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
				b_0  = _mm256_load_pd( &Bp[12] ); // prefetch

				// k = 3
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
				b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
				c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
				a_0  = _mm256_load_pd( &Ap0[16] ); // prefetch
				b_0  = _mm256_load_pd( &Bp[16] ); // prefetch
				a_4  = _mm256_load_pd( &Ap1[16] ); // prefetch

				Ap0 += 16;
				Ap1 += 16;
				Ap2 += 16;
				Bp  += 16;
				k  += 4;

				if(kadd>=8)
					{

					// k = 4
					a_4  = _mm256_blend_pd( zeros, a_4, 0x1 );
					c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
					c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
					c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
					b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
					c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
					c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
					a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
					c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
					a_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
					b_0  = _mm256_load_pd( &Bp[4] ); // prefetch
								
					// k = 5
					a_4  = _mm256_blend_pd( zeros, a_4, 0x3 );
					c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
					c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
					c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
					b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
					c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
					c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
					a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
					c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
					a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
					b_0  = _mm256_load_pd( &Bp[8] ); // prefetch

					// k = 6
					a_4  = _mm256_blend_pd( zeros, a_4, 0x7 );
					c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
					c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
					c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
					b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
					c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
					c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
					a_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
					c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
					a_4  = _mm256_load_pd( &Ap1[12] ); // prefetch
					b_0  = _mm256_load_pd( &Bp[12] ); // prefetch

					// k = 7
					c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
					c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
					c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
					b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
					c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
					c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
					a_0  = _mm256_load_pd( &Ap0[16] ); // prefetch
					c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
					a_4  = _mm256_load_pd( &Ap1[16] ); // prefetch
					b_0  = _mm256_load_pd( &Bp[16] ); // prefetch
					a_8  = _mm256_load_pd( &Ap2[16] ); // prefetch
				
					Ap0 += 16;
					Ap1 += 16;
					Ap2 += 16;
					Bp  += 16;
					k  += 4;

					if(kadd>=12)
						{

						// k = 8
						a_8  = _mm256_blend_pd( zeros, a_8, 0x1 );
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
						c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
						c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
						c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );
						b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
						c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
						c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
						c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
						a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
						c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
						a_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
						c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
						b_0  = _mm256_load_pd( &Bp[4] ); // prefetch
						a_8  = _mm256_load_pd( &Ap2[4] ); // prefetch
									
						// k = 9
						a_8  = _mm256_blend_pd( zeros, a_8, 0x3 );
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
						c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
						c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
						c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );
						b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
						c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
						c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
						c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
						a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
						c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
						a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
						c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
						b_0  = _mm256_load_pd( &Bp[8] ); // prefetch
						a_8  = _mm256_load_pd( &Ap2[8] ); // prefetch

						// k = 10
						a_8  = _mm256_blend_pd( zeros, a_8, 0x7 );
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
						c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
						c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
						c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );
						b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
						c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
						c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
						c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
						a_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
						c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
						a_4  = _mm256_load_pd( &Ap1[12] ); // prefetch
						c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
						b_0  = _mm256_load_pd( &Bp[12] ); // prefetch
						a_8  = _mm256_load_pd( &Ap2[12] ); // prefetch

						// k = 11
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
						c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
						c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
						c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );
						b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
						c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
						c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
						c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
						a_0  = _mm256_load_pd( &Ap0[16] ); // prefetch
						c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
						a_4  = _mm256_load_pd( &Ap1[16] ); // prefetch
						c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
						b_0  = _mm256_load_pd( &Bp[16] ); // prefetch
						a_8  = _mm256_load_pd( &Ap2[16] ); // prefetch
										
						Ap0 += 16;
						Ap1 += 16;
						Ap2 += 16;
						Bp  += 16;
						k  += 4;

						}
					else
						{

						if(kadd>8)
							{

							// k = 8
							a_8  = _mm256_blend_pd( zeros, a_8, 0x1 );
							c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
							c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
							c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );
							b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
							c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
							c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
							c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );
							b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
							c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
							c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
							c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );
							b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
							c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
							a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
							c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
							a_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
							c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
							b_0  = _mm256_load_pd( &Bp[4] ); // prefetch
							a_8  = _mm256_load_pd( &Ap2[4] ); // prefetch
		
							k += 1;

							if(kadd>9)
								{

								// k = 9
								a_8  = _mm256_blend_pd( zeros, a_8, 0x3 );
								c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
								c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
								c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );
								b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
								c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
								c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
								c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );
								b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
								c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
								c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
								c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );
								b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
								c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
								a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
								c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
								a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
								c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
								b_0  = _mm256_load_pd( &Bp[8] ); // prefetch
								a_8  = _mm256_load_pd( &Ap2[8] ); // prefetch

								k += 1;

								if(kadd>10)
									{

									// k = 10
									a_8  = _mm256_blend_pd( zeros, a_8, 0x7 );
									c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
									c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
									c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );
									b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
									c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
									c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
									c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );
									b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
									c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
									c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
									c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );
									b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
									c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
									a_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
									c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
									a_4  = _mm256_load_pd( &Ap1[12] ); // prefetch
									c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
									b_0  = _mm256_load_pd( &Bp[12] ); // prefetch
									a_8  = _mm256_load_pd( &Ap2[12] ); // prefetch

									k += 1;

									}
								}
							}
						}
					}
				else
					{

					if(kadd>4)
						{

						// k = 4
						a_4  = _mm256_blend_pd( zeros, a_4, 0x1 );
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
						c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
						b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
						c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
						c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
						a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
						c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
						a_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
						b_0  = _mm256_load_pd( &Bp[4] ); // prefetch

						k  += 1;

						if(kadd>5)
							{
							
							// k = 5
							a_4  = _mm256_blend_pd( zeros, a_4, 0x3 );
							c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
							c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
							b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
							c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
							c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
							b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
							c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
							c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
							b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
							c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
							a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
							c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
							a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
							b_0  = _mm256_load_pd( &Bp[8] ); // prefetch

							k  += 1;

							if(kadd>6)
								{	

								// k = 6
								a_4  = _mm256_blend_pd( zeros, a_4, 0x7 );
								c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
								c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
								b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
								c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
								c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
								b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
								c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
								c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
								b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
								c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
								a_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
								c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
								a_4  = _mm256_load_pd( &Ap1[12] ); // prefetch
								b_0  = _mm256_load_pd( &Bp[12] ); // prefetch

								k  += 1;

								}

							}

						}

					}

				}
			else // kadd = {1 2 3}
				{

				zeros  = _mm256_setzero_pd();

				// k = 0
				a_0  = _mm256_blend_pd( zeros, a_0, 0x1 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
				b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
				c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
				b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
				a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
				b_0  = _mm256_load_pd( &Bp[4] ); // prefetch

				k  += 1;

				if(kadd>1)
					{
					
					// k = 1
					a_0  = _mm256_blend_pd( zeros, a_0, 0x3 );
					c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
					b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
					c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
					b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
					a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
					b_0  = _mm256_load_pd( &Bp[8] ); // prefetch

					k  += 1;

					if(kadd>2)
						{

						// k = 2
						a_0  = _mm256_blend_pd( zeros, a_0, 0x7 );
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
						b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
						c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
						b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
						a_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
						b_0  = _mm256_load_pd( &Bp[12] ); // prefetch

						k  += 1;

						}

					}

				}

			}

		for(; k<kadd-3; k+=4)
			{
			
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
			c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
			c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bp[4] ); // prefetch
			a_8  = _mm256_load_pd( &Ap2[4] ); // prefetch
			
			
			
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
			c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
			c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bp[8] ); // prefetch
			a_8  = _mm256_load_pd( &Ap2[8] ); // prefetch



			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
			c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Ap1[12] ); // prefetch
			c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bp[12] ); // prefetch
			a_8  = _mm256_load_pd( &Ap2[12] ); // prefetch


			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Ap0[16] ); // prefetch
			c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Ap1[16] ); // prefetch
			c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bp[16] ); // prefetch
			a_8  = _mm256_load_pd( &Ap2[16] ); // prefetch
			
			Ap0 += 16;
			Ap1 += 16;
			Ap2 += 16;
			Bp  += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
			c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
			c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bp[4] );
			a_8  = _mm256_load_pd( &Ap2[4] ); // prefetch
			
			
			
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
			c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
			c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bp[8] );
			a_8  = _mm256_load_pd( &Ap2[8] ); // prefetch
				
			
			Ap0 += 8;
			Ap1 += 8;
			Ap2 += 8;
			Bp  += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_0, c_02 );
	/*		a_0  = _mm256_load_pd( &Ap0[4] ); // prefetch*/
			c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
	/*		a_4  = _mm256_load_pd( &Ap1[4] ); // prefetch*/
			c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
	//		b_0  = _mm256_load_pd( &Bp[4] );
	/*		a_8  = _mm256_load_pd( &Ap2[4] ); // prefetch*/

			}
		}

	if(ksub>0)
		{

		//d_print_mat(4, 4, A0, 4);
		//d_print_mat(4, 4, A1, 4);

		// prefetch
		a_0 = _mm256_load_pd( &Am0[0] );
		a_4 = _mm256_load_pd( &Am1[0] );
		a_8 = _mm256_load_pd( &Am2[0] );
		b_0 = _mm256_load_pd( &Bm[0] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
			c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fnmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fnmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fnmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fnmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fnmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fnmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fnmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fnmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Am0[4] ); // prefetch
			c_42 = _mm256_fnmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Am1[4] ); // prefetch
			c_82 = _mm256_fnmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bm[4] ); // prefetch
			a_8  = _mm256_load_pd( &Am2[4] ); // prefetch
			
			
			
			c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fnmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fnmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fnmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fnmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fnmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fnmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fnmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fnmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Am0[8] ); // prefetch
			c_42 = _mm256_fnmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Am1[8] ); // prefetch
			c_82 = _mm256_fnmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bm[8] ); // prefetch
			a_8  = _mm256_load_pd( &Am2[8] ); // prefetch



			c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fnmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fnmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fnmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fnmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fnmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fnmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fnmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fnmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Am0[12] ); // prefetch
			c_42 = _mm256_fnmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Am1[12] ); // prefetch
			c_82 = _mm256_fnmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bm[12] ); // prefetch
			a_8  = _mm256_load_pd( &Am2[12] ); // prefetch


			c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
			c_80 = _mm256_fnmadd_pd( a_8, b_0, c_80 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_01 = _mm256_fnmadd_pd( a_0, b_0, c_01 );
			c_41 = _mm256_fnmadd_pd( a_4, b_0, c_41 );
			c_81 = _mm256_fnmadd_pd( a_8, b_0, c_81 );

			b_0  = _mm256_permute2f128_pd( b_0, b_0, 0x1 );
			c_03 = _mm256_fnmadd_pd( a_0, b_0, c_03 );
			c_43 = _mm256_fnmadd_pd( a_4, b_0, c_43 );
			c_83 = _mm256_fnmadd_pd( a_8, b_0, c_83 );

			b_0  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_02 = _mm256_fnmadd_pd( a_0, b_0, c_02 );
			a_0  = _mm256_load_pd( &Am0[16] ); // prefetch
			c_42 = _mm256_fnmadd_pd( a_4, b_0, c_42 );
			a_4  = _mm256_load_pd( &Am1[16] ); // prefetch
			c_82 = _mm256_fnmadd_pd( a_8, b_0, c_82 );
			b_0  = _mm256_load_pd( &Bm[16] ); // prefetch
			a_8  = _mm256_load_pd( &Am2[16] ); // prefetch
			
			
			Am0 += 16;
			Am1 += 16;
			Am2 += 16;
			Bm  += 16;

			}

		}

	__m256d
		d_00, d_01, d_02, d_03,
		d_40, d_41, d_42, d_43,
		d_80, d_81, d_82, d_83,
		e_00, e_01, e_02, e_03;

		e_00 = _mm256_blend_pd( c_00, c_01, 0xa );
		e_01 = _mm256_blend_pd( c_00, c_01, 0x5 );
		e_02 = _mm256_blend_pd( c_02, c_03, 0xa );
		e_03 = _mm256_blend_pd( c_02, c_03, 0x5 );
		
		d_00 = _mm256_blend_pd( e_00, e_02, 0xc );
		d_02 = _mm256_blend_pd( e_00, e_02, 0x3 );
		d_01 = _mm256_blend_pd( e_01, e_03, 0xc );
		d_03 = _mm256_blend_pd( e_01, e_03, 0x3 );

		e_00 = _mm256_blend_pd( c_40, c_41, 0xa );
		e_01 = _mm256_blend_pd( c_40, c_41, 0x5 );
		e_02 = _mm256_blend_pd( c_42, c_43, 0xa );
		e_03 = _mm256_blend_pd( c_42, c_43, 0x5 );
		
		d_40 = _mm256_blend_pd( e_00, e_02, 0xc );
		d_42 = _mm256_blend_pd( e_00, e_02, 0x3 );
		d_41 = _mm256_blend_pd( e_01, e_03, 0xc );
		d_43 = _mm256_blend_pd( e_01, e_03, 0x3 );

		e_00 = _mm256_blend_pd( c_80, c_81, 0xa );
		e_01 = _mm256_blend_pd( c_80, c_81, 0x5 );
		e_02 = _mm256_blend_pd( c_82, c_83, 0xa );
		e_03 = _mm256_blend_pd( c_82, c_83, 0x5 );
		
		d_80 = _mm256_blend_pd( e_00, e_02, 0xc );
		d_82 = _mm256_blend_pd( e_00, e_02, 0x3 );
		d_81 = _mm256_blend_pd( e_01, e_03, 0xc );
		d_83 = _mm256_blend_pd( e_01, e_03, 0x3 );

	if(alg!=0)
		{
		c_00 = _mm256_load_pd( &C0[0+bs*0] );
		c_01 = _mm256_load_pd( &C0[0+bs*1] );
		c_02 = _mm256_load_pd( &C0[0+bs*2] );
		c_03 = _mm256_load_pd( &C0[0+bs*3] );

		d_00 = _mm256_add_pd( d_00, c_00 );
		d_01 = _mm256_add_pd( d_01, c_01 );
		d_02 = _mm256_add_pd( d_02, c_02 );
		d_03 = _mm256_add_pd( d_03, c_03 );

		c_40 = _mm256_load_pd( &C1[0+bs*0] );
		c_41 = _mm256_load_pd( &C1[0+bs*1] );
		c_42 = _mm256_load_pd( &C1[0+bs*2] );
		c_43 = _mm256_load_pd( &C1[0+bs*3] );

		d_40 = _mm256_add_pd( d_40, c_40 );
		d_41 = _mm256_add_pd( d_41, c_41 );
		d_42 = _mm256_add_pd( d_42, c_42 );
		d_43 = _mm256_add_pd( d_43, c_43 );

		c_80 = _mm256_load_pd( &C2[0+bs*0] );
		c_81 = _mm256_load_pd( &C2[0+bs*1] );
		c_82 = _mm256_load_pd( &C2[0+bs*2] );
		c_83 = _mm256_load_pd( &C2[0+bs*3] );

		d_80 = _mm256_add_pd( d_80, c_80 );
		d_81 = _mm256_add_pd( d_81, c_81 );
		d_82 = _mm256_add_pd( d_82, c_82 );
		d_83 = _mm256_add_pd( d_83, c_83 );
		}
		
	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00 = _mm256_mul_pd( d_00, a_00 );
	d_40 = _mm256_mul_pd( d_40, a_00 );
	d_80 = _mm256_mul_pd( d_80, a_00 );
	_mm256_store_pd( &D0[0+bs*0], d_00 );
	_mm256_store_pd( &D1[0+bs*0], d_40 );
	_mm256_store_pd( &D2[0+bs*0], d_80 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	d_01 = _mm256_fnmadd_pd( d_00, a_10, d_01 );
	d_41 = _mm256_fnmadd_pd( d_40, a_10, d_41 );
	d_81 = _mm256_fnmadd_pd( d_80, a_10, d_81 );
	d_01 = _mm256_mul_pd( d_01, a_11 );
	d_41 = _mm256_mul_pd( d_41, a_11 );
	d_81 = _mm256_mul_pd( d_81, a_11 );
	_mm256_store_pd( &D0[0+bs*1], d_01 );
	_mm256_store_pd( &D1[0+bs*1], d_41 );
	_mm256_store_pd( &D2[0+bs*1], d_81 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	d_02 = _mm256_fnmadd_pd( d_00, a_20, d_02 );
	d_42 = _mm256_fnmadd_pd( d_40, a_20, d_42 );
	d_82 = _mm256_fnmadd_pd( d_80, a_20, d_82 );
	d_02 = _mm256_fnmadd_pd( d_01, a_21, d_02 );
	d_42 = _mm256_fnmadd_pd( d_41, a_21, d_42 );
	d_82 = _mm256_fnmadd_pd( d_81, a_21, d_82 );
	d_02 = _mm256_mul_pd( d_02, a_22 );
	d_42 = _mm256_mul_pd( d_42, a_22 );
	d_82 = _mm256_mul_pd( d_82, a_22 );
	_mm256_store_pd( &D0[0+bs*2], d_02 );
	_mm256_store_pd( &D1[0+bs*2], d_42 );
	_mm256_store_pd( &D2[0+bs*2], d_82 );

	a_30 = _mm256_broadcast_sd( &fact[6] );
	a_31 = _mm256_broadcast_sd( &fact[7] );
	a_32 = _mm256_broadcast_sd( &fact[8] );
	a_33 = _mm256_broadcast_sd( &fact[9] );
	d_03 = _mm256_fnmadd_pd( d_00, a_30, d_03 );
	d_43 = _mm256_fnmadd_pd( d_40, a_30, d_43 );
	d_83 = _mm256_fnmadd_pd( d_80, a_30, d_83 );
	d_03 = _mm256_fnmadd_pd( d_01, a_31, d_03 );
	d_43 = _mm256_fnmadd_pd( d_41, a_31, d_43 );
	d_83 = _mm256_fnmadd_pd( d_81, a_31, d_83 );
	d_03 = _mm256_fnmadd_pd( d_02, a_32, d_03 );
	d_43 = _mm256_fnmadd_pd( d_42, a_32, d_43 );
	d_83 = _mm256_fnmadd_pd( d_82, a_32, d_83 );
	d_03 = _mm256_mul_pd( d_03, a_33 );
	d_43 = _mm256_mul_pd( d_43, a_33 );
	d_83 = _mm256_mul_pd( d_83, a_33 );
	_mm256_store_pd( &D0[0+bs*3], d_03 );
	_mm256_store_pd( &D1[0+bs*3], d_43 );
	_mm256_store_pd( &D2[0+bs*3], d_83 );

	}



// normal-transposed, 8x4 with data packed in 4
void kernel_dgemm_dtrsm_nt_8x4_lib4(int tri, int kadd, int ksub, double *Ap0, int sdap, double *Bp, double *Am0, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg)
	{

	double *Ap1 = Ap0 + 4*sdap;
	double *Am1 = Am0 + 4*sdam;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	
	int k;
	
	__m256d
		zeros,
		a_0, a_4, A_0, A_4,
		b_0, b_1, b_2,
		c_00, c_01, c_03, c_02,
		c_40, c_41, c_43, c_42;
	
	// zero registers
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

	//printf("\n%d\n", kadd);

	if(kadd>0)
		{

		// prefetch
		a_0 = _mm256_load_pd( &Ap0[0] );
		a_4 = _mm256_load_pd( &Ap1[0] );
		b_0 = _mm256_broadcast_pd( (__m128d *) &Bp[0] );
		b_2 = _mm256_broadcast_pd( (__m128d *) &Bp[2] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0  = _mm256_blend_pd( zeros, a_0, 0x1 );
				A_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
				b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
				c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
				b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
				b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
				c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
						
				// k = 1
				A_0  = _mm256_blend_pd( zeros, A_0, 0x3 );
				a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
				b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
				b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
				c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
				b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
				c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
				b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch
				c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );

				// k = 2
				a_0  = _mm256_blend_pd( zeros, a_0, 0x7 );
				A_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
				b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
				c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
				b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
				b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[14] ); // prefetch
				c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );

				// k = 3
				a_0  = _mm256_load_pd( &Ap0[16] ); // prefetch
				a_4  = _mm256_load_pd( &Ap1[16] ); // prefetch
				b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
				b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
				c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
				b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
				c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
				b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[18] ); // prefetch
				c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
						
				Ap0 += 16;
				Ap1 += 16;
				Bp  += 16;
				k  += 4;

				if(kadd>=8)
					{

					// k = 4
					a_4  = _mm256_blend_pd( zeros, a_4, 0x1 );
					A_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
					A_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
					b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
					c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
					b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
					c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
					c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
					b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
					c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
					c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
					b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
					c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
					c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
								
					// k = 5
					A_4  = _mm256_blend_pd( zeros, A_4, 0x3 );
					a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
					a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
					b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
					c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
					b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
					c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
					c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
					b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
					c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
					c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
					b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch
					c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
					c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );

					// k = 6
					a_4  = _mm256_blend_pd( zeros, a_4, 0x7 );
					A_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
					A_4  = _mm256_load_pd( &Ap1[12] ); // prefetch
					b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
					c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
					b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
					c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
					c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
					b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
					c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
					c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
					b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[14] ); // prefetch
					c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
					c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
						
					// k = 7
					a_0  = _mm256_load_pd( &Ap0[16] ); // prefetch
					a_4  = _mm256_load_pd( &Ap1[16] ); // prefetch
					b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
					c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
					b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
					c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
					c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
					b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
					c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
					c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
					b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[18] ); // prefetch
					c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
					c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );
					
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
						a_4  = _mm256_blend_pd( zeros, a_4, 0x1 );
						A_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
						A_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
						b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
						b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
						c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
						c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
						b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
						c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
						b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
						c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
						c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );

						k  += 1;

						if(kadd>5)
							{
							
							// k = 5
							A_4  = _mm256_blend_pd( zeros, A_4, 0x3 );
							a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
							a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
							b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
							c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
							c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
							b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
							c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
							c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
							b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
							c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
							c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
							b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch
							c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
							c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );

							k  += 1;

							if(kadd>6)
								{	

								// k = 6
								a_4  = _mm256_blend_pd( zeros, a_4, 0x7 );
								A_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
								A_4  = _mm256_load_pd( &Ap1[12] ); // prefetch
								b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
								c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
								c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
								b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
								c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
								c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
								b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
								c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
								c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
								b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[14] ); // prefetch
								c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
								c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );

								k  += 1;

								}

							}

						}

					}

				}
			else // kadd = {1 2 3}
				{

				a_0  = _mm256_blend_pd( zeros, a_0, 0x1 );
				A_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
				b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
				c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
				b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
				b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
				c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );

				k  += 1;

				if(kadd>1)
					{
					
					// k = 1
					A_0  = _mm256_blend_pd( zeros, A_0, 0x3 );
					a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
					b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
					c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
					b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
					c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
					b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
					c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
					b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch
					c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );

					k  += 1;

					if(kadd>2)
						{

						// k = 2
						a_0  = _mm256_blend_pd( zeros, a_0, 0x7 );
						A_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
						b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
						c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
						b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
						b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[14] ); // prefetch
						c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );

						k  += 1;

						}

					}

				}

			}

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			A_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
			A_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
			c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
			c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
			
	/*	__builtin_prefetch( A+40 );*/
			a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
			a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
			c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
			c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch
			c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
			c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );
		
	/*	__builtin_prefetch( A+48 );*/
			A_0  = _mm256_load_pd( &Ap0[12] ); // prefetch
			A_4  = _mm256_load_pd( &Ap1[12] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
			c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
			c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[14] ); // prefetch
			c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
		
	/*	__builtin_prefetch( A+56 );*/
			a_0  = _mm256_load_pd( &Ap0[16] ); // prefetch
			a_4  = _mm256_load_pd( &Ap1[16] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
			c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
			c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
			c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[18] ); // prefetch
			c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
			c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );

			Ap0 += 16;
			Ap1 += 16;
			Bp  += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			A_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
			A_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
			c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
			c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
			
			a_0  = _mm256_load_pd( &Ap0[8] ); // prefetch
			a_4  = _mm256_load_pd( &Ap1[8] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
			c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
			c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch
			c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
			c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );
				
			
			Ap0 += 8;
			Ap1 += 8;
			Bp  += 8;

			}

		for(; k<kadd; k+=1)
			{
			
	//		A_0  = _mm256_load_pd( &Ap0[4] ); // prefetch
	//		A_4  = _mm256_load_pd( &Ap1[4] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
	//		b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
			c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
			c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
	//		b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
			c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
			c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
					
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
		a_0 = _mm256_load_pd( &Am0[0] );
		a_4 = _mm256_load_pd( &Am1[0] );
		b_0 = _mm256_broadcast_pd( (__m128d *) &Bm[0] );
		b_2 = _mm256_broadcast_pd( (__m128d *) &Bm[2] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			A_0  = _mm256_load_pd( &Am0[4] ); // prefetch
			A_4  = _mm256_load_pd( &Am1[4] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bm[4] ); // prefetch
			c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
			c_41 = _mm256_fnmadd_pd( a_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
			c_42 = _mm256_fnmadd_pd( a_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bm[6] ); // prefetch
			c_03 = _mm256_fnmadd_pd( a_0, b_1, c_03 );
			c_43 = _mm256_fnmadd_pd( a_4, b_1, c_43 );
			
	/*	__builtin_prefetch( A+40 );*/
			a_0  = _mm256_load_pd( &Am0[8] ); // prefetch
			a_4  = _mm256_load_pd( &Am1[8] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fnmadd_pd( A_0, b_0, c_00 );
			c_40 = _mm256_fnmadd_pd( A_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bm[8] ); // prefetch
			c_01 = _mm256_fnmadd_pd( A_0, b_1, c_01 );
			c_41 = _mm256_fnmadd_pd( A_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fnmadd_pd( A_0, b_2, c_02 );
			c_42 = _mm256_fnmadd_pd( A_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bm[10] ); // prefetch
			c_03 = _mm256_fnmadd_pd( A_0, b_1, c_03 );
			c_43 = _mm256_fnmadd_pd( A_4, b_1, c_43 );
		
	/*	__builtin_prefetch( A+48 );*/
			A_0  = _mm256_load_pd( &Am0[12] ); // prefetch
			A_4  = _mm256_load_pd( &Am1[12] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
			c_40 = _mm256_fnmadd_pd( a_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bm[12] ); // prefetch
			c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
			c_41 = _mm256_fnmadd_pd( a_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
			c_42 = _mm256_fnmadd_pd( a_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bm[14] ); // prefetch
			c_03 = _mm256_fnmadd_pd( a_0, b_1, c_03 );
			c_43 = _mm256_fnmadd_pd( a_4, b_1, c_43 );
		
	/*	__builtin_prefetch( A+56 );*/
			a_0  = _mm256_load_pd( &Am0[16] ); // prefetch
			a_4  = _mm256_load_pd( &Am1[16] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fnmadd_pd( A_0, b_0, c_00 );
			c_40 = _mm256_fnmadd_pd( A_4, b_0, c_40 );
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bm[16] ); // prefetch
			c_01 = _mm256_fnmadd_pd( A_0, b_1, c_01 );
			c_41 = _mm256_fnmadd_pd( A_4, b_1, c_41 );
			b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fnmadd_pd( A_0, b_2, c_02 );
			c_42 = _mm256_fnmadd_pd( A_4, b_2, c_42 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bm[18] ); // prefetch
			c_03 = _mm256_fnmadd_pd( A_0, b_1, c_03 );
			c_43 = _mm256_fnmadd_pd( A_4, b_1, c_43 );
		
			Am0 += 16;
			Am1 += 16;
			Bm  += 16;

			}

		}

	__m256d
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33,
		d_40_50_60_70, d_41_51_61_71, d_42_52_62_72, d_43_53_63_73;


	if(alg==0)
		{
		d_00_10_20_30 = _mm256_blend_pd( c_00, c_01, 0xa );
		d_01_11_21_31 = _mm256_blend_pd( c_00, c_01, 0x5 );
		d_02_12_22_32 = _mm256_blend_pd( c_02, c_03, 0xa );
		d_03_13_23_33 = _mm256_blend_pd( c_02, c_03, 0x5 );
		d_40_50_60_70 = _mm256_blend_pd( c_40, c_41, 0xa );
		d_41_51_61_71 = _mm256_blend_pd( c_40, c_41, 0x5 );
		d_42_52_62_72 = _mm256_blend_pd( c_42, c_43, 0xa );
		d_43_53_63_73 = _mm256_blend_pd( c_42, c_43, 0x5 );
		}
	else
		{
		c_00_10_20_30 = _mm256_blend_pd( c_00, c_01, 0xa );
		c_01_11_21_31 = _mm256_blend_pd( c_00, c_01, 0x5 );
		c_02_12_22_32 = _mm256_blend_pd( c_02, c_03, 0xa );
		c_03_13_23_33 = _mm256_blend_pd( c_02, c_03, 0x5 );
		d_00_10_20_30 = _mm256_load_pd( &C0[0+bs*0] );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+bs*1] );
		d_02_12_22_32 = _mm256_load_pd( &C0[0+bs*2] );
		d_03_13_23_33 = _mm256_load_pd( &C0[0+bs*3] );
		d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
		d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
		d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
		d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );

		c_40_50_60_70 = _mm256_blend_pd( c_40, c_41, 0xa );
		c_41_51_61_71 = _mm256_blend_pd( c_40, c_41, 0x5 );
		c_42_52_62_72 = _mm256_blend_pd( c_42, c_43, 0xa );
		c_43_53_63_73 = _mm256_blend_pd( c_42, c_43, 0x5 );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+bs*0] );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+bs*1] );
		d_42_52_62_72 = _mm256_load_pd( &C1[0+bs*2] );
		d_43_53_63_73 = _mm256_load_pd( &C1[0+bs*3] );
		d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
		d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
		d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
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
	d_01_11_21_31 = _mm256_fnmadd_pd( d_00_10_20_30, a_10, d_01_11_21_31 );
	d_41_51_61_71 = _mm256_fnmadd_pd( d_40_50_60_70, a_10, d_41_51_61_71 );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
	_mm256_store_pd( &D0[0+bs*1], d_01_11_21_31 );
	_mm256_store_pd( &D1[0+bs*1], d_41_51_61_71 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	d_02_12_22_32 = _mm256_fnmadd_pd( d_00_10_20_30, a_20, d_02_12_22_32 );
	d_42_52_62_72 = _mm256_fnmadd_pd( d_40_50_60_70, a_20, d_42_52_62_72 );
	d_02_12_22_32 = _mm256_fnmadd_pd( d_01_11_21_31, a_21, d_02_12_22_32 );
	d_42_52_62_72 = _mm256_fnmadd_pd( d_41_51_61_71, a_21, d_42_52_62_72 );
	d_02_12_22_32 = _mm256_mul_pd( d_02_12_22_32, a_22 );
	d_42_52_62_72 = _mm256_mul_pd( d_42_52_62_72, a_22 );
	_mm256_store_pd( &D0[0+bs*2], d_02_12_22_32 );
	_mm256_store_pd( &D1[0+bs*2], d_42_52_62_72 );

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
	_mm256_store_pd( &D0[0+bs*3], d_03_13_23_33 );
	_mm256_store_pd( &D1[0+bs*3], d_43_53_63_73 );

	}



// normal-transposed, 8x2 with data packed in 4
void kernel_dgemm_dtrsm_nt_8x2_lib4(int tri, int kadd, int ksub, double *Ap0, int sdap, double *Bp, double *Am0, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg)
	{
	
	double *Ap1 = Ap0 + 4*sdap;
	double *Am1 = Am0 + 4*sdam;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	
	int k;
	
	__m256d
		zeros,
		a_0123, a_4567, //A_0123,
		b_0101, b_1010,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_20_31, c_01_10_21_30,
		c_40_51_60_71, c_41_50_61_70,
		C_00_11_20_31, C_01_10_21_30,
		C_40_51_60_71, C_41_50_61_70;
	
	// zero registers
	zeros = _mm256_setzero_pd();
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	c_40_51_60_71 = _mm256_setzero_pd();
	c_41_50_61_70 = _mm256_setzero_pd();
	C_00_11_20_31 = _mm256_setzero_pd();
	C_01_10_21_30 = _mm256_setzero_pd();
	C_40_51_60_71 = _mm256_setzero_pd();
	C_41_50_61_70 = _mm256_setzero_pd();

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
			c_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			c_40_51_60_71 = _mm256_fmadd_pd( a_4567, b_0101, c_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			c_41_50_61_70 = _mm256_fmadd_pd( a_4567, b_1010, c_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, C_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			C_40_51_60_71 = _mm256_fmadd_pd( a_4567, b_0101, C_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			C_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, C_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			C_41_50_61_70 = _mm256_fmadd_pd( a_4567, b_1010, C_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
		

	/*	__builtin_prefetch( A+48 );*/
			c_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			c_40_51_60_71 = _mm256_fmadd_pd( a_4567, b_0101, c_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
			c_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap0[12] ); // prefetch
			c_41_50_61_70 = _mm256_fmadd_pd( a_4567, b_1010, c_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Ap1[12] ); // prefetch
		

	/*	__builtin_prefetch( A+56 );*/
			C_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, C_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			C_40_51_60_71 = _mm256_fmadd_pd( a_4567, b_0101, C_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
			C_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, C_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap0[16] ); // prefetch
			C_41_50_61_70 = _mm256_fmadd_pd( a_4567, b_1010, C_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Ap1[16] ); // prefetch
				
			Ap0 += 16;
			Ap1 += 16;
			Bp  += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			c_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			c_40_51_60_71 = _mm256_fmadd_pd( a_4567, b_0101, c_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			c_41_50_61_70 = _mm256_fmadd_pd( a_4567, b_1010, c_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, C_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			C_40_51_60_71 = _mm256_fmadd_pd( a_4567, b_0101, C_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			C_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, C_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap0[8] ); // prefetch
			C_41_50_61_70 = _mm256_fmadd_pd( a_4567, b_1010, C_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Ap1[8] ); // prefetch
			
			
			Ap0 += 8;
			Ap1 += 8;
			Bp  += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			c_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			c_40_51_60_71 = _mm256_fmadd_pd( a_4567, b_0101, c_40_51_60_71 );
			//b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			//a_0123        = _mm256_load_pd( &Ap0[4] ); // prefetch
			c_41_50_61_70 = _mm256_fmadd_pd( a_4567, b_1010, c_41_50_61_70 );
			//a_4567        = _mm256_load_pd( &Ap1[4] ); // prefetch
			
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
			c_00_11_20_31 = _mm256_fnmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			c_40_51_60_71 = _mm256_fnmadd_pd( a_4567, b_0101, c_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[4] ); // prefetch
			c_01_10_21_30 = _mm256_fnmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Am0[4] ); // prefetch
			c_41_50_61_70 = _mm256_fnmadd_pd( a_4567, b_1010, c_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Am1[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11_20_31 = _mm256_fnmadd_pd( a_0123, b_0101, C_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			C_40_51_60_71 = _mm256_fnmadd_pd( a_4567, b_0101, C_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[8] ); // prefetch
			C_01_10_21_30 = _mm256_fnmadd_pd( a_0123, b_1010, C_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Am0[8] ); // prefetch
			C_41_50_61_70 = _mm256_fnmadd_pd( a_4567, b_1010, C_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Am1[8] ); // prefetch
		

	/*	__builtin_prefetch( A+48 );*/
			c_00_11_20_31 = _mm256_fnmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			c_40_51_60_71 = _mm256_fnmadd_pd( a_4567, b_0101, c_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[12] ); // prefetch
			c_01_10_21_30 = _mm256_fnmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Am0[12] ); // prefetch
			c_41_50_61_70 = _mm256_fnmadd_pd( a_4567, b_1010, c_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Am1[12] ); // prefetch
		

	/*	__builtin_prefetch( A+56 );*/
			C_00_11_20_31 = _mm256_fnmadd_pd( a_0123, b_0101, C_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			C_40_51_60_71 = _mm256_fnmadd_pd( a_4567, b_0101, C_40_51_60_71 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[16] ); // prefetch
			C_01_10_21_30 = _mm256_fnmadd_pd( a_0123, b_1010, C_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Am0[16] ); // prefetch
			C_41_50_61_70 = _mm256_fnmadd_pd( a_4567, b_1010, C_41_50_61_70 );
			a_4567        = _mm256_load_pd( &Am1[16] ); // prefetch

			Am0 += 16;
			Am1 += 16;
			Bm  += 16;

			}

		}

	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, C_00_11_20_31 );
	c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, C_40_51_60_71 );
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, C_01_10_21_30 );
	c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, C_41_50_61_70 );

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
	d_01_11_21_31 = _mm256_fnmadd_pd( d_00_10_20_30, a_10, d_01_11_21_31 );
	d_41_51_61_71 = _mm256_fnmadd_pd( d_40_50_60_70, a_10, d_41_51_61_71 );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	d_41_51_61_71 = _mm256_mul_pd( d_41_51_61_71, a_11 );
	_mm256_store_pd( &D0[0+bs*1], d_01_11_21_31 );
	_mm256_store_pd( &D1[0+bs*1], d_41_51_61_71 );

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dgemm_dtrsm_nt_4x4_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;

	int k;
	
	__m256d
		zeros, 
		a_0, A_0,
		b_0, B_0, b_1, b_2, B_2, b_3,
		c_00, c_01, c_03, c_02,
		C_00, C_01, C_03, C_02;
	
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

	if(kadd>0)
		{

		// prefetch
		a_0 = _mm256_load_pd( &Ap[0] );
		b_0 = _mm256_broadcast_pd( (__m128d *) &Bp[0] );
		b_2 = _mm256_broadcast_pd( (__m128d *) &Bp[2] );

		if(tri==1)
			{

			if(kadd>=4)
				{

				// k = 0
				a_0  = _mm256_blend_pd( zeros, a_0, 0x1 );
				B_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
				A_0  = _mm256_load_pd( &Ap[4] ); // prefetch
				b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
				b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
				c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
				B_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
					
				// k = 1
				A_0  = _mm256_blend_pd( zeros, A_0, 0x3 );
				b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
				a_0  = _mm256_load_pd( &Ap[8] ); // prefetch
				b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
				C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
				C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
				b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
				C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
				C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
				b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch

				// k = 2
				a_0  = _mm256_blend_pd( zeros, a_0, 0x7 );
				B_0  = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
				A_0  = _mm256_load_pd( &Ap[12] ); // prefetch
				b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
				b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
				c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
				B_2  = _mm256_broadcast_pd( (__m128d *) &Bp[14] ); // prefetch

				// k = 3
				b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
				a_0  = _mm256_load_pd( &Ap[16] ); // prefetch
				b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
				C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
				C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
				b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
				C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
				C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
				b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[18] ); // prefetch
					
				Ap += 16;
				Bp += 16;
				k += 4;


				}
			else
				{

				// k = 0
				a_0  = _mm256_blend_pd( zeros, a_0, 0x1 );
				B_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
				A_0  = _mm256_load_pd( &Ap[4] ); // prefetch
				b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
				c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
				c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
				b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
				c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
				c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
				B_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch

				k += 1;

				if(kadd>1)
					{
					
					// k = 1
					A_0  = _mm256_blend_pd( zeros, A_0, 0x3 );
					b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
					a_0  = _mm256_load_pd( &Ap[8] ); // prefetch
					b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
					C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
					C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
					b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
					C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
					C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
					b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch

					k += 1;

					if(kadd>2)
						{

						// k = 2
						a_0  = _mm256_blend_pd( zeros, a_0, 0x7 );
						B_0  = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
						A_0  = _mm256_load_pd( &Ap[12] ); // prefetch
						b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
						c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
						c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
						b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
						c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
						c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
						B_2  = _mm256_broadcast_pd( (__m128d *) &Bp[14] ); // prefetch

						k += 1;

						}

					}

				}

			}

		for(; k<kadd-3; k+=4)
			{
			
	/*	__builtin_prefetch( A+32 );*/
			B_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			A_0  = _mm256_load_pd( &Ap[4] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
			b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
			c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
			B_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			a_0  = _mm256_load_pd( &Ap[8] ); // prefetch
			b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
			C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
			C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
			b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
			C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
			C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch


	/*	__builtin_prefetch( A+48 );*/
			B_0  = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
			A_0  = _mm256_load_pd( &Ap[12] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
			b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
			c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
			B_2  = _mm256_broadcast_pd( (__m128d *) &Bp[14] ); // prefetch


	/*	__builtin_prefetch( A+56 );*/
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
			a_0  = _mm256_load_pd( &Ap[16] ); // prefetch
			b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
			C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
			C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
			b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
			C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
			C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[18] ); // prefetch
			
			Ap += 16;
			Bp += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			B_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			A_0  = _mm256_load_pd( &Ap[4] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
			b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
			c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
			B_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
			
			
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			a_0  = _mm256_load_pd( &Ap[8] ); // prefetch
			b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
			C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
			C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
			b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
			C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
			C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bp[10] ); // prefetch
			
			
			Ap += 8;
			Bp += 8;

			}

		for(; k<kadd; k+=1)
			{
			
//			B_0  = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
//			A_0  = _mm256_load_pd( &Ap[4] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
			c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
			b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
			c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
//			B_2  = _mm256_broadcast_pd( (__m128d *) &Bp[6] ); // prefetch
			
//			Ap += 4; // keep it !!!
//			Bp += 4; // keep it !!!

			}

		}

	if(ksub>0)
		{
		
		// prefetch
		a_0 = _mm256_load_pd( &Am[0] );
		b_0 = _mm256_broadcast_pd( (__m128d *) &Bm[0] );
		b_2 = _mm256_broadcast_pd( (__m128d *) &Bm[2] );

		for(k=0; k<ksub-3; k+=4) // correction in cholesky is multiple of block size 4
			{
			
	/*	__builtin_prefetch( A+32 );*/
			B_0  = _mm256_broadcast_pd( (__m128d *) &Bm[4] ); // prefetch
			A_0  = _mm256_load_pd( &Am[4] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
			c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
			b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
			c_03 = _mm256_fnmadd_pd( a_0, b_3, c_03 );
			B_2  = _mm256_broadcast_pd( (__m128d *) &Bm[6] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bm[8] ); // prefetch
			a_0  = _mm256_load_pd( &Am[8] ); // prefetch
			b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
			C_00 = _mm256_fnmadd_pd( A_0, B_0, C_00 );
			C_01 = _mm256_fnmadd_pd( A_0, b_1, C_01 );
			b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
			C_02 = _mm256_fnmadd_pd( A_0, B_2, C_02 );
			C_03 = _mm256_fnmadd_pd( A_0, b_3, C_03 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bm[10] ); // prefetch


	/*	__builtin_prefetch( A+48 );*/
			B_0  = _mm256_broadcast_pd( (__m128d *) &Bm[12] ); // prefetch
			A_0  = _mm256_load_pd( &Am[12] ); // prefetch
			b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
			c_00 = _mm256_fnmadd_pd( a_0, b_0, c_00 );
			c_01 = _mm256_fnmadd_pd( a_0, b_1, c_01 );
			b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
			c_02 = _mm256_fnmadd_pd( a_0, b_2, c_02 );
			c_03 = _mm256_fnmadd_pd( a_0, b_3, c_03 );
			B_2  = _mm256_broadcast_pd( (__m128d *) &Bm[14] ); // prefetch


	/*	__builtin_prefetch( A+56 );*/
			b_0  = _mm256_broadcast_pd( (__m128d *) &Bm[16] ); // prefetch
			a_0  = _mm256_load_pd( &Am[16] ); // prefetch
			b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
			C_00 = _mm256_fnmadd_pd( A_0, B_0, C_00 );
			C_01 = _mm256_fnmadd_pd( A_0, b_1, C_01 );
			b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
			C_02 = _mm256_fnmadd_pd( A_0, B_2, C_02 );
			C_03 = _mm256_fnmadd_pd( A_0, b_3, C_03 );
			b_2  = _mm256_broadcast_pd( (__m128d *) &Bm[18] ); // prefetch
			
			Am += 16;
			Bm += 16;

			}

		}

	c_00 = _mm256_add_pd( c_00, C_00 );
	c_01 = _mm256_add_pd( c_01, C_01 );
	c_03 = _mm256_add_pd( c_03, C_03 );
	c_02 = _mm256_add_pd( c_02, C_02 );

	__m256d
		d_00, d_01, d_02, d_03;

	d_00 = _mm256_blend_pd( c_00, c_01, 0xa );
	d_01 = _mm256_blend_pd( c_00, c_01, 0x5 );
	d_02 = _mm256_blend_pd( c_02, c_03, 0xa );
	d_03 = _mm256_blend_pd( c_02, c_03, 0x5 );
	
	if(alg!=0)
		{
		c_00 = _mm256_load_pd( &C[0+bs*0] );
		c_01 = _mm256_load_pd( &C[0+bs*1] );
		c_02 = _mm256_load_pd( &C[0+bs*2] );
		c_03 = _mm256_load_pd( &C[0+bs*3] );

		d_00 = _mm256_add_pd( d_00, c_00 );
		d_01 = _mm256_add_pd( d_01, c_01 );
		d_02 = _mm256_add_pd( d_02, c_02 );
		d_03 = _mm256_add_pd( d_03, c_03 );
		}

	__m256d
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;
	
	a_00 = _mm256_broadcast_sd( &fact[0] );
	d_00 = _mm256_mul_pd( d_00, a_00 );
	_mm256_store_pd( &D[0+bs*0], d_00 );

	a_10 = _mm256_broadcast_sd( &fact[1] );
	a_11 = _mm256_broadcast_sd( &fact[2] );
	d_01 = _mm256_fnmadd_pd( d_00, a_10, d_01 );
	d_01 = _mm256_mul_pd( d_01, a_11 );
	_mm256_store_pd( &D[0+bs*1], d_01 );

	a_20 = _mm256_broadcast_sd( &fact[3] );
	a_21 = _mm256_broadcast_sd( &fact[4] );
	a_22 = _mm256_broadcast_sd( &fact[5] );
	d_02 = _mm256_fnmadd_pd( d_00, a_20, d_02 );
	d_02 = _mm256_fnmadd_pd( d_01, a_21, d_02 );
	d_02 = _mm256_mul_pd( d_02, a_22 );
	_mm256_store_pd( &D[0+bs*2], d_02 );

	a_30 = _mm256_broadcast_sd( &fact[6] );
	a_31 = _mm256_broadcast_sd( &fact[7] );
	a_32 = _mm256_broadcast_sd( &fact[8] );
	a_33 = _mm256_broadcast_sd( &fact[9] );
	d_03 = _mm256_fnmadd_pd( d_00, a_30, d_03 );
	d_03 = _mm256_fnmadd_pd( d_01, a_31, d_03 );
	d_03 = _mm256_fnmadd_pd( d_02, a_32, d_03 );
	d_03 = _mm256_mul_pd( d_03, a_33 );
	_mm256_store_pd( &D[0+bs*3], d_03 );

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_dgemm_dtrsm_nt_4x2_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;

	int k;
	
	__m256d
		zeros,
		a_0123,
		b_0101, b_1010,
		ab_temp, // temporary results
		c_00_11_20_31, c_01_10_21_30, C_00_11_20_31, C_01_10_21_30,
		d_00_11_20_31, d_01_10_21_30, D_00_11_20_31, D_01_10_21_30;

	// zero registers
	zeros = _mm256_setzero_pd();
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	C_00_11_20_31 = _mm256_setzero_pd();
	C_01_10_21_30 = _mm256_setzero_pd();
	d_00_11_20_31 = _mm256_setzero_pd();
	d_01_10_21_30 = _mm256_setzero_pd();
	D_00_11_20_31 = _mm256_setzero_pd();
	D_01_10_21_30 = _mm256_setzero_pd();

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
			c_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, C_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			C_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, C_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch


	/*	__builtin_prefetch( A+48 );*/
			d_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, d_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[12] ); // prefetch
			d_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, d_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap[12] ); // prefetch


	/*	__builtin_prefetch( A+56 );*/
			D_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, D_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[16] ); // prefetch
			D_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, D_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap[16] ); // prefetch
			
			Ap += 16;
			Bp += 16;

			}
		
		for(; k<kadd-1; k+=2)
			{
			
			c_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
		
			
			C_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, C_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[8] ); // prefetch
			C_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, C_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Ap[8] ); // prefetch
			
			
			Ap += 8;
			Bp += 8;

			}

		for(; k<kadd; k+=1)
			{
			
			c_00_11_20_31 = _mm256_fmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			//b_0101        = _mm256_broadcast_pd( (__m128d *) &Bp[4] ); // prefetch
			c_01_10_21_30 = _mm256_fmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			//a_0123        = _mm256_load_pd( &Ap[4] ); // prefetch
			
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
			c_00_11_20_31 = _mm256_fnmadd_pd( a_0123, b_0101, c_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[4] ); // prefetch
			c_01_10_21_30 = _mm256_fnmadd_pd( a_0123, b_1010, c_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Am[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11_20_31 = _mm256_fnmadd_pd( a_0123, b_0101, C_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[8] ); // prefetch
			C_01_10_21_30 = _mm256_fnmadd_pd( a_0123, b_1010, C_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Am[8] ); // prefetch


	/*	__builtin_prefetch( A+48 );*/
			d_00_11_20_31 = _mm256_fnmadd_pd( a_0123, b_0101, d_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[12] ); // prefetch
			d_01_10_21_30 = _mm256_fnmadd_pd( a_0123, b_1010, d_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Am[12] ); // prefetch


	/*	__builtin_prefetch( A+56 );*/
			D_00_11_20_31 = _mm256_fnmadd_pd( a_0123, b_0101, D_00_11_20_31 );
			b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
			b_0101        = _mm256_broadcast_pd( (__m128d *) &Bm[16] ); // prefetch
			D_01_10_21_30 = _mm256_fnmadd_pd( a_0123, b_1010, D_01_10_21_30 );
			a_0123        = _mm256_load_pd( &Am[16] ); // prefetch
			
			Am += 16;
			Bm += 16;

			}

		}

	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, d_00_11_20_31 );
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, d_01_10_21_30 );
	C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, D_00_11_20_31 );
	C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, D_01_10_21_30 );
	
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
	d_01_11_21_31 = _mm256_fnmadd_pd( d_00_10_20_30, a_10, d_01_11_21_31 );
	d_01_11_21_31 = _mm256_mul_pd( d_01_11_21_31, a_11 );
	_mm256_store_pd( &D[0+bs*1], d_01_11_21_31 );

	}



// normal-transposed, 2x4 with data packed in 4
void kernel_dgemm_dtrsm_nt_2x4_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
	{
	
	const int bs = 4;

	int k;
	
	__m256d
		zeros,
		a_0101,
		b_0123, b_1032,
		ab_temp, // temporary results
		c_00_11_02_13, c_01_10_03_12, C_00_11_02_13, C_01_10_03_12,
		d_00_11_02_13, d_01_10_03_12, D_00_11_02_13, D_01_10_03_12;

	// zero registers
	zeros = _mm256_setzero_pd();
	c_00_11_02_13 = _mm256_setzero_pd();
	c_01_10_03_12 = _mm256_setzero_pd();
	C_00_11_02_13 = _mm256_setzero_pd();
	C_01_10_03_12 = _mm256_setzero_pd();
	d_00_11_02_13 = _mm256_setzero_pd();
	d_01_10_03_12 = _mm256_setzero_pd();
	D_00_11_02_13 = _mm256_setzero_pd();
	D_01_10_03_12 = _mm256_setzero_pd();

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
			c_00_11_02_13 = _mm256_fmadd_pd( a_0101, b_0123, c_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			c_01_10_03_12 = _mm256_fmadd_pd( a_0101, b_1032, c_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11_02_13 = _mm256_fmadd_pd( a_0101, b_0123, C_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			C_01_10_03_12 = _mm256_fmadd_pd( a_0101, b_1032, C_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[8] ); // prefetch


	/*	__builtin_prefetch( A+48 );*/
			d_00_11_02_13 = _mm256_fmadd_pd( a_0101, b_0123, d_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[12] ); // prefetch
			d_01_10_03_12 = _mm256_fmadd_pd( a_0101, b_1032, d_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[12] ); // prefetch


	/*	__builtin_prefetch( A+56 );*/
			D_00_11_02_13 = _mm256_fmadd_pd( a_0101, b_0123, D_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[16] ); // prefetch
			D_01_10_03_12 = _mm256_fmadd_pd( a_0101, b_1032, D_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[16] ); // prefetch
				
			Ap += 16;
			Bp += 16;

			}
		
		if(kadd%4>=2)
			{
			
			c_00_11_02_13 = _mm256_fmadd_pd( a_0101, b_0123, c_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			c_01_10_03_12 = _mm256_fmadd_pd( a_0101, b_1032, c_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[4] ); // prefetch
			
			
			C_00_11_02_13 = _mm256_fmadd_pd( a_0101, b_0123, C_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bp[8] ); // prefetch
			C_01_10_03_12 = _mm256_fmadd_pd( a_0101, b_1032, C_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[8] ); // prefetch

		
			Ap += 8;
			Bp += 8;

			}

		if(kadd%2==1)
			{
			
			c_00_11_02_13 = _mm256_fmadd_pd( a_0101, b_0123, c_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
//			b_0123        = _mm256_load_pd( &Bp[4] ); // prefetch
			c_01_10_03_12 = _mm256_fmadd_pd( a_0101, b_1032, c_01_10_03_12 );
//			a_0101        = _mm256_broadcast_pd( (__m128d *) &Ap[4] ); // prefetch
			
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
			c_00_11_02_13 = _mm256_fnmadd_pd( a_0101, b_0123, c_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[4] ); // prefetch
			c_01_10_03_12 = _mm256_fnmadd_pd( a_0101, b_1032, c_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Am[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11_02_13 = _mm256_fnmadd_pd( a_0101, b_0123, C_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[8] ); // prefetch
			C_01_10_03_12 = _mm256_fnmadd_pd( a_0101, b_1032, C_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Am[8] ); // prefetch


	/*	__builtin_prefetch( A+48 );*/
			d_00_11_02_13 = _mm256_fnmadd_pd( a_0101, b_0123, d_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[12] ); // prefetch
			d_01_10_03_12 = _mm256_fnmadd_pd( a_0101, b_1032, d_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Am[12] ); // prefetch


	/*	__builtin_prefetch( A+56 );*/
			D_00_11_02_13 = _mm256_fnmadd_pd( a_0101, b_0123, D_00_11_02_13 );
			b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
			b_0123        = _mm256_load_pd( &Bm[16] ); // prefetch
			D_01_10_03_12 = _mm256_fnmadd_pd( a_0101, b_1032, D_01_10_03_12 );
			a_0101        = _mm256_broadcast_pd( (__m128d *) &Am[16] ); // prefetch
				
			Am += 16;
			Bm += 16;

			}

		}

	c_00_11_02_13 = _mm256_add_pd( c_00_11_02_13, d_00_11_02_13 );
	c_01_10_03_12 = _mm256_add_pd( c_01_10_03_12, d_01_10_03_12 );
	C_00_11_02_13 = _mm256_add_pd( C_00_11_02_13, D_00_11_02_13 );
	C_01_10_03_12 = _mm256_add_pd( C_01_10_03_12, D_01_10_03_12 );
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
void kernel_dgemm_dtrsm_nt_2x2_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg)
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
			c_00_11 = _mm_fmadd_pd( a_01, b_01, c_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[4] ); // prefetch
			c_01_10 = _mm_fmadd_pd( a_01, b_10, c_01_10 );
			a_01    = _mm_load_pd( &Ap[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11 = _mm_fmadd_pd( a_01, b_01, C_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[8] ); // prefetch
			C_01_10 = _mm_fmadd_pd( a_01, b_10, C_01_10 );
			a_01    = _mm_load_pd( &Ap[8] ); // prefetch


	/*	__builtin_prefetch( A+48 );*/
			c_00_11 = _mm_fmadd_pd( a_01, b_01, c_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[12] ); // prefetch
			c_01_10 = _mm_fmadd_pd( a_01, b_10, c_01_10 );
			a_01    = _mm_load_pd( &Ap[12] ); // prefetch


	/*	__builtin_prefetch( A+56 );*/
			C_00_11 = _mm_fmadd_pd( a_01, b_01, C_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[16] ); // prefetch
			C_01_10 = _mm_fmadd_pd( a_01, b_10, C_01_10 );
			a_01    = _mm_load_pd( &Ap[16] ); // prefetch
			
			Ap += 16;
			Bp += 16;

			}
		
		if(kadd%4>=2)
			{
			
			c_00_11 = _mm_fmadd_pd( a_01, b_01, c_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[4] ); // prefetch
			c_01_10 = _mm_fmadd_pd( a_01, b_10, c_01_10 );
			a_01    = _mm_load_pd( &Ap[4] ); // prefetch
		
			
			C_00_11 = _mm_fmadd_pd( a_01, b_01, C_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bp[8] ); // prefetch
			C_01_10 = _mm_fmadd_pd( a_01, b_10, C_01_10 );
			a_01    = _mm_load_pd( &Ap[8] ); // prefetch
			
			Ap += 8;
			Bp += 8;

			}

		if(kadd%2==1)
			{
			
			c_00_11 = _mm_fmadd_pd( a_01, b_01, c_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			//b_01    = _mm_load_pd( &B[4] ); // prefetch
			c_01_10 = _mm_fmadd_pd( a_01, b_10, c_01_10 );
			//a_01    = _mm_load_pd( &A[4] ); // prefetch
		
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
			c_00_11 = _mm_fnmadd_pd( a_01, b_01, c_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bm[4] ); // prefetch
			c_01_10 = _mm_fnmadd_pd( a_01, b_10, c_01_10 );
			a_01    = _mm_load_pd( &Am[4] ); // prefetch
			
			
	/*	__builtin_prefetch( A+40 );*/
			C_00_11 = _mm_fnmadd_pd( a_01, b_01, C_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bm[8] ); // prefetch
			C_01_10 = _mm_fnmadd_pd( a_01, b_10, C_01_10 );
			a_01    = _mm_load_pd( &Am[8] ); // prefetch


	/*	__builtin_prefetch( A+48 );*/
			c_00_11 = _mm_fnmadd_pd( a_01, b_01, c_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bm[12] ); // prefetch
			c_01_10 = _mm_fnmadd_pd( a_01, b_10, c_01_10 );
			a_01    = _mm_load_pd( &Am[12] ); // prefetch


	/*	__builtin_prefetch( A+56 );*/
			C_00_11 = _mm_fnmadd_pd( a_01, b_01, C_00_11 );
			b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
			b_01    = _mm_load_pd( &Bm[16] ); // prefetch
			C_01_10 = _mm_fnmadd_pd( a_01, b_10, C_01_10 );
			a_01    = _mm_load_pd( &Am[16] ); // prefetch
			
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
	
	a_00 = _mm_loaddup_pd( &fact[0] );
	d_00_10 = _mm_mul_pd( d_00_10, a_00 );
	_mm_store_pd( &D[0+bs*0], d_00_10 );

	a_10 = _mm_loaddup_pd( &fact[1] );
	a_11 = _mm_loaddup_pd( &fact[2] );
	d_01_11 = _mm_fnmadd_pd( d_00_10, a_10, d_01_11 );
	d_01_11 = _mm_mul_pd( d_01_11, a_11 );
	_mm_store_pd( &D[0+bs*1], d_01_11 );

	}




