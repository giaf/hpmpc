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
void kernel_dsyrk_nt_12x4_lib4(int kadd, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg)
	{
	
	double *A1 = A0 + 4*sda;
	double *A2 = A0 + 8*sda;
	double *C1 = C0 + 4*sdc;
	double *C2 = C0 + 8*sdc;
	double *D1 = D0 + 4*sdd;
	double *D2 = D0 + 8*sdd;
	
	const int bs = 4;
	const int ldc = bs;
	
	int k;
	
	__m256d
		a_0, a_4, a_8,
		b_0,
		c_00, c_01, c_03, c_02,
		c_40, c_41, c_43, c_42,
		c_80, c_81, c_83, c_82;
	
	// prefetch
	a_0 = _mm256_load_pd( &A0[0] );
	b_0 = _mm256_load_pd( &B[0] );
	a_4 = _mm256_load_pd( &A1[0] );
	a_8 = _mm256_load_pd( &A2[0] );

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

	for(k=0; k<kadd-3; k+=4)
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
		a_0  = _mm256_load_pd( &A0[4] ); // prefetch
		c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
		a_4  = _mm256_load_pd( &A1[4] ); // prefetch
		c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
		b_0  = _mm256_load_pd( &B[4] ); // prefetch
		a_8  = _mm256_load_pd( &A2[4] ); // prefetch
		
		
		
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
		a_0  = _mm256_load_pd( &A0[8] ); // prefetch
		c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
		a_4  = _mm256_load_pd( &A1[8] ); // prefetch
		c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
		b_0  = _mm256_load_pd( &B[8] ); // prefetch
		a_8  = _mm256_load_pd( &A2[8] ); // prefetch



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
		a_0  = _mm256_load_pd( &A0[12] ); // prefetch
		c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
		a_4  = _mm256_load_pd( &A1[12] ); // prefetch
		c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
		b_0  = _mm256_load_pd( &B[12] ); // prefetch
		a_8  = _mm256_load_pd( &A2[12] ); // prefetch


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
		a_0  = _mm256_load_pd( &A0[16] ); // prefetch
		c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
		a_4  = _mm256_load_pd( &A1[16] ); // prefetch
		c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
		b_0  = _mm256_load_pd( &B[16] ); // prefetch
		a_8  = _mm256_load_pd( &A2[16] ); // prefetch
		
		A0 += 16;
		A1 += 16;
		A2 += 16;
		B  += 16;

		}
	
	if(kadd%4>=2)
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
		a_0  = _mm256_load_pd( &A0[4] ); // prefetch
		c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
		a_4  = _mm256_load_pd( &A1[4] ); // prefetch
		c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
		b_0  = _mm256_load_pd( &B[4] );
		a_8  = _mm256_load_pd( &A2[4] ); // prefetch
		
		
		
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
		a_0  = _mm256_load_pd( &A0[8] ); // prefetch
		c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
		a_4  = _mm256_load_pd( &A1[8] ); // prefetch
		c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
		b_0  = _mm256_load_pd( &B[8] );
		a_8  = _mm256_load_pd( &A2[8] ); // prefetch
		
		
		A0 += 8;
		A1 += 8;
		A2 += 8;
		B  += 8;

		}

	if(kadd%2==1)
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
/*		a_0  = _mm256_load_pd( &A0[4] ); // prefetch*/
		c_42 = _mm256_fmadd_pd( a_4, b_0, c_42 );
/*		a_4  = _mm256_load_pd( &A1[4] ); // prefetch*/
		c_82 = _mm256_fmadd_pd( a_8, b_0, c_82 );
//		b_0  = _mm256_load_pd( &B[4] );
/*		a_8  = _mm256_load_pd( &A2[4] ); // prefetch*/
			}

	__m256d
		e_00, e_01, e_02, e_03,
		e_40, e_41, e_42, e_43,
		e_80, e_81, e_82, e_83,
		d_00, d_01, d_02, d_03,
		d_40, d_41, d_42, d_43,
		d_80, d_81, d_82, d_83;


	e_00 = _mm256_blend_pd( c_00, c_01, 0xa );
	e_01 = _mm256_blend_pd( c_00, c_01, 0x5 );
	e_02 = _mm256_blend_pd( c_02, c_03, 0xa );
	e_03 = _mm256_blend_pd( c_02, c_03, 0x5 );

	c_00 = _mm256_blend_pd( e_00, e_02, 0xc );
	c_02 = _mm256_blend_pd( e_00, e_02, 0x3 );
	c_01 = _mm256_blend_pd( e_01, e_03, 0xc );
	c_03 = _mm256_blend_pd( e_01, e_03, 0x3 );

	e_40 = _mm256_blend_pd( c_40, c_41, 0xa );
	e_41 = _mm256_blend_pd( c_40, c_41, 0x5 );
	e_42 = _mm256_blend_pd( c_42, c_43, 0xa );
	e_43 = _mm256_blend_pd( c_42, c_43, 0x5 );

	c_40 = _mm256_blend_pd( e_40, e_42, 0xc );
	c_42 = _mm256_blend_pd( e_40, e_42, 0x3 );
	c_41 = _mm256_blend_pd( e_41, e_43, 0xc );
	c_43 = _mm256_blend_pd( e_41, e_43, 0x3 );
	
	e_80 = _mm256_blend_pd( c_80, c_81, 0xa );
	e_81 = _mm256_blend_pd( c_80, c_81, 0x5 );
	e_82 = _mm256_blend_pd( c_82, c_83, 0xa );
	e_83 = _mm256_blend_pd( c_82, c_83, 0x5 );

	c_80 = _mm256_blend_pd( e_80, e_82, 0xc );
	c_82 = _mm256_blend_pd( e_80, e_82, 0x3 );
	c_81 = _mm256_blend_pd( e_81, e_83, 0xc );
	c_83 = _mm256_blend_pd( e_81, e_83, 0x3 );
	
	if(alg==0) // C = A * B'
		{
		d_01 = _mm256_load_pd( &D0[0+ldc*1] );
		d_02 = _mm256_load_pd( &D0[0+ldc*2] );
		d_03 = _mm256_load_pd( &D0[0+ldc*3] );

		c_01 = _mm256_blend_pd( c_01, d_01, 0x1 );
		c_02 = _mm256_blend_pd( c_02, d_02, 0x3 );
		c_03 = _mm256_blend_pd( c_03, d_03, 0x7 );

		_mm256_store_pd( &D0[0+ldc*0], c_00 );
		_mm256_store_pd( &D0[0+ldc*1], c_01 );
		_mm256_store_pd( &D0[0+ldc*2], c_02 );
		_mm256_store_pd( &D0[0+ldc*3], c_03 );
		_mm256_store_pd( &D1[0+ldc*0], c_40 );
		_mm256_store_pd( &D1[0+ldc*1], c_41 );
		_mm256_store_pd( &D1[0+ldc*2], c_42 );
		_mm256_store_pd( &D1[0+ldc*3], c_43 );
		_mm256_store_pd( &D2[0+ldc*0], c_80 );
		_mm256_store_pd( &D2[0+ldc*1], c_81 );
		_mm256_store_pd( &D2[0+ldc*2], c_82 );
		_mm256_store_pd( &D2[0+ldc*3], c_83 );
		}
	else 
		{
		if(alg==1) // C += A * B'
			{
			d_00 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01 = _mm256_load_pd( &C0[0+ldc*1] );
			d_02 = _mm256_load_pd( &C0[0+ldc*2] );
			d_03 = _mm256_load_pd( &C0[0+ldc*3] );

			d_00 = _mm256_add_pd( d_00, c_00 );
			d_01 = _mm256_add_pd( d_01, c_01 );
			d_02 = _mm256_add_pd( d_02, c_02 );
			d_03 = _mm256_add_pd( d_03, c_03 );

			d_40 = _mm256_load_pd( &C1[0+ldc*0] );
			d_41 = _mm256_load_pd( &C1[0+ldc*1] );
			d_42 = _mm256_load_pd( &C1[0+ldc*2] );
			d_43 = _mm256_load_pd( &C1[0+ldc*3] );
		
			d_40 = _mm256_add_pd( d_40, c_40 );
			d_41 = _mm256_add_pd( d_41, c_41 );
			d_42 = _mm256_add_pd( d_42, c_42 );
			d_43 = _mm256_add_pd( d_43, c_43 );

			d_80 = _mm256_load_pd( &C2[0+ldc*0] );
			d_81 = _mm256_load_pd( &C2[0+ldc*1] );
			d_82 = _mm256_load_pd( &C2[0+ldc*2] );
			d_83 = _mm256_load_pd( &C2[0+ldc*3] );
		
			d_80 = _mm256_add_pd( d_80, c_80 );
			d_81 = _mm256_add_pd( d_81, c_81 );
			d_82 = _mm256_add_pd( d_82, c_82 );
			d_83 = _mm256_add_pd( d_83, c_83 );
			}
		else // C -= A * B'
			{
			d_00 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01 = _mm256_load_pd( &C0[0+ldc*1] );
			d_02 = _mm256_load_pd( &C0[0+ldc*2] );
			d_03 = _mm256_load_pd( &C0[0+ldc*3] );

			d_00 = _mm256_sub_pd( d_00, c_00 );
			d_01 = _mm256_sub_pd( d_01, c_01 );
			d_02 = _mm256_sub_pd( d_02, c_02 );
			d_03 = _mm256_sub_pd( d_03, c_03 );

			d_40 = _mm256_load_pd( &C1[0+ldc*0] );
			d_41 = _mm256_load_pd( &C1[0+ldc*1] );
			d_42 = _mm256_load_pd( &C1[0+ldc*2] );
			d_43 = _mm256_load_pd( &C1[0+ldc*3] );
		
			d_40 = _mm256_sub_pd( d_40, c_40 );
			d_41 = _mm256_sub_pd( d_41, c_41 );
			d_42 = _mm256_sub_pd( d_42, c_42 );
			d_43 = _mm256_sub_pd( d_43, c_43 );

			d_80 = _mm256_load_pd( &C2[0+ldc*0] );
			d_81 = _mm256_load_pd( &C2[0+ldc*1] );
			d_82 = _mm256_load_pd( &C2[0+ldc*2] );
			d_83 = _mm256_load_pd( &C2[0+ldc*3] );
		
			d_80 = _mm256_sub_pd( d_80, c_80 );
			d_81 = _mm256_sub_pd( d_81, c_81 );
			d_82 = _mm256_sub_pd( d_82, c_82 );
			d_83 = _mm256_sub_pd( d_83, c_83 );
			}

		c_01 = _mm256_load_pd( &D0[0+ldc*1] );
		c_02 = _mm256_load_pd( &D0[0+ldc*2] );
		c_03 = _mm256_load_pd( &D0[0+ldc*3] );

		d_01 = _mm256_blend_pd( d_01, c_01, 0x1 );
		d_02 = _mm256_blend_pd( d_02, c_02, 0x3 );
		d_03 = _mm256_blend_pd( d_03, c_03, 0x7 );

		_mm256_store_pd( &D0[0+ldc*0], d_00 );
		_mm256_store_pd( &D0[0+ldc*1], d_01 );
		_mm256_store_pd( &D0[0+ldc*2], d_02 );
		_mm256_store_pd( &D0[0+ldc*3], d_03 );
		_mm256_store_pd( &D1[0+ldc*0], d_40 );
		_mm256_store_pd( &D1[0+ldc*1], d_41 );
		_mm256_store_pd( &D1[0+ldc*2], d_42 );
		_mm256_store_pd( &D1[0+ldc*3], d_43 );
		_mm256_store_pd( &D2[0+ldc*0], d_80 );
		_mm256_store_pd( &D2[0+ldc*1], d_81 );
		_mm256_store_pd( &D2[0+ldc*2], d_82 );
		_mm256_store_pd( &D2[0+ldc*3], d_83 );
		}

	}



// normal-transposed, 8x4 with data packed in 4
void kernel_dsyrk_nt_8x4_lib4(int kadd, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg)
	{
	
	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	const int ldc = bs;
	
	int k;
	
	__m256d
		a_0, a_4, A_0, A_4,
		b_0, b_1, b_2,
		c_00, c_01, c_03, c_02,
		c_40, c_41, c_43, c_42;
	
	// prefetch
	a_0 = _mm256_load_pd( &A0[0] );
	a_4 = _mm256_load_pd( &A1[0] );
	b_0 = _mm256_broadcast_pd( (__m128d *) &B[0] );
	b_2 = _mm256_broadcast_pd( (__m128d *) &B[2] );

	// zero registers
	c_00 = _mm256_setzero_pd();
	c_01 = _mm256_setzero_pd();
	c_03 = _mm256_setzero_pd();
	c_02 = _mm256_setzero_pd();
	c_40 = _mm256_setzero_pd();
	c_41 = _mm256_setzero_pd();
	c_43 = _mm256_setzero_pd();
	c_42 = _mm256_setzero_pd();

	for(k=0; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		A_0  = _mm256_load_pd( &A0[4] ); // prefetch
		A_4  = _mm256_load_pd( &A1[4] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
		c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
		c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
		c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
		c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
		
/*	__builtin_prefetch( A+40 );*/
		a_0  = _mm256_load_pd( &A0[8] ); // prefetch
		a_4  = _mm256_load_pd( &A1[8] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
		c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
		c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
		c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch
		c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
		c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );
	
/*	__builtin_prefetch( A+48 );*/
		A_0  = _mm256_load_pd( &A0[12] ); // prefetch
		A_4  = _mm256_load_pd( &A1[12] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
		c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
		c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
		c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[14] ); // prefetch
		c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
		c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
	
/*	__builtin_prefetch( A+56 );*/
		a_0  = _mm256_load_pd( &A0[16] ); // prefetch
		a_4  = _mm256_load_pd( &A1[16] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
		c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
		c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
		c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[18] ); // prefetch
		c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
		c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kadd%4>=2)
		{
		
		A_0  = _mm256_load_pd( &A0[4] ); // prefetch
		A_4  = _mm256_load_pd( &A1[4] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
		c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
		c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
		c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
		c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
		
		a_0  = _mm256_load_pd( &A0[8] ); // prefetch
		a_4  = _mm256_load_pd( &A1[8] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( A_0, b_0, c_00 );
		c_40 = _mm256_fmadd_pd( A_4, b_0, c_40 );
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_01 = _mm256_fmadd_pd( A_0, b_1, c_01 );
		c_41 = _mm256_fmadd_pd( A_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( A_0, b_2, c_02 );
		c_42 = _mm256_fmadd_pd( A_4, b_2, c_42 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch
		c_03 = _mm256_fmadd_pd( A_0, b_1, c_03 );
		c_43 = _mm256_fmadd_pd( A_4, b_1, c_43 );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kadd%2==1)
		{
		
//		A_0  = _mm256_load_pd( &A0[4] ); // prefetch
//		A_4  = _mm256_load_pd( &A1[4] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
		c_40 = _mm256_fmadd_pd( a_4, b_0, c_40 );
//		b_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
		c_41 = _mm256_fmadd_pd( a_4, b_1, c_41 );
		b_1  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
		c_42 = _mm256_fmadd_pd( a_4, b_2, c_42 );
//		b_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		c_03 = _mm256_fmadd_pd( a_0, b_1, c_03 );
		c_43 = _mm256_fmadd_pd( a_4, b_1, c_43 );
			}

	__m256d
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33,
		d_40_50_60_70, d_41_51_61_71, d_42_52_62_72, d_43_53_63_73;

	c_00_10_20_30 = _mm256_blend_pd( c_00, c_01, 0xa );
	c_01_11_21_31 = _mm256_blend_pd( c_00, c_01, 0x5 );
	c_02_12_22_32 = _mm256_blend_pd( c_02, c_03, 0xa );
	c_03_13_23_33 = _mm256_blend_pd( c_02, c_03, 0x5 );

	c_40_50_60_70 = _mm256_blend_pd( c_40, c_41, 0xa );
	c_41_51_61_71 = _mm256_blend_pd( c_40, c_41, 0x5 );
	c_42_52_62_72 = _mm256_blend_pd( c_42, c_43, 0xa );
	c_43_53_63_73 = _mm256_blend_pd( c_42, c_43, 0x5 );
	
	if(alg==0) // C = A * B'
		{
		d_01_11_21_31 = _mm256_load_pd( &D0[0+ldc*1] );
		d_02_12_22_32 = _mm256_load_pd( &D0[0+ldc*2] );
		d_03_13_23_33 = _mm256_load_pd( &D0[0+ldc*3] );

		c_01_11_21_31 = _mm256_blend_pd( c_01_11_21_31, d_01_11_21_31, 0x1 );
		c_02_12_22_32 = _mm256_blend_pd( c_02_12_22_32, d_02_12_22_32, 0x3 );
		c_03_13_23_33 = _mm256_blend_pd( c_03_13_23_33, d_03_13_23_33, 0x7 );

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

		c_01_11_21_31 = _mm256_load_pd( &D0[0+ldc*1] );
		c_02_12_22_32 = _mm256_load_pd( &D0[0+ldc*2] );
		c_03_13_23_33 = _mm256_load_pd( &D0[0+ldc*3] );

		d_01_11_21_31 = _mm256_blend_pd( d_01_11_21_31, c_01_11_21_31, 0x1 );
		d_02_12_22_32 = _mm256_blend_pd( d_02_12_22_32, c_02_12_22_32, 0x3 );
		d_03_13_23_33 = _mm256_blend_pd( d_03_13_23_33, c_03_13_23_33, 0x7 );

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



// normal-transposed, 4x4 with data packed in 4
void kernel_dsyrk_nt_4x4_lib4(int kadd, double *A, double *B, double *C, double *D, int alg)
	{
	
	const int bs = 4;
	const int ldc = bs;
	
	int k;
	
	__m256d
		a_0, A_0,
		b_0, B_0, b_1, b_2, B_2, b_3,
		c_00, c_01, c_03, c_02,
		C_00, C_01, C_03, C_02;
	
	// prefetch
	a_0 = _mm256_load_pd( &A[0] );
	b_0 = _mm256_broadcast_pd( (__m128d *) &B[0] );
	b_2 = _mm256_broadcast_pd( (__m128d *) &B[2] );

	// zero registers
	c_00 = _mm256_setzero_pd();
	c_01 = _mm256_setzero_pd();
	c_03 = _mm256_setzero_pd();
	c_02 = _mm256_setzero_pd();
	C_00 = _mm256_setzero_pd();
	C_01 = _mm256_setzero_pd();
	C_03 = _mm256_setzero_pd();
	C_02 = _mm256_setzero_pd();


	for(k=0; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		B_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		A_0  = _mm256_load_pd( &A[4] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
		c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
		b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
		c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
		B_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		
		
/*	__builtin_prefetch( A+40 );*/
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		a_0  = _mm256_load_pd( &A[8] ); // prefetch
		b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
		C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
		C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
		b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
		C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
		C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch


/*	__builtin_prefetch( A+48 );*/
		B_0  = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		A_0  = _mm256_load_pd( &A[12] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
		c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
		b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
		c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
		B_2  = _mm256_broadcast_pd( (__m128d *) &B[14] ); // prefetch


/*	__builtin_prefetch( A+56 );*/
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		a_0  = _mm256_load_pd( &A[16] ); // prefetch
		b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
		C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
		C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
		b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
		C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
		C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[18] ); // prefetch
		
	
		A += 16;
		B += 16;

		}
	
	if(kadd%4>=2)
		{
		
		B_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		A_0  = _mm256_load_pd( &A[4] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
		c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
		b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
		c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
		B_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		
		
		b_0  = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		a_0  = _mm256_load_pd( &A[8] ); // prefetch
		b_1  = _mm256_shuffle_pd( B_0, B_0, 0x5 );
		C_00 = _mm256_fmadd_pd( A_0, B_0, C_00 );
		C_01 = _mm256_fmadd_pd( A_0, b_1, C_01 );
		b_3  = _mm256_shuffle_pd( B_2, B_2, 0x5 );
		C_02 = _mm256_fmadd_pd( A_0, B_2, C_02 );
		C_03 = _mm256_fmadd_pd( A_0, b_3, C_03 );
		b_2  = _mm256_broadcast_pd( (__m128d *) &B[10] ); // prefetch

	
		
		A += 8;
		B += 8;

		}

	c_00 = _mm256_add_pd( c_00, C_00 );
	c_01 = _mm256_add_pd( c_01, C_01 );
	c_03 = _mm256_add_pd( c_03, C_03 );
	c_02 = _mm256_add_pd( c_02, C_02 );

	if(kadd%2==1)
		{
		
//		B_0  = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
//		A_0  = _mm256_load_pd( &A[4] ); // prefetch
		b_1  = _mm256_shuffle_pd( b_0, b_0, 0x5 );
		c_00 = _mm256_fmadd_pd( a_0, b_0, c_00 );
		c_01 = _mm256_fmadd_pd( a_0, b_1, c_01 );
		b_3  = _mm256_shuffle_pd( b_2, b_2, 0x5 );
		c_02 = _mm256_fmadd_pd( a_0, b_2, c_02 );
		c_03 = _mm256_fmadd_pd( a_0, b_3, c_03 );
//		B_2  = _mm256_broadcast_pd( (__m128d *) &B[6] ); // prefetch
		
//		A += 4; // keep it !!!
//		B += 4; // keep it !!!
		
		}



	__m256d
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33;

	c_00_10_20_30 = _mm256_blend_pd( c_00, c_01, 0xa );
	c_01_11_21_31 = _mm256_blend_pd( c_00, c_01, 0x5 );
	c_02_12_22_32 = _mm256_blend_pd( c_02, c_03, 0xa );
	c_03_13_23_33 = _mm256_blend_pd( c_02, c_03, 0x5 );
		
	if(alg==0) // C = A * B'
		{
		d_01_11_21_31 = _mm256_load_pd( &D[0+ldc*1] );
		d_02_12_22_32 = _mm256_load_pd( &D[0+ldc*2] );
		d_03_13_23_33 = _mm256_load_pd( &D[0+ldc*3] );

		c_01_11_21_31 = _mm256_blend_pd( c_01_11_21_31, d_01_11_21_31, 0x1 );
		c_02_12_22_32 = _mm256_blend_pd( c_02_12_22_32, d_02_12_22_32, 0x3 );
		c_03_13_23_33 = _mm256_blend_pd( c_03_13_23_33, d_03_13_23_33, 0x7 );

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

		c_01_11_21_31 = _mm256_load_pd( &D[0+ldc*1] );
		c_02_12_22_32 = _mm256_load_pd( &D[0+ldc*2] );
		c_03_13_23_33 = _mm256_load_pd( &D[0+ldc*3] );

		d_01_11_21_31 = _mm256_blend_pd( d_01_11_21_31, c_01_11_21_31, 0x1 );
		d_02_12_22_32 = _mm256_blend_pd( d_02_12_22_32, c_02_12_22_32, 0x3 );
		d_03_13_23_33 = _mm256_blend_pd( d_03_13_23_33, c_03_13_23_33, 0x7 );

		_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
		_mm256_store_pd( &D[0+ldc*2], d_02_12_22_32 );
		_mm256_store_pd( &D[0+ldc*3], d_03_13_23_33 );
		}

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_dsyrk_nt_4x2_lib4(int kadd, double *A, double *B, double *C, double *D, int alg)
	{
	
	const int bs = 4;
	const int ldc = bs;
	
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

	for(k=0; k<kadd-3; k+=4)
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
	
	if(kadd%4>=2)
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

	if(kadd%2==1)
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
		
	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, C_00_11_20_31 );
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, C_01_10_21_30 );

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		d_00_10_20_30, d_01_11_21_31;

	c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
	c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
		
	if(alg==0) // C = A * B'
		{
		d_01_11_21_31 = _mm256_load_pd( &D[0+ldc*1] );

		c_01_11_21_31 = _mm256_blend_pd( c_01_11_21_31, d_01_11_21_31, 0x1 );

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

		c_01_11_21_31 = _mm256_load_pd( &D[0+ldc*1] );

		d_01_11_21_31 = _mm256_blend_pd( d_01_11_21_31, c_01_11_21_31, 0x1 );

		_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
		}


	}



// normal-transposed, 2x2 with data packed in 4
void kernel_dsyrk_nt_2x2_lib4(int kadd, double *A, double *B, double *C, double *D, int alg)
	{
	
	const int bs = 4;
	const int ldc = bs;
	
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
		


	c_00_11 = _mm_add_pd( c_00_11, C_00_11 );
	c_01_10 = _mm_add_pd( c_01_10, C_01_10 );

	__m128d
		c_00_10, c_01_11,
		d_00_10, d_01_11;

	c_00_10 = _mm_blend_pd( c_00_11, c_01_10, 0x2 );
	c_01_11 = _mm_blend_pd( c_00_11, c_01_10, 0x1 );
		
	if(alg==0) // C = A * B'
		{
		d_01_11 = _mm_load_pd( &D[0+ldc*1] );

		c_01_11 = _mm_blend_pd( c_01_11, d_01_11, 0x1 );

		_mm_store_pd( &D[0+ldc*0], c_00_10 );
		_mm_store_pd( &D[0+ldc*1], c_01_11 );
		}
	else 
		{
		d_00_10 = _mm_load_pd( &C[0+ldc*0] );
		d_01_11 = _mm_load_pd( &C[0+ldc*1] );
		
		if(alg==1) // C += A * B'
			{
			d_00_10 = _mm_add_pd( d_00_10, c_00_10 );
			d_01_11 = _mm_add_pd( d_01_11, c_01_11 );
			}
		else // C -= A * B'
			{
			d_00_10 = _mm_sub_pd( d_00_10, c_00_10 );
			d_01_11 = _mm_sub_pd( d_01_11, c_01_11 );
			}

		c_01_11 = _mm_load_pd( &D[0+ldc*1] );

		d_01_11 = _mm_blend_pd( d_01_11, c_01_11, 0x1 );

		_mm_store_pd( &D[0+ldc*0], d_00_10 );
		_mm_store_pd( &D[0+ldc*1], d_01_11 );
		}

	}

