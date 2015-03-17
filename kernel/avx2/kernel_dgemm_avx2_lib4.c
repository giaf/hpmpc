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



void kernel_dgemm_nt_12x4_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td)
	{
	
	if(kmax<=0)
		return;
	
	double *A1 = A0 + 4*sda;
	double *A2 = A0 + 8*sda;
	double *C1 = C0 + 4*sdc;
	double *C2 = C0 + 8*sdc;
	double *D1 = D0 + 4*sdd;
	double *D2 = D0 + 8*sdd;
	
	const int ldc = 4;

	int k;
	
	__m256d
		a_0, a_4, a_8,
		b_0,
		c_00, c_01, c_03, c_02,
		c_40, c_41, c_43, c_42,
		c_80, c_81, c_83, c_82;
	
	// prefetch
	a_0 = _mm256_load_pd( &A0[0] );
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

	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0  = _mm256_load_pd( &B[0] );
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
		a_8  = _mm256_load_pd( &A2[4] ); // prefetch
		
		
		
		b_0  = _mm256_load_pd( &B[4] );
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
		a_8  = _mm256_load_pd( &A2[8] ); // prefetch



		b_0  = _mm256_load_pd( &B[8] );
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
		a_8  = _mm256_load_pd( &A2[12] ); // prefetch


		b_0  = _mm256_load_pd( &B[12] );
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
		a_8  = _mm256_load_pd( &A2[16] ); // prefetch
		
		A0 += 16;
		A1 += 16;
		A2 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		b_0  = _mm256_load_pd( &B[0] );
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
		a_8  = _mm256_load_pd( &A2[4] ); // prefetch
		
		
		
		b_0  = _mm256_load_pd( &B[4] );
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
		a_8  = _mm256_load_pd( &A2[8] ); // prefetch
		
		
		A0 += 8;
		A1 += 8;
		A2 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		b_0  = _mm256_load_pd( &B[0] );
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
/*		a_8  = _mm256_load_pd( &A2[4] ); // prefetch*/
		
		}

	__m256d
		e_00, e_01, e_02, e_03,
		e_10, e_12, e_50, e_52, e_90, e_92,
		e_40, e_41, e_42, e_43,
		e_80, e_81, e_82, e_83,
		c_10, c_20, c_30,
		c_50, c_60, c_70,
		c_90, c_a0, c_b0,
		d_10, d_20, d_30,
		d_50, d_60, d_70,
		d_90, d_a0, d_b0,
		d_00, d_01, d_02, d_03,
		d_40, d_41, d_42, d_43,
		d_80, d_81, d_82, d_83;

	if(alg==0) // D = A * B'
		{
		if(td==0) // AB = A * B'
			{
			e_00 = _mm256_blend_pd( c_00, c_01, 0xa );
			e_01 = _mm256_blend_pd( c_00, c_01, 0x5 );
			e_02 = _mm256_blend_pd( c_02, c_03, 0xa );
			e_03 = _mm256_blend_pd( c_02, c_03, 0x5 );

			c_00 = _mm256_blend_pd( e_00, e_02, 0xc );
			c_02 = _mm256_blend_pd( e_00, e_02, 0x3 );
			c_01 = _mm256_blend_pd( e_01, e_03, 0xc );
			c_03 = _mm256_blend_pd( e_01, e_03, 0x3 );

			_mm256_store_pd( &D0[0+ldc*0], c_00 );
			_mm256_store_pd( &D0[0+ldc*1], c_01 );
			_mm256_store_pd( &D0[0+ldc*2], c_02 );
			_mm256_store_pd( &D0[0+ldc*3], c_03 );

			e_40 = _mm256_blend_pd( c_40, c_41, 0xa );
			e_41 = _mm256_blend_pd( c_40, c_41, 0x5 );
			e_42 = _mm256_blend_pd( c_42, c_43, 0xa );
			e_43 = _mm256_blend_pd( c_42, c_43, 0x5 );

			c_40 = _mm256_blend_pd( e_40, e_42, 0xc );
			c_42 = _mm256_blend_pd( e_40, e_42, 0x3 );
			c_41 = _mm256_blend_pd( e_41, e_43, 0xc );
			c_43 = _mm256_blend_pd( e_41, e_43, 0x3 );

			_mm256_store_pd( &D1[0+ldc*0], c_40 );
			_mm256_store_pd( &D1[0+ldc*1], c_41 );
			_mm256_store_pd( &D1[0+ldc*2], c_42 );
			_mm256_store_pd( &D1[0+ldc*3], c_43 );

			e_80 = _mm256_blend_pd( c_80, c_81, 0xa );
			e_81 = _mm256_blend_pd( c_80, c_81, 0x5 );
			e_82 = _mm256_blend_pd( c_82, c_83, 0xa );
			e_83 = _mm256_blend_pd( c_82, c_83, 0x5 );
			
			c_80 = _mm256_blend_pd( e_80, e_82, 0xc );
			c_82 = _mm256_blend_pd( e_80, e_82, 0x3 );
			c_81 = _mm256_blend_pd( e_81, e_83, 0xc );
			c_83 = _mm256_blend_pd( e_81, e_83, 0x3 );

			_mm256_store_pd( &D2[0+ldc*0], c_80 );
			_mm256_store_pd( &D2[0+ldc*1], c_81 );
			_mm256_store_pd( &D2[0+ldc*2], c_82 );
			_mm256_store_pd( &D2[0+ldc*3], c_83 );
			}
		else // AB = t( A * B' )
			{
			e_00 = _mm256_unpacklo_pd( c_00, c_01 );
			e_10 = _mm256_unpackhi_pd( c_01, c_00 );
			e_02 = _mm256_unpacklo_pd( c_02, c_03 );
			e_12 = _mm256_unpackhi_pd( c_03, c_02 );

			c_00 = _mm256_permute2f128_pd( e_00, e_02, 0x20 );
			c_10 = _mm256_permute2f128_pd( e_10, e_12, 0x20 );
			c_20 = _mm256_permute2f128_pd( e_02, e_00, 0x31 );
			c_30 = _mm256_permute2f128_pd( e_12, e_10, 0x31 );

			_mm256_store_pd( &D0[0+ldc*0], c_00 );
			_mm256_store_pd( &D0[0+ldc*1], c_10 );
			_mm256_store_pd( &D0[0+ldc*2], c_20 );
			_mm256_store_pd( &D0[0+ldc*3], c_30 );

			e_40 = _mm256_unpacklo_pd( c_40, c_41 );
			e_50 = _mm256_unpackhi_pd( c_41, c_40 );
			e_42 = _mm256_unpacklo_pd( c_42, c_43 );
			e_52 = _mm256_unpackhi_pd( c_43, c_42 );

			c_40 = _mm256_permute2f128_pd( e_40, e_42, 0x20 );
			c_50 = _mm256_permute2f128_pd( e_50, e_52, 0x20 );
			c_60 = _mm256_permute2f128_pd( e_42, e_40, 0x31 );
			c_70 = _mm256_permute2f128_pd( e_52, e_50, 0x31 );

			_mm256_store_pd( &D1[0+ldc*0], c_40 );
			_mm256_store_pd( &D1[0+ldc*1], c_50 );
			_mm256_store_pd( &D1[0+ldc*2], c_60 );
			_mm256_store_pd( &D1[0+ldc*3], c_70 );

			e_80 = _mm256_unpacklo_pd( c_80, c_81 );
			e_90 = _mm256_unpackhi_pd( c_81, c_80 );
			e_82 = _mm256_unpacklo_pd( c_82, c_83 );
			e_92 = _mm256_unpackhi_pd( c_83, c_82 );

			c_80 = _mm256_permute2f128_pd( e_80, e_82, 0x20 );
			c_90 = _mm256_permute2f128_pd( e_90, e_92, 0x20 );
			c_a0 = _mm256_permute2f128_pd( e_82, e_80, 0x31 );
			c_b0 = _mm256_permute2f128_pd( e_92, e_90, 0x31 );

			_mm256_store_pd( &D2[0+ldc*0], c_80 );
			_mm256_store_pd( &D2[0+ldc*1], c_90 );
			_mm256_store_pd( &D2[0+ldc*2], c_a0 );
			_mm256_store_pd( &D2[0+ldc*3], c_b0 );
			}
		}
	else
		{
		if(tc==0) // C
			{
			e_00 = _mm256_blend_pd( c_00, c_01, 0xa );
			e_01 = _mm256_blend_pd( c_00, c_01, 0x5 );
			e_02 = _mm256_blend_pd( c_02, c_03, 0xa );
			e_03 = _mm256_blend_pd( c_02, c_03, 0x5 );
			
			c_00 = _mm256_blend_pd( e_00, e_02, 0xc );
			c_02 = _mm256_blend_pd( e_00, e_02, 0x3 );
			c_01 = _mm256_blend_pd( e_01, e_03, 0xc );
			c_03 = _mm256_blend_pd( e_01, e_03, 0x3 );

			d_00 = _mm256_load_pd( &C0[0+ldc*0] );
			d_01 = _mm256_load_pd( &C0[0+ldc*1] );
			d_02 = _mm256_load_pd( &C0[0+ldc*2] );
			d_03 = _mm256_load_pd( &C0[0+ldc*3] );
			
			if(alg==1) // AB = A * B'
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
			else // AB = - A * B'
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

			if(td==0) // AB + C 
				{
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
			else // t(AB + C)
				{
				e_00 = _mm256_unpacklo_pd( d_00, d_01 );
				e_10 = _mm256_unpackhi_pd( d_00, d_01 );
				e_02 = _mm256_unpacklo_pd( d_02, d_03 );
				e_12 = _mm256_unpackhi_pd( d_02, d_03 );

				d_00 = _mm256_permute2f128_pd( e_00, e_02, 0x20 );
				d_20 = _mm256_permute2f128_pd( e_00, e_02, 0x31 );
				d_10 = _mm256_permute2f128_pd( e_10, e_12, 0x20 );
				d_30 = _mm256_permute2f128_pd( e_10, e_12, 0x31 );

				_mm256_store_pd( &D0[0+ldc*0], d_00 );
				_mm256_store_pd( &D0[0+ldc*1], d_10 );
				_mm256_store_pd( &D0[0+ldc*2], d_20 );
				_mm256_store_pd( &D0[0+ldc*3], d_30 );

				e_40 = _mm256_unpacklo_pd( d_40, d_41 );
				e_50 = _mm256_unpackhi_pd( d_40, d_41 );
				e_42 = _mm256_unpacklo_pd( d_42, d_43 );
				e_52 = _mm256_unpackhi_pd( d_42, d_43 );

				d_40 = _mm256_permute2f128_pd( e_40, e_42, 0x20 );
				d_60 = _mm256_permute2f128_pd( e_40, e_42, 0x31 );
				d_50 = _mm256_permute2f128_pd( e_50, e_52, 0x20 );
				d_70 = _mm256_permute2f128_pd( e_50, e_52, 0x31 );

				_mm256_store_pd( &D0[0+ldc*4], d_40 );
				_mm256_store_pd( &D0[0+ldc*5], d_50 );
				_mm256_store_pd( &D0[0+ldc*6], d_60 );
				_mm256_store_pd( &D0[0+ldc*7], d_70 );

				e_80 = _mm256_unpacklo_pd( d_80, d_81 );
				e_90 = _mm256_unpackhi_pd( d_80, d_81 );
				e_82 = _mm256_unpacklo_pd( d_82, d_83 );
				e_92 = _mm256_unpackhi_pd( d_82, d_83 );

				d_80 = _mm256_permute2f128_pd( e_80, e_82, 0x20 );
				d_90 = _mm256_permute2f128_pd( e_80, e_82, 0x31 );
				d_a0 = _mm256_permute2f128_pd( e_90, e_92, 0x20 );
				d_b0 = _mm256_permute2f128_pd( e_90, e_92, 0x31 );

				_mm256_store_pd( &D0[0+ldc*8], d_80 );
				_mm256_store_pd( &D0[0+ldc*9], d_90 );
				_mm256_store_pd( &D0[0+ldc*10], d_a0 );
				_mm256_store_pd( &D0[0+ldc*11], d_b0 );
				}
			}
		else // t(C)
			{

			e_00 = _mm256_unpacklo_pd( c_00, c_01 );
			e_10 = _mm256_unpackhi_pd( c_01, c_00 );
			e_02 = _mm256_unpacklo_pd( c_02, c_03 );
			e_12 = _mm256_unpackhi_pd( c_03, c_02 );

			c_00 = _mm256_permute2f128_pd( e_00, e_02, 0x20 );
			c_10 = _mm256_permute2f128_pd( e_10, e_12, 0x20 );
			c_20 = _mm256_permute2f128_pd( e_02, e_00, 0x31 );
			c_30 = _mm256_permute2f128_pd( e_12, e_10, 0x31 );

			e_40 = _mm256_unpacklo_pd( c_40, c_41 );
			e_50 = _mm256_unpackhi_pd( c_41, c_40 );
			e_42 = _mm256_unpacklo_pd( c_42, c_43 );
			e_52 = _mm256_unpackhi_pd( c_43, c_42 );

			c_40 = _mm256_permute2f128_pd( e_40, e_42, 0x20 );
			c_50 = _mm256_permute2f128_pd( e_50, e_52, 0x20 );
			c_60 = _mm256_permute2f128_pd( e_42, e_40, 0x31 );
			c_70 = _mm256_permute2f128_pd( e_52, e_50, 0x31 );

			e_80 = _mm256_unpacklo_pd( c_80, c_81 );
			e_90 = _mm256_unpackhi_pd( c_81, c_80 );
			e_82 = _mm256_unpacklo_pd( c_82, c_83 );
			e_92 = _mm256_unpackhi_pd( c_83, c_82 );

			c_80 = _mm256_permute2f128_pd( e_80, e_82, 0x20 );
			c_90 = _mm256_permute2f128_pd( e_90, e_92, 0x20 );
			c_a0 = _mm256_permute2f128_pd( e_82, e_80, 0x31 );
			c_b0 = _mm256_permute2f128_pd( e_92, e_90, 0x31 );

			if(alg==1) // AB = A*B'
				{
				d_00 = _mm256_load_pd( &C0[0+ldc*0] );
				d_01 = _mm256_load_pd( &C0[0+ldc*1] );
				d_02 = _mm256_load_pd( &C0[0+ldc*2] );
				d_03 = _mm256_load_pd( &C0[0+ldc*3] );

				d_00 = _mm256_add_pd( d_00, c_00 );
				d_01 = _mm256_add_pd( d_01, c_10 );
				d_02 = _mm256_add_pd( d_02, c_20 );
				d_03 = _mm256_add_pd( d_03, c_30 );

				d_40 = _mm256_load_pd( &C0[0+ldc*4] );
				d_41 = _mm256_load_pd( &C0[0+ldc*5] );
				d_42 = _mm256_load_pd( &C0[0+ldc*6] );
				d_43 = _mm256_load_pd( &C0[0+ldc*7] );

				d_40 = _mm256_add_pd( d_40, c_40 );
				d_41 = _mm256_add_pd( d_41, c_50 );
				d_42 = _mm256_add_pd( d_42, c_60 );
				d_43 = _mm256_add_pd( d_43, c_70 );

				d_80 = _mm256_load_pd( &C0[0+ldc*8] );
				d_81 = _mm256_load_pd( &C0[0+ldc*9] );
				d_82 = _mm256_load_pd( &C0[0+ldc*10] );
				d_83 = _mm256_load_pd( &C0[0+ldc*11] );

				d_80 = _mm256_add_pd( d_80, c_80 );
				d_81 = _mm256_add_pd( d_81, c_90 );
				d_82 = _mm256_add_pd( d_82, c_a0 );
				d_83 = _mm256_add_pd( d_83, c_b0 );
				}
			else // AB = - A*B'
				{
				d_00 = _mm256_load_pd( &C0[0+ldc*0] );
				d_01 = _mm256_load_pd( &C0[0+ldc*1] );
				d_02 = _mm256_load_pd( &C0[0+ldc*2] );
				d_03 = _mm256_load_pd( &C0[0+ldc*3] );

				d_00 = _mm256_sub_pd( d_00, c_00 );
				d_01 = _mm256_sub_pd( d_01, c_10 );
				d_02 = _mm256_sub_pd( d_02, c_20 );
				d_03 = _mm256_sub_pd( d_03, c_30 );

				d_40 = _mm256_load_pd( &C0[0+ldc*4] );
				d_41 = _mm256_load_pd( &C0[0+ldc*5] );
				d_42 = _mm256_load_pd( &C0[0+ldc*6] );
				d_43 = _mm256_load_pd( &C0[0+ldc*7] );

				d_40 = _mm256_sub_pd( d_40, c_40 );
				d_41 = _mm256_sub_pd( d_41, c_50 );
				d_42 = _mm256_sub_pd( d_42, c_60 );
				d_43 = _mm256_sub_pd( d_43, c_70 );

				d_80 = _mm256_load_pd( &C0[0+ldc*8] );
				d_81 = _mm256_load_pd( &C0[0+ldc*9] );
				d_82 = _mm256_load_pd( &C0[0+ldc*10] );
				d_83 = _mm256_load_pd( &C0[0+ldc*11] );

				d_80 = _mm256_sub_pd( d_80, c_80 );
				d_81 = _mm256_sub_pd( d_81, c_90 );
				d_82 = _mm256_sub_pd( d_82, c_a0 );
				d_83 = _mm256_sub_pd( d_83, c_b0 );
				}

			if(td==0) // t( t(AB) + C )
				{
				e_00 = _mm256_unpacklo_pd( d_00, d_01 );
				e_10 = _mm256_unpackhi_pd( d_00, d_01 );
				e_02 = _mm256_unpacklo_pd( d_02, d_03 );
				e_12 = _mm256_unpackhi_pd( d_02, d_03 );

				c_00 = _mm256_permute2f128_pd( e_00, e_02, 0x20 );
				c_20 = _mm256_permute2f128_pd( e_00, e_02, 0x31 );
				c_10 = _mm256_permute2f128_pd( e_10, e_12, 0x20 );
				c_30 = _mm256_permute2f128_pd( e_10, e_12, 0x31 );

				_mm256_store_pd( &D0[0+ldc*0], c_00 );
				_mm256_store_pd( &D0[0+ldc*1], c_10 );
				_mm256_store_pd( &D0[0+ldc*2], c_20 );
				_mm256_store_pd( &D0[0+ldc*3], c_30 );

				e_40 = _mm256_unpacklo_pd( d_40, d_41 );
				e_50 = _mm256_unpackhi_pd( d_40, d_41 );
				e_42 = _mm256_unpacklo_pd( d_42, d_43 );
				e_52 = _mm256_unpackhi_pd( d_42, d_43 );

				c_40 = _mm256_permute2f128_pd( e_40, e_42, 0x20 );
				c_60 = _mm256_permute2f128_pd( e_40, e_42, 0x31 );
				c_50 = _mm256_permute2f128_pd( e_50, e_52, 0x20 );
				c_70 = _mm256_permute2f128_pd( e_50, e_52, 0x31 );

				_mm256_store_pd( &D1[0+ldc*0], c_40 );
				_mm256_store_pd( &D1[0+ldc*1], c_50 );
				_mm256_store_pd( &D1[0+ldc*2], c_60 );
				_mm256_store_pd( &D1[0+ldc*3], c_70 );

				e_80 = _mm256_unpacklo_pd( d_80, d_81 );
				e_90 = _mm256_unpackhi_pd( d_80, d_81 );
				e_82 = _mm256_unpacklo_pd( d_82, d_83 );
				e_92 = _mm256_unpackhi_pd( d_82, d_83 );

				c_80 = _mm256_permute2f128_pd( e_80, e_82, 0x20 );
				c_90 = _mm256_permute2f128_pd( e_80, e_82, 0x31 );
				c_a0 = _mm256_permute2f128_pd( e_90, e_92, 0x20 );
				c_b0 = _mm256_permute2f128_pd( e_90, e_92, 0x31 );

				_mm256_store_pd( &D2[0+ldc*0], c_80 );
				_mm256_store_pd( &D2[0+ldc*1], c_90 );
				_mm256_store_pd( &D2[0+ldc*2], c_a0 );
				_mm256_store_pd( &D2[0+ldc*3], c_b0 );
				}
			else // t(AB) + C
				{
				_mm256_store_pd( &D0[0+ldc*0], d_00 );
				_mm256_store_pd( &D0[0+ldc*1], d_01 );
				_mm256_store_pd( &D0[0+ldc*2], d_02 );
				_mm256_store_pd( &D0[0+ldc*3], d_03 );

				_mm256_store_pd( &D0[0+ldc*4], d_40 );
				_mm256_store_pd( &D0[0+ldc*5], d_41 );
				_mm256_store_pd( &D0[0+ldc*6], d_42 );
				_mm256_store_pd( &D0[0+ldc*7], d_43 );

				_mm256_store_pd( &D0[0+ldc*8], d_80 );
				_mm256_store_pd( &D0[0+ldc*9], d_81 );
				_mm256_store_pd( &D0[0+ldc*10], d_82 );
				_mm256_store_pd( &D0[0+ldc*11], d_83 );
				}

			}

		}

	}



// normal-transposed, 8x4 with data packed in 4
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
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		c_00_11_22_33 = _mm256_fmadd_pd( a_0123, b_0123, c_00_11_22_33 );
		c_40_51_62_73 = _mm256_fmadd_pd( a_4567, b_0123, c_40_51_62_73 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_fmadd_pd( a_0123, b_1032, c_01_10_23_32 );
		c_41_50_63_72 = _mm256_fmadd_pd( a_4567, b_1032, c_41_50_63_72 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_fmadd_pd( a_0123, b_3210, c_03_12_21_30 );
		c_43_52_61_70 = _mm256_fmadd_pd( a_4567, b_3210, c_43_52_61_70 );
		c_02_13_20_31 = _mm256_fmadd_pd( a_0123, b_2301, c_02_13_20_31 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		c_42_53_60_71 = _mm256_fmadd_pd( a_4567, b_2301, c_42_53_60_71 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		
		
/*	__builtin_prefetch( A+40 );*/
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		c_00_11_22_33 = _mm256_fmadd_pd( a_0123, b_0123, c_00_11_22_33 );
		c_40_51_62_73 = _mm256_fmadd_pd( a_4567, b_0123, c_40_51_62_73 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_fmadd_pd( a_0123, b_1032, c_01_10_23_32 );
		c_41_50_63_72 = _mm256_fmadd_pd( a_4567, b_1032, c_41_50_63_72 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_fmadd_pd( a_0123, b_3210, c_03_12_21_30 );
		c_43_52_61_70 = _mm256_fmadd_pd( a_4567, b_3210, c_43_52_61_70 );
		c_02_13_20_31 = _mm256_fmadd_pd( a_0123, b_2301, c_02_13_20_31 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		c_42_53_60_71 = _mm256_fmadd_pd( a_4567, b_2301, c_42_53_60_71 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch


/*	__builtin_prefetch( A+48 );*/
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		c_00_11_22_33 = _mm256_fmadd_pd( a_0123, b_0123, c_00_11_22_33 );
		c_40_51_62_73 = _mm256_fmadd_pd( a_4567, b_0123, c_40_51_62_73 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_fmadd_pd( a_0123, b_1032, c_01_10_23_32 );
		c_41_50_63_72 = _mm256_fmadd_pd( a_4567, b_1032, c_41_50_63_72 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_fmadd_pd( a_0123, b_3210, c_03_12_21_30 );
		c_43_52_61_70 = _mm256_fmadd_pd( a_4567, b_3210, c_43_52_61_70 );
		c_02_13_20_31 = _mm256_fmadd_pd( a_0123, b_2301, c_02_13_20_31 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		c_42_53_60_71 = _mm256_fmadd_pd( a_4567, b_2301, c_42_53_60_71 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch


/*	__builtin_prefetch( A+56 );*/
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		c_00_11_22_33 = _mm256_fmadd_pd( a_0123, b_0123, c_00_11_22_33 );
		c_40_51_62_73 = _mm256_fmadd_pd( a_4567, b_0123, c_40_51_62_73 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_fmadd_pd( a_0123, b_1032, c_01_10_23_32 );
		c_41_50_63_72 = _mm256_fmadd_pd( a_4567, b_1032, c_41_50_63_72 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_fmadd_pd( a_0123, b_3210, c_03_12_21_30 );
		c_43_52_61_70 = _mm256_fmadd_pd( a_4567, b_3210, c_43_52_61_70 );
		c_02_13_20_31 = _mm256_fmadd_pd( a_0123, b_2301, c_02_13_20_31 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		c_42_53_60_71 = _mm256_fmadd_pd( a_4567, b_2301, c_42_53_60_71 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		c_00_11_22_33 = _mm256_fmadd_pd( a_0123, b_0123, c_00_11_22_33 );
		c_40_51_62_73 = _mm256_fmadd_pd( a_4567, b_0123, c_40_51_62_73 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_fmadd_pd( a_0123, b_1032, c_01_10_23_32 );
		c_41_50_63_72 = _mm256_fmadd_pd( a_4567, b_1032, c_41_50_63_72 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_fmadd_pd( a_0123, b_3210, c_03_12_21_30 );
		c_43_52_61_70 = _mm256_fmadd_pd( a_4567, b_3210, c_43_52_61_70 );
		c_02_13_20_31 = _mm256_fmadd_pd( a_0123, b_2301, c_02_13_20_31 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		c_42_53_60_71 = _mm256_fmadd_pd( a_4567, b_2301, c_42_53_60_71 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		
		
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		c_00_11_22_33 = _mm256_fmadd_pd( a_0123, b_0123, c_00_11_22_33 );
		c_40_51_62_73 = _mm256_fmadd_pd( a_4567, b_0123, c_40_51_62_73 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_fmadd_pd( a_0123, b_1032, c_01_10_23_32 );
		c_41_50_63_72 = _mm256_fmadd_pd( a_4567, b_1032, c_41_50_63_72 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_fmadd_pd( a_0123, b_3210, c_03_12_21_30 );
		c_43_52_61_70 = _mm256_fmadd_pd( a_4567, b_3210, c_43_52_61_70 );
		c_02_13_20_31 = _mm256_fmadd_pd( a_0123, b_2301, c_02_13_20_31 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		c_42_53_60_71 = _mm256_fmadd_pd( a_4567, b_2301, c_42_53_60_71 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		c_00_11_22_33 = _mm256_fmadd_pd( a_0123, b_0123, c_00_11_22_33 );
		c_40_51_62_73 = _mm256_fmadd_pd( a_4567, b_0123, c_40_51_62_73 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		c_01_10_23_32 = _mm256_fmadd_pd( a_0123, b_1032, c_01_10_23_32 );
		c_41_50_63_72 = _mm256_fmadd_pd( a_4567, b_1032, c_41_50_63_72 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		c_03_12_21_30 = _mm256_fmadd_pd( a_0123, b_3210, c_03_12_21_30 );
		c_43_52_61_70 = _mm256_fmadd_pd( a_4567, b_3210, c_43_52_61_70 );
		c_02_13_20_31 = _mm256_fmadd_pd( a_0123, b_2301, c_02_13_20_31 );
/*		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch*/
		c_42_53_60_71 = _mm256_fmadd_pd( a_4567, b_2301, c_42_53_60_71 );
/*		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch*/
		
		}

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



// normal-transposed, 4x4 with data packed in 4
void kernel_dgemm_nt_4x4_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td)
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
		}	}

