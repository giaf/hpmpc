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




void kernel_stran_8_lib8(int kmax, int kna, float *A, int sda, float *C) // TODO 8 ???
	{
	
	// kmax is at least 4 !!!
	
	int k;

	const int bs = 8;

	__m256
		v0, v1, v2, v3, v4, v5, v6, v7,
		v8, v9, va, vb, vc, vd, ve, vf;

	k=0;

// TODO

	for(; k<kmax-7; k+=8)
		{

		v0 = _mm256_load_ps( &A[0+bs*0] ); // 00 10 20 30
		v1 = _mm256_load_ps( &A[0+bs*1] ); // 01 11 21 31
		v8 = _mm256_unpacklo_ps( v0, v1 ); // 00 01 10 11
		v9 = _mm256_unpackhi_ps( v0, v1 ); // 20 21 30 31

		v2 = _mm256_load_ps( &A[0+bs*2] ); // 02 12 22 32
		v3 = _mm256_load_ps( &A[0+bs*3] ); // 03 13 23 33
		va = _mm256_unpacklo_ps( v2, v3 ); // 02 03 12 13
		vb = _mm256_unpackhi_ps( v2, v3 ); // 22 23 32 33

		v4 = _mm256_load_ps( &A[0+bs*4] ); // 04 14 24 34
		v5 = _mm256_load_ps( &A[0+bs*5] ); // 05 15 25 35
		vc = _mm256_unpacklo_ps( v4, v5 ); // 04 05 14 15
		vd = _mm256_unpackhi_ps( v4, v5 ); // 24 25 34 35

		v6 = _mm256_load_ps( &A[0+bs*6] ); // 06 16 26 36
		v7 = _mm256_load_ps( &A[0+bs*7] ); // 07 17 27 37
		ve = _mm256_unpacklo_ps( v6, v7 ); // 06 07 16 17
		vf = _mm256_unpackhi_ps( v6, v7 ); // 26 27 36 37
		
		A += bs*sda;
		
		v0 = _mm256_shuffle_ps( v8, va, 0x44 ); // 00 01 02 03
		v4 = _mm256_shuffle_ps( vc, ve, 0x44 );
		_mm256_store_ps( &C[0+bs*0], _mm256_permute2f128_ps( v0, v4, 0x20 ) );
		_mm256_store_ps( &C[0+bs*4], _mm256_permute2f128_ps( v0, v4, 0x31 ) );

		v1 = _mm256_shuffle_ps( v8, va, 0xee ); // 10 11 12 13
		v5 = _mm256_shuffle_ps( vc, ve, 0xee );
		_mm256_store_ps( &C[0+bs*1], _mm256_permute2f128_ps( v1, v5, 0x20 ) );
		_mm256_store_ps( &C[0+bs*5], _mm256_permute2f128_ps( v1, v5, 0x31 ) );

		v2 = _mm256_shuffle_ps( v9, vb, 0x44 );
		v6 = _mm256_shuffle_ps( vd, vf, 0x44 );
		_mm256_store_ps( &C[0+bs*2], _mm256_permute2f128_ps( v2, v6, 0x20 ) );
		_mm256_store_ps( &C[0+bs*6], _mm256_permute2f128_ps( v2, v6, 0x31 ) );

		v3 = _mm256_shuffle_ps( v9, vb, 0xee );
		v7 = _mm256_shuffle_ps( vd, vf, 0xee );
		_mm256_store_ps( &C[0+bs*3], _mm256_permute2f128_ps( v3, v7, 0x20 ) );
		_mm256_store_ps( &C[0+bs*7], _mm256_permute2f128_ps( v3, v7, 0x31 ) );

		C += bs*bs;

		}
	
	}

